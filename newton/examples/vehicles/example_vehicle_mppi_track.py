# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle MPPI Track
#
# Races the rc_car fixture around a procedurally generated closed track
# using an MPPI controller whose rollout model is the simulator itself:
# num-samples replicated worlds are collocated at the origin (cross-world
# collision is filtered), world 0 is the rendered hero executing the
# optimized command and worlds 1..K-1 evaluate noise-perturbed command
# sequences every frame. Track generation, out-of-bounds tests, and
# checkpoint progress come from the track_gen package.
#
# Command: python -m newton.examples vehicle_mppi_track --viewer gl
#
###########################################################################

import json
import math
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.vehicles as nv

try:
    import track_gen
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.collision import CollisionChecker
    from track_gen.props import PropSampler
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("This example requires the track_gen package: pip install -e <path-to-track_gen>") from exc

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))

TRACK_HALF_WIDTH = 0.5  # [m]
TRACK_SCALE = 17.0  # calibrated to a measured ~22 m mean track footprint
TRACK_N_MAX = 512
CONE_SPACING = 0.5  # [m]
CAR_HALF_EXTENTS = (0.29, 0.15)  # oriented OOB box [m] (Slash-class rc car)
MAX_TRACK_ATTEMPTS = 32

# reference speed profile (curvature-limited steady state + forward accel pass
# + backward braking pass, the standard racing decomposition): braking
# foresight lives in the profile, not the MPPI horizon
A_LAT_MAX = 12.0  # [m/s^2] usable lateral acceleration for v_ss = sqrt(a_lat/|kappa|)
A_ACC = 6.0  # [m/s^2] forward pass acceleration limit
A_BRK = 10.0  # [m/s^2] backward pass braking limit
V_CAP = 13.6  # [m/s] profile ceiling (matches the 0.8 drive command cap)
PROX_MARGIN = 0.15  # [m] graded wall-proximity band


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.kernel
def _gather_car_pose(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    chassis: wp.array[wp.int32],
    pos: wp.array[wp.vec2f],
    yaw: wp.array[float],
    speed: wp.array[float],
):
    e = wp.tid()
    tf = body_q[chassis[e]]
    p = wp.transform_get_translation(tf)
    pos[e] = wp.vec2f(p[0], p[1])
    yaw[e] = _quat_yaw(wp.transform_get_rotation(tf))
    qd = body_qd[chassis[e]]
    # linear velocity is the first three entries (world frame)
    speed[e] = wp.sqrt(qd[0] * qd[0] + qd[1] * qd[1])


@wp.kernel
def _project_centerline(
    center: wp.array[wp.vec2f],
    cum_s: wp.array[float],
    v_profile: wp.array[float],
    count: int,
    pos: wp.array[wp.vec2f],
    s_out: wp.array[float],
    v_des: wp.array[float],
):
    e = wp.tid()
    p = pos[e]
    best_d2 = float(1.0e12)
    best_s = float(0.0)
    best_i = int(0)
    for i in range(count):
        a = center[i]
        j = i + 1
        if j == count:
            j = 0
        b = center[j]
        ab = b - a
        denom = wp.max(wp.dot(ab, ab), 1.0e-9)
        u = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + u * ab
        d2 = wp.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_s = cum_s[i] + u * wp.sqrt(wp.dot(ab, ab))
            best_i = i
    s_out[e] = best_s
    v_des[e] = v_profile[best_i]


@wp.kernel
def _accumulate_lap_distance(
    s: wp.array[float],
    total_len: float,
    s_prev: wp.array[float],
    total_s: wp.array[float],
):
    ds = s[0] - s_prev[0]
    if ds < -0.5 * total_len:
        ds += total_len
    elif ds > 0.5 * total_len:
        ds -= total_len
    total_s[0] = total_s[0] + ds
    s_prev[0] = s[0]


@wp.kernel
def _broadcast_slice_tf(snap: wp.array[wp.transform], n_per: int, dst: wp.array[wp.transform]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_sv(snap: wp.array[wp.spatial_vector], n_per: int, dst: wp.array[wp.spatial_vector]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_f32(snap: wp.array[float], n_per: int, dst: wp.array[float]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _apply_sample_commands(
    samples: wp.array3d[float],
    t: int,
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    drive[v] = samples[v, t, 0]
    steer[v] = samples[v, t, 1]
    brake[v] = 0.0


@wp.kernel
def _apply_nominal_command(
    nominal: wp.array2d[float],
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    drive[v] = nominal[0, 0]
    steer[v] = nominal[0, 1]
    brake[v] = 0.0


@wp.kernel
def _zero_plan_buffers(costs: wp.array[float], dead: wp.array[wp.int32], back_dist: wp.array[float]):
    e = wp.tid()
    costs[e] = 0.0
    dead[e] = 0
    back_dist[e] = 0.0


@wp.kernel
def _record_start_oob(oob: wp.array[wp.int32], start_oob: wp.array[wp.int32]):
    e = wp.tid()
    start_oob[e] = oob[e]


@wp.kernel
def _accumulate_cost(
    s: wp.array[float],
    v_des: wp.array[float],
    speed: wp.array[float],
    oob: wp.array[wp.int32],
    clearance: wp.array[float],
    start_oob: wp.array[wp.int32],
    samples: wp.array3d[float],
    t: int,
    horizon: int,
    total_len: float,
    params: wp.array[float],  # [w_progress, w_speed, w_steer, kill, w_rate, w_prox, w_term]
    s_prev: wp.array[float],
    back_dist: wp.array[float],
    dead: wp.array[wp.int32],
    costs: wp.array[float],
):
    e = wp.tid()
    if dead[e] == 1:
        return
    # time-decayed kill: crashing later is strictly cheaper, which gives the
    # planner a braking gradient even when every sample ends in a wall
    kill = params[3] * wp.pow(0.9, float(t))
    if oob[e] == 1 and start_oob[e] == 0:
        dead[e] = 1
        costs[e] = costs[e] + kill
        return
    # arc-length progress along the centerline, unwrapped at the lap seam
    ds = s[e] - s_prev[e]
    if ds < -0.5 * total_len:
        ds += total_len
    elif ds > 0.5 * total_len:
        ds -= total_len
    s_prev[e] = s[e]
    if wp.abs(ds) > 5.0:  # teleport/NaN guard
        ds = 0.0
    if start_oob[e] == 0:
        # backward travel along the track beyond a small back-out budget is
        # killed, so reverse stays a brake/recovery tool (this also covers
        # turned-around forward driving, which velocity checks would miss)
        back_dist[e] = back_dist[e] + wp.max(0.0, -ds)
        if back_dist[e] > 1.0:
            dead[e] = 1
            costs[e] = costs[e] + kill
            return
    c = -params[0] * ds
    # one-sided reference-speed penalty: the profile caps corner-entry speed,
    # progress reward alone pushes speed up everywhere else
    over = wp.max(0.0, speed[e] - v_des[e])
    c += params[1] * over * over
    if t == horizon - 1:
        # terminal over-speed cost: a cheap stand-in for a value function at
        # the horizon end (Vazquez-style terminal speed limit)
        c += params[6] * over * over
    steer = samples[e, t, 1]
    c += params[2] * steer * steer
    if t > 0:
        # small command-rate penalty for smooth trajectories
        dd = samples[e, t, 0] - samples[e, t - 1, 0]
        dst = samples[e, t, 1] - samples[e, t - 1, 1]
        c += params[4] * (dd * dd + dst * dst)
    # graded wall proximity: a gradient before contact instead of a cliff at it
    c += params[5] * wp.max(0.0, PROX_MARGIN - clearance[e])
    if start_oob[e] == 1:
        # recovery mode (rollout began outside the band, e.g. after a crash):
        # no kills; penalize distance outside the band so plans steer back in
        c += params[0] * wp.max(0.0, -clearance[e])
    costs[e] = costs[e] + c


@wp.kernel
def _record_ribbon(
    body_q: wp.array[wp.transform],
    chassis0: int,
    t: int,
    ribbon: wp.array[wp.vec3],
):
    p = wp.transform_get_translation(body_q[chassis0])
    ribbon[t] = wp.vec3(p[0], p[1], p[2] + 0.05)


def _speed_profile(center, seg_len):
    """Curvature-limited speed profile with forward-accel and backward-braking passes."""
    prev = np.roll(center, 1, axis=0)
    nxt = np.roll(center, -1, axis=0)
    v1 = center - prev
    v2 = nxt - center
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    angles = np.arctan2(cross, (v1 * v2).sum(axis=1))
    ds = 0.5 * (np.linalg.norm(v1, axis=1) + np.linalg.norm(v2, axis=1))
    kappa = np.abs(angles) / np.maximum(ds, 1e-6)
    kappa = np.convolve(np.concatenate([kappa[-2:], kappa, kappa[:2]]), np.ones(5) / 5.0, mode="valid")
    v = np.minimum(np.sqrt(A_LAT_MAX / np.maximum(kappa, 1e-6)), V_CAP)
    n = len(center)
    for _ in range(2):  # two wrap laps so the closed loop converges
        for i in range(n):  # forward pass: acceleration limit
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * A_ACC * seg_len[i]))
        for i in range(n - 1, -1, -1):  # backward pass: braking limit
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * A_BRK * seg_len[i]))
    return v.astype(np.float32)


def _build_model(num_worlds):
    manifest = json.loads((_ASSET_DIR / "manifest.json").read_text())
    asset = next(a for a in manifest["assets"] if a["name"] == "rc_car")
    rd = asset["reference_dimensions"]

    car = newton.ModelBuilder()
    nv.register_vehicle_attributes(car)
    newton.solvers.SolverMuJoCo.register_custom_attributes(car)
    car.add_usd(str(_ASSET_DIR / asset["file"]))
    nv.configure_wheel_axle_joints(car, axle_joint_labels=asset["axle_joint_labels"])

    joint_by_label = {label: i for i, label in enumerate(car.joint_label)}
    shape_by_label = {label: i for i, label in enumerate(car.shape_label)}
    nv.set_vehicle(
        car,
        0,
        drive_mode=int(nv.DriveMode.ACKERMANN),
        wheelbase=rd["wheelbase_m"],
        track_width=rd["track_width_m"],
        steer_limit=math.radians(rd["steering_limit_deg"]),
    )
    steering = asset["steering_joint_labels"]
    for wheel_id, (body_label, shape_label) in enumerate(
        zip(asset["wheel_body_labels"], asset["wheel_shape_labels"], strict=True)
    ):
        name = body_label.split("/")[-1]
        front = "front" in name
        left = "left" in name
        steer_joint = joint_by_label[steering[0 if left else 1]] if front else -1
        nv.add_wheel(
            car,
            shape=shape_by_label[shape_label],
            vehicle_id=0,
            wheel_id=wheel_id,
            radius=rd["wheel_radius_m"],
            width=rd["wheel_width_m"],
            driven=True,
            steerable=front,
            side=(-1 if left else 1),
            axle_row=(0 if front else 1),
            steer_joint=steer_joint,
        )

    scene = newton.ModelBuilder()
    nv.register_vehicle_attributes(scene)
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(car, num_worlds, spacing=(0.0, 0.0, 0.0))
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    scene.add_ground_plane(cfg=terrain_cfg)
    model = scene.finalize()

    joint_type = model.joint_type.numpy()
    joint_child = model.joint_child.numpy()
    free_children = joint_child[joint_type == int(newton.JointType.FREE)]
    if len(free_children) != num_worlds:
        raise RuntimeError(f"expected {num_worlds} free joints, found {len(free_children)}")
    return model, np.sort(free_children).astype(np.int32)


def _generate_track(num_envs, seed, device):
    """Generates one bezier track shared by all envs; retries invalid seeds."""
    for attempt in range(MAX_TRACK_ATTEMPTS):
        seeds = wp.array(np.full(num_envs, seed + attempt, dtype=np.int32), dtype=wp.int32, device=device)
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=str(device))
        config = TrackGenConfig(
            num_envs=num_envs,
            generator="bezier",
            scale=TRACK_SCALE,
            half_width=TRACK_HALF_WIDTH,
            N_max=TRACK_N_MAX,
            device=str(device),
        )
        track = TrackGenerator(config, rng).generate()
        if bool(track.valid.numpy()[0]):
            return track, seed + attempt
    raise RuntimeError(f"no valid track after {MAX_TRACK_ATTEMPTS} attempts from seed {seed}")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self._test_mode = getattr(args, "test", False)

        self.num_worlds = 32 if self._test_mode else args.num_samples
        self.model, chassis_ids = _build_model(self.num_worlds)
        self.chassis = wp.array(chassis_ids, dtype=wp.int32, device=self.model.device)
        self._chassis0 = int(chassis_ids[0])
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.dofs_per_world = self.model.joint_coord_count // self.num_worlds
        self.vel_dofs_per_world = self.model.joint_dof_count // self.num_worlds

        self.vehicles = nv.WheeledVehicles(
            self.model,
            config=nv.WheeledConfig(
                max_wheel_speed=315.0,
                motor_max_torque=1.0,
                angular_damping=0.0005,
                friction=2.0,
                longitudinal_stiffness=20.0,
                lateral_stiffness=40.0,
            ),
        )
        self.vehicles.configure_solver_contacts()

        # --- track generation and spawn placement ------------------------
        # spawn edits model.joint_q, so it must precede solver/state creation
        # (mjData qpos and State.joint_q are initialized from model.joint_q)
        self.track, self.track_seed = _generate_track(self.num_worlds, args.track_seed, self.model.device)
        self._spawn_on_track()

        # njmax/nconmax are per world; size the shared contact buffer for all worlds
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=192, nconmax=48)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=max(1024, 48 * self.num_worlds))
        self.contacts = self.model.contacts(pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- centerline arc-length progress, speed profile, collision ----

        self.car_pos = wp.zeros(self.num_worlds, dtype=wp.vec2f, device=self.model.device)
        self.car_yaw = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_speed = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_s = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_v_des = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_half_extents = wp.array(
            np.tile(np.array(CAR_HALF_EXTENTS, dtype=np.float32), (self.num_worlds, 1)),
            dtype=wp.vec2f,
            device=self.model.device,
        )
        center = self._env0_polyline(self.track.center)
        seg_len = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
        self.track_len = float(seg_len.sum())
        cum_s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])]).astype(np.float32)
        v_profile = _speed_profile(center, seg_len)
        self._center = wp.array(center, dtype=wp.vec2f, device=self.model.device)
        self._cum_s = wp.array(cum_s, dtype=wp.float32, device=self.model.device)
        self._v_profile = wp.array(v_profile, dtype=wp.float32, device=self.model.device)
        self._center_count = len(center)
        self.checker = CollisionChecker(self.track, max_boxes=1, method="segments")
        self.checker.bind_inputs(position=self.car_pos, yaw=self.car_yaw, half_extents=self.car_half_extents)

        # --- MPPI planner and plan-cycle buffers -------------------------
        horizon = 8 if self._test_mode else args.horizon
        self.rollout_substeps = 2 if self._test_mode else args.rollout_substeps
        if self.rollout_substeps < 1:
            raise ValueError("rollout-substeps must be >= 1")
        total_substeps = horizon * self.rollout_substeps + self.sim_substeps
        if total_substeps % 2 != 0:
            raise ValueError(
                "horizon * rollout_substeps + 8 must be even so state buffers "
                "return to their starting roles each frame (CUDA graph replay)"
            )
        self.planner = nv.ControllerMPPI(
            config=nv.ControllerMPPI.Config(
                num_samples=self.num_worlds,
                horizon=horizon,
                dim=2,
                sigma=(0.35, 0.45),
                temperature=0.05,
                beta=0.7,
                # drive in [-0.3, 0.8]: enough negative torque to brake hard
                # for corners (measured backward travel is cost-killed, so
                # reverse cannot become a cruise mode) and a top-speed cap
                # that keeps the horizon ahead of the braking distance
                bounds_lo=(-0.3, -1.0),
                bounds_hi=(0.8, 1.0),
            ),
            device=self.model.device,
        )
        device = self.model.device
        E = self.num_worlds
        self.costs = wp.zeros(E, dtype=wp.float32, device=device)
        self.dead = wp.zeros(E, dtype=wp.int32, device=device)
        self.s_prev = wp.zeros(E, dtype=wp.float32, device=device)
        self.start_oob = wp.zeros(E, dtype=wp.int32, device=device)
        self.back_dist = wp.zeros(E, dtype=wp.float32, device=device)
        # [w_progress, w_speed, w_steer, kill, w_rate, w_prox, w_term]
        self.cost_params = wp.array([30.0, 2.0, 0.05, 1000.0, 2.0, 10.0, 20.0], dtype=wp.float32, device=device)
        self.ribbon = wp.zeros(horizon, dtype=wp.vec3, device=device)
        # hero lap odometer (display + tests): unwrapped centerline arc length
        self.total_s = wp.zeros(1, dtype=wp.float32, device=device)
        self.frame_s_prev = wp.zeros(1, dtype=wp.float32, device=device)

        # hero-slice snapshots (world 0 leads every per-world array)
        self.snap_joint_q = wp.zeros(self.dofs_per_world, dtype=wp.float32, device=device)
        self.snap_joint_qd = wp.zeros(self.vel_dofs_per_world, dtype=wp.float32, device=device)
        self.snap_body_q = wp.zeros(self.bodies_per_world, dtype=wp.transform, device=device)
        self.snap_body_qd = wp.zeros(self.bodies_per_world, dtype=wp.spatial_vector, device=device)
        wheels_per_world = self.vehicles.dynamics.omega.shape[0] // E
        self.wheels_per_world = wheels_per_world
        self.snap_omega = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_long = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_lat = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_fz = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)

        self.graph = None
        self._telemetry = {
            "speed": 0.0,
            "laps": 0,
            "meters": 0.0,
            "v_des": 0.0,
            "ess": 0.0,
            "alive": 1.0,
            "best_cost": 0.0,
            "mean_cost": 0.0,
            "drive": 0.0,
            "steer": 0.0,
            "hero_oob": 0,
        }
        self._nominal_plan = np.zeros((horizon, 2), dtype=np.float32)
        self.ui_temperature = self.planner.config.temperature
        self.ui_sigma_drive, self.ui_sigma_steer = self.planner.config.sigma
        self.ui_cost = list(self.cost_params.numpy())

        # prime pose/projection buffers so the lap odometer starts at the
        # spawn arc length (eager launches, before any CUDA graph capture)
        self._gather_and_track()
        wp.copy(self.frame_s_prev, self.car_s, count=1)

        self._init_track_render()
        self.follow_camera = True
        self.viewer.set_model(self.model)
        # render only the hero world, exactly where it simulates: all worlds
        # are collocated at the origin, so disable the viewer's automatic
        # per-world grid offsets (they would draw the car away from the track)
        if hasattr(self.viewer, "set_visible_worlds"):
            self.viewer.set_visible_worlds([0])
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

    # --- track helpers ---------------------------------------------------

    def _env0_polyline(self, flat_vec2):
        count = int(self.track.count.numpy()[0])
        return flat_vec2.numpy()[:count]

    def _spawn_on_track(self):
        center = self._env0_polyline(self.track.center)
        p0, p1 = center[0], center[1]
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        joint_q = self.model.joint_q.numpy()
        joint_type = self.model.joint_type.numpy()
        q_start = self.model.joint_q_start.numpy()
        for j in np.flatnonzero(joint_type == int(newton.JointType.FREE)):
            qs = int(q_start[j])
            joint_q[qs + 0] = p0[0]
            joint_q[qs + 1] = p0[1]
            # keep the authored spawn height joint_q[qs + 2]
            joint_q[qs + 3 : qs + 7] = [q[0], q[1], q[2], q[3]]
        self.model.joint_q.assign(joint_q)

    def _init_track_render(self):
        device = self.model.device
        inner = self._env0_polyline(self.track.inner)
        outer = self._env0_polyline(self.track.outer)

        def _loop_lines(poly, z):
            pts = np.column_stack([poly, np.full(len(poly), z, dtype=np.float32)])
            starts = pts
            ends = np.roll(pts, -1, axis=0)
            return (
                wp.array(starts, dtype=wp.vec3, device=device),
                wp.array(ends, dtype=wp.vec3, device=device),
            )

        self._boundary_lines = [_loop_lines(inner, 0.01), _loop_lines(outer, 0.01)]

        # cone poses along both boundaries (env 0 only)
        xforms = []
        for boundary in ("inner", "outer"):
            props = PropSampler(self.track, spacing=CONE_SPACING, boundary=boundary, mode="points").sample()
            n = int(props.count.numpy()[0])
            pos = props.position.numpy()[:n]
            yaw = props.yaw.numpy()[:n]
            for p, y in zip(pos, yaw, strict=True):
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(y))
                xforms.append(wp.transform(wp.vec3(float(p[0]), float(p[1]), 0.0), q))
        self._cone_xforms = wp.array(xforms, dtype=wp.transform, device=device)
        self._cone_colors = wp.array([wp.vec3(1.0, 0.35, 0.05)], dtype=wp.vec3, device=device)
        self._cone_mesh = self._load_cone_mesh()

    @staticmethod
    def _load_cone_mesh():
        stage = Usd.Stage.Open(newton.examples.get_asset("cone.usda"))
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cone"))
        vertices = np.array(usd_mesh.GetPointsAttr().Get())
        indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        mesh = newton.Mesh(vertices, indices)
        mesh.finalize()
        return mesh

    # --- MPPI plan/execute cycle ------------------------------------------

    def _snapshot_hero(self):
        wp.copy(self.snap_joint_q, self.state_0.joint_q, count=self.dofs_per_world)
        wp.copy(self.snap_joint_qd, self.state_0.joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.snap_body_q, self.state_0.body_q, count=self.bodies_per_world)
        wp.copy(self.snap_body_qd, self.state_0.body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(self.snap_omega, dyn.omega, count=self.wheels_per_world)
        wp.copy(self.snap_trans_long, dyn.trans_long, count=self.wheels_per_world)
        wp.copy(self.snap_trans_lat, dyn.trans_lat, count=self.wheels_per_world)
        wp.copy(self.snap_fz, patch.fz, count=self.wheels_per_world)

    def _broadcast_hero(self):
        # Broadcasting joint_q/joint_qd is what actually teleports the particles:
        # SolverMuJoCo re-ingests state joint coords into mjData qpos/qvel every
        # step (update_data_interval=1), so no mujoco_warp.reset_data call is
        # needed on rollout refresh. The only mjData state that survives the
        # teleport is qacc_warmstart (constraint warm start); measured impact on
        # racing metrics is nil, so it is deliberately left untouched.
        dev = self.model.device
        E = self.num_worlds
        wp.launch(
            _broadcast_slice_f32,
            dim=(E, self.dofs_per_world),
            inputs=[self.snap_joint_q, self.dofs_per_world, self.state_0.joint_q],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_f32,
            dim=(E, self.vel_dofs_per_world),
            inputs=[self.snap_joint_qd, self.vel_dofs_per_world, self.state_0.joint_qd],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_tf,
            dim=(E, self.bodies_per_world),
            inputs=[self.snap_body_q, self.bodies_per_world, self.state_0.body_q],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_sv,
            dim=(E, self.bodies_per_world),
            inputs=[self.snap_body_qd, self.bodies_per_world, self.state_0.body_qd],
            device=dev,
        )
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        n = self.wheels_per_world
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_omega, n, dyn.omega], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_long, n, dyn.trans_long], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_lat, n, dyn.trans_lat], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_fz, n, patch.fz], device=dev)

    def _restore_hero(self):
        wp.copy(self.state_0.joint_q, self.snap_joint_q, count=self.dofs_per_world)
        wp.copy(self.state_0.joint_qd, self.snap_joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.state_0.body_q, self.snap_body_q, count=self.bodies_per_world)
        wp.copy(self.state_0.body_qd, self.snap_body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(dyn.omega, self.snap_omega, count=self.wheels_per_world)
        wp.copy(dyn.trans_long, self.snap_trans_long, count=self.wheels_per_world)
        wp.copy(dyn.trans_lat, self.snap_trans_lat, count=self.wheels_per_world)
        wp.copy(patch.fz, self.snap_fz, count=self.wheels_per_world)

    def _gather_and_track(self):
        dev = self.model.device
        wp.launch(
            _gather_car_pose,
            dim=self.num_worlds,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.chassis,
                self.car_pos,
                self.car_yaw,
                self.car_speed,
            ],
            device=dev,
        )
        wp.launch(
            _project_centerline,
            dim=self.num_worlds,
            inputs=[self._center, self._cum_s, self._v_profile, self._center_count, self.car_pos],
            outputs=[self.car_s, self.car_v_des],
            device=dev,
        )
        # query() refreshes and returns the same preallocated contact buffers
        self.contact = self.checker.query()

    def _plan_and_execute(self):
        cmd = self.vehicles.commands
        dev = self.model.device
        horizon = self.planner.config.horizon
        rollout_dt = self.frame_dt / self.rollout_substeps

        self._snapshot_hero()
        self._broadcast_hero()
        self.planner.sample()
        wp.launch(_zero_plan_buffers, dim=self.num_worlds, inputs=[self.costs, self.dead, self.back_dist], device=dev)
        self._gather_and_track()
        wp.copy(self.s_prev, self.car_s, count=self.num_worlds)
        wp.launch(_record_start_oob, dim=self.num_worlds, inputs=[self.contact.oob, self.start_oob], device=dev)

        for t in range(horizon):
            wp.launch(
                _apply_sample_commands,
                dim=self.num_worlds,
                inputs=[self.planner.samples, t, cmd.drive, cmd.steer, cmd.brake],
                device=dev,
            )
            for _ in range(self.rollout_substeps):
                self._substep(rollout_dt)
            self._gather_and_track()
            wp.launch(
                _accumulate_cost,
                dim=self.num_worlds,
                inputs=[
                    self.car_s,
                    self.car_v_des,
                    self.car_speed,
                    self.contact.oob,
                    self.contact.distance,
                    self.start_oob,
                    self.planner.samples,
                    t,
                    horizon,
                    self.track_len,
                    self.cost_params,
                    self.s_prev,
                    self.back_dist,
                    self.dead,
                    self.costs,
                ],
                device=dev,
            )
            wp.launch(
                _record_ribbon,
                dim=1,
                inputs=[self.state_0.body_q, self._chassis0, t, self.ribbon],
                device=dev,
            )

        self.planner.update(self.costs)
        self._restore_hero()
        wp.launch(
            _apply_nominal_command,
            dim=self.num_worlds,
            inputs=[self.planner.nominal, cmd.drive, cmd.steer, cmd.brake],
            device=dev,
        )
        for _ in range(self.sim_substeps):
            self._substep(self.sim_dt)
        self.planner.shift()
        self._gather_and_track()
        # hero lap odometer for telemetry and tests (rollouts never touch it:
        # it only advances here, once per executed frame)
        wp.launch(
            _accumulate_lap_distance,
            dim=1,
            inputs=[self.car_s, self.track_len, self.frame_s_prev, self.total_s],
            device=dev,
        )

    def step(self):
        if self.graph is None and self.model.device.is_cuda:
            track_gen.set_capturing(True)
            with wp.ScopedCapture() as capture:
                self._plan_and_execute()
            self.graph = capture.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._plan_and_execute()
        self.sim_time += self.frame_dt
        self._update_telemetry()

    def _update_telemetry(self):
        t = self._telemetry
        qd = self.state_0.body_qd.numpy()[self._chassis0]
        t["speed"] = float(np.linalg.norm(qd[:2]))  # linear velocity is entries 0:3
        total = float(self.total_s.numpy()[0])
        t["meters"] = total
        t["laps"] = int(total // self.track_len) if total >= 0.0 else 0
        t["v_des"] = float(self.car_v_des.numpy()[0])
        t["ess"] = float(self.planner.ess.numpy()[0])
        t["hero_oob"] = int(self.contact.oob.numpy()[0])
        dead = self.dead.numpy()
        costs = self.costs.numpy()
        t["alive"] = 1.0 - float(dead.mean())
        t["best_cost"] = float(costs.min())
        t["mean_cost"] = float(costs.mean())
        nominal = self.planner.nominal.numpy()
        t["drive"], t["steer"] = float(nominal[0, 0]), float(nominal[0, 1])
        self._nominal_plan = nominal

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        ui.separator()
        ui.text("Controller output")
        t = self._telemetry
        ui.text(f"Drive: {t['drive']:+.2f}   Steer: {t['steer']:+.2f}")
        ui.plot_lines("drive plan", np.ascontiguousarray(self._nominal_plan[:, 0]))
        ui.plot_lines("steer plan", np.ascontiguousarray(self._nominal_plan[:, 1]))
        ui.separator()
        ui.text("Race")
        ui.text(f"Speed: {t['speed']:.2f} m/s (ref {t['v_des']:.2f})")
        ui.text(f"Laps: {t['laps']}   Distance: {t['meters']:.0f} m")
        ui.text(f"OOB: {t['hero_oob']}")
        ui.separator()
        ui.text("Planner")
        ui.text(f"Alive: {100.0 * t['alive']:.0f}%   ESS: {t['ess']:.0f}")
        ui.text(f"Cost best/mean: {t['best_cost']:.1f} / {t['mean_cost']:.1f}")
        changed_t, self.ui_temperature = ui.slider_float("Temperature", self.ui_temperature, 0.005, 0.5)
        if changed_t:
            self.planner.set_temperature(self.ui_temperature)
        changed_d, self.ui_sigma_drive = ui.slider_float("Sigma drive", self.ui_sigma_drive, 0.05, 1.0)
        changed_s, self.ui_sigma_steer = ui.slider_float("Sigma steer", self.ui_sigma_steer, 0.05, 1.0)
        if changed_d or changed_s:
            self.planner.sigma.assign(np.array([self.ui_sigma_drive, self.ui_sigma_steer], dtype=np.float32))
        changed = False
        labels = ("W progress", "W speed", "W steer", "Kill penalty", "W rate", "W proximity", "W terminal")
        for i, label in enumerate(labels):
            hi = 1.0 if i == 2 else (2000.0 if i == 3 else (20.0 if i in (1, 4) else 100.0))
            c, self.ui_cost[i] = ui.slider_float(label, self.ui_cost[i], 0.0, hi)
            changed = changed or c
        if changed:
            self.cost_params.assign(np.array(self.ui_cost, dtype=np.float32))

    def _substep(self, dt):
        self.state_0.clear_forces()
        self.vehicles.update_controls(self.control)
        self.model.collide(self.state_0, self.contacts)
        self.vehicles.apply(self.state_0, self.contacts, dt)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.vehicles.latch_loads(self.contacts)
        self.state_0, self.state_1 = self.state_1, self.state_0

    # --- rendering -------------------------------------------------------

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        tf = self.state_0.body_q.numpy()[self._chassis0]
        x, y, z, w = tf[3], tf[4], tf[5], tf[6]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = tf[:3] - 2.8 * forward + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.viewer.set_camera(
            pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])),
            pitch=-18.0,
            yaw=math.degrees(yaw),
        )

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes(
            "/track/cones",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self._cone_xforms,
            colors=self._cone_colors,
            geo_src=self._cone_mesh,
        )
        for i, (starts, ends) in enumerate(self._boundary_lines):
            self.viewer.log_lines(f"/track/boundary_{i}", starts, ends, (0.35, 0.35, 0.4))
        horizon = self.planner.config.horizon
        self.viewer.log_lines("/mppi/plan", self.ribbon[: horizon - 1], self.ribbon[1:horizon], (0.1, 0.9, 0.3))
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()[: self.bodies_per_world]).all():
            raise ValueError("non-finite hero poses")
        if not np.isfinite(self.costs.numpy()).all():
            raise ValueError("non-finite MPPI costs")

    def test_final(self):
        hero_q = self.state_0.body_q.numpy()[self._chassis0]
        if not np.isfinite(hero_q).all():
            raise ValueError("non-finite hero pose")
        meters = float(self.total_s.numpy()[0])
        if meters < 4.0:
            raise ValueError(f"hero covered only {meters:.2f} m of centerline")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # 512 samples beat 1024/2048 in tuning sweeps (the controller is
        # sim-step-bound, not sample-bound: at fixed temperature, larger
        # sample counts pull the softmax average toward the mean)
        parser.add_argument(
            "--num-samples",
            type=int,
            default=512,
            help="MPPI samples K (= simulated worlds; sample 0 is the hero)",
        )
        parser.add_argument("--horizon", type=int, default=48, help="MPPI planning horizon in control steps")
        parser.add_argument("--rollout-substeps", type=int, default=4, help="solver substeps per rollout control step")
        parser.add_argument("--track-seed", type=int, default=0, help="base seed for track generation")
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
