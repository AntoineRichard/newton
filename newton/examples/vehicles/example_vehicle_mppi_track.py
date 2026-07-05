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
    from track_gen.checkpoints import CheckpointSampler
    from track_gen.collision import CollisionChecker
    from track_gen.progress import ProgressTracker
    from track_gen.props import PropSampler
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("This example requires the track_gen package: pip install -e <path-to-track_gen>") from exc

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))

TRACK_HALF_WIDTH = 0.5  # [m]
TRACK_SCALE = 17.0  # calibrated for a ~20 m footprint
TRACK_N_MAX = 512
CONE_SPACING = 0.5  # [m]
CHECKPOINT_SPACING = 1.0  # [m]
CAR_HALF_EXTENTS = (0.29, 0.15)  # oriented OOB box [m] (Slash-class rc car)
MAX_TRACK_ATTEMPTS = 32


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.kernel
def _gather_car_pose(
    body_q: wp.array[wp.transform],
    chassis: wp.array[wp.int32],
    pos: wp.array[wp.vec2f],
    yaw: wp.array[float],
):
    e = wp.tid()
    tf = body_q[chassis[e]]
    p = wp.transform_get_translation(tf)
    pos[e] = wp.vec2f(p[0], p[1])
    yaw[e] = _quat_yaw(wp.transform_get_rotation(tf))


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
def _broadcast_env_i32(snap: wp.array[wp.int32], dst: wp.array[wp.int32]):
    dst[wp.tid()] = snap[0]


@wp.kernel
def _restore_env0_i32(snap: wp.array[wp.int32], dst: wp.array[wp.int32]):
    dst[0] = snap[0]


@wp.kernel
def _broadcast_env_vec2(snap: wp.array[wp.vec2f], dst: wp.array[wp.vec2f]):
    dst[wp.tid()] = snap[0]


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
def _zero_plan_buffers(costs: wp.array[float], dead: wp.array[wp.int32]):
    e = wp.tid()
    costs[e] = 0.0
    dead[e] = 0


@wp.kernel
def _record_start_oob(oob: wp.array[wp.int32], start_oob: wp.array[wp.int32]):
    e = wp.tid()
    start_oob[e] = oob[e]


@wp.kernel
def _accumulate_cost(
    dist_to_next: wp.array[float],
    passed: wp.array[wp.int32],
    oob: wp.array[wp.int32],
    clearance: wp.array[float],
    start_oob: wp.array[wp.int32],
    samples: wp.array3d[float],
    t: int,
    params: wp.array[float],  # [w_progress, w_pass, w_steer, kill_penalty]
    dist_prev: wp.array[float],
    dead: wp.array[wp.int32],
    costs: wp.array[float],
):
    e = wp.tid()
    if dead[e] == 1:
        return
    if oob[e] == 1 and start_oob[e] == 0:
        # entered a wall during the rollout: kill the particle
        dead[e] = 1
        costs[e] = costs[e] + params[3]
        return
    d = dist_to_next[e]
    progress = dist_prev[e] - d
    # NaN/garbage guard (dist is NaN until the first tracker update)
    if progress != progress or progress > 1.0e3 or progress < -1.0e3:
        progress = 0.0
    steer = samples[e, t, 1]
    costs[e] = costs[e] - params[0] * progress - params[1] * float(passed[e]) + params[2] * steer * steer
    if start_oob[e] == 1:
        # recovery mode (rollout began outside the band, e.g. after a crash):
        # no kills; penalize distance outside the band so plans steer back in
        costs[e] = costs[e] + params[0] * wp.max(0.0, -clearance[e])
    dist_prev[e] = d


@wp.kernel
def _record_ribbon(
    body_q: wp.array[wp.transform],
    chassis0: int,
    t: int,
    ribbon: wp.array[wp.vec3],
):
    p = wp.transform_get_translation(body_q[chassis0])
    ribbon[t] = wp.vec3(p[0], p[1], p[2] + 0.05)


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

    # hide every shape outside world 0 so the viewer draws a single car
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    flags[worlds >= 1] &= ~int(newton.ShapeFlags.VISIBLE)
    model.shape_flags.assign(flags)

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

        # --- checkpoints, progress, collision ----------------------------

        self.car_pos = wp.zeros(self.num_worlds, dtype=wp.vec2f, device=self.model.device)
        self.car_yaw = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_half_extents = wp.array(
            np.tile(np.array(CAR_HALF_EXTENTS, dtype=np.float32), (self.num_worlds, 1)),
            dtype=wp.vec2f,
            device=self.model.device,
        )
        self.sampler = CheckpointSampler(self.track, spacing=CHECKPOINT_SPACING)
        self.checkpoints = self.sampler.sample()
        self.tracker = ProgressTracker(self.checkpoints, position=self.car_pos)
        self.checker = CollisionChecker(self.track, max_boxes=1, method="segments")
        self.checker.bind_inputs(position=self.car_pos, yaw=self.car_yaw, half_extents=self.car_half_extents)

        # --- MPPI planner and plan-cycle buffers -------------------------
        horizon = 8 if self._test_mode else args.horizon
        self.rollout_substeps = 2 if self._test_mode else args.rollout_substeps
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
                bounds_lo=(-0.3, -1.0),  # limited reverse, full steering
                bounds_hi=(1.0, 1.0),
            ),
            device=self.model.device,
        )
        device = self.model.device
        E = self.num_worlds
        self.costs = wp.zeros(E, dtype=wp.float32, device=device)
        self.dead = wp.zeros(E, dtype=wp.int32, device=device)
        self.dist_prev = wp.zeros(E, dtype=wp.float32, device=device)
        self.start_oob = wp.zeros(E, dtype=wp.int32, device=device)
        # [w_progress, w_pass, w_steer, kill_penalty]
        self.cost_params = wp.array([30.0, 30.0, 0.05, 200.0], dtype=wp.float32, device=device)
        self.ribbon = wp.zeros(horizon, dtype=wp.vec3, device=device)

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
        self.snap_prev_pos = wp.zeros(1, dtype=wp.vec2f, device=device)
        self.snap_next = wp.zeros(1, dtype=wp.int32, device=device)
        self.snap_laps = wp.zeros(1, dtype=wp.int32, device=device)
        self.snap_progress = wp.zeros(1, dtype=wp.int32, device=device)

        self.graph = None
        self._telemetry = {
            "speed": 0.0,
            "laps": 0,
            "progress": 0,
            "dist": 0.0,
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

        self._init_track_render()
        self.follow_camera = True
        self.viewer.set_model(self.model)
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
        wp.copy(self.snap_prev_pos, self.tracker._prev_pos, count=1)
        wp.copy(self.snap_next, self.tracker._next, count=1)
        wp.copy(self.snap_laps, self.tracker._laps, count=1)
        wp.copy(self.snap_progress, self.tracker._progress, count=1)

    def _broadcast_hero(self):
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
        wp.launch(_broadcast_env_vec2, dim=E, inputs=[self.snap_prev_pos, self.tracker._prev_pos], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_next, self.tracker._next], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_laps, self.tracker._laps], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_progress, self.tracker._progress], device=dev)

    def _restore_hero(self):
        dev = self.model.device
        wp.copy(self.state_0.joint_q, self.snap_joint_q, count=self.dofs_per_world)
        wp.copy(self.state_0.joint_qd, self.snap_joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.state_0.body_q, self.snap_body_q, count=self.bodies_per_world)
        wp.copy(self.state_0.body_qd, self.snap_body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(dyn.omega, self.snap_omega, count=self.wheels_per_world)
        wp.copy(dyn.trans_long, self.snap_trans_long, count=self.wheels_per_world)
        wp.copy(dyn.trans_lat, self.snap_trans_lat, count=self.wheels_per_world)
        wp.copy(patch.fz, self.snap_fz, count=self.wheels_per_world)
        wp.copy(self.tracker._prev_pos, self.snap_prev_pos, count=1)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_next, self.tracker._next], device=dev)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_laps, self.tracker._laps], device=dev)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_progress, self.tracker._progress], device=dev)

    def _gather_and_track(self):
        wp.launch(
            _gather_car_pose,
            dim=self.num_worlds,
            inputs=[self.state_0.body_q, self.chassis, self.car_pos, self.car_yaw],
            device=self.model.device,
        )
        # update()/query() refresh and return the same preallocated buffers
        self.events = self.tracker.update()
        self.contact = self.checker.query()

    def _plan_and_execute(self):
        cmd = self.vehicles.commands
        dev = self.model.device
        horizon = self.planner.config.horizon
        rollout_dt = self.frame_dt / self.rollout_substeps

        self._snapshot_hero()
        self._broadcast_hero()
        self.planner.sample()
        wp.launch(_zero_plan_buffers, dim=self.num_worlds, inputs=[self.costs, self.dead], device=dev)
        self._gather_and_track()
        wp.copy(self.dist_prev, self.events.dist_to_next, count=self.num_worlds)
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
                    self.events.dist_to_next,
                    self.events.passed,
                    self.contact.oob,
                    self.contact.distance,
                    self.start_oob,
                    self.planner.samples,
                    t,
                    self.cost_params,
                    self.dist_prev,
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
        events = self.events
        t["laps"] = int(events.laps.numpy()[0])
        t["progress"] = int(events.progress.numpy()[0])
        t["dist"] = float(events.dist_to_next.numpy()[0])
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
        ui.text(f"Speed: {t['speed']:.2f} m/s")
        ui.text(f"Laps: {t['laps']}   Checkpoints: {t['progress']}")
        ui.text(f"Dist to next: {t['dist']:.2f} m   OOB: {t['hero_oob']}")
        ui.separator()
        ui.text("Planner")
        ui.text(f"Alive: {100.0 * t['alive']:.0f}%")
        ui.text(f"Cost best/mean: {t['best_cost']:.1f} / {t['mean_cost']:.1f}")
        changed_t, self.ui_temperature = ui.slider_float("Temperature", self.ui_temperature, 0.005, 0.5)
        if changed_t:
            self.planner.set_temperature(self.ui_temperature)
        changed_d, self.ui_sigma_drive = ui.slider_float("Sigma drive", self.ui_sigma_drive, 0.05, 1.0)
        changed_s, self.ui_sigma_steer = ui.slider_float("Sigma steer", self.ui_sigma_steer, 0.05, 1.0)
        if changed_d or changed_s:
            self.planner.sigma.assign(np.array([self.ui_sigma_drive, self.ui_sigma_steer], dtype=np.float32))

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
        progress = int(self.events.progress.numpy()[0])
        if progress < 2:
            raise ValueError(f"hero passed only {progress} checkpoints")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--num-samples",
            type=int,
            default=1024,
            help="MPPI samples K (= simulated worlds; sample 0 is the hero)",
        )
        parser.add_argument("--horizon", type=int, default=32, help="MPPI planning horizon in control steps")
        parser.add_argument("--rollout-substeps", type=int, default=4, help="solver substeps per rollout control step")
        parser.add_argument("--track-seed", type=int, default=0, help="base seed for track generation")
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
