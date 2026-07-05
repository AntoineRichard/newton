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
    raise ImportError(
        "This example requires the track_gen package: pip install -e <path-to-track_gen>"
    ) from exc

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
        # njmax/nconmax are per world; size the shared contact buffer for all worlds
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=192, nconmax=48)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=max(1024, 48 * self.num_worlds))
        self.contacts = self.model.contacts(pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # --- track, checkpoints, progress, collision --------------------
        self.track, self.track_seed = _generate_track(self.num_worlds, args.track_seed, self.model.device)
        self._spawn_on_track()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(newton.examples.get_asset("cone.usda"))
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cone"))
        vertices = np.array(usd_mesh.GetPointsAttr().Get())
        indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        mesh = newton.Mesh(vertices, indices)
        mesh.finalize()
        return mesh

    # --- per-frame -------------------------------------------------------

    def step(self):
        # temporary open-loop drive; replaced by MPPI in the next commit
        self.vehicles.set_commands(drive=0.3, steer=0.2, brake=0.0)
        for _ in range(self.sim_substeps):
            self._substep(self.sim_dt)
        wp.launch(
            _gather_car_pose,
            dim=self.num_worlds,
            inputs=[self.state_0.body_q, self.chassis, self.car_pos, self.car_yaw],
            device=self.model.device,
        )
        self.tracker.update()
        self.checker.query()
        self.sim_time += self.frame_dt

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
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("non-finite body poses")

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("non-finite final poses")

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
        parser.add_argument(
            "--rollout-substeps", type=int, default=4, help="solver substeps per rollout control step"
        )
        parser.add_argument("--track-seed", type=int, default=0, help="base seed for track generation")
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
