# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Terrain Contact
#
# Drives the simplified Ackermann RC car fixture over a small terrain course.
# Axle joints are fixed; wheel spin is integrated analytically from drive
# torque and tire reaction moments, front steering targets are written to the
# solver control, and the camera follows the car for interactive driving.
#
# Command: python -m newton.examples wheeled_terrain_contact
#
###########################################################################

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd

ASSET_DIR = Path(newton.examples.get_asset("wheeled"))
MANIFEST_PATH = ASSET_DIR / "manifest.json"
VEHICLE_NAME = "rc_car"

COURSE_MU = 0.95
TIRE_FRICTION_MU = 1.8
TIRE_FALLBACK_NORMAL_LOAD = 14.0
TIRE_LONGITUDINAL_STIFFNESS = 120.0
TIRE_LATERAL_STIFFNESS = 140.0
MAX_WHEEL_ANGULAR_SPEED = 28.0
MAX_WHEEL_DRIVE_TORQUE = 0.65
COAST_BRAKE_TORQUE = 0.05
WHEEL_INERTIA = 0.01
WHEEL_ANGULAR_DAMPING = 0.008
WHEEL_ROLLING_RESISTANCE_TORQUE = 0.01


@wp.kernel
def _update_moment_drive_commands(
    wheel_drive_command: wp.array[wp.float32],
    moment_wheel_angular_speed: wp.array[wp.float32],
    max_drive_torque: float,
    coast_brake_torque: float,
    drive_torque: wp.array[wp.float32],
    brake_torque: wp.array[wp.float32],
    tire_wheel_angular_speed: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    command = wheel_drive_command[wheel_id]
    tire_wheel_angular_speed[wheel_id] = moment_wheel_angular_speed[wheel_id]
    if wp.abs(command) <= 1.0e-4:
        drive_torque[wheel_id] = 0.0
        brake_torque[wheel_id] = coast_brake_torque
    else:
        drive_torque[wheel_id] = command * max_drive_torque
        brake_torque[wheel_id] = 0.0


def _yaw_from_transform_row(transform_row):
    qx, qy, qz, qw = transform_row[3], transform_row[4], transform_row[5], transform_row[6]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _shape_config(*, gap: float = 0.0, margin: float = 0.0, mu: float = COURSE_MU) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.gap = gap
    cfg.margin = margin
    cfg.mu = mu
    return cfg


def _configure_zero_gap_builder(builder: newton.ModelBuilder) -> None:
    builder.rigid_gap = 0.0
    builder.default_shape_cfg.gap = 0.0
    builder.default_shape_cfg.margin = 0.0


def _add_course_terrain(scene: newton.ModelBuilder) -> None:
    terrain_cfg = _shape_config(gap=0.0, margin=0.0)
    scene.add_ground_plane(cfg=terrain_cfg, label="ground")

    for index, x_pos in enumerate((0.15, 0.36, 0.57)):
        scene.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(x_pos, 0.0, 0.012), wp.quat_identity()),
            hx=0.035,
            hy=0.55,
            hz=0.012,
            cfg=terrain_cfg,
            color=wp.vec3(0.45, 0.36, 0.27),
            label=f"speed_bump_{index}",
        )

    ramp_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.05)
    scene.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(1.1, 0.0, 0.04), ramp_rotation),
        hx=0.45,
        hy=0.55,
        hz=0.02,
        cfg=terrain_cfg,
        color=wp.vec3(0.34, 0.42, 0.50),
        label="low_jump_ramp",
    )

    scene.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(1.85, 0.0, 0.018), wp.quat_identity()),
        hx=0.055,
        hy=0.55,
        hz=0.018,
        cfg=terrain_cfg,
        color=wp.vec3(0.36, 0.35, 0.32),
        label="curb_ridge",
    )

    vertices = np.array(
        [
            [2.30, -0.58, 0.010],
            [2.80, -0.58, 0.034],
            [3.30, -0.58, 0.014],
            [2.30, 0.58, 0.018],
            [2.80, 0.58, 0.006],
            [3.30, 0.58, 0.030],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4], dtype=np.int32)
    terrain_mesh = newton.Mesh(vertices, indices, compute_inertia=False)
    scene.add_shape_mesh(
        -1,
        mesh=terrain_mesh,
        cfg=terrain_cfg,
        color=wp.vec3(0.26, 0.43, 0.34),
        label="mesh_ripple",
    )


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.world_count = args.world_count
        self.drive_scale = 1.0
        self.cycle_enabled = True
        self.follow_camera = True
        self.manual_drive = 0.0
        self.manual_steering = 0.0
        self._max_active_wheels = 0
        self._max_chassis_lift = 0.0
        self._max_patch_normal_tilt = 0.0
        self._max_moment_wheel_speed = 0.0

        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(MANIFEST_PATH)}
        asset = assets[VEHICLE_NAME]

        world = newton.ModelBuilder()
        _configure_zero_gap_builder(world)
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        newton.wheeled.register_wheeled_custom_attributes(world)
        world.add_usd(
            str(asset.file),
            xform=wp.transform(wp.vec3(-0.55, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=False,
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
        )
        newton.wheeled.apply_wheeled_manifest(world, MANIFEST_PATH, asset_names=(VEHICLE_NAME,))
        newton.wheeled.configure_wheel_axle_joints(world, axle_joint_labels=asset.axle_joint_labels)

        scene = newton.ModelBuilder()
        _configure_zero_gap_builder(scene)
        scene.replicate(world, self.world_count)
        _add_course_terrain(scene)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.wheeled_metadata = newton.wheeled.build_wheeled_metadata(self.model)
        self.patch_state = newton.wheeled.WheelContactPatchState(self.model, self.wheeled_metadata)
        self.tire_control = newton.wheeled.WheelTireControl(self.model, self.wheeled_metadata)
        self.tire_state = newton.wheeled.WheelTireState(self.model, self.wheeled_metadata)
        self.moment_control = newton.wheeled.WheelMomentControl(self.model, self.wheeled_metadata)
        self.moment_state = newton.wheeled.WheelMomentState(self.model, self.wheeled_metadata)
        self.vehicle_layout = newton.wheeled.build_wheeled_vehicle_layout(
            self.model,
            self.wheeled_metadata,
            manifest_path=MANIFEST_PATH,
            asset_names=(VEHICLE_NAME,),
        )
        self.vehicle_control = newton.wheeled.WheeledVehicleControl(self.vehicle_layout)
        self.vehicle_state = newton.wheeled.WheeledVehicleState(self.vehicle_layout)
        self.motor_config = newton.wheeled.WheeledMotorConfig(
            self.vehicle_layout,
            max_wheel_angular_speed=MAX_WHEEL_ANGULAR_SPEED,
        )
        self.steering_config = newton.wheeled.WheeledSteeringConfig(self.vehicle_layout)
        newton.wheeled.configure_wheel_tire_control(
            self.tire_control,
            friction_mu=TIRE_FRICTION_MU,
            fallback_normal_load=TIRE_FALLBACK_NORMAL_LOAD,
            longitudinal_stiffness=TIRE_LONGITUDINAL_STIFFNESS,
            lateral_stiffness=TIRE_LATERAL_STIFFNESS,
        )
        newton.wheeled.configure_wheel_moment_control(
            self.moment_control,
            wheel_inertia=WHEEL_INERTIA,
            angular_damping=WHEEL_ANGULAR_DAMPING,
            rolling_resistance_torque=WHEEL_ROLLING_RESISTANCE_TORQUE,
        )
        newton.wheeled.configure_mujoco_wheel_contacts(self.model, self.wheeled_metadata)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            disable_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=max(512 * self.world_count, self.model.rigid_contact_max),
            nconmax=max(256 * self.world_count, self.model.rigid_contact_max),
            iterations=20,
            ls_iterations=100,
        )

        self._chassis_body_indices = [
            body_index for body_index, label in enumerate(self.model.body_label) if str(label).endswith("chassis")
        ]
        self._tracked_chassis_body = self._chassis_body_indices[0]
        self._initial_chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices].copy()

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((1.5, 1.2, 0.0))
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 68.0

        self.drive_commands = np.zeros(self.vehicle_layout.drive_channel_count, dtype=np.float32)
        self.steering_commands = np.zeros(self.vehicle_layout.steering_channel_count, dtype=np.float32)
        expected_wheels = self.world_count * len(asset.wheel_body_labels)
        if self.wheeled_metadata.wheel_count != expected_wheels:
            raise ValueError(f"expected {expected_wheels} wheels, found {self.wheeled_metadata.wheel_count}")

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow Camera", self.follow_camera)
        _changed, self.cycle_enabled = ui.checkbox("Cycle commands", self.cycle_enabled)
        _changed, self.drive_scale = ui.slider_float("Drive scale", self.drive_scale, 0.0, 2.0)
        if not self.cycle_enabled:
            _changed, self.manual_drive = ui.slider_float("Drive", self.manual_drive, -1.0, 1.0)
            _changed, self.manual_steering = ui.slider_float("Steering", self.manual_steering, -1.0, 1.0)

    def _update_vehicle_commands(self):
        if self.cycle_enabled:
            cycle_time = self.sim_time % 8.0
            if cycle_time < 4.0:
                drive = 0.85 * self.drive_scale
                steering = 0.0
            elif cycle_time < 5.5:
                drive = 0.45 * self.drive_scale
                steering = 0.35
            elif cycle_time < 7.0:
                drive = -0.5 * self.drive_scale
                steering = -0.25
            else:
                drive = 0.0
                steering = 0.0
        else:
            drive = self.manual_drive * self.drive_scale
            steering = self.manual_steering

        self.drive_commands.fill(np.clip(drive, -1.0, 1.0))
        self.steering_commands.fill(np.clip(steering, -1.0, 1.0))
        newton.wheeled.configure_wheeled_vehicle_control(
            self.vehicle_control,
            drive_command=self.drive_commands,
            steering_command=self.steering_commands,
        )

    def _set_follow_camera(self):
        chassis_q = self.state_0.body_q.numpy()[self._tracked_chassis_body]
        yaw = _yaw_from_transform_row(chassis_q)
        car_pos = chassis_q[:3]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        camera_pos = car_pos - 3.2 * forward + np.array([0.0, 0.0, 1.35], dtype=np.float32)
        self.viewer.set_camera(pos=wp.vec3(*camera_pos), pitch=-19.0, yaw=math.degrees(yaw))

    def _update_terrain_diagnostics(self):
        active = self.patch_state.active.numpy().astype(bool)
        self._max_active_wheels = max(self._max_active_wheels, int(np.count_nonzero(active)))
        if np.any(active):
            normals = self.patch_state.normal.numpy()[active]
            normal_tilt = np.linalg.norm(normals[:, :2], axis=1)
            self._max_patch_normal_tilt = max(self._max_patch_normal_tilt, float(np.max(normal_tilt)))

        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        chassis_lift = chassis_q[:, 2] - self._initial_chassis_q[:, 2]
        self._max_chassis_lift = max(self._max_chassis_lift, float(np.max(chassis_lift)))
        wheel_speed = self.moment_state.wheel_angular_speed.numpy()
        self._max_moment_wheel_speed = max(self._max_moment_wheel_speed, float(np.max(np.abs(wheel_speed))))

    def step(self):
        self._update_vehicle_commands()

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.control.clear()
            self.viewer.apply_forces(self.state_0)
            newton.wheeled.update_wheeled_vehicle_controls(
                self.model,
                self.control,
                self.wheeled_metadata,
                self.vehicle_layout,
                self.vehicle_control,
                self.vehicle_state,
                self.tire_control,
                motor_config=self.motor_config,
                steering_config=self.steering_config,
            )
            wp.launch(
                _update_moment_drive_commands,
                dim=self.wheeled_metadata.wheel_count,
                inputs=[
                    self.vehicle_state.wheel_drive_command,
                    self.moment_state.wheel_angular_speed,
                    MAX_WHEEL_DRIVE_TORQUE,
                    COAST_BRAKE_TORQUE,
                ],
                outputs=[
                    self.moment_control.drive_torque,
                    self.moment_control.brake_torque,
                    self.tire_control.wheel_angular_speed,
                ],
                device=self.model.device,
            )
            self.model.collide(self.state_0, self.contacts)
            newton.wheeled.update_wheel_contact_patches(
                self.model,
                self.state_0,
                self.contacts,
                self.wheeled_metadata,
                self.patch_state,
            )
            newton.wheeled.apply_wheel_tire_forces(
                self.model,
                self.state_0,
                self.wheeled_metadata,
                self.patch_state,
                self.tire_control,
                self.tire_state,
            )
            newton.wheeled.update_wheel_moments(
                self.model,
                self.state_0,
                self.wheeled_metadata,
                self.patch_state,
                self.tire_state,
                self.moment_control,
                self.moment_state,
                self.sim_dt,
                tire_control=self.tire_control,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self._update_terrain_diagnostics()
        self.sim_time += self.frame_dt

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "terrain-controlled wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )
        if self._max_moment_wheel_speed <= 0.0:
            raise ValueError("terrain car controls did not spin analytical wheel moments")

        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        planar_motion = np.linalg.norm(chassis_q[:, :2] - self._initial_chassis_q[:, :2], axis=1)
        if float(np.max(planar_motion)) < 0.35:
            raise ValueError("terrain car did not translate under scripted commands")
        if self._max_active_wheels < min(4, self.wheeled_metadata.wheel_count):
            raise ValueError("terrain car did not maintain wheel contact patches")
        if self.sim_time >= 2.5 and self._max_chassis_lift < 0.004 and self._max_patch_normal_tilt < 0.02:
            raise ValueError("terrain car did not interact with the bump/ramp course")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=420, world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
