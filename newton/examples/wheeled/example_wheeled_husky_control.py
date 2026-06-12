# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Husky Control
#
# Drives the simplified Husky skid-steer fixture through the vehicle
# command-channel mapper. Axle joints are fixed and wheel speeds are
# analytical tire inputs written from left/right drive channels.
#
# Command: python -m newton.examples wheeled_husky_control
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
VEHICLE_NAME = "husky"
TIRE_FRICTION_MU = 1.2
TIRE_FALLBACK_NORMAL_LOAD = 80.0 * 9.81 / 4.0
TIRE_LONGITUDINAL_STIFFNESS = 900.0
TIRE_LATERAL_STIFFNESS = 1200.0
MAX_WHEEL_ANGULAR_SPEED = 8.0


def _yaw_from_transform_row(transform_row):
    qx, qy, qz, qw = transform_row[3], transform_row[4], transform_row[5], transform_row[6]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _angle_delta(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


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
        self.manual_left_drive = 0.0
        self.manual_right_drive = 0.0

        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(MANIFEST_PATH)}
        asset = assets[VEHICLE_NAME]

        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        newton.wheeled.register_wheeled_custom_attributes(world)
        world.add_usd(
            str(asset.file),
            enable_self_collisions=False,
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
        )
        newton.wheeled.apply_wheeled_manifest(world, MANIFEST_PATH, asset_names=(VEHICLE_NAME,))
        newton.wheeled.configure_wheel_axle_joints(world, axle_joint_labels=asset.axle_joint_labels)

        scene = newton.ModelBuilder()
        scene.replicate(world, self.world_count)
        scene.add_ground_plane(label="ground")
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
        newton.wheeled.configure_mujoco_wheel_contacts(self.model, self.wheeled_metadata)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            disable_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=max(256 * self.world_count, self.model.rigid_contact_max),
            nconmax=max(128 * self.world_count, self.model.rigid_contact_max),
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
            self.viewer.set_world_offsets((1.8, 1.8, 0.0))
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

        self.drive_commands = np.zeros(self.vehicle_layout.drive_channel_count, dtype=np.float32)
        self.drive_channel_side = np.zeros(self.vehicle_layout.drive_channel_count, dtype=np.int32)
        for wheel_id, channel in enumerate(self.vehicle_layout.wheel_drive_channel_host):
            if channel >= 0 and self.drive_channel_side[channel] == 0:
                self.drive_channel_side[channel] = self.vehicle_layout.wheel_side_host[wheel_id]

        expected_wheels = self.world_count * len(asset.wheel_body_labels)
        if self.wheeled_metadata.wheel_count != expected_wheels:
            raise ValueError(f"expected {expected_wheels} wheels, found {self.wheeled_metadata.wheel_count}")

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow Camera", self.follow_camera)
        _changed, self.cycle_enabled = ui.checkbox("Cycle commands", self.cycle_enabled)
        _changed, self.drive_scale = ui.slider_float("Drive scale", self.drive_scale, 0.0, 2.0)
        if not self.cycle_enabled:
            _changed, self.manual_left_drive = ui.slider_float("Left drive", self.manual_left_drive, -1.0, 1.0)
            _changed, self.manual_right_drive = ui.slider_float("Right drive", self.manual_right_drive, -1.0, 1.0)

    def _update_vehicle_commands(self):
        if self.cycle_enabled:
            cycle_time = self.sim_time % 6.0
            if cycle_time < 2.0:
                left_drive = 0.55 * self.drive_scale
                right_drive = 0.55 * self.drive_scale
            elif cycle_time < 3.0:
                left_drive = -0.45 * self.drive_scale
                right_drive = 0.45 * self.drive_scale
            elif cycle_time < 5.0:
                left_drive = -0.5 * self.drive_scale
                right_drive = -0.5 * self.drive_scale
            else:
                left_drive = 0.0
                right_drive = 0.0
        else:
            left_drive = self.manual_left_drive * self.drive_scale
            right_drive = self.manual_right_drive * self.drive_scale

        left_drive = np.clip(left_drive, -1.0, 1.0)
        right_drive = np.clip(right_drive, -1.0, 1.0)
        self.drive_commands.fill(0.0)
        for channel, side in enumerate(self.drive_channel_side):
            if side == newton.wheeled.WheeledVehicleLayout.WheelSide.LEFT:
                self.drive_commands[channel] = left_drive
            else:
                self.drive_commands[channel] = right_drive

        newton.wheeled.configure_wheeled_vehicle_control(self.vehicle_control, drive_command=self.drive_commands)

    def _set_follow_camera(self):
        chassis_q = self.state_0.body_q.numpy()[self._tracked_chassis_body]
        yaw = _yaw_from_transform_row(chassis_q)
        husky_pos = chassis_q[:3]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        camera_pos = husky_pos - 3.6 * forward + np.array([0.0, 0.0, 1.6], dtype=np.float32)
        self.viewer.set_camera(pos=wp.vec3(*camera_pos), pitch=-18.0, yaw=math.degrees(yaw))

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

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
            "husky-controlled wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )
        if np.max(np.abs(self.vehicle_state.wheel_angular_speed.numpy())) <= 0.0:
            raise ValueError("Husky controls did not write wheel speed targets")

        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        planar_motion = np.linalg.norm(chassis_q[:, :2] - self._initial_chassis_q[:, :2], axis=1)
        if float(np.max(planar_motion)) < 0.2:
            raise ValueError("Husky did not translate under scripted commands")
        if self.sim_time >= 2.9:
            yaw_delta = np.array(
                [
                    abs(_angle_delta(_yaw_from_transform_row(current), _yaw_from_transform_row(initial)))
                    for current, initial in zip(chassis_q, self._initial_chassis_q, strict=True)
                ],
                dtype=np.float32,
            )
            if float(np.max(yaw_delta)) < 0.03:
                raise ValueError("Husky did not rotate under scripted commands")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=360, world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
