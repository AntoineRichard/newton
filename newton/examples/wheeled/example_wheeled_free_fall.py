# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Free Fall
#
# Drops the simplified Ackermann RC car fixture with no terrain or ground.
# Wheel spin is integrated analytically from drive torque and tire reaction
# moments, front steering targets are written to the solver control, and the
# camera follows the car through repeated free-fall resets.
#
# Command: python -m newton.examples wheeled_free_fall
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

RESET_HEIGHT = -100.0
TIRE_FRICTION_MU = 1.0
TIRE_FALLBACK_NORMAL_LOAD = 14.0
TIRE_LONGITUDINAL_STIFFNESS = 70.0
TIRE_LATERAL_STIFFNESS = 60.0
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


def _configure_zero_gap_builder(builder: newton.ModelBuilder) -> None:
    builder.rigid_gap = 0.0
    builder.default_shape_cfg.gap = 0.0
    builder.default_shape_cfg.margin = 0.0


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
        self._max_moment_wheel_speed = 0.0
        self._reset_count = 0

        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(MANIFEST_PATH)}
        asset = assets[VEHICLE_NAME]

        world = newton.ModelBuilder()
        _configure_zero_gap_builder(world)
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        newton.wheeled.register_wheeled_custom_attributes(world)
        world.add_usd(
            str(asset.file),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=False,
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
        )
        newton.wheeled.apply_wheeled_manifest(world, MANIFEST_PATH, asset_names=(VEHICLE_NAME,))
        newton.wheeled.configure_wheel_axle_joints(world, axle_joint_labels=asset.axle_joint_labels)

        scene = newton.ModelBuilder()
        _configure_zero_gap_builder(scene)
        scene.replicate(world, self.world_count)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self.state_1.assign(self.state_0)
        self._initial_model_joint_q = wp.clone(self.model.joint_q)
        self._initial_model_joint_qd = wp.clone(self.model.joint_qd)

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
            njmax=max(256 * self.world_count, self.model.rigid_contact_max),
            nconmax=max(128 * self.world_count, self.model.rigid_contact_max),
            iterations=20,
            ls_iterations=100,
        )

        self._chassis_body_indices = [
            body_index for body_index, label in enumerate(self.model.body_label) if str(label).endswith("chassis")
        ]
        self._tracked_chassis_body = self._chassis_body_indices[0]
        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        self._initial_chassis_z = float(np.min(chassis_q[:, 2]))
        self._min_chassis_z = self._initial_chassis_z

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
            if cycle_time < 3.0:
                drive = 0.85 * self.drive_scale
                steering = 0.0
            elif cycle_time < 5.0:
                drive = 0.55 * self.drive_scale
                steering = 0.45
            elif cycle_time < 6.5:
                drive = -0.6 * self.drive_scale
                steering = -0.35
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
        camera_pos = car_pos - 3.0 * forward + np.array([0.0, 0.0, 1.4], dtype=np.float32)
        self.viewer.set_camera(pos=wp.vec3(*camera_pos), pitch=-20.0, yaw=math.degrees(yaw))

    def _clear_transient_state(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.control.clear()
        self.contacts.clear()
        self.patch_state.clear()
        self.tire_state.clear(clear_previous_normal_load=True)
        self.moment_state.clear(clear_wheel_angular_speed=True)
        self.vehicle_state.clear()
        self.tire_control.wheel_angular_speed.zero_()
        self.moment_control.drive_torque.zero_()
        self.moment_control.brake_torque.zero_()

    def _reset_if_too_low(self):
        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        min_z = float(np.min(chassis_q[:, 2]))
        self._min_chassis_z = min(self._min_chassis_z, min_z)
        if min_z >= RESET_HEIGHT:
            return

        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.model.joint_q.assign(self._initial_model_joint_q)
        self.model.joint_qd.assign(self._initial_model_joint_qd)
        self._clear_transient_state()
        self._reset_count += 1

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

        self._max_moment_wheel_speed = max(
            self._max_moment_wheel_speed,
            float(np.max(np.abs(self.moment_state.wheel_angular_speed.numpy()))),
        )
        self._reset_if_too_low()
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
            "free-falling wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 150.0,
        )
        if self._max_moment_wheel_speed <= 0.0:
            raise ValueError("free-fall car controls did not spin analytical wheel moments")
        if self.sim_time >= 1.0 and self._min_chassis_z > self._initial_chassis_z - 1.0:
            raise ValueError("free-fall car did not fall under gravity")
        if self.sim_time >= 4.8 and self._reset_count == 0:
            raise ValueError(f"free-fall car did not reset below {RESET_HEIGHT} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=480, world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
