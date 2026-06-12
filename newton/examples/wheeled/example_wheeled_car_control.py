# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Car Control
#
# Drives the simplified Ackermann RC car fixture through the vehicle
# command-channel mapper. Axle joints are fixed; analytical wheel speeds are
# integrated from drive torque and tire reaction moments, and front steering
# targets are written to the solver control.
#
# Command: python -m newton.examples wheeled_car_control --tire-model fiala
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
TIRE_FRICTION_MU = 1.8
TIRE_FALLBACK_NORMAL_LOAD = 10.0
TIRE_LONGITUDINAL_STIFFNESS = 120.0
TIRE_LINEAR_LATERAL_STIFFNESS = 140.0
TIRE_FIALA_CORNERING_STIFFNESS = 350.0
MAX_WHEEL_ANGULAR_SPEED = 14.0
MAX_WHEEL_DRIVE_TORQUE = 0.08
COAST_BRAKE_TORQUE = 0.03
WHEEL_INERTIA = 0.01
WHEEL_ANGULAR_DAMPING = 0.01
WHEEL_ROLLING_RESISTANCE_TORQUE = 0.008


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
        self.tire_model = args.tire_model
        self.drive_scale = args.drive_scale
        self.max_wheel_drive_torque = MAX_WHEEL_DRIVE_TORQUE
        self.tire_friction_mu = args.tire_friction_mu
        self.tire_longitudinal_stiffness = args.tire_longitudinal_stiffness
        if args.tire_lateral_stiffness is None:
            self.tire_lateral_stiffness = TIRE_FIALA_CORNERING_STIFFNESS
            if self.tire_model == "linear":
                self.tire_lateral_stiffness = TIRE_LINEAR_LATERAL_STIFFNESS
        else:
            self.tire_lateral_stiffness = args.tire_lateral_stiffness
        self.cycle_enabled = True
        self.follow_camera = True
        self.manual_drive = 0.0
        self.manual_steering = 0.0

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
        self._configure_tire_control()
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
        self._initial_chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices].copy()
        self._max_wheel_radius = max(self.wheeled_metadata.wheel_radius, default=0.0)
        self._previous_diag_speed = 0.0
        self._previous_diag_yaw = _yaw_from_transform_row(self._initial_chassis_q[0])
        self._previous_diag_time = self.sim_time
        self._diag_speed = 0.0
        self._diag_accel = 0.0
        self._diag_yaw_rate = 0.0
        self._diag_wheel_angular_speed = 0.0
        self._diag_wheel_surface_speed = 0.0
        self._diag_longitudinal_slip = 0.0
        self._diag_lateral_slip_angle = 0.0
        self._diag_combined_slip_scale = 1.0
        self._diag_drive_torque = 0.0

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((1.4, 1.2, 0.0))
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

        self.drive_commands = np.zeros(self.vehicle_layout.drive_channel_count, dtype=np.float32)
        self.steering_commands = np.zeros(self.vehicle_layout.steering_channel_count, dtype=np.float32)
        expected_wheels = self.world_count * len(asset.wheel_body_labels)
        if self.wheeled_metadata.wheel_count != expected_wheels:
            raise ValueError(f"expected {expected_wheels} wheels, found {self.wheeled_metadata.wheel_count}")

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow Camera", self.follow_camera)
        _changed, self.cycle_enabled = ui.checkbox("Cycle commands", self.cycle_enabled)
        _changed, self.drive_scale = ui.slider_float("Drive scale", self.drive_scale, 0.0, 8.0)
        tire_changed, self.tire_friction_mu = ui.slider_float("Tire mu", self.tire_friction_mu, 0.2, 3.0)
        long_changed, self.tire_longitudinal_stiffness = ui.slider_float(
            "Longitudinal stiffness", self.tire_longitudinal_stiffness, 0.0, 500.0
        )
        lat_changed, self.tire_lateral_stiffness = ui.slider_float(
            "Lateral stiffness", self.tire_lateral_stiffness, 0.0, 800.0
        )
        if tire_changed or long_changed or lat_changed:
            self._configure_tire_control()
        if not self.cycle_enabled:
            _changed, self.manual_drive = ui.slider_float("Drive", self.manual_drive, -1.0, 1.0)
            _changed, self.manual_steering = ui.slider_float("Steering", self.manual_steering, -1.0, 1.0)

        ui.separator()
        ui.text("Diagnostics")
        ui.text(f"Speed: {self._diag_speed:.2f} m/s")
        ui.text(f"Accel: {self._diag_accel:.2f} m/s^2")
        ui.text(f"Yaw rate: {math.degrees(self._diag_yaw_rate):.1f} deg/s")
        ui.text(f"Wheel omega: {self._diag_wheel_angular_speed:.1f} rad/s")
        ui.text(f"Wheel surface: {self._diag_wheel_surface_speed:.2f} m/s")
        ui.text(f"Drive torque: {self._diag_drive_torque:.2f} N*m")
        ui.text(f"Slip ratio: {self._diag_longitudinal_slip:.2f}")
        ui.text(f"Slip angle: {math.degrees(self._diag_lateral_slip_angle):.1f} deg")
        ui.text(f"Force scale: {self._diag_combined_slip_scale:.2f}")

    def _configure_tire_control(self):
        newton.wheeled.configure_wheel_tire_control(
            self.tire_control,
            tire_model=self.tire_model,
            friction_mu=self.tire_friction_mu,
            fallback_normal_load=TIRE_FALLBACK_NORMAL_LOAD,
            longitudinal_stiffness=self.tire_longitudinal_stiffness,
            lateral_stiffness=self.tire_lateral_stiffness,
        )

    def _update_moment_drive_commands(self):
        self.max_wheel_drive_torque = MAX_WHEEL_DRIVE_TORQUE * max(self.drive_scale, 0.0)
        wp.launch(
            _update_moment_drive_commands,
            dim=self.wheeled_metadata.wheel_count,
            inputs=[
                self.vehicle_state.wheel_drive_command,
                self.moment_state.wheel_angular_speed,
                self.max_wheel_drive_torque,
                COAST_BRAKE_TORQUE,
            ],
            outputs=[
                self.moment_control.drive_torque,
                self.moment_control.brake_torque,
                self.tire_control.wheel_angular_speed,
            ],
            device=self.model.device,
        )

    def _update_vehicle_commands(self):
        if self.cycle_enabled:
            cycle_time = self.sim_time % 8.0
            if cycle_time < 2.0:
                drive = 0.65
                steering = 0.0
            elif cycle_time < 3.4:
                drive = 0.4
                steering = 0.25
            elif cycle_time < 5.2:
                drive = 0.55
                steering = 0.0
            elif cycle_time < 6.4:
                drive = -0.35
                steering = 0.0
            else:
                drive = 0.0
                steering = 0.0
        else:
            drive = self.manual_drive
            steering = self.manual_steering

        self.drive_commands.fill(drive)
        self.steering_commands.fill(steering)
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
        camera_pos = car_pos - 2.8 * forward + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.viewer.set_camera(pos=wp.vec3(*camera_pos), pitch=-18.0, yaw=math.degrees(yaw))

    def _update_diagnostics(self):
        chassis_q = self.state_0.body_q.numpy()[self._tracked_chassis_body]
        chassis_qd = self.state_0.body_qd.numpy()[self._tracked_chassis_body]
        speed = float(np.linalg.norm(chassis_qd[:2]))
        dt = max(float(self.sim_time - self._previous_diag_time), self.frame_dt)
        yaw = _yaw_from_transform_row(chassis_q)

        wheel_angular_speed = self.moment_state.wheel_angular_speed.numpy()
        if wheel_angular_speed.size:
            max_wheel_angular_speed = float(np.max(np.abs(wheel_angular_speed)))
        else:
            max_wheel_angular_speed = 0.0

        wheel_count = int(self.wheeled_metadata.wheel_count)
        if wheel_count:
            active = self.patch_state.active.numpy().astype(bool)
            mask = active if np.any(active) else np.ones(wheel_count, dtype=bool)

            longitudinal_slip = self.tire_state.longitudinal_slip_ratio.numpy()
            lateral_slip_angle = self.tire_state.lateral_slip_angle.numpy()
            combined_slip_scale = self.tire_state.combined_slip_scale.numpy()
            applied_longitudinal = self.tire_state.applied_longitudinal_force.numpy()
            applied_lateral = self.tire_state.applied_lateral_force.numpy()
            drive_torque = self.moment_control.drive_torque.numpy()

            force_mask = mask & ((np.abs(applied_longitudinal) + np.abs(applied_lateral)) > 1.0e-6)
            if np.any(force_mask):
                force_scale = float(np.min(combined_slip_scale[force_mask]))
            else:
                force_scale = 1.0

            max_longitudinal_slip = float(np.max(np.abs(longitudinal_slip[mask])))
            max_lateral_slip_angle = float(np.max(np.abs(lateral_slip_angle[mask])))
            max_drive_torque = float(np.max(np.abs(drive_torque)))
        else:
            max_longitudinal_slip = 0.0
            max_lateral_slip_angle = 0.0
            force_scale = 1.0
            max_drive_torque = 0.0

        self._diag_speed = speed
        self._diag_accel = (speed - self._previous_diag_speed) / dt
        self._diag_yaw_rate = _angle_delta(yaw, self._previous_diag_yaw) / dt
        self._diag_wheel_angular_speed = max_wheel_angular_speed
        self._diag_wheel_surface_speed = max_wheel_angular_speed * self._max_wheel_radius
        self._diag_longitudinal_slip = max_longitudinal_slip
        self._diag_lateral_slip_angle = max_lateral_slip_angle
        self._diag_combined_slip_scale = force_scale
        self._diag_drive_torque = max_drive_torque
        self._previous_diag_speed = speed
        self._previous_diag_yaw = yaw
        self._previous_diag_time = self.sim_time

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
            self._update_moment_drive_commands()
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

        self.sim_time += self.frame_dt
        self._update_diagnostics()

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
            "car-controlled wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )
        if np.max(np.abs(self.moment_state.wheel_angular_speed.numpy())) <= 0.0:
            raise ValueError("car controls did not integrate wheel moment speeds")
        diagnostics = (
            self._diag_speed,
            self._diag_accel,
            self._diag_yaw_rate,
            self._diag_wheel_angular_speed,
            self._diag_wheel_surface_speed,
            self._diag_longitudinal_slip,
            self._diag_lateral_slip_angle,
            self._diag_combined_slip_scale,
        )
        if not all(math.isfinite(value) for value in diagnostics):
            raise ValueError("car control diagnostics are not finite")

        chassis_q = self.state_0.body_q.numpy()[self._chassis_body_indices]
        planar_motion = np.linalg.norm(chassis_q[:, :2] - self._initial_chassis_q[:, :2], axis=1)
        if float(np.max(planar_motion)) < 0.2:
            raise ValueError("car did not translate under scripted commands")
        if self.sim_time >= 2.9:
            yaw_delta = np.array(
                [
                    abs(_angle_delta(_yaw_from_transform_row(current), _yaw_from_transform_row(initial)))
                    for current, initial in zip(chassis_q, self._initial_chassis_q, strict=True)
                ],
                dtype=np.float32,
            )
            if float(np.max(yaw_delta)) < 0.03:
                raise ValueError("car did not steer under scripted commands")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--tire-model",
            choices=("linear", "fiala"),
            default="linear",
            help="Analytical tire model for lateral force generation.",
        )
        parser.add_argument(
            "--drive-scale",
            type=float,
            default=1.0,
            help="Scale applied to the maximum analytical wheel drive torque.",
        )
        parser.add_argument(
            "--tire-friction-mu",
            type=float,
            default=TIRE_FRICTION_MU,
            help="Tire friction override used by the analytical tire model.",
        )
        parser.add_argument(
            "--tire-longitudinal-stiffness",
            type=float,
            default=TIRE_LONGITUDINAL_STIFFNESS,
            help="Longitudinal slip stiffness used by the analytical tire model.",
        )
        parser.add_argument(
            "--tire-lateral-stiffness",
            type=float,
            default=None,
            help="Lateral stiffness override; defaults depend on --tire-model.",
        )
        parser.set_defaults(num_frames=360, world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
