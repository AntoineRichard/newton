# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle RC Car (Ackermann)
#
# A rear-wheel-drive, front-steer Ackermann car driven through the
# newton.vehicles layer. The front wheels are mounted on real revolute steering
# joints (PD position servos tracked by the MuJoCo solver); the command mapper
# writes per-wheel Ackermann inner/outer angles. The wrapped solver owns
# collision and normal support; the WheeledVehicles controller owns the
# analytical wheel spin and the brush tire forces (front-wheel lateral grip
# turns the car). A follow camera tracks the car and a UI panel shows telemetry
# (speed, yaw rate, wheel speed); tick "Manual control" for throttle/steering/
# brake sliders, otherwise a scripted demo loop drives it. (W/A/S/D fly the
# camera, so driving is via the panel sliders.)
#
# Command: python -m newton.examples vehicle_rc_car --viewer gl
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.vehicles as nv

_R = 0.06  # wheel radius [m]
_HALF_WB = 0.16  # half wheelbase [m]
_HALF_TRACK = 0.12  # half track [m]


def _build():
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    builder.add_ground_plane(cfg=terrain_cfg)

    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    chassis = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, _R), wp.quat_identity()))
    chassis_cfg = newton.ModelBuilder.ShapeConfig()
    chassis_cfg.has_shape_collision = False
    builder.add_shape_box(chassis, xform=wp.transform(), hx=0.16, hy=0.1, hz=0.04, cfg=chassis_cfg)
    chassis_free = builder.add_joint_free(child=chassis)

    nv.set_vehicle(
        builder,
        0,
        drive_mode=int(nv.DriveMode.ACKERMANN),
        wheelbase=2.0 * _HALF_WB,
        track_width=2.0 * _HALF_TRACK,
        steer_limit=0.5,
    )

    def wheel_shape(body):
        return builder.add_shape_cylinder(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axis_q), radius=_R, half_height=0.025
        )

    # rear wheels: rigidly attached to the chassis, driven
    for i, y in enumerate((_HALF_TRACK, -_HALF_TRACK)):
        s = builder.add_shape_cylinder(
            chassis, xform=wp.transform(wp.vec3(-_HALF_WB, y, 0.0), axis_q), radius=_R, half_height=0.025
        )
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=0,
            wheel_id=i,
            radius=_R,
            width=0.05,
            driven=True,
            steerable=False,
            side=(-1 if y > 0 else 1),
            axle_row=1,
        )

    # front wheels: separate bodies on vertical revolute steering joints (PD servo)
    steer_joints = []
    for j, y in enumerate((_HALF_TRACK, -_HALF_TRACK)):
        fw = builder.add_link(xform=wp.transform(wp.vec3(_HALF_WB, y, _R), wp.quat_identity()))
        steer_joint = builder.add_joint_revolute(
            parent=chassis,
            child=fw,
            parent_xform=wp.transform(wp.vec3(_HALF_WB, y, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            target_ke=40.0,
            target_kd=4.0,
            limit_lower=-0.7,
            limit_upper=0.7,
        )
        steer_joints.append(steer_joint)
        s = wheel_shape(fw)
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=0,
            wheel_id=2 + j,
            radius=_R,
            width=0.05,
            driven=False,
            steerable=True,
            side=(-1 if y > 0 else 1),
            axle_row=0,
            steer_joint=steer_joint,
        )
    builder.add_articulation([chassis_free, *steer_joints], label="rc_car")
    return builder.finalize()


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        # interactive control via the UI panel; scripted path under --test
        self._interactive = not getattr(args, "test", False)

        self.model = _build()
        self.vehicles = nv.WheeledVehicles(self.model, config=nv.WheeledConfig(max_wheel_speed=14.0))
        self.vehicles.configure_solver_contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=256, nconmax=128)

        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._chassis = 0  # chassis is body 0
        self._initial = self.state_0.body_q.numpy()[self._chassis].copy()

        # control + telemetry state (driven from the UI panel; W/A/S/D stay with the camera)
        self.follow_camera = True
        self.manual = False
        self.manual_drive = 0.0
        self.manual_steer = 0.0
        self.manual_brake = 0.0
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._omega = 0.0
        self._prev_yaw = _yaw(self._initial)

        self.viewer.set_model(self.model)
        self._set_follow_camera()

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        _changed, self.manual = ui.checkbox("Manual control", self.manual)
        if self.manual:
            _changed, self.manual_drive = ui.slider_float("Throttle", self.manual_drive, -1.0, 1.0)
            _changed, self.manual_steer = ui.slider_float("Steering", self.manual_steer, -1.0, 1.0)
            _changed, self.manual_brake = ui.slider_float("Brake", self.manual_brake, 0.0, 1.0)
        else:
            ui.text("(scripted demo - tick 'Manual control' to drive)")
        ui.separator()
        ui.text("Telemetry")
        ui.text(f"Speed: {self._speed:.2f} m/s")
        ui.text(f"Yaw rate: {math.degrees(self._yaw_rate):.1f} deg/s")
        ui.text(f"Wheel omega: {self._omega:.1f} rad/s")

    def _command(self):
        if not self._interactive:
            return 1.0, 0.6, 0.0  # scripted under --test: drive forward and steer
        if self.manual:
            return self.manual_drive, self.manual_steer, self.manual_brake
        # gentle scripted demo loop until the user ticks "Manual control"
        cycle = self.sim_time % 8.0
        if cycle < 3.0:
            return 0.7, 0.0, 0.0
        if cycle < 5.5:
            return 0.5, 0.7, 0.0
        if cycle < 6.5:
            return 0.0, 0.0, 1.0
        return 0.6, -0.7, 0.0

    def step(self):
        drive, steer, brake = self._command()
        self.vehicles.set_commands(drive=drive, steer=steer, brake=brake)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.vehicles.update_controls(self.control)
            self.model.collide(self.state_0, self.contacts)
            self.vehicles.apply(self.state_0, self.contacts, self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)
            self.vehicles.latch_loads(self.contacts)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        self._update_telemetry()

    def _update_telemetry(self):
        q = self.state_0.body_q.numpy()[self._chassis]
        qd = self.state_0.body_qd.numpy()[self._chassis]
        self._speed = float(np.linalg.norm(qd[:2]))
        yaw = _yaw(q)
        self._yaw_rate = ((yaw - self._prev_yaw + math.pi) % (2.0 * math.pi) - math.pi) / self.frame_dt
        self._prev_yaw = yaw
        omega = self.vehicles.dynamics.omega.numpy()
        self._omega = float(np.max(np.abs(omega))) if omega.size else 0.0

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        q = self.state_0.body_q.numpy()[self._chassis]
        yaw = _yaw(q)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = q[:3] - 1.4 * forward + np.array([0.0, 0.0, 0.6], dtype=np.float32)
        self.viewer.set_camera(pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])), pitch=-20.0, yaw=math.degrees(yaw))

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.body_q.numpy()[self._chassis]
        if not np.isfinite(q).all():
            raise ValueError("non-finite chassis pose")
        dx = float(q[0] - self._initial[0])
        yaw = _yaw(q) - _yaw(self._initial)
        if dx < 0.2:
            raise ValueError(f"car did not drive forward (dx {dx:.3f} m)")
        if abs(yaw) < 0.1:
            raise ValueError(f"car did not turn while steering (yaw {yaw:.3f} rad)")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


def _yaw(transform_row):
    x, y, z, w = transform_row[3], transform_row[4], transform_row[5], transform_row[6]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
