# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle Husky (skid-steer)
#
# A four-wheel skid-steer vehicle driven through the newton.vehicles layer.
# The wrapped MuJoCo solver owns collision and normal support (Newton-detected
# contacts, condim=1 on the wheels); the WheeledVehicles controller owns the
# analytical wheel spin and the brush tire forces. The vehicle drives forward,
# then spins in place from a left/right wheel-speed differential.
#
# A follow camera tracks the (world-0) chassis and a UI panel shows telemetry;
# untick "Cycle commands" for throttle/steering/brake sliders (W/A/S/D fly the
# camera). Pass --world-count N to replicate the vehicle.
#
# Command: python -m newton.examples vehicle_husky
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.vehicles as nv

_R = 0.09  # wheel radius [m]
_CORNERS = ((0.25, 0.18), (0.25, -0.18), (-0.25, 0.18), (-0.25, -0.18))


def _build(num_worlds):
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    builder.add_ground_plane(cfg=terrain_cfg)

    vehicle = newton.ModelBuilder()
    nv.register_vehicle_attributes(vehicle)
    newton.solvers.SolverMuJoCo.register_custom_attributes(vehicle)
    car = vehicle.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _R), wp.quat_identity()))
    chassis_cfg = newton.ModelBuilder.ShapeConfig()
    chassis_cfg.has_shape_collision = False
    vehicle.add_shape_box(car, xform=wp.transform(), hx=0.22, hy=0.16, hz=0.05, cfg=chassis_cfg)
    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    nv.set_vehicle(vehicle, 0, drive_mode=int(nv.DriveMode.SKID_STEER), track_width=0.36)
    for i, (x, y) in enumerate(_CORNERS):
        s = vehicle.add_shape_cylinder(car, xform=wp.transform(wp.vec3(x, y, 0.0), axis_q), radius=_R, half_height=0.04)
        nv.add_wheel(
            vehicle,
            shape=s,
            vehicle_id=0,
            wheel_id=i,
            radius=_R,
            width=0.08,
            driven=True,
            side=(-1 if y > 0 else 1),
            axle_row=(0 if x > 0 else 1),
        )

    spacing = 1.5
    for w in range(num_worlds):
        builder.add_builder(vehicle, xform=wp.transform(wp.vec3(float(w) * spacing, 0.0, 0.0), wp.quat_identity()))
    model = builder.finalize()
    joint_type = model.joint_type.numpy()
    joint_child = model.joint_child.numpy()
    # world-0 chassis = child of the first free joint (camera/telemetry target)
    chassis = int(joint_child[list(joint_type).index(int(newton.JointType.FREE))])
    return model, chassis


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self._interactive = not getattr(args, "test", False)

        self.model, self._chassis = _build(getattr(args, "world_count", 1))
        self.vehicles = nv.WheeledVehicles(self.model, config=nv.WheeledConfig(max_wheel_speed=10.0))
        self.vehicles.configure_solver_contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=256, nconmax=128)

        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial = self.state_0.body_q.numpy().copy()
        self._max_disp = 0.0
        self._max_yaw = 0.0

        # interactive control + telemetry (driven from the UI panel)
        self.follow_camera = True
        self.cycle_enabled = True  # scripted demo loop; uncheck to drive with sliders
        self.manual_drive = 0.0
        self.manual_steer = 0.0
        self.manual_brake = 0.0
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._omega = 0.0
        self._slip = 0.0
        self._prev_yaw = _yaw(self._initial[self._chassis])

        self.viewer.set_model(self.model)
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        _changed, self.cycle_enabled = ui.checkbox("Cycle commands", self.cycle_enabled)
        if not self.cycle_enabled:
            _changed, self.manual_drive = ui.slider_float("Throttle", self.manual_drive, -1.0, 1.0)
            _changed, self.manual_steer = ui.slider_float("Steering", self.manual_steer, -1.0, 1.0)
            _changed, self.manual_brake = ui.slider_float("Brake", self.manual_brake, 0.0, 1.0)
        ui.separator()
        ui.text("Telemetry")
        ui.text(f"Speed: {self._speed:.2f} m/s")
        ui.text(f"Yaw rate: {math.degrees(self._yaw_rate):.1f} deg/s")
        ui.text(f"Wheel omega: {self._omega:.1f} rad/s")
        ui.text(f"Slip ratio: {self._slip:.2f}")

    def _command(self):
        if not self._interactive:
            # scripted under --test: drive forward, then spin in place
            return (1.0, 0.0, 0.0) if self.sim_time < 1.5 else (0.0, 1.0, 0.0)
        if not self.cycle_enabled:
            return self.manual_drive, self.manual_steer, self.manual_brake
        cycle = self.sim_time % 8.0
        if cycle < 3.0:
            return 0.9, 0.0, 0.0
        if cycle < 5.0:
            return 0.0, 1.0, 0.0
        if cycle < 6.0:
            return 0.0, 0.0, 1.0
        return 0.0, -1.0, 0.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.vehicles.update_controls(self.control)
            self.model.collide(self.state_0, self.contacts)
            self.vehicles.apply(self.state_0, self.contacts, self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)
            self.vehicles.latch_loads(self.contacts)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        drive, steer, brake = self._command()
        self.vehicles.set_commands(drive=drive, steer=steer, brake=brake)
        self.simulate()
        self.sim_time += self.frame_dt
        self._track()
        self._update_telemetry()

    def _track(self):
        q = self.state_0.body_q.numpy()
        disp = np.linalg.norm(q[:, :2] - self._initial[:, :2], axis=1)
        self._max_disp = max(self._max_disp, float(np.max(disp)))
        for cur, init in zip(q, self._initial, strict=True):
            self._max_yaw = max(self._max_yaw, abs(_yaw(cur) - _yaw(init)))

    def _update_telemetry(self):
        qd = self.state_0.body_qd.numpy()[self._chassis]
        self._speed = float(np.linalg.norm(qd[:2]))
        yaw = _yaw(self.state_0.body_q.numpy()[self._chassis])
        self._yaw_rate = ((yaw - self._prev_yaw + math.pi) % (2.0 * math.pi) - math.pi) / self.frame_dt
        self._prev_yaw = yaw
        omega = self.vehicles.dynamics.omega.numpy()
        kappa = self.vehicles.dynamics.kappa.numpy()
        self._omega = float(np.max(np.abs(omega))) if omega.size else 0.0
        self._slip = float(np.max(np.abs(kappa))) if kappa.size else 0.0

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        q = self.state_0.body_q.numpy()[self._chassis]
        yaw = _yaw(q)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = q[:3] - 2.0 * forward + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.viewer.set_camera(
            pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])), pitch=-18.0, yaw=math.degrees(yaw)
        )

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("non-finite body poses")
        if not np.isfinite(self.state_0.body_qd.numpy()).all():
            raise ValueError("non-finite body velocities")
        utilization = self.vehicles.dynamics.impulse_utilization.numpy()
        if not np.isfinite(utilization).all():
            raise ValueError(f"non-finite impulse utilization {utilization}")
        if float(utilization.min()) < 0.0 or float(utilization.max()) > 1.0 + 1e-4:
            raise ValueError(f"impulse utilization outside [0, 1]: {utilization}")

    def test_final(self):
        qd = self.state_0.body_qd.numpy()
        if not np.isfinite(qd).all():
            raise ValueError("non-finite velocities")
        if self._max_disp < 0.3:
            raise ValueError(f"husky did not drive forward (max displacement {self._max_disp:.3f} m)")
        if self._max_yaw < 0.2:
            raise ValueError(f"husky did not rotate (max yaw {self._max_yaw:.3f} rad)")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=240, world_count=1)
        return parser


def _yaw(transform_row):
    x, y, z, w = transform_row[3], transform_row[4], transform_row[5], transform_row[6]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
