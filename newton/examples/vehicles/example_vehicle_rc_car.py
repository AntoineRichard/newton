# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle RC Car (Ackermann, rc_car.usda)
#
# Loads the authored rc_car.usda fixture (a sprung, front-steer Ackermann RC car
# with real prismatic suspension, revolute steering, and physical axle joints)
# and drives it through the newton.vehicles layer. The asset's physical axle
# (wheel-spin) joints are converted to fixed via configure_wheel_axle_joints so
# wheel spin is analytical; suspension and steering remain solver joints. Wheels
# are annotated from the manifest labels.
#
# A follow camera tracks the car and a UI panel shows telemetry; untick
# "Cycle commands" for throttle/steering/brake sliders (W/A/S/D fly the camera).
#
# Command: python -m newton.examples vehicle_rc_car --viewer gl
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

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))


def _build():
    manifest = json.loads((_ASSET_DIR / "manifest.json").read_text())
    asset = next(a for a in manifest["assets"] if a["name"] == "rc_car")
    rd = asset["reference_dimensions"]

    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    builder.add_ground_plane(cfg=terrain_cfg)

    builder.add_usd(str(_ASSET_DIR / asset["file"]))
    # physical axle (wheel-spin) joints -> fixed, so wheel spin is analytical
    nv.configure_wheel_axle_joints(builder, axle_joint_labels=asset["axle_joint_labels"])

    joint_by_label = {label: i for i, label in enumerate(builder.joint_label)}
    shape_by_label = {label: i for i, label in enumerate(builder.shape_label)}
    nv.set_vehicle(
        builder,
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
            builder,
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

    model = builder.finalize()
    joint_type = model.joint_type.numpy()
    joint_child = model.joint_child.numpy()
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

        self.model, self._chassis = _build()
        # Drivetrain is modelled on a Traxxas Slash 4x4 VXL: Castle 1415 2400 Kv motor
        # (torque constant Kt = 9.55/Kv = 3.98 mN*m/A) on 4S (~14.8 V nominal) through
        # the stock 11.82:1 final drive at ~85% efficiency.
        #   max_wheel_speed = Kv * V * 2pi/60 / FDR = 2400 * 14.8 * 0.1047 / 11.82
        #                   ~= 315 rad/s  -> top speed ~= w*r = 315 * 0.055 ~= 17 m/s
        #   motor_max_torque = Kt * I_peak * FDR * eta / 4_wheels
        #                    = 0.00398 * 100 A * 11.82 * 0.85 / 4 ~= 1.0 N*m per wheel
        # ~100 A is a typical launch burst for this class (the 4S/90C pack is not the
        # limit); refine from a logged peak current. angular_damping is kept low because
        # the 85% efficiency already accounts for drivetrain loss -- a large value would
        # cap the top speed at motor_max_torque/angular_damping rather than the no-load
        # wheel speed.
        self.vehicles = nv.WheeledVehicles(
            self.model,
            config=nv.WheeledConfig(
                max_wheel_speed=315.0,
                motor_max_torque=1.0,
                angular_damping=0.0005,
                # Grippy RC tires: mu ~ 1 (one g of grip) is a realistic, well-behaved
                # value for a soft RC compound on a grippy surface; rubber can exceed 1
                # and mu ~= peak lateral g, so measure it on the real car and slide up if
                # the surface warrants (higher mu means larger tire forces, which stresses
                # the explicit tire integration -- raise sim_substeps if you push it). The
                # lateral slip stiffness is set ~2x longitudinal so the car turns in
                # crisply; both still saturate at the same mu*Fz circle, so this shapes how
                # fast grip builds, not its maximum.
                friction=1.0,
                longitudinal_stiffness=20.0,
                lateral_stiffness=40.0,
            ),
        )
        self.vehicles.configure_solver_contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=256, nconmax=128)

        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._initial = self.state_0.body_q.numpy()[self._chassis].copy()

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
        self._prev_yaw = _yaw(self._initial)

        # live handling tuning, pushed into the controller's per-wheel arrays each
        # interactive step (sliders in the UI panel); initialized from the config
        cfg = self.vehicles.config
        self.linear_tire = False
        self.tune_c_long = cfg.longitudinal_stiffness
        self.tune_c_lat = cfg.lateral_stiffness
        self.tune_friction = cfg.friction if cfg.friction > 0.0 else 1.0  # tire mu
        self.tune_motor_torque = cfg.motor_max_torque
        self.tune_top_speed = cfg.max_wheel_speed

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
        ui.text("Handling")
        _changed, self.linear_tire = ui.checkbox("Linear tire (else brush)", self.linear_tire)
        _changed, self.tune_c_long = ui.slider_float("Long. stiffness", self.tune_c_long, 1.0, 100.0)
        _changed, self.tune_c_lat = ui.slider_float("Lat. stiffness", self.tune_c_lat, 1.0, 100.0)
        _changed, self.tune_friction = ui.slider_float("Tire mu", self.tune_friction, 0.2, 3.0)
        _changed, self.tune_motor_torque = ui.slider_float("Motor torque [N*m]", self.tune_motor_torque, 0.2, 8.0)
        _changed, self.tune_top_speed = ui.slider_float("Top wheel speed [rad/s]", self.tune_top_speed, 50.0, 400.0)
        ui.separator()
        ui.text("Telemetry")
        ui.text(f"Speed: {self._speed:.2f} m/s")
        ui.text(f"Yaw rate: {math.degrees(self._yaw_rate):.1f} deg/s")
        ui.text(f"Wheel omega: {self._omega:.1f} rad/s")
        ui.text(f"Slip ratio: {self._slip:.2f}")

    def _command(self):
        if not self._interactive:
            # scripted under --test: settle, launch straight, then steer. Steering
            # starts late so test_final's world-frame dx check measures the straight
            # launch, not how far around a circle the car happens to end up.
            if self.sim_time < 0.5:
                return 0.0, 0.0, 0.0
            if self.sim_time < 3.0:
                return 1.0, 0.0, 0.0
            return 1.0, 0.6, 0.0
        if not self.cycle_enabled:
            return self.manual_drive, self.manual_steer, self.manual_brake
        cycle = self.sim_time % 8.0
        if cycle < 3.0:
            return 0.8, 0.0, 0.0
        if cycle < 5.5:
            return 0.6, 0.7, 0.0
        if cycle < 6.5:
            return 0.0, 0.0, 1.0
        return 0.6, -0.7, 0.0

    def _apply_tuning(self):
        # push the UI handling values into the controller's per-wheel arrays (all
        # wheels identical for this car); takes effect on the next substep.
        dyn = self.vehicles.dynamics
        dyn.tire_model.fill_(int(nv.TireModel.LINEAR if self.linear_tire else nv.TireModel.BRUSH))
        dyn.c_long.fill_(float(self.tune_c_long))
        dyn.c_lat.fill_(float(self.tune_c_lat))
        dyn.mu_override.fill_(float(self.tune_friction))
        dyn.tau_max.fill_(float(self.tune_motor_torque))
        dyn.max_speed.fill_(float(self.tune_top_speed))

    def step(self):
        if self._interactive:
            self._apply_tuning()
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
        kappa = self.vehicles.dynamics.kappa.numpy()
        self._omega = float(np.max(np.abs(omega))) if omega.size else 0.0
        self._slip = float(np.max(np.abs(kappa))) if kappa.size else 0.0

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        q = self.state_0.body_q.numpy()[self._chassis]
        yaw = _yaw(q)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = q[:3] - 2.8 * forward + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.viewer.set_camera(
            pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])), pitch=-18.0, yaw=math.degrees(yaw)
        )

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
        fz = self.vehicles.patch.fz.numpy()
        if not np.isfinite(fz).all() or float(fz.min()) <= 0.0:
            raise ValueError(f"unexpected wheel loads {fz}")
        dx = float(q[0] - self._initial[0])
        yaw = _yaw(q) - _yaw(self._initial)
        if dx < 0.2:
            raise ValueError(f"rc car did not drive forward (dx {dx:.3f} m)")
        if abs(yaw) < 0.1:
            raise ValueError(f"rc car did not turn while steering (yaw {yaw:.3f} rad)")

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
