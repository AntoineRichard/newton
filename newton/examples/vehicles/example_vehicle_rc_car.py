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
# turns the car). The car drives forward while steering and follows a curve.
#
# Command: python -m newton.examples vehicle_rc_car
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
        # drive from the keyboard interactively; fall back to a scripted path under --test
        self._keyboard = not getattr(args, "test", False)

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
        self.viewer.set_model(self.model)
        if self._keyboard:
            print("Drive the RC car:  W/S or Up/Down = throttle/reverse,  A/D or Left/Right = steer,  Space = brake")

    def _command(self):
        if not self._keyboard:
            return 1.0, 0.6, 0.0  # scripted: drive forward and steer (used under --test)
        down = self.viewer.is_key_down
        drive = (1.0 if down("w") or down("up") else 0.0) - (1.0 if down("s") or down("down") else 0.0)
        steer = (1.0 if down("a") or down("left") else 0.0) - (1.0 if down("d") or down("right") else 0.0)
        brake = 1.0 if down("space") else 0.0
        return drive, steer, brake

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

    def render(self):
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
