# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle Sprung (Ackermann with real suspension)
#
# An Ackermann car with real prismatic spring/damper suspension on every wheel
# (handled by the MuJoCo solver) and revolute steering joints on the front. The
# newton.vehicles layer is agnostic to the suspension: it reads each wheel body's
# pose and applies the brush tire forces, and the suspension joint transmits the
# reaction to the chassis. Because every wheel sits on its own spring, the load
# distribution is determinate, so the contact-load smoothing band-aid is turned
# off here (load_filter=1.0) and the per-wheel loads stay even.
#
# Command: python -m newton.examples vehicle_sprung
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.vehicles as nv

_R = 0.06  # wheel radius [m]
_H = 0.12  # chassis height [m]
_HALF_WB = 0.16  # half wheelbase [m]
_HALF_TRACK = 0.12  # half track [m]
_SUSP_KE = 3000.0
_SUSP_KD = 120.0


def _build():
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    builder.add_ground_plane(cfg=terrain_cfg)

    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    chassis = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, _H), wp.quat_identity()))
    chassis_cfg = newton.ModelBuilder.ShapeConfig()
    chassis_cfg.has_shape_collision = False
    builder.add_shape_box(chassis, xform=wp.transform(), hx=0.16, hy=0.1, hz=0.04, cfg=chassis_cfg)
    joints = [builder.add_joint_free(child=chassis)]

    nv.set_vehicle(
        builder,
        0,
        drive_mode=int(nv.DriveMode.ACKERMANN),
        wheelbase=2.0 * _HALF_WB,
        track_width=2.0 * _HALF_TRACK,
        steer_limit=0.5,
    )

    def suspension(parent, x, y, child):
        # vertical prismatic spring/damper between parent and child at the wheel column
        return builder.add_joint_prismatic(
            parent=parent,
            child=child,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(x, y, _R - _H), wp.quat_identity()),
            child_xform=wp.transform(),
            target_ke=_SUSP_KE,
            target_kd=_SUSP_KD,
            target_pos=0.0,
            limit_lower=-0.05,
            limit_upper=0.05,
        )

    def wheel_cyl(body):
        return builder.add_shape_cylinder(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axis_q), radius=_R, half_height=0.02
        )

    wid = 0
    for x in (_HALF_WB, -_HALF_WB):
        front = x > 0
        for y in (_HALF_TRACK, -_HALF_TRACK):
            if front:
                # chassis -> suspension -> strut -> steering -> wheel
                strut = builder.add_link(xform=wp.transform(wp.vec3(x, y, _R), wp.quat_identity()))
                builder.add_shape_box(strut, hx=0.02, hy=0.02, hz=0.02, cfg=chassis_cfg)  # tiny mass/inertia
                joints.append(suspension(chassis, x, y, strut))
                wheel = builder.add_link(xform=wp.transform(wp.vec3(x, y, _R), wp.quat_identity()))
                steer = builder.add_joint_revolute(
                    parent=strut,
                    child=wheel,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(),
                    child_xform=wp.transform(),
                    target_ke=40.0,
                    target_kd=4.0,
                    limit_lower=-0.7,
                    limit_upper=0.7,
                )
                joints.append(steer)
                nv.add_wheel(
                    builder,
                    shape=wheel_cyl(wheel),
                    vehicle_id=0,
                    wheel_id=wid,
                    radius=_R,
                    width=0.04,
                    driven=True,
                    steerable=True,
                    side=(-1 if y > 0 else 1),
                    axle_row=0,
                    steer_joint=steer,
                )
            else:
                # chassis -> suspension -> wheel (rear, driven)
                wheel = builder.add_link(xform=wp.transform(wp.vec3(x, y, _R), wp.quat_identity()))
                joints.append(suspension(chassis, x, y, wheel))
                nv.add_wheel(
                    builder,
                    shape=wheel_cyl(wheel),
                    vehicle_id=0,
                    wheel_id=wid,
                    radius=_R,
                    width=0.04,
                    driven=True,
                    steerable=False,
                    side=(-1 if y > 0 else 1),
                    axle_row=1,
                )
            wid += 1

    builder.add_articulation(joints, label="sprung_car")
    return builder.finalize(), chassis


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.model, self._chassis = _build()
        # suspension makes the load determinate, so the load-smoothing band-aid is off
        self.vehicles = nv.WheeledVehicles(self.model, config=nv.WheeledConfig(max_wheel_speed=12.0, load_filter=1.0))
        self.vehicles.configure_solver_contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=256, nconmax=128)

        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial = self.state_0.body_q.numpy()[self._chassis].copy()
        self.viewer.set_model(self.model)

    def step(self):
        # settle on the springs, then drive forward and steer
        if self.sim_time < 0.5:
            self.vehicles.set_commands(drive=0.0, steer=0.0)
        else:
            self.vehicles.set_commands(drive=1.0, steer=0.5)
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
        # loads stay even with the band-aid off (suspension makes them determinate)
        fz = self.vehicles.patch.fz.numpy()
        if not np.isfinite(fz).all() or float(fz.min()) <= 0.0:
            raise ValueError(f"unexpected wheel loads {fz}")
        dx = float(q[0] - self._initial[0])
        yaw = _yaw(q) - _yaw(self._initial)
        if dx < 0.2:
            raise ValueError(f"sprung car did not drive forward (dx {dx:.3f} m)")
        if abs(yaw) < 0.1:
            raise ValueError(f"sprung car did not turn while steering (yaw {yaw:.3f} rad)")
        # chassis rides on the springs, not collapsed or launched
        if not (0.05 < float(q[2]) < 0.2):
            raise ValueError(f"chassis ride height out of range: {float(q[2]):.3f} m")

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
