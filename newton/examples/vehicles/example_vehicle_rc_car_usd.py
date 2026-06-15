# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle RC Car (USD asset, sprung Ackermann)
#
# Loads the authored rc_car.usda fixture (a sprung, front-steer Ackermann RC car
# with real prismatic suspension, revolute steering, and physical axle joints)
# and drives it through the newton.vehicles layer. The asset's physical axle
# (wheel-spin) joints are converted to fixed via configure_wheel_axle_joints so
# wheel spin is analytical; suspension and steering remain solver joints. Wheels
# are annotated from the manifest labels.
#
# Command: python -m newton.examples vehicle_rc_car_usd
#
###########################################################################

import json
import math
from pathlib import Path

import numpy as np

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

        self.model, self._chassis = _build()
        # the asset is sprung, so the load is determinate and the band-aid is off
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
        if self.sim_time < 0.5:
            self.vehicles.set_commands(drive=0.0, steer=0.0)
        else:
            self.vehicles.set_commands(drive=1.0, steer=0.6)
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
