# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Drive
#
# Shows the Phase 00 wheeled assets driving forward and backward under
# MuJoCo contact dynamics. Wheel axle joints are commanded with explicit
# torque control.
#
# Command: python -m newton.examples wheeled_drive
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

ASSET_DIR = Path(newton.examples.get_asset("wheeled"))
MANIFEST_PATH = ASSET_DIR / "manifest.json"
VEHICLE_NAMES = ("rc_car", "husky")
VEHICLE_OFFSETS = {
    "rc_car": wp.vec3(0.0, -0.85, 0.0),
    "husky": wp.vec3(0.0, 1.0, 0.0),
}
VEHICLE_TORQUE_GAINS = {
    "rc_car": 0.03,
    "husky": 1.0,
}
VEHICLE_TORQUE_LIMITS = {
    "rc_car": 0.08,
    "husky": 4.0,
}


@wp.kernel
def _apply_axle_velocity_torques(
    joint_qd: wp.array[float],
    axle_dof_indices: wp.array[int],
    inv_wheel_radius: wp.array[float],
    torque_gain: wp.array[float],
    torque_limit: wp.array[float],
    command_speed: wp.array[float],
    joint_f: wp.array[float],
):
    axle_id = wp.tid()
    dof_index = axle_dof_indices[axle_id]
    target_angular_speed = command_speed[0] * inv_wheel_radius[axle_id]
    torque = (target_angular_speed - joint_qd[dof_index]) * torque_gain[axle_id]
    limit = torque_limit[axle_id]
    if torque > limit:
        torque = limit
    elif torque < -limit:
        torque = -limit
    joint_f[dof_index] = torque


def _vehicle_from_label(label: str) -> str:
    for vehicle_name in VEHICLE_NAMES:
        if f"/{vehicle_name}/" in label:
            return vehicle_name
    raise ValueError(f"could not infer wheeled vehicle from label: {label}")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.world_count = args.world_count
        self.drive_scale = 1.0
        self.cycle_enabled = True
        self.manual_speed = 0.0

        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(MANIFEST_PATH)}

        world = newton.ModelBuilder()
        for vehicle_name in VEHICLE_NAMES:
            world.add_usd(
                str(assets[vehicle_name].file),
                xform=wp.transform(VEHICLE_OFFSETS[vehicle_name], wp.quat_identity()),
                enable_self_collisions=False,
                schema_resolvers=[newton.usd.SchemaResolverPhysx()],
            )

        scene = newton.ModelBuilder()
        scene.replicate(world, self.world_count)
        scene.add_ground_plane(label="ground")
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=True,
            njmax=256 * self.world_count,
            nconmax=128 * self.world_count,
        )

        axle_dof_indices = []
        inv_wheel_radius = []
        torque_gain = []
        torque_limit = []
        joint_qd_start = self.model.joint_qd_start.numpy()
        for joint_index, label in enumerate(self.model.joint_label):
            label_text = str(label)
            if "axle" not in label_text:
                continue
            vehicle_name = _vehicle_from_label(label_text)
            axle_dof_indices.append(int(joint_qd_start[joint_index]))
            inv_wheel_radius.append(1.0 / assets[vehicle_name].wheel_radius)
            torque_gain.append(VEHICLE_TORQUE_GAINS[vehicle_name])
            torque_limit.append(VEHICLE_TORQUE_LIMITS[vehicle_name])

        expected_axles = self.world_count * sum(len(assets[name].wheel_body_labels) for name in VEHICLE_NAMES)
        if len(axle_dof_indices) != expected_axles:
            raise ValueError(f"expected {expected_axles} axle joints, found {len(axle_dof_indices)}")

        self.axle_count = len(axle_dof_indices)
        self.axle_dof_indices = wp.array(axle_dof_indices, dtype=wp.int32, device=self.model.device)
        self.inv_wheel_radius = wp.array(inv_wheel_radius, dtype=wp.float32, device=self.model.device)
        self.torque_gain = wp.array(torque_gain, dtype=wp.float32, device=self.model.device)
        self.torque_limit = wp.array(torque_limit, dtype=wp.float32, device=self.model.device)
        self.command_speed = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((2.6, 2.4, 0.0))
        self.viewer.set_camera(pos=wp.vec3(7.0, -9.0, 5.2), pitch=-35.0, yaw=-135.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

    def gui(self, ui):
        _changed, self.cycle_enabled = ui.checkbox("Cycle forward/back", self.cycle_enabled)
        _changed, self.drive_scale = ui.slider_float("Drive scale", self.drive_scale, 0.0, 2.0)
        if not self.cycle_enabled:
            _changed, self.manual_speed = ui.slider_float("Manual speed", self.manual_speed, -1.0, 1.0)

    def step(self):
        if self.cycle_enabled:
            cycle_time = self.sim_time % 6.0
            if cycle_time < 2.0:
                command_speed = 0.7 * self.drive_scale
            elif cycle_time < 3.0:
                command_speed = 0.0
            elif cycle_time < 5.0:
                command_speed = -0.7 * self.drive_scale
            else:
                command_speed = 0.0
        else:
            command_speed = self.manual_speed * self.drive_scale
        self.command_speed.assign(np.array([command_speed], dtype=np.float32))

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _apply_axle_velocity_torques,
                dim=self.axle_count,
                inputs=[
                    self.state_0.joint_qd,
                    self.axle_dof_indices,
                    self.inv_wheel_radius,
                    self.torque_gain,
                    self.torque_limit,
                    self.command_speed,
                    self.control.joint_f,
                ],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=360, world_count=32)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
