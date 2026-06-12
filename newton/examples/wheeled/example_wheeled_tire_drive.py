# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Tire Drive
#
# Shows the Phase 00 wheeled assets driven by the Phase 3 tire helper.
# MuJoCo solves normal contact constraints from Newton contacts. Wheel contact
# pairs are configured as normal-only contacts, and longitudinal/lateral tire
# forces are applied explicitly from analytical wheel speeds.
#
# Command: python -m newton.examples wheeled_tire_drive
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd

ASSET_DIR = Path(newton.examples.get_asset("wheeled"))
MANIFEST_PATH = ASSET_DIR / "manifest.json"
VEHICLE_NAMES = ("rc_car", "husky")
VEHICLE_OFFSETS = {
    "rc_car": wp.vec3(0.0, -0.85, 0.0),
    "husky": wp.vec3(0.0, 1.0, 0.0),
}
TIRE_FRICTION_MU = 0.8
TIRE_FALLBACK_NORMAL_LOAD = 10.0
TIRE_LONGITUDINAL_STIFFNESS = 40.0
TIRE_LATERAL_STIFFNESS = 30.0


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
        self.manual_speed = 0.0

        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(MANIFEST_PATH)}

        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        newton.wheeled.register_wheeled_custom_attributes(world)
        for vehicle_name in VEHICLE_NAMES:
            world.add_usd(
                str(assets[vehicle_name].file),
                xform=wp.transform(VEHICLE_OFFSETS[vehicle_name], wp.quat_identity()),
                enable_self_collisions=False,
                schema_resolvers=[newton.usd.SchemaResolverPhysx()],
            )
        newton.wheeled.apply_wheeled_manifest(world, MANIFEST_PATH, asset_names=VEHICLE_NAMES)
        newton.wheeled.configure_wheel_axle_joints(
            world,
            axle_joint_labels=[
                label for vehicle_name in VEHICLE_NAMES for label in assets[vehicle_name].axle_joint_labels
            ],
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

        self.wheeled_metadata = newton.wheeled.build_wheeled_metadata(self.model)
        self.patch_state = newton.wheeled.WheelContactPatchState(self.model, self.wheeled_metadata)
        self.tire_control = newton.wheeled.WheelTireControl(self.model, self.wheeled_metadata)
        self.tire_state = newton.wheeled.WheelTireState(self.model, self.wheeled_metadata)
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

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((2.6, 2.4, 0.0))
        self.viewer.set_camera(pos=wp.vec3(7.0, -9.0, 5.2), pitch=-35.0, yaw=-135.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

        self._chassis_body_indices = [
            body_index
            for body_index, label in enumerate(self.model.body_label)
            if str(label).endswith("chassis") or "_chassis" in str(label)
        ]
        self._initial_chassis_x = self.state_0.body_q.numpy()[self._chassis_body_indices, 0].copy()
        self.wheel_angular_speed_per_speed = 1.0 / np.asarray(self.wheeled_metadata.wheel_radius, dtype=np.float32)
        self.wheel_angular_speed_command = np.zeros(self.wheeled_metadata.wheel_count, dtype=np.float32)

        expected_wheels = self.world_count * sum(len(assets[name].wheel_body_labels) for name in VEHICLE_NAMES)
        if self.wheeled_metadata.wheel_count != expected_wheels:
            raise ValueError(f"expected {expected_wheels} wheels, found {self.wheeled_metadata.wheel_count}")

    def gui(self, ui):
        _changed, self.cycle_enabled = ui.checkbox("Cycle forward/back", self.cycle_enabled)
        _changed, self.drive_scale = ui.slider_float("Drive scale", self.drive_scale, 0.0, 2.0)
        if not self.cycle_enabled:
            _changed, self.manual_speed = ui.slider_float("Manual speed", self.manual_speed, -1.0, 1.0)

    def step(self):
        if self.cycle_enabled:
            cycle_time = self.sim_time % 6.0
            if cycle_time < 2.0:
                command_speed = 0.6 * self.drive_scale
            elif cycle_time < 3.0:
                command_speed = 0.0
            elif cycle_time < 5.0:
                command_speed = -0.6 * self.drive_scale
            else:
                command_speed = 0.0
        else:
            command_speed = self.manual_speed * self.drive_scale
        self.wheel_angular_speed_command[:] = command_speed * self.wheel_angular_speed_per_speed
        self.tire_control.wheel_angular_speed.assign(self.wheel_angular_speed_command)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "tire-driven wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )

        if self.sim_time <= 2.1:
            chassis_x = self.state_0.body_q.numpy()[self._chassis_body_indices, 0]
            mean_forward_motion = float(np.mean(chassis_x - self._initial_chassis_x))
            if mean_forward_motion < 0.02:
                raise ValueError("tire-driven wheeled vehicles did not move forward")

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
