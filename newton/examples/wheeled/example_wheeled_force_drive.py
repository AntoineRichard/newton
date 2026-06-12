# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Wheeled Force Drive
#
# Shows the Phase 00 wheeled assets driven by the Phase 2 force helper.
# MuJoCo handles the normal contact constraints with a tiny material-friction
# floor, while Newton contacts feed wheel contact patches and longitudinal drive
# forces. The demo also applies the matching axle reaction torque so the
# wheels do not free-spin from the injected patch force.
#
# Command: python -m newton.examples wheeled_force_drive
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
VEHICLE_FRICTION_MU = {
    "rc_car": 0.8,
    "husky": 0.8,
}
VEHICLE_DRIVE_FORCE_PER_SPEED = {
    "rc_car": 1.0,
    "husky": 16.0,
}
VEHICLE_NORMAL_LOADS = {
    "rc_car": 4.0 * 9.81 / 4.0,
    "husky": 80.0 * 9.81 / 4.0,
}


@wp.kernel
def _update_wheel_forward_axes_from_chassis(
    body_q: wp.array[wp.transform],
    wheel_body_indices: wp.array[wp.int32],
    chassis_body_indices: wp.array[wp.int32],
    forward_axis_body: wp.array[wp.vec3],
):
    wheel_id = wp.tid()
    wheel_body = wheel_body_indices[wheel_id]
    chassis_body = chassis_body_indices[wheel_id]

    wheel_q = wp.transform_get_rotation(body_q[wheel_body])
    chassis_q = wp.transform_get_rotation(body_q[chassis_body])
    chassis_forward_world = wp.quat_rotate(chassis_q, wp.vec3(1.0, 0.0, 0.0))
    forward_axis_body[wheel_id] = wp.quat_rotate_inv(wheel_q, chassis_forward_world)


@wp.kernel
def _apply_axle_patch_reaction_torques(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    wheel_body_indices: wp.array[wp.int32],
    axle_dof_indices: wp.array[wp.int32],
    axle_wheel_indices: wp.array[wp.int32],
    patch_center: wp.array[wp.vec3],
    axle_axis_body: wp.array[wp.vec3],
    longitudinal_direction: wp.array[wp.vec3],
    applied_force: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
):
    axle_id = wp.tid()
    dof_index = axle_dof_indices[axle_id]
    wheel_id = axle_wheel_indices[axle_id]
    body_index = wheel_body_indices[wheel_id]

    force = applied_force[wheel_id]
    if wp.abs(force) <= 1.0e-6:
        joint_f[dof_index] = 0.0
        return

    X_wb = body_q[body_index]
    axle_world = wp.normalize(wp.transform_vector(X_wb, axle_axis_body[wheel_id]))
    com_world = wp.transform_point(X_wb, body_com[body_index])
    patch_offset = patch_center[wheel_id] - com_world
    force_world = longitudinal_direction[wheel_id] * force
    patch_torque_world = wp.cross(patch_offset, force_world)
    joint_f[dof_index] = -wp.dot(patch_torque_world, axle_world)


def _vehicle_from_label(label: str) -> str:
    for vehicle_name in VEHICLE_NAMES:
        if f"/{vehicle_name}/" in label:
            return vehicle_name
    raise ValueError(f"could not infer wheeled vehicle from label: {label}")


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
        newton.wheeled.register_wheeled_custom_attributes(world)
        for vehicle_name in VEHICLE_NAMES:
            world.add_usd(
                str(assets[vehicle_name].file),
                xform=wp.transform(VEHICLE_OFFSETS[vehicle_name], wp.quat_identity()),
                enable_self_collisions=False,
                schema_resolvers=[newton.usd.SchemaResolverPhysx()],
            )
        newton.wheeled.apply_wheeled_manifest(world, MANIFEST_PATH, asset_names=VEHICLE_NAMES)

        scene = newton.ModelBuilder()
        scene.replicate(world, self.world_count)
        scene.add_ground_plane(label="ground")
        self.model = scene.finalize()
        self._minimize_solver_material_friction()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.wheeled_metadata = newton.wheeled.build_wheeled_metadata(self.model)
        self.patch_state = newton.wheeled.WheelContactPatchState(self.model, self.wheeled_metadata)
        self.drive_control = newton.wheeled.WheelDriveControl(self.model, self.wheeled_metadata)
        self.drive_state = newton.wheeled.WheelDriveState(self.model, self.wheeled_metadata)
        self._configure_drive_frames()
        self._configure_axle_patch_reaction_torques()
        self._configure_drive_control()

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

        self._initial_chassis_x = self.state_0.body_q.numpy()[self._chassis_body_indices, 0].copy()

        expected_wheels = self.world_count * sum(len(assets[name].wheel_body_labels) for name in VEHICLE_NAMES)
        if self.wheeled_metadata.wheel_count != expected_wheels:
            raise ValueError(f"expected {expected_wheels} wheels, found {self.wheeled_metadata.wheel_count}")

    def _minimize_solver_material_friction(self):
        min_friction = 1.0e-4
        self.model.shape_material_mu.fill_(min_friction)
        self.model.shape_material_mu_torsional.fill_(min_friction)
        self.model.shape_material_mu_rolling.fill_(min_friction)

    def _configure_drive_frames(self):
        body_world = self.model.body_world.numpy()
        chassis_by_vehicle_world = {}
        chassis_body_indices = []
        for body_index, label in enumerate(self.model.body_label):
            label_text = str(label)
            if not (label_text.endswith("chassis") or "_chassis" in label_text):
                continue
            vehicle_name = _vehicle_from_label(label_text)
            world_id = int(body_world[body_index])
            chassis_by_vehicle_world[(vehicle_name, world_id)] = body_index
            chassis_body_indices.append(body_index)

        wheel_chassis_body_indices = []
        for body_index in self.wheeled_metadata.wheel_body_indices:
            label_text = str(self.model.body_label[body_index])
            vehicle_name = _vehicle_from_label(label_text)
            world_id = int(body_world[body_index])
            try:
                chassis_body_index = chassis_by_vehicle_world[(vehicle_name, world_id)]
            except KeyError as exc:
                raise ValueError(f"could not find chassis for wheel body {label_text} in world {world_id}") from exc
            wheel_chassis_body_indices.append(chassis_body_index)

        self._chassis_body_indices = chassis_body_indices
        self._wheel_body_indices = wp.array(
            np.array(self.wheeled_metadata.wheel_body_indices, dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )
        self._wheel_chassis_body_indices = wp.array(
            np.array(wheel_chassis_body_indices, dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )

    def _configure_axle_patch_reaction_torques(self):
        wheel_by_key = {}
        for wheel_index, body_index in enumerate(self.wheeled_metadata.wheel_body_indices):
            body_leaf = str(self.model.body_label[body_index]).rsplit("/", 1)[-1]
            wheel_by_key[body_leaf.removesuffix("_wheel")] = wheel_index

        axle_dof_indices = []
        axle_wheel_indices = []
        joint_qd_start = self.model.joint_qd_start.numpy()
        for joint_index, label in enumerate(self.model.joint_label):
            joint_leaf = str(label).rsplit("/", 1)[-1]
            if not joint_leaf.endswith("_axle"):
                continue
            wheel_key = joint_leaf.removesuffix("_axle")
            try:
                wheel_index = wheel_by_key[wheel_key]
            except KeyError as exc:
                raise ValueError(f"could not map axle joint {label} to a wheel body") from exc
            axle_dof_indices.append(int(joint_qd_start[joint_index]))
            axle_wheel_indices.append(wheel_index)

        if len(axle_dof_indices) != self.wheeled_metadata.wheel_count:
            raise ValueError(f"expected {self.wheeled_metadata.wheel_count} axle joints, found {len(axle_dof_indices)}")

        self.axle_count = len(axle_dof_indices)
        self.axle_dof_indices = wp.array(axle_dof_indices, dtype=wp.int32, device=self.model.device)
        self.axle_wheel_indices = wp.array(axle_wheel_indices, dtype=wp.int32, device=self.model.device)

    def _configure_drive_control(self):
        wheel_count = self.wheeled_metadata.wheel_count
        friction_mu = np.zeros(wheel_count, dtype=np.float32)
        fallback_normal_load = np.zeros(wheel_count, dtype=np.float32)
        drive_torque_per_speed = np.zeros(wheel_count, dtype=np.float32)

        for wheel_index, body_index in enumerate(self.wheeled_metadata.wheel_body_indices):
            vehicle_name = _vehicle_from_label(str(self.model.body_label[body_index]))
            friction_mu[wheel_index] = VEHICLE_FRICTION_MU[vehicle_name]
            fallback_normal_load[wheel_index] = VEHICLE_NORMAL_LOADS[vehicle_name]
            drive_torque_per_speed[wheel_index] = (
                VEHICLE_DRIVE_FORCE_PER_SPEED[vehicle_name] * self.wheeled_metadata.wheel_radius[wheel_index]
            )

        self.drive_control.friction_mu.assign(friction_mu)
        self.drive_control.fallback_normal_load.assign(fallback_normal_load)
        self.drive_torque_per_speed = drive_torque_per_speed
        self.drive_torque_command = np.zeros(wheel_count, dtype=np.float32)

    def _update_drive_forward_axes(self):
        wp.launch(
            _update_wheel_forward_axes_from_chassis,
            dim=self.wheeled_metadata.wheel_count,
            inputs=[
                self.state_0.body_q,
                self._wheel_body_indices,
                self._wheel_chassis_body_indices,
            ],
            outputs=[self.drive_control.forward_axis_body],
            device=self.model.device,
        )

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
        self.drive_torque_command[:] = command_speed * self.drive_torque_per_speed
        self.drive_control.drive_torque.assign(self.drive_torque_command)

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
            self._update_drive_forward_axes()
            newton.wheeled.apply_wheel_drive_forces(
                self.model,
                self.state_0,
                self.wheeled_metadata,
                self.patch_state,
                self.drive_control,
                self.drive_state,
            )
            wp.launch(
                _apply_axle_patch_reaction_torques,
                dim=self.axle_count,
                inputs=[
                    self.state_0.body_q,
                    self.model.body_com,
                    self._wheel_body_indices,
                    self.axle_dof_indices,
                    self.axle_wheel_indices,
                    self.patch_state.center,
                    self.drive_control.axle_axis_body,
                    self.drive_state.longitudinal_direction,
                    self.drive_state.applied_force,
                ],
                outputs=[self.control.joint_f],
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
            "force-driven wheeled bodies have bounded velocity",
            lambda q, qd: max(abs(qd)) < 100.0,
        )

        if self.sim_time <= 2.1:
            chassis_x = self.state_0.body_q.numpy()[self._chassis_body_indices, 0]
            mean_forward_motion = float(np.mean(chassis_x - self._initial_chassis_x))
            if mean_forward_motion < 0.02:
                raise ValueError("force-driven wheeled vehicles did not move forward")

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
