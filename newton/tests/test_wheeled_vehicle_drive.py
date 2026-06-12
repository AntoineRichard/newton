# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _build_two_wheel_model(*, device="cpu"):
    builder = newton.ModelBuilder(gravity=0.0)
    newton.wheeled.register_wheeled_custom_attributes(builder)

    wheel_bodies = []
    wheel_shapes = []
    for wheel_id, y in enumerate((0.0, 1.0)):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(wheel_id), y, 0.5), wp.quat_identity()),
            label=f"wheel_body_{wheel_id}",
            custom_attributes={"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": wheel_id},
        )
        shape = builder.add_shape_sphere(
            body,
            radius=0.5,
            label=f"wheel_shape_{wheel_id}",
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": wheel_id,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.5,
                "wheeled:wheel_width": 0.2,
            },
        )
        wheel_bodies.append(body)
        wheel_shapes.append(shape)

    model = builder.finalize(device=device)
    metadata = newton.wheeled.build_wheeled_metadata(model)
    return model, metadata, tuple(wheel_bodies), tuple(wheel_shapes)


def _build_shared_body_two_wheel_model(*, device="cpu"):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="chassis")
    wheel_shapes = (
        builder.add_shape_sphere(body, xform=wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity()), radius=0.1),
        builder.add_shape_sphere(body, xform=wp.transform(wp.vec3(0.0, 0.5, 0.0), wp.quat_identity()), radius=0.1),
    )
    model = builder.finalize(device=device)
    metadata = newton.wheeled.WheeledModelMetadata(
        wheel_count=2,
        vehicle_count=1,
        wheel_shape_indices=wheel_shapes,
        wheel_body_indices=(body, body),
        wheel_vehicle_ids=(0, 0),
        wheel_radius=(0.5, 0.5),
        wheel_width=(0.2, 0.2),
        vehicle_wheel_counts=(2,),
    )
    return model, metadata, body


def _make_patch_state(model, metadata, *, active=(True, True)):
    patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
    wheel_count = metadata.wheel_count
    active_values = np.array(active[:wheel_count], dtype=bool)
    centers = np.zeros((wheel_count, 3), dtype=np.float32)
    centers[:, 2] = 0.0
    if wheel_count > 1:
        centers[1, 0] = 1.0
    patch_state.active.assign(active_values)
    patch_state.contact_count.assign(active_values.astype(np.int32))
    patch_state.center.assign(centers)
    patch_state.normal.assign(np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (wheel_count, 1)))
    patch_state.friction_mu_seed.assign(np.full(wheel_count, 0.75, dtype=np.float32))
    return patch_state


class TestWheelDriveControlAndState(unittest.TestCase):
    def test_public_import_and_defaults(self):
        from newton.wheeled import (  # noqa: PLC0415
            WheelDriveControl,
            WheelDriveState,
            apply_wheel_drive_forces,
            update_wheel_drive_normal_loads,
        )

        self.assertTrue(callable(apply_wheel_drive_forces))
        self.assertTrue(callable(update_wheel_drive_normal_loads))

        model, metadata, *_ = _build_two_wheel_model()
        control = WheelDriveControl(model, metadata)
        drive_state = WheelDriveState(model, metadata)

        self.assertEqual(control.wheel_count, metadata.wheel_count)
        self.assertEqual(drive_state.wheel_count, metadata.wheel_count)
        self.assertEqual(control.drive_torque.shape, (metadata.wheel_count,))
        self.assertEqual(control.brake_torque.shape, (metadata.wheel_count,))
        self.assertEqual(control.forward_axis_body.shape, (metadata.wheel_count,))
        self.assertEqual(drive_state.longitudinal_direction.shape, (metadata.wheel_count,))

        np.testing.assert_array_equal(control.enabled.numpy(), np.array([True, True]))
        np.testing.assert_allclose(control.drive_torque.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(control.brake_torque.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(control.target_speed.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(control.target_speed_gain.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(control.friction_mu.numpy(), np.full(2, -1.0, dtype=np.float32))
        np.testing.assert_allclose(control.fallback_normal_load.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(
            control.forward_axis_body.numpy(),
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            control.axle_axis_body.numpy(),
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(drive_state.normal_load.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(drive_state.requested_force.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(drive_state.applied_force.numpy(), np.zeros(2, dtype=np.float32))

    def test_clear_preserves_previous_normal_load_by_default(self):
        model, metadata, *_ = _build_two_wheel_model()
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        drive_state.normal_load.assign(np.array([1.0, 2.0], dtype=np.float32))
        drive_state.previous_normal_load.assign(np.array([3.0, 4.0], dtype=np.float32))
        drive_state.applied_force.assign(np.array([5.0, 6.0], dtype=np.float32))

        drive_state.clear()

        np.testing.assert_allclose(drive_state.normal_load.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(drive_state.applied_force.numpy(), np.zeros(2, dtype=np.float32))

        drive_state.clear(clear_previous_normal_load=True)

        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.zeros(2, dtype=np.float32))


class TestWheelDriveForces(unittest.TestCase):
    def test_direction_projection_and_speed_diagnostics(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        control.fallback_normal_load.assign(np.array([10.0, 0.0], dtype=np.float32))
        state.body_qd.assign(np.array([[3.0, 0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        np.testing.assert_allclose(
            drive_state.longitudinal_direction.numpy()[0],
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        self.assertAlmostEqual(float(drive_state.wheel_angular_speed.numpy()[0]), 2.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.longitudinal_speed.numpy()[0]), 2.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.slip_speed.numpy()[0]), 1.0, delta=1e-6)
        np.testing.assert_allclose(state.body_f.numpy()[bodies[0]], np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_degenerate_projected_direction_applies_no_force(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        patch_state.normal.assign(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        control.drive_torque.assign(np.array([10.0, 0.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([10.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        np.testing.assert_allclose(state.body_f.numpy()[bodies[0]], np.zeros(6, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), 0.0, delta=1e-6)

    def test_drive_force_clips_and_accumulates_wrench(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        control.drive_torque.assign(np.array([10.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([0.5, -1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([10.0, 0.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        self.assertAlmostEqual(float(drive_state.normal_load.numpy()[0]), 10.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.requested_force.numpy()[0]), 20.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.friction_limit.numpy()[0]), 5.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), 5.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[bodies[0]],
            np.array([5.0, 0.0, 0.0, 0.0, -2.5, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_inactive_disabled_and_zero_load_wheels_apply_no_force(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()

        cases = (
            {"active": (False, False), "enabled": True, "load": 10.0},
            {"active": (True, False), "enabled": False, "load": 10.0},
            {"active": (True, False), "enabled": True, "load": 0.0},
        )
        for case in cases:
            with self.subTest(case=case):
                state = model.state()
                patch_state = _make_patch_state(model, metadata, active=case["active"])
                control = newton.wheeled.WheelDriveControl(model, metadata)
                drive_state = newton.wheeled.WheelDriveState(model, metadata)
                control.enabled.assign(np.array([case["enabled"], True], dtype=bool))
                control.drive_torque.assign(np.array([10.0, 0.0], dtype=np.float32))
                control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))
                control.fallback_normal_load.assign(np.array([case["load"], 0.0], dtype=np.float32))

                newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

                np.testing.assert_allclose(state.body_f.numpy()[bodies[0]], np.zeros(6, dtype=np.float32), atol=1e-6)
                self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), 0.0, delta=1e-6)

    def test_brake_opposes_motion_and_does_not_start_stopped_wheel(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        control.brake_torque.assign(np.array([4.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([20.0, 0.0], dtype=np.float32))

        moving_state = model.state()
        moving_state.body_qd.assign(
            np.array([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        newton.wheeled.apply_wheel_drive_forces(model, moving_state, metadata, patch_state, control, drive_state)

        self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), -8.0, delta=1e-6)
        np.testing.assert_allclose(
            moving_state.body_f.numpy()[bodies[0]],
            np.array([-8.0, 0.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

        stopped_state = model.state()
        drive_state.clear()
        newton.wheeled.apply_wheel_drive_forces(model, stopped_state, metadata, patch_state, control, drive_state)

        self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), 0.0, delta=1e-6)
        np.testing.assert_allclose(stopped_state.body_f.numpy()[bodies[0]], np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_normal_load_latching_and_fallback(self):
        model, metadata, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, True))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        patch_state.normal_force.assign(np.array([12.0, 0.0], dtype=np.float32))
        newton.wheeled.update_wheel_drive_normal_loads(patch_state, drive_state)

        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.array([12.0, 0.0], dtype=np.float32))

        control.drive_torque.assign(np.array([100.0, 100.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, 1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([3.0, 3.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        np.testing.assert_allclose(drive_state.normal_load.numpy(), np.array([12.0, 3.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(
            drive_state.applied_force.numpy(), np.array([12.0, 3.0], dtype=np.float32), atol=1e-6
        )

        patch_state.active.assign(np.array([False, True], dtype=bool))
        patch_state.normal_force.assign(np.array([0.0, 4.0], dtype=np.float32))
        newton.wheeled.update_wheel_drive_normal_loads(patch_state, drive_state)
        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.array([12.0, 4.0], dtype=np.float32))

        newton.wheeled.update_wheel_drive_normal_loads(patch_state, drive_state, clear_inactive=True)
        np.testing.assert_allclose(drive_state.previous_normal_load.numpy(), np.array([0.0, 4.0], dtype=np.float32))

    def test_target_speed_gain_adds_clipped_force(self):
        model, metadata, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        control.target_speed.assign(np.array([3.0, 0.0], dtype=np.float32))
        control.target_speed_gain.assign(np.array([4.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([20.0, 0.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        self.assertAlmostEqual(float(drive_state.requested_force.numpy()[0]), 12.0, delta=1e-6)
        self.assertAlmostEqual(float(drive_state.applied_force.numpy()[0]), 12.0, delta=1e-6)


class TestWheelDriveSolverFlow(unittest.TestCase):
    def test_drive_force_accelerates_body_in_solver_step(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        state_0 = model.state()
        state_1 = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

        control.drive_torque.assign(np.array([5.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0, 0.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state_0, metadata, patch_state, control, drive_state)
        solver.step(state_0, state_1, None, None, 0.1)

        self.assertGreater(float(state_1.body_qd.numpy()[bodies[0], 0]), 0.0)

    def test_brake_force_reduces_forward_velocity_in_solver_step(self):
        model, metadata, bodies, *_ = _build_two_wheel_model()
        state_0 = model.state()
        state_1 = model.state()
        patch_state = _make_patch_state(model, metadata, active=(True, False))
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

        state_0.body_qd.assign(
            np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        control.brake_torque.assign(np.array([5.0, 0.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, -1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0, 0.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state_0, metadata, patch_state, control, drive_state)
        solver.step(state_0, state_1, None, None, 0.1)

        self.assertLess(float(state_1.body_qd.numpy()[bodies[0], 0]), 2.0)

    def test_opposite_wheel_forces_create_yaw_torque(self):
        model, metadata, body = _build_shared_body_two_wheel_model()
        state_0 = model.state()
        state_1 = model.state()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

        patch_state.active.assign(np.array([True, True], dtype=bool))
        patch_state.contact_count.assign(np.array([1, 1], dtype=np.int32))
        patch_state.center.assign(np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32))
        patch_state.normal.assign(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        patch_state.friction_mu_seed.assign(np.array([1.0, 1.0], dtype=np.float32))
        control.drive_torque.assign(np.array([5.0, -5.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0, 1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0, 50.0], dtype=np.float32))

        newton.wheeled.apply_wheel_drive_forces(model, state_0, metadata, patch_state, control, drive_state)
        solver.step(state_0, state_1, None, None, 0.1)

        self.assertGreater(float(state_1.body_qd.numpy()[body, 5]), 0.0)

    def test_newton_contact_flow_applies_drive_force_on_flat_patch(self):
        builder = newton.ModelBuilder(gravity=0.0)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        builder.add_ground_plane(label="terrain")
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()), label="wheel_body")
        builder.add_shape_sphere(
            body,
            radius=0.5,
            label="wheel_shape",
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": 0,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.5,
                "wheeled:wheel_width": 0.2,
            },
        )
        builder.custom_attributes["wheeled:is_wheel_body"].values[body] = True
        builder.custom_attributes["wheeled:wheel_body_id"].values[body] = 0

        model = builder.finalize(device="cpu")
        metadata = newton.wheeled.build_wheeled_metadata(model)
        state = model.state()
        contacts = model.contacts()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)

        model.collide(state, contacts)
        newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)
        control.drive_torque.assign(np.array([5.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0], dtype=np.float32))
        newton.wheeled.apply_wheel_drive_forces(model, state, metadata, patch_state, control, drive_state)

        self.assertTrue(bool(patch_state.active.numpy()[0]))
        self.assertGreater(float(state.body_f.numpy()[body, 0]), 0.0)

    def test_mujoco_newton_contact_flow_accepts_drive_forces(self):
        builder = newton.ModelBuilder(gravity=0.0)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        builder.add_ground_plane(label="terrain")
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()), label="wheel_body")
        builder.add_shape_sphere(
            body,
            radius=0.5,
            label="wheel_shape",
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": 0,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.5,
                "wheeled:wheel_width": 0.2,
            },
        )
        builder.custom_attributes["wheeled:is_wheel_body"].values[body] = True
        builder.custom_attributes["wheeled:wheel_body_id"].values[body] = 0

        model = builder.finalize(device="cpu")
        metadata = newton.wheeled.build_wheeled_metadata(model)
        try:
            solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=64, nconmax=32)
        except ImportError as exc:
            self.skipTest(f"MuJoCo or dependencies are not installed: {exc}")

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.contacts()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        control = newton.wheeled.WheelDriveControl(model, metadata)
        drive_state = newton.wheeled.WheelDriveState(model, metadata)
        sim_control = model.control()

        state_0.clear_forces()
        model.collide(state_0, contacts)
        newton.wheeled.update_wheel_contact_patches(model, state_0, contacts, metadata, patch_state)
        control.drive_torque.assign(np.array([5.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0], dtype=np.float32))
        newton.wheeled.apply_wheel_drive_forces(model, state_0, metadata, patch_state, control, drive_state)
        solver.step(state_0, state_1, sim_control, contacts, 1.0 / 240.0)

        self.assertGreater(float(drive_state.applied_force.numpy()[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
