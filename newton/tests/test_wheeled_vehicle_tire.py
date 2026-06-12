# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _build_one_wheel_model(*, device="cpu", mujoco_attrs=False):
    builder = newton.ModelBuilder(gravity=0.0)
    if mujoco_attrs:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    newton.wheeled.register_wheeled_custom_attributes(builder)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        label="wheel_body",
        custom_attributes={"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": 0},
    )
    shape = builder.add_shape_sphere(
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
    model = builder.finalize(device=device)
    metadata = newton.wheeled.build_wheeled_metadata(model)
    return model, metadata, body, shape


def _build_two_wheel_model(*, device="cpu"):
    builder = newton.ModelBuilder(gravity=0.0)
    newton.wheeled.register_wheeled_custom_attributes(builder)
    wheel_bodies = []
    wheel_shapes = []
    for wheel_id in range(2):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(float(wheel_id), 0.0, 0.5), wp.quat_identity()),
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


def _fiala_lateral_force(tan_alpha, cornering_stiffness, friction_limit):
    abs_tan_alpha = abs(tan_alpha)
    if abs_tan_alpha <= 1.0e-6 or cornering_stiffness <= 0.0 or friction_limit <= 0.0:
        return 0.0

    transition_tan_alpha = 3.0 * friction_limit / cornering_stiffness
    if abs_tan_alpha >= transition_tan_alpha:
        return -friction_limit * np.sign(tan_alpha)

    return (
        -cornering_stiffness * tan_alpha
        + cornering_stiffness**2 * abs_tan_alpha * tan_alpha / (3.0 * friction_limit)
        - cornering_stiffness**3 * tan_alpha**3 / (27.0 * friction_limit**2)
    )


def _make_patch_state(model, metadata, *, active=True, center_z=0.0):
    patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
    wheel_count = int(metadata.wheel_count)
    active_values = np.full(wheel_count, bool(active), dtype=bool)
    centers = np.zeros((wheel_count, 3), dtype=np.float32)
    centers[:, 0] = np.arange(wheel_count, dtype=np.float32)
    centers[:, 2] = center_z
    patch_state.active.assign(active_values)
    patch_state.contact_count.assign(active_values.astype(np.int32))
    patch_state.center.assign(centers)
    patch_state.normal.assign(np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (wheel_count, 1)))
    patch_state.friction_mu_seed.assign(np.full(wheel_count, 0.75, dtype=np.float32))
    return patch_state


class TestWheelTireControlAndState(unittest.TestCase):
    def test_public_import_and_defaults(self):
        from newton.wheeled import (  # noqa: PLC0415
            WheelTireControl,
            WheelTireState,
            apply_wheel_tire_forces,
            configure_mujoco_wheel_contacts,
            configure_wheel_axle_joints,
            configure_wheel_tire_control,
            update_wheel_tire_normal_loads,
        )

        self.assertTrue(callable(apply_wheel_tire_forces))
        self.assertTrue(callable(update_wheel_tire_normal_loads))
        self.assertTrue(callable(configure_mujoco_wheel_contacts))
        self.assertTrue(callable(configure_wheel_axle_joints))
        self.assertTrue(callable(configure_wheel_tire_control))

        model, metadata, *_ = _build_one_wheel_model()
        control = WheelTireControl(model, metadata)
        tire_state = WheelTireState(model, metadata)

        self.assertEqual(control.wheel_count, metadata.wheel_count)
        self.assertEqual(tire_state.wheel_count, metadata.wheel_count)
        self.assertEqual(control.tire_model.shape, (metadata.wheel_count,))
        self.assertEqual(control.wheel_angular_speed.shape, (metadata.wheel_count,))
        self.assertEqual(control.longitudinal_stiffness.shape, (metadata.wheel_count,))
        self.assertEqual(control.lateral_stiffness.shape, (metadata.wheel_count,))
        self.assertEqual(tire_state.applied_longitudinal_force.shape, (metadata.wheel_count,))
        self.assertEqual(tire_state.applied_lateral_force.shape, (metadata.wheel_count,))

        np.testing.assert_array_equal(control.enabled.numpy(), np.array([True]))
        np.testing.assert_array_equal(
            control.tire_model.numpy(), np.array([newton.wheeled.WheelTireControl.TireModel.SATURATED_LINEAR])
        )
        np.testing.assert_allclose(control.wheel_angular_speed.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.friction_mu.numpy(), np.array([-1.0], dtype=np.float32))
        np.testing.assert_allclose(control.fallback_normal_load.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.longitudinal_stiffness.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.lateral_stiffness.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.min_reference_speed.numpy(), np.array([0.1], dtype=np.float32))
        np.testing.assert_allclose(control.forward_axis_body.numpy(), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(control.axle_axis_body.numpy(), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        np.testing.assert_allclose(tire_state.normal_load.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(tire_state.requested_longitudinal_force.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(tire_state.applied_lateral_force.numpy(), np.zeros(1, dtype=np.float32))

    def test_clear_preserves_previous_normal_load_by_default(self):
        model, metadata, *_ = _build_one_wheel_model()
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        tire_state.normal_load.assign(np.array([1.0], dtype=np.float32))
        tire_state.previous_normal_load.assign(np.array([2.0], dtype=np.float32))
        tire_state.applied_longitudinal_force.assign(np.array([3.0], dtype=np.float32))

        tire_state.clear()

        np.testing.assert_allclose(tire_state.normal_load.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.array([2.0], dtype=np.float32))
        np.testing.assert_allclose(tire_state.applied_longitudinal_force.numpy(), np.zeros(1, dtype=np.float32))

        tire_state.clear(clear_previous_normal_load=True)

        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.zeros(1, dtype=np.float32))


class TestWheelTireConfigurationHelpers(unittest.TestCase):
    def test_configure_wheel_tire_control_broadcasts_and_assigns_arrays(self):
        model, metadata, *_ = _build_one_wheel_model()
        control = newton.wheeled.WheelTireControl(model, metadata)

        newton.wheeled.configure_wheel_tire_control(
            control,
            enabled=False,
            tire_model="fiala",
            friction_mu=0.9,
            fallback_normal_load=[12.0],
            longitudinal_stiffness=40.0,
            lateral_stiffness=[30.0],
            min_reference_speed=0.2,
            forward_axis_body=[0.0, 1.0, 0.0],
            axle_axis_body=[[1.0, 0.0, 0.0]],
        )

        np.testing.assert_array_equal(control.enabled.numpy(), np.array([False]))
        np.testing.assert_array_equal(
            control.tire_model.numpy(), np.array([newton.wheeled.WheelTireControl.TireModel.FIALA])
        )
        np.testing.assert_allclose(control.friction_mu.numpy(), np.array([0.9], dtype=np.float32))
        np.testing.assert_allclose(control.fallback_normal_load.numpy(), np.array([12.0], dtype=np.float32))
        np.testing.assert_allclose(control.longitudinal_stiffness.numpy(), np.array([40.0], dtype=np.float32))
        np.testing.assert_allclose(control.lateral_stiffness.numpy(), np.array([30.0], dtype=np.float32))
        np.testing.assert_allclose(control.min_reference_speed.numpy(), np.array([0.2], dtype=np.float32))
        np.testing.assert_allclose(control.forward_axis_body.numpy(), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(control.axle_axis_body.numpy(), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def test_configure_wheel_tire_control_rejects_wrong_shapes(self):
        model, metadata, *_ = _build_one_wheel_model()
        control = newton.wheeled.WheelTireControl(model, metadata)

        with self.assertRaisesRegex(ValueError, "friction_mu"):
            newton.wheeled.configure_wheel_tire_control(control, friction_mu=[1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "tire_model"):
            newton.wheeled.configure_wheel_tire_control(control, tire_model="pacejka")
        with self.assertRaisesRegex(ValueError, "tire_model"):
            newton.wheeled.configure_wheel_tire_control(control, tire_model=["linear", "fiala"])
        with self.assertRaisesRegex(ValueError, "forward_axis_body"):
            newton.wheeled.configure_wheel_tire_control(control, forward_axis_body=[[1.0, 0.0]])


class TestWheelMomentControlAndState(unittest.TestCase):
    def test_public_import_and_defaults(self):
        from newton.wheeled import (  # noqa: PLC0415
            WheelMomentControl,
            WheelMomentState,
            configure_wheel_moment_control,
            update_wheel_moments,
        )

        self.assertTrue(callable(configure_wheel_moment_control))
        self.assertTrue(callable(update_wheel_moments))

        model, metadata, *_ = _build_one_wheel_model()
        control = WheelMomentControl(model, metadata)
        moment_state = WheelMomentState(model, metadata)

        self.assertEqual(control.wheel_count, metadata.wheel_count)
        self.assertEqual(moment_state.wheel_count, metadata.wheel_count)
        self.assertEqual(control.drive_torque.shape, (metadata.wheel_count,))
        self.assertEqual(control.brake_torque.shape, (metadata.wheel_count,))
        self.assertEqual(control.wheel_inertia.shape, (metadata.wheel_count,))
        self.assertEqual(control.axle_axis_body.shape, (metadata.wheel_count,))
        self.assertEqual(moment_state.wheel_angular_speed.shape, (metadata.wheel_count,))
        self.assertEqual(moment_state.tire_reaction_torque.shape, (metadata.wheel_count,))

        np.testing.assert_array_equal(control.enabled.numpy(), np.array([True]))
        np.testing.assert_array_equal(control.apply_body_reaction_torque.numpy(), np.array([True]))
        np.testing.assert_allclose(control.drive_torque.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.brake_torque.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.wheel_inertia.numpy(), np.ones(1, dtype=np.float32))
        np.testing.assert_allclose(control.angular_damping.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.rolling_resistance_torque.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(control.axle_axis_body.numpy(), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(moment_state.wheel_angular_speed.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(moment_state.net_torque.numpy(), np.zeros(1, dtype=np.float32))

    def test_clear_preserves_wheel_speed_by_default(self):
        model, metadata, *_ = _build_one_wheel_model()
        moment_state = newton.wheeled.WheelMomentState(model, metadata)
        moment_state.wheel_angular_speed.assign(np.array([3.0], dtype=np.float32))
        moment_state.net_torque.assign(np.array([2.0], dtype=np.float32))

        moment_state.clear()

        np.testing.assert_allclose(moment_state.wheel_angular_speed.numpy(), np.array([3.0], dtype=np.float32))
        np.testing.assert_allclose(moment_state.net_torque.numpy(), np.zeros(1, dtype=np.float32))

        moment_state.clear(clear_wheel_angular_speed=True)

        np.testing.assert_allclose(moment_state.wheel_angular_speed.numpy(), np.zeros(1, dtype=np.float32))

    def test_configure_wheel_moment_control_broadcasts_and_assigns_arrays(self):
        model, metadata, *_ = _build_one_wheel_model()
        control = newton.wheeled.WheelMomentControl(model, metadata)

        newton.wheeled.configure_wheel_moment_control(
            control,
            enabled=False,
            drive_torque=2.0,
            brake_torque=[1.0],
            wheel_inertia=0.25,
            angular_damping=[0.1],
            rolling_resistance_torque=0.2,
            apply_body_reaction_torque=False,
            axle_axis_body=[1.0, 0.0, 0.0],
        )

        np.testing.assert_array_equal(control.enabled.numpy(), np.array([False]))
        np.testing.assert_array_equal(control.apply_body_reaction_torque.numpy(), np.array([False]))
        np.testing.assert_allclose(control.drive_torque.numpy(), np.array([2.0], dtype=np.float32))
        np.testing.assert_allclose(control.brake_torque.numpy(), np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(control.wheel_inertia.numpy(), np.array([0.25], dtype=np.float32))
        np.testing.assert_allclose(control.angular_damping.numpy(), np.array([0.1], dtype=np.float32))
        np.testing.assert_allclose(control.rolling_resistance_torque.numpy(), np.array([0.2], dtype=np.float32))
        np.testing.assert_allclose(control.axle_axis_body.numpy(), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def test_configure_wheel_moment_control_rejects_invalid_values(self):
        model, metadata, *_ = _build_one_wheel_model()
        control = newton.wheeled.WheelMomentControl(model, metadata)

        with self.assertRaisesRegex(ValueError, "wheel_inertia"):
            newton.wheeled.configure_wheel_moment_control(control, wheel_inertia=0.0)
        with self.assertRaisesRegex(ValueError, "brake_torque"):
            newton.wheeled.configure_wheel_moment_control(control, brake_torque=-1.0)
        with self.assertRaisesRegex(ValueError, "axle_axis_body"):
            newton.wheeled.configure_wheel_moment_control(control, axle_axis_body=[[1.0, 0.0]])


class TestWheelMomentDynamics(unittest.TestCase):
    def test_drive_torque_integrates_speed_and_body_reaction(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=False)
        tire_control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        moment_control = newton.wheeled.WheelMomentControl(model, metadata)
        moment_state = newton.wheeled.WheelMomentState(model, metadata)

        newton.wheeled.configure_wheel_moment_control(moment_control, drive_torque=2.0, wheel_inertia=0.5)

        newton.wheeled.update_wheel_moments(
            model,
            state,
            metadata,
            patch_state,
            tire_state,
            moment_control,
            moment_state,
            0.25,
            tire_control=tire_control,
        )

        self.assertAlmostEqual(float(moment_state.wheel_angular_speed.numpy()[0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_control.wheel_angular_speed.numpy()[0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.wheel_angular_acceleration.numpy()[0]), 4.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.net_torque.numpy()[0]), 2.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.body_reaction_torque.numpy()[0]), -2.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([0.0, 0.0, 0.0, 0.0, -2.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_brake_can_hold_stopped_wheel_against_drive_torque(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=False)
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        moment_control = newton.wheeled.WheelMomentControl(model, metadata)
        moment_state = newton.wheeled.WheelMomentState(model, metadata)

        newton.wheeled.configure_wheel_moment_control(
            moment_control,
            drive_torque=1.0,
            brake_torque=2.0,
            wheel_inertia=1.0,
        )

        newton.wheeled.update_wheel_moments(
            model,
            state,
            metadata,
            patch_state,
            tire_state,
            moment_control,
            moment_state,
            1.0,
        )

        self.assertAlmostEqual(float(moment_state.wheel_angular_speed.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.net_torque.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.brake_torque.numpy()[0]), -1.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.body_reaction_torque.numpy()[0]), 0.0, delta=1e-6)
        np.testing.assert_allclose(state.body_f.numpy()[body], np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_tire_reaction_torque_cancels_locked_body_axle_moment(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata)
        tire_control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        moment_control = newton.wheeled.WheelMomentControl(model, metadata)
        moment_state = newton.wheeled.WheelMomentState(model, metadata)

        tire_control.wheel_angular_speed.assign(np.array([4.0], dtype=np.float32))
        tire_control.longitudinal_stiffness.assign(np.array([2.0], dtype=np.float32))
        tire_control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        tire_control.fallback_normal_load.assign(np.array([10.0], dtype=np.float32))
        moment_state.wheel_angular_speed.assign(np.array([4.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, tire_control, tire_state)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([4.0, 0.0, 0.0, 0.0, -2.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

        newton.wheeled.update_wheel_moments(
            model,
            state,
            metadata,
            patch_state,
            tire_state,
            moment_control,
            moment_state,
            0.1,
            tire_control=tire_control,
        )

        self.assertAlmostEqual(float(moment_state.tire_reaction_torque.numpy()[0]), -2.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.net_torque.numpy()[0]), -2.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.wheel_angular_speed.numpy()[0]), 3.8, delta=1e-6)
        self.assertAlmostEqual(float(tire_control.wheel_angular_speed.numpy()[0]), 3.8, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.body_reaction_torque.numpy()[0]), 2.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_inactive_patch_ignores_stale_tire_reaction_force(self):
        model, metadata, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=False)
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        moment_control = newton.wheeled.WheelMomentControl(model, metadata)
        moment_state = newton.wheeled.WheelMomentState(model, metadata)

        tire_state.applied_longitudinal_force.assign(np.array([10.0], dtype=np.float32))
        moment_state.wheel_angular_speed.assign(np.array([1.0], dtype=np.float32))

        newton.wheeled.update_wheel_moments(
            model,
            state,
            metadata,
            patch_state,
            tire_state,
            moment_control,
            moment_state,
            0.5,
        )

        self.assertAlmostEqual(float(moment_state.wheel_angular_speed.numpy()[0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.tire_reaction_torque.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(moment_state.net_torque.numpy()[0]), 0.0, delta=1e-6)

    def test_multi_wheel_moments_update_in_one_batch(self):
        model, metadata, *_ = _build_two_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata, active=False)
        tire_state = newton.wheeled.WheelTireState(model, metadata)
        moment_control = newton.wheeled.WheelMomentControl(model, metadata)
        moment_state = newton.wheeled.WheelMomentState(model, metadata)

        newton.wheeled.configure_wheel_moment_control(
            moment_control,
            drive_torque=[1.0, -2.0],
            wheel_inertia=[0.5, 2.0],
            apply_body_reaction_torque=False,
        )

        newton.wheeled.update_wheel_moments(
            model,
            state,
            metadata,
            patch_state,
            tire_state,
            moment_control,
            moment_state,
            0.5,
        )

        np.testing.assert_allclose(moment_state.wheel_angular_speed.numpy(), np.array([1.0, -0.5]), atol=1e-6)
        np.testing.assert_allclose(moment_state.wheel_angular_acceleration.numpy(), np.array([2.0, -1.0]), atol=1e-6)
        np.testing.assert_allclose(moment_state.net_torque.numpy(), np.array([1.0, -2.0]), atol=1e-6)


class TestWheelAxleJointConfiguration(unittest.TestCase):
    def test_configure_wheel_axle_joints_converts_revolute_to_fixed(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="test_dof",
                frequency=newton.Model.AttributeFrequency.JOINT_DOF,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=-1.0,
            )
        )
        chassis = builder.add_body(label="chassis")
        wheel = builder.add_body(label="wheel")
        slider = builder.add_body(label="slider")
        axle = builder.add_joint_revolute(chassis, wheel, axis=wp.vec3(0.0, 1.0, 0.0), label="wheel_axle")
        slide = builder.add_joint_prismatic(
            chassis,
            slider,
            axis=wp.vec3(1.0, 0.0, 0.0),
            label="kept_slide",
            custom_attributes={"test_dof": 7.0},
        )

        old_dof_count = builder.joint_dof_count
        old_coord_count = builder.joint_coord_count
        old_constraint_count = builder.joint_constraint_count
        old_slide_q_start = builder.joint_q_start[slide]
        old_slide_qd_start = builder.joint_qd_start[slide]
        old_slide_cts_start = builder.joint_cts_start[slide]

        converted = newton.wheeled.configure_wheel_axle_joints(builder, axle_joint_labels=["wheel_axle"])

        self.assertEqual(converted, (axle,))
        self.assertEqual(builder.joint_type[axle], newton.JointType.FIXED)
        self.assertEqual(builder.joint_dof_dim[axle], (0, 0))
        self.assertEqual(builder.joint_q_start[slide], old_slide_q_start - 1)
        self.assertEqual(builder.joint_qd_start[slide], old_slide_qd_start - 1)
        self.assertEqual(builder.joint_cts_start[slide], old_slide_cts_start + 1)
        self.assertEqual(builder.joint_dof_count, old_dof_count - 1)
        self.assertEqual(builder.joint_coord_count, old_coord_count - 1)
        self.assertEqual(builder.joint_constraint_count, old_constraint_count + 1)
        self.assertEqual(builder.custom_attributes["test_dof"].values, {old_slide_qd_start - 1: 7.0})

        model = builder.finalize(device="cpu")

        self.assertEqual(int(model.joint_type.numpy()[axle]), int(newton.JointType.FIXED))
        self.assertEqual(int(model.joint_qd_start.numpy()[slide]), old_slide_qd_start - 1)
        self.assertEqual(model.joint_qd.shape, (old_dof_count - 1,))

    def test_configure_wheel_axle_joints_can_resolve_by_wheel_body_label(self):
        builder = newton.ModelBuilder(gravity=0.0)
        chassis = builder.add_body(label="chassis")
        wheel = builder.add_body(label="wheel")
        axle = builder.add_joint_revolute(chassis, wheel, axis=wp.vec3(0.0, 1.0, 0.0), label="wheel_axle")

        converted = newton.wheeled.configure_wheel_axle_joints(builder, wheel_body_labels=["wheel"])

        self.assertEqual(converted, (axle,))
        self.assertEqual(builder.joint_type[axle], newton.JointType.FIXED)

    def test_configure_wheel_axle_joints_rejects_non_revolute(self):
        builder = newton.ModelBuilder(gravity=0.0)
        chassis = builder.add_body(label="chassis")
        wheel = builder.add_body(label="wheel")
        builder.add_joint_prismatic(chassis, wheel, axis=wp.vec3(1.0, 0.0, 0.0), label="not_axle")

        with self.assertRaisesRegex(ValueError, "must be revolute"):
            newton.wheeled.configure_wheel_axle_joints(builder, axle_joint_labels=["not_axle"])


class TestMujocoWheelContactConfiguration(unittest.TestCase):
    def test_configures_wheel_geom_condim_and_priority(self):
        model, metadata, _, shape = _build_one_wheel_model(mujoco_attrs=True)

        self.assertEqual(int(model.mujoco.condim.numpy()[shape]), 3)
        self.assertEqual(int(model.mujoco.geom_priority.numpy()[shape]), 0)

        newton.wheeled.configure_mujoco_wheel_contacts(model, metadata)

        self.assertEqual(int(model.mujoco.condim.numpy()[shape]), 1)
        self.assertEqual(int(model.mujoco.geom_priority.numpy()[shape]), 1)

        newton.wheeled.configure_mujoco_wheel_contacts(model, metadata, condim=3, priority=4)

        self.assertEqual(int(model.mujoco.condim.numpy()[shape]), 3)
        self.assertEqual(int(model.mujoco.geom_priority.numpy()[shape]), 4)

    def test_requires_registered_mujoco_attributes(self):
        model, metadata, *_ = _build_one_wheel_model()

        with self.assertRaisesRegex(ValueError, "SolverMuJoCo.register_custom_attributes"):
            newton.wheeled.configure_mujoco_wheel_contacts(model, metadata)


class TestWheelTireForces(unittest.TestCase):
    def test_longitudinal_velocity_slip_applies_force_and_wrench(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        control.wheel_angular_speed.assign(np.array([4.0], dtype=np.float32))
        control.longitudinal_stiffness.assign(np.array([2.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([10.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        self.assertAlmostEqual(float(tire_state.wheel_angular_speed.numpy()[0]), 4.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.longitudinal_speed.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.longitudinal_slip_speed.numpy()[0]), 2.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.longitudinal_slip_ratio.numpy()[0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.requested_longitudinal_force.numpy()[0]), 4.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_longitudinal_force.numpy()[0]), 4.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.combined_slip_scale.numpy()[0]), 1.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([4.0, 0.0, 0.0, 0.0, -2.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_combined_lateral_longitudinal_forces_saturate_force_circle(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        state.body_qd.assign(np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        patch_state = _make_patch_state(model, metadata)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        control.wheel_angular_speed.assign(np.array([4.0], dtype=np.float32))
        control.longitudinal_stiffness.assign(np.array([2.0], dtype=np.float32))
        control.lateral_stiffness.assign(np.array([3.0], dtype=np.float32))
        control.friction_mu.assign(np.array([0.25], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([10.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        self.assertAlmostEqual(float(tire_state.requested_longitudinal_force.numpy()[0]), 4.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.requested_lateral_force.numpy()[0]), -3.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.friction_limit.numpy()[0]), 2.5, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.combined_slip_scale.numpy()[0]), 0.5, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_longitudinal_force.numpy()[0]), 2.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_lateral_force.numpy()[0]), -1.5, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([2.0, -1.5, 0.0, -0.75, -1.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_fiala_lateral_force_matches_brush_formula_before_saturation(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        state.body_qd.assign(np.array([[1.0, 0.05, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        patch_state = _make_patch_state(model, metadata)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        control.tire_model.assign(np.array([newton.wheeled.WheelTireControl.TireModel.FIALA], dtype=np.int32))
        control.lateral_stiffness.assign(np.array([200.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([100.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        expected = _fiala_lateral_force(0.05, 200.0, 100.0)
        self.assertAlmostEqual(float(tire_state.lateral_slip_angle.numpy()[0]), np.arctan2(0.05, 1.0), delta=1e-6)
        self.assertAlmostEqual(float(tire_state.requested_lateral_force.numpy()[0]), expected, delta=1e-5)
        self.assertAlmostEqual(float(tire_state.applied_lateral_force.numpy()[0]), expected, delta=1e-5)
        self.assertAlmostEqual(float(tire_state.combined_slip_scale.numpy()[0]), 1.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([0.0, expected, 0.0, 0.5 * expected, 0.0, 0.0], dtype=np.float32),
            atol=1e-5,
        )

    def test_fiala_lateral_force_saturates_at_friction_limit(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        state.body_qd.assign(np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        patch_state = _make_patch_state(model, metadata)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        control.tire_model.assign(np.array([newton.wheeled.WheelTireControl.TireModel.FIALA], dtype=np.int32))
        control.lateral_stiffness.assign(np.array([1000.0], dtype=np.float32))
        control.friction_mu.assign(np.array([0.5], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([100.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        self.assertAlmostEqual(float(tire_state.friction_limit.numpy()[0]), 50.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.requested_lateral_force.numpy()[0]), -50.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_lateral_force.numpy()[0]), -50.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.combined_slip_scale.numpy()[0]), 1.0, delta=1e-6)
        np.testing.assert_allclose(
            state.body_f.numpy()[body],
            np.array([0.0, -50.0, 0.0, -25.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_physical_body_spin_does_not_replace_analytical_wheel_speed(self):
        model, metadata, body, *_ = _build_one_wheel_model()
        state = model.state()
        state.body_qd.assign(np.array([[0.0, 0.0, 0.0, 0.0, 20.0, 0.0]], dtype=np.float32))
        patch_state = _make_patch_state(model, metadata, center_z=0.5)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        control.longitudinal_stiffness.assign(np.array([2.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([10.0], dtype=np.float32))

        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        self.assertAlmostEqual(float(tire_state.wheel_angular_speed.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.longitudinal_slip_speed.numpy()[0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_longitudinal_force.numpy()[0]), 0.0, delta=1e-6)
        np.testing.assert_allclose(state.body_f.numpy()[body], np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_normal_load_latching_and_fallback(self):
        model, metadata, *_ = _build_one_wheel_model()
        state = model.state()
        patch_state = _make_patch_state(model, metadata)
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        patch_state.normal_force.assign(np.array([12.0], dtype=np.float32))
        newton.wheeled.update_wheel_tire_normal_loads(patch_state, tire_state)
        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.array([12.0], dtype=np.float32))

        control.wheel_angular_speed.assign(np.array([100.0], dtype=np.float32))
        control.longitudinal_stiffness.assign(np.array([100.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([3.0], dtype=np.float32))
        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)
        self.assertAlmostEqual(float(tire_state.normal_load.numpy()[0]), 12.0, delta=1e-6)
        self.assertAlmostEqual(float(tire_state.applied_longitudinal_force.numpy()[0]), 12.0, delta=1e-6)

        patch_state.active.assign(np.array([False], dtype=bool))
        patch_state.normal_force.assign(np.array([0.0], dtype=np.float32))
        newton.wheeled.update_wheel_tire_normal_loads(patch_state, tire_state)
        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.array([12.0], dtype=np.float32))

        newton.wheeled.update_wheel_tire_normal_loads(patch_state, tire_state, clear_inactive=True)
        np.testing.assert_allclose(tire_state.previous_normal_load.numpy(), np.zeros(1, dtype=np.float32))

    def test_newton_contact_flow_applies_tire_force_on_flat_patch(self):
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
        control = newton.wheeled.WheelTireControl(model, metadata)
        tire_state = newton.wheeled.WheelTireState(model, metadata)

        model.collide(state, contacts)
        newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)
        control.wheel_angular_speed.assign(np.array([10.0], dtype=np.float32))
        control.longitudinal_stiffness.assign(np.array([2.0], dtype=np.float32))
        control.friction_mu.assign(np.array([1.0], dtype=np.float32))
        control.fallback_normal_load.assign(np.array([50.0], dtype=np.float32))
        newton.wheeled.apply_wheel_tire_forces(model, state, metadata, patch_state, control, tire_state)

        self.assertTrue(bool(patch_state.active.numpy()[0]))
        self.assertGreater(float(tire_state.applied_longitudinal_force.numpy()[0]), 0.0)
        self.assertGreater(float(state.body_f.numpy()[body, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
