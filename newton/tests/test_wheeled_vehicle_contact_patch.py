# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _build_synthetic_wheel_model(*, device="cpu"):
    builder = newton.ModelBuilder(gravity=0.0)
    newton.wheeled.register_wheeled_custom_attributes(builder)

    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 0.7
    terrain_shape = builder.add_ground_plane(cfg=terrain_cfg, label="terrain")

    wheel0_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="wheel0")
    wheel0_shape = builder.add_shape_sphere(
        wheel0_body,
        radius=0.5,
        label="wheel0_shape",
        custom_attributes={
            "wheeled:is_wheel": True,
            "wheeled:wheel_id": 0,
            "wheeled:vehicle_id": 0,
            "wheeled:wheel_radius": 0.5,
            "wheeled:wheel_width": 0.2,
        },
    )
    builder.custom_attributes["wheeled:is_wheel_body"].values[wheel0_body] = True
    builder.custom_attributes["wheeled:wheel_body_id"].values[wheel0_body] = 0

    wheel1_body = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()), label="wheel1")
    wheel1_shape = builder.add_shape_sphere(
        wheel1_body,
        radius=0.5,
        label="wheel1_shape",
        custom_attributes={
            "wheeled:is_wheel": True,
            "wheeled:wheel_id": 1,
            "wheeled:vehicle_id": 0,
            "wheeled:wheel_radius": 0.5,
            "wheeled:wheel_width": 0.2,
        },
    )
    builder.custom_attributes["wheeled:is_wheel_body"].values[wheel1_body] = True
    builder.custom_attributes["wheeled:wheel_body_id"].values[wheel1_body] = 1

    non_wheel_body = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()))
    non_wheel_shape = builder.add_shape_sphere(non_wheel_body, radius=0.25, label="non_wheel")

    model = builder.finalize(device=device)
    metadata = newton.wheeled.build_wheeled_metadata(model)
    return model, metadata, terrain_shape, wheel0_shape, wheel1_shape, non_wheel_shape


class TestWheelContactPatchState(unittest.TestCase):
    def test_public_import(self):
        self.assertTrue(hasattr(newton.wheeled, "WheelContactPatchState"))
        self.assertTrue(hasattr(newton.wheeled, "update_wheel_contact_patches"))

    def test_synthetic_contact_grouping_orientation_material_and_force(self):
        model, metadata, terrain_shape, wheel0_shape, wheel1_shape, non_wheel_shape = _build_synthetic_wheel_model()
        contacts = newton.Contacts(
            rigid_contact_max=4,
            soft_contact_max=0,
            device=model.device,
            requested_attributes={"force"},
        )
        contacts.rigid_contact_count.assign(np.array([4], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(
            np.array([wheel0_shape, terrain_shape, wheel0_shape, terrain_shape], dtype=np.int32)
        )
        contacts.rigid_contact_shape1.assign(
            np.array([terrain_shape, wheel1_shape, terrain_shape, non_wheel_shape], dtype=np.int32)
        )
        contacts.rigid_contact_point0.assign(
            np.array(
                [
                    [0.1, 0.0, -0.5],
                    [0.0, 0.0, 0.0],
                    [-0.1, 0.0, -0.5],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        contacts.rigid_contact_point1.assign(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.5],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.25],
                ],
                dtype=np.float32,
            )
        )
        contacts.rigid_contact_normal.assign(
            np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        contacts.force.assign(
            np.array(
                [
                    [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -12.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )

        state = model.state()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)

        np.testing.assert_array_equal(patch_state.active.numpy(), np.array([True, True]))
        np.testing.assert_array_equal(patch_state.contact_count.numpy(), np.array([2, 1], dtype=np.int32))
        np.testing.assert_array_equal(patch_state.terrain_shape_index.numpy(), np.array([terrain_shape, terrain_shape]))
        np.testing.assert_allclose(patch_state.normal.numpy(), np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), atol=1e-6)
        np.testing.assert_allclose(patch_state.friction_mu_seed.numpy(), np.array([0.7, 0.7]), rtol=1e-6)
        np.testing.assert_allclose(patch_state.normal_force.numpy(), np.array([12.0, 12.0]), rtol=1e-6)
        np.testing.assert_allclose(patch_state.center.numpy()[0], np.array([0.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(patch_state.center.numpy()[1], np.array([1.0, 0.0, 0.0]), atol=1e-6)

    def test_missing_force_buffer_leaves_normal_force_zero(self):
        model, metadata, terrain_shape, wheel0_shape, *_ = _build_synthetic_wheel_model()
        contacts = newton.Contacts(rigid_contact_max=1, soft_contact_max=0, device=model.device)
        contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([wheel0_shape], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([terrain_shape], dtype=np.int32))
        contacts.rigid_contact_point0.assign(np.array([[0.0, 0.0, -0.5]], dtype=np.float32))
        contacts.rigid_contact_point1.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        contacts.rigid_contact_normal.assign(np.array([[0.0, 0.0, -1.0]], dtype=np.float32))

        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        newton.wheeled.update_wheel_contact_patches(model, model.state(), contacts, metadata, patch_state)

        np.testing.assert_allclose(patch_state.normal_force.numpy(), np.array([0.0, 0.0]))

    def test_rejects_mismatched_metadata(self):
        model, metadata, *_ = _build_synthetic_wheel_model()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        contacts = newton.Contacts(rigid_contact_max=1, soft_contact_max=0, device=model.device)
        mismatched = newton.wheeled.WheeledModelMetadata(
            wheel_count=metadata.wheel_count,
            vehicle_count=metadata.vehicle_count,
            wheel_shape_indices=tuple(reversed(metadata.wheel_shape_indices)),
            wheel_body_indices=metadata.wheel_body_indices,
            wheel_vehicle_ids=metadata.wheel_vehicle_ids,
            wheel_radius=metadata.wheel_radius,
            wheel_width=metadata.wheel_width,
            vehicle_wheel_counts=metadata.vehicle_wheel_counts,
        )

        with self.assertRaisesRegex(ValueError, "metadata used to construct"):
            newton.wheeled.update_wheel_contact_patches(model, model.state(), contacts, mismatched, patch_state)

    def test_clear_restores_inactive_defaults(self):
        model, metadata, *_ = _build_synthetic_wheel_model()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        patch_state.active.fill_(True)
        patch_state.contact_count.fill_(3)
        patch_state.terrain_shape_index.fill_(5)
        patch_state.normal_force.fill_(10.0)

        patch_state.clear()

        np.testing.assert_array_equal(patch_state.active.numpy(), np.array([False, False]))
        np.testing.assert_array_equal(patch_state.contact_count.numpy(), np.array([0, 0], dtype=np.int32))
        np.testing.assert_array_equal(patch_state.terrain_shape_index.numpy(), np.array([-1, -1], dtype=np.int32))
        np.testing.assert_allclose(patch_state.normal_force.numpy(), np.array([0.0, 0.0]))

    def test_synthetic_multi_world_flat_wheel_ids(self):
        builder = newton.ModelBuilder(gravity=0.0)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        terrain_shapes = []
        wheel_shapes = []
        for wheel_id, x in enumerate((0.0, 1.0)):
            builder.begin_world()
            terrain_shapes.append(builder.add_ground_plane(label=f"terrain_{wheel_id}"))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.5), wp.quat_identity()),
                label=f"wheel_body_{wheel_id}",
                custom_attributes={"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": wheel_id},
            )
            wheel_shapes.append(
                builder.add_shape_sphere(
                    body,
                    radius=0.5,
                    label=f"wheel_shape_{wheel_id}",
                    custom_attributes={
                        "wheeled:is_wheel": True,
                        "wheeled:wheel_id": wheel_id,
                        "wheeled:vehicle_id": wheel_id,
                        "wheeled:wheel_radius": 0.5,
                        "wheeled:wheel_width": 0.2,
                    },
                )
            )
            builder.end_world()

        model = builder.finalize(device="cpu")
        metadata = newton.wheeled.build_wheeled_metadata(model)
        contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)
        contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array(wheel_shapes, dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array(terrain_shapes, dtype=np.int32))
        contacts.rigid_contact_point0.assign(np.array([[0.0, 0.0, -0.5], [0.0, 0.0, -0.5]], dtype=np.float32))
        contacts.rigid_contact_point1.assign(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        contacts.rigid_contact_normal.assign(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32))

        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        newton.wheeled.update_wheel_contact_patches(model, model.state(), contacts, metadata, patch_state)

        self.assertEqual(model.world_count, 2)
        np.testing.assert_array_equal(patch_state.active.numpy(), np.array([True, True]))
        np.testing.assert_array_equal(patch_state.contact_count.numpy(), np.array([1, 1], dtype=np.int32))
        np.testing.assert_array_equal(patch_state.terrain_shape_index.numpy(), np.array(terrain_shapes, dtype=np.int32))


class TestWheelContactPatchCollisionPipeline(unittest.TestCase):
    def test_sphere_wheel_on_plane_from_newton_contacts(self):
        builder = newton.ModelBuilder(gravity=0.0)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        terrain_shape = builder.add_ground_plane(label="terrain")
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
        model.collide(state, contacts)

        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        newton.wheeled.update_wheel_contact_patches(model, state, contacts, metadata, patch_state)

        self.assertTrue(bool(patch_state.active.numpy()[0]))
        self.assertGreaterEqual(int(patch_state.contact_count.numpy()[0]), 1)
        self.assertEqual(int(patch_state.terrain_shape_index.numpy()[0]), terrain_shape)
        self.assertGreater(patch_state.normal.numpy()[0, 2], 0.7)
        self.assertGreaterEqual(float(patch_state.patch_area.numpy()[0]), 0.0)
        self.assertEqual(float(patch_state.normal_force.numpy()[0]), 0.0)

    def test_mujoco_solver_uses_same_newton_contacts_for_support(self):
        builder = newton.ModelBuilder(gravity=-9.81)
        newton.wheeled.register_wheeled_custom_attributes(builder)
        builder.add_ground_plane(label="terrain")
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.75), wp.quat_identity()), label="wheel_body")
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

        contacts = model.contacts()
        patch_state = newton.wheeled.WheelContactPatchState(model, metadata)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        for _ in range(30):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            newton.wheeled.update_wheel_contact_patches(model, state_0, contacts, metadata, patch_state)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        self.assertGreater(state_0.body_q.numpy()[body, 2], 0.25)


if __name__ == "__main__":
    unittest.main()
