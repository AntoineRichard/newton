# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_VEHICLE_NAMES = ("rc_car", "husky")


def _add_wheel(builder, *, vehicle_id, wheel_id, chassis, name, radius=0.1, steer=False):
    body = builder.add_body(label=f"vehicle{vehicle_id}_{name}_wheel")
    builder.add_shape_sphere(
        body,
        radius=radius,
        label=f"vehicle{vehicle_id}_{name}_shape",
        custom_attributes={
            "wheeled:is_wheel": True,
            "wheeled:wheel_id": wheel_id,
            "wheeled:vehicle_id": vehicle_id,
            "wheeled:wheel_radius": radius,
            "wheeled:wheel_width": 0.05,
        },
    )
    builder.custom_attributes["wheeled:is_wheel_body"].values[body] = True
    builder.custom_attributes["wheeled:wheel_body_id"].values[body] = wheel_id
    joint = -1
    if steer:
        joint = builder.add_joint_revolute(
            chassis,
            body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            label=f"vehicle{vehicle_id}_{name}_steering",
        )
    return body, joint


def _build_two_vehicle_model(*, device="cpu"):
    builder = newton.ModelBuilder(gravity=0.0)
    newton.wheeled.register_wheeled_custom_attributes(builder)

    steering_joints = {}
    wheel_id = 0
    for vehicle_id in range(2):
        chassis = builder.add_body(label=f"vehicle{vehicle_id}_chassis")
        for name, steer in (
            ("front_left", vehicle_id == 0),
            ("rear_left", False),
            ("front_right", vehicle_id == 0),
            ("rear_right", False),
        ):
            _body, joint = _add_wheel(
                builder,
                vehicle_id=vehicle_id,
                wheel_id=wheel_id,
                chassis=chassis,
                name=name,
                steer=steer,
            )
            if joint >= 0:
                steering_joints[name] = joint
            wheel_id += 1

    model = builder.finalize(device=device)
    metadata = newton.wheeled.build_wheeled_metadata(model)
    return model, metadata, steering_joints


def _explicit_layout(model, metadata, steering_joints):
    qd_start = model.joint_qd_start.numpy()
    steering_dof = np.full(metadata.wheel_count, -1, dtype=np.int32)
    steering_dof[0] = int(qd_start[steering_joints["front_left"]])
    steering_dof[2] = int(qd_start[steering_joints["front_right"]])

    return newton.wheeled.build_wheeled_vehicle_layout(
        model,
        metadata,
        vehicle_geometry_kind=[
            newton.wheeled.WheeledVehicleLayout.GeometryKind.ACKERMANN,
            newton.wheeled.WheeledVehicleLayout.GeometryKind.SKID_STEER,
        ],
        wheel_drive_channel=[0, 0, 0, 0, 1, 1, 2, 2],
        wheel_steering_channel=[0, -1, 0, -1, -1, -1, -1, -1],
        wheel_steering_joint_dof_index=steering_dof,
        wheel_side=[-1, -1, 1, 1, -1, -1, 1, 1],
        wheel_axle=[1, -1, 1, -1, 1, -1, 1, -1],
        vehicle_wheelbase=[0.32, 0.5],
        vehicle_track_width=[0.3, 0.6],
        vehicle_steering_limit=[0.5, 0.0],
    )


class TestWheeledVehicleDriveModePublicApi(unittest.TestCase):
    def test_public_imports(self):
        from newton.wheeled import (  # noqa: PLC0415
            WheeledMotorConfig,
            WheeledSteeringConfig,
            WheeledVehicleControl,
            WheeledVehicleLayout,
            WheeledVehicleState,
            build_wheeled_vehicle_layout,
            configure_wheeled_vehicle_control,
            update_wheeled_vehicle_controls,
        )

        self.assertTrue(callable(WheeledMotorConfig))
        self.assertTrue(callable(WheeledSteeringConfig))
        self.assertTrue(callable(WheeledVehicleControl))
        self.assertTrue(callable(WheeledVehicleLayout))
        self.assertTrue(callable(WheeledVehicleState))
        self.assertTrue(callable(build_wheeled_vehicle_layout))
        self.assertTrue(callable(configure_wheeled_vehicle_control))
        self.assertTrue(callable(update_wheeled_vehicle_controls))

    def test_channel_control_defaults_do_not_expose_geometry_specific_fields(self):
        model, metadata, steering_joints = _build_two_vehicle_model()
        layout = _explicit_layout(model, metadata, steering_joints)
        control = newton.wheeled.WheeledVehicleControl(layout)
        state = newton.wheeled.WheeledVehicleState(layout)

        self.assertEqual(control.enabled.shape, (2,))
        self.assertEqual(control.drive_command.shape, (3,))
        self.assertEqual(control.steering_command.shape, (1,))
        self.assertFalse(hasattr(control, "left_drive_command"))
        self.assertFalse(hasattr(control, "right_drive_command"))
        self.assertFalse(hasattr(control, "steering_angle"))
        np.testing.assert_array_equal(control.enabled.numpy(), np.array([True, True]))
        np.testing.assert_allclose(control.drive_command.numpy(), np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(state.wheel_angular_speed.numpy(), np.zeros(8, dtype=np.float32))


class TestWheeledVehicleLayout(unittest.TestCase):
    def test_explicit_layout_allocates_channels_and_roles(self):
        model, metadata, steering_joints = _build_two_vehicle_model()
        layout = _explicit_layout(model, metadata, steering_joints)

        self.assertEqual(layout.vehicle_count, 2)
        self.assertEqual(layout.wheel_count, 8)
        self.assertEqual(layout.drive_channel_count, 3)
        self.assertEqual(layout.steering_channel_count, 1)
        self.assertEqual(layout.vehicle_drive_channels, ((0,), (1, 2)))
        self.assertEqual(layout.vehicle_steering_channels, ((0,), ()))
        self.assertEqual(layout.wheel_drive_channel_host, (0, 0, 0, 0, 1, 1, 2, 2))
        self.assertEqual(layout.wheel_steering_channel_host, (0, -1, 0, -1, -1, -1, -1, -1))

    def test_layout_rejects_steering_channel_without_dof(self):
        model, metadata, steering_joints = _build_two_vehicle_model()
        qd_start = model.joint_qd_start.numpy()
        steering_dof = np.full(metadata.wheel_count, -1, dtype=np.int32)
        steering_dof[0] = int(qd_start[steering_joints["front_left"]])

        with self.assertRaisesRegex(ValueError, "steering channel"):
            newton.wheeled.build_wheeled_vehicle_layout(
                model,
                metadata,
                vehicle_geometry_kind=["ackermann", "skid_steer"],
                wheel_drive_channel=[0, 0, 0, 0, 1, 1, 2, 2],
                wheel_steering_channel=[0, -1, 0, -1, -1, -1, -1, -1],
                wheel_steering_joint_dof_index=steering_dof,
            )


class TestWheeledVehicleCommandMapping(unittest.TestCase):
    def test_update_maps_skid_steer_and_ackermann_channels(self):
        model, metadata, steering_joints = _build_two_vehicle_model()
        layout = _explicit_layout(model, metadata, steering_joints)
        control = newton.wheeled.WheeledVehicleControl(layout)
        state = newton.wheeled.WheeledVehicleState(layout)
        tire_control = newton.wheeled.WheelTireControl(model, metadata)
        sim_control = model.control()
        motor_config = newton.wheeled.WheeledMotorConfig(layout, max_wheel_angular_speed=[10.0, 20.0, 20.0])
        steering_config = newton.wheeled.WheeledSteeringConfig(layout, max_steering_angle=[0.4])

        newton.wheeled.configure_wheeled_vehicle_control(
            control,
            drive_command=[0.5, -0.5, 0.5],
            steering_command=[0.5],
        )
        newton.wheeled.update_wheeled_vehicle_controls(
            model,
            sim_control,
            metadata,
            layout,
            control,
            state,
            tire_control,
            motor_config=motor_config,
            steering_config=steering_config,
        )

        np.testing.assert_allclose(
            tire_control.wheel_angular_speed.numpy(),
            np.array([5.0, 5.0, 5.0, 5.0, -10.0, -10.0, 10.0, 10.0], dtype=np.float32),
        )
        np.testing.assert_allclose(state.drive_command.numpy(), np.array([0.5, -0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(state.steering_command.numpy(), np.array([0.5], dtype=np.float32))
        np.testing.assert_allclose(state.steering_angle.numpy(), np.array([0.2], dtype=np.float32))

        target_pos = sim_control.joint_target_pos.numpy()
        left_angle = target_pos[int(model.joint_qd_start.numpy()[steering_joints["front_left"]])]
        right_angle = target_pos[int(model.joint_qd_start.numpy()[steering_joints["front_right"]])]
        self.assertGreater(left_angle, right_angle)
        self.assertGreater(left_angle, 0.0)
        self.assertGreater(right_angle, 0.0)

    def test_update_clamps_commands_and_respects_enabled(self):
        model, metadata, steering_joints = _build_two_vehicle_model()
        layout = _explicit_layout(model, metadata, steering_joints)
        control = newton.wheeled.WheeledVehicleControl(layout)
        state = newton.wheeled.WheeledVehicleState(layout)
        tire_control = newton.wheeled.WheelTireControl(model, metadata)
        sim_control = model.control()
        motor_config = newton.wheeled.WheeledMotorConfig(layout, max_wheel_angular_speed=10.0)

        newton.wheeled.configure_wheeled_vehicle_control(
            control,
            enabled=[False, True],
            drive_command=[2.0, -2.0, 0.25],
            steering_command=[2.0],
        )
        newton.wheeled.update_wheeled_vehicle_controls(
            model,
            sim_control,
            metadata,
            layout,
            control,
            state,
            tire_control,
            motor_config=motor_config,
        )

        np.testing.assert_allclose(
            tire_control.wheel_angular_speed.numpy(),
            np.array([0.0, 0.0, 0.0, 0.0, -10.0, -10.0, 2.5, 2.5], dtype=np.float32),
        )
        np.testing.assert_allclose(state.drive_command.numpy(), np.array([1.0, -1.0, 0.25], dtype=np.float32))
        np.testing.assert_allclose(state.wheel_steering_angle.numpy()[:4], np.zeros(4, dtype=np.float32))


class TestWheeledVehicleManifestLayout(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_manifest_layout_handles_mixed_replicated_assets(self):
        assets = {asset.name: asset for asset in newton.wheeled.load_wheeled_manifest(_MANIFEST_PATH)}
        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        newton.wheeled.register_wheeled_custom_attributes(world)
        for vehicle_name in _VEHICLE_NAMES:
            world.add_usd(
                str(assets[vehicle_name].file),
                enable_self_collisions=False,
                schema_resolvers=[newton.usd.SchemaResolverPhysx()],
            )
        newton.wheeled.apply_wheeled_manifest(world, _MANIFEST_PATH, asset_names=_VEHICLE_NAMES)
        newton.wheeled.configure_wheel_axle_joints(
            world,
            axle_joint_labels=[
                label for vehicle_name in _VEHICLE_NAMES for label in assets[vehicle_name].axle_joint_labels
            ],
        )

        scene = newton.ModelBuilder()
        scene.replicate(world, 2)
        model = scene.finalize(device="cpu")
        metadata = newton.wheeled.build_wheeled_metadata(model)
        layout = newton.wheeled.build_wheeled_vehicle_layout(
            model,
            metadata,
            manifest_path=_MANIFEST_PATH,
            asset_names=_VEHICLE_NAMES,
        )

        self.assertEqual(layout.vehicle_count, 4)
        self.assertEqual(layout.wheel_count, 16)
        self.assertEqual(layout.drive_channel_count, 6)
        self.assertEqual(layout.steering_channel_count, 2)
        self.assertEqual(
            layout.vehicle_geometry_kind_host,
            (
                newton.wheeled.WheeledVehicleLayout.GeometryKind.ACKERMANN,
                newton.wheeled.WheeledVehicleLayout.GeometryKind.SKID_STEER,
                newton.wheeled.WheeledVehicleLayout.GeometryKind.ACKERMANN,
                newton.wheeled.WheeledVehicleLayout.GeometryKind.SKID_STEER,
            ),
        )
        self.assertEqual(layout.vehicle_steering_channels[0], (0,))
        self.assertEqual(layout.vehicle_steering_channels[1], ())
        self.assertEqual(layout.vehicle_steering_channels[2], (1,))
        self.assertEqual(layout.vehicle_steering_channels[3], ())

        vehicle_control = newton.wheeled.WheeledVehicleControl(layout)
        vehicle_state = newton.wheeled.WheeledVehicleState(layout)
        tire_control = newton.wheeled.WheelTireControl(model, metadata)
        sim_control = model.control()
        newton.wheeled.configure_wheeled_vehicle_control(
            vehicle_control,
            drive_command=[0.25] * layout.drive_channel_count,
            steering_command=[0.5] * layout.steering_channel_count,
        )
        newton.wheeled.update_wheeled_vehicle_controls(
            model,
            sim_control,
            metadata,
            layout,
            vehicle_control,
            vehicle_state,
            tire_control,
        )

        self.assertGreater(np.max(np.abs(tire_control.wheel_angular_speed.numpy())), 0.0)
        self.assertGreater(np.max(np.abs(sim_control.joint_target_pos.numpy())), 0.0)


if __name__ == "__main__":
    unittest.main()
