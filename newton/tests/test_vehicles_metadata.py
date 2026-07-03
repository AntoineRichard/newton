# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.vehicles as nv
from newton._src.geometry.flags import ShapeFlags
from newton.tests.unittest_utils import USD_AVAILABLE

_ASSET_DIR = Path(__file__).resolve().parents[1] / "examples" / "assets" / "wheeled"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"


def _asset_by_name(name: str):
    assets = nv.load_vehicle_manifest(_MANIFEST_PATH)
    return {asset.name: asset for asset in assets}[name]


def _import_fixture(asset):
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    builder.add_usd(str(asset.file))
    return builder


# wheel layout: (x, y, axle_row) -> front (row 0) at +x, rear (row 1) at -x
_LAYOUT = ((0.5, 0.3, 0), (0.5, -0.3, 0), (-0.5, 0.3, 1), (-0.5, -0.3, 1))


def _add_car(builder, *, vehicle_id, wheel_id_start, drive_mode, steer_front):
    nv.set_vehicle(builder, vehicle_id, drive_mode=int(drive_mode), wheelbase=1.0, track_width=0.6, steer_limit=0.5)
    chassis = builder.add_body()
    shapes = []
    for i, (x, y, axle_row) in enumerate(_LAYOUT):
        wb = builder.add_body(xform=wp.transform(wp.vec3(x, y, 0.1), wp.quat_identity()))
        s = builder.add_shape_cylinder(wb, radius=0.1, half_height=0.05)
        steerable = steer_front and axle_row == 0
        steer_joint = builder.add_joint_revolute(chassis, wb, axis=(0.0, 0.0, 1.0)) if steerable else -1
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=vehicle_id,
            wheel_id=wheel_id_start + i,
            radius=0.1,
            width=0.1,
            driven=True,
            steerable=steerable,
            side=(-1 if y > 0 else 1),
            axle_row=axle_row,
            steer_joint=steer_joint,
        )
        shapes.append(s)
    return shapes


class TestVehiclesImport(unittest.TestCase):
    def test_public_import(self):
        self.assertTrue(hasattr(nv, "WheeledVehicles"))
        self.assertTrue(hasattr(nv, "register_vehicle_attributes"))
        self.assertTrue(hasattr(nv, "add_wheel"))
        self.assertEqual(int(nv.DriveMode.ACKERMANN), 1)
        self.assertEqual(int(nv.WheeledVehicles.DriveMode.SKID_STEER), 2)


class TestVehicleMetadata(unittest.TestCase):
    def test_register_and_finalize(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        b = builder.add_body()
        builder.add_shape_cylinder(b, radius=0.1, half_height=0.05)
        model = builder.finalize()
        ns = getattr(model, nv.VEHICLE_NAMESPACE)
        self.assertTrue(hasattr(ns, "is_wheel"))
        self.assertEqual(len(ns.is_wheel.numpy()), model.shape_count)

    def test_add_wheel_sets_attrs_and_flag(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        shapes = _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        ns = getattr(model, nv.VEHICLE_NAMESPACE)
        is_wheel = ns.is_wheel.numpy()
        radius = ns.radius.numpy()
        flags = model.shape_flags.numpy()
        for s in shapes:
            self.assertTrue(bool(is_wheel[s]))
            self.assertAlmostEqual(float(radius[s]), 0.1, places=5)
            self.assertTrue(int(flags[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

    def test_read_flat_tables(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 4)
        self.assertEqual(data.vehicle_count, 1)
        self.assertTrue((data.wheel_vehicle.numpy() == 0).all())
        self.assertTrue((abs(data.radius.numpy() - 0.1) < 1e-5).all())
        self.assertEqual(int(data.drive_mode.numpy()[0]), int(nv.DriveMode.ACKERMANN))
        self.assertAlmostEqual(float(data.wheelbase.numpy()[0]), 1.0, places=5)
        self.assertEqual(int(data.vehicle_wheel_count.numpy()[0]), 4)

    def test_steer_dof_resolution(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        steer_dof = data.steer_dof.numpy()
        steerable = data.steerable.numpy()
        axle_row = data.axle_row.numpy()
        for w in range(data.wheel_count):
            if axle_row[w] == 0:  # front wheels are steerable
                self.assertEqual(int(steerable[w]), 1)
                self.assertGreaterEqual(int(steer_dof[w]), 0)
            else:
                self.assertEqual(int(steer_dof[w]), -1)

    def test_heterogeneous_two_vehicles(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        _add_car(builder, vehicle_id=1, wheel_id_start=4, drive_mode=nv.DriveMode.SKID_STEER, steer_front=False)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        self.assertEqual(int(data.drive_mode.numpy()[0]), int(nv.DriveMode.ACKERMANN))
        self.assertEqual(int(data.drive_mode.numpy()[1]), int(nv.DriveMode.SKID_STEER))
        self.assertEqual(list(data.wheel_vehicle.numpy()), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_replication_preserves_ids(self):
        sub = newton.ModelBuilder()
        nv.register_vehicle_attributes(sub)
        _add_car(sub, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)

        main = newton.ModelBuilder()
        main.add_builder(sub)
        main.add_builder(sub)
        model = main.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        self.assertEqual(list(data.wheel_vehicle.numpy()), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_controller_construction(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        vehicles = nv.WheeledVehicles(model)
        self.assertEqual(vehicles.wheel_count, 4)
        self.assertEqual(vehicles.vehicle_count, 1)


class TestVehicleManifestLoading(unittest.TestCase):
    def test_load_manifest(self):
        assets = nv.load_vehicle_manifest(_MANIFEST_PATH)
        by_name = {asset.name: asset for asset in assets}

        self.assertEqual(set(by_name), {"rc_car", "husky"})

        rc_car = by_name["rc_car"]
        self.assertEqual(len(rc_car.wheel_body_labels), 4)
        self.assertEqual(len(rc_car.wheel_shape_labels), 4)
        self.assertAlmostEqual(rc_car.wheel_radius, 0.055)
        self.assertAlmostEqual(rc_car.wheel_width, 0.045)
        self.assertEqual(rc_car.drive_mode, int(nv.DriveMode.ACKERMANN))
        self.assertAlmostEqual(rc_car.wheelbase, 0.324)
        self.assertAlmostEqual(rc_car.track_width, 0.296)
        self.assertAlmostEqual(rc_car.steer_limit, math.radians(35.0))
        self.assertEqual(len(rc_car.steering_joint_labels), 2)
        self.assertEqual(len(rc_car.axle_joint_labels), 4)
        self.assertTrue(rc_car.file.exists())

        husky = by_name["husky"]
        self.assertEqual(husky.drive_mode, int(nv.DriveMode.SKID_STEER))
        self.assertAlmostEqual(husky.wheel_radius, 0.1625)
        self.assertAlmostEqual(husky.wheel_width, 0.13)
        self.assertEqual(husky.steering_joint_labels, ())
        self.assertAlmostEqual(husky.steer_limit, 0.0)

    def test_rejects_invalid_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            (tmp_dir / "asset.usda").write_text("#usda 1.0\n")

            manifest = {
                "version": 1,
                "assets": [
                    {
                        "name": "bad",
                        "file": "asset.usda",
                        "reference_dimensions": {"wheel_radius_m": 0.1, "wheel_width_m": 0.2},
                        "wheel_body_labels": ["/body"],
                        "wheel_shape_labels": [],
                    }
                ],
            }
            path = tmp_dir / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bad.*wheel_shape_labels"):
                nv.load_vehicle_manifest(path)

            manifest["assets"][0]["wheel_shape_labels"] = ["/shape"]
            manifest["assets"].append(dict(manifest["assets"][0]))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate.*bad"):
                nv.load_vehicle_manifest(path)

            manifest["assets"] = [manifest["assets"][0]]
            manifest["assets"][0]["vehicle_type"] = "hovercraft"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bad.*vehicle_type"):
                nv.load_vehicle_manifest(path)

            del manifest["assets"][0]["vehicle_type"]
            manifest["assets"][0]["file"] = "missing.usda"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bad.*file"):
                nv.load_vehicle_manifest(path)


class TestVehicleManifestRoleValidation(unittest.TestCase):
    @staticmethod
    def _synthetic(body_labels, steering_joint_labels=()):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        shape_labels = []
        for body_label in body_labels:
            body = builder.add_body(mass=1.0, label=body_label)
            shape_label = body_label + "/geom"
            builder.add_shape_cylinder(body, radius=0.1, half_height=0.05, label=shape_label)
            shape_labels.append(shape_label)
        asset = nv.VehicleAssetMetadata(
            name="synthetic",
            file=_MANIFEST_PATH,
            drive_mode=int(nv.DriveMode.GENERIC),
            wheel_radius=0.1,
            wheel_width=0.05,
            wheelbase=0.0,
            track_width=0.0,
            steer_limit=0.0,
            wheel_body_labels=tuple(body_labels),
            wheel_shape_labels=tuple(shape_labels),
            steering_joint_labels=tuple(steering_joint_labels),
        )
        return builder, asset

    def test_label_missing_front_rear_raises(self):
        builder, asset = self._synthetic(["/car/left_wheel"])
        with self.assertRaisesRegex(ValueError, "synthetic.*left_wheel.*'front' or 'rear'"):
            nv.apply_vehicle_manifest(builder, asset)

    def test_label_missing_left_right_raises(self):
        builder, asset = self._synthetic(["/car/front_wheel"])
        with self.assertRaisesRegex(ValueError, "synthetic.*front_wheel.*'left' or 'right'"):
            nv.apply_vehicle_manifest(builder, asset)

    def test_unmatched_steering_side_raises(self):
        builder, asset = self._synthetic(["/car/front_left_wheel"], steering_joint_labels=["/car/steer_right"])
        with self.assertRaisesRegex(ValueError, "synthetic.*exactly one steering joint label containing 'left'"):
            nv.apply_vehicle_manifest(builder, asset)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestVehicleManifestApplication(unittest.TestCase):
    def test_apply_manifest_to_imported_rc_car(self):
        asset = _asset_by_name("rc_car")
        builder = _import_fixture(asset)

        nv.apply_vehicle_manifest(builder, asset)

        model = builder.finalize(device="cpu")
        data = nv.read_vehicle_model_data(model)

        self.assertEqual(data.wheel_count, 4)
        self.assertEqual(data.vehicle_count, 1)
        np.testing.assert_array_equal(data.wheel_vehicle.numpy(), [0, 0, 0, 0])
        np.testing.assert_allclose(data.radius.numpy(), [0.055] * 4)
        np.testing.assert_allclose(data.width.numpy(), [0.045] * 4)
        self.assertEqual(int(data.drive_mode.numpy()[0]), int(nv.DriveMode.ACKERMANN))
        np.testing.assert_allclose(data.wheelbase.numpy(), [0.324])
        np.testing.assert_allclose(data.steer_limit.numpy(), [math.radians(35.0)], rtol=1e-6)

        # Manifest order is FL, RL, FR, RR: wheel ids follow the manifest, so the
        # wheel tables map back to the manifest-labelled shapes in that order.
        wheel_shapes = data.wheel_shape.numpy()
        labels = [model.shape_label[int(s)] for s in wheel_shapes]
        self.assertEqual(labels, list(asset.wheel_shape_labels))

        np.testing.assert_array_equal(data.side.numpy(), [-1, -1, 1, 1])
        np.testing.assert_array_equal(data.axle_row.numpy(), [0, 1, 0, 1])
        np.testing.assert_array_equal(data.steerable.numpy(), [1, 0, 1, 0])
        np.testing.assert_array_equal(data.driven.numpy(), [1, 1, 1, 1])
        steer_dof = data.steer_dof.numpy()
        self.assertTrue((steer_dof[[0, 2]] >= 0).all())
        np.testing.assert_array_equal(steer_dof[[1, 3]], [-1, -1])

        # shape_to_wheel is the inverse of wheel_shape.
        shape_to_wheel = data.shape_to_wheel.numpy()
        for wheel_id, shape in enumerate(wheel_shapes):
            self.assertEqual(int(shape_to_wheel[int(shape)]), wheel_id)

    def test_apply_manifest_multiple_assets(self):
        rc_car = _asset_by_name("rc_car")
        husky = _asset_by_name("husky")
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        builder.add_usd(str(rc_car.file))
        builder.add_usd(str(husky.file))

        nv.apply_vehicle_manifest(builder, rc_car)
        nv.apply_vehicle_manifest(builder, husky)

        model = builder.finalize(device="cpu")
        data = nv.read_vehicle_model_data(model)

        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        np.testing.assert_array_equal(data.wheel_vehicle.numpy(), [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(data.vehicle_wheel_count.numpy(), [4, 4])
        np.testing.assert_array_equal(
            data.drive_mode.numpy(), [int(nv.DriveMode.ACKERMANN), int(nv.DriveMode.SKID_STEER)]
        )
        np.testing.assert_allclose(data.radius.numpy(), [0.055] * 4 + [0.1625] * 4)
        np.testing.assert_array_equal(data.steerable.numpy(), [1, 0, 1, 0, 0, 0, 0, 0])

    def test_apply_manifest_replicated(self):
        asset = _asset_by_name("rc_car")
        template = _import_fixture(asset)
        nv.apply_vehicle_manifest(template, asset)

        scene = newton.ModelBuilder()
        scene.replicate(template, 2)
        model = scene.finalize(device="cpu")
        data = nv.read_vehicle_model_data(model)

        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        np.testing.assert_array_equal(data.wheel_vehicle.numpy(), [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(data.radius.numpy(), [0.055] * 8)

    def test_missing_manifest_label_raises(self):
        asset = _asset_by_name("husky")
        builder = _import_fixture(asset)
        broken = replace(asset, wheel_shape_labels=("/missing/shape", *asset.wheel_shape_labels[1:]))

        with self.assertRaisesRegex(ValueError, "husky.*wheel shape.*missing"):
            nv.apply_vehicle_manifest(builder, broken)

    def test_shape_body_mismatch_raises(self):
        asset = _asset_by_name("rc_car")
        builder = _import_fixture(asset)
        swapped = replace(
            asset,
            wheel_shape_labels=(
                asset.wheel_shape_labels[1],
                asset.wheel_shape_labels[0],
                *asset.wheel_shape_labels[2:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "rc_car.*attached to body"):
            nv.apply_vehicle_manifest(builder, swapped)


if __name__ == "__main__":
    unittest.main()
