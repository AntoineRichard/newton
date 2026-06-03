# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import newton
from newton.tests.unittest_utils import USD_AVAILABLE

_ASSET_DIR = Path(__file__).resolve().parents[1] / "examples" / "assets" / "wheeled"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_TEST_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "wheeled"


def _asset_by_name(name: str):
    assets = newton.wheeled.load_wheeled_manifest(_MANIFEST_PATH)
    return {asset.name: asset for asset in assets}[name]


def _load_fixture_builder(asset, *, register: bool = True):
    builder = newton.ModelBuilder()
    if register:
        newton.wheeled.register_wheeled_custom_attributes(builder)
    builder.add_usd(str(asset.file), schema_resolvers=[newton.usd.SchemaResolverPhysx()])
    return builder


class TestWheeledMetadataRegistration(unittest.TestCase):
    def test_public_import_and_registration(self):
        self.assertTrue(hasattr(newton.wheeled, "register_wheeled_custom_attributes"))

        builder = newton.ModelBuilder()
        newton.wheeled.register_wheeled_custom_attributes(builder)

        expected = {
            "wheeled:is_wheel",
            "wheeled:wheel_id",
            "wheeled:vehicle_id",
            "wheeled:wheel_radius",
            "wheeled:wheel_width",
            "wheeled:is_wheel_body",
            "wheeled:wheel_body_id",
        }
        self.assertTrue(expected.issubset(builder.custom_attributes))

    def test_shape_and_body_defaults_after_finalize(self):
        builder = newton.ModelBuilder()
        newton.wheeled.register_wheeled_custom_attributes(builder)

        wheel_body = builder.add_body(
            mass=1.0,
            label="wheel_body",
            custom_attributes={
                "wheeled:is_wheel_body": True,
                "wheeled:wheel_body_id": 0,
            },
        )
        non_wheel_body = builder.add_body(mass=1.0, label="chassis")
        wheel_shape = builder.add_shape_sphere(
            wheel_body,
            radius=0.2,
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": 0,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.2,
                "wheeled:wheel_width": 0.1,
            },
        )
        non_wheel_shape = builder.add_shape_box(non_wheel_body, hx=0.1, hy=0.1, hz=0.1)

        model = builder.finalize(device="cpu")

        self.assertEqual(model.wheeled.is_wheel.numpy()[wheel_shape], 1)
        self.assertEqual(model.wheeled.is_wheel.numpy()[non_wheel_shape], 0)
        self.assertEqual(model.wheeled.wheel_id.numpy()[wheel_shape], 0)
        self.assertEqual(model.wheeled.wheel_id.numpy()[non_wheel_shape], -1)
        self.assertEqual(model.wheeled.vehicle_id.numpy()[non_wheel_shape], -1)
        self.assertAlmostEqual(model.wheeled.wheel_radius.numpy()[wheel_shape], 0.2)
        self.assertAlmostEqual(model.wheeled.wheel_width.numpy()[wheel_shape], 0.1)
        self.assertEqual(model.wheeled.is_wheel_body.numpy()[wheel_body], 1)
        self.assertEqual(model.wheeled.is_wheel_body.numpy()[non_wheel_body], 0)
        self.assertEqual(model.wheeled.wheel_body_id.numpy()[wheel_body], 0)
        self.assertEqual(model.wheeled.wheel_body_id.numpy()[non_wheel_body], -1)


class TestWheeledManifestLoading(unittest.TestCase):
    def test_load_phase_0_manifest(self):
        assets = newton.wheeled.load_wheeled_manifest(_MANIFEST_PATH)
        by_name = {asset.name: asset for asset in assets}

        self.assertEqual(set(by_name), {"rc_car", "husky"})
        self.assertFalse(hasattr(by_name["rc_car"], "vehicle_type"))
        self.assertFalse(hasattr(by_name["rc_car"], "steering_joint_labels"))
        self.assertFalse(hasattr(by_name["rc_car"], "suspension_joint_labels"))
        self.assertEqual(len(by_name["rc_car"].wheel_body_labels), 4)
        self.assertEqual(len(by_name["rc_car"].wheel_shape_labels), 4)
        self.assertAlmostEqual(by_name["rc_car"].wheel_radius, 0.055)
        self.assertAlmostEqual(by_name["rc_car"].wheel_width, 0.045)
        self.assertAlmostEqual(by_name["husky"].wheel_radius, 0.1625)
        self.assertAlmostEqual(by_name["husky"].wheel_width, 0.13)

    def test_rejects_invalid_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            (tmp_dir / "asset.usda").write_text("#usda 1.0\n")

            manifest = {
                "version": 1,
                "namespace": "wheeled",
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
                newton.wheeled.load_wheeled_manifest(path)

            manifest["assets"][0]["wheel_shape_labels"] = ["/shape"]
            manifest["assets"].append(dict(manifest["assets"][0]))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate.*bad"):
                newton.wheeled.load_wheeled_manifest(path)

            manifest["assets"] = [manifest["assets"][0]]
            manifest["assets"][0]["file"] = "missing.usda"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bad.*file"):
                newton.wheeled.load_wheeled_manifest(path)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestWheeledManifestAnnotation(unittest.TestCase):
    def test_apply_manifest_metadata_to_imported_rc_car(self):
        asset = _asset_by_name("rc_car")
        builder = _load_fixture_builder(asset)

        wheels = newton.wheeled.apply_wheeled_manifest_metadata(builder, asset, vehicle_id=3)

        self.assertEqual(len(wheels), 4)
        self.assertEqual([wheel.wheel_id for wheel in wheels], [0, 1, 2, 3])
        self.assertEqual([wheel.vehicle_id for wheel in wheels], [3, 3, 3, 3])
        self.assertNotIn("wheeled:steering_joint_index", builder.custom_attributes)
        self.assertNotIn("wheeled:suspension_joint_index", builder.custom_attributes)

        model = builder.finalize(device="cpu")
        table = newton.wheeled.build_wheeled_metadata(model, wheels)
        self.assertEqual(table.wheel_count, 4)
        self.assertEqual(table.vehicle_count, 4)
        self.assertEqual(table.wheel_vehicle_ids, (3, 3, 3, 3))
        self.assertEqual(table.wheel_radius, (0.055, 0.055, 0.055, 0.055))
        self.assertEqual(table.wheel_width, (0.045, 0.045, 0.045, 0.045))

    def test_missing_manifest_label_raises(self):
        asset = _asset_by_name("husky")
        builder = _load_fixture_builder(asset)
        broken = replace(asset, wheel_shape_labels=("/missing/shape", *asset.wheel_shape_labels[1:]))

        with self.assertRaisesRegex(ValueError, "husky.*wheel shape.*missing"):
            newton.wheeled.apply_wheeled_manifest_metadata(builder, broken, vehicle_id=0)

    def test_apply_manifest_multiple_assets(self):
        rc_car = _asset_by_name("rc_car")
        husky = _asset_by_name("husky")
        builder = newton.ModelBuilder()
        newton.wheeled.register_wheeled_custom_attributes(builder)
        builder.add_usd(str(rc_car.file), schema_resolvers=[newton.usd.SchemaResolverPhysx()])
        builder.add_usd(str(husky.file), schema_resolvers=[newton.usd.SchemaResolverPhysx()])

        wheels = newton.wheeled.apply_wheeled_manifest(builder, _MANIFEST_PATH)
        model = builder.finalize(device="cpu")
        table = newton.wheeled.build_wheeled_metadata(model, wheels)

        self.assertEqual(table.wheel_count, 8)
        self.assertEqual(table.vehicle_count, 2)
        self.assertEqual(table.wheel_vehicle_ids, (0, 0, 0, 0, 1, 1, 1, 1))
        self.assertEqual(table.vehicle_wheel_counts, (4, 4))
        self.assertEqual(sorted(table.wheel_shape_indices), list(table.wheel_shape_indices))


class TestWheeledModelMetadata(unittest.TestCase):
    def test_builds_table_from_authored_model_attributes(self):
        builder = newton.ModelBuilder()
        newton.wheeled.register_wheeled_custom_attributes(builder)

        body = builder.add_body(
            mass=1.0,
            custom_attributes={"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": 0},
        )
        builder.add_shape_sphere(
            body,
            radius=0.1,
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": 0,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.1,
                "wheeled:wheel_width": 0.05,
            },
        )

        model = builder.finalize(device="cpu")
        wheels = newton.wheeled.read_wheeled_metadata(model)
        table = newton.wheeled.build_wheeled_metadata(model)

        self.assertEqual(len(wheels), 1)
        self.assertEqual(table.to_dict()["wheel_count"], 1)
        self.assertEqual(table.wheel_body_indices, (body,))
        np.testing.assert_allclose(table.wheel_radius, (0.1,))

    def test_rejects_inconsistent_authored_ids(self):
        builder = newton.ModelBuilder()
        newton.wheeled.register_wheeled_custom_attributes(builder)
        body = builder.add_body(
            mass=1.0,
            custom_attributes={"wheeled:is_wheel_body": True, "wheeled:wheel_body_id": 1},
        )
        builder.add_shape_sphere(
            body,
            radius=0.1,
            custom_attributes={
                "wheeled:is_wheel": True,
                "wheeled:wheel_id": 0,
                "wheeled:vehicle_id": 0,
                "wheeled:wheel_radius": 0.1,
                "wheeled:wheel_width": 0.05,
            },
        )

        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "wheel_body_id"):
            newton.wheeled.read_wheeled_metadata(model)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestWheeledAuthoredUsdMetadata(unittest.TestCase):
    def test_authored_usda_matches_runtime_annotation(self):
        authored_paths = {
            "rc_car": _TEST_ASSET_DIR / "rc_car_wheeled_attrs.usda",
            "husky": _TEST_ASSET_DIR / "husky_wheeled_attrs.usda",
        }
        for name, authored_path in authored_paths.items():
            with self.subTest(name=name):
                self.assertTrue(authored_path.exists(), authored_path)

                asset = _asset_by_name(name)
                runtime_builder = _load_fixture_builder(asset)
                runtime_wheels = newton.wheeled.apply_wheeled_manifest_metadata(runtime_builder, asset, vehicle_id=0)
                runtime_model = runtime_builder.finalize(device="cpu")
                runtime_table = newton.wheeled.build_wheeled_metadata(runtime_model, runtime_wheels)

                authored_builder = newton.ModelBuilder()
                newton.wheeled.register_wheeled_custom_attributes(authored_builder)
                authored_builder.add_usd(str(authored_path), schema_resolvers=[newton.usd.SchemaResolverPhysx()])
                authored_model = authored_builder.finalize(device="cpu")
                authored_table = newton.wheeled.build_wheeled_metadata(authored_model)

                self.assertEqual(authored_table.wheel_count, runtime_table.wheel_count)
                self.assertEqual(authored_table.wheel_vehicle_ids, runtime_table.wheel_vehicle_ids)
                np.testing.assert_allclose(authored_table.wheel_radius, runtime_table.wheel_radius)
                np.testing.assert_allclose(authored_table.wheel_width, runtime_table.wheel_width)


if __name__ == "__main__":
    unittest.main()
