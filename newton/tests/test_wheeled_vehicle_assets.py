# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path

from newton._src.utils.wheeled_asset_inspection import inspect_usd_asset, match_labels
from newton.tests.unittest_utils import USD_AVAILABLE

_ASSET_DIR = Path(__file__).resolve().parents[1] / "examples" / "assets" / "wheeled"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_REQUIRED_ASSET_NAMES = {"rc_car", "husky"}
_REQUIRED_ASSET_KEYS = {
    "name",
    "file",
    "vehicle_type",
    "description",
    "wheel_body_labels",
    "wheel_shape_labels",
    "suspension_joint_labels",
    "steering_joint_labels",
}


class TestWheeledVehicleAssetManifest(unittest.TestCase):
    def test_manifest_exists(self):
        self.assertTrue(_MANIFEST_PATH.exists(), f"Missing wheeled asset manifest: {_MANIFEST_PATH}")

    def test_manifest_schema_and_assets(self):
        manifest = json.loads(_MANIFEST_PATH.read_text())
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["namespace"], "wheeled")
        self.assertIsInstance(manifest["assets"], list)
        self.assertEqual({asset["name"] for asset in manifest["assets"]}, _REQUIRED_ASSET_NAMES)

        for asset in manifest["assets"]:
            self.assertTrue(_REQUIRED_ASSET_KEYS.issubset(asset), asset)
            self.assertTrue(asset["file"].endswith(".usda"), asset["file"])
            self.assertIn(asset["vehicle_type"], {"ackermann", "skid_steer"})
            self.assertIsInstance(asset["description"], str)
            self.assertGreater(len(asset["description"]), 0)
            for key in (
                "wheel_body_labels",
                "wheel_shape_labels",
                "suspension_joint_labels",
                "steering_joint_labels",
            ):
                self.assertIsInstance(asset[key], list)

    def test_manifest_files_exist(self):
        manifest = json.loads(_MANIFEST_PATH.read_text())
        for asset in manifest["assets"]:
            path = _ASSET_DIR / asset["file"]
            self.assertTrue(path.exists(), f"Missing asset file for {asset['name']}: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty asset file for {asset['name']}: {path}")


class TestWheeledVehicleAssetInspection(unittest.TestCase):
    def test_match_labels_is_case_insensitive(self):
        labels = ["/World/Robot/front_left_WHEEL", "/World/Robot/chassis", "/World/Robot/rear_tire"]
        self.assertEqual(
            match_labels(labels, ["wheel", "tire"]),
            ["/World/Robot/front_left_WHEEL", "/World/Robot/rear_tire"],
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_inspect_existing_usd_reports_counts(self):
        asset_path = Path(__file__).resolve().parent / "assets" / "cube_cylinder.usda"
        report = inspect_usd_asset(asset_path)

        self.assertEqual(report["path"], str(asset_path))
        self.assertGreater(report["body_count"], 0)
        self.assertGreater(report["shape_count"], 0)
        self.assertIn("body_labels", report)
        self.assertIn("joint_labels", report)
        self.assertIn("shape_labels", report)
        self.assertIn("candidate_wheel_body_labels", report)
        self.assertIn("candidate_wheel_shape_labels", report)


class TestWheeledVehicleInspectionScript(unittest.TestCase):
    def test_script_file_exists(self):
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_wheeled_assets.py"
        self.assertTrue(script_path.exists(), f"Missing inspection script: {script_path}")
