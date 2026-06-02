# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path


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
