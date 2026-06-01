# Wheeled Vehicle Phase 0 Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the reference wheeled-vehicle asset intake path, inspection tooling, tests, and asset report needed before implementing the wheeled solver wrapper.

**Architecture:** Phase 0 stays out of solver runtime code. It adds canonical asset locations, a manifest that records the reference vehicles and identified labels, an internal inspection helper plus CLI, and tests that prove the assets load through `ModelBuilder.add_usd()` and expose enough labels for Phase 1 wheel metadata mapping.

**Tech Stack:** Python stdlib, `unittest`, Newton `ModelBuilder.add_usd()`, Warp-backed Newton model finalization, optional USD support guarded by existing `USD_AVAILABLE` test utility.

---

## Scope

Phase 0 produces asset intake artifacts only. It does not add `SolverWheeledVehicle`, wheel contact kernels, tire models, drive modes, raycasts, examples, or public API symbols.

The implementation requires two user-provided USD files. Store them under canonical names inside the repo:

- `newton/examples/assets/wheeled/rc_car.usda`
- `newton/examples/assets/wheeled/husky.usda`

If the files are not available when execution reaches the asset-copy step, stop at that step and ask the user to provide them. Do not substitute unrelated assets.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/examples/assets/wheeled/rc_car.usda` | Create | Ackermann RC-car reference asset supplied by the user |
| `newton/examples/assets/wheeled/husky.usda` | Create | Skid-steer AGV reference asset supplied by the user |
| `newton/examples/assets/wheeled/manifest.json` | Create | Canonical reference asset manifest and identified Phase 1 labels |
| `newton/_src/utils/wheeled_asset_inspection.py` | Create | Internal asset inspection helpers used by tests and script |
| `scripts/inspect_wheeled_assets.py` | Create | CLI that loads the manifest assets and emits JSON/Markdown reports |
| `newton/tests/test_wheeled_vehicle_assets.py` | Create | Manifest, inspection-helper, and reference-asset load tests |
| `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md` | Create | Human-readable asset inspection report and metadata-gap log |
| `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md` | Modify | Link the Phase 0 plan and report |

## Task 1: Manifest Contract

**Files:**
- Create: `newton/tests/test_wheeled_vehicle_assets.py`
- Create: `newton/examples/assets/wheeled/manifest.json`

- [ ] **Step 1: Write the failing manifest tests**

Create `newton/tests/test_wheeled_vehicle_assets.py` with this content:

```python
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
```

- [ ] **Step 2: Run the manifest tests to verify they fail**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: failure from `test_manifest_exists` because `newton/examples/assets/wheeled/manifest.json` does not exist.

- [ ] **Step 3: Add the asset directory, supplied assets, and initial manifest**

Create `newton/examples/assets/wheeled/`. Copy the user-provided assets into these exact paths:

```text
newton/examples/assets/wheeled/rc_car.usda
newton/examples/assets/wheeled/husky.usda
```

Create `newton/examples/assets/wheeled/manifest.json` with this content:

```json
{
  "version": 1,
  "namespace": "wheeled",
  "assets": [
    {
      "name": "rc_car",
      "file": "rc_car.usda",
      "vehicle_type": "ackermann",
      "description": "Reference Ackermann RC car asset for wheeled-vehicle solver development.",
      "wheel_body_labels": [],
      "wheel_shape_labels": [],
      "suspension_joint_labels": [],
      "steering_joint_labels": []
    },
    {
      "name": "husky",
      "file": "husky.usda",
      "vehicle_type": "skid_steer",
      "description": "Reference skid-steer AGV asset for wheeled-vehicle solver development.",
      "wheel_body_labels": [],
      "wheel_shape_labels": [],
      "suspension_joint_labels": [],
      "steering_joint_labels": []
    }
  ]
}
```

- [ ] **Step 4: Run the manifest tests to verify they pass**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: the three manifest tests pass. USD-dependent tests added in later tasks do not exist yet.

- [ ] **Step 5: Commit manifest and reference asset intake**

Run:

```bash
git add newton/tests/test_wheeled_vehicle_assets.py \
  newton/examples/assets/wheeled/manifest.json \
  newton/examples/assets/wheeled/rc_car.usda \
  newton/examples/assets/wheeled/husky.usda
git commit -m "Add wheeled vehicle reference assets"
```

Commit body:

```text
Add canonical reference asset locations and a manifest for the Ackermann RC car
and skid-steer AGV assets. These files are the Phase 0 inputs for inspecting
wheel, suspension, and steering labels before solver implementation.
```

## Task 2: Asset Inspection Helper

**Files:**
- Create: `newton/_src/utils/wheeled_asset_inspection.py`
- Modify: `newton/tests/test_wheeled_vehicle_assets.py`

- [ ] **Step 1: Add failing tests for inspection helpers**

Append this code to `newton/tests/test_wheeled_vehicle_assets.py`:

```python
from newton.tests.unittest_utils import USD_AVAILABLE


class TestWheeledVehicleAssetInspection(unittest.TestCase):
    def test_match_labels_is_case_insensitive(self):
        from newton._src.utils.wheeled_asset_inspection import match_labels

        labels = ["/World/Robot/front_left_WHEEL", "/World/Robot/chassis", "/World/Robot/rear_tire"]
        self.assertEqual(
            match_labels(labels, ["wheel", "tire"]),
            ["/World/Robot/front_left_WHEEL", "/World/Robot/rear_tire"],
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_inspect_existing_usd_reports_counts(self):
        from newton._src.utils.wheeled_asset_inspection import inspect_usd_asset

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
```

- [ ] **Step 2: Run the inspection tests to verify they fail**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: failure with `ModuleNotFoundError: No module named 'newton._src.utils.wheeled_asset_inspection'`.

- [ ] **Step 3: Implement the inspection helper**

Create `newton/_src/utils/wheeled_asset_inspection.py` with this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspection helpers for wheeled-vehicle reference assets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import newton

_DEFAULT_WHEEL_PATTERNS = ("wheel", "tire", "tyre")
_DEFAULT_SUSPENSION_PATTERNS = ("suspension", "spring", "shock", "damper", "strut")
_DEFAULT_STEERING_PATTERNS = ("steer", "steering")


def match_labels(labels: list[str], patterns: list[str] | tuple[str, ...]) -> list[str]:
    """Return labels containing any case-insensitive pattern."""
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [label for label in labels if any(pattern.search(label) for pattern in compiled)]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


def _joint_summary(builder: newton.ModelBuilder) -> list[dict[str, Any]]:
    rows = []
    for joint_index, label in enumerate(builder.joint_label):
        parent = builder.joint_parent[joint_index]
        child = builder.joint_child[joint_index]
        rows.append(
            {
                "index": joint_index,
                "label": label,
                "type": int(builder.joint_type[joint_index]),
                "parent": "world" if parent < 0 else builder.body_label[parent],
                "child": builder.body_label[child],
            }
        )
    return rows


def inspect_usd_asset(asset_path: str | Path) -> dict[str, Any]:
    """Load a USD asset through ModelBuilder and return structural metadata."""
    path = Path(asset_path)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    result = builder.add_usd(path)

    body_labels = list(builder.body_label)
    joint_labels = list(builder.joint_label)
    shape_labels = list(builder.shape_label)

    return {
        "path": str(path),
        "body_count": builder.body_count,
        "joint_count": builder.joint_count,
        "shape_count": builder.shape_count,
        "body_labels": body_labels,
        "joint_labels": joint_labels,
        "shape_labels": shape_labels,
        "joints": _joint_summary(builder),
        "candidate_wheel_body_labels": match_labels(body_labels, _DEFAULT_WHEEL_PATTERNS),
        "candidate_wheel_shape_labels": match_labels(shape_labels, _DEFAULT_WHEEL_PATTERNS),
        "candidate_suspension_joint_labels": match_labels(joint_labels, _DEFAULT_SUSPENSION_PATTERNS),
        "candidate_steering_joint_labels": match_labels(joint_labels, _DEFAULT_STEERING_PATTERNS),
        "path_body_map": dict(result.get("path_body_map", {})),
        "path_joint_map": dict(result.get("path_joint_map", {})),
        "path_shape_map": dict(result.get("path_shape_map", {})),
    }


def inspect_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Inspect every asset listed by a wheeled reference asset manifest."""
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    asset_dir = path.parent
    reports = []
    for asset in manifest["assets"]:
        report = inspect_usd_asset(asset_dir / asset["file"])
        report["name"] = asset["name"]
        report["vehicle_type"] = asset["vehicle_type"]
        report["manifest"] = asset
        reports.append(report)
    return reports


def format_markdown_report(reports: list[dict[str, Any]]) -> str:
    """Format inspection results as a Markdown report."""
    lines = ["# Wheeled Vehicle Phase 0 Asset Report", ""]
    for report in reports:
        lines.extend(
            [
                f"## {report['name']} ({report['vehicle_type']})",
                "",
                f"- Path: `{report['path']}`",
                f"- Bodies: {report['body_count']}",
                f"- Joints: {report['joint_count']}",
                f"- Shapes: {report['shape_count']}",
                "",
                "### Candidate Labels",
                "",
                f"- Wheel bodies: {report['candidate_wheel_body_labels']}",
                f"- Wheel shapes: {report['candidate_wheel_shape_labels']}",
                f"- Suspension joints: {report['candidate_suspension_joint_labels']}",
                f"- Steering joints: {report['candidate_steering_joint_labels']}",
                "",
                "### Manifest Labels",
                "",
                f"- Wheel bodies: {report['manifest']['wheel_body_labels']}",
                f"- Wheel shapes: {report['manifest']['wheel_shape_labels']}",
                f"- Suspension joints: {report['manifest']['suspension_joint_labels']}",
                f"- Steering joints: {report['manifest']['steering_joint_labels']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run the inspection tests to verify they pass**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: manifest and inspection tests pass. USD-dependent tests may be skipped only when `usd-core` is unavailable in the environment.

- [ ] **Step 5: Commit the inspection helper**

Run:

```bash
git add newton/_src/utils/wheeled_asset_inspection.py newton/tests/test_wheeled_vehicle_assets.py
git commit -m "Add wheeled asset inspection helper"
```

Commit body:

```text
Add an internal helper for loading USD reference assets through ModelBuilder and
summarizing labels, counts, and candidate wheel-related prims. Tests cover the
pure label matcher and a known USD fixture.
```

## Task 3: Inspection CLI

**Files:**
- Create: `scripts/inspect_wheeled_assets.py`
- Modify: `newton/tests/test_wheeled_vehicle_assets.py`

- [ ] **Step 1: Add a failing CLI smoke test**

Append this code to `newton/tests/test_wheeled_vehicle_assets.py`:

```python
class TestWheeledVehicleInspectionScript(unittest.TestCase):
    def test_script_file_exists(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_wheeled_assets.py"
        self.assertTrue(script_path.exists(), f"Missing inspection script: {script_path}")
```

- [ ] **Step 2: Run the script smoke test to verify it fails**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: failure from `test_script_file_exists` because `scripts/inspect_wheeled_assets.py` does not exist.

- [ ] **Step 3: Implement the CLI script**

Create `scripts/inspect_wheeled_assets.py` with this content:

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect wheeled-vehicle reference assets listed in a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newton._src.utils.wheeled_asset_inspection import format_markdown_report, inspect_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("newton/examples/assets/wheeled/manifest.json"),
        help="Path to the wheeled reference asset manifest.",
    )
    parser.add_argument("--output-json", type=Path, help="Write raw inspection data to this JSON path.")
    parser.add_argument("--output-md", type=Path, help="Write a Markdown inspection report to this path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = inspect_manifest(args.manifest)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(format_markdown_report(reports))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script smoke test to verify it passes**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: all tests in `test_wheeled_vehicle_assets` pass, with USD-dependent tests skipped only when `usd-core` is unavailable.

- [ ] **Step 5: Run the CLI against the reference manifest**

Run:

```bash
uv run --extra dev scripts/inspect_wheeled_assets.py \
  --manifest newton/examples/assets/wheeled/manifest.json \
  --output-json /tmp/wheeled_vehicle_phase_0_assets.json \
  --output-md /tmp/wheeled_vehicle_phase_0_assets.md
```

Expected: command exits with status 0 and writes both `/tmp/wheeled_vehicle_phase_0_assets.json` and `/tmp/wheeled_vehicle_phase_0_assets.md`.

- [ ] **Step 6: Commit the CLI**

Run:

```bash
git add scripts/inspect_wheeled_assets.py newton/tests/test_wheeled_vehicle_assets.py
git commit -m "Add wheeled asset inspection script"
```

Commit body:

```text
Add a small CLI around the wheeled asset inspection helper so Phase 0 can produce
repeatable JSON and Markdown reports from the reference asset manifest.
```

## Task 4: Reference Asset Label Identification

**Files:**
- Modify: `newton/examples/assets/wheeled/manifest.json`
- Modify: `newton/tests/test_wheeled_vehicle_assets.py`

- [ ] **Step 1: Add failing tests that require identified labels**

Append this code to `newton/tests/test_wheeled_vehicle_assets.py`:

```python
class TestWheeledVehicleReferenceAssetLabels(unittest.TestCase):
    def test_manifest_identifies_required_phase_1_labels(self):
        manifest = json.loads(_MANIFEST_PATH.read_text())
        assets = {asset["name"]: asset for asset in manifest["assets"]}

        rc_car = assets["rc_car"]
        self.assertGreaterEqual(len(rc_car["wheel_body_labels"]), 4)
        self.assertGreaterEqual(len(rc_car["wheel_shape_labels"]), 4)
        self.assertGreaterEqual(len(rc_car["steering_joint_labels"]), 1)

        husky = assets["husky"]
        self.assertGreaterEqual(len(husky["wheel_body_labels"]), 4)
        self.assertGreaterEqual(len(husky["wheel_shape_labels"]), 4)
        self.assertEqual(husky["steering_joint_labels"], [])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_manifest_labels_exist_in_loaded_assets(self):
        from newton._src.utils.wheeled_asset_inspection import inspect_manifest

        reports = {report["name"]: report for report in inspect_manifest(_MANIFEST_PATH)}
        for name, report in reports.items():
            labels = set(report["body_labels"] + report["shape_labels"] + report["joint_labels"])
            manifest = report["manifest"]
            for key in (
                "wheel_body_labels",
                "wheel_shape_labels",
                "suspension_joint_labels",
                "steering_joint_labels",
            ):
                missing = sorted(set(manifest[key]).difference(labels))
                self.assertEqual(missing, [], f"{name} manifest has labels not found in loaded asset for {key}")
```

- [ ] **Step 2: Run the label tests to verify they fail**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: failure from `test_manifest_identifies_required_phase_1_labels` because the manifest label lists are still empty.

- [ ] **Step 3: Generate inspection output for label selection**

Run:

```bash
uv run --extra dev scripts/inspect_wheeled_assets.py \
  --manifest newton/examples/assets/wheeled/manifest.json \
  --output-json /tmp/wheeled_vehicle_phase_0_assets.json \
  --output-md /tmp/wheeled_vehicle_phase_0_assets.md
```

Open `/tmp/wheeled_vehicle_phase_0_assets.json` and select the exact body, shape, suspension-joint, and steering-joint labels that Phase 1 should use. Use labels from `body_labels`, `shape_labels`, and `joint_labels` only.

- [ ] **Step 4: Update the manifest with identified labels**

Edit `newton/examples/assets/wheeled/manifest.json` so each asset's label lists contain exact strings copied from the inspection JSON:

- `wheel_body_labels` entries must come from that asset's `body_labels` array.
- `wheel_shape_labels` entries must come from that asset's `shape_labels` array.
- `suspension_joint_labels` entries must come from that asset's `joint_labels` array.
- `steering_joint_labels` entries must come from that asset's `joint_labels` array.

For the Ackermann RC car, record at least four wheel body labels, at least four wheel shape labels, and at least one steering joint label. For Husky, record at least four wheel body labels and at least four wheel shape labels; keep `steering_joint_labels` empty unless the asset contains explicit steer joints. Use more than four wheel labels if either asset has additional wheels.

- [ ] **Step 5: Run the label tests to verify they pass**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
```

Expected: all tests in `test_wheeled_vehicle_assets` pass, with USD-dependent tests skipped only when `usd-core` is unavailable.

- [ ] **Step 6: Commit identified labels**

Run:

```bash
git add newton/examples/assets/wheeled/manifest.json newton/tests/test_wheeled_vehicle_assets.py
git commit -m "Identify wheeled reference asset labels"
```

Commit body:

```text
Record the wheel, suspension, and steering labels discovered from the reference
assets. These labels define the static asset mapping that Phase 1 uses when
adding wheeled custom attributes.
```

## Task 5: Phase 0 Report And Roadmap Link

**Files:**
- Create: `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`
- Modify: `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`

- [ ] **Step 1: Generate the report from the manifest**

Run:

```bash
uv run --extra dev scripts/inspect_wheeled_assets.py \
  --manifest newton/examples/assets/wheeled/manifest.json \
  --output-json /tmp/wheeled_vehicle_phase_0_assets.json \
  --output-md docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md
```

Expected: command exits with status 0 and writes `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`.

- [ ] **Step 2: Add report notes for Phase 1**

Append this section to `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`:

```markdown
## Phase 1 Metadata Decisions

- Mark every `wheel_shape_labels` entry with `wheeled:is_wheel = 1`.
- Use `wheel_shape_labels` as the source of explicit `wheeled:wheel_radius` values when shape inference is ambiguous.
- Use `wheel_body_labels` as the receiving bodies for wheel support forces.
- Keep Husky `steering_joint_labels` empty for skid-steer control.
- Treat missing suspension labels as an asset limitation, not a Phase 1 blocker, because the first plane-contact wrapper can use shape/body metadata only.
```

- [ ] **Step 3: Link the report from the roadmap**

In `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`, keep the existing Phase 0 plan link and add this report link immediately below it:

```markdown
Report: `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`
```

- [ ] **Step 4: Verify docs and tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
rg -n "2026-06-01-wheeled-vehicle-phase-0-assets" docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md
rg -n "Phase 1 Metadata Decisions" docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md
```

Expected:

- `newton.tests` reports all `test_wheeled_vehicle_assets` tests passing, with USD-dependent tests skipped only when `usd-core` is unavailable.
- The first `rg` command prints the plan and report links.
- The second `rg` command prints the report section heading.

- [ ] **Step 5: Commit the report and roadmap update**

Run:

```bash
git add docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md \
  docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md
git commit -m "Document wheeled phase 0 asset findings"
```

Commit body:

```text
Add the generated reference asset inspection report and link the Phase 0 plan
and report from the wheeled vehicle roadmap. The report records the labels and
metadata decisions needed before Phase 1 solver work.
```

## Final Verification

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
uvx pre-commit run -a
```

Expected:

- The focused asset tests pass, with USD-dependent tests skipped only when `usd-core` is unavailable.
- `pre-commit` exits with status 0.

## Handoff Notes

- If `uvx pre-commit run -a` changes formatting, review the diff and commit those formatting changes with the related task commit if they affect only Phase 0 files.
- Do not implement solver kernels or public solver exports while executing this plan.
- If either reference asset cannot be loaded by `ModelBuilder.add_usd()`, stop and record the import error in the final response before changing solver design.
