# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspection helpers for wheeled-vehicle reference assets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import newton
from newton.usd import SchemaResolverPhysx

_DEFAULT_WHEEL_PATTERNS = ("wheel", "tire", "tyre")
_DEFAULT_SUSPENSION_PATTERNS = ("suspension", "spring", "shock", "damper", "strut")
_DEFAULT_STEERING_PATTERNS = ("steer", "steering")


def match_labels(labels: list[str], patterns: list[str] | tuple[str, ...]) -> list[str]:
    """Return labels containing any case-insensitive pattern."""
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [label for label in labels if any(pattern.search(label) for pattern in compiled)]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


def _stringify_mapping(mapping: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in mapping.items()}


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
                "q_start": builder.joint_q_start[joint_index],
                "qd_start": builder.joint_qd_start[joint_index],
            }
        )
    return rows


def inspect_usd_asset(asset_path: str | Path) -> dict[str, Any]:
    """Load a USD asset through ModelBuilder and return structural metadata."""
    path = Path(asset_path)
    builder = newton.ModelBuilder()
    result = builder.add_usd(
        str(path),
        floating=False,
        enable_self_collisions=False,
        schema_resolvers=[SchemaResolverPhysx()],
    )

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
        "path_body_map": _stringify_mapping(result.get("path_body_map", {})),
        "path_joint_map": _stringify_mapping(result.get("path_joint_map", {})),
        "path_shape_map": _stringify_mapping(result.get("path_shape_map", {})),
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
