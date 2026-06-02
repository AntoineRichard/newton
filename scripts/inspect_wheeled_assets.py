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

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")

    markdown = format_markdown_report(reports)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)

    if args.output_json is None and args.output_md is None:
        print(markdown, end="")


if __name__ == "__main__":
    main()
