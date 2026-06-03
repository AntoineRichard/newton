# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wheeled-vehicle metadata and contact patch helpers."""

from ._src.wheeled.contact_patch import WheelContactPatchState, update_wheel_contact_patches
from ._src.wheeled.metadata import (
    WheeledAssetMetadata,
    WheeledModelMetadata,
    WheelMetadata,
    apply_wheeled_manifest,
    apply_wheeled_manifest_metadata,
    build_wheeled_metadata,
    load_wheeled_manifest,
    read_wheeled_metadata,
    register_wheeled_custom_attributes,
)

__all__ = [
    "WheelContactPatchState",
    "WheelMetadata",
    "WheeledAssetMetadata",
    "WheeledModelMetadata",
    "apply_wheeled_manifest",
    "apply_wheeled_manifest_metadata",
    "build_wheeled_metadata",
    "load_wheeled_manifest",
    "read_wheeled_metadata",
    "register_wheeled_custom_attributes",
    "update_wheel_contact_patches",
]
