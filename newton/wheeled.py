# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wheeled-vehicle metadata, contact patch, drive, and tire helpers."""

from ._src.wheeled.contact_patch import WheelContactPatchState, update_wheel_contact_patches
from ._src.wheeled.drive import (
    WheelDriveControl,
    WheelDriveState,
    apply_wheel_drive_forces,
    update_wheel_drive_normal_loads,
)
from ._src.wheeled.joints import configure_wheel_axle_joints
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
from ._src.wheeled.moment import (
    WheelMomentControl,
    WheelMomentState,
    configure_wheel_moment_control,
    update_wheel_moments,
)
from ._src.wheeled.mujoco import configure_mujoco_wheel_contacts
from ._src.wheeled.tire import (
    WheelTireControl,
    WheelTireState,
    apply_wheel_tire_forces,
    configure_wheel_tire_control,
    update_wheel_tire_normal_loads,
)
from ._src.wheeled.vehicle import (
    WheeledMotorConfig,
    WheeledSteeringConfig,
    WheeledVehicleControl,
    WheeledVehicleLayout,
    WheeledVehicleState,
    build_wheeled_vehicle_layout,
    configure_wheeled_vehicle_control,
    update_wheeled_vehicle_controls,
)

__all__ = [
    "WheelContactPatchState",
    "WheelDriveControl",
    "WheelDriveState",
    "WheelMetadata",
    "WheelMomentControl",
    "WheelMomentState",
    "WheelTireControl",
    "WheelTireState",
    "WheeledAssetMetadata",
    "WheeledModelMetadata",
    "WheeledMotorConfig",
    "WheeledSteeringConfig",
    "WheeledVehicleControl",
    "WheeledVehicleLayout",
    "WheeledVehicleState",
    "apply_wheel_drive_forces",
    "apply_wheel_tire_forces",
    "apply_wheeled_manifest",
    "apply_wheeled_manifest_metadata",
    "build_wheeled_metadata",
    "build_wheeled_vehicle_layout",
    "configure_mujoco_wheel_contacts",
    "configure_wheel_axle_joints",
    "configure_wheel_moment_control",
    "configure_wheel_tire_control",
    "configure_wheeled_vehicle_control",
    "load_wheeled_manifest",
    "read_wheeled_metadata",
    "register_wheeled_custom_attributes",
    "update_wheel_contact_patches",
    "update_wheel_drive_normal_loads",
    "update_wheel_moments",
    "update_wheel_tire_normal_loads",
    "update_wheeled_vehicle_controls",
]
