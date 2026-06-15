# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wheeled-vehicle simulation layer.

A cohesive :class:`WheeledVehicles` controller wraps a rigid solver (MuJoCo Warp
first): the solver owns collision and normal support while this layer owns the
analytical wheel spin and a brush combined-slip tire model, supporting
heterogeneous vehicles (Ackermann, skid-steer, generic) in a single model.
"""

from ._src.vehicles import (
    VEHICLE_NAMESPACE,
    DriveInput,
    DriveMode,
    TireModel,
    VehicleModelData,
    WheeledConfig,
    WheeledVehicles,
    add_wheel,
    configure_wheel_axle_joints,
    configure_wheel_solver_contacts,
    read_vehicle_model_data,
    register_vehicle_attributes,
    set_vehicle,
)

__all__ = [
    "VEHICLE_NAMESPACE",
    "DriveInput",
    "DriveMode",
    "TireModel",
    "VehicleModelData",
    "WheeledConfig",
    "WheeledVehicles",
    "add_wheel",
    "configure_wheel_axle_joints",
    "configure_wheel_solver_contacts",
    "read_vehicle_model_data",
    "register_vehicle_attributes",
    "set_vehicle",
]
