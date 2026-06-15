# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal implementation of the Newton wheeled-vehicle layer.

Public symbols are re-exported from :mod:`newton.vehicles`; examples and docs
must import from there, never from ``newton._src``.
"""

from .controller import DriveInput, DriveMode, TireModel, WheeledConfig, WheeledVehicles
from .joints import configure_wheel_axle_joints
from .metadata import (
    VEHICLE_NAMESPACE,
    VehicleModelData,
    add_wheel,
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
