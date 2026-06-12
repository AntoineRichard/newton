# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The cohesive :class:`WheeledVehicles` controller object."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from newton._src.sim import Model, ModelBuilder

from .metadata import VehicleModelData, read_vehicle_model_data, register_vehicle_attributes


class DriveMode(IntEnum):
    """Per-vehicle command-mapping topology."""

    GENERIC = 0
    """Each driven wheel pulls the drive command directly; steerable wheels steer."""
    ACKERMANN = 1
    """Front-steer car: drive command to driven wheels, steer command to Ackermann angles."""
    SKID_STEER = 2
    """Differential drive: steer command biases left/right wheel speeds, no steering joints."""


class TireModel(IntEnum):
    """Selectable per-wheel tire-force model."""

    BRUSH = 0
    """Elastic-bristle brush model with intrinsic combined-slip saturation (default)."""
    LINEAR = 1
    """Linear slip-to-force with a friction-circle clip."""


class DriveInput(IntEnum):
    """How a normalized drive command is realized."""

    SPEED = 0
    """Command maps to a target wheel speed via a torque-limited servo."""
    TORQUE = 1
    """Command maps directly to motor torque."""


@dataclass
class WheeledConfig:
    """Default per-wheel parameters and integration options for a vehicle layer.

    All values are defaults applied uniformly at construction; later phases allow
    per-wheel overrides. Stiffnesses are model-dependent (see :class:`TireModel`).

    Args:
        tire_model: Default :class:`TireModel`.
        drive_input: Default :class:`DriveInput`.
        longitudinal_stiffness: Brush longitudinal slip stiffness [N] (force at unit slip).
        lateral_stiffness: Brush lateral slip stiffness [N].
        friction: Default tire friction coefficient; <0 uses the contact material seed.
        wheel_inertia: Wheel rotational inertia about the axle [kg·m²].
        angular_damping: Wheel angular damping [N·m·s/rad].
        rolling_resistance: Rolling-resistance torque magnitude [N·m].
        motor_kp: Speed-servo proportional gain [N·m·s/rad].
        motor_max_torque: Motor torque clamp [N·m].
        max_wheel_speed: Wheel angular speed at unit drive command [rad/s].
        brake_max_torque: Brake torque at unit brake command [N·m].
        fallback_normal_load: Normal load used before solver forces are latched [N].
        min_reference_speed: Speed floor for slip regularization [m/s].
        apply_reaction_torque: Whether to apply the motor axle reaction torque to the wheel body.
    """

    tire_model: int = int(TireModel.BRUSH)
    drive_input: int = int(DriveInput.SPEED)
    longitudinal_stiffness: float = 2.0e4
    lateral_stiffness: float = 2.0e4
    friction: float = -1.0
    wheel_inertia: float = 0.01
    angular_damping: float = 0.01
    rolling_resistance: float = 0.0
    motor_kp: float = 5.0
    motor_max_torque: float = 20.0
    max_wheel_speed: float = 60.0
    brake_max_torque: float = 20.0
    fallback_normal_load: float = 0.0
    min_reference_speed: float = 0.5
    apply_reaction_torque: bool = True


class WheeledVehicles:
    """Cohesive controller for wheeled vehicles over a finalized :class:`Model`.

    Owns flat device tables and per-wheel tire/spin state, preserving full
    heterogeneity (mixed Ackermann/skid-steer/generic vehicles in one model). The
    runtime methods (:meth:`set_commands`, :meth:`update_controls`, :meth:`apply`,
    :meth:`latch_loads`) execute as batched Warp kernels with no Python loops over
    wheels or vehicles.

    Args:
        model: Finalized model carrying ``vehicle:*`` custom attributes.
        config: Default parameters; uses :class:`WheeledConfig` defaults if omitted.
    """

    DriveMode = DriveMode
    TireModel = TireModel
    DriveInput = DriveInput
    Config = WheeledConfig

    def __init__(self, model: Model, *, config: WheeledConfig | None = None):
        self.model = model
        self.config = config if config is not None else WheeledConfig()
        self.data: VehicleModelData = read_vehicle_model_data(model)
        self.wheel_count = self.data.wheel_count
        self.vehicle_count = self.data.vehicle_count

    @staticmethod
    def register_attributes(builder: ModelBuilder) -> None:
        """Register the ``vehicle:*`` custom attributes on ``builder``."""
        register_vehicle_attributes(builder)
