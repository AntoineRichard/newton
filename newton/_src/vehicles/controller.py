# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The cohesive :class:`WheeledVehicles` controller object."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from newton._src.sim import Model, ModelBuilder

from .contact import WheelContactPatch, latch_wheel_loads, update_wheel_contact_patches
from .metadata import (
    VehicleModelData,
    configure_wheel_solver_contacts,
    read_vehicle_model_data,
    register_vehicle_attributes,
)
from .vehicle import VehicleCommands, update_vehicle_controls
from .wheel import WheelDynamics, apply_wheel_dynamics


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

    All values are defaults applied uniformly at construction. Stiffnesses are
    model-dependent (see :class:`TireModel`).

    Args:
        tire_model: Default :class:`TireModel`.
        drive_input: Default :class:`DriveInput`.
        longitudinal_stiffness: Longitudinal slip stiffness per unit normal load [1/rad]
            (linear-regime slope is ``stiffness * Fz``; saturates near ``3*mu/stiffness`` slip).
        lateral_stiffness: Lateral slip stiffness per unit normal load [1/rad].
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
        apply_reaction_torque: Whether to apply the motor axle reaction torque to the wheel
            body. Off by default; it only affects pitch/weight-transfer, not the primary
            traction, and is left opt-in pending broader validation.
    """

    tire_model: int = int(TireModel.BRUSH)
    drive_input: int = int(DriveInput.SPEED)
    longitudinal_stiffness: float = 20.0
    lateral_stiffness: float = 20.0
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
    load_filter: float = 0.2
    apply_reaction_torque: bool = False


class WheeledVehicles:
    """Cohesive controller for wheeled vehicles over a finalized :class:`Model`.

    Owns flat device tables, per-wheel tire/spin state, and per-vehicle command
    buffers, preserving full heterogeneity (mixed Ackermann/skid-steer/generic
    vehicles in one model). The runtime methods execute as batched Warp kernels
    with no Python loops over wheels or vehicles.

    Typical per-substep use::

        vehicles.set_commands(drive=1.0, steer=0.2)  # per frame
        vehicles.update_controls(control)
        model.collide(state, contacts)
        vehicles.apply(state, contacts, dt)
        solver.step(state, next_state, control, contacts, dt)
        solver.update_contacts(contacts, state)
        vehicles.latch_loads(contacts)

    Args:
        model: Finalized model carrying ``vehicle:*`` custom attributes. Construct
            before ``model.contacts()`` so contact forces are available for load latching.
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
        device = self.data.device

        self.patch = WheelContactPatch(self.wheel_count, device=device)
        self.dynamics = WheelDynamics(self.wheel_count, device=device)
        self.commands = VehicleCommands(self.vehicle_count, device=device)
        self._init_params()

        # Ensure contact forces are allocated for load latching.
        model.request_contact_attributes("force")

    def _init_params(self) -> None:
        c = self.config

        def fill(arr, value):
            host = arr.numpy()
            host[...] = value
            arr.assign(host)

        d = self.dynamics
        fill(d.tire_model, int(c.tire_model))
        fill(d.drive_input, int(c.drive_input))
        fill(d.c_long, c.longitudinal_stiffness)
        fill(d.c_lat, c.lateral_stiffness)
        fill(d.mu_override, c.friction)
        fill(d.inertia, c.wheel_inertia)
        fill(d.damping, c.angular_damping)
        fill(d.rolling_resistance, c.rolling_resistance)
        fill(d.kp, c.motor_kp)
        fill(d.tau_max, c.motor_max_torque)
        fill(d.max_speed, c.max_wheel_speed)
        fill(d.brake_max, c.brake_max_torque)
        fill(d.fallback_load, c.fallback_normal_load)
        fill(d.min_ref, c.min_reference_speed)
        fill(d.apply_reaction, 1 if c.apply_reaction_torque else 0)

    @staticmethod
    def register_attributes(builder: ModelBuilder) -> None:
        """Register the ``vehicle:*`` custom attributes on ``builder``."""
        register_vehicle_attributes(builder)

    def configure_solver_contacts(self, *, condim: int = 1, priority: int = 1) -> None:
        """Make wheel-ground contacts normal-only (``condim=1``) so the tire model owns
        tangential force. Requires ``SolverMuJoCo.register_custom_attributes(builder)``
        before ``finalize``."""
        configure_wheel_solver_contacts(self.model, self.data, condim=condim, priority=priority)

    def set_commands(self, *, drive=None, steer=None, brake=None) -> None:
        """Set normalized per-vehicle commands.

        Each argument is a scalar (broadcast to all vehicles) or an array of
        length ``vehicle_count``. ``drive`` and ``steer`` are in [-1, 1];
        ``brake`` is in [0, 1].
        """
        for arr, value in ((self.commands.drive, drive), (self.commands.steer, steer), (self.commands.brake, brake)):
            if value is None:
                continue
            host = arr.numpy()
            host[...] = np.asarray(value, dtype=np.float32)
            arr.assign(host)

    def update_controls(self, control) -> None:
        """Map commands to per-wheel drive/brake targets and steering joint targets."""
        update_vehicle_controls(control, self.data, self.dynamics, self.commands)

    def apply(self, state, contacts, dt: float) -> None:
        """Extract patches, compute tire forces, accumulate into ``state.body_f``, and
        integrate analytical wheel spin. Call after ``model.collide`` and before
        ``solver.step``."""
        update_wheel_contact_patches(self.model, state, contacts, self.data, self.patch)
        apply_wheel_dynamics(self.model, state, self.data, self.patch, self.dynamics, dt)

    def latch_loads(self, contacts) -> None:
        """Latch solver-reported normal loads for the next step. Call after
        ``solver.update_contacts``."""
        latch_wheel_loads(self.model, contacts, self.data, self.patch, alpha=self.config.load_filter)
