# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, ClassVar

import warp as wp

from ...core.reset import normalize_reset_world_mask
from ...core.types import Devicelike
from ..actuator import _require_array
from .base import PowerSupplyBase

_SECONDS_PER_HOUR = wp.constant(3600.0)


@wp.func
def _open_circuit_voltage(state_of_charge: float, cell_voltage: wp.array[float]) -> float:
    """Interpolate the cell open-circuit voltage on its uniform state-of-charge grid."""
    n = cell_voltage.shape[0]
    x = wp.clamp(state_of_charge, 0.0, 1.0) * float(n - 1)
    i = wp.min(int(x), n - 2)
    t = x - float(i)
    return cell_voltage[i] * (1.0 - t) + cell_voltage[i + 1] * t


@wp.kernel
def _battery_step_kernel(
    actuator_start: wp.array[wp.int32],
    actuators: wp.array[wp.int32],
    cell_voltage: wp.array[float],
    series_cells: wp.array[float],
    capacity: wp.array[float],
    resistance: wp.array[float],
    polarization_resistance: wp.array[float],
    polarization_capacitance: wp.array[float],
    min_voltage: wp.array[float],
    max_charge_current: wp.array[float],
    dt: float,
    actuator_current: wp.array[float],
    state_of_charge_in: wp.array[float],
    polarization_voltage_in: wp.array[float],
    cutoff_in: wp.array[wp.int32],
    current: wp.array[float],
    state_of_charge_out: wp.array[float],
    polarization_voltage_out: wp.array[float],
    cutoff_out: wp.array[wp.int32],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    start = actuator_start[b]
    end = actuator_start[b + 1]
    for k in range(start, end):
        current[b] += actuator_current[actuators[k]]

    state_of_charge = state_of_charge_in[b]
    cutoff = cutoff_in[b]
    cell_current = current[b]
    if cutoff != 0:
        cell_current = 0.0
    elif cell_current < 0.0:
        # Regeneration charges the cells up to the charge limit and until full;
        # the rest is assumed dissipated outside the battery, e.g. in a brake resistor.
        cell_current = wp.max(cell_current, -max_charge_current[b])
        if state_of_charge >= 1.0:
            cell_current = 0.0

    time_constant = polarization_resistance[b] * polarization_capacitance[b]
    decay = float(0.0)
    if time_constant > 0.0:
        decay = wp.exp(-dt / time_constant)
    polarization_voltage = polarization_voltage_in[b] * decay + polarization_resistance[b] * cell_current * (
        1.0 - decay
    )
    state_of_charge = wp.clamp(
        state_of_charge - cell_current * dt / (_SECONDS_PER_HOUR * capacity[b]),
        0.0,
        1.0,
    )
    voltage = (
        series_cells[b] * _open_circuit_voltage(state_of_charge, cell_voltage)
        - resistance[b] * cell_current
        - polarization_voltage
    )
    if cutoff == 0:
        if voltage < min_voltage[b] or (state_of_charge <= 0.0 and cell_current > 0.0):
            cutoff = 1
    if cutoff != 0:
        voltage = 0.0

    state_of_charge_out[b] = state_of_charge
    polarization_voltage_out[b] = polarization_voltage
    cutoff_out[b] = cutoff
    for k in range(start, end):
        supply_voltage[actuators[k]] = voltage


@wp.kernel
def _battery_reset_kernel(
    actuator_start: wp.array[wp.int32],
    actuators: wp.array[wp.int32],
    supply_world: wp.array[wp.int32],
    world_count: int,
    cell_voltage: wp.array[float],
    series_cells: wp.array[float],
    initial_state_of_charge: wp.array[float],
    world_mask: wp.array[wp.bool],
    state_of_charge: wp.array[float],
    polarization_voltage: wp.array[float],
    cutoff: wp.array[wp.int32],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    world = supply_world[b]
    if world < 0:
        world = world_count
    if world_mask:
        if not world_mask[world]:
            return
    if state_of_charge:
        state_of_charge[b] = initial_state_of_charge[b]
        polarization_voltage[b] = 0.0
        cutoff[b] = 0
    if supply_voltage:
        voltage = series_cells[b] * _open_circuit_voltage(initial_state_of_charge[b], cell_voltage)
        for k in range(actuator_start[b], actuator_start[b + 1]):
            supply_voltage[actuators[k]] = voltage


class PowerSupplyBattery(PowerSupplyBase):
    """Stateful lithium-ion battery pack with a discharge curve, regeneration and a low-voltage cutoff.

    Each pack is :math:`n` cells in series, modeled by the first-order
    Thevenin equivalent circuit common in battery management systems: an
    open-circuit voltage that depends on the state of charge :math:`s`, an
    ohmic resistance :math:`R_0`, and one polarization branch
    :math:`R_1 \\parallel C_1` for the slower voltage relaxation under load.
    Over the signed sum :math:`I` of the currents the actuators draw, the
    terminal voltage is

    .. math::

        V = n\\, U_{\\text{oc}}(s) - R_0 I - V_1, \\qquad
        C_1 \\frac{dV_1}{dt} = I - \\frac{V_1}{R_1}, \\qquad
        \\frac{ds}{dt} = -\\frac{I}{Q}

    with :math:`Q` the capacity. :meth:`step` integrates :math:`V_1` exactly
    over a step, holding :math:`I` constant, and :math:`s` with forward Euler.

    :math:`U_{\\text{oc}}` interpolates :attr:`OCV_CELL_VOLTAGE`, the
    open-circuit voltage of a lithium-ion NMC cell, as in the 18650 and 21700
    packs of legged robots: 4.2 V full, a plateau around 3.6 to 3.9 V, and a
    steep knee below 10 % to 3.0 V empty. Override it in a subclass for
    another chemistry.

    **Regeneration.** A negative :math:`I` from regenerating motors charges
    the pack, raising :math:`s` and the terminal voltage. The pack accepts
    at most ``max_charge_current`` and nothing once full; the excess is
    assumed dissipated outside the pack, e.g. in a brake resistor, and does
    not change the voltage.

    **Cutoff.** Like the undervoltage protection of a battery management
    system, the pack disconnects once its terminal voltage drops below
    ``min_voltage``, or once it is empty while discharging. It then supplies
    0 V until :meth:`reset`, so a heavy load trips it before the pack is
    empty, as with a real pack.

    The state of the packs, :class:`State`, holds the state of charge, the
    polarization voltage and the cutoff flag, and is double-buffered like
    the state of an :class:`~newton.actuators.Actuator`. The supply and the
    motors form the same explicit feedback loop as with
    :class:`~newton.actuators.PowerSupplyDCSource`.
    """

    OCV_CELL_VOLTAGE: ClassVar[tuple[float, ...]] = (
        3.00, 3.35, 3.46, 3.52, 3.57, 3.60, 3.63, 3.66, 3.69, 3.72, 3.75,
        3.78, 3.81, 3.85, 3.89, 3.93, 3.98, 4.03, 4.08, 4.14, 4.20,
    )  # fmt: skip
    """Open-circuit voltage of one cell [V] at the states of charge 0, 0.05, ..., 1, evenly spaced."""

    CUTOFF_CELL_VOLTAGE: ClassVar[float] = 3.0
    """Default cutoff voltage of one cell [V], setting the default ``min_voltage``."""

    class State(PowerSupplyBase.State):
        """State of the battery packs of one :class:`PowerSupplyBattery`, one entry per pack."""

        def __init__(self, num_power_supplies: int, device: Devicelike, requires_grad: bool = False):
            self.state_of_charge = wp.zeros(num_power_supplies, dtype=float, device=device, requires_grad=requires_grad)
            """State of charge, from 0 (empty) to 1 (full), shape (B,)."""
            self.polarization_voltage = wp.zeros(
                num_power_supplies, dtype=float, device=device, requires_grad=requires_grad
            )
            """Voltage across the polarization branch [V], shape (B,)."""
            self.cutoff = wp.zeros(num_power_supplies, dtype=wp.int32, device=device)
            """Nonzero once the pack has disconnected, until :meth:`PowerSupplyBattery.reset`, shape (B,)."""

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments. ``series_cells``, the number of
                cells in series, and ``capacity`` [A·h] are required.
                ``resistance`` [Ω], ``polarization_resistance`` [Ω] and
                ``polarization_capacitance`` [F] of the pack default to 0.
                ``min_voltage`` [V] defaults to ``series_cells`` times
                :attr:`CUTOFF_CELL_VOLTAGE`, ``initial_state_of_charge`` to 1,
                and ``max_charge_current`` [A] to infinity, so that
                regeneration charges the pack without limit; 0 disables it.

        Returns:
            Complete arguments with defaults filled in.

        Raises:
            ValueError: A required argument is missing, an argument is
                unknown, *series_cells* or *capacity* is not positive,
                another argument is negative, or *initial_state_of_charge*
                is above 1.
        """
        known = {
            "series_cells",
            "capacity",
            "resistance",
            "polarization_resistance",
            "polarization_capacitance",
            "min_voltage",
            "initial_state_of_charge",
            "max_charge_current",
        }
        if unknown := set(args) - known:
            raise ValueError(f"{cls.__name__} got unknown argument(s) {sorted(unknown)}; expected {sorted(known)}")
        for name in ("series_cells", "capacity"):
            if name not in args:
                raise ValueError(f"{cls.__name__} requires '{name}' argument")
            if not args[name] > 0.0:
                raise ValueError(f"{name} must be positive, got {args[name]}")
        series_cells = float(args["series_cells"])
        resolved = {
            "series_cells": series_cells,
            "capacity": float(args["capacity"]),
            "resistance": float(args.get("resistance", 0.0)),
            "polarization_resistance": float(args.get("polarization_resistance", 0.0)),
            "polarization_capacitance": float(args.get("polarization_capacitance", 0.0)),
            "min_voltage": float(args.get("min_voltage", series_cells * cls.CUTOFF_CELL_VOLTAGE)),
            "initial_state_of_charge": float(args.get("initial_state_of_charge", 1.0)),
            "max_charge_current": float(args.get("max_charge_current", math.inf)),
        }
        for name, value in resolved.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if resolved["initial_state_of_charge"] > 1.0:
            raise ValueError(f"initial_state_of_charge must be at most 1, got {resolved['initial_state_of_charge']}")
        return resolved

    def __init__(
        self,
        series_cells: wp.array[float],
        capacity: wp.array[float],
        resistance: wp.array[float],
        polarization_resistance: wp.array[float],
        polarization_capacitance: wp.array[float],
        min_voltage: wp.array[float],
        initial_state_of_charge: wp.array[float],
        max_charge_current: wp.array[float],
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        *,
        world: wp.array[wp.int32] | None = None,
        world_count: int = 0,
    ):
        """Initialize the battery packs.

        Args:
            series_cells: Number of cells in series :math:`n`. Shape ``(B,)``.
            capacity: Capacity :math:`Q` [A·h]. Shape ``(B,)``.
            resistance: Ohmic resistance :math:`R_0` of the pack [Ω]. Shape ``(B,)``.
            polarization_resistance: Polarization resistance :math:`R_1` of
                the pack [Ω]. Shape ``(B,)``.
            polarization_capacitance: Polarization capacitance :math:`C_1` of
                the pack [F]. Shape ``(B,)``.
            min_voltage: Terminal voltage below which the pack disconnects [V].
                Shape ``(B,)``.
            initial_state_of_charge: State of charge after :meth:`reset`,
                from 0 to 1. Shape ``(B,)``.
            max_charge_current: Largest regenerative current the pack
                accepts [A]. Shape ``(B,)``.
            actuator_start: See :meth:`PowerSupplyBase.__init__`.
            actuators: See :meth:`PowerSupplyBase.__init__`.
            world: See :meth:`PowerSupplyBase.__init__`.
            world_count: See :meth:`PowerSupplyBase.__init__`.

        Raises:
            ValueError: An array does not have one entry per pack, or
                :attr:`OCV_CELL_VOLTAGE` has fewer than two entries.
        """
        super().__init__(actuator_start, actuators, world=world, world_count=world_count)
        self.series_cells = series_cells
        """Number of cells in series, shape (B,)."""
        self.capacity = capacity
        """Capacity [A·h], shape (B,)."""
        self.resistance = resistance
        """Ohmic resistance of the pack [Ω], shape (B,)."""
        self.polarization_resistance = polarization_resistance
        """Polarization resistance of the pack [Ω], shape (B,)."""
        self.polarization_capacitance = polarization_capacitance
        """Polarization capacitance of the pack [F], shape (B,)."""
        self.min_voltage = min_voltage
        """Cutoff voltage of the pack [V], shape (B,)."""
        self.initial_state_of_charge = initial_state_of_charge
        """State of charge after :meth:`reset`, shape (B,)."""
        self.max_charge_current = max_charge_current
        """Largest regenerative current the pack accepts [A], shape (B,)."""
        self._check_shapes(
            "series_cells",
            "capacity",
            "resistance",
            "polarization_resistance",
            "polarization_capacitance",
            "min_voltage",
            "initial_state_of_charge",
            "max_charge_current",
        )
        if len(self.OCV_CELL_VOLTAGE) < 2:
            raise ValueError("OCV_CELL_VOLTAGE must have at least two entries")
        self._cell_voltage = wp.array(self.OCV_CELL_VOLTAGE, dtype=float, device=self.device)
        self._requires_grad = any(
            getattr(self, name).requires_grad
            for name in (
                "series_cells",
                "capacity",
                "resistance",
                "polarization_resistance",
                "polarization_capacitance",
                "initial_state_of_charge",
            )
        )
        self._current = wp.zeros(
            self.num_power_supplies, dtype=float, device=self.device, requires_grad=self._requires_grad
        )

    def is_stateful(self) -> bool:
        """Return ``True``: the packs keep their charge, polarization and cutoff."""
        return True

    def state(self) -> PowerSupplyBattery.State:
        """Create a state with every pack at its initial state of charge and connected.

        Returns:
            A new state.
        """
        state = PowerSupplyBattery.State(self.num_power_supplies, self.device, requires_grad=self._requires_grad)
        self._launch_reset(state, None, None)
        return state

    def step(
        self,
        sim_control: Any,
        current_state: PowerSupplyBase.State | None = None,
        next_state: PowerSupplyBase.State | None = None,
        dt: float | None = None,
        *,
        next_control: Any | None = None,
    ) -> None:
        """Discharge or charge each pack by the current of its actuators and update their supply voltage.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_current``,
                or the written control no ``actuator_supply_voltage``.
            ValueError: One of these arrays is ``None``, *dt* is ``None``,
                or a state is missing.

        See :meth:`PowerSupplyBase.step` for the arguments.
        """
        self._check_step(current_state, next_state, dt)
        actuator_current = _require_array(sim_control, "actuator_supply_current")
        if self._current.requires_grad or actuator_current.requires_grad:
            current = wp.zeros_like(self._current, requires_grad=True)
        else:
            current = self._current
            current.zero_()
        if next_control is None:
            next_control = sim_control
        wp.launch(
            _battery_step_kernel,
            dim=self.num_power_supplies,
            inputs=[
                self.actuator_start,
                self.actuators,
                self._cell_voltage,
                self.series_cells,
                self.capacity,
                self.resistance,
                self.polarization_resistance,
                self.polarization_capacitance,
                self.min_voltage,
                self.max_charge_current,
                float(dt),
                actuator_current,
                current_state.state_of_charge,
                current_state.polarization_voltage,
                current_state.cutoff,
            ],
            outputs=[
                current,
                next_state.state_of_charge,
                next_state.polarization_voltage,
                next_state.cutoff,
                _require_array(next_control, "actuator_supply_voltage"),
            ],
            device=self.device,
        )

    def reset(
        self,
        sim_control: Any,
        state: PowerSupplyBase.State | None = None,
        *,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Recharge and reconnect the reset packs, and restore their open-circuit voltage.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_voltage``.
            ValueError: ``actuator_supply_voltage`` is ``None``, or
                *world_mask* does not have shape ``(world_count + 1,)``.

        See :meth:`PowerSupplyBase.reset` for the arguments.
        """
        world_mask = normalize_reset_world_mask(world_mask, world_count=self.world_count, device=self.device)
        self._launch_reset(state, _require_array(sim_control, "actuator_supply_voltage"), world_mask)

    def _launch_reset(
        self,
        state: PowerSupplyBattery.State | None,
        supply_voltage: wp.array[float] | None,
        world_mask: wp.array[wp.bool] | None,
    ) -> None:
        wp.launch(
            _battery_reset_kernel,
            dim=self.num_power_supplies,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.world,
                self.world_count,
                self._cell_voltage,
                self.series_cells,
                self.initial_state_of_charge,
                world_mask,
            ],
            outputs=[
                state.state_of_charge if state is not None else None,
                state.polarization_voltage if state is not None else None,
                state.cutoff if state is not None else None,
                supply_voltage,
            ],
            device=self.device,
        )
