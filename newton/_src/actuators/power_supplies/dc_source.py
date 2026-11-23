# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.reset import normalize_reset_world_mask
from ..actuator import _require_array
from .base import PowerSupplyBase


@wp.kernel
def _dc_source_step_kernel(
    actuator_start: wp.array[wp.int32],
    actuators: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    resistance: wp.array[float],
    capacitance: wp.array[float],
    dt: float,
    actuator_current: wp.array[float],
    voltage_in: wp.array[float],
    current: wp.array[float],
    voltage_out: wp.array[float],
):
    b = wp.tid()
    start = actuator_start[b]
    end = actuator_start[b + 1]
    if start == end:
        return
    for k in range(start, end):
        current[b] += actuator_current[actuators[k]]
    steady_voltage = nominal_voltage[b] - resistance[b] * current[b]
    time_constant = resistance[b] * capacitance[b]
    decay = float(0.0)
    if time_constant > 0.0:
        decay = wp.exp(-dt / time_constant)
    # Every actuator of the supply holds the capacitor voltage; read it from the first.
    voltage = steady_voltage + (voltage_in[actuators[start]] - steady_voltage) * decay
    for k in range(start, end):
        voltage_out[actuators[k]] = voltage


@wp.kernel
def _dc_source_reset_kernel(
    actuator_start: wp.array[wp.int32],
    actuators: wp.array[wp.int32],
    supply_world: wp.array[wp.int32],
    world_count: int,
    nominal_voltage: wp.array[float],
    world_mask: wp.array[wp.bool],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    world = supply_world[b]
    if world < 0:
        world = world_count
    if world_mask:
        if not world_mask[world]:
            return
    for k in range(actuator_start[b], actuator_start[b + 1]):
        supply_voltage[actuators[k]] = nominal_voltage[b]


class PowerSupplyDCSource(PowerSupplyBase):
    """Linear DC source: an ideal voltage source behind a series resistance, with a capacitor across its output.

    The source of nominal voltage :math:`E` feeds the bus through the series
    resistance :math:`R`; the capacitor :math:`C`, e.g. the bus capacitance of
    the motor drivers, sits in parallel with the load. The bus voltage
    :math:`V` follows

    .. math::

        C \\frac{dV}{dt} = \\frac{E - V}{R} - I

    over the signed sum :math:`I` of the currents the actuators draw, e.g.
    :math:`d\\, I_{\\text{motor}}` for :class:`~newton.actuators.MotorDC`. A
    regenerating actuator offsets a driving one, and a net regenerative
    current raises :math:`V` above :math:`E`, as the source absorbs it.
    Each supply sums its actuators in a fixed order, so the voltage is
    deterministic.

    :meth:`step` integrates this equation exactly over a step of length
    :math:`\\Delta t`, holding :math:`I` constant:

    .. math::

        V_{k+1} = E - R I + (V_k - E + R I)\\, \\alpha,
        \\qquad \\alpha = e^{-\\Delta t / (R C)}

    With :math:`R C = 0`, :math:`\\alpha = 0` and the voltage is
    :math:`E - R I`, the constant-resistance supply of `BAM
    <https://github.com/Rhoban/bam>`_. The capacitor voltage :math:`V_k` is
    the ``actuator_supply_voltage`` on ``sim_control``, so the source keeps no
    state of its own and :meth:`is_stateful` returns ``False``.

    The supply and the motors form an explicit feedback loop. For
    :class:`~newton.actuators.MotorDC` behind a DUTY-configured
    :class:`~newton.actuators.DrivePD` and a
    :class:`~newton.actuators.ClampingCurrentDuty` without an active current
    limit, the load acts as the conductance
    :math:`G = \\sum d_{\\max}^2 / R_{\\text{motor}}` over the actuators
    sharing the supply, and the loop settles while
    :math:`(1 - \\alpha)(1 + R G) < 2`. The capacitor thus widens the stable
    range, which without it is :math:`R G < 1`. Above that bound, or while
    a current limit is active, the voltage may oscillate instead of
    converging.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments. ``nominal_voltage`` [V] is
                required; ``resistance`` [Ω] and ``capacitance`` [F] default
                to 0.

        Returns:
            Complete arguments with defaults filled in.

        Raises:
            ValueError: *nominal_voltage* is missing or negative,
                *resistance* or *capacitance* is negative, or an argument
                is unknown.
        """
        known = {"nominal_voltage", "resistance", "capacitance"}
        if unknown := set(args) - known:
            raise ValueError(f"{cls.__name__} got unknown argument(s) {sorted(unknown)}; expected {sorted(known)}")
        if "nominal_voltage" not in args:
            raise ValueError(f"{cls.__name__} requires 'nominal_voltage' argument")
        resolved = {
            "nominal_voltage": float(args["nominal_voltage"]),
            "resistance": float(args.get("resistance", 0.0)),
            "capacitance": float(args.get("capacitance", 0.0)),
        }
        for name, value in resolved.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        return resolved

    def __init__(
        self,
        nominal_voltage: wp.array[float],
        resistance: wp.array[float],
        capacitance: wp.array[float],
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        *,
        world: wp.array[wp.int32] | None = None,
        world_count: int = 0,
    ):
        """Initialize the DC sources.

        Args:
            nominal_voltage: Open-circuit voltage :math:`E` [V]. Shape ``(B,)``.
            resistance: Series resistance :math:`R` [Ω]. Shape ``(B,)``.
            capacitance: Output capacitance :math:`C` [F]. Shape ``(B,)``.
            actuator_start: See :meth:`PowerSupplyBase.__init__`.
            actuators: See :meth:`PowerSupplyBase.__init__`.
            world: See :meth:`PowerSupplyBase.__init__`.
            world_count: See :meth:`PowerSupplyBase.__init__`.

        Raises:
            ValueError: An array does not have one entry per supply.
        """
        super().__init__(actuator_start, actuators, world=world, world_count=world_count)
        self.nominal_voltage = nominal_voltage
        """Open-circuit voltage [V], shape (B,)."""
        self.resistance = resistance
        """Series resistance [Ω], shape (B,)."""
        self.capacitance = capacitance
        """Output capacitance [F], shape (B,)."""
        self._check_shapes("nominal_voltage", "resistance", "capacitance")
        self._current = wp.zeros(
            self.num_power_supplies,
            dtype=float,
            device=self.device,
            requires_grad=nominal_voltage.requires_grad or resistance.requires_grad or capacitance.requires_grad,
        )

    def step(
        self,
        sim_control: Any,
        current_state: PowerSupplyBase.State | None = None,
        next_state: PowerSupplyBase.State | None = None,
        dt: float | None = None,
        *,
        next_control: Any | None = None,
    ) -> None:
        """Move each bus voltage toward the source voltage under the current of its actuators.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_current``
                or ``actuator_supply_voltage``, or *next_control* no
                ``actuator_supply_voltage``.
            ValueError: One of these arrays is ``None``, or *dt* is ``None``.

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
            _dc_source_step_kernel,
            dim=self.num_power_supplies,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.nominal_voltage,
                self.resistance,
                self.capacitance,
                float(dt),
                actuator_current,
                _require_array(sim_control, "actuator_supply_voltage"),
            ],
            outputs=[current, _require_array(next_control, "actuator_supply_voltage")],
            device=self.device,
        )

    def reset(
        self,
        sim_control: Any,
        state: PowerSupplyBase.State | None = None,
        *,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Charge the capacitor of each reset supply to the nominal voltage.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_voltage``.
            ValueError: ``actuator_supply_voltage`` is ``None``, or
                *world_mask* does not have shape ``(world_count + 1,)``.

        See :meth:`PowerSupplyBase.reset` for the arguments.
        """
        world_mask = normalize_reset_world_mask(world_mask, world_count=self.world_count, device=self.device)
        wp.launch(
            _dc_source_reset_kernel,
            dim=self.num_power_supplies,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.world,
                self.world_count,
                self.nominal_voltage,
                world_mask,
            ],
            outputs=[_require_array(sim_control, "actuator_supply_voltage")],
            device=self.device,
        )
