# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp


class PowerSupplyBase:
    """Base class for the power supplies of actuators.

    A power supply powers the motors of one or more actuators (see
    :class:`~newton.actuators.MotorDC`).  One object holds several power
    supplies of the same type: its parameter arrays have one entry per
    supply.  :meth:`step` and :meth:`reset` read and write these arrays on
    ``sim_control``, with one entry per actuator of the model, indexed like
    :attr:`Actuator.actuator_indices <newton.actuators.Actuator.actuator_indices>`:

    * input ``actuator_supply_current``: the current each motor draws from
      its supply [A], written by the motor; negative while the motor
      regenerates;
    * output ``actuator_supply_voltage``: the supply voltage of each motor
      [V], read by the motor.

    The motors and the supplies form an explicit loop: the motors read the
    voltage of the previous step.  With one ``Control`` per step, e.g. to
    differentiate several steps on one tape, pass the next step's ``Control``
    as *next_control* to :meth:`step`.

    A stateful supply (see :meth:`is_stateful`) keeps its internal state,
    e.g. the charge of a battery, in :class:`State` objects created with
    :meth:`state`, double-buffered like the state of an
    :class:`~newton.actuators.Actuator`.

    Subclasses must override :meth:`resolve_arguments`, :meth:`step` and
    :meth:`reset`, and stateful subclasses also :meth:`is_stateful` and
    :meth:`state`.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values before they are batched into Warp arrays.
    ``__init__`` receives pre-built arrays and validates shapes only —
    reading back array contents would force a synchronous device-to-host
    copy on every construction.
    """

    class State:
        """Internal state of the power supplies of one object, one entry per supply.

        Subclasses hold their state in :class:`warp.array` attributes, which
        :meth:`assign` copies.
        """

        def assign(self, other: PowerSupplyBase.State) -> None:
            """Copy the state held by *other* into this one.

            Use it in place of the final swap of an odd-length CUDA graph
            region, as with :meth:`Actuator.State.assign <newton.actuators.Actuator.State.assign>`.

            Args:
                other: State of the same type to copy from.

            Raises:
                TypeError: *other* is not of the same type.
            """
            if type(other) is not type(self):
                raise TypeError(f"Cannot assign {type(other).__name__} to {type(self).__name__}")
            for name, value in vars(self).items():
                if isinstance(value, wp.array):
                    wp.copy(value, getattr(other, name))

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def __init__(
        self,
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        *,
        world: wp.array[wp.int32] | None = None,
        world_count: int = 0,
    ):
        """Initialize the power supplies.

        Args:
            actuator_start: Start of each supply's actuators in *actuators*.
                Shape ``(B + 1,)``.
            actuators: Actuators of all supplies, sorted by supply, so that
                supply ``b`` powers
                ``actuators[actuator_start[b]:actuator_start[b + 1]]``.
            world: World of each supply, ``-1`` for none. Shape ``(B,)``.
                ``None`` places every supply outside any world.
            world_count: Number of worlds, which sets the shape of the mask in
                :meth:`reset`.

        Raises:
            ValueError: *world* does not have one entry per supply.
        """
        self.num_power_supplies = len(actuator_start) - 1
        """Number of power supplies B."""
        self.device = actuator_start.device
        self.actuator_start = actuator_start
        """Start of each supply's actuators in :attr:`actuators`, shape (B + 1,)."""
        self.actuators = actuators
        """Actuators of all supplies, sorted by supply."""
        self.world = (
            world if world is not None else wp.full(self.num_power_supplies, -1, dtype=wp.int32, device=self.device)
        )
        """World of each supply, ``-1`` for none, shape (B,)."""
        self.world_count = world_count
        """Number of worlds."""
        if self.world.shape != (self.num_power_supplies,):
            raise ValueError(f"world shape {self.world.shape} must be ({self.num_power_supplies},)")

    def _check_shapes(self, *names: str) -> None:
        """Raise ``ValueError`` unless each named array has one entry per supply."""
        for name in names:
            shape = getattr(self, name).shape
            if shape != (self.num_power_supplies,):
                raise ValueError(f"{name} shape {shape} must be ({self.num_power_supplies},)")

    def is_stateful(self) -> bool:
        """Return whether :meth:`step` needs :class:`State` objects.

        Returns:
            ``True`` if the supplies keep internal state.
        """
        return False

    def state(self) -> PowerSupplyBase.State | None:
        """Create a state at the initial condition of every supply.

        Returns:
            A new state, or ``None`` for a stateless supply.
        """
        return None

    def step(
        self,
        sim_control: Any,
        current_state: PowerSupplyBase.State | None = None,
        next_state: PowerSupplyBase.State | None = None,
        dt: float | None = None,
        *,
        next_control: Any | None = None,
    ) -> None:
        """Advance the supplies by *dt* and update the supply voltage of their actuators.

        Reads ``actuator_supply_current`` from *sim_control* and writes
        ``actuator_supply_voltage`` to *next_control* for the actuators of the
        supplies. Call once per step, after the actuators have stepped.

        Args:
            sim_control: Object carrying ``actuator_supply_current`` and the
                current ``actuator_supply_voltage``, e.g. :class:`~newton.Control`.
            current_state: Current state of a stateful supply, ``None``
                otherwise.
            next_state: State to write for a stateful supply, ``None``
                otherwise.
            dt: Timestep [s].
            next_control: Object carrying ``actuator_supply_voltage`` for the
                next step. ``None`` writes it to *sim_control*.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement step")

    def reset(
        self,
        sim_control: Any,
        state: PowerSupplyBase.State | None = None,
        *,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Reset the supplies and the supply voltage of their actuators.

        Writes ``actuator_supply_voltage`` for the actuators of the reset
        supplies, and their initial condition into *state*. Call after
        resetting worlds.

        Args:
            sim_control: Object carrying ``actuator_supply_voltage``, e.g.
                :class:`~newton.Control`.
            state: State of a stateful supply to reset, i.e. the
                *current_state* of the next :meth:`step`. ``None`` resets the
                voltage only.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``
                specifying which worlds to reset. Entries before the last select
                local worlds by index, and the final entry selects supplies
                whose world is ``-1``. If ``None``, all supplies are reset.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement reset")

    def _check_step(
        self,
        current_state: PowerSupplyBase.State | None,
        next_state: PowerSupplyBase.State | None,
        dt: float | None,
    ) -> None:
        if dt is None:
            raise ValueError(f"{type(self).__name__}.step() requires dt")
        if self.is_stateful() and (current_state is None or next_state is None):
            raise ValueError(
                f"{type(self).__name__} is stateful and requires current_state and next_state; create them with state()"
            )
