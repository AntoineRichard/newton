# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical wheel rotational-moment helpers for wheeled vehicles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from newton._src.sim import Model, State

from .contact_patch import WheelContactPatchState
from .metadata import WheeledModelMetadata
from .tire import WheelTireControl, WheelTireState


class WheelMomentControl:
    """Per-wheel analytical rotational-moment commands and configuration.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        enabled: Whether analytical wheel moment integration is active, shape
            ``(wheel_count,)``.
        drive_torque: Drive torque applied to the analytical wheel [N·m], shape
            ``(wheel_count,)``. Positive values increase positive analytical
            wheel angular speed.
        brake_torque: Brake torque magnitude opposing wheel rotation [N·m],
            shape ``(wheel_count,)``. Negative values are treated as zero.
        wheel_inertia: Analytical wheel rotational inertia [kg·m²], shape
            ``(wheel_count,)``.
        angular_damping: Viscous wheel damping [N·m·s/rad], shape
            ``(wheel_count,)``.
        rolling_resistance_torque: Constant rolling resistance magnitude [N·m],
            shape ``(wheel_count,)``.
        apply_body_reaction_torque: Whether to accumulate equal-and-opposite
            axle/body torques into :attr:`State.body_f`, shape
            ``(wheel_count,)``.
        axle_axis_body: Positive analytical wheel axis in body frame, shape
            ``(wheel_count,)``.
    """

    def __init__(self, model: Model, wheeled_metadata: WheeledModelMetadata):
        self.wheel_count = int(wheeled_metadata.wheel_count)
        self.body_count = int(model.body_count)
        self.device = model.device
        self._wheeled_metadata = wheeled_metadata

        wheel_body_indices = tuple(int(index) for index in wheeled_metadata.wheel_body_indices)
        wheel_radius = tuple(float(radius) for radius in wheeled_metadata.wheel_radius)
        _validate_metadata_arrays(self.wheel_count, self.body_count, wheel_body_indices, wheel_radius)

        axle_axes = np.zeros((self.wheel_count, 3), dtype=np.float32)
        if self.wheel_count:
            axle_axes[:, 1] = 1.0

        with wp.ScopedDevice(self.device):
            self.enabled = wp.full(self.wheel_count, True, dtype=wp.bool)
            self.drive_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.brake_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.wheel_inertia = wp.full(self.wheel_count, 1.0, dtype=wp.float32)
            self.angular_damping = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.rolling_resistance_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.apply_body_reaction_torque = wp.full(self.wheel_count, True, dtype=wp.bool)
            self.axle_axis_body = wp.array(axle_axes, dtype=wp.vec3)

            self._wheel_body_indices = wp.array(np.array(wheel_body_indices, dtype=np.int32), dtype=wp.int32)
            self._wheel_radius = wp.array(np.array(wheel_radius, dtype=np.float32), dtype=wp.float32)

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="moment control",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


class WheelMomentState:
    """Per-wheel analytical rotational state and moment diagnostics.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        wheel_angular_speed: Integrated analytical wheel angular speed [rad/s],
            shape ``(wheel_count,)``.
        wheel_angular_acceleration: Analytical angular acceleration [rad/s²],
            shape ``(wheel_count,)``.
        net_torque: Net torque integrated by the analytical wheel [N·m], shape
            ``(wheel_count,)``.
        drive_torque: Drive torque used by the current update [N·m], shape
            ``(wheel_count,)``.
        brake_torque: Signed brake torque used by the current update [N·m],
            shape ``(wheel_count,)``.
        tire_reaction_torque: Tire reaction torque from the previous tire solve
            [N·m], shape ``(wheel_count,)``.
        damping_torque: Signed viscous damping torque [N·m], shape
            ``(wheel_count,)``.
        rolling_resistance_torque: Signed rolling resistance torque [N·m], shape
            ``(wheel_count,)``.
        body_reaction_torque: Signed axle/body reaction torque accumulated into
            :attr:`State.body_f` [N·m], shape ``(wheel_count,)``.
    """

    def __init__(self, model: Model, wheeled_metadata: WheeledModelMetadata):
        self.wheel_count = int(wheeled_metadata.wheel_count)
        self.body_count = int(model.body_count)
        self.device = model.device
        self._wheeled_metadata = wheeled_metadata

        wheel_body_indices = tuple(int(index) for index in wheeled_metadata.wheel_body_indices)
        wheel_radius = tuple(float(radius) for radius in wheeled_metadata.wheel_radius)
        _validate_metadata_arrays(self.wheel_count, self.body_count, wheel_body_indices, wheel_radius)

        with wp.ScopedDevice(self.device):
            self.wheel_angular_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.wheel_angular_acceleration = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.net_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.drive_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.brake_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.tire_reaction_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.damping_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.rolling_resistance_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.body_reaction_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self._tire_speed_output = wp.zeros(self.wheel_count, dtype=wp.float32)

    def clear(self, *, clear_wheel_angular_speed: bool = False) -> None:
        """Reset per-step wheel moment diagnostics.

        Args:
            clear_wheel_angular_speed: Also clear integrated wheel angular speed
                [rad/s]. By default angular speed is preserved across steps.
        """

        if self.wheel_count == 0:
            return

        wp.launch(
            kernel=_clear_wheel_moment_state,
            dim=self.wheel_count,
            inputs=[clear_wheel_angular_speed],
            outputs=[
                self.wheel_angular_speed,
                self.wheel_angular_acceleration,
                self.net_torque,
                self.drive_torque,
                self.brake_torque,
                self.tire_reaction_torque,
                self.damping_torque,
                self.rolling_resistance_torque,
                self.body_reaction_torque,
            ],
            device=self.device,
        )

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="moment state",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


def update_wheel_moments(
    model: Model,
    state: State,
    wheeled_metadata: WheeledModelMetadata,
    patch_state: WheelContactPatchState,
    tire_state: WheelTireState,
    moment_control: WheelMomentControl,
    moment_state: WheelMomentState,
    dt: float,
    *,
    tire_control: WheelTireControl | None = None,
) -> None:
    """Integrate analytical wheel rotational moments.

    This helper advances per-wheel angular speed from drive torque, brake torque,
    tire reaction torque, damping, rolling resistance, and wheel inertia. It is
    intended for the locked-wheel or visual-wheel path where physical axle spin
    is not owned by the rigid solver. Call it after
    :func:`apply_wheel_tire_forces` when using tire reaction torque from the
    current step.

    Args:
        model: Model that owns wheel body and COM arrays.
        state: Current simulation state. Optional body reaction moments are
            accumulated into `body_f`.
        wheeled_metadata: Phase 1A wheel metadata.
        patch_state: Contact patch diagnostics for the current collision pass.
        tire_state: Tire diagnostics from the previous tire-force solve.
        moment_control: Per-wheel moment commands and configuration.
        moment_state: Integrated wheel moment state and diagnostics.
        dt: Integration step [s].
        tire_control: Optional tire control to receive the integrated analytical
            wheel angular speed [rad/s] for the next tire solve.

    Raises:
        ValueError: If `dt` is not positive or input buffers are incompatible.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    moment_control._validate_update_inputs(model, wheeled_metadata)
    moment_state._validate_update_inputs(model, wheeled_metadata)
    _validate_patch_state(patch_state, wheeled_metadata)
    _validate_tire_state(tire_state, wheeled_metadata)
    if tire_control is not None:
        tire_control._validate_update_inputs(model, wheeled_metadata)
    _validate_state_arrays(state)

    if moment_state.wheel_count == 0:
        return

    tire_speed_output = (
        tire_control.wheel_angular_speed if tire_control is not None else moment_state._tire_speed_output
    )

    wp.launch(
        kernel=_update_wheel_moments,
        dim=moment_state.wheel_count,
        inputs=[
            state.body_q,
            model.body_com,
            moment_control._wheel_body_indices,
            moment_control._wheel_radius,
            patch_state.active,
            patch_state.center,
            moment_control.enabled,
            moment_control.drive_torque,
            moment_control.brake_torque,
            moment_control.wheel_inertia,
            moment_control.angular_damping,
            moment_control.rolling_resistance_torque,
            moment_control.apply_body_reaction_torque,
            moment_control.axle_axis_body,
            tire_state.applied_longitudinal_force,
            tire_state.longitudinal_direction,
            moment_state.wheel_angular_speed,
            float(dt),
        ],
        outputs=[
            state.body_f,
            moment_state.wheel_angular_speed,
            tire_speed_output,
            moment_state.wheel_angular_acceleration,
            moment_state.net_torque,
            moment_state.drive_torque,
            moment_state.brake_torque,
            moment_state.tire_reaction_torque,
            moment_state.damping_torque,
            moment_state.rolling_resistance_torque,
            moment_state.body_reaction_torque,
        ],
        device=moment_state.device,
    )


def configure_wheel_moment_control(
    moment_control: WheelMomentControl,
    *,
    enabled: bool | Sequence[bool] | np.ndarray | None = None,
    drive_torque: float | Sequence[float] | np.ndarray | None = None,
    brake_torque: float | Sequence[float] | np.ndarray | None = None,
    wheel_inertia: float | Sequence[float] | np.ndarray | None = None,
    angular_damping: float | Sequence[float] | np.ndarray | None = None,
    rolling_resistance_torque: float | Sequence[float] | np.ndarray | None = None,
    apply_body_reaction_torque: bool | Sequence[bool] | np.ndarray | None = None,
    axle_axis_body: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
) -> None:
    """Assign static per-wheel analytical moment configuration.

    Scalar values are broadcast to every wheel. Array-like values must either
    match ``(wheel_count,)`` for scalar fields or ``(wheel_count, 3)`` for body
    axes. A single axis with shape ``(3,)`` is broadcast to every wheel.

    Args:
        moment_control: Wheel moment control object to configure.
        enabled: Whether moment integration is active, shape ``(wheel_count,)``.
        drive_torque: Drive torque [N·m], shape ``(wheel_count,)``.
        brake_torque: Brake torque magnitude [N·m], shape ``(wheel_count,)``.
        wheel_inertia: Wheel rotational inertia [kg·m²], shape
            ``(wheel_count,)``. Values must be positive.
        angular_damping: Viscous wheel damping [N·m·s/rad], shape
            ``(wheel_count,)``.
        rolling_resistance_torque: Constant rolling resistance magnitude [N·m],
            shape ``(wheel_count,)``.
        apply_body_reaction_torque: Whether to accumulate axle/body reaction
            torque into `State.body_f`, shape ``(wheel_count,)``.
        axle_axis_body: Positive analytical wheel axis in body frame, shape
            ``(wheel_count, 3)`` or ``(3,)``.

    Raises:
        ValueError: If an array-like value does not match the expected shape or
            if non-negative fields receive invalid values.
    """

    wheel_count = moment_control.wheel_count

    enabled_values = _optional_bool_config(enabled, wheel_count, "enabled")
    if enabled_values is not None:
        moment_control.enabled.assign(enabled_values)

    reaction_values = _optional_bool_config(apply_body_reaction_torque, wheel_count, "apply_body_reaction_torque")
    if reaction_values is not None:
        moment_control.apply_body_reaction_torque.assign(reaction_values)

    signed_drive = _optional_float_config(drive_torque, wheel_count, "drive_torque")
    if signed_drive is not None:
        moment_control.drive_torque.assign(signed_drive)

    for name, values, destination, strictly_positive in (
        ("brake_torque", brake_torque, moment_control.brake_torque, False),
        ("wheel_inertia", wheel_inertia, moment_control.wheel_inertia, True),
        ("angular_damping", angular_damping, moment_control.angular_damping, False),
        (
            "rolling_resistance_torque",
            rolling_resistance_torque,
            moment_control.rolling_resistance_torque,
            False,
        ),
    ):
        config_values = _optional_float_config(values, wheel_count, name)
        if config_values is None:
            continue
        if strictly_positive:
            if np.any(config_values <= 0.0):
                raise ValueError(f"{name} values must be positive")
        elif np.any(config_values < 0.0):
            raise ValueError(f"{name} values must be non-negative")
        destination.assign(config_values)

    axle_axes = _optional_vec3_config(axle_axis_body, wheel_count, "axle_axis_body")
    if axle_axes is not None:
        moment_control.axle_axis_body.assign(axle_axes)


def _optional_float_config(
    values: float | Sequence[float] | np.ndarray | None, wheel_count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return np.full(wheel_count, float(array), dtype=np.float32)
    if array.shape != (wheel_count,):
        raise ValueError(f"{name} must be a scalar or have shape ({wheel_count},), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _optional_bool_config(
    values: bool | Sequence[bool] | np.ndarray | None, wheel_count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=bool)
    if array.ndim == 0:
        return np.full(wheel_count, bool(array), dtype=bool)
    if array.shape != (wheel_count,):
        raise ValueError(f"{name} must be a scalar or have shape ({wheel_count},), got {array.shape}")
    return np.ascontiguousarray(array, dtype=bool)


def _optional_vec3_config(
    values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None, wheel_count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=np.float32)
    if array.shape == (3,):
        return np.repeat(array.reshape(1, 3), wheel_count, axis=0).astype(np.float32)
    if array.shape != (wheel_count, 3):
        raise ValueError(f"{name} must have shape (3,) or ({wheel_count}, 3), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_metadata_arrays(
    wheel_count: int,
    body_count: int,
    wheel_body_indices: tuple[int, ...],
    wheel_radius: tuple[float, ...],
) -> None:
    if len(wheel_body_indices) != wheel_count:
        raise ValueError(
            "wheeled metadata wheel_body_indices length must match wheel_count "
            f"({len(wheel_body_indices)} != {wheel_count})"
        )
    if len(wheel_radius) != wheel_count:
        raise ValueError(
            f"wheeled metadata wheel_radius length must match wheel_count ({len(wheel_radius)} != {wheel_count})"
        )
    for wheel_id, body_index in enumerate(wheel_body_indices):
        if body_index < 0 or body_index >= body_count:
            raise ValueError(f"wheel {wheel_id} has invalid body index {body_index}")
    for wheel_id, radius in enumerate(wheel_radius):
        if radius <= 0.0:
            raise ValueError(f"wheel {wheel_id} has non-positive radius")


def _validate_model_metadata_binding(
    *,
    owner_name: str,
    owner_wheel_count: int,
    owner_body_count: int,
    owner_metadata: WheeledModelMetadata,
    model: Model,
    wheeled_metadata: WheeledModelMetadata,
) -> None:
    if int(model.body_count) != owner_body_count:
        raise ValueError(
            f"{owner_name} body_count {owner_body_count} does not match model body_count {model.body_count}"
        )
    _validate_metadata_binding(
        owner_name=owner_name,
        owner_wheel_count=owner_wheel_count,
        owner_metadata=owner_metadata,
        wheeled_metadata=wheeled_metadata,
    )


def _validate_metadata_binding(
    *,
    owner_name: str,
    owner_wheel_count: int,
    owner_metadata: WheeledModelMetadata,
    wheeled_metadata: WheeledModelMetadata,
) -> None:
    if int(wheeled_metadata.wheel_count) != owner_wheel_count:
        raise ValueError(
            f"{owner_name} wheel_count does not match wheeled metadata wheel_count "
            f"({owner_wheel_count} != {wheeled_metadata.wheel_count})"
        )
    if wheeled_metadata is not owner_metadata:
        raise ValueError(f"{owner_name} must be updated with the wheeled metadata used to construct it")


def _validate_patch_state(patch_state: WheelContactPatchState, wheeled_metadata: WheeledModelMetadata) -> None:
    if int(patch_state.wheel_count) != int(wheeled_metadata.wheel_count):
        raise ValueError(
            "patch state wheel_count does not match wheeled metadata wheel_count "
            f"({patch_state.wheel_count} != {wheeled_metadata.wheel_count})"
        )
    if patch_state._wheeled_metadata is not wheeled_metadata:
        raise ValueError("patch state must use the same wheeled metadata as the moment state")


def _validate_tire_state(tire_state: WheelTireState, wheeled_metadata: WheeledModelMetadata) -> None:
    if int(tire_state.wheel_count) != int(wheeled_metadata.wheel_count):
        raise ValueError(
            "tire state wheel_count does not match wheeled metadata wheel_count "
            f"({tire_state.wheel_count} != {wheeled_metadata.wheel_count})"
        )
    if tire_state._wheeled_metadata is not wheeled_metadata:
        raise ValueError("tire state must use the same wheeled metadata as the moment state")


def _validate_state_arrays(state: State) -> None:
    if state.body_q is None:
        raise ValueError("state.body_q is required to update wheel moments")
    if state.body_f is None:
        raise ValueError("state.body_f is required to update wheel moments")


@wp.func
def _safe_normalize(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3()


@wp.func
def _sign_from_value(value: float) -> float:
    if value > 1.0e-5:
        return 1.0
    if value < -1.0e-5:
        return -1.0
    return 0.0


@wp.kernel
def _clear_wheel_moment_state(
    clear_wheel_angular_speed: bool,
    wheel_angular_speed: wp.array[wp.float32],
    wheel_angular_acceleration: wp.array[wp.float32],
    net_torque: wp.array[wp.float32],
    drive_torque: wp.array[wp.float32],
    brake_torque: wp.array[wp.float32],
    tire_reaction_torque: wp.array[wp.float32],
    damping_torque: wp.array[wp.float32],
    rolling_resistance_torque: wp.array[wp.float32],
    body_reaction_torque: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    if clear_wheel_angular_speed:
        wheel_angular_speed[wheel_id] = 0.0
    wheel_angular_acceleration[wheel_id] = 0.0
    net_torque[wheel_id] = 0.0
    drive_torque[wheel_id] = 0.0
    brake_torque[wheel_id] = 0.0
    tire_reaction_torque[wheel_id] = 0.0
    damping_torque[wheel_id] = 0.0
    rolling_resistance_torque[wheel_id] = 0.0
    body_reaction_torque[wheel_id] = 0.0


@wp.kernel
def _update_wheel_moments(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    wheel_body_indices: wp.array[wp.int32],
    wheel_radius: wp.array[wp.float32],
    patch_active: wp.array[wp.bool],
    patch_center: wp.array[wp.vec3],
    enabled: wp.array[wp.bool],
    commanded_drive_torque: wp.array[wp.float32],
    commanded_brake_torque: wp.array[wp.float32],
    wheel_inertia: wp.array[wp.float32],
    angular_damping: wp.array[wp.float32],
    rolling_resistance_torque_magnitude: wp.array[wp.float32],
    apply_body_reaction_torque: wp.array[wp.bool],
    axle_axis_body: wp.array[wp.vec3],
    applied_longitudinal_force: wp.array[wp.float32],
    longitudinal_direction: wp.array[wp.vec3],
    previous_wheel_angular_speed: wp.array[wp.float32],
    dt: float,
    body_f: wp.array[wp.spatial_vector],
    wheel_angular_speed: wp.array[wp.float32],
    tire_control_wheel_angular_speed: wp.array[wp.float32],
    wheel_angular_acceleration: wp.array[wp.float32],
    net_torque: wp.array[wp.float32],
    drive_torque: wp.array[wp.float32],
    brake_torque: wp.array[wp.float32],
    tire_reaction_torque: wp.array[wp.float32],
    damping_torque: wp.array[wp.float32],
    rolling_resistance_torque: wp.array[wp.float32],
    body_reaction_torque: wp.array[wp.float32],
):
    wheel_id = wp.tid()

    omega = previous_wheel_angular_speed[wheel_id]
    wheel_angular_speed[wheel_id] = omega
    tire_control_wheel_angular_speed[wheel_id] = omega
    wheel_angular_acceleration[wheel_id] = 0.0
    net_torque[wheel_id] = 0.0
    drive_torque[wheel_id] = 0.0
    brake_torque[wheel_id] = 0.0
    tire_reaction_torque[wheel_id] = 0.0
    damping_torque[wheel_id] = 0.0
    rolling_resistance_torque[wheel_id] = 0.0
    body_reaction_torque[wheel_id] = 0.0

    body_index = wheel_body_indices[wheel_id]
    radius = wheel_radius[wheel_id]
    inertia = wheel_inertia[wheel_id]
    if body_index < 0 or radius <= 1.0e-6 or inertia <= 1.0e-8:
        return

    if not enabled[wheel_id]:
        return

    drive = commanded_drive_torque[wheel_id]
    brake_magnitude = commanded_brake_torque[wheel_id]
    if brake_magnitude < 0.0:
        brake_magnitude = 0.0

    damping = angular_damping[wheel_id]
    if damping < 0.0:
        damping = 0.0
    damping_signed = -damping * omega

    tire_reaction = 0.0
    tire_force = 0.0
    if patch_active[wheel_id]:
        tire_force = applied_longitudinal_force[wheel_id]
        tire_reaction = -tire_force * radius

    rolling_magnitude = rolling_resistance_torque_magnitude[wheel_id]
    if rolling_magnitude < 0.0:
        rolling_magnitude = 0.0

    rolling_signed = 0.0
    if rolling_magnitude > 0.0:
        rolling_sign = _sign_from_value(omega)
        if rolling_sign == 0.0:
            rolling_sign = _sign_from_value(drive + tire_reaction)
        if rolling_sign != 0.0:
            rolling_signed = -rolling_sign * rolling_magnitude

    non_brake_torque = drive + tire_reaction + damping_signed + rolling_signed
    brake_signed = 0.0
    if brake_magnitude > 0.0:
        brake_sign = _sign_from_value(omega)
        if brake_sign == 0.0:
            if wp.abs(non_brake_torque) <= brake_magnitude:
                brake_signed = -non_brake_torque
            else:
                brake_sign = _sign_from_value(non_brake_torque)
                brake_signed = -brake_sign * brake_magnitude
        else:
            brake_signed = -brake_sign * brake_magnitude

    net = non_brake_torque + brake_signed
    alpha = net / inertia
    next_omega = omega + alpha * dt

    if brake_magnitude > 0.0 and omega * next_omega < 0.0 and wp.abs(non_brake_torque) <= brake_magnitude:
        next_omega = 0.0
        alpha = -omega / dt
        net = alpha * inertia
        brake_signed = net - non_brake_torque

    if wp.abs(next_omega) < 1.0e-7:
        next_omega = 0.0

    wheel_angular_speed[wheel_id] = next_omega
    tire_control_wheel_angular_speed[wheel_id] = next_omega
    wheel_angular_acceleration[wheel_id] = alpha
    net_torque[wheel_id] = net
    drive_torque[wheel_id] = drive
    brake_torque[wheel_id] = brake_signed
    tire_reaction_torque[wheel_id] = tire_reaction
    damping_torque[wheel_id] = damping_signed
    rolling_resistance_torque[wheel_id] = rolling_signed

    if not apply_body_reaction_torque[wheel_id]:
        return

    X_wb = body_q[body_index]
    axle_world = _safe_normalize(wp.transform_vector(X_wb, axle_axis_body[wheel_id]))
    if wp.length(axle_world) <= 1.0e-6:
        return

    body_reaction = -(drive + brake_signed + damping_signed + rolling_signed)
    if patch_active[wheel_id] and wp.abs(tire_force) > 1.0e-7:
        com_world = wp.transform_point(X_wb, body_com[body_index])
        patch_offset = patch_center[wheel_id] - com_world
        force_world = longitudinal_direction[wheel_id] * tire_force
        contact_torque_world = wp.cross(patch_offset, force_world)
        contact_axle_torque = wp.dot(contact_torque_world, axle_world)
        body_reaction = body_reaction - contact_axle_torque

    body_reaction_torque[wheel_id] = body_reaction
    if wp.abs(body_reaction) > 1.0e-7:
        wp.atomic_add(body_f, body_index, wp.spatial_vector(wp.vec3(), axle_world * body_reaction))
