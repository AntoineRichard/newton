# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tire-force helpers for wheeled vehicles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from newton._src.sim import Model, State

from .contact_patch import WheelContactPatchState
from .metadata import WheeledModelMetadata

_TIRE_MODEL_SATURATED_LINEAR = 0
_TIRE_MODEL_FIALA = 1
_TIRE_MODEL_NAME_TO_ID = {
    "saturated_linear": _TIRE_MODEL_SATURATED_LINEAR,
    "saturated-linear": _TIRE_MODEL_SATURATED_LINEAR,
    "linear": _TIRE_MODEL_SATURATED_LINEAR,
    "fiala": _TIRE_MODEL_FIALA,
    "brush": _TIRE_MODEL_FIALA,
    "brush_fiala": _TIRE_MODEL_FIALA,
    "brush-fiala": _TIRE_MODEL_FIALA,
}


class WheelTireControl:
    """Per-wheel analytical tire model commands and configuration.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        enabled: Whether tire forces are active, shape ``(wheel_count,)``.
        tire_model: Tire model identifier, shape ``(wheel_count,)``. Use
            ``TireModel`` constants.
        wheel_angular_speed: Analytical wheel angular speed [rad/s], shape
            ``(wheel_count,)``.
        friction_mu: Optional tire friction override, shape ``(wheel_count,)``.
            Negative values use the contact patch material seed.
        fallback_normal_load: Explicit normal load used when no latched solver
            normal load is available [N], shape ``(wheel_count,)``.
        forward_axis_body: Wheel forward axis in body frame, shape
            ``(wheel_count, 3)``.
        axle_axis_body: Wheel axle axis in body frame, shape
            ``(wheel_count, 3)``.
            This records the wheel convention for callers and future tire
            models; the saturated-linear tire model uses the analytical angular
            speed command directly.
        longitudinal_stiffness: Gain from longitudinal slip speed to force
            [N/(m/s)], shape ``(wheel_count,)``.
        lateral_stiffness: Lateral tire stiffness, shape ``(wheel_count,)``.
            The saturated-linear model uses [N/(m/s)]; the Fiala model uses
            cornering stiffness [N/rad].
        min_reference_speed: Lower bound for slip diagnostics [m/s], shape
            ``(wheel_count,)``.
    """

    class TireModel:
        """Tire model identifiers for :attr:`WheelTireControl.tire_model`.

        Attributes:
            SATURATED_LINEAR: Saturated linear velocity-slip model.
            FIALA: Brush/Fiala lateral model with saturated-linear
                longitudinal force.
        """

        SATURATED_LINEAR = _TIRE_MODEL_SATURATED_LINEAR
        FIALA = _TIRE_MODEL_FIALA

    def __init__(self, model: Model, wheeled_metadata: WheeledModelMetadata):
        self.wheel_count = int(wheeled_metadata.wheel_count)
        self.body_count = int(model.body_count)
        self.device = model.device
        self._wheeled_metadata = wheeled_metadata

        wheel_body_indices = tuple(int(index) for index in wheeled_metadata.wheel_body_indices)
        wheel_radius = tuple(float(radius) for radius in wheeled_metadata.wheel_radius)
        _validate_metadata_arrays(self.wheel_count, self.body_count, wheel_body_indices, wheel_radius)

        forward_axes = np.zeros((self.wheel_count, 3), dtype=np.float32)
        axle_axes = np.zeros((self.wheel_count, 3), dtype=np.float32)
        if self.wheel_count:
            forward_axes[:, 0] = 1.0
            axle_axes[:, 1] = 1.0

        with wp.ScopedDevice(self.device):
            self.enabled = wp.full(self.wheel_count, True, dtype=wp.bool)
            self.tire_model = wp.full(self.wheel_count, self.TireModel.SATURATED_LINEAR, dtype=wp.int32)
            self.wheel_angular_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.friction_mu = wp.full(self.wheel_count, -1.0, dtype=wp.float32)
            self.fallback_normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.forward_axis_body = wp.array(forward_axes, dtype=wp.vec3)
            self.axle_axis_body = wp.array(axle_axes, dtype=wp.vec3)
            self.longitudinal_stiffness = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.lateral_stiffness = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.min_reference_speed = wp.full(self.wheel_count, 0.1, dtype=wp.float32)

            self._wheel_body_indices = wp.array(np.array(wheel_body_indices, dtype=np.int32), dtype=wp.int32)
            self._wheel_radius = wp.array(np.array(wheel_radius, dtype=np.float32), dtype=wp.float32)

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="tire control",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


class WheelTireState:
    """Per-wheel tire diagnostics and latched normal loads.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        normal_load: Normal load used for the current force solve [N], shape
            ``(wheel_count,)``.
        previous_normal_load: Normal load latched from the previous solver
            contact-force report [N], shape ``(wheel_count,)``.
        longitudinal_direction: World-space forward tangent direction, shape
            ``(wheel_count,)``.
        lateral_direction: World-space lateral tangent direction, shape
            ``(wheel_count,)``.
        wheel_angular_speed: Analytical wheel angular speed used by the tire
            model [rad/s], shape ``(wheel_count,)``.
        longitudinal_speed: Contact-point speed along the longitudinal tangent
            [m/s], shape ``(wheel_count,)``.
        lateral_speed: Contact-point speed along the lateral tangent [m/s],
            shape ``(wheel_count,)``.
        longitudinal_slip_speed: Wheel surface speed minus longitudinal patch
            speed [m/s], shape ``(wheel_count,)``.
        longitudinal_slip_ratio: Diagnostic longitudinal slip ratio, shape
            ``(wheel_count,)``.
        lateral_slip_angle: Diagnostic lateral slip angle [rad], shape
            ``(wheel_count,)``.
        requested_longitudinal_force: Unclipped longitudinal tire force [N],
            shape ``(wheel_count,)``.
        requested_lateral_force: Unclipped lateral tire force [N], shape
            ``(wheel_count,)``.
        applied_longitudinal_force: Saturated longitudinal tire force [N],
            shape ``(wheel_count,)``.
        applied_lateral_force: Saturated lateral tire force [N], shape
            ``(wheel_count,)``.
        friction_limit: Coulomb force-circle limit [N], shape
            ``(wheel_count,)``.
        combined_slip_scale: Force saturation scale, shape ``(wheel_count,)``.
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
            self.normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.previous_normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_direction = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self.lateral_direction = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self.wheel_angular_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.lateral_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_slip_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_slip_ratio = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.lateral_slip_angle = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.requested_longitudinal_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.requested_lateral_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.applied_longitudinal_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.applied_lateral_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.friction_limit = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.combined_slip_scale = wp.zeros(self.wheel_count, dtype=wp.float32)

    def clear(self, *, clear_previous_normal_load: bool = False) -> None:
        """Reset per-step tire diagnostics.

        Args:
            clear_previous_normal_load: Also clear the latched solver normal
                load [N]. By default the latch is preserved across steps.
        """

        if self.wheel_count == 0:
            return

        wp.launch(
            kernel=_clear_wheel_tire_state,
            dim=self.wheel_count,
            inputs=[clear_previous_normal_load],
            outputs=[
                self.normal_load,
                self.previous_normal_load,
                self.longitudinal_direction,
                self.lateral_direction,
                self.wheel_angular_speed,
                self.longitudinal_speed,
                self.lateral_speed,
                self.longitudinal_slip_speed,
                self.longitudinal_slip_ratio,
                self.lateral_slip_angle,
                self.requested_longitudinal_force,
                self.requested_lateral_force,
                self.applied_longitudinal_force,
                self.applied_lateral_force,
                self.friction_limit,
                self.combined_slip_scale,
            ],
            device=self.device,
        )

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="tire state",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


def apply_wheel_tire_forces(
    model: Model,
    state: State,
    wheeled_metadata: WheeledModelMetadata,
    patch_state: WheelContactPatchState,
    tire_control: WheelTireControl,
    tire_state: WheelTireState,
) -> None:
    """Apply analytical tire forces at wheel contact patches.

    The helper consumes Phase 1B contact patch diagnostics, computes
    longitudinal and configured lateral tire forces, clips the combined force by
    the Coulomb force circle, and accumulates the resulting world-frame wrench
    into :attr:`State.body_f`. Normal support, suspension, steering, and any
    physical joints remain owned by the main rigid solver.

    Args:
        model: Model that owns wheel body and COM arrays.
        state: Current simulation state. `body_f` receives accumulated wrenches.
        wheeled_metadata: Phase 1A wheel metadata.
        patch_state: Contact patch diagnostics for the current collision pass.
        tire_control: Per-wheel analytical tire commands and configuration.
        tire_state: Per-wheel diagnostics and normal-load latch.
    """

    tire_control._validate_update_inputs(model, wheeled_metadata)
    tire_state._validate_update_inputs(model, wheeled_metadata)
    _validate_patch_state(patch_state, wheeled_metadata)
    _validate_state_arrays(state)

    if tire_state.wheel_count == 0:
        return

    wp.launch(
        kernel=_apply_wheel_tire_forces,
        dim=tire_state.wheel_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            tire_control._wheel_body_indices,
            tire_control._wheel_radius,
            patch_state.active,
            patch_state.center,
            patch_state.normal,
            patch_state.friction_mu_seed,
            tire_control.enabled,
            tire_control.tire_model,
            tire_control.wheel_angular_speed,
            tire_control.friction_mu,
            tire_control.fallback_normal_load,
            tire_control.forward_axis_body,
            tire_control.longitudinal_stiffness,
            tire_control.lateral_stiffness,
            tire_control.min_reference_speed,
            tire_state.previous_normal_load,
        ],
        outputs=[
            state.body_f,
            tire_state.normal_load,
            tire_state.longitudinal_direction,
            tire_state.lateral_direction,
            tire_state.wheel_angular_speed,
            tire_state.longitudinal_speed,
            tire_state.lateral_speed,
            tire_state.longitudinal_slip_speed,
            tire_state.longitudinal_slip_ratio,
            tire_state.lateral_slip_angle,
            tire_state.requested_longitudinal_force,
            tire_state.requested_lateral_force,
            tire_state.applied_longitudinal_force,
            tire_state.applied_lateral_force,
            tire_state.friction_limit,
            tire_state.combined_slip_scale,
        ],
        device=tire_state.device,
    )


def update_wheel_tire_normal_loads(
    patch_state: WheelContactPatchState,
    tire_state: WheelTireState,
    *,
    clear_inactive: bool = False,
) -> None:
    """Latch usable Phase 1B normal-force diagnostics for the next tire solve.

    Positive `patch_state.normal_force` values from active patches replace
    `tire_state.previous_normal_load`. Inactive patches and non-positive force
    reports preserve the previous latch by default so a wheel can keep using
    the last reported solver load across the pre-step/post-step handoff. Set
    `clear_inactive` to clear inactive wheel latches explicitly.

    Args:
        patch_state: Contact patch diagnostics containing normal forces [N].
        tire_state: Destination tire diagnostics and normal-load latch.
        clear_inactive: Clear latches for inactive patches.
    """

    _validate_patch_tire_binding(patch_state, tire_state)

    if tire_state.wheel_count == 0:
        return

    wp.launch(
        kernel=_update_wheel_tire_normal_loads,
        dim=tire_state.wheel_count,
        inputs=[patch_state.active, patch_state.normal_force, clear_inactive],
        outputs=[tire_state.previous_normal_load],
        device=tire_state.device,
    )


def configure_wheel_tire_control(
    tire_control: WheelTireControl,
    *,
    enabled: bool | Sequence[bool] | np.ndarray | None = None,
    tire_model: int | str | Sequence[int] | Sequence[str] | np.ndarray | None = None,
    friction_mu: float | Sequence[float] | np.ndarray | None = None,
    fallback_normal_load: float | Sequence[float] | np.ndarray | None = None,
    longitudinal_stiffness: float | Sequence[float] | np.ndarray | None = None,
    lateral_stiffness: float | Sequence[float] | np.ndarray | None = None,
    min_reference_speed: float | Sequence[float] | np.ndarray | None = None,
    forward_axis_body: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    axle_axis_body: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
) -> None:
    """Assign static per-wheel tire control configuration.

    Scalar values are broadcast to every wheel. Array-like values must either
    match ``(wheel_count,)`` for scalar fields or ``(wheel_count, 3)`` for body
    axes. A single axis with shape ``(3,)`` is broadcast to every wheel. Runtime
    wheel speed commands should usually be written to
    :attr:`WheelTireControl.wheel_angular_speed` each step.

    Args:
        tire_control: Tire control object to configure.
        enabled: Whether tire forces are active, shape ``(wheel_count,)``.
        tire_model: Tire model id or name, shape ``(wheel_count,)``. Valid
            names include ``"saturated_linear"``, ``"linear"``, ``"fiala"``,
            and ``"brush"``.
        friction_mu: Optional tire friction override, shape ``(wheel_count,)``.
            Negative values use the contact patch material seed.
        fallback_normal_load: Explicit normal load fallback [N], shape
            ``(wheel_count,)``.
        longitudinal_stiffness: Gain from longitudinal slip speed to force
            [N/(m/s)], shape ``(wheel_count,)``.
        lateral_stiffness: Lateral tire stiffness, shape ``(wheel_count,)``.
            The saturated-linear model uses [N/(m/s)]; the Fiala model uses
            cornering stiffness [N/rad].
        min_reference_speed: Lower bound for slip diagnostics [m/s], shape
            ``(wheel_count,)``.
        forward_axis_body: Wheel forward axis in body frame, shape
            ``(wheel_count, 3)`` or ``(3,)``.
        axle_axis_body: Wheel axle axis in body frame, shape
            ``(wheel_count, 3)`` or ``(3,)``.

    Raises:
        ValueError: If an array-like value does not match the expected shape.
    """

    wheel_count = tire_control.wheel_count

    enabled_values = _optional_bool_config(enabled, wheel_count, "enabled")
    if enabled_values is not None:
        tire_control.enabled.assign(enabled_values)

    tire_model_values = _optional_tire_model_config(tire_model, wheel_count, "tire_model")
    if tire_model_values is not None:
        tire_control.tire_model.assign(tire_model_values)

    for name, values, destination in (
        ("friction_mu", friction_mu, tire_control.friction_mu),
        ("fallback_normal_load", fallback_normal_load, tire_control.fallback_normal_load),
        ("longitudinal_stiffness", longitudinal_stiffness, tire_control.longitudinal_stiffness),
        ("lateral_stiffness", lateral_stiffness, tire_control.lateral_stiffness),
        ("min_reference_speed", min_reference_speed, tire_control.min_reference_speed),
    ):
        config_values = _optional_float_config(values, wheel_count, name)
        if config_values is not None:
            destination.assign(config_values)

    forward_axes = _optional_vec3_config(forward_axis_body, wheel_count, "forward_axis_body")
    if forward_axes is not None:
        tire_control.forward_axis_body.assign(forward_axes)

    axle_axes = _optional_vec3_config(axle_axis_body, wheel_count, "axle_axis_body")
    if axle_axes is not None:
        tire_control.axle_axis_body.assign(axle_axes)


def _optional_tire_model_config(
    values: int | str | Sequence[int] | Sequence[str] | np.ndarray | None, wheel_count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return np.full(wheel_count, _coerce_tire_model_id(array.item(), name), dtype=np.int32)
    if array.shape != (wheel_count,):
        raise ValueError(f"{name} must be a scalar or have shape ({wheel_count},), got {array.shape}")
    return np.array([_coerce_tire_model_id(value, name) for value in array.tolist()], dtype=np.int32)


def _coerce_tire_model_id(value: object, name: str) -> int:
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_")
        if key in _TIRE_MODEL_NAME_TO_ID:
            return _TIRE_MODEL_NAME_TO_ID[key]
        raise ValueError(f"{name} has unknown tire model {value!r}")

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a tire model id or name, got {value!r}")
    if isinstance(value, (int, np.integer)):
        model_id = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        model_id = int(value)
    else:
        raise ValueError(f"{name} must be a tire model id or name, got {value!r}")

    if model_id not in (_TIRE_MODEL_SATURATED_LINEAR, _TIRE_MODEL_FIALA):
        raise ValueError(f"{name} has unknown tire model id {model_id}")
    return model_id


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
        raise ValueError("patch state must use the same wheeled metadata as the tire state")


def _validate_patch_tire_binding(patch_state: WheelContactPatchState, tire_state: WheelTireState) -> None:
    if int(patch_state.wheel_count) != int(tire_state.wheel_count):
        raise ValueError(
            "patch state wheel_count does not match tire state wheel_count "
            f"({patch_state.wheel_count} != {tire_state.wheel_count})"
        )
    if patch_state._wheeled_metadata is not tire_state._wheeled_metadata:
        raise ValueError("patch state and tire state must use the same wheeled metadata")


def _validate_state_arrays(state: State) -> None:
    if state.body_q is None:
        raise ValueError("state.body_q is required to apply wheel tire forces")
    if state.body_qd is None:
        raise ValueError("state.body_qd is required to apply wheel tire forces")
    if state.body_f is None:
        raise ValueError("state.body_f is required to apply wheel tire forces")


@wp.func
def _safe_normalize(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3()


@wp.func
def _sign_nonzero(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


@wp.func
def _fiala_lateral_force(tan_alpha: float, cornering_stiffness: float, friction_limit: float) -> float:
    if cornering_stiffness <= 0.0 or friction_limit <= 0.0:
        return 0.0

    abs_tan_alpha = wp.abs(tan_alpha)
    if abs_tan_alpha <= 1.0e-6:
        return 0.0

    transition_tan_alpha = 3.0 * friction_limit / cornering_stiffness
    if abs_tan_alpha >= transition_tan_alpha:
        return -friction_limit * _sign_nonzero(tan_alpha)

    stiffness_squared = cornering_stiffness * cornering_stiffness
    stiffness_cubed = stiffness_squared * cornering_stiffness
    tan_alpha_squared = tan_alpha * tan_alpha
    return (
        -cornering_stiffness * tan_alpha
        + stiffness_squared * abs_tan_alpha * tan_alpha / (3.0 * friction_limit)
        - stiffness_cubed * tan_alpha_squared * tan_alpha / (27.0 * friction_limit * friction_limit)
    )


@wp.kernel
def _clear_wheel_tire_state(
    clear_previous_normal_load: bool,
    normal_load: wp.array[wp.float32],
    previous_normal_load: wp.array[wp.float32],
    longitudinal_direction: wp.array[wp.vec3],
    lateral_direction: wp.array[wp.vec3],
    wheel_angular_speed: wp.array[wp.float32],
    longitudinal_speed: wp.array[wp.float32],
    lateral_speed: wp.array[wp.float32],
    longitudinal_slip_speed: wp.array[wp.float32],
    longitudinal_slip_ratio: wp.array[wp.float32],
    lateral_slip_angle: wp.array[wp.float32],
    requested_longitudinal_force: wp.array[wp.float32],
    requested_lateral_force: wp.array[wp.float32],
    applied_longitudinal_force: wp.array[wp.float32],
    applied_lateral_force: wp.array[wp.float32],
    friction_limit: wp.array[wp.float32],
    combined_slip_scale: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    normal_load[wheel_id] = 0.0
    if clear_previous_normal_load:
        previous_normal_load[wheel_id] = 0.0
    longitudinal_direction[wheel_id] = wp.vec3()
    lateral_direction[wheel_id] = wp.vec3()
    wheel_angular_speed[wheel_id] = 0.0
    longitudinal_speed[wheel_id] = 0.0
    lateral_speed[wheel_id] = 0.0
    longitudinal_slip_speed[wheel_id] = 0.0
    longitudinal_slip_ratio[wheel_id] = 0.0
    lateral_slip_angle[wheel_id] = 0.0
    requested_longitudinal_force[wheel_id] = 0.0
    requested_lateral_force[wheel_id] = 0.0
    applied_longitudinal_force[wheel_id] = 0.0
    applied_lateral_force[wheel_id] = 0.0
    friction_limit[wheel_id] = 0.0
    combined_slip_scale[wheel_id] = 0.0


@wp.kernel
def _apply_wheel_tire_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    wheel_body_indices: wp.array[wp.int32],
    wheel_radius: wp.array[wp.float32],
    patch_active: wp.array[wp.bool],
    patch_center: wp.array[wp.vec3],
    patch_normal: wp.array[wp.vec3],
    patch_friction_mu_seed: wp.array[wp.float32],
    enabled: wp.array[wp.bool],
    tire_model: wp.array[wp.int32],
    commanded_wheel_angular_speed: wp.array[wp.float32],
    friction_mu: wp.array[wp.float32],
    fallback_normal_load: wp.array[wp.float32],
    forward_axis_body: wp.array[wp.vec3],
    longitudinal_stiffness: wp.array[wp.float32],
    lateral_stiffness: wp.array[wp.float32],
    min_reference_speed: wp.array[wp.float32],
    previous_normal_load: wp.array[wp.float32],
    body_f: wp.array[wp.spatial_vector],
    normal_load: wp.array[wp.float32],
    longitudinal_direction: wp.array[wp.vec3],
    lateral_direction: wp.array[wp.vec3],
    wheel_angular_speed: wp.array[wp.float32],
    longitudinal_speed: wp.array[wp.float32],
    lateral_speed: wp.array[wp.float32],
    longitudinal_slip_speed: wp.array[wp.float32],
    longitudinal_slip_ratio: wp.array[wp.float32],
    lateral_slip_angle: wp.array[wp.float32],
    requested_longitudinal_force: wp.array[wp.float32],
    requested_lateral_force: wp.array[wp.float32],
    applied_longitudinal_force: wp.array[wp.float32],
    applied_lateral_force: wp.array[wp.float32],
    friction_limit: wp.array[wp.float32],
    combined_slip_scale: wp.array[wp.float32],
):
    wheel_id = wp.tid()

    normal_load[wheel_id] = 0.0
    longitudinal_direction[wheel_id] = wp.vec3()
    lateral_direction[wheel_id] = wp.vec3()
    wheel_angular_speed[wheel_id] = 0.0
    longitudinal_speed[wheel_id] = 0.0
    lateral_speed[wheel_id] = 0.0
    longitudinal_slip_speed[wheel_id] = 0.0
    longitudinal_slip_ratio[wheel_id] = 0.0
    lateral_slip_angle[wheel_id] = 0.0
    requested_longitudinal_force[wheel_id] = 0.0
    requested_lateral_force[wheel_id] = 0.0
    applied_longitudinal_force[wheel_id] = 0.0
    applied_lateral_force[wheel_id] = 0.0
    friction_limit[wheel_id] = 0.0
    combined_slip_scale[wheel_id] = 0.0

    if not enabled[wheel_id] or not patch_active[wheel_id]:
        return

    body_index = wheel_body_indices[wheel_id]
    radius = wheel_radius[wheel_id]
    if body_index < 0 or radius <= 1.0e-6:
        return

    support_normal = _safe_normalize(patch_normal[wheel_id])
    if wp.length(support_normal) <= 1.0e-6:
        return

    X_wb = body_q[body_index]
    forward_world = wp.transform_vector(X_wb, forward_axis_body[wheel_id])
    projected_forward = forward_world - support_normal * wp.dot(forward_world, support_normal)
    forward_tangent = _safe_normalize(projected_forward)
    if wp.length(forward_tangent) <= 1.0e-6:
        return

    lateral_tangent = _safe_normalize(wp.cross(support_normal, forward_tangent))
    if wp.length(lateral_tangent) <= 1.0e-6:
        return

    twist = body_qd[body_index]
    linear_velocity = wp.spatial_top(twist)
    angular_velocity = wp.spatial_bottom(twist)
    com_world = wp.transform_point(X_wb, body_com[body_index])
    patch_offset = patch_center[wheel_id] - com_world
    patch_velocity = linear_velocity + wp.cross(angular_velocity, patch_offset)

    commanded_omega = commanded_wheel_angular_speed[wheel_id]
    wheel_surface_speed = commanded_omega * radius
    longitudinal_patch_speed = wp.dot(patch_velocity, forward_tangent)
    lateral_patch_speed = wp.dot(patch_velocity, lateral_tangent)
    longitudinal_slip = wheel_surface_speed - longitudinal_patch_speed

    reference_speed = min_reference_speed[wheel_id]
    if reference_speed < 1.0e-4:
        reference_speed = 1.0e-4
    if wp.abs(longitudinal_patch_speed) > reference_speed:
        reference_speed = wp.abs(longitudinal_patch_speed)
    if wp.abs(wheel_surface_speed) > reference_speed:
        reference_speed = wp.abs(wheel_surface_speed)

    longitudinal_direction[wheel_id] = forward_tangent
    lateral_direction[wheel_id] = lateral_tangent
    wheel_angular_speed[wheel_id] = commanded_omega
    longitudinal_speed[wheel_id] = longitudinal_patch_speed
    lateral_speed[wheel_id] = lateral_patch_speed
    longitudinal_slip_speed[wheel_id] = longitudinal_slip
    longitudinal_slip_ratio[wheel_id] = longitudinal_slip / reference_speed
    lateral_slip_angle[wheel_id] = wp.atan2(lateral_patch_speed, reference_speed)

    load = previous_normal_load[wheel_id]
    if load <= 0.0:
        load = fallback_normal_load[wheel_id]
    if load <= 0.0:
        return
    normal_load[wheel_id] = load

    mu = friction_mu[wheel_id]
    if mu < 0.0:
        mu = patch_friction_mu_seed[wheel_id]
    if mu < 0.0:
        mu = 0.0

    limit = mu * load
    friction_limit[wheel_id] = limit
    if limit <= 0.0:
        return

    requested_longitudinal = longitudinal_stiffness[wheel_id] * longitudinal_slip
    requested_lateral = -lateral_stiffness[wheel_id] * lateral_patch_speed
    if tire_model[wheel_id] == wp.static(_TIRE_MODEL_FIALA):
        requested_lateral = _fiala_lateral_force(
            lateral_patch_speed / reference_speed, lateral_stiffness[wheel_id], limit
        )
    requested_longitudinal_force[wheel_id] = requested_longitudinal
    requested_lateral_force[wheel_id] = requested_lateral

    requested_norm = wp.sqrt(requested_longitudinal * requested_longitudinal + requested_lateral * requested_lateral)
    if requested_norm <= 1.0e-6:
        return

    scale = 1.0
    if requested_norm > limit:
        scale = limit / requested_norm
    combined_slip_scale[wheel_id] = scale

    applied_longitudinal = requested_longitudinal * scale
    applied_lateral = requested_lateral * scale
    applied_longitudinal_force[wheel_id] = applied_longitudinal
    applied_lateral_force[wheel_id] = applied_lateral

    force_world = forward_tangent * applied_longitudinal + lateral_tangent * applied_lateral
    torque_world = wp.cross(patch_offset, force_world)
    wp.atomic_add(body_f, body_index, wp.spatial_vector(force_world, torque_world))


@wp.kernel
def _update_wheel_tire_normal_loads(
    patch_active: wp.array[wp.bool],
    patch_normal_force: wp.array[wp.float32],
    clear_inactive: bool,
    previous_normal_load: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    if patch_active[wheel_id]:
        normal_force = patch_normal_force[wheel_id]
        if normal_force > 0.0:
            previous_normal_load[wheel_id] = normal_force
    elif clear_inactive:
        previous_normal_load[wheel_id] = 0.0
