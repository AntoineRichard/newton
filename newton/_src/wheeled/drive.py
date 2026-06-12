# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Model, State

from .contact_patch import WheelContactPatchState
from .metadata import WheeledModelMetadata


class WheelDriveControl:
    """Per-wheel longitudinal drive and brake commands.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        enabled: Whether drive and brake commands are active, shape
            ``(wheel_count,)``.
        drive_torque: Requested drive torque at each wheel [N·m], shape
            ``(wheel_count,)``.
        brake_torque: Requested brake torque magnitude at each wheel [N·m],
            shape ``(wheel_count,)``.
        target_speed: Optional target longitudinal speed [m/s], shape
            ``(wheel_count,)``. Disabled when `target_speed_gain` is zero.
        target_speed_gain: Proportional gain from speed error to force
            [N/(m/s)], shape ``(wheel_count,)``.
        friction_mu: Optional tire friction override, shape ``(wheel_count,)``.
            Negative values use the contact patch material seed.
        fallback_normal_load: Explicit normal load used when no latched solver
            normal load is available [N], shape ``(wheel_count,)``.
        forward_axis_body: Wheel forward axis in body frame, shape
            ``(wheel_count,)``.
        axle_axis_body: Wheel axle axis in body frame, shape
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

        forward_axes = np.zeros((self.wheel_count, 3), dtype=np.float32)
        axle_axes = np.zeros((self.wheel_count, 3), dtype=np.float32)
        if self.wheel_count:
            forward_axes[:, 0] = 1.0
            axle_axes[:, 1] = 1.0

        with wp.ScopedDevice(self.device):
            self.enabled = wp.full(self.wheel_count, True, dtype=wp.bool)
            self.drive_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.brake_torque = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.target_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.target_speed_gain = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.friction_mu = wp.full(self.wheel_count, -1.0, dtype=wp.float32)
            self.fallback_normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.forward_axis_body = wp.array(forward_axes, dtype=wp.vec3)
            self.axle_axis_body = wp.array(axle_axes, dtype=wp.vec3)

            self._wheel_body_indices = wp.array(np.array(wheel_body_indices, dtype=np.int32), dtype=wp.int32)
            self._wheel_radius = wp.array(np.array(wheel_radius, dtype=np.float32), dtype=wp.float32)

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="drive control",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


class WheelDriveState:
    """Per-wheel diagnostics and latched normal loads for drive/brake forces.

    Args:
        model: Model that owns the wheel bodies.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        normal_load: Normal load used for the current force solve [N], shape
            ``(wheel_count,)``.
        previous_normal_load: Normal load latched from the previous solver
            contact-force report [N], shape ``(wheel_count,)``.
        longitudinal_direction: World-space tangent direction used for force
            application, shape ``(wheel_count,)``.
        wheel_angular_speed: Wheel angular speed around the configured axle
            [rad/s], shape ``(wheel_count,)``.
        longitudinal_speed: Contact-point speed along the longitudinal
            direction [m/s], shape ``(wheel_count,)``.
        slip_speed: Longitudinal speed minus axle angular speed times radius
            [m/s], shape ``(wheel_count,)``.
        requested_force: Unclipped longitudinal force request [N], shape
            ``(wheel_count,)``.
        applied_force: Clipped longitudinal force applied at the patch [N],
            shape ``(wheel_count,)``.
        friction_limit: Coulomb longitudinal force limit [N], shape
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

        with wp.ScopedDevice(self.device):
            self.normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.previous_normal_load = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_direction = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self.wheel_angular_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.longitudinal_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.slip_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.requested_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.applied_force = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.friction_limit = wp.zeros(self.wheel_count, dtype=wp.float32)

    def clear(self, *, clear_previous_normal_load: bool = False) -> None:
        """Reset per-step drive diagnostics.

        Args:
            clear_previous_normal_load: Also clear the latched solver normal
                load [N]. By default the latch is preserved across steps.
        """

        if self.wheel_count == 0:
            return

        wp.launch(
            kernel=_clear_wheel_drive_state,
            dim=self.wheel_count,
            inputs=[clear_previous_normal_load],
            outputs=[
                self.normal_load,
                self.previous_normal_load,
                self.longitudinal_direction,
                self.wheel_angular_speed,
                self.longitudinal_speed,
                self.slip_speed,
                self.requested_force,
                self.applied_force,
                self.friction_limit,
            ],
            device=self.device,
        )

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        _validate_model_metadata_binding(
            owner_name="drive state",
            owner_wheel_count=self.wheel_count,
            owner_body_count=self.body_count,
            owner_metadata=self._wheeled_metadata,
            model=model,
            wheeled_metadata=wheeled_metadata,
        )


def apply_wheel_drive_forces(
    model: Model,
    state: State,
    wheeled_metadata: WheeledModelMetadata,
    patch_state: WheelContactPatchState,
    drive_control: WheelDriveControl,
    drive_state: WheelDriveState,
) -> None:
    """Apply Coulomb-limited longitudinal wheel drive/brake forces.

    The function consumes Phase 1B contact patch diagnostics and accumulates a
    world-frame external wrench into :attr:`State.body_f` for each active wheel.
    Normal contact support remains owned by the wrapped rigid solver.

    Args:
        model: Model that owns wheel body and COM arrays.
        state: Current simulation state. `body_f` receives accumulated wrenches.
        wheeled_metadata: Phase 1A wheel metadata.
        patch_state: Contact patch diagnostics for the current collision pass.
        drive_control: Per-wheel drive/brake commands and configuration.
        drive_state: Per-wheel diagnostics and normal-load latch.
    """

    drive_control._validate_update_inputs(model, wheeled_metadata)
    drive_state._validate_update_inputs(model, wheeled_metadata)
    _validate_patch_state(patch_state, wheeled_metadata)
    _validate_state_arrays(state)

    if drive_state.wheel_count == 0:
        return

    wp.launch(
        kernel=_apply_wheel_drive_forces,
        dim=drive_state.wheel_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            drive_control._wheel_body_indices,
            drive_control._wheel_radius,
            patch_state.active,
            patch_state.center,
            patch_state.normal,
            patch_state.friction_mu_seed,
            drive_control.enabled,
            drive_control.drive_torque,
            drive_control.brake_torque,
            drive_control.target_speed,
            drive_control.target_speed_gain,
            drive_control.friction_mu,
            drive_control.fallback_normal_load,
            drive_control.forward_axis_body,
            drive_control.axle_axis_body,
            drive_state.previous_normal_load,
        ],
        outputs=[
            state.body_f,
            drive_state.normal_load,
            drive_state.longitudinal_direction,
            drive_state.wheel_angular_speed,
            drive_state.longitudinal_speed,
            drive_state.slip_speed,
            drive_state.requested_force,
            drive_state.applied_force,
            drive_state.friction_limit,
        ],
        device=drive_state.device,
    )


def update_wheel_drive_normal_loads(
    patch_state: WheelContactPatchState,
    drive_state: WheelDriveState,
    *,
    clear_inactive: bool = False,
) -> None:
    """Latch usable Phase 1B normal-force diagnostics for the next drive solve.

    Positive `patch_state.normal_force` values from active patches replace
    `drive_state.previous_normal_load`. Inactive patches and non-positive force
    reports preserve the previous latch by default so a wheel can keep using the
    last reported solver load across the pre-step/post-step handoff. Set
    `clear_inactive` to clear inactive wheel latches explicitly.

    Args:
        patch_state: Contact patch diagnostics containing normal forces [N].
        drive_state: Destination drive diagnostics and normal-load latch.
        clear_inactive: Clear latches for inactive patches.
    """

    _validate_patch_drive_binding(patch_state, drive_state)

    if drive_state.wheel_count == 0:
        return

    wp.launch(
        kernel=_update_wheel_drive_normal_loads,
        dim=drive_state.wheel_count,
        inputs=[patch_state.active, patch_state.normal_force, clear_inactive],
        outputs=[drive_state.previous_normal_load],
        device=drive_state.device,
    )


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
        raise ValueError("patch state must use the same wheeled metadata as the drive state")


def _validate_patch_drive_binding(patch_state: WheelContactPatchState, drive_state: WheelDriveState) -> None:
    if int(patch_state.wheel_count) != int(drive_state.wheel_count):
        raise ValueError(
            "patch state wheel_count does not match drive state wheel_count "
            f"({patch_state.wheel_count} != {drive_state.wheel_count})"
        )
    if patch_state._wheeled_metadata is not drive_state._wheeled_metadata:
        raise ValueError("patch state and drive state must use the same wheeled metadata")


def _validate_state_arrays(state: State) -> None:
    if state.body_q is None:
        raise ValueError("state.body_q is required to apply wheel drive forces")
    if state.body_qd is None:
        raise ValueError("state.body_qd is required to apply wheel drive forces")
    if state.body_f is None:
        raise ValueError("state.body_f is required to apply wheel drive forces")


@wp.func
def _safe_normalize(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3()


@wp.func
def _clip_scalar(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@wp.func
def _sign_from_speed(speed: float) -> float:
    if speed > 1.0e-5:
        return 1.0
    if speed < -1.0e-5:
        return -1.0
    return 0.0


@wp.kernel
def _clear_wheel_drive_state(
    clear_previous_normal_load: bool,
    normal_load: wp.array[wp.float32],
    previous_normal_load: wp.array[wp.float32],
    longitudinal_direction: wp.array[wp.vec3],
    wheel_angular_speed: wp.array[wp.float32],
    longitudinal_speed: wp.array[wp.float32],
    slip_speed: wp.array[wp.float32],
    requested_force: wp.array[wp.float32],
    applied_force: wp.array[wp.float32],
    friction_limit: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    normal_load[wheel_id] = 0.0
    if clear_previous_normal_load:
        previous_normal_load[wheel_id] = 0.0
    longitudinal_direction[wheel_id] = wp.vec3()
    wheel_angular_speed[wheel_id] = 0.0
    longitudinal_speed[wheel_id] = 0.0
    slip_speed[wheel_id] = 0.0
    requested_force[wheel_id] = 0.0
    applied_force[wheel_id] = 0.0
    friction_limit[wheel_id] = 0.0


@wp.kernel
def _apply_wheel_drive_forces(
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
    drive_torque: wp.array[wp.float32],
    brake_torque: wp.array[wp.float32],
    target_speed: wp.array[wp.float32],
    target_speed_gain: wp.array[wp.float32],
    friction_mu: wp.array[wp.float32],
    fallback_normal_load: wp.array[wp.float32],
    forward_axis_body: wp.array[wp.vec3],
    axle_axis_body: wp.array[wp.vec3],
    previous_normal_load: wp.array[wp.float32],
    body_f: wp.array[wp.spatial_vector],
    normal_load: wp.array[wp.float32],
    longitudinal_direction: wp.array[wp.vec3],
    wheel_angular_speed: wp.array[wp.float32],
    longitudinal_speed: wp.array[wp.float32],
    slip_speed: wp.array[wp.float32],
    requested_force: wp.array[wp.float32],
    applied_force: wp.array[wp.float32],
    friction_limit: wp.array[wp.float32],
):
    wheel_id = wp.tid()

    normal_load[wheel_id] = 0.0
    longitudinal_direction[wheel_id] = wp.vec3()
    wheel_angular_speed[wheel_id] = 0.0
    longitudinal_speed[wheel_id] = 0.0
    slip_speed[wheel_id] = 0.0
    requested_force[wheel_id] = 0.0
    applied_force[wheel_id] = 0.0
    friction_limit[wheel_id] = 0.0

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
    direction = _safe_normalize(projected_forward)
    if wp.length(direction) <= 1.0e-6:
        return

    axle_world = _safe_normalize(wp.transform_vector(X_wb, axle_axis_body[wheel_id]))
    twist = body_qd[body_index]
    linear_velocity = wp.spatial_top(twist)
    angular_velocity = wp.spatial_bottom(twist)
    com_world = wp.transform_point(X_wb, body_com[body_index])
    patch_offset = patch_center[wheel_id] - com_world
    patch_velocity = linear_velocity + wp.cross(angular_velocity, patch_offset)
    angular_speed = wp.dot(angular_velocity, axle_world)
    contact_speed = wp.dot(patch_velocity, direction)
    slip = contact_speed - angular_speed * radius

    longitudinal_direction[wheel_id] = direction
    wheel_angular_speed[wheel_id] = angular_speed
    longitudinal_speed[wheel_id] = contact_speed
    slip_speed[wheel_id] = slip

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

    drive_force = drive_torque[wheel_id] / radius

    brake_force = 0.0
    brake_magnitude = brake_torque[wheel_id] / radius
    if brake_magnitude > 0.0:
        brake_speed = slip
        if wp.abs(brake_speed) <= 1.0e-5:
            brake_speed = contact_speed
        speed_sign = _sign_from_speed(brake_speed)
        if speed_sign != 0.0:
            brake_force = -speed_sign * brake_magnitude

    target_force = 0.0
    speed_gain = target_speed_gain[wheel_id]
    if speed_gain > 0.0:
        target_force = (target_speed[wheel_id] - contact_speed) * speed_gain

    requested = drive_force + brake_force + target_force
    applied = _clip_scalar(requested, limit)
    requested_force[wheel_id] = requested
    applied_force[wheel_id] = applied
    if wp.abs(applied) <= 1.0e-6:
        return

    force_world = direction * applied
    torque_world = wp.cross(patch_offset, force_world)
    wp.atomic_add(body_f, body_index, wp.spatial_vector(force_world, torque_world))


@wp.kernel
def _update_wheel_drive_normal_loads(
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
