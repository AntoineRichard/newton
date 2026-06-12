# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vehicle geometry and command mapping helpers for wheeled vehicles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.sim import Control, JointType, Model

from .metadata import WheeledModelMetadata
from .tire import WheelTireControl


class WheeledVehicleLayout:
    """Flat vehicle geometry and wheel-role layout.

    The layout maps generic drive and steering command channels to wheels. It
    describes vehicle geometry and actuator wiring; it does not prescribe a
    chassis-speed or yaw-rate controller.

    Args:
        model: Model containing the wheeled bodies and steering joints.
        wheeled_metadata: Resolved wheel metadata used to size wheel arrays.
        vehicle_geometry_kind: Geometry kind per vehicle.
        wheel_drive_channel: Drive command channel per wheel. Use ``-1`` for
            non-driven wheels.
        wheel_steering_channel: Steering command channel per wheel. Use ``-1``
            for non-steerable wheels.
        wheel_steering_joint_dof_index: Steering target DOF index per wheel.
            Use ``-1`` for non-steerable wheels.
        wheel_side: Optional side role per wheel: ``LEFT``, ``RIGHT``, or
            ``UNKNOWN``.
        wheel_axle: Optional longitudinal role per wheel: ``FRONT``, ``REAR``,
            or ``UNKNOWN``.
        vehicle_wheelbase: Optional vehicle wheelbase [m], shape
            ``(vehicle_count,)``.
        vehicle_track_width: Optional vehicle track width [m], shape
            ``(vehicle_count,)``.
        vehicle_steering_limit: Optional steering limit [rad], shape
            ``(vehicle_count,)``.
    """

    class GeometryKind:
        """Vehicle geometry kind constants."""

        GENERIC = 0
        ACKERMANN = 1
        SKID_STEER = 2

    class WheelSide:
        """Wheel side role constants."""

        UNKNOWN = 0
        LEFT = -1
        RIGHT = 1

    class WheelAxle:
        """Wheel longitudinal role constants."""

        UNKNOWN = 0
        REAR = -1
        FRONT = 1

    DISABLED_CHANNEL = -1

    def __init__(
        self,
        model: Model,
        wheeled_metadata: WheeledModelMetadata,
        *,
        vehicle_geometry_kind: int | str | Sequence[int | str] | np.ndarray | None = None,
        wheel_drive_channel: Sequence[int] | np.ndarray | None = None,
        wheel_steering_channel: Sequence[int] | np.ndarray | None = None,
        wheel_steering_joint_dof_index: Sequence[int] | np.ndarray | None = None,
        wheel_side: Sequence[int] | np.ndarray | None = None,
        wheel_axle: Sequence[int] | np.ndarray | None = None,
        vehicle_wheelbase: float | Sequence[float] | np.ndarray | None = None,
        vehicle_track_width: float | Sequence[float] | np.ndarray | None = None,
        vehicle_steering_limit: float | Sequence[float] | np.ndarray | None = None,
    ):
        self.vehicle_count = int(wheeled_metadata.vehicle_count)
        self.wheel_count = int(wheeled_metadata.wheel_count)
        self.body_count = int(model.body_count)
        self.joint_dof_count = int(model.joint_dof_count)
        self.device = model.device
        self._wheeled_metadata = wheeled_metadata

        if self.vehicle_count < 0 or self.wheel_count < 0:
            raise ValueError("wheeled metadata counts must be non-negative")
        if len(wheeled_metadata.wheel_vehicle_ids) != self.wheel_count:
            raise ValueError("wheeled metadata wheel_vehicle_ids length must match wheel_count")

        vehicle_ids = _int_array(wheeled_metadata.wheel_vehicle_ids, self.wheel_count, "wheel_vehicle_ids")
        for wheel_id, vehicle_id in enumerate(vehicle_ids):
            if vehicle_id < 0 or vehicle_id >= self.vehicle_count:
                raise ValueError(f"wheel {wheel_id} has invalid vehicle id {vehicle_id}")

        geometry = _geometry_array(vehicle_geometry_kind, self.vehicle_count)
        drive_channel = _channel_array(wheel_drive_channel, self.wheel_count, "wheel_drive_channel")
        steering_channel = _channel_array(wheel_steering_channel, self.wheel_count, "wheel_steering_channel")
        steering_dof = _channel_array(
            wheel_steering_joint_dof_index, self.wheel_count, "wheel_steering_joint_dof_index"
        )
        side = _role_array(wheel_side, self.wheel_count, "wheel_side", (-1, 0, 1))
        axle = _role_array(wheel_axle, self.wheel_count, "wheel_axle", (-1, 0, 1))
        wheelbase = _optional_vehicle_float_array(vehicle_wheelbase, self.vehicle_count, "vehicle_wheelbase")
        track_width = _optional_vehicle_float_array(vehicle_track_width, self.vehicle_count, "vehicle_track_width")
        steering_limit = _optional_vehicle_float_array(
            vehicle_steering_limit, self.vehicle_count, "vehicle_steering_limit"
        )

        if wheel_drive_channel is None:
            drive_channel[:] = self.DISABLED_CHANNEL
        if wheel_steering_channel is None:
            steering_channel[:] = self.DISABLED_CHANNEL
        if wheel_steering_joint_dof_index is None:
            steering_dof[:] = self.DISABLED_CHANNEL

        for wheel_id, dof_index in enumerate(steering_dof):
            if dof_index == self.DISABLED_CHANNEL:
                continue
            if dof_index < 0 or dof_index >= self.joint_dof_count:
                raise ValueError(f"wheel {wheel_id} steering joint DOF index {dof_index} is out of range")
            if steering_channel[wheel_id] < 0:
                raise ValueError(f"wheel {wheel_id} has steering joint DOF but no steering channel")

        for wheel_id, channel in enumerate(steering_channel):
            if channel >= 0 and steering_dof[wheel_id] < 0:
                raise ValueError(f"wheel {wheel_id} has a steering channel but no steering joint DOF index")

        self.drive_channel_count = _channel_count(drive_channel)
        self.steering_channel_count = _channel_count(steering_channel)

        drive_channel_vehicle_ids = _channel_vehicle_ids(vehicle_ids, drive_channel, self.drive_channel_count, "drive")
        steering_channel_vehicle_ids = _channel_vehicle_ids(
            vehicle_ids, steering_channel, self.steering_channel_count, "steering"
        )

        self.vehicle_drive_channels = _vehicle_channels(vehicle_ids, drive_channel, self.vehicle_count)
        self.vehicle_steering_channels = _vehicle_channels(vehicle_ids, steering_channel, self.vehicle_count)
        self.drive_channel_vehicle_ids_host = tuple(int(value) for value in drive_channel_vehicle_ids)
        self.steering_channel_vehicle_ids_host = tuple(int(value) for value in steering_channel_vehicle_ids)

        self.wheel_vehicle_ids_host = tuple(int(value) for value in vehicle_ids)
        self.wheel_drive_channel_host = tuple(int(value) for value in drive_channel)
        self.wheel_steering_channel_host = tuple(int(value) for value in steering_channel)
        self.wheel_steering_joint_dof_index_host = tuple(int(value) for value in steering_dof)
        self.wheel_side_host = tuple(int(value) for value in side)
        self.wheel_axle_host = tuple(int(value) for value in axle)
        self.vehicle_geometry_kind_host = tuple(int(value) for value in geometry)

        with wp.ScopedDevice(self.device):
            self.vehicle_geometry_kind = wp.array(geometry, dtype=wp.int32)
            self.vehicle_wheelbase = wp.array(wheelbase, dtype=wp.float32)
            self.vehicle_track_width = wp.array(track_width, dtype=wp.float32)
            self.vehicle_steering_limit = wp.array(steering_limit, dtype=wp.float32)
            self.wheel_vehicle_ids = wp.array(vehicle_ids, dtype=wp.int32)
            self.wheel_drive_channel = wp.array(drive_channel, dtype=wp.int32)
            self.wheel_steering_channel = wp.array(steering_channel, dtype=wp.int32)
            self.wheel_steering_joint_dof_index = wp.array(steering_dof, dtype=wp.int32)
            self.wheel_side = wp.array(side, dtype=wp.int32)
            self.wheel_axle = wp.array(axle, dtype=wp.int32)
            self.drive_channel_vehicle_ids = wp.array(drive_channel_vehicle_ids, dtype=wp.int32)
            self.steering_channel_vehicle_ids = wp.array(steering_channel_vehicle_ids, dtype=wp.int32)

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        if int(model.body_count) != self.body_count:
            raise ValueError(f"layout body_count {self.body_count} does not match model body_count {model.body_count}")
        if int(model.joint_dof_count) != self.joint_dof_count:
            raise ValueError(
                f"layout joint_dof_count {self.joint_dof_count} does not match model joint_dof_count "
                f"{model.joint_dof_count}"
            )
        if wheeled_metadata is not self._wheeled_metadata:
            raise ValueError("layout must be updated with the wheeled metadata used to construct it")
        if int(wheeled_metadata.wheel_count) != self.wheel_count:
            raise ValueError("layout wheel_count does not match wheeled metadata wheel_count")


class WheeledVehicleControl:
    """Normalized vehicle command channel buffers.

    Args:
        layout: Vehicle layout defining channel counts.

    Attributes:
        enabled: Whether each vehicle consumes commands, shape
            ``(vehicle_count,)``.
        drive_command: Normalized drive command channels, shape
            ``(drive_channel_count,)``.
        steering_command: Normalized steering command channels, shape
            ``(steering_channel_count,)``.
    """

    def __init__(self, layout: WheeledVehicleLayout):
        self.vehicle_count = layout.vehicle_count
        self.drive_channel_count = layout.drive_channel_count
        self.steering_channel_count = layout.steering_channel_count
        self.device = layout.device
        self._layout = layout

        with wp.ScopedDevice(self.device):
            self.enabled = wp.full(self.vehicle_count, True, dtype=wp.bool)
            self.drive_command = wp.zeros(self.drive_channel_count, dtype=wp.float32)
            self.steering_command = wp.zeros(self.steering_channel_count, dtype=wp.float32)

    def _validate_update_inputs(self, layout: WheeledVehicleLayout) -> None:
        if layout is not self._layout:
            raise ValueError("vehicle control must use the layout used to construct it")
        if self.drive_channel_count != layout.drive_channel_count:
            raise ValueError("vehicle control drive_channel_count does not match layout")
        if self.steering_channel_count != layout.steering_channel_count:
            raise ValueError("vehicle control steering_channel_count does not match layout")


class WheeledVehicleState:
    """Diagnostics for vehicle command mapping.

    Args:
        layout: Vehicle layout defining channel and wheel counts.

    Attributes:
        wheel_drive_command: Per-wheel clipped normalized drive command, shape
            ``(wheel_count,)``.
        wheel_angular_speed: Per-wheel target angular speed [rad/s], shape
            ``(wheel_count,)``.
        wheel_steering_angle: Per-wheel steering target [rad], shape
            ``(wheel_count,)``.
        drive_command: Per-drive-channel clipped command, shape
            ``(drive_channel_count,)``.
        steering_command: Per-steering-channel clipped command, shape
            ``(steering_channel_count,)``.
        steering_angle: Per-steering-channel target [rad], shape
            ``(steering_channel_count,)``.
    """

    def __init__(self, layout: WheeledVehicleLayout):
        self.vehicle_count = layout.vehicle_count
        self.wheel_count = layout.wheel_count
        self.drive_channel_count = layout.drive_channel_count
        self.steering_channel_count = layout.steering_channel_count
        self.device = layout.device
        self._layout = layout

        with wp.ScopedDevice(self.device):
            self.wheel_drive_command = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.wheel_angular_speed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.wheel_steering_angle = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.drive_command = wp.zeros(self.drive_channel_count, dtype=wp.float32)
            self.steering_command = wp.zeros(self.steering_channel_count, dtype=wp.float32)
            self.steering_angle = wp.zeros(self.steering_channel_count, dtype=wp.float32)

    def clear(self) -> None:
        """Reset vehicle command mapping diagnostics."""

        self.wheel_drive_command.zero_()
        self.wheel_angular_speed.zero_()
        self.wheel_steering_angle.zero_()
        self.drive_command.zero_()
        self.steering_command.zero_()
        self.steering_angle.zero_()

    def _validate_update_inputs(self, layout: WheeledVehicleLayout) -> None:
        if layout is not self._layout:
            raise ValueError("vehicle state must use the layout used to construct it")


class WheeledMotorConfig:
    """Open-loop drive-channel command mapping configuration.

    Args:
        layout: Vehicle layout defining drive channel count.
        max_wheel_angular_speed: Command scale [rad/s], scalar or shape
            ``(drive_channel_count,)``.
        command_lower: Lower normalized command clamp, scalar or shape
            ``(drive_channel_count,)``.
        command_upper: Upper normalized command clamp, scalar or shape
            ``(drive_channel_count,)``.
    """

    def __init__(
        self,
        layout: WheeledVehicleLayout,
        *,
        max_wheel_angular_speed: float | Sequence[float] | np.ndarray = 12.0,
        command_lower: float | Sequence[float] | np.ndarray = -1.0,
        command_upper: float | Sequence[float] | np.ndarray = 1.0,
    ):
        self.drive_channel_count = layout.drive_channel_count
        self.device = layout.device
        self._layout = layout

        max_speed = _float_array(max_wheel_angular_speed, self.drive_channel_count, "max_wheel_angular_speed")
        lower = _float_array(command_lower, self.drive_channel_count, "command_lower")
        upper = _float_array(command_upper, self.drive_channel_count, "command_upper")
        _validate_bounds(lower, upper, "motor command")

        with wp.ScopedDevice(self.device):
            self.max_wheel_angular_speed = wp.array(max_speed, dtype=wp.float32)
            self.command_lower = wp.array(lower, dtype=wp.float32)
            self.command_upper = wp.array(upper, dtype=wp.float32)

    def _validate_update_inputs(self, layout: WheeledVehicleLayout) -> None:
        if layout is not self._layout:
            raise ValueError("motor config must use the layout used to construct it")


class WheeledSteeringConfig:
    """Open-loop steering-channel command mapping configuration.

    Args:
        layout: Vehicle layout defining steering channel count.
        max_steering_angle: Command scale [rad], scalar or shape
            ``(steering_channel_count,)``. Defaults to the owning vehicle's
            layout steering limit when available.
        command_lower: Lower normalized command clamp, scalar or shape
            ``(steering_channel_count,)``.
        command_upper: Upper normalized command clamp, scalar or shape
            ``(steering_channel_count,)``.
    """

    def __init__(
        self,
        layout: WheeledVehicleLayout,
        *,
        max_steering_angle: float | Sequence[float] | np.ndarray | None = None,
        command_lower: float | Sequence[float] | np.ndarray = -1.0,
        command_upper: float | Sequence[float] | np.ndarray = 1.0,
    ):
        self.steering_channel_count = layout.steering_channel_count
        self.device = layout.device
        self._layout = layout

        if max_steering_angle is None:
            steering_limit = layout.vehicle_steering_limit.numpy()
            max_angle = np.zeros(self.steering_channel_count, dtype=np.float32)
            for channel, vehicle_id in enumerate(layout.steering_channel_vehicle_ids_host):
                if vehicle_id >= 0 and steering_limit[vehicle_id] > 0.0:
                    max_angle[channel] = steering_limit[vehicle_id]
                else:
                    max_angle[channel] = 0.5
        else:
            max_angle = _float_array(max_steering_angle, self.steering_channel_count, "max_steering_angle")
        lower = _float_array(command_lower, self.steering_channel_count, "command_lower")
        upper = _float_array(command_upper, self.steering_channel_count, "command_upper")
        _validate_bounds(lower, upper, "steering command")

        with wp.ScopedDevice(self.device):
            self.max_steering_angle = wp.array(max_angle, dtype=wp.float32)
            self.command_lower = wp.array(lower, dtype=wp.float32)
            self.command_upper = wp.array(upper, dtype=wp.float32)

    def _validate_update_inputs(self, layout: WheeledVehicleLayout) -> None:
        if layout is not self._layout:
            raise ValueError("steering config must use the layout used to construct it")


def build_wheeled_vehicle_layout(
    model: Model,
    wheeled_metadata: WheeledModelMetadata,
    *,
    manifest_path: str | Path | None = None,
    asset_names: Sequence[str] | None = None,
    vehicle_geometry_kind: int | str | Sequence[int | str] | np.ndarray | None = None,
    wheel_drive_channel: Sequence[int] | np.ndarray | None = None,
    wheel_steering_channel: Sequence[int] | np.ndarray | None = None,
    wheel_steering_joint_dof_index: Sequence[int] | np.ndarray | None = None,
    wheel_side: Sequence[int] | np.ndarray | None = None,
    wheel_axle: Sequence[int] | np.ndarray | None = None,
    vehicle_wheelbase: float | Sequence[float] | np.ndarray | None = None,
    vehicle_track_width: float | Sequence[float] | np.ndarray | None = None,
    vehicle_steering_limit: float | Sequence[float] | np.ndarray | None = None,
) -> WheeledVehicleLayout:
    """Build a wheeled vehicle geometry layout.

    Args:
        model: Model containing finalized wheeled assets.
        wheeled_metadata: Resolved wheel metadata for `model`.
        manifest_path: Optional Phase 00 manifest used as a layout hint.
        asset_names: Optional manifest asset names used in the imported order.
        vehicle_geometry_kind: Explicit geometry kind per vehicle.
        wheel_drive_channel: Explicit drive channel per wheel.
        wheel_steering_channel: Explicit steering channel per wheel.
        wheel_steering_joint_dof_index: Explicit steering target DOF per wheel.
        wheel_side: Optional side role per wheel.
        wheel_axle: Optional longitudinal role per wheel.
        vehicle_wheelbase: Optional vehicle wheelbase [m].
        vehicle_track_width: Optional vehicle track width [m].
        vehicle_steering_limit: Optional steering limit [rad].

    Returns:
        Vehicle layout object with host and device-side flat arrays.
    """

    if manifest_path is not None and wheel_drive_channel is None:
        manifest_layout = _build_manifest_layout(model, wheeled_metadata, manifest_path, asset_names=asset_names)
        return WheeledVehicleLayout(model, wheeled_metadata, **manifest_layout)

    return WheeledVehicleLayout(
        model,
        wheeled_metadata,
        vehicle_geometry_kind=vehicle_geometry_kind,
        wheel_drive_channel=wheel_drive_channel,
        wheel_steering_channel=wheel_steering_channel,
        wheel_steering_joint_dof_index=wheel_steering_joint_dof_index,
        wheel_side=wheel_side,
        wheel_axle=wheel_axle,
        vehicle_wheelbase=vehicle_wheelbase,
        vehicle_track_width=vehicle_track_width,
        vehicle_steering_limit=vehicle_steering_limit,
    )


def configure_wheeled_vehicle_control(
    vehicle_control: WheeledVehicleControl,
    *,
    enabled: bool | Sequence[bool] | np.ndarray | None = None,
    drive_command: float | Sequence[float] | np.ndarray | None = None,
    steering_command: float | Sequence[float] | np.ndarray | None = None,
) -> None:
    """Assign normalized vehicle command channels.

    Scalar command values are broadcast to every matching command channel.
    Array-like values must match the corresponding channel count.

    Args:
        vehicle_control: Vehicle command buffers to configure.
        enabled: Per-vehicle enabled flags, shape ``(vehicle_count,)``.
        drive_command: Normalized drive command channels, shape
            ``(drive_channel_count,)``.
        steering_command: Normalized steering command channels, shape
            ``(steering_channel_count,)``.

    Raises:
        ValueError: If an array-like value does not match the expected shape.
    """

    enabled_values = _optional_bool_config(enabled, vehicle_control.vehicle_count, "enabled")
    if enabled_values is not None:
        vehicle_control.enabled.assign(enabled_values)

    drive_values = _optional_float_config(drive_command, vehicle_control.drive_channel_count, "drive_command")
    if drive_values is not None:
        vehicle_control.drive_command.assign(drive_values)

    steering_values = _optional_float_config(
        steering_command, vehicle_control.steering_channel_count, "steering_command"
    )
    if steering_values is not None:
        vehicle_control.steering_command.assign(steering_values)


def update_wheeled_vehicle_controls(
    model: Model,
    sim_control: Control,
    wheeled_metadata: WheeledModelMetadata,
    layout: WheeledVehicleLayout,
    vehicle_control: WheeledVehicleControl,
    vehicle_state: WheeledVehicleState,
    tire_control: WheelTireControl,
    *,
    motor_config: WheeledMotorConfig | None = None,
    steering_config: WheeledSteeringConfig | None = None,
) -> None:
    """Map vehicle command channels into tire and steering actuator targets.

    Args:
        model: Model containing the target steering DOF arrays.
        sim_control: Newton control object receiving steering targets.
        wheeled_metadata: Resolved wheel metadata for `model`.
        layout: Vehicle layout mapping wheels to command channels.
        vehicle_control: Normalized command channel buffers.
        vehicle_state: Mapping diagnostics destination.
        tire_control: Tire control receiving wheel angular speed [rad/s].
        motor_config: Optional drive-channel scaling and clamp configuration.
        steering_config: Optional steering-channel scaling and clamp
            configuration.
    """

    layout._validate_update_inputs(model, wheeled_metadata)
    vehicle_control._validate_update_inputs(layout)
    vehicle_state._validate_update_inputs(layout)
    tire_control._validate_update_inputs(model, wheeled_metadata)

    if motor_config is None:
        motor_config = WheeledMotorConfig(layout)
    else:
        motor_config._validate_update_inputs(layout)
    if steering_config is None:
        steering_config = WheeledSteeringConfig(layout)
    else:
        steering_config._validate_update_inputs(layout)

    if sim_control.joint_target_pos is None:
        raise ValueError("sim_control.joint_target_pos is required to update wheeled steering targets")

    if layout.drive_channel_count:
        wp.launch(
            _clip_drive_channels,
            dim=layout.drive_channel_count,
            inputs=[
                vehicle_control.drive_command,
                motor_config.command_lower,
                motor_config.command_upper,
            ],
            outputs=[vehicle_state.drive_command],
            device=layout.device,
        )
    if layout.steering_channel_count:
        wp.launch(
            _clip_steering_channels,
            dim=layout.steering_channel_count,
            inputs=[
                vehicle_control.steering_command,
                steering_config.command_lower,
                steering_config.command_upper,
                steering_config.max_steering_angle,
            ],
            outputs=[vehicle_state.steering_command, vehicle_state.steering_angle],
            device=layout.device,
        )

    if layout.wheel_count:
        wp.launch(
            _apply_wheel_vehicle_controls,
            dim=layout.wheel_count,
            inputs=[
                vehicle_control.enabled,
                layout.vehicle_geometry_kind,
                layout.vehicle_wheelbase,
                layout.vehicle_track_width,
                layout.wheel_vehicle_ids,
                layout.wheel_drive_channel,
                layout.wheel_steering_channel,
                layout.wheel_steering_joint_dof_index,
                layout.wheel_side,
                layout.wheel_axle,
                vehicle_state.drive_command,
                motor_config.max_wheel_angular_speed,
                vehicle_state.steering_angle,
            ],
            outputs=[
                tire_control.wheel_angular_speed,
                sim_control.joint_target_pos,
                vehicle_state.wheel_drive_command,
                vehicle_state.wheel_angular_speed,
                vehicle_state.wheel_steering_angle,
            ],
            device=layout.device,
        )


def _build_manifest_layout(
    model: Model,
    wheeled_metadata: WheeledModelMetadata,
    manifest_path: str | Path,
    *,
    asset_names: Sequence[str] | None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text())
    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise ValueError("wheeled manifest requires an assets list")
    if asset_names is not None:
        requested = tuple(asset_names)
        by_name = {asset.get("name"): asset for asset in assets if isinstance(asset, dict)}
        assets = [by_name[name] for name in requested]
    assets = [asset for asset in assets if isinstance(asset, dict)]
    if not assets:
        raise ValueError("wheeled manifest contains no usable assets")

    vehicle_count = int(wheeled_metadata.vehicle_count)
    wheel_count = int(wheeled_metadata.wheel_count)
    vehicle_geometry_kind = np.zeros(vehicle_count, dtype=np.int32)
    vehicle_wheelbase = np.zeros(vehicle_count, dtype=np.float32)
    vehicle_track_width = np.zeros(vehicle_count, dtype=np.float32)
    vehicle_steering_limit = np.zeros(vehicle_count, dtype=np.float32)
    wheel_drive_channel = np.full(wheel_count, -1, dtype=np.int32)
    wheel_steering_channel = np.full(wheel_count, -1, dtype=np.int32)
    wheel_steering_dof = np.full(wheel_count, -1, dtype=np.int32)
    wheel_side = np.zeros(wheel_count, dtype=np.int32)
    wheel_axle = np.zeros(wheel_count, dtype=np.int32)

    wheels_by_vehicle: list[list[int]] = [[] for _ in range(vehicle_count)]
    for wheel_id, vehicle_id in enumerate(wheeled_metadata.wheel_vehicle_ids):
        wheels_by_vehicle[int(vehicle_id)].append(wheel_id)

    drive_channel_count = 0
    steering_channel_count = 0
    body_labels = tuple(str(label) for label in model.body_label)
    for vehicle_id, wheel_ids in enumerate(wheels_by_vehicle):
        if not wheel_ids:
            continue
        asset = assets[vehicle_id % len(assets)]
        geometry_text = str(asset.get("vehicle_type", "generic"))
        geometry_kind = _geometry_kind(geometry_text)
        vehicle_geometry_kind[vehicle_id] = geometry_kind

        reference = asset.get("reference_dimensions") if isinstance(asset.get("reference_dimensions"), dict) else {}
        vehicle_wheelbase[vehicle_id] = float(reference.get("wheelbase_m", 0.0) or 0.0)
        vehicle_track_width[vehicle_id] = float(reference.get("track_width_m", 0.0) or 0.0)
        steering_limit_deg = float(reference.get("steering_limit_deg", 0.0) or 0.0)
        vehicle_steering_limit[vehicle_id] = np.deg2rad(steering_limit_deg) if steering_limit_deg > 0.0 else 0.0

        if geometry_kind == WheeledVehicleLayout.GeometryKind.SKID_STEER:
            left_channel = drive_channel_count
            right_channel = drive_channel_count + 1
            drive_channel_count += 2
            for wheel_id in wheel_ids:
                body_label = body_labels[int(wheeled_metadata.wheel_body_indices[wheel_id])]
                side = _infer_side(body_label)
                axle = _infer_axle(body_label)
                if side == WheeledVehicleLayout.WheelSide.UNKNOWN:
                    raise ValueError(f"could not infer left/right role for skid-steer wheel label: {body_label}")
                wheel_side[wheel_id] = side
                wheel_axle[wheel_id] = axle
                wheel_drive_channel[wheel_id] = (
                    left_channel if side == WheeledVehicleLayout.WheelSide.LEFT else right_channel
                )
            continue

        if geometry_kind == WheeledVehicleLayout.GeometryKind.ACKERMANN:
            drive_channel = drive_channel_count
            steering_channel = steering_channel_count
            drive_channel_count += 1
            steering_channel_count += 1
            steering_joint_labels = _optional_str_tuple(asset.get("steering_joint_labels"))
            steering_joint_by_side = _joint_labels_by_side(steering_joint_labels, "steering")
            if vehicle_wheelbase[vehicle_id] <= 0.0 or vehicle_track_width[vehicle_id] <= 0.0:
                raise ValueError("Ackermann manifest layout requires positive wheelbase and track width")
            if vehicle_steering_limit[vehicle_id] <= 0.0:
                raise ValueError("Ackermann manifest layout requires a positive steering limit")
            for wheel_id in wheel_ids:
                body_index = int(wheeled_metadata.wheel_body_indices[wheel_id])
                body_label = body_labels[body_index]
                side = _infer_side(body_label)
                axle = _infer_axle(body_label)
                wheel_side[wheel_id] = side
                wheel_axle[wheel_id] = axle
                wheel_drive_channel[wheel_id] = drive_channel
                if axle == WheeledVehicleLayout.WheelAxle.FRONT:
                    steering_joint_label = steering_joint_by_side.get(side)
                    if steering_joint_label is None:
                        raise ValueError(f"could not resolve steering joint label for front wheel {body_label}")
                    wheel_steering_channel[wheel_id] = steering_channel
                    wheel_steering_dof[wheel_id] = _resolve_manifest_steering_dof(
                        model,
                        steering_joint_label,
                        asset_instance_index=vehicle_id // len(assets),
                    )
            continue

        drive_channel = drive_channel_count
        drive_channel_count += 1
        for wheel_id in wheel_ids:
            body_label = body_labels[int(wheeled_metadata.wheel_body_indices[wheel_id])]
            wheel_side[wheel_id] = _infer_side(body_label)
            wheel_axle[wheel_id] = _infer_axle(body_label)
            wheel_drive_channel[wheel_id] = drive_channel

    return {
        "vehicle_geometry_kind": vehicle_geometry_kind,
        "wheel_drive_channel": wheel_drive_channel,
        "wheel_steering_channel": wheel_steering_channel,
        "wheel_steering_joint_dof_index": wheel_steering_dof,
        "wheel_side": wheel_side,
        "wheel_axle": wheel_axle,
        "vehicle_wheelbase": vehicle_wheelbase,
        "vehicle_track_width": vehicle_track_width,
        "vehicle_steering_limit": vehicle_steering_limit,
    }


def _optional_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("manifest joint label fields must be arrays of strings")
    labels = tuple(str(label) for label in value)
    if any(not label for label in labels):
        raise ValueError("manifest joint label fields must contain non-empty strings")
    return labels


def _joint_labels_by_side(labels: Sequence[str], kind: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for label in labels:
        side = _infer_side(label)
        if side == WheeledVehicleLayout.WheelSide.UNKNOWN:
            continue
        if side in out:
            raise ValueError(f"manifest contains multiple {kind} joint labels for side {side}")
        out[side] = label
    return out


def _resolve_manifest_steering_dof(model: Model, manifest_joint_label: str, *, asset_instance_index: int) -> int:
    joint_qd_start = model.joint_qd_start.numpy()
    joint_type = model.joint_type.numpy()
    matches = []
    for joint_index, label in enumerate(model.joint_label):
        if not _matches_manifest_label(str(label), manifest_joint_label):
            continue
        if int(joint_type[joint_index]) != int(JointType.REVOLUTE):
            raise ValueError(f"steering joint {model.joint_label[joint_index]} must be revolute")
        matches.append(int(joint_qd_start[joint_index]))
    if asset_instance_index < 0 or asset_instance_index >= len(matches):
        raise ValueError(
            f"expected steering joint {manifest_joint_label} for asset instance {asset_instance_index}, "
            f"found {len(matches)} matches"
        )
    return matches[asset_instance_index]


def _matches_manifest_label(label: str, manifest_label: str) -> bool:
    return label == manifest_label or label.endswith(manifest_label)


def _infer_side(label: str) -> int:
    lower = label.lower()
    if "left" in lower:
        return WheeledVehicleLayout.WheelSide.LEFT
    if "right" in lower:
        return WheeledVehicleLayout.WheelSide.RIGHT
    return WheeledVehicleLayout.WheelSide.UNKNOWN


def _infer_axle(label: str) -> int:
    lower = label.lower()
    if "front" in lower:
        return WheeledVehicleLayout.WheelAxle.FRONT
    if "rear" in lower or "back" in lower:
        return WheeledVehicleLayout.WheelAxle.REAR
    return WheeledVehicleLayout.WheelAxle.UNKNOWN


def _geometry_kind(value: int | str) -> int:
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key in ("generic", "none"):
            return WheeledVehicleLayout.GeometryKind.GENERIC
        if key == "ackermann":
            return WheeledVehicleLayout.GeometryKind.ACKERMANN
        if key in ("skid_steer", "skidsteer"):
            return WheeledVehicleLayout.GeometryKind.SKID_STEER
        raise ValueError(f"unknown wheeled vehicle geometry kind: {value}")
    value = int(value)
    if value not in (
        WheeledVehicleLayout.GeometryKind.GENERIC,
        WheeledVehicleLayout.GeometryKind.ACKERMANN,
        WheeledVehicleLayout.GeometryKind.SKID_STEER,
    ):
        raise ValueError(f"unknown wheeled vehicle geometry kind: {value}")
    return value


def _geometry_array(values: int | str | Sequence[int | str] | np.ndarray | None, count: int) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.int32)
    if isinstance(values, str) or np.asarray(values).ndim == 0:
        value = values.item() if isinstance(values, np.ndarray) else values
        return np.full(count, _geometry_kind(value), dtype=np.int32)
    raw = list(values)
    if len(raw) != count:
        raise ValueError(f"vehicle_geometry_kind must have shape ({count},), got ({len(raw)},)")
    return np.array([_geometry_kind(value) for value in raw], dtype=np.int32)


def _int_array(values: Sequence[int] | np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.int32)


def _channel_array(values: Sequence[int] | np.ndarray | None, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.full(count, -1, dtype=np.int32)
    array = _int_array(values, count, name)
    if np.any(array < -1):
        raise ValueError(f"{name} entries must be -1 or non-negative channel indices")
    return array


def _role_array(
    values: Sequence[int] | np.ndarray | None, count: int, name: str, allowed: tuple[int, ...]
) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.int32)
    array = _int_array(values, count, name)
    allowed_set = {int(value) for value in allowed}
    for value in array:
        if int(value) not in allowed_set:
            raise ValueError(f"{name} entries must be one of {sorted(allowed_set)}, got {int(value)}")
    return array


def _optional_vehicle_float_array(
    values: float | Sequence[float] | np.ndarray | None, count: int, name: str
) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.float32)
    return _float_array(values, count, name)


def _float_array(values: float | Sequence[float] | np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return np.full(count, float(array), dtype=np.float32)
    if array.shape != (count,):
        raise ValueError(f"{name} must be a scalar or have shape ({count},), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _optional_float_config(
    values: float | Sequence[float] | np.ndarray | None, count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None
    return _float_array(values, count, name)


def _optional_bool_config(
    values: bool | Sequence[bool] | np.ndarray | None, count: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=bool)
    if array.ndim == 0:
        return np.full(count, bool(array), dtype=bool)
    if array.shape != (count,):
        raise ValueError(f"{name} must be a scalar or have shape ({count},), got {array.shape}")
    return np.ascontiguousarray(array, dtype=bool)


def _channel_count(channels: np.ndarray) -> int:
    active = channels[channels >= 0]
    if active.size == 0:
        return 0
    return int(np.max(active)) + 1


def _channel_vehicle_ids(
    wheel_vehicle_ids: np.ndarray, channels: np.ndarray, channel_count: int, channel_name: str
) -> np.ndarray:
    out = np.full(channel_count, -1, dtype=np.int32)
    for wheel_id, channel in enumerate(channels):
        if channel < 0:
            continue
        vehicle_id = int(wheel_vehicle_ids[wheel_id])
        if out[channel] == -1:
            out[channel] = vehicle_id
        elif int(out[channel]) != vehicle_id:
            raise ValueError(f"{channel_name} channel {channel} is assigned to multiple vehicles")
    return out


def _vehicle_channels(
    wheel_vehicle_ids: np.ndarray, channels: np.ndarray, vehicle_count: int
) -> tuple[tuple[int, ...], ...]:
    out: list[set[int]] = [set() for _ in range(vehicle_count)]
    for wheel_id, channel in enumerate(channels):
        if channel >= 0:
            out[int(wheel_vehicle_ids[wheel_id])].add(int(channel))
    return tuple(tuple(sorted(channels_for_vehicle)) for channels_for_vehicle in out)


def _validate_bounds(lower: np.ndarray, upper: np.ndarray, name: str) -> None:
    if np.any(lower > upper):
        raise ValueError(f"{name} lower bounds must be <= upper bounds")


@wp.kernel
def _clip_drive_channels(
    command: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    clipped: wp.array[wp.float32],
):
    channel = wp.tid()
    clipped[channel] = wp.clamp(command[channel], lower[channel], upper[channel])


@wp.kernel
def _clip_steering_channels(
    command: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    max_angle: wp.array[wp.float32],
    clipped: wp.array[wp.float32],
    steering_angle: wp.array[wp.float32],
):
    channel = wp.tid()
    value = wp.clamp(command[channel], lower[channel], upper[channel])
    clipped[channel] = value
    steering_angle[channel] = value * max_angle[channel]


@wp.func
def _ackermann_angle(center_angle: float, wheelbase: float, track_width: float, side: int) -> float:
    abs_center = wp.abs(center_angle)
    if abs_center <= 1.0e-5 or wheelbase <= 0.0 or track_width <= 0.0 or side == 0:
        return center_angle

    turn_sign = float(1.0)
    if center_angle < 0.0:
        turn_sign = -1.0
    radius = wheelbase / wp.tan(abs_center)
    denom = radius + float(side) * turn_sign * 0.5 * track_width
    if denom <= 1.0e-4:
        denom = 1.0e-4
    return turn_sign * wp.atan(wheelbase / denom)


@wp.kernel
def _apply_wheel_vehicle_controls(
    vehicle_enabled: wp.array[wp.bool],
    vehicle_geometry_kind: wp.array[wp.int32],
    vehicle_wheelbase: wp.array[wp.float32],
    vehicle_track_width: wp.array[wp.float32],
    wheel_vehicle_ids: wp.array[wp.int32],
    wheel_drive_channel: wp.array[wp.int32],
    wheel_steering_channel: wp.array[wp.int32],
    wheel_steering_joint_dof_index: wp.array[wp.int32],
    wheel_side: wp.array[wp.int32],
    wheel_axle: wp.array[wp.int32],
    clipped_drive_command: wp.array[wp.float32],
    max_wheel_angular_speed: wp.array[wp.float32],
    steering_angle_by_channel: wp.array[wp.float32],
    tire_wheel_angular_speed: wp.array[wp.float32],
    joint_target_pos: wp.array[wp.float32],
    wheel_drive_command: wp.array[wp.float32],
    wheel_angular_speed: wp.array[wp.float32],
    wheel_steering_angle: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    vehicle_id = wheel_vehicle_ids[wheel_id]

    drive_value = float(0.0)
    angular_speed = float(0.0)
    drive_channel = wheel_drive_channel[wheel_id]
    if vehicle_enabled[vehicle_id] and drive_channel >= 0:
        drive_value = clipped_drive_command[drive_channel]
        angular_speed = drive_value * max_wheel_angular_speed[drive_channel]
    tire_wheel_angular_speed[wheel_id] = angular_speed
    wheel_drive_command[wheel_id] = drive_value
    wheel_angular_speed[wheel_id] = angular_speed

    steering_angle = float(0.0)
    steering_channel = wheel_steering_channel[wheel_id]
    steering_dof = wheel_steering_joint_dof_index[wheel_id]
    if vehicle_enabled[vehicle_id] and steering_channel >= 0 and steering_dof >= 0:
        steering_angle = steering_angle_by_channel[steering_channel]
        if (
            vehicle_geometry_kind[vehicle_id] == int(1)
            and wheel_axle[wheel_id] == int(1)
            and wheel_side[wheel_id] != int(0)
        ):
            steering_angle = _ackermann_angle(
                steering_angle,
                vehicle_wheelbase[vehicle_id],
                vehicle_track_width[vehicle_id],
                wheel_side[wheel_id],
            )
        joint_target_pos[steering_dof] = steering_angle
    wheel_steering_angle[wheel_id] = steering_angle
