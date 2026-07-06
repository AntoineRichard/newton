# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wheeled-vehicle metadata: custom-attribute registration, build-time
annotation helpers, and reading finalized models into flat device tables.

The metadata layer answers two questions for the runtime kernels without any
per-wheel Python work in the step loop:

* which shapes/bodies are wheels, and which vehicle owns each wheel, and
* the per-wheel role (driven/steerable/side) and per-vehicle drive geometry.

Wheel identity uses the proven Newton custom-attribute pattern so that wheel and
vehicle ids remain correct when template builders are replicated/merged
(``references=`` remaps indices on :meth:`ModelBuilder.add_builder`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.sim import Model, ModelBuilder

VEHICLE_NAMESPACE = "vehicle"
_VEHICLE_FREQUENCY = f"{VEHICLE_NAMESPACE}:vehicle"
_VEHICLE_INDEX_ATTR = f"{VEHICLE_NAMESPACE}:vehicle_index"
_WHEEL_FREQUENCY = f"{VEHICLE_NAMESPACE}:wheel"
_WHEEL_INDEX_ATTR = f"{VEHICLE_NAMESPACE}:wheel_index"

# SHAPE-frequency attributes carrying per-wheel identity, geometry, and role.
# (name, dtype, default, references)
_SHAPE_SPECS = (
    ("is_wheel", wp.bool, False, None),
    ("wheel_id", wp.int32, -1, _WHEEL_FREQUENCY),
    ("vehicle_id", wp.int32, -1, _VEHICLE_FREQUENCY),
    ("radius", wp.float32, 0.0, None),
    ("width", wp.float32, 0.0, None),
    ("driven", wp.bool, False, None),
    ("steerable", wp.bool, False, None),
    ("side", wp.int32, 0, None),  # -1 left, 0 center, +1 right
    ("axle_row", wp.int32, 0, None),  # 0 front, 1 rear, ...
    ("steer_joint", wp.int32, -1, "joint"),  # steering joint index, or -1
    ("forward_axis", wp.vec3, wp.vec3(1.0, 0.0, 0.0), None),  # wheel body frame
    ("axle_axis", wp.vec3, wp.vec3(0.0, 1.0, 0.0), None),  # wheel body frame
)

# BODY-frequency attributes mapping wheel bodies back to wheel ids.
_BODY_SPECS = (
    ("is_wheel_body", wp.bool, False, None),
    ("wheel_body_id", wp.int32, -1, _WHEEL_FREQUENCY),
)

# Per-vehicle drive geometry, indexed by the vehicle custom frequency.
_VEHICLE_SPECS = (
    ("drive_mode", wp.int32, 0, None),
    ("wheelbase", wp.float32, 0.0, None),
    ("track_width", wp.float32, 0.0, None),
    ("steer_limit", wp.float32, 0.0, None),
)


@dataclass
class VehicleModelData:
    """Flat, device-resident wheel/vehicle tables consumed by the runtime kernels.

    All per-wheel arrays have length ``wheel_count`` and are indexed by the flat
    wheel id; all per-vehicle arrays have length ``vehicle_count`` and are indexed
    by the flat vehicle id. The tables are built once at construction; the step
    loop never reindexes or rediscovers them.

    Args:
        wheel_count: Number of wheels across all vehicles.
        vehicle_count: Number of vehicles.
        device: Warp device holding the arrays.
        wheel_shape: Shape index per wheel.
        wheel_body: Body index per wheel.
        wheel_vehicle: Vehicle id per wheel.
        radius: Wheel radius [m] per wheel.
        width: Wheel width [m] per wheel.
        driven: 1 if the wheel receives drive command, else 0, per wheel.
        steerable: 1 if the wheel receives steering command, else 0, per wheel.
        side: -1 left / 0 center / +1 right, per wheel.
        axle_row: Axle row (0 front, 1 rear, ...), per wheel.
        steer_dof: Steering joint DOF index, or -1, per wheel.
        forward_axis: Wheel forward axis in body frame, per wheel.
        axle_axis: Wheel spin axis in body frame, per wheel.
        wheel_center: Wheel-shape center in the body frame [m], per wheel (used to
            place the tire wrench at the wheel's ground contact rather than the body COM).
        drive_mode: :class:`DriveMode` value per vehicle.
        wheelbase: Wheelbase [m] per vehicle.
        track_width: Track width [m] per vehicle.
        steer_limit: Steering angle limit [rad] per vehicle.
        vehicle_wheel_count: Wheel count per vehicle (diagnostic).
    """

    wheel_count: int
    vehicle_count: int
    shape_count: int
    device: wp.context.Device
    shape_to_wheel: wp.array[wp.int32]
    wheel_shape: wp.array[wp.int32]
    wheel_body: wp.array[wp.int32]
    wheel_vehicle: wp.array[wp.int32]
    radius: wp.array[wp.float32]
    width: wp.array[wp.float32]
    driven: wp.array[wp.int32]
    steerable: wp.array[wp.int32]
    side: wp.array[wp.int32]
    axle_row: wp.array[wp.int32]
    steer_dof: wp.array[wp.int32]
    forward_axis: wp.array[wp.vec3]
    axle_axis: wp.array[wp.vec3]
    wheel_center: wp.array[wp.vec3]
    drive_mode: wp.array[wp.int32]
    wheelbase: wp.array[wp.float32]
    track_width: wp.array[wp.float32]
    steer_limit: wp.array[wp.float32]
    vehicle_wheel_count: wp.array[wp.int32]


@dataclass(frozen=True)
class VehicleAssetMetadata:
    """Vehicle asset metadata loaded from a fixture manifest.

    Args:
        name: Stable asset name.
        file: Absolute USD asset path.
        drive_mode: :class:`DriveMode` value parsed from the manifest vehicle type.
        wheel_radius: Wheel radius [m].
        wheel_width: Wheel width [m].
        wheelbase: Front-to-rear axle distance [m] (0 if not specified).
        track_width: Left-to-right wheel distance [m] (0 if not specified).
        steer_limit: Maximum steering angle [rad] (0 if not specified).
        wheel_body_labels: Wheel body labels, one per wheel.
        wheel_shape_labels: Wheel collision shape labels, one per wheel.
        steering_joint_labels: Optional steering joint labels.
        axle_joint_labels: Optional physical axle (wheel-spin) joint labels.
    """

    name: str
    file: Path
    drive_mode: int
    wheel_radius: float
    wheel_width: float
    wheelbase: float
    track_width: float
    steer_limit: float
    wheel_body_labels: tuple[str, ...]
    wheel_shape_labels: tuple[str, ...]
    steering_joint_labels: tuple[str, ...] = ()
    axle_joint_labels: tuple[str, ...] = ()


def register_vehicle_attributes(builder: ModelBuilder) -> None:
    """Register the ``vehicle:*`` custom attributes on ``builder``.

    Must be called before any :func:`add_wheel`/:func:`set_vehicle` annotation and
    before :meth:`ModelBuilder.finalize`.

    Args:
        builder: Model builder to receive the attributes.
    """
    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="vehicle",
            namespace=VEHICLE_NAMESPACE,
            usd_prim_filter=_usd_is_vehicle_prim,
        )
    )
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="vehicle_index",
            frequency=_VEHICLE_FREQUENCY,
            assignment=Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=-1,
            namespace=VEHICLE_NAMESPACE,
            references=_VEHICLE_FREQUENCY,
            usd_attribute_name="newton:vehicle:vehicle_id",
        )
    )
    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="wheel",
            namespace=VEHICLE_NAMESPACE,
            usd_prim_filter=_usd_is_wheel_prim,
        )
    )
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="wheel_index",
            frequency=_WHEEL_FREQUENCY,
            assignment=Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=-1,
            namespace=VEHICLE_NAMESPACE,
            references=_WHEEL_FREQUENCY,
            usd_attribute_name="newton:vehicle:wheel_id",
        )
    )
    for name, dtype, default, references in _SHAPE_SPECS:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                frequency=Model.AttributeFrequency.SHAPE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                namespace=VEHICLE_NAMESPACE,
                references=references,
            )
        )
    for name, dtype, default, references in _BODY_SPECS:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                namespace=VEHICLE_NAMESPACE,
                references=references,
            )
        )
    for name, dtype, default, references in _VEHICLE_SPECS:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                frequency=_VEHICLE_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                namespace=VEHICLE_NAMESPACE,
                references=references,
            )
        )


def set_vehicle(
    builder: ModelBuilder,
    vehicle_id: int,
    *,
    drive_mode: int,
    wheelbase: float = 0.0,
    track_width: float = 0.0,
    steer_limit: float = 0.0,
) -> None:
    """Set per-vehicle drive geometry on ``builder``.

    Args:
        builder: Builder with registered ``vehicle:*`` attributes.
        vehicle_id: Flat vehicle id (>= 0).
        drive_mode: :class:`DriveMode` value.
        wheelbase: Front-to-rear axle distance [m] (Ackermann).
        track_width: Left-to-right wheel distance [m].
        steer_limit: Maximum steering angle [rad].
    """
    if vehicle_id < 0:
        raise ValueError("vehicle_id must be non-negative")
    _require_registered(builder)
    current = int(builder._custom_frequency_counts.get(_VEHICLE_FREQUENCY, 0))
    if vehicle_id != current:
        raise ValueError(
            f"set_vehicle must be called once per vehicle in increasing id order; "
            f"expected vehicle_id {current}, got {vehicle_id}"
        )
    # Custom-frequency rows are appended atomically: every per-vehicle attribute
    # must be supplied in one call so they share the same row.
    builder.add_custom_values(
        **{
            _VEHICLE_INDEX_ATTR: vehicle_id,
            f"{VEHICLE_NAMESPACE}:drive_mode": int(drive_mode),
            f"{VEHICLE_NAMESPACE}:wheelbase": float(wheelbase),
            f"{VEHICLE_NAMESPACE}:track_width": float(track_width),
            f"{VEHICLE_NAMESPACE}:steer_limit": float(steer_limit),
        }
    )


def add_wheel(
    builder: ModelBuilder,
    *,
    shape: int,
    vehicle_id: int,
    wheel_id: int,
    radius: float,
    width: float,
    body: int | None = None,
    driven: bool = True,
    steerable: bool = False,
    side: int = 0,
    axle_row: int = 0,
    steer_joint: int = -1,
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    axle_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> None:
    """Annotate ``shape`` as a wheel.

    Args:
        builder: Builder with registered ``vehicle:*`` attributes.
        shape: Wheel collision shape index.
        vehicle_id: Owning vehicle id (>= 0).
        wheel_id: Flat wheel id (>= 0), unique across the model.
        radius: Wheel radius [m].
        width: Wheel width [m].
        body: Wheel body index. Defaults to the shape's attached body.
        driven: Whether the wheel receives drive commands.
        steerable: Whether the wheel receives steering commands.
        side: -1 left, 0 center, +1 right (skid-steer differential, Ackermann L/R).
        axle_row: Axle row index (0 front, 1 rear, ...).
        steer_joint: Steering joint index controlling this wheel, or -1.
        forward_axis: Wheel forward (rolling) axis in the wheel body frame.
        axle_axis: Wheel spin axis in the wheel body frame.
    """
    if vehicle_id < 0 or wheel_id < 0:
        raise ValueError("vehicle_id and wheel_id must be non-negative")
    if radius <= 0.0 or width <= 0.0:
        raise ValueError("radius and width must be positive")
    _require_registered(builder)
    if body is None:
        body = int(builder.shape_body[shape])
    elif int(builder.shape_body[shape]) != body:
        raise ValueError(f"shape {shape} is attached to body {int(builder.shape_body[shape])}, not {body}")

    vehicle_rows = int(builder._custom_frequency_counts.get(_VEHICLE_FREQUENCY, 0))
    if vehicle_id >= vehicle_rows:
        raise ValueError(f"call set_vehicle({vehicle_id}, ...) before adding its wheels")
    _reserve_rows(builder, _WHEEL_FREQUENCY, _WHEEL_INDEX_ATTR, wheel_id + 1)

    _set(builder, f"{VEHICLE_NAMESPACE}:is_wheel", shape, True)
    _set(builder, f"{VEHICLE_NAMESPACE}:wheel_id", shape, int(wheel_id))
    _set(builder, f"{VEHICLE_NAMESPACE}:vehicle_id", shape, int(vehicle_id))
    _set(builder, f"{VEHICLE_NAMESPACE}:radius", shape, float(radius))
    _set(builder, f"{VEHICLE_NAMESPACE}:width", shape, float(width))
    _set(builder, f"{VEHICLE_NAMESPACE}:driven", shape, bool(driven))
    _set(builder, f"{VEHICLE_NAMESPACE}:steerable", shape, bool(steerable))
    _set(builder, f"{VEHICLE_NAMESPACE}:side", shape, int(side))
    _set(builder, f"{VEHICLE_NAMESPACE}:axle_row", shape, int(axle_row))
    _set(builder, f"{VEHICLE_NAMESPACE}:steer_joint", shape, int(steer_joint))
    _set(builder, f"{VEHICLE_NAMESPACE}:forward_axis", shape, wp.vec3(*forward_axis))
    _set(builder, f"{VEHICLE_NAMESPACE}:axle_axis", shape, wp.vec3(*axle_axis))
    _set(builder, f"{VEHICLE_NAMESPACE}:is_wheel_body", body, True)
    _set(builder, f"{VEHICLE_NAMESPACE}:wheel_body_id", body, int(wheel_id))


def load_vehicle_manifest(path: str | Path) -> tuple[VehicleAssetMetadata, ...]:
    """Load a vehicle fixture manifest.

    The manifest is a JSON file with an ``assets`` list. Each asset names a USD
    file (relative to the manifest), the wheel body/shape labels, reference
    dimensions (wheel radius/width [m], and optionally wheelbase/track width [m]
    and steering limit [deg]), and optional steering/axle joint labels. The
    asset's ``vehicle_type`` (``"generic"``, ``"ackermann"``, or ``"skid_steer"``)
    maps to a :class:`DriveMode` value.

    Args:
        path: Manifest path.

    Returns:
        Parsed vehicle asset metadata entries in manifest order.

    Raises:
        ValueError: If the manifest does not match the metadata contract.
    """
    from .controller import DriveMode  # noqa: PLC0415  (deferred: controller imports this module)

    drive_modes = {
        "generic": int(DriveMode.GENERIC),
        "ackermann": int(DriveMode.ACKERMANN),
        "skid_steer": int(DriveMode.SKID_STEER),
    }

    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise ValueError(f"vehicle manifest file is not readable: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"vehicle manifest file is not valid JSON: {manifest_path}") from exc

    assets = data.get("assets")
    if not isinstance(assets, list):
        raise ValueError("vehicle manifest requires an assets list")

    seen_names: set[str] = set()
    parsed: list[VehicleAssetMetadata] = []
    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            raise ValueError("vehicle manifest asset entries must be objects")

        name = _require_str(raw_asset, "name", "<unknown>")
        if name in seen_names:
            raise ValueError(f"duplicate vehicle asset name: {name}")
        seen_names.add(name)

        file_name = _require_str(raw_asset, "file", name)
        asset_file = manifest_path.parent / file_name
        if not asset_file.exists():
            raise ValueError(f"asset {name} has invalid file: {asset_file}")

        vehicle_type = raw_asset.get("vehicle_type", "generic")
        if vehicle_type not in drive_modes:
            raise ValueError(
                f"asset {name} has unknown vehicle_type {vehicle_type!r}; expected one of {sorted(drive_modes)}"
            )

        wheel_body_labels = _require_str_tuple(raw_asset, "wheel_body_labels", name)
        wheel_shape_labels = _require_str_tuple(raw_asset, "wheel_shape_labels", name)
        steering_joint_labels = _optional_str_tuple(raw_asset, "steering_joint_labels", name)
        axle_joint_labels = _optional_str_tuple(raw_asset, "axle_joint_labels", name)
        if len(wheel_body_labels) != len(wheel_shape_labels):
            raise ValueError(
                f"asset {name} wheel_shape_labels length must match wheel_body_labels length "
                f"({len(wheel_shape_labels)} != {len(wheel_body_labels)})"
            )
        if not wheel_body_labels:
            raise ValueError(f"asset {name} wheel_body_labels must not be empty")
        if axle_joint_labels and len(axle_joint_labels) != len(wheel_body_labels):
            raise ValueError(
                f"asset {name} axle_joint_labels length must match wheel_body_labels length "
                f"({len(axle_joint_labels)} != {len(wheel_body_labels)})"
            )

        reference_dimensions = raw_asset.get("reference_dimensions")
        if not isinstance(reference_dimensions, dict):
            raise ValueError(f"asset {name} reference_dimensions must be an object")
        wheel_radius = _require_positive_float(reference_dimensions, "wheel_radius_m", name)
        wheel_width = _require_positive_float(reference_dimensions, "wheel_width_m", name)
        wheelbase = _optional_positive_float(reference_dimensions, "wheelbase_m", name)
        track_width = _optional_positive_float(reference_dimensions, "track_width_m", name)
        steer_limit = np.radians(_optional_positive_float(reference_dimensions, "steering_limit_deg", name))

        parsed.append(
            VehicleAssetMetadata(
                name=name,
                file=asset_file,
                drive_mode=drive_modes[vehicle_type],
                wheel_radius=wheel_radius,
                wheel_width=wheel_width,
                wheelbase=wheelbase,
                track_width=track_width,
                steer_limit=float(steer_limit),
                wheel_body_labels=wheel_body_labels,
                wheel_shape_labels=wheel_shape_labels,
                steering_joint_labels=steering_joint_labels,
                axle_joint_labels=axle_joint_labels,
            )
        )

    return tuple(parsed)


def apply_vehicle_manifest(
    builder: ModelBuilder,
    asset: VehicleAssetMetadata,
    *,
    vehicle_id: int | None = None,
    wheel_id_start: int | None = None,
) -> None:
    """Annotate an already-imported manifest asset via :func:`set_vehicle`/:func:`add_wheel`.

    Resolves the asset's wheel body/shape labels against ``builder`` and stamps
    the per-vehicle drive geometry and per-wheel identity/role attributes. Wheel
    roles are derived from the wheel body label leaf names, which must contain
    exactly one of ``front``/``rear`` (axle row) and exactly one of
    ``left``/``right`` (side); front wheels of a
    steering-equipped asset are steerable and bound to the steering joint whose
    label leaf contains the matching side.

    Call once per asset in vehicle order; ``vehicle_id``/``wheel_id_start``
    default to continuing from the rows already annotated on ``builder``.

    Args:
        builder: Builder with registered ``vehicle:*`` attributes containing the
            imported asset.
        asset: Manifest metadata for the asset.
        vehicle_id: Flat vehicle id. Defaults to the next unused id.
        wheel_id_start: First flat wheel id. Defaults to the next unused id.

    Raises:
        ValueError: If labels are missing or inconsistent.
    """
    _require_registered(builder)
    if vehicle_id is None:
        vehicle_id = int(builder._custom_frequency_counts.get(_VEHICLE_FREQUENCY, 0))
    if wheel_id_start is None:
        wheel_id_start = int(builder._custom_frequency_counts.get(_WHEEL_FREQUENCY, 0))
    if wheel_id_start < 0:
        raise ValueError(f"asset {asset.name} wheel_id_start must be non-negative")

    body_by_label = _label_lookup(builder.body_label, asset.name, "body")
    shape_by_label = _label_lookup(builder.shape_label, asset.name, "shape")
    joint_by_label = _label_lookup(builder.joint_label, asset.name, "joint")

    set_vehicle(
        builder,
        vehicle_id,
        drive_mode=asset.drive_mode,
        wheelbase=asset.wheelbase,
        track_width=asset.track_width,
        steer_limit=asset.steer_limit,
    )

    for offset, (body_label, shape_label) in enumerate(
        zip(asset.wheel_body_labels, asset.wheel_shape_labels, strict=True)
    ):
        try:
            body_index = body_by_label[body_label]
        except KeyError as exc:
            raise ValueError(f"asset {asset.name} missing wheel body label: {body_label}") from exc
        try:
            shape_index = shape_by_label[shape_label]
        except KeyError as exc:
            raise ValueError(f"asset {asset.name} missing wheel shape label: {shape_label}") from exc

        attached_body = int(builder.shape_body[shape_index])
        if attached_body != body_index:
            raise ValueError(
                f"asset {asset.name} wheel shape {shape_label} is attached to body {attached_body}, "
                f"not manifest wheel body {body_label} ({body_index})"
            )

        leaf = body_label.rsplit("/", 1)[-1]
        front = "front" in leaf
        if front == ("rear" in leaf):
            raise ValueError(
                f"asset {asset.name} wheel body label {body_label} must contain exactly one of 'front' or 'rear'"
            )
        left = "left" in leaf
        if left == ("right" in leaf):
            raise ValueError(
                f"asset {asset.name} wheel body label {body_label} must contain exactly one of 'left' or 'right'"
            )
        steer_joint = -1
        steerable = front and bool(asset.steering_joint_labels)
        if steerable:
            side_key = "left" if left else "right"
            matches = [label for label in asset.steering_joint_labels if side_key in label.rsplit("/", 1)[-1]]
            if len(matches) != 1:
                raise ValueError(
                    f"asset {asset.name} needs exactly one steering joint label containing "
                    f"{side_key!r}, found {matches}"
                )
            try:
                steer_joint = joint_by_label[matches[0]]
            except KeyError as exc:
                raise ValueError(f"asset {asset.name} missing steering joint label: {matches[0]}") from exc

        add_wheel(
            builder,
            shape=shape_index,
            vehicle_id=vehicle_id,
            wheel_id=wheel_id_start + offset,
            radius=asset.wheel_radius,
            width=asset.wheel_width,
            body=body_index,
            driven=True,
            steerable=steerable,
            side=(-1 if left else 1),
            axle_row=(0 if front else 1),
            steer_joint=steer_joint,
        )


def read_vehicle_model_data(model: Model, *, device: wp.context.Devicelike | None = None) -> VehicleModelData:
    """Read finalized ``model.vehicle`` attributes into flat device tables.

    Args:
        model: Finalized model carrying ``vehicle:*`` custom attributes.
        device: Target device for the tables. Defaults to the model device.

    Returns:
        Flat, device-resident wheel/vehicle tables.

    Raises:
        ValueError: If the authored attributes are missing or inconsistent.
    """
    ns = getattr(model, VEHICLE_NAMESPACE, None)
    if ns is None:
        raise ValueError("model does not carry vehicle:* custom attributes")
    device = model.device if device is None else wp.get_device(device)

    is_wheel = _np(ns, "is_wheel")
    wheel_id = _np(ns, "wheel_id")
    vehicle_id = _np(ns, "vehicle_id")
    radius = _np(ns, "radius")
    width = _np(ns, "width")
    driven = _np(ns, "driven")
    steerable = _np(ns, "steerable")
    side = _np(ns, "side")
    axle_row = _np(ns, "axle_row")
    steer_joint = _np(ns, "steer_joint")
    forward_axis = _np(ns, "forward_axis")
    axle_axis = _np(ns, "axle_axis")
    shape_body = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()
    joint_qd_start = model.joint_qd_start.numpy()

    wheel_shape_indices = np.nonzero(is_wheel)[0]
    rows = sorted((int(wheel_id[s]), int(s)) for s in wheel_shape_indices)
    ids = [wid for wid, _ in rows]
    if ids != list(range(len(ids))):
        raise ValueError(f"wheel ids must be contiguous 0..N-1, got {ids}")
    wheel_count = len(rows)
    shapes = [s for _, s in rows]
    shape_count = int(model.shape_count)
    s2w = np.full(shape_count, -1, dtype=np.int32)
    for wid, s in rows:
        s2w[s] = wid

    w_shape = np.array(shapes, dtype=np.int32)
    w_body = np.array([int(shape_body[s]) for s in shapes], dtype=np.int32)
    w_vehicle = np.array([int(vehicle_id[s]) for s in shapes], dtype=np.int32)
    w_radius = np.array([float(radius[s]) for s in shapes], dtype=np.float32)
    w_width = np.array([float(width[s]) for s in shapes], dtype=np.float32)
    w_driven = np.array([1 if bool(driven[s]) else 0 for s in shapes], dtype=np.int32)
    w_steerable = np.array([1 if bool(steerable[s]) else 0 for s in shapes], dtype=np.int32)
    w_side = np.array([int(side[s]) for s in shapes], dtype=np.int32)
    w_axle = np.array([int(axle_row[s]) for s in shapes], dtype=np.int32)
    w_fwd = np.array([forward_axis[s] for s in shapes], dtype=np.float32).reshape(-1, 3)
    w_axle_axis = np.array([axle_axis[s] for s in shapes], dtype=np.float32).reshape(-1, 3)
    # wheel-shape center in its body frame (shape_transform translation)
    w_center = np.array([shape_transform[s][:3] for s in shapes], dtype=np.float32).reshape(-1, 3)

    # Resolve steering joint -> DOF index (revolute steering joints have one DOF).
    w_steer_dof = np.full(wheel_count, -1, dtype=np.int32)
    for i, s in enumerate(shapes):
        j = int(steer_joint[s])
        if j >= 0:
            w_steer_dof[i] = int(joint_qd_start[j])

    vehicle_count = int(w_vehicle.max()) + 1 if wheel_count else 0
    drive_mode = _np(ns, "drive_mode")
    wheelbase = _np(ns, "wheelbase")
    track_width = _np(ns, "track_width")
    steer_limit = _np(ns, "steer_limit")
    v_count = max(vehicle_count, int(len(drive_mode)))
    v_drive_mode = np.zeros(v_count, dtype=np.int32)
    v_wheelbase = np.zeros(v_count, dtype=np.float32)
    v_track = np.zeros(v_count, dtype=np.float32)
    v_steer_limit = np.zeros(v_count, dtype=np.float32)
    v_drive_mode[: len(drive_mode)] = drive_mode.astype(np.int32)
    v_wheelbase[: len(wheelbase)] = wheelbase.astype(np.float32)
    v_track[: len(track_width)] = track_width.astype(np.float32)
    v_steer_limit[: len(steer_limit)] = steer_limit.astype(np.float32)

    v_wheel_count = np.zeros(v_count, dtype=np.int32)
    for v in w_vehicle:
        v_wheel_count[int(v)] += 1

    def arr(a, dtype):
        return wp.array(a, dtype=dtype, device=device)

    return VehicleModelData(
        wheel_count=wheel_count,
        vehicle_count=v_count,
        shape_count=shape_count,
        device=wp.get_device(device),
        shape_to_wheel=arr(s2w, wp.int32),
        wheel_shape=arr(w_shape, wp.int32),
        wheel_body=arr(w_body, wp.int32),
        wheel_vehicle=arr(w_vehicle, wp.int32),
        radius=arr(w_radius, wp.float32),
        width=arr(w_width, wp.float32),
        driven=arr(w_driven, wp.int32),
        steerable=arr(w_steerable, wp.int32),
        side=arr(w_side, wp.int32),
        axle_row=arr(w_axle, wp.int32),
        steer_dof=arr(w_steer_dof, wp.int32),
        forward_axis=wp.array(w_fwd, dtype=wp.vec3, device=device),
        axle_axis=wp.array(w_axle_axis, dtype=wp.vec3, device=device),
        wheel_center=wp.array(w_center, dtype=wp.vec3, device=device),
        drive_mode=arr(v_drive_mode, wp.int32),
        wheelbase=arr(v_wheelbase, wp.float32),
        track_width=arr(v_track, wp.float32),
        steer_limit=arr(v_steer_limit, wp.float32),
        vehicle_wheel_count=arr(v_wheel_count, wp.int32),
    )


def configure_wheel_solver_contacts(
    model: Model,
    data: VehicleModelData,
    *,
    condim: int = 1,
    priority: int = 1,
    gap: float = 0.0,
    radial_stiffness: float | None = None,
) -> None:
    """Configure wheel-ground contacts: normal-only, ground-level patch, optional
    radial compliance.

    Sets, on the wheel shapes:

    * MuJoCo ``condim`` (default 1, normal-only) and ``geom_priority`` (default 1,
      above the terrain) so the wrapped solver provides only normal support while
      the tire model owns all tangential force (avoiding double-counting, and the
      NaN that ``condim=3`` with zero friction would produce).
    * the contact ``gap`` (default 0): a zero gap removes the spurious analytic
      plane-cylinder margin contact, so the contact patch sits at ground level
      instead of being biased up the wheel.
    * optionally the contact stiffness ``shape_material_ke`` (``radial_stiffness``):
      a lower value lets the wheel sink under load, representing tire radial
      compliance and widening the fore-aft footprint. ``None`` leaves it unchanged.

    The MuJoCo ``condim``/``geom_priority`` settings require
    ``SolverMuJoCo.register_custom_attributes(builder)`` before
    :meth:`ModelBuilder.finalize`; all settings must be applied before the solver
    is constructed.

    Args:
        model: Finalized model carrying ``mujoco:*`` attributes.
        data: Vehicle tables from :func:`read_vehicle_model_data`.
        condim: MuJoCo contact dimensionality for wheel geoms (1 = normal-only).
        priority: MuJoCo geom priority for wheel geoms (must exceed the terrain's).
        gap: Contact gap [m] for wheel shapes (0 keeps the patch at ground level;
            a small positive value can help fast vehicles avoid tunneling).
        radial_stiffness: Optional wheel contact stiffness [N/m]; lower = more
            sink (radial compliance). ``None`` leaves the existing value.

    Raises:
        ValueError: If the model lacks the MuJoCo condim/priority attributes.
    """
    if data.wheel_count == 0:
        return
    ns = getattr(model, "mujoco", None)
    if ns is None or not hasattr(ns, "condim") or not hasattr(ns, "geom_priority"):
        raise ValueError(
            "model lacks mujoco:condim/geom_priority; call "
            "SolverMuJoCo.register_custom_attributes(builder) before finalize"
        )
    wheel_shapes = data.wheel_shape.numpy()
    condim_values = ns.condim.numpy()
    priority_values = ns.geom_priority.numpy()
    for s in wheel_shapes:
        condim_values[int(s)] = condim
        priority_values[int(s)] = priority
    ns.condim.assign(condim_values)
    ns.geom_priority.assign(priority_values)

    if model.shape_gap is not None:
        gap_values = model.shape_gap.numpy()
        for s in wheel_shapes:
            gap_values[int(s)] = gap
        model.shape_gap.assign(gap_values)

    if radial_stiffness is not None and model.shape_material_ke is not None:
        ke_values = model.shape_material_ke.numpy()
        for s in wheel_shapes:
            ke_values[int(s)] = radial_stiffness
        model.shape_material_ke.assign(ke_values)


# --- helpers ---------------------------------------------------------------


def _usd_is_vehicle_prim(prim: Any, _context: dict[str, Any]) -> bool:
    return _usd_has_true_attribute(prim, "newton:vehicle:is_vehicle")


def _usd_is_wheel_prim(prim: Any, _context: dict[str, Any]) -> bool:
    return _usd_has_true_attribute(prim, "newton:vehicle:is_wheel")


def _usd_has_true_attribute(prim: Any, attribute_name: str) -> bool:
    if not prim or prim.IsPseudoRoot():
        return False
    attr = prim.GetAttribute(attribute_name)
    if not attr or not attr.HasAuthoredValueOpinion():
        return False
    return bool(attr.Get())


def _require_str(raw: dict[str, Any], key: str, asset_name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"asset {asset_name} requires non-empty {key}")
    return value


def _require_str_tuple(raw: dict[str, Any], key: str, asset_name: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"asset {asset_name} requires list {key}")
    return _str_tuple(value, key, asset_name)


def _optional_str_tuple(raw: dict[str, Any], key: str, asset_name: str) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"asset {asset_name} {key} must be a list when provided")
    return _str_tuple(value, key, asset_name)


def _str_tuple(value: list[Any], key: str, asset_name: str) -> tuple[str, ...]:
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"asset {asset_name} {key} entries must be non-empty strings")
        out.append(item)
    return tuple(out)


def _require_positive_float(raw: dict[str, Any], key: str, asset_name: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"asset {asset_name} requires numeric reference_dimensions.{key}")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"asset {asset_name} reference_dimensions.{key} must be positive")
    return value


def _optional_positive_float(raw: dict[str, Any], key: str, asset_name: str) -> float:
    if raw.get(key) is None:
        return 0.0
    return _require_positive_float(raw, key, asset_name)


def _label_lookup(labels: list[str], asset_name: str, label_kind: str) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, label in enumerate(labels):
        if label in lookup:
            raise ValueError(f"asset {asset_name} has duplicate {label_kind} label in builder: {label}")
        lookup[label] = index
    return lookup


def _require_registered(builder: ModelBuilder) -> None:
    if f"{VEHICLE_NAMESPACE}:is_wheel" not in builder.custom_attributes:
        raise ValueError("vehicle:* custom attributes are not registered; call register_vehicle_attributes first")


def _reserve_rows(builder: ModelBuilder, frequency: str, index_attr: str, count: int) -> None:
    current = int(builder._custom_frequency_counts.get(frequency, 0))
    for index in range(current, count):
        builder.add_custom_values(**{index_attr: index})


def _set(builder: ModelBuilder, key: str, index: int, value: object) -> None:
    attr = builder.custom_attributes[key]
    if attr.values is None:
        attr.values = {}
    attr.values[index] = value


def _np(namespace: object, name: str) -> np.ndarray:
    value = getattr(namespace, name, None)
    if value is None:
        raise ValueError(f"missing vehicle:{name}")
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)
