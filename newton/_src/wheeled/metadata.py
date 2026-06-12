# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.sim import Model, ModelBuilder

_WHEELED_NAMESPACE = "wheeled"
_VEHICLE_FREQUENCY = f"{_WHEELED_NAMESPACE}:vehicle"
_VEHICLE_INDEX_ATTR = f"{_WHEELED_NAMESPACE}:vehicle_index"
_WHEEL_FREQUENCY = f"{_WHEELED_NAMESPACE}:wheel"
_WHEEL_INDEX_ATTR = f"{_WHEELED_NAMESPACE}:wheel_index"


@dataclass(frozen=True)
class WheeledAssetMetadata:
    """Wheeled fixture metadata loaded from a fixture manifest.

    Args:
        name: Stable asset name.
        file: Absolute USDA asset path.
        wheel_body_labels: Wheel body labels.
        wheel_shape_labels: Wheel collision shape labels.
        wheel_radius: Wheel radius [m].
        wheel_width: Wheel width [m].
        axle_joint_labels: Optional axle joint labels.
    """

    name: str
    file: Path
    wheel_body_labels: tuple[str, ...]
    wheel_shape_labels: tuple[str, ...]
    wheel_radius: float
    wheel_width: float
    axle_joint_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class WheelMetadata:
    """Resolved host-side wheel metadata row.

    Args:
        wheel_id: Globally flat wheel index.
        vehicle_id: Vehicle instance index that owns the wheel.
        body_index: Resolved Newton body index.
        shape_index: Resolved Newton shape index.
        radius: Wheel radius [m].
        width: Wheel width [m].
    """

    wheel_id: int
    vehicle_id: int
    body_index: int
    shape_index: int
    radius: float
    width: float


@dataclass(frozen=True)
class WheeledModelMetadata:
    """Deterministic host-side wheeled model metadata table.

    Args:
        wheel_count: Number of wheels.
        vehicle_count: Number of flat vehicle id slots.
        wheel_shape_indices: Shape indices per wheel.
        wheel_body_indices: Body indices per wheel.
        wheel_vehicle_ids: Vehicle id per wheel.
        wheel_radius: Wheel radius [m] per wheel.
        wheel_width: Wheel width [m] per wheel.
        vehicle_wheel_counts: Wheel counts per vehicle id.
    """

    wheel_count: int
    vehicle_count: int
    wheel_shape_indices: tuple[int, ...]
    wheel_body_indices: tuple[int, ...]
    wheel_vehicle_ids: tuple[int, ...]
    wheel_radius: tuple[float, ...]
    wheel_width: tuple[float, ...]
    vehicle_wheel_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostics dictionary."""
        return {
            "wheel_count": self.wheel_count,
            "vehicle_count": self.vehicle_count,
            "wheel_shape_indices": list(self.wheel_shape_indices),
            "wheel_body_indices": list(self.wheel_body_indices),
            "wheel_vehicle_ids": list(self.wheel_vehicle_ids),
            "wheel_radius": list(self.wheel_radius),
            "wheel_width": list(self.wheel_width),
            "vehicle_wheel_counts": list(self.vehicle_wheel_counts),
        }


def register_wheeled_custom_attributes(builder: ModelBuilder) -> None:
    """Register Phase 1A wheeled metadata custom attributes on a builder.

    Args:
        builder: Model builder to receive `wheeled:*` attributes.
    """

    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="vehicle",
            namespace=_WHEELED_NAMESPACE,
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
            namespace=_WHEELED_NAMESPACE,
            references=_VEHICLE_FREQUENCY,
            usd_attribute_name="newton:wheeled:vehicle_id",
        )
    )

    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="wheel",
            namespace=_WHEELED_NAMESPACE,
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
            namespace=_WHEELED_NAMESPACE,
            references=_WHEEL_FREQUENCY,
            usd_attribute_name="newton:wheeled:wheel_id",
        )
    )

    specs = (
        ("is_wheel", Model.AttributeFrequency.SHAPE, wp.bool, False, None),
        ("wheel_id", Model.AttributeFrequency.SHAPE, wp.int32, -1, _WHEEL_FREQUENCY),
        ("vehicle_id", Model.AttributeFrequency.SHAPE, wp.int32, -1, _VEHICLE_FREQUENCY),
        ("wheel_radius", Model.AttributeFrequency.SHAPE, wp.float32, 0.0, None),
        ("wheel_width", Model.AttributeFrequency.SHAPE, wp.float32, 0.0, None),
        ("is_wheel_body", Model.AttributeFrequency.BODY, wp.bool, False, None),
        ("wheel_body_id", Model.AttributeFrequency.BODY, wp.int32, -1, _WHEEL_FREQUENCY),
    )
    for name, frequency, dtype, default, references in specs:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                frequency=frequency,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                namespace=_WHEELED_NAMESPACE,
                references=references,
            )
        )


def load_wheeled_manifest(path: str | Path) -> tuple[WheeledAssetMetadata, ...]:
    """Load the Phase 0 wheeled fixture manifest.

    Args:
        path: Manifest path.

    Returns:
        Parsed wheeled asset metadata entries in manifest order.

    Raises:
        ValueError: If the manifest does not match the Phase 1A metadata contract.
    """

    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise ValueError(f"wheeled manifest file is not readable: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"wheeled manifest file is not valid JSON: {manifest_path}") from exc

    assets = data.get("assets")
    if not isinstance(assets, list):
        raise ValueError("wheeled manifest requires an assets list")

    seen_names: set[str] = set()
    parsed: list[WheeledAssetMetadata] = []
    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            raise ValueError("wheeled manifest asset entries must be objects")

        name = _require_str(raw_asset, "name", "<unknown>")
        if name in seen_names:
            raise ValueError(f"duplicate wheeled asset name: {name}")
        seen_names.add(name)

        file_name = _require_str(raw_asset, "file", name)
        asset_file = manifest_path.parent / file_name
        if not asset_file.exists():
            raise ValueError(f"asset {name} has invalid file: {asset_file}")

        wheel_body_labels = _require_str_tuple(raw_asset, "wheel_body_labels", name)
        wheel_shape_labels = _require_str_tuple(raw_asset, "wheel_shape_labels", name)
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

        parsed.append(
            WheeledAssetMetadata(
                name=name,
                file=asset_file,
                wheel_body_labels=wheel_body_labels,
                wheel_shape_labels=wheel_shape_labels,
                wheel_radius=wheel_radius,
                wheel_width=wheel_width,
                axle_joint_labels=axle_joint_labels,
            )
        )

    return tuple(parsed)


def apply_wheeled_manifest_metadata(
    builder: ModelBuilder,
    asset: WheeledAssetMetadata,
    vehicle_id: int,
    *,
    wheel_id_start: int = 0,
) -> list[WheelMetadata]:
    """Resolve manifest labels and annotate imported fixture bodies/shapes.

    Args:
        builder: Builder containing the imported fixture.
        asset: Manifest metadata for the fixture.
        vehicle_id: Vehicle instance id.
        wheel_id_start: First globally flat wheel id.

    Returns:
        Resolved wheel metadata rows.

    Raises:
        ValueError: If labels are missing or inconsistent.
    """

    if vehicle_id < 0:
        raise ValueError(f"asset {asset.name} vehicle_id must be non-negative")
    if wheel_id_start < 0:
        raise ValueError(f"asset {asset.name} wheel_id_start must be non-negative")
    _require_registered(builder)

    body_by_label = _label_lookup(builder.body_label, asset.name, "body")
    shape_by_label = _label_lookup(builder.shape_label, asset.name, "shape")

    wheels: list[WheelMetadata] = []
    _reserve_custom_frequency_rows(
        builder,
        frequency=_VEHICLE_FREQUENCY,
        index_attribute=_VEHICLE_INDEX_ATTR,
        count=vehicle_id + 1,
    )
    _reserve_custom_frequency_rows(
        builder,
        frequency=_WHEEL_FREQUENCY,
        index_attribute=_WHEEL_INDEX_ATTR,
        count=wheel_id_start + len(asset.wheel_body_labels),
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

        wheel_id = wheel_id_start + offset
        _set_builder_attr(builder, "wheeled:is_wheel", shape_index, True)
        _set_builder_attr(builder, "wheeled:wheel_id", shape_index, wheel_id)
        _set_builder_attr(builder, "wheeled:vehicle_id", shape_index, vehicle_id)
        _set_builder_attr(builder, "wheeled:wheel_radius", shape_index, asset.wheel_radius)
        _set_builder_attr(builder, "wheeled:wheel_width", shape_index, asset.wheel_width)
        _set_builder_attr(builder, "wheeled:is_wheel_body", body_index, True)
        _set_builder_attr(builder, "wheeled:wheel_body_id", body_index, wheel_id)
        wheels.append(
            WheelMetadata(
                wheel_id=wheel_id,
                vehicle_id=vehicle_id,
                body_index=body_index,
                shape_index=shape_index,
                radius=asset.wheel_radius,
                width=asset.wheel_width,
            )
        )

    return wheels


def apply_wheeled_manifest(
    builder: ModelBuilder,
    manifest_path: str | Path,
    *,
    asset_names: Sequence[str] | None = None,
) -> list[WheelMetadata]:
    """Apply manifest metadata for one or more already-imported fixture assets.

    Args:
        builder: Builder containing imported fixture assets.
        manifest_path: Wheeled fixture manifest path.
        asset_names: Optional manifest asset names to apply.

    Returns:
        Resolved flat wheel metadata rows.
    """

    assets = load_wheeled_manifest(manifest_path)
    if asset_names is not None:
        requested = tuple(asset_names)
        requested_set = set(requested)
        assets_by_name = {asset.name: asset for asset in assets}
        missing = sorted(requested_set.difference(assets_by_name))
        if missing:
            raise ValueError(f"wheeled manifest does not contain requested assets: {missing}")
        assets = tuple(assets_by_name[name] for name in requested)

    wheels: list[WheelMetadata] = []
    for vehicle_id, asset in enumerate(assets):
        wheels.extend(
            apply_wheeled_manifest_metadata(
                builder,
                asset,
                vehicle_id=vehicle_id,
                wheel_id_start=len(wheels),
            )
        )
    return wheels


def read_wheeled_metadata(model: Model) -> tuple[WheelMetadata, ...]:
    """Read finalized `model.wheeled` attributes into wheel metadata rows.

    Args:
        model: Finalized model with registered wheeled custom attributes.

    Returns:
        Wheel metadata rows sorted by wheel id.

    Raises:
        ValueError: If authored model attributes are inconsistent.
    """

    namespace = getattr(model, _WHEELED_NAMESPACE, None)
    if namespace is None:
        raise ValueError("model does not contain wheeled custom attributes")

    is_wheel = _as_numpy(getattr(namespace, "is_wheel", None), "wheeled:is_wheel")
    wheel_ids = _as_numpy(getattr(namespace, "wheel_id", None), "wheeled:wheel_id")
    vehicle_ids = _as_numpy(getattr(namespace, "vehicle_id", None), "wheeled:vehicle_id")
    wheel_radius = _as_numpy(getattr(namespace, "wheel_radius", None), "wheeled:wheel_radius")
    wheel_width = _as_numpy(getattr(namespace, "wheel_width", None), "wheeled:wheel_width")
    is_wheel_body = _as_numpy(getattr(namespace, "is_wheel_body", None), "wheeled:is_wheel_body")
    wheel_body_ids = _as_numpy(getattr(namespace, "wheel_body_id", None), "wheeled:wheel_body_id")
    shape_body = _as_numpy(model.shape_body, "shape_body")

    body_by_wheel_id: dict[int, int] = {}
    for body_index, is_body in enumerate(is_wheel_body):
        body_wheel_id = int(wheel_body_ids[body_index])
        if bool(is_body):
            if body_wheel_id < 0:
                raise ValueError(f"wheel body {body_index} has negative wheeled:wheel_body_id")
            previous = body_by_wheel_id.get(body_wheel_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate wheeled:wheel_body_id {body_wheel_id} on bodies {previous} and {body_index}"
                )
            body_by_wheel_id[body_wheel_id] = body_index
        elif body_wheel_id >= 0:
            raise ValueError(f"body {body_index} has wheeled:wheel_body_id without wheeled:is_wheel_body")

    rows: list[WheelMetadata] = []
    seen_wheel_ids: set[int] = set()
    for shape_index, is_shape_wheel in enumerate(is_wheel):
        shape_wheel_id = int(wheel_ids[shape_index])
        if not bool(is_shape_wheel):
            if shape_wheel_id >= 0:
                raise ValueError(f"shape {shape_index} has wheeled:wheel_id without wheeled:is_wheel")
            continue

        if shape_wheel_id < 0:
            raise ValueError(f"wheel shape {shape_index} has negative wheeled:wheel_id")
        if shape_wheel_id in seen_wheel_ids:
            raise ValueError(f"duplicate wheeled:wheel_id {shape_wheel_id}")
        seen_wheel_ids.add(shape_wheel_id)

        vehicle_id = int(vehicle_ids[shape_index])
        if vehicle_id < 0:
            raise ValueError(f"wheel shape {shape_index} has negative wheeled:vehicle_id")
        radius = float(wheel_radius[shape_index])
        width = float(wheel_width[shape_index])
        if radius <= 0.0:
            raise ValueError(f"wheel shape {shape_index} has non-positive wheeled:wheel_radius")
        if width <= 0.0:
            raise ValueError(f"wheel shape {shape_index} has non-positive wheeled:wheel_width")

        body_index = int(shape_body[shape_index])
        expected_body_index = body_by_wheel_id.get(shape_wheel_id)
        if expected_body_index is None:
            raise ValueError(f"wheel shape {shape_index} has no body with matching wheeled:wheel_body_id")
        if expected_body_index != body_index:
            raise ValueError(
                f"wheel shape {shape_index} has wheeled:wheel_id {shape_wheel_id} but shape_body is {body_index} "
                f"and wheeled:wheel_body_id is on body {expected_body_index}"
            )

        rows.append(
            WheelMetadata(
                wheel_id=shape_wheel_id,
                vehicle_id=vehicle_id,
                body_index=body_index,
                shape_index=shape_index,
                radius=radius,
                width=width,
            )
        )

    return tuple(_sorted_contiguous_rows(rows))


def build_wheeled_metadata(model: Model, wheel_metadata: Sequence[WheelMetadata] | None = None) -> WheeledModelMetadata:
    """Build deterministic host-side wheel metadata tables.

    Args:
        model: Finalized Newton model.
        wheel_metadata: Optional resolved rows. If omitted, rows are read from
            finalized `model.wheeled` custom attributes.

    Returns:
        Host-side wheeled model metadata.
    """

    rows = read_wheeled_metadata(model) if wheel_metadata is None else tuple(wheel_metadata)
    sorted_rows = tuple(_sorted_contiguous_rows(rows))

    shape_count = int(getattr(model, "shape_count", len(getattr(model, "shape_label", ()))))
    body_count = int(getattr(model, "body_count", len(getattr(model, "body_label", ()))))
    for row in sorted_rows:
        if row.shape_index < 0 or row.shape_index >= shape_count:
            raise ValueError(f"wheel {row.wheel_id} has invalid shape_index {row.shape_index}")
        if row.body_index < 0 or row.body_index >= body_count:
            raise ValueError(f"wheel {row.wheel_id} has invalid body_index {row.body_index}")
        if row.vehicle_id < 0:
            raise ValueError(f"wheel {row.wheel_id} has negative vehicle_id {row.vehicle_id}")
        if row.radius <= 0.0:
            raise ValueError(f"wheel {row.wheel_id} has non-positive radius")
        if row.width <= 0.0:
            raise ValueError(f"wheel {row.wheel_id} has non-positive width")

    vehicle_count = max((row.vehicle_id for row in sorted_rows), default=-1) + 1
    vehicle_wheel_counts = [0] * vehicle_count
    for row in sorted_rows:
        vehicle_wheel_counts[row.vehicle_id] += 1

    return WheeledModelMetadata(
        wheel_count=len(sorted_rows),
        vehicle_count=vehicle_count,
        wheel_shape_indices=tuple(row.shape_index for row in sorted_rows),
        wheel_body_indices=tuple(row.body_index for row in sorted_rows),
        wheel_vehicle_ids=tuple(row.vehicle_id for row in sorted_rows),
        wheel_radius=tuple(row.radius for row in sorted_rows),
        wheel_width=tuple(row.width for row in sorted_rows),
        vehicle_wheel_counts=tuple(vehicle_wheel_counts),
    )


def _require_str(raw: dict[str, Any], key: str, asset_name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"asset {asset_name} requires non-empty {key}")
    return value


def _require_str_tuple(raw: dict[str, Any], key: str, asset_name: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"asset {asset_name} requires list {key}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"asset {asset_name} {key} entries must be non-empty strings")
        out.append(item)
    return tuple(out)


def _optional_str_tuple(raw: dict[str, Any], key: str, asset_name: str) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"asset {asset_name} {key} must be a list when provided")
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


def _require_registered(builder: ModelBuilder) -> None:
    missing = sorted(
        key
        for key in (
            _VEHICLE_INDEX_ATTR,
            _WHEEL_INDEX_ATTR,
            "wheeled:is_wheel",
            "wheeled:wheel_id",
            "wheeled:vehicle_id",
            "wheeled:wheel_radius",
            "wheeled:wheel_width",
            "wheeled:is_wheel_body",
            "wheeled:wheel_body_id",
        )
        if key not in builder.custom_attributes
    )
    if missing:
        raise ValueError(f"wheeled custom attributes are not registered: {missing}")


def _reserve_custom_frequency_rows(
    builder: ModelBuilder,
    *,
    frequency: str,
    index_attribute: str,
    count: int,
) -> None:
    if count <= 0:
        return
    _require_registered(builder)
    current = int(builder._custom_frequency_counts.get(frequency, 0))
    for index in range(current, count):
        builder.add_custom_values(**{index_attribute: index})


def _usd_is_vehicle_prim(prim: Any, _context: dict[str, Any]) -> bool:
    return _usd_has_true_attribute(prim, "newton:wheeled:is_vehicle")


def _usd_is_wheel_prim(prim: Any, _context: dict[str, Any]) -> bool:
    return _usd_has_true_attribute(prim, "newton:wheeled:is_wheel")


def _usd_has_true_attribute(prim: Any, attribute_name: str) -> bool:
    if not prim or prim.IsPseudoRoot():
        return False
    attr = prim.GetAttribute(attribute_name)
    if not attr or not attr.HasAuthoredValueOpinion():
        return False
    return bool(attr.Get())


def _label_lookup(labels: Sequence[str], asset_name: str, label_kind: str) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, label in enumerate(labels):
        if label in lookup:
            raise ValueError(f"asset {asset_name} has duplicate {label_kind} label in builder: {label}")
        lookup[label] = index
    return lookup


def _set_builder_attr(builder: ModelBuilder, key: str, index: int, value: object) -> None:
    attr = builder.custom_attributes[key]
    if attr.values is None:
        attr.values = {}
    attr.values[index] = value


def _as_numpy(value: object, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing {name}")
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _sorted_contiguous_rows(rows: Sequence[WheelMetadata]) -> tuple[WheelMetadata, ...]:
    sorted_rows = tuple(sorted(rows, key=lambda row: row.wheel_id))
    for expected_id, row in enumerate(sorted_rows):
        if row.wheel_id != expected_id:
            raise ValueError(
                f"wheeled metadata wheel ids must be contiguous from 0; expected {expected_id}, got {row.wheel_id}"
            )
    return sorted_rows
