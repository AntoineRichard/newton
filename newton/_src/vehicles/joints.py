# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint helpers for wheeled vehicles.

Analytical wheel spin (see :mod:`newton._src.vehicles.wheel`) represents wheel
rotation with tire control inputs rather than a physical revolute axle DOF. When
an imported asset carries physical axle (wheel-spin) joints they must be locked,
otherwise a free axle spins instead of staying rigid and pollutes the
contact-point velocity used for slip. :func:`configure_wheel_axle_joints`
converts those revolute joints to fixed joints on the builder (with the required
coordinate/DOF/constraint index remapping) before finalization.
"""

from __future__ import annotations

from collections.abc import Sequence

from newton import JointType, ModelBuilder

from ._compat import set_joint_type_fixed


def configure_wheel_axle_joints(
    builder: ModelBuilder,
    *,
    axle_joint_labels: Sequence[str] | None = None,
    wheel_body_labels: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Convert wheel axle joints on a builder to fixed joints.

    Intended for analytical tire-model setups where wheel spin is represented by
    tire control inputs instead of physical revolute axle DOFs. Must be called
    before :meth:`ModelBuilder.finalize`. Selected joints are converted to
    :class:`JointType.FIXED`; their revolute coordinate/DOF data is removed and
    their constraint data is replaced with fixed-joint constraints.

    Args:
        builder: Builder containing imported wheeled assets.
        axle_joint_labels: Explicit axle joint labels to convert.
        wheel_body_labels: Optional wheel body labels. When explicit axle joint
            labels are omitted, the helper converts revolute joints whose child
            body is one of these wheel bodies.

    Returns:
        Tuple of converted joint indices.

    Raises:
        ValueError: If labels are missing, no joints can be resolved, or a
            selected joint is not a revolute joint.
    """
    joint_indices = _resolve_axle_joint_indices(
        builder,
        axle_joint_labels=axle_joint_labels,
        wheel_body_labels=wheel_body_labels,
    )
    _validate_joints_convertible(builder, joint_indices)
    for joint_index in joint_indices:
        set_joint_type_fixed(builder, joint_index)
    return joint_indices


def _validate_joints_convertible(builder: ModelBuilder, joint_indices: Sequence[int]) -> None:
    """Validate all selected joints up front so no joint is converted on error."""
    for joint_index in joint_indices:
        if joint_index < 0 or joint_index >= len(builder.joint_type):
            raise ValueError(f"axle joint index {joint_index} is out of range")
        joint_type = builder.joint_type[joint_index]
        if joint_type not in (JointType.FIXED, JointType.REVOLUTE):
            raise ValueError(
                f"axle joint {joint_index} ('{builder.joint_label[joint_index]}') must be revolute or fixed, "
                f"got {joint_type}"
            )


def _resolve_axle_joint_indices(
    builder: ModelBuilder,
    *,
    axle_joint_labels: Sequence[str] | None,
    wheel_body_labels: Sequence[str] | None,
) -> tuple[int, ...]:
    if axle_joint_labels is not None:
        if not axle_joint_labels:
            raise ValueError("axle_joint_labels must not be empty")
        joint_by_label = _unique_label_lookup(builder.joint_label, "joint")
        indices = []
        for label in axle_joint_labels:
            try:
                indices.append(joint_by_label[label])
            except KeyError as exc:
                raise ValueError(f"missing axle joint label: {label}") from exc
        return _deduplicate_indices(indices, "axle_joint_labels")

    if wheel_body_labels is None:
        raise ValueError("configure_wheel_axle_joints requires axle_joint_labels or wheel_body_labels")
    if not wheel_body_labels:
        raise ValueError("wheel_body_labels must not be empty")

    body_by_label = _unique_label_lookup(builder.body_label, "body")
    wheel_body_indices = set()
    for label in wheel_body_labels:
        try:
            wheel_body_indices.add(body_by_label[label])
        except KeyError as exc:
            raise ValueError(f"missing wheel body label: {label}") from exc

    indices = [
        joint_index
        for joint_index, child in enumerate(builder.joint_child)
        if child in wheel_body_indices and builder.joint_type[joint_index] == JointType.REVOLUTE
    ]
    return _deduplicate_indices(indices, "wheel_body_labels")


def _unique_label_lookup(labels: Sequence[str], entity_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, label in enumerate(labels):
        label_text = str(label)
        if label_text in out:
            raise ValueError(f"duplicate {entity_name} label: {label_text}")
        out[label_text] = index
    return out


def _deduplicate_indices(indices: Sequence[int], source_name: str) -> tuple[int, ...]:
    if not indices:
        raise ValueError(f"{source_name} did not resolve any axle joints")
    out = tuple(dict.fromkeys(int(index) for index in indices))
    if len(out) != len(indices):
        raise ValueError(f"{source_name} resolved duplicate axle joints")
    return out
