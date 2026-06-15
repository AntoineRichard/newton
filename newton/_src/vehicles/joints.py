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
from typing import Any

from newton._src.sim import JointType, Model, ModelBuilder


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
    _convert_joints_to_fixed(builder, joint_indices)
    return joint_indices


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


def _convert_joints_to_fixed(builder: ModelBuilder, joint_indices: Sequence[int]) -> None:
    selected = {int(index) for index in joint_indices}
    for joint_index in selected:
        if joint_index < 0 or joint_index >= len(builder.joint_type):
            raise ValueError(f"axle joint index {joint_index} is out of range")
        joint_type = builder.joint_type[joint_index]
        if joint_type == JointType.FIXED:
            continue
        if joint_type != JointType.REVOLUTE:
            raise ValueError(
                f"axle joint {joint_index} ('{builder.joint_label[joint_index]}') must be revolute or fixed, "
                f"got {joint_type}"
            )

    old_q_start = list(builder.joint_q_start)
    old_qd_start = list(builder.joint_qd_start)
    old_cts_start = list(builder.joint_cts_start)
    old_q = list(builder.joint_q)
    old_qd = list(builder.joint_qd)
    old_cts = list(builder.joint_cts)
    old_joint_f = list(builder.joint_f)
    old_joint_act = list(builder.joint_act)
    old_axis = list(builder.joint_axis)
    old_target_pos = list(builder.joint_target_pos)
    old_target_vel = list(builder.joint_target_vel)
    old_target_mode = list(builder.joint_target_mode)
    old_target_ke = list(builder.joint_target_ke)
    old_target_kd = list(builder.joint_target_kd)
    old_limit_lower = list(builder.joint_limit_lower)
    old_limit_upper = list(builder.joint_limit_upper)
    old_limit_ke = list(builder.joint_limit_ke)
    old_limit_kd = list(builder.joint_limit_kd)
    old_armature = list(builder.joint_armature)
    old_effort_limit = list(builder.joint_effort_limit)
    old_velocity_limit = list(builder.joint_velocity_limit)
    old_friction = list(builder.joint_friction)

    q_index_remap: dict[int, int] = {}
    qd_index_remap: dict[int, int] = {}
    cts_index_remap: dict[int, int] = {}

    new_q: list[Any] = []
    new_qd: list[Any] = []
    new_cts: list[Any] = []
    new_joint_f: list[Any] = []
    new_joint_act: list[Any] = []
    new_axis: list[Any] = []
    new_target_pos: list[Any] = []
    new_target_vel: list[Any] = []
    new_target_mode: list[Any] = []
    new_target_ke: list[Any] = []
    new_target_kd: list[Any] = []
    new_limit_lower: list[Any] = []
    new_limit_upper: list[Any] = []
    new_limit_ke: list[Any] = []
    new_limit_kd: list[Any] = []
    new_armature: list[Any] = []
    new_effort_limit: list[Any] = []
    new_velocity_limit: list[Any] = []
    new_friction: list[Any] = []
    new_q_start: list[int] = []
    new_qd_start: list[int] = []
    new_cts_start: list[int] = []

    for joint_index in range(len(builder.joint_type)):
        q_start, q_end = _joint_slice(old_q_start, len(old_q), joint_index)
        qd_start, qd_end = _joint_slice(old_qd_start, len(old_qd), joint_index)
        cts_start, cts_end = _joint_slice(old_cts_start, len(old_cts), joint_index)

        new_q_start.append(len(new_q))
        new_qd_start.append(len(new_qd))
        new_cts_start.append(len(new_cts))

        if joint_index in selected:
            builder.joint_type[joint_index] = JointType.FIXED
            builder.joint_dof_dim[joint_index] = (0, 0)
            new_cts.extend(0.0 for _ in range(JointType.FIXED.constraint_count(0)))
            continue

        for old_index in range(q_start, q_end):
            q_index_remap[old_index] = len(new_q)
            new_q.append(old_q[old_index])
        for old_index in range(qd_start, qd_end):
            qd_index_remap[old_index] = len(new_qd)
            new_qd.append(old_qd[old_index])
            new_joint_f.append(old_joint_f[old_index])
            new_joint_act.append(old_joint_act[old_index])
            new_axis.append(old_axis[old_index])
            new_target_pos.append(old_target_pos[old_index])
            new_target_vel.append(old_target_vel[old_index])
            new_target_mode.append(old_target_mode[old_index])
            new_target_ke.append(old_target_ke[old_index])
            new_target_kd.append(old_target_kd[old_index])
            new_limit_lower.append(old_limit_lower[old_index])
            new_limit_upper.append(old_limit_upper[old_index])
            new_limit_ke.append(old_limit_ke[old_index])
            new_limit_kd.append(old_limit_kd[old_index])
            new_armature.append(old_armature[old_index])
            new_effort_limit.append(old_effort_limit[old_index])
            new_velocity_limit.append(old_velocity_limit[old_index])
            new_friction.append(old_friction[old_index])
        for old_index in range(cts_start, cts_end):
            cts_index_remap[old_index] = len(new_cts)
            new_cts.append(old_cts[old_index])

    builder.joint_q = new_q
    builder.joint_qd = new_qd
    builder.joint_cts = new_cts
    builder.joint_f = new_joint_f
    builder.joint_act = new_joint_act
    builder.joint_axis = new_axis
    builder.joint_target_pos = new_target_pos
    builder.joint_target_vel = new_target_vel
    builder.joint_target_mode = new_target_mode
    builder.joint_target_ke = new_target_ke
    builder.joint_target_kd = new_target_kd
    builder.joint_limit_lower = new_limit_lower
    builder.joint_limit_upper = new_limit_upper
    builder.joint_limit_ke = new_limit_ke
    builder.joint_limit_kd = new_limit_kd
    builder.joint_armature = new_armature
    builder.joint_effort_limit = new_effort_limit
    builder.joint_velocity_limit = new_velocity_limit
    builder.joint_friction = new_friction
    builder.joint_q_start = new_q_start
    builder.joint_qd_start = new_qd_start
    builder.joint_cts_start = new_cts_start
    builder.joint_dof_count = len(new_qd)
    builder.joint_coord_count = len(new_q)
    builder.joint_constraint_count = len(new_cts)

    _remap_custom_frequency(builder, Model.AttributeFrequency.JOINT_COORD, q_index_remap)
    _remap_custom_frequency(builder, Model.AttributeFrequency.JOINT_DOF, qd_index_remap)
    _remap_custom_frequency(builder, Model.AttributeFrequency.JOINT_CONSTRAINT, cts_index_remap)


def _joint_slice(starts: Sequence[int], total: int, joint_index: int) -> tuple[int, int]:
    start = int(starts[joint_index])
    end = int(starts[joint_index + 1]) if joint_index + 1 < len(starts) else int(total)
    return start, end


def _remap_custom_frequency(builder: ModelBuilder, frequency: Model.AttributeFrequency, remap: dict[int, int]) -> None:
    for custom_attr in builder.get_custom_attributes_by_frequency([frequency]):
        values = custom_attr.values
        if not isinstance(values, dict):
            continue
        custom_attr.values = {remap[old_index]: value for old_index, value in values.items() if old_index in remap}
