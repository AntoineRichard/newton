# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Forward-compatibility shims over pending newton public APIs.

The wheeled-vehicle layer needs exactly two builder capabilities that newton
does not expose publicly yet:

* :func:`get_custom_frequency_count` — read how many rows a custom attribute
  frequency currently has (proposed upstream in newton-physics/newton#3361).
* :func:`set_joint_type_fixed` — convert an existing builder joint to
  :class:`newton.JointType.FIXED` before finalization (proposed upstream in
  newton-physics/newton#3362).

Each shim prefers the public :class:`newton.ModelBuilder` method when it
exists, so this module upgrades automatically once the referenced upstream
issues land — at which point the fallback bodies (and eventually this whole
module) can be deleted. Until then, the fallbacks below are the only places in
the package that touch newton private API (``builder._custom_frequency_counts``)
or vendor builder array surgery.
"""

from __future__ import annotations

from typing import Any

from newton import JointType, Model, ModelBuilder


def get_custom_frequency_count(builder: ModelBuilder, frequency: str) -> int:
    """Return the number of rows registered for a custom attribute frequency.

    Uses ``builder.get_custom_frequency_count`` when available (upstream
    newton-physics/newton#3361); otherwise falls back to reading the private
    ``_custom_frequency_counts`` mapping.
    """
    if hasattr(builder, "get_custom_frequency_count"):
        return int(builder.get_custom_frequency_count(frequency))
    return int(builder._custom_frequency_counts.get(frequency, 0))


def set_joint_type_fixed(builder: ModelBuilder, joint: int) -> None:
    """Convert a builder joint to :class:`newton.JointType.FIXED`.

    Uses ``builder.set_joint_type`` when available (upstream
    newton-physics/newton#3362); otherwise runs a vendored fallback that
    rewrites the builder joint arrays directly (removing the joint's
    coordinate/DOF data, replacing its constraint data with fixed-joint
    constraints, and remapping custom-attribute indices).

    Args:
        builder: Builder containing the joint. Must not be finalized yet.
        joint: Joint index. Must reference a revolute or already-fixed joint.

    Raises:
        ValueError: If the joint index is out of range or the joint is neither
            revolute nor fixed.
    """
    if hasattr(builder, "set_joint_type"):
        builder.set_joint_type(joint, JointType.FIXED)
        return
    _convert_joint_to_fixed_fallback(builder, int(joint))


def _convert_joint_to_fixed_fallback(builder: ModelBuilder, joint_index: int) -> None:
    if joint_index < 0 or joint_index >= len(builder.joint_type):
        raise ValueError(f"axle joint index {joint_index} is out of range")
    joint_type = builder.joint_type[joint_index]
    if joint_type == JointType.FIXED:
        return
    if joint_type != JointType.REVOLUTE:
        raise ValueError(
            f"axle joint {joint_index} ('{builder.joint_label[joint_index]}') must be revolute or fixed, "
            f"got {joint_type}"
        )

    selected = {joint_index}

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

    for index in range(len(builder.joint_type)):
        q_start, q_end = _joint_slice(old_q_start, len(old_q), index)
        qd_start, qd_end = _joint_slice(old_qd_start, len(old_qd), index)
        cts_start, cts_end = _joint_slice(old_cts_start, len(old_cts), index)

        new_q_start.append(len(new_q))
        new_qd_start.append(len(new_qd))
        new_cts_start.append(len(new_cts))

        if index in selected:
            builder.joint_type[index] = JointType.FIXED
            builder.joint_dof_dim[index] = (0, 0)
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


def _joint_slice(starts: list[int], total: int, joint_index: int) -> tuple[int, int]:
    start = int(starts[joint_index])
    end = int(starts[joint_index + 1]) if joint_index + 1 < len(starts) else int(total)
    return start, end


def _remap_custom_frequency(builder: ModelBuilder, frequency: Model.AttributeFrequency, remap: dict[int, int]) -> None:
    for custom_attr in builder.get_custom_attributes_by_frequency([frequency]):
        values = custom_attr.values
        if not isinstance(values, dict):
            continue
        custom_attr.values = {remap[old_index]: value for old_index, value in values.items() if old_index in remap}
