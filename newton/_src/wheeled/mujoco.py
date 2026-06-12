# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-specific helpers for wheeled vehicles."""

from __future__ import annotations

from newton._src.sim import Model

from .metadata import WheeledModelMetadata


def configure_mujoco_wheel_contacts(
    model: Model,
    wheeled_metadata: WheeledModelMetadata,
    *,
    condim: int = 1,
    priority: int = 1,
) -> None:
    """Configure MuJoCo contact parameters for wheel shapes.

    The helper sets the MuJoCo contact dimension and geom priority on all wheel
    shapes referenced by the wheeled metadata. The default values make wheel
    contacts normal-only when the opposing terrain uses the default lower
    priority.

    Args:
        model: Finalized model containing MuJoCo custom attributes.
        wheeled_metadata: Wheel metadata for `model`.
        condim: MuJoCo contact dimensionality for wheel geoms.
        priority: MuJoCo geom priority for wheel geoms.

    Raises:
        ValueError: If `model` does not contain the required MuJoCo attributes
            or if wheel shape metadata is inconsistent with the model.
    """

    if condim < 1:
        raise ValueError(f"MuJoCo wheel contact condim must be positive, got {condim}")
    if priority < 0:
        raise ValueError(f"MuJoCo wheel geom priority must be non-negative, got {priority}")

    if not hasattr(model, "mujoco") or not hasattr(model.mujoco, "condim"):
        raise ValueError(
            "model does not have mujoco:condim attributes; call "
            "SolverMuJoCo.register_custom_attributes(builder) before finalizing the model"
        )
    if not hasattr(model.mujoco, "geom_priority"):
        raise ValueError(
            "model does not have mujoco:geom_priority attributes; call "
            "SolverMuJoCo.register_custom_attributes(builder) before finalizing the model"
        )

    wheel_shape_indices = tuple(int(index) for index in wheeled_metadata.wheel_shape_indices)
    if len(wheel_shape_indices) != int(wheeled_metadata.wheel_count):
        raise ValueError(
            "wheeled metadata wheel_shape_indices length must match wheel_count "
            f"({len(wheel_shape_indices)} != {wheeled_metadata.wheel_count})"
        )

    shape_count = int(model.shape_count)
    for wheel_id, shape_index in enumerate(wheel_shape_indices):
        if shape_index < 0 or shape_index >= shape_count:
            raise ValueError(f"wheel {wheel_id} has invalid shape index {shape_index}")

    condim_values = model.mujoco.condim.numpy()
    priority_values = model.mujoco.geom_priority.numpy()
    for shape_index in wheel_shape_indices:
        condim_values[shape_index] = condim
        priority_values[shape_index] = priority
    model.mujoco.condim.assign(condim_values)
    model.mujoco.geom_priority.assign(priority_values)
