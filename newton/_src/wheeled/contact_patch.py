# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Contacts, Model, State

from .metadata import WheeledModelMetadata


class WheelContactPatchState:
    """Wheel-indexed reduction of Newton rigid contacts.

    The object is a reusable per-wheel view over the latest :class:`Contacts`
    buffers. It does not replace or mutate collision contacts; call
    :func:`update_wheel_contact_patches` after collision generation to refresh
    the arrays.

    Args:
        model: Model that owns the contact and wheel shape indices.
        wheeled_metadata: Phase 1A wheel metadata used to size and map wheels.

    Attributes:
        active: Active contact flag per wheel, shape ``(wheel_count,)``.
        contact_count: Number of active rigid contacts per wheel, shape
            ``(wheel_count,)``.
        terrain_shape_index: Counterpart shape index per wheel, or ``-1``,
            shape ``(wheel_count,)``.
        center: Estimated contact patch center [m], shape ``(wheel_count,)``.
        normal: Estimated support normal acting on the wheel, shape
            ``(wheel_count,)``.
        patch_u_extent: Contact cloud extent along the first tangent [m], shape
            ``(wheel_count,)``.
        patch_v_extent: Contact cloud extent along the second tangent [m], shape
            ``(wheel_count,)``.
        patch_area: Estimated contact patch area [m^2], shape
            ``(wheel_count,)``.
        friction_mu_seed: Counterpart shape friction seed, shape
            ``(wheel_count,)``.
        normal_force: Optional accumulated normal-force diagnostic [N], shape
            ``(wheel_count,)``.
    """

    def __init__(self, model: Model, wheeled_metadata: WheeledModelMetadata):
        self.wheel_count = int(wheeled_metadata.wheel_count)
        self.shape_count = int(model.shape_count)
        self.device = model.device
        self.wheel_shape_indices = tuple(int(index) for index in wheeled_metadata.wheel_shape_indices)
        self._wheeled_metadata = wheeled_metadata

        if len(self.wheel_shape_indices) != self.wheel_count:
            raise ValueError(
                "wheeled metadata wheel_shape_indices length must match wheel_count "
                f"({len(self.wheel_shape_indices)} != {self.wheel_count})"
            )

        shape_wheel_ids = np.full(self.shape_count, -1, dtype=np.int32)
        for wheel_id, shape_index in enumerate(self.wheel_shape_indices):
            if shape_index < 0 or shape_index >= self.shape_count:
                raise ValueError(f"wheel {wheel_id} has invalid shape index {shape_index}")
            if shape_wheel_ids[shape_index] >= 0:
                raise ValueError(f"shape {shape_index} is assigned to more than one wheel")
            shape_wheel_ids[shape_index] = wheel_id

        with wp.ScopedDevice(self.device):
            self.active = wp.zeros(self.wheel_count, dtype=wp.bool)
            self.contact_count = wp.zeros(self.wheel_count, dtype=wp.int32)
            self.terrain_shape_index = wp.full(self.wheel_count, -1, dtype=wp.int32)
            self.center = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self.normal = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self.patch_u_extent = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.patch_v_extent = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.patch_area = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.friction_mu_seed = wp.zeros(self.wheel_count, dtype=wp.float32)
            self.normal_force = wp.zeros(self.wheel_count, dtype=wp.float32)

            self._shape_wheel_ids = wp.array(shape_wheel_ids, dtype=wp.int32)
            self._point_sum = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self._normal_sum = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self._terrain_shape_min = wp.full(self.wheel_count, self.shape_count, dtype=wp.int32)
            self._tangent_u = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self._tangent_v = wp.zeros(self.wheel_count, dtype=wp.vec3)
            self._u_min = wp.full(self.wheel_count, 1.0e20, dtype=wp.float32)
            self._u_max = wp.full(self.wheel_count, -1.0e20, dtype=wp.float32)
            self._v_min = wp.full(self.wheel_count, 1.0e20, dtype=wp.float32)
            self._v_max = wp.full(self.wheel_count, -1.0e20, dtype=wp.float32)
            self._empty_force = wp.zeros(1, dtype=wp.spatial_vector)

    def clear(self) -> None:
        """Reset all public diagnostics and internal reduction buffers."""
        if self.wheel_count == 0:
            return

        wp.launch(
            kernel=_clear_wheel_contact_patch_state,
            dim=self.wheel_count,
            inputs=[self.shape_count],
            outputs=[
                self.active,
                self.contact_count,
                self.terrain_shape_index,
                self.center,
                self.normal,
                self.patch_u_extent,
                self.patch_v_extent,
                self.patch_area,
                self.friction_mu_seed,
                self.normal_force,
                self._point_sum,
                self._normal_sum,
                self._terrain_shape_min,
                self._tangent_u,
                self._tangent_v,
                self._u_min,
                self._u_max,
                self._v_min,
                self._v_max,
            ],
            device=self.device,
        )

    def _validate_update_inputs(self, model: Model, wheeled_metadata: WheeledModelMetadata) -> None:
        if int(model.shape_count) != self.shape_count:
            raise ValueError(
                f"patch state shape_count {self.shape_count} does not match model shape_count {model.shape_count}"
            )
        if int(wheeled_metadata.wheel_count) != self.wheel_count:
            raise ValueError(
                "patch state wheel_count does not match wheeled metadata wheel_count "
                f"({self.wheel_count} != {wheeled_metadata.wheel_count})"
            )
        if wheeled_metadata is not self._wheeled_metadata:
            raise ValueError("patch state must be updated with the wheeled metadata used to construct it")


def update_wheel_contact_patches(
    model: Model,
    state: State,
    contacts: Contacts,
    wheeled_metadata: WheeledModelMetadata,
    patch_state: WheelContactPatchState,
) -> None:
    """Update wheel contact patch diagnostics from Newton rigid contacts.

    Args:
        model: Model that owns the shapes and material fields.
        state: Current simulation state.
        contacts: Newton contact buffers populated by collision generation.
        wheeled_metadata: Phase 1A wheel metadata used for wheel identity.
        patch_state: Destination wheel-indexed contact patch state.
    """

    patch_state._validate_update_inputs(model, wheeled_metadata)
    patch_state.clear()

    if patch_state.wheel_count == 0 or contacts.rigid_contact_max == 0:
        return

    force = contacts.force if contacts.force is not None else patch_state._empty_force
    force_available = contacts.force is not None

    wp.launch(
        kernel=_accumulate_wheel_contact_patches,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            model.shape_body,
            state.body_q,
            patch_state._shape_wheel_ids,
            patch_state.shape_count,
            force,
            force_available,
        ],
        outputs=[
            patch_state.contact_count,
            patch_state._point_sum,
            patch_state._normal_sum,
            patch_state._terrain_shape_min,
            patch_state.normal_force,
        ],
        device=patch_state.device,
    )

    wp.launch(
        kernel=_prepare_wheel_contact_patch_state,
        dim=patch_state.wheel_count,
        inputs=[patch_state.shape_count, model.shape_material_mu],
        outputs=[
            patch_state.active,
            patch_state.contact_count,
            patch_state.terrain_shape_index,
            patch_state.center,
            patch_state.normal,
            patch_state.friction_mu_seed,
            patch_state._point_sum,
            patch_state._normal_sum,
            patch_state._terrain_shape_min,
            patch_state._tangent_u,
            patch_state._tangent_v,
        ],
        device=patch_state.device,
    )

    wp.launch(
        kernel=_accumulate_wheel_contact_patch_extents,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            model.shape_body,
            state.body_q,
            patch_state._shape_wheel_ids,
            patch_state.shape_count,
            patch_state.center,
            patch_state._tangent_u,
            patch_state._tangent_v,
        ],
        outputs=[
            patch_state._u_min,
            patch_state._u_max,
            patch_state._v_min,
            patch_state._v_max,
        ],
        device=patch_state.device,
    )

    wp.launch(
        kernel=_finalize_wheel_contact_patch_extents,
        dim=patch_state.wheel_count,
        inputs=[patch_state.contact_count],
        outputs=[
            patch_state.patch_u_extent,
            patch_state.patch_v_extent,
            patch_state.patch_area,
            patch_state._u_min,
            patch_state._u_max,
            patch_state._v_min,
            patch_state._v_max,
        ],
        device=patch_state.device,
    )


@wp.func
def _contact_wheel_id(
    shape0: int,
    shape1: int,
    shape_wheel_ids: wp.array[wp.int32],
    shape_count: int,
) -> int:
    wheel0 = wp.int32(-1)
    wheel1 = wp.int32(-1)
    if shape0 >= 0 and shape0 < shape_count:
        wheel0 = shape_wheel_ids[shape0]
    if shape1 >= 0 and shape1 < shape_count:
        wheel1 = shape_wheel_ids[shape1]

    if wheel0 >= 0 and wheel1 >= 0:
        if wheel0 <= wheel1:
            return wheel0
        return wheel1
    if wheel0 >= 0:
        return wheel0
    return wheel1


@wp.func
def _contact_wheel_is_shape0(
    shape0: int,
    shape1: int,
    shape_wheel_ids: wp.array[wp.int32],
    shape_count: int,
    wheel_id: int,
) -> bool:
    if shape0 >= 0 and shape0 < shape_count:
        if shape_wheel_ids[shape0] == wheel_id:
            return True
    if shape1 >= 0 and shape1 < shape_count:
        if shape_wheel_ids[shape1] == wheel_id:
            return False
    return True


@wp.func
def _shape_point_world(
    shape_index: int,
    point_body: wp.vec3,
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
) -> wp.vec3:
    body_index = shape_body[shape_index]
    if body_index >= 0:
        return wp.transform_point(body_q[body_index], point_body)
    return point_body


@wp.func
def _safe_normalize(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3()


@wp.func
def _patch_tangent_u(normal: wp.vec3) -> wp.vec3:
    if wp.length(normal) <= 1.0e-6:
        return wp.vec3()
    if wp.abs(normal[2]) > 0.9:
        return wp.vec3(1.0, 0.0, 0.0)
    return _safe_normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), normal))


@wp.kernel
def _clear_wheel_contact_patch_state(
    shape_count: int,
    active: wp.array[wp.bool],
    contact_count: wp.array[wp.int32],
    terrain_shape_index: wp.array[wp.int32],
    center: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    patch_u_extent: wp.array[wp.float32],
    patch_v_extent: wp.array[wp.float32],
    patch_area: wp.array[wp.float32],
    friction_mu_seed: wp.array[wp.float32],
    normal_force: wp.array[wp.float32],
    point_sum: wp.array[wp.vec3],
    normal_sum: wp.array[wp.vec3],
    terrain_shape_min: wp.array[wp.int32],
    tangent_u: wp.array[wp.vec3],
    tangent_v: wp.array[wp.vec3],
    u_min: wp.array[wp.float32],
    u_max: wp.array[wp.float32],
    v_min: wp.array[wp.float32],
    v_max: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    active[wheel_id] = False
    contact_count[wheel_id] = wp.int32(0)
    terrain_shape_index[wheel_id] = wp.int32(-1)
    center[wheel_id] = wp.vec3()
    normal[wheel_id] = wp.vec3()
    patch_u_extent[wheel_id] = 0.0
    patch_v_extent[wheel_id] = 0.0
    patch_area[wheel_id] = 0.0
    friction_mu_seed[wheel_id] = 0.0
    normal_force[wheel_id] = 0.0
    point_sum[wheel_id] = wp.vec3()
    normal_sum[wheel_id] = wp.vec3()
    terrain_shape_min[wheel_id] = wp.int32(shape_count)
    tangent_u[wheel_id] = wp.vec3()
    tangent_v[wheel_id] = wp.vec3()
    u_min[wheel_id] = 1.0e20
    u_max[wheel_id] = -1.0e20
    v_min[wheel_id] = 1.0e20
    v_max[wheel_id] = -1.0e20


@wp.kernel
def _accumulate_wheel_contact_patches(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_wheel_ids: wp.array[wp.int32],
    shape_count: int,
    contact_force: wp.array[wp.spatial_vector],
    force_available: bool,
    contact_count: wp.array[wp.int32],
    point_sum: wp.array[wp.vec3],
    normal_sum: wp.array[wp.vec3],
    terrain_shape_min: wp.array[wp.int32],
    normal_force: wp.array[wp.float32],
):
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    shape0 = rigid_contact_shape0[contact_id]
    shape1 = rigid_contact_shape1[contact_id]
    if shape0 < 0 or shape1 < 0 or shape0 >= shape_count or shape1 >= shape_count:
        return

    wheel_id = _contact_wheel_id(shape0, shape1, shape_wheel_ids, shape_count)
    if wheel_id < 0:
        return

    wheel_is_shape0 = _contact_wheel_is_shape0(shape0, shape1, shape_wheel_ids, shape_count, wheel_id)
    wheel_shape = shape1
    terrain_shape = shape0
    wheel_point = rigid_contact_point1[contact_id]
    support_normal = rigid_contact_normal[contact_id]
    if wheel_is_shape0:
        wheel_shape = shape0
        terrain_shape = shape1
        wheel_point = rigid_contact_point0[contact_id]
        support_normal = -rigid_contact_normal[contact_id]

    point_world = _shape_point_world(wheel_shape, wheel_point, shape_body, body_q)
    wp.atomic_add(point_sum, wheel_id, point_world)
    wp.atomic_add(normal_sum, wheel_id, support_normal)
    wp.atomic_add(contact_count, wheel_id, wp.int32(1))
    wp.atomic_min(terrain_shape_min, wheel_id, terrain_shape)

    if force_available:
        force_on_body0 = wp.spatial_top(contact_force[contact_id])
        force_on_wheel = force_on_body0
        if not wheel_is_shape0:
            force_on_wheel = -force_on_body0
        projected = wp.dot(force_on_wheel, support_normal)
        if projected > 0.0:
            wp.atomic_add(normal_force, wheel_id, projected)


@wp.kernel
def _prepare_wheel_contact_patch_state(
    shape_count: int,
    shape_material_mu: wp.array[wp.float32],
    active: wp.array[wp.bool],
    contact_count: wp.array[wp.int32],
    terrain_shape_index: wp.array[wp.int32],
    center: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    friction_mu_seed: wp.array[wp.float32],
    point_sum: wp.array[wp.vec3],
    normal_sum: wp.array[wp.vec3],
    terrain_shape_min: wp.array[wp.int32],
    tangent_u: wp.array[wp.vec3],
    tangent_v: wp.array[wp.vec3],
):
    wheel_id = wp.tid()
    count = contact_count[wheel_id]
    if count <= 0:
        return

    active[wheel_id] = True
    inv_count = 1.0 / wp.float32(count)
    center[wheel_id] = point_sum[wheel_id] * inv_count

    support_normal = _safe_normalize(normal_sum[wheel_id])
    normal[wheel_id] = support_normal
    tangent = _patch_tangent_u(support_normal)
    tangent_u[wheel_id] = tangent
    tangent_v[wheel_id] = _safe_normalize(wp.cross(support_normal, tangent))

    terrain_shape = terrain_shape_min[wheel_id]
    if terrain_shape >= 0 and terrain_shape < shape_count:
        terrain_shape_index[wheel_id] = terrain_shape
        friction_mu_seed[wheel_id] = shape_material_mu[terrain_shape]


@wp.kernel
def _accumulate_wheel_contact_patch_extents(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_wheel_ids: wp.array[wp.int32],
    shape_count: int,
    center: wp.array[wp.vec3],
    tangent_u: wp.array[wp.vec3],
    tangent_v: wp.array[wp.vec3],
    u_min: wp.array[wp.float32],
    u_max: wp.array[wp.float32],
    v_min: wp.array[wp.float32],
    v_max: wp.array[wp.float32],
):
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    shape0 = rigid_contact_shape0[contact_id]
    shape1 = rigid_contact_shape1[contact_id]
    if shape0 < 0 or shape1 < 0 or shape0 >= shape_count or shape1 >= shape_count:
        return

    wheel_id = _contact_wheel_id(shape0, shape1, shape_wheel_ids, shape_count)
    if wheel_id < 0:
        return

    wheel_is_shape0 = _contact_wheel_is_shape0(shape0, shape1, shape_wheel_ids, shape_count, wheel_id)
    wheel_shape = shape1
    wheel_point = rigid_contact_point1[contact_id]
    if wheel_is_shape0:
        wheel_shape = shape0
        wheel_point = rigid_contact_point0[contact_id]

    point_world = _shape_point_world(wheel_shape, wheel_point, shape_body, body_q)
    relative = point_world - center[wheel_id]
    u = wp.dot(relative, tangent_u[wheel_id])
    v = wp.dot(relative, tangent_v[wheel_id])
    wp.atomic_min(u_min, wheel_id, u)
    wp.atomic_max(u_max, wheel_id, u)
    wp.atomic_min(v_min, wheel_id, v)
    wp.atomic_max(v_max, wheel_id, v)


@wp.kernel
def _finalize_wheel_contact_patch_extents(
    contact_count: wp.array[wp.int32],
    patch_u_extent: wp.array[wp.float32],
    patch_v_extent: wp.array[wp.float32],
    patch_area: wp.array[wp.float32],
    u_min: wp.array[wp.float32],
    u_max: wp.array[wp.float32],
    v_min: wp.array[wp.float32],
    v_max: wp.array[wp.float32],
):
    wheel_id = wp.tid()
    if contact_count[wheel_id] <= 0:
        return

    u_extent = 0.0
    v_extent = 0.0
    if u_max[wheel_id] >= u_min[wheel_id]:
        u_extent = u_max[wheel_id] - u_min[wheel_id]
    if v_max[wheel_id] >= v_min[wheel_id]:
        v_extent = v_max[wheel_id] - v_min[wheel_id]

    patch_u_extent[wheel_id] = u_extent
    patch_v_extent[wheel_id] = v_extent
    patch_area[wheel_id] = u_extent * v_extent
