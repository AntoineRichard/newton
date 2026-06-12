# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-wheel contact-patch extraction from Newton rigid contacts.

The wrapped solver owns collision and normal support; this module reads the
Newton ``Contacts`` (never mutates them) and reduces them to one
:class:`WheelContactPatch` row per wheel: contact center, support normal,
tangent footprint, terrain shape/material seed, and the solver-reported normal
load latched for the tire model. All work is batched Warp kernels.
"""

from __future__ import annotations

import warp as wp

from .metadata import VehicleModelData

_BIG_F = 1.0e20
_BIG_I = 1 << 30


@wp.func
def _safe_normalize(v: wp.vec3) -> wp.vec3:
    n = wp.length(v)
    if n > 1.0e-9:
        return v / n
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _tangent_u(n: wp.vec3) -> wp.vec3:
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    return _safe_normalize(a - n * wp.dot(a, n))


@wp.func
def _shape_point_world(
    shape: int, point: wp.vec3, shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform]
) -> wp.vec3:
    body = shape_body[shape]
    if body >= 0:
        return wp.transform_point(body_q[body], point)
    return point


class WheelContactPatch:
    """Per-wheel contact-patch state as flat device arrays (length ``wheel_count``).

    Args:
        wheel_count: Number of wheels.
        device: Warp device for the arrays.

    Attributes:
        active: 1 if the wheel has at least one contact this step.
        contact_count: Number of contacts grouped to the wheel.
        terrain_shape: Counterpart terrain shape index, or -1.
        center: World-space patch center [m].
        normal: Unit support normal acting on the wheel.
        tangent_u, tangent_v: Orthonormal tangent basis of the patch.
        tangent_extent: Footprint extents (u, v) [m].
        area: Estimated patch area [m²].
        normal_load: Solver-reported normal load measured this step [N] (diagnostic).
        friction_seed: Terrain ``shape_material_mu``.
        fz: Latched normal load [N] used by the tire model (persists across
            :func:`update_wheel_contact_patches`; written by :func:`latch_wheel_loads`).
    """

    def __init__(self, wheel_count: int, device: wp.context.Devicelike | None = None):
        self.wheel_count = wheel_count
        n = max(int(wheel_count), 1)
        self.device = wp.get_device(device)

        def z(dt):
            return wp.zeros(n, dtype=dt, device=self.device)

        self.active = z(wp.bool)
        self.contact_count = z(wp.int32)
        self.terrain_shape = wp.full(n, -1, dtype=wp.int32, device=self.device)
        self.center = z(wp.vec3)
        self.normal = z(wp.vec3)
        self.tangent_u = z(wp.vec3)
        self.tangent_v = z(wp.vec3)
        self.tangent_extent = z(wp.vec2)
        self.area = z(wp.float32)
        self.normal_load = z(wp.float32)
        self.friction_seed = z(wp.float32)
        self.fz = z(wp.float32)
        # scratch accumulators
        self._point_sum = z(wp.vec3)
        self._normal_sum = z(wp.vec3)
        self._terrain_min = z(wp.int32)
        self._u_min = z(wp.float32)
        self._u_max = z(wp.float32)
        self._v_min = z(wp.float32)
        self._v_max = z(wp.float32)


def update_wheel_contact_patches(model, state, contacts, data: VehicleModelData, patch: WheelContactPatch) -> None:
    """Extract per-wheel contact patches from ``contacts`` (geometry only).

    Reads the Newton rigid contacts and ``state.body_q`` to compute, per wheel,
    the contact center, support normal, tangent footprint, terrain shape, and
    material seed. Does not touch ``patch.fz`` (see :func:`latch_wheel_loads`).

    Args:
        model: Finalized model.
        state: State whose ``body_q`` matches the contacts (the collide state).
        contacts: Newton contacts populated by ``model.collide``.
        data: Vehicle tables.
        patch: Patch state to update in place.
    """
    if data.wheel_count == 0:
        return
    patch.active.zero_()
    patch.contact_count.zero_()
    patch.terrain_shape.fill_(-1)
    patch.center.zero_()
    patch.normal.zero_()
    patch.tangent_u.zero_()
    patch.tangent_v.zero_()
    patch.tangent_extent.zero_()
    patch.area.zero_()
    patch.normal_load.zero_()
    patch.friction_seed.zero_()
    patch._point_sum.zero_()
    patch._normal_sum.zero_()
    patch._terrain_min.fill_(_BIG_I)
    patch._u_min.fill_(_BIG_F)
    patch._u_max.fill_(-_BIG_F)
    patch._v_min.fill_(_BIG_F)
    patch._v_max.fill_(-_BIG_F)

    n_contacts = contacts.rigid_contact_max
    wp.launch(
        _accumulate_patches,
        dim=n_contacts,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            model.shape_body,
            state.body_q,
            data.shape_to_wheel,
            data.shape_count,
            patch._point_sum,
            patch._normal_sum,
            patch.contact_count,
            patch._terrain_min,
        ],
        device=patch.device,
    )
    wp.launch(
        _prepare_patches,
        dim=data.wheel_count,
        inputs=[
            data.shape_count,
            model.shape_material_mu,
            patch._point_sum,
            patch._normal_sum,
            patch.contact_count,
            patch._terrain_min,
            patch.active,
            patch.terrain_shape,
            patch.center,
            patch.normal,
            patch.tangent_u,
            patch.tangent_v,
            patch.friction_seed,
        ],
        device=patch.device,
    )
    wp.launch(
        _accumulate_extents,
        dim=n_contacts,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            model.shape_body,
            state.body_q,
            data.shape_to_wheel,
            data.shape_count,
            patch.center,
            patch.tangent_u,
            patch.tangent_v,
            patch._u_min,
            patch._u_max,
            patch._v_min,
            patch._v_max,
        ],
        device=patch.device,
    )
    wp.launch(
        _finalize_extents,
        dim=data.wheel_count,
        inputs=[patch.active, patch._u_min, patch._u_max, patch._v_min, patch._v_max, patch.tangent_extent, patch.area],
        device=patch.device,
    )


def latch_wheel_loads(model, contacts, data: VehicleModelData, patch: WheelContactPatch) -> None:
    """Latch solver-reported per-wheel normal load into ``patch.fz`` and
    ``patch.normal_load``.

    Call after ``solver.update_contacts`` so ``contacts.force`` is populated.
    No-op if contact forces are unavailable (leaves the previous ``fz``).

    Args:
        model: Finalized model.
        contacts: Contacts with ``force`` populated.
        data: Vehicle tables.
        patch: Patch state; ``fz`` and ``normal_load`` updated in place.
    """
    if data.wheel_count == 0 or contacts.force is None:
        return
    patch.fz.zero_()
    wp.launch(
        _latch_loads,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_normal,
            contacts.force,
            data.shape_to_wheel,
            data.shape_count,
            patch.fz,
        ],
        device=patch.device,
    )
    wp.copy(patch.normal_load, patch.fz)


# --- kernels ---------------------------------------------------------------


@wp.kernel
def _accumulate_patches(
    rigid_contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_to_wheel: wp.array[wp.int32],
    shape_count: int,
    point_sum: wp.array[wp.vec3],
    normal_sum: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    terrain_min: wp.array[wp.int32],
):
    cid = wp.tid()
    if cid >= rigid_contact_count[0]:
        return
    s0 = shape0[cid]
    s1 = shape1[cid]
    if s0 < 0 or s1 < 0 or s0 >= shape_count or s1 >= shape_count:
        return

    w0 = shape_to_wheel[s0]
    w1 = shape_to_wheel[s1]
    # Only handle wheel-vs-nonwheel contacts (ignore wheel-wheel).
    if (w0 >= 0) == (w1 >= 0):
        return

    if w1 >= 0:
        wheel_id = w1
        wheel_shape = s1
        terrain_shape = s0
        wheel_point = point1[cid]
        support_normal = normal[cid]
    else:
        wheel_id = w0
        wheel_shape = s0
        terrain_shape = s1
        wheel_point = point0[cid]
        support_normal = -normal[cid]

    point_world = _shape_point_world(wheel_shape, wheel_point, shape_body, body_q)
    wp.atomic_add(point_sum, wheel_id, point_world)
    wp.atomic_add(normal_sum, wheel_id, support_normal)
    wp.atomic_add(contact_count, wheel_id, wp.int32(1))
    wp.atomic_min(terrain_min, wheel_id, terrain_shape)


@wp.kernel
def _prepare_patches(
    shape_count: int,
    shape_material_mu: wp.array[wp.float32],
    point_sum: wp.array[wp.vec3],
    normal_sum: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    terrain_min: wp.array[wp.int32],
    active: wp.array[wp.bool],
    terrain_shape: wp.array[wp.int32],
    center: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    tangent_u: wp.array[wp.vec3],
    tangent_v: wp.array[wp.vec3],
    friction_seed: wp.array[wp.float32],
):
    w = wp.tid()
    count = contact_count[w]
    if count <= 0:
        return
    active[w] = True
    center[w] = point_sum[w] / wp.float32(count)
    n = _safe_normalize(normal_sum[w])
    normal[w] = n
    u = _tangent_u(n)
    tangent_u[w] = u
    tangent_v[w] = _safe_normalize(wp.cross(n, u))
    t = terrain_min[w]
    if t >= 0 and t < shape_count:
        terrain_shape[w] = t
        friction_seed[w] = shape_material_mu[t]


@wp.kernel
def _accumulate_extents(
    rigid_contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_to_wheel: wp.array[wp.int32],
    shape_count: int,
    center: wp.array[wp.vec3],
    tangent_u: wp.array[wp.vec3],
    tangent_v: wp.array[wp.vec3],
    u_min: wp.array[wp.float32],
    u_max: wp.array[wp.float32],
    v_min: wp.array[wp.float32],
    v_max: wp.array[wp.float32],
):
    cid = wp.tid()
    if cid >= rigid_contact_count[0]:
        return
    s0 = shape0[cid]
    s1 = shape1[cid]
    if s0 < 0 or s1 < 0 or s0 >= shape_count or s1 >= shape_count:
        return
    w0 = shape_to_wheel[s0]
    w1 = shape_to_wheel[s1]
    if (w0 >= 0) == (w1 >= 0):
        return
    if w1 >= 0:
        wheel_id = w1
        wheel_shape = s1
        wheel_point = point1[cid]
    else:
        wheel_id = w0
        wheel_shape = s0
        wheel_point = point0[cid]

    p = _shape_point_world(wheel_shape, wheel_point, shape_body, body_q) - center[wheel_id]
    u = wp.dot(p, tangent_u[wheel_id])
    v = wp.dot(p, tangent_v[wheel_id])
    wp.atomic_min(u_min, wheel_id, u)
    wp.atomic_max(u_max, wheel_id, u)
    wp.atomic_min(v_min, wheel_id, v)
    wp.atomic_max(v_max, wheel_id, v)


@wp.kernel
def _finalize_extents(
    active: wp.array[wp.bool],
    u_min: wp.array[wp.float32],
    u_max: wp.array[wp.float32],
    v_min: wp.array[wp.float32],
    v_max: wp.array[wp.float32],
    tangent_extent: wp.array[wp.vec2],
    area: wp.array[wp.float32],
):
    w = wp.tid()
    if not active[w]:
        return
    eu = wp.max(u_max[w] - u_min[w], 0.0)
    ev = wp.max(v_max[w] - v_min[w], 0.0)
    tangent_extent[w] = wp.vec2(eu, ev)
    area[w] = eu * ev


@wp.kernel
def _latch_loads(
    rigid_contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],
    shape_to_wheel: wp.array[wp.int32],
    shape_count: int,
    fz: wp.array[wp.float32],
):
    cid = wp.tid()
    if cid >= rigid_contact_count[0]:
        return
    s0 = shape0[cid]
    s1 = shape1[cid]
    if s0 < 0 or s1 < 0 or s0 >= shape_count or s1 >= shape_count:
        return
    w0 = shape_to_wheel[s0]
    w1 = shape_to_wheel[s1]
    if (w0 >= 0) == (w1 >= 0):
        return

    # contact_force stores the wrench on body0; project the force on the wheel
    # along the support normal acting on the wheel.
    force_on_body0 = wp.spatial_top(contact_force[cid])
    if w1 >= 0:
        wheel_id = w1
        support_normal = normal[cid]
        force_on_wheel = -force_on_body0  # force on body1 = -force on body0
    else:
        wheel_id = w0
        support_normal = -normal[cid]
        force_on_wheel = force_on_body0
    projected = wp.dot(force_on_wheel, support_normal)
    if projected > 0.0:
        wp.atomic_add(fz, wheel_id, projected)
