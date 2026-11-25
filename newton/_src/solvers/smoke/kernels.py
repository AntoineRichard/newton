# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for :class:`~newton.solvers.SolverSmoke`.

Sparse layout
-------------
Space is cut into cubic cells of edge ``h``. Cells are grouped into bricks
(tiles) of ``TILE^3`` cells, and only active tiles store data. A rebuildable
NanoVDB *index* volume whose voxels are tiles maps a tile coordinate to its
slot; ``neighbor[slot, 27]`` caches the slots of the 26 surrounding tiles so
stencils avoid tree lookups.

Every world has its own grid: world ``w`` is shifted by ``w * WORLD_TILE_STRIDE``
tiles along x in index space, so worlds never share a tile.

Scalars live at cell centers in ``(slot, TILE, TILE, TILE)`` arrays. Velocity is
staggered (MAC): a tile stores the low face of each of its cells along each
axis, plus an extra layer for its high faces, e.g. ``u`` has shape
``(slot, TILE + 1, TILE, TILE)``. The extra layer is only used when the
neighbor tile on that side is inactive; otherwise that face belongs to the
neighbor's first layer.

Outside the active tiles the air is still and clean with zero pressure (an
open boundary). Below the ground plane (cell z < 0) everything is solid.
"""

import warp as wp

from ...geometry.kernels import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder, sdf_ellipsoid, sdf_sphere
from ...geometry.types import Axis, GeoType

TILE = wp.constant(8)
WORLD_TILE_STRIDE = wp.constant(1 << 17)
INVALID_TILE = wp.constant(wp.vec3i(-(1 << 30), -(1 << 30), -(1 << 30)))

OFFSET_U = wp.constant(wp.vec3(0.0, 0.5, 0.5))
OFFSET_V = wp.constant(wp.vec3(0.5, 0.0, 0.5))
OFFSET_W = wp.constant(wp.vec3(0.5, 0.5, 0.0))
OFFSET_C = wp.constant(wp.vec3(0.5, 0.5, 0.5))

AXIS_Z = wp.constant(int(Axis.Z))


@wp.struct
class SparseGrid:
    volume: wp.uint64
    """Index volume whose active voxels are tiles."""
    tile_coord: wp.array[wp.vec3i]
    """Tile coordinate of each slot (world offset included); ``INVALID_TILE`` past the active count."""
    neighbor: wp.array2d[wp.int32]
    """Slots of the 27 tiles around each slot (``-1`` if inactive)."""
    origin: wp.vec3
    """World position of cell (0, 0, 0)'s lower corner [m]; its z is the ground height."""
    h: float
    """Cell edge length [m]."""


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


@wp.func
def floor_div(a: int, b: int) -> int:
    return a // b  # Warp's // floors; / truncates toward zero


@wp.func
def tile_of(g: wp.vec3i) -> wp.vec3i:
    return wp.vec3i(floor_div(g[0], TILE), floor_div(g[1], TILE), floor_div(g[2], TILE))


@wp.func
def world_offset(world: int) -> wp.vec3i:
    """Index-space cell offset of a world's grid."""
    return wp.vec3i(world * WORLD_TILE_STRIDE * TILE, 0, 0)


@wp.func
def tile_world(t: wp.vec3i) -> int:
    return floor_div(t[0] + WORLD_TILE_STRIDE / 2, WORLD_TILE_STRIDE)


@wp.func
def slot_valid(grid: SparseGrid, slot: int) -> bool:
    """Whether ``slot`` holds an active tile; slots past the active count are unused."""
    return grid.tile_coord[slot][0] != INVALID_TILE[0]


@wp.func
def find_slot(grid: SparseGrid, ctx_slot: int, ctx_tile: wp.vec3i, t: wp.vec3i) -> int:
    """Slot of tile ``t``, using ``ctx_slot``'s neighbor table when ``t`` is adjacent to ``ctx_tile``."""
    d = t - ctx_tile
    if ctx_slot >= 0 and wp.abs(d[0]) <= 1 and wp.abs(d[1]) <= 1 and wp.abs(d[2]) <= 1:
        return grid.neighbor[ctx_slot, (d[0] + 1) * 9 + (d[1] + 1) * 3 + (d[2] + 1)]
    return wp.volume_lookup_index(grid.volume, t[0], t[1], t[2])


@wp.func
def axis_step(axis: int) -> wp.vec3i:
    if axis == 0:
        return wp.vec3i(1, 0, 0)
    if axis == 1:
        return wp.vec3i(0, 1, 0)
    return wp.vec3i(0, 0, 1)


@wp.func
def axis_offset(axis: int) -> wp.vec3:
    if axis == 0:
        return OFFSET_U
    if axis == 1:
        return OFFSET_V
    return OFFSET_W


@wp.func
def cell_position(grid: SparseGrid, g: wp.vec3i, world: int, offset: wp.vec3) -> wp.vec3:
    """World position [m] of a sample at ``offset`` (in cells) from cell ``g``'s lower corner."""
    local = g - world_offset(world)
    return grid.origin + (wp.vec3(float(local[0]), float(local[1]), float(local[2])) + offset) * grid.h


@wp.func
def thread_cell(grid: SparseGrid, slot: int, i: int, j: int, k: int) -> wp.vec3i:
    return grid.tile_coord[slot] * TILE + wp.vec3i(i, j, k)


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


@wp.func
def read_cell(
    grid: SparseGrid, f: wp.array4d[float], g: wp.vec3i, ctx_slot: int, ctx_tile: wp.vec3i, outside: float
) -> float:
    t = tile_of(g)
    s = find_slot(grid, ctx_slot, ctx_tile, t)
    if s < 0:
        return outside
    l = g - t * TILE
    return f[s, l[0], l[1], l[2]]


@wp.func
def read_face(
    grid: SparseGrid, f: wp.array4d[float], axis: int, g: wp.vec3i, ctx_slot: int, ctx_tile: wp.vec3i
) -> float:
    """Velocity on the low face of cell ``g`` along ``axis`` (zero outside the active tiles)."""
    t = tile_of(g)
    l = g - t * TILE
    s = find_slot(grid, ctx_slot, ctx_tile, t)
    if s >= 0:
        return f[s, l[0], l[1], l[2]]
    if l[axis] == 0:
        # The face may be the high boundary face of the tile below.
        e = axis_step(axis)
        s = find_slot(grid, ctx_slot, ctx_tile, t - e)
        if s >= 0:
            l = l + e * TILE
            return f[s, l[0], l[1], l[2]]
    return 0.0


@wp.func
def is_solid(grid: SparseGrid, solid: wp.array4d[wp.int32], g: wp.vec3i, ctx_slot: int, ctx_tile: wp.vec3i) -> bool:
    if g[2] < 0:
        return True  # ground
    t = tile_of(g)
    s = find_slot(grid, ctx_slot, ctx_tile, t)
    if s < 0:
        return False
    l = g - t * TILE
    return solid[s, l[0], l[1], l[2]] != 0


@wp.func
def floor_index(q: wp.vec3) -> wp.vec3i:
    return wp.vec3i(int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2])))


@wp.func
def sample_scalar(
    grid: SparseGrid,
    f: wp.array4d[float],
    world: int,
    q: wp.vec3,
    ctx_slot: int,
    ctx_tile: wp.vec3i,
) -> float:
    """Trilinear sample at local index-space position ``q`` (cell-center samples at integer ``q``)."""
    base = floor_index(q)
    fr = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    g = base + world_offset(world)
    c000 = read_cell(grid, f, g, ctx_slot, ctx_tile, 0.0)
    c100 = read_cell(grid, f, g + wp.vec3i(1, 0, 0), ctx_slot, ctx_tile, 0.0)
    c010 = read_cell(grid, f, g + wp.vec3i(0, 1, 0), ctx_slot, ctx_tile, 0.0)
    c110 = read_cell(grid, f, g + wp.vec3i(1, 1, 0), ctx_slot, ctx_tile, 0.0)
    c001 = read_cell(grid, f, g + wp.vec3i(0, 0, 1), ctx_slot, ctx_tile, 0.0)
    c101 = read_cell(grid, f, g + wp.vec3i(1, 0, 1), ctx_slot, ctx_tile, 0.0)
    c011 = read_cell(grid, f, g + wp.vec3i(0, 1, 1), ctx_slot, ctx_tile, 0.0)
    c111 = read_cell(grid, f, g + wp.vec3i(1, 1, 1), ctx_slot, ctx_tile, 0.0)
    c00 = wp.lerp(c000, c100, fr[0])
    c10 = wp.lerp(c010, c110, fr[0])
    c01 = wp.lerp(c001, c101, fr[0])
    c11 = wp.lerp(c011, c111, fr[0])
    return wp.lerp(wp.lerp(c00, c10, fr[1]), wp.lerp(c01, c11, fr[1]), fr[2])


@wp.func
def sample_face(
    grid: SparseGrid,
    f: wp.array4d[float],
    axis: int,
    world: int,
    q: wp.vec3,
    ctx_slot: int,
    ctx_tile: wp.vec3i,
) -> float:
    base = floor_index(q)
    fr = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    g = base + world_offset(world)
    c000 = read_face(grid, f, axis, g, ctx_slot, ctx_tile)
    c100 = read_face(grid, f, axis, g + wp.vec3i(1, 0, 0), ctx_slot, ctx_tile)
    c010 = read_face(grid, f, axis, g + wp.vec3i(0, 1, 0), ctx_slot, ctx_tile)
    c110 = read_face(grid, f, axis, g + wp.vec3i(1, 1, 0), ctx_slot, ctx_tile)
    c001 = read_face(grid, f, axis, g + wp.vec3i(0, 0, 1), ctx_slot, ctx_tile)
    c101 = read_face(grid, f, axis, g + wp.vec3i(1, 0, 1), ctx_slot, ctx_tile)
    c011 = read_face(grid, f, axis, g + wp.vec3i(0, 1, 1), ctx_slot, ctx_tile)
    c111 = read_face(grid, f, axis, g + wp.vec3i(1, 1, 1), ctx_slot, ctx_tile)
    c00 = wp.lerp(c000, c100, fr[0])
    c10 = wp.lerp(c010, c110, fr[0])
    c01 = wp.lerp(c001, c101, fr[0])
    c11 = wp.lerp(c011, c111, fr[0])
    return wp.lerp(wp.lerp(c00, c10, fr[1]), wp.lerp(c01, c11, fr[1]), fr[2])


@wp.func
def local_index(grid: SparseGrid, p: wp.vec3) -> wp.vec3:
    """Local (per-world) index-space position of world point ``p``."""
    return (p - grid.origin) / grid.h


@wp.func
def sample_velocity(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    world: int,
    p: wp.vec3,
    ctx_slot: int,
    ctx_tile: wp.vec3i,
) -> wp.vec3:
    q = local_index(grid, p)
    return wp.vec3(
        sample_face(grid, u, 0, world, q - OFFSET_U, ctx_slot, ctx_tile),
        sample_face(grid, v, 1, world, q - OFFSET_V, ctx_slot, ctx_tile),
        sample_face(grid, w, 2, world, q - OFFSET_W, ctx_slot, ctx_tile),
    )


@wp.func
def point_tile(grid: SparseGrid, world: int, p: wp.vec3) -> wp.vec3i:
    return tile_of(floor_index(local_index(grid, p)) + world_offset(world))


@wp.func
def face_owned(grid: SparseGrid, slot: int, axis: int, fi: int, fj: int, fk: int) -> bool:
    """Whether this tile stores face ``(fi, fj, fk)``; the extra layer is owned only at the active-region boundary."""
    f = wp.vec3i(fi, fj, fk)
    if f[axis] < TILE:
        return True
    d = axis_step(axis)
    return grid.neighbor[slot, (d[0] + 1) * 9 + (d[1] + 1) * 3 + (d[2] + 1)] < 0


# ---------------------------------------------------------------------------
# Topology: activation, rebuild bookkeeping, remap
# ---------------------------------------------------------------------------


@wp.func
def write_activation(points: wp.array[wp.vec3i], mask: wp.array[wp.int32], index: int, t: wp.vec3i, active: bool):
    points[index] = t
    # Tiles fully below ground hold only solid cells and are never stored.
    mask[index] = wp.where(active and t[2] >= 0 and tile_world(t) >= 0, 1, 0)


@wp.kernel
def activate_boxes_kernel(
    grid: SparseGrid,
    centers: wp.array[wp.vec3],
    radii: wp.array[float],
    worlds: wp.array[wp.int32],
    active: wp.array[wp.int32],
    padding: float,
    offset: int,
    points: wp.array[wp.vec3i],
    mask: wp.array[wp.int32],
):
    """Activate the bricks (up to 3 per axis) under a padded box around each point."""
    n, c = wp.tid()
    r = radii[n] + padding
    world = wp.max(worlds[n], 0)
    lo = point_tile(grid, world, centers[n] - wp.vec3(r))
    hi = point_tile(grid, world, centers[n] + wp.vec3(r))
    t = lo + wp.vec3i(c // 9, (c // 3) % 3, c % 3)
    in_box = t[0] <= hi[0] and t[1] <= hi[1] and t[2] <= hi[2]
    write_activation(points, mask, offset + n * 27 + c, t, active[n] != 0 and in_box)


@wp.kernel
def tracer_activation_inputs_kernel(
    particle_age: wp.array[float],
    lifetime: float,
    active: wp.array[wp.int32],
):
    tid = wp.tid()
    active[tid] = wp.where(particle_age[tid] < lifetime, 1, 0)


@wp.kernel
def emitter_activation_inputs_kernel(
    emitter_rate: wp.array[float],
    emitter_heat_rate: wp.array[float],
    active: wp.array[wp.int32],
):
    tid = wp.tid()
    active[tid] = wp.where(emitter_rate[tid] > 0.0 or emitter_heat_rate[tid] > 0.0, 1, 0)


@wp.kernel
def activate_colliders_kernel(
    grid: SparseGrid,
    shapes: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    shape_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    padding: float,
    offset: int,
    points: wp.array[wp.vec3i],
    mask: wp.array[wp.int32],
):
    """Activate up to 4x4x4 tiles around each moving collider so the air it pushes is simulated."""
    n, a, b, c = wp.tid()
    s = shapes[n]
    body = shape_body[s]
    index = offset + ((n * 4 + a) * 4 + b) * 4 + c
    if body < 0:
        write_activation(points, mask, index, wp.vec3i(0), False)
        return
    X = wp.transform_multiply(body_q[body], shape_transform[s])
    center = wp.transform_get_translation(X)
    world = wp.max(shape_world[s], 0)
    r = shape_radius[s] + padding
    lo = point_tile(grid, world, center - wp.vec3(r))
    hi = point_tile(grid, world, center + wp.vec3(r))
    t = lo + wp.vec3i(a, b, c)
    in_range = t[0] <= hi[0] and t[1] <= hi[1] and t[2] <= hi[2]
    write_activation(points, mask, index, t, in_range)


@wp.kernel
def mark_content_kernel(
    grid: SparseGrid,
    density: wp.array4d[float],
    temperature: wp.array4d[float],
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    smoke_threshold: float,
    speed_threshold: float,
    content: wp.array[wp.int32],
):
    """Flag tiles holding smoke or heat (2) or only moving air (1)."""
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    if density[slot, i, j, k] > smoke_threshold or wp.abs(temperature[slot, i, j, k]) > smoke_threshold:
        wp.atomic_max(content, slot, 2)
    elif (
        wp.abs(u[slot, i, j, k]) > speed_threshold
        or wp.abs(v[slot, i, j, k]) > speed_threshold
        or wp.abs(w[slot, i, j, k]) > speed_threshold
    ):
        wp.atomic_max(content, slot, 1)


@wp.kernel
def activate_content_kernel(
    grid: SparseGrid,
    content: wp.array[wp.int32],
    offset: int,
    points: wp.array[wp.vec3i],
    mask: wp.array[wp.int32],
):
    """Keep tiles with smoke and their 26 neighbors active, and tiles with only moving air alone.

    Moving air does not grow the region: the pressure solve spreads small velocities
    everywhere, so dilating around them would grow the region every step.
    """
    slot, n = wp.tid()
    index = offset + slot * 27 + n
    level = 0
    if slot_valid(grid, slot):
        level = content[slot]
    d = wp.vec3i(n // 9 - 1, (n // 3) % 3 - 1, n % 3 - 1)
    keep = level == 2 or (level == 1 and n == 13)
    write_activation(points, mask, index, grid.tile_coord[slot] + d, keep)


@wp.kernel
def anchor_worlds_kernel(offset: int, points: wp.array[wp.vec3i], mask: wp.array[wp.int32]):
    """Keep one tile per world active so the volume is never empty."""
    world = wp.tid()
    write_activation(points, mask, offset + world, wp.vec3i(world * WORLD_TILE_STRIDE, 0, 0), True)


@wp.kernel
def assign_tile_coords_kernel(
    volume: wp.uint64,
    points: wp.array[wp.vec3i],
    mask: wp.array[wp.int32],
    tile_coord: wp.array[wp.vec3i],
):
    tid = wp.tid()
    if mask[tid] == 0:
        return
    t = points[tid]
    slot = wp.volume_lookup_index(volume, t[0], t[1], t[2])
    if slot >= 0:
        tile_coord[slot] = t


@wp.kernel
def build_neighbors_kernel(grid: SparseGrid, neighbor: wp.array2d[wp.int32]):
    slot, n = wp.tid()
    if not slot_valid(grid, slot):
        neighbor[slot, n] = -1
        return
    d = wp.vec3i(n // 9 - 1, (n // 3) % 3 - 1, n % 3 - 1)
    t = grid.tile_coord[slot] + d
    neighbor[slot, n] = wp.volume_lookup_index(grid.volume, t[0], t[1], t[2])


@wp.kernel
def remap_field_kernel(
    volume: wp.uint64,
    old_tile_coord: wp.array[wp.vec3i],
    src: wp.array4d[float],
    dst: wp.array4d[float],
):
    """Move each surviving tile's data from its old slot to its new slot."""
    slot, i, j, k = wp.tid()
    t = old_tile_coord[slot]
    if t[0] == INVALID_TILE[0]:
        return
    s = wp.volume_lookup_index(volume, t[0], t[1], t[2])
    if s >= 0:
        dst[s, i, j, k] = src[slot, i, j, k]


# ---------------------------------------------------------------------------
# Colliders
# ---------------------------------------------------------------------------


@wp.func
def shape_signed_distance(geo_type: int, scale: wp.vec3, mesh: wp.uint64, x: wp.vec3, max_dist: float) -> float:
    """Signed distance [m] from shape-local point ``x``; negative inside."""
    if geo_type == GeoType.SPHERE:
        return sdf_sphere(x, scale[0])
    if geo_type == GeoType.BOX:
        return sdf_box(x, scale[0], scale[1], scale[2])
    if geo_type == GeoType.CAPSULE:
        return sdf_capsule(x, scale[0], scale[1], AXIS_Z)
    if geo_type == GeoType.CYLINDER:
        return sdf_cylinder(x, scale[0], scale[1], AXIS_Z, -1.0, scale[2])
    if geo_type == GeoType.CONE:
        return sdf_cone(x, scale[0], scale[1], AXIS_Z)
    if geo_type == GeoType.ELLIPSOID:
        return sdf_ellipsoid(x, scale)
    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        query = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(x, scale), max_dist)
        if query.result:
            closest = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), scale)
            return query.sign * wp.length(x - closest)
    return 1.0e6


@wp.kernel
def rasterize_colliders_kernel(
    grid: SparseGrid,
    world_shape_start: wp.array[wp.int32],
    world_shapes: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    solid: wp.array4d[wp.int32],
    solid_velocity: wp.array4d[wp.vec3],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    g = thread_cell(grid, slot, i, j, k)
    world = tile_world(grid.tile_coord[slot])
    p = cell_position(grid, g, world, OFFSET_C)
    reach = 0.87 * grid.h  # half the cell diagonal
    for c in range(world_shape_start[world], world_shape_start[world + 1]):
        s = world_shapes[c]
        b = shape_body[s]
        X_wb = wp.transform_identity()
        if b >= 0:
            X_wb = body_q[b]
        X_ws = wp.transform_multiply(X_wb, shape_transform[s])
        if wp.length(p - wp.transform_get_translation(X_ws)) > shape_radius[s] + reach:
            continue
        x_local = wp.transform_point(wp.transform_inverse(X_ws), p)
        if shape_signed_distance(shape_type[s], shape_scale[s], shape_source_ptr[s], x_local, reach) < 0.0:
            vel = wp.vec3(0.0)
            if b >= 0:
                qd = body_qd[b]
                vel = wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), p - wp.transform_point(X_wb, body_com[b]))
            solid[slot, i, j, k] = 1
            solid_velocity[slot, i, j, k] = vel
            return
    solid[slot, i, j, k] = 0
    solid_velocity[slot, i, j, k] = wp.vec3(0.0)


@wp.kernel
def enforce_solid_faces_kernel(
    grid: SparseGrid,
    vel: wp.array4d[float],
    axis: int,
    solid: wp.array4d[wp.int32],
    solid_velocity: wp.array4d[wp.vec3],
):
    """Set faces touching a solid cell to the solid's velocity (no flow through solids)."""
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, axis, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(fi, fj, fk)
    e = axis_step(axis)
    if axis == 2 and g[2] == 0:
        vel[slot, fi, fj, fk] = 0.0  # ground
        return
    s1 = is_solid(grid, solid, g, slot, tile)
    s0 = is_solid(grid, solid, g - e, slot, tile)
    if not s0 and not s1:
        return
    total = wp.vec3(0.0)
    count = float(0.0)
    if s0:
        t0 = tile_of(g - e)
        l0 = g - e - t0 * TILE
        n0 = find_slot(grid, slot, tile, t0)
        if n0 >= 0:
            total += solid_velocity[n0, l0[0], l0[1], l0[2]]
            count += 1.0
    if s1:
        t1 = tile_of(g)
        n1 = find_slot(grid, slot, tile, t1)
        if n1 >= 0:
            l1 = g - t1 * TILE
            total += solid_velocity[n1, l1[0], l1[1], l1[2]]
            count += 1.0
    if count > 0.0:
        vel[slot, fi, fj, fk] = total[axis] / count


# ---------------------------------------------------------------------------
# Advection, emitters, forces
# ---------------------------------------------------------------------------


@wp.kernel
def advect_face_kernel(
    grid: SparseGrid,
    src: wp.array4d[float],
    axis: int,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    dt: float,
    decay: float,
    dst: wp.array4d[float],
):
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, axis, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    world = tile_world(tile)
    offset = axis_offset(axis)
    p = cell_position(grid, tile * TILE + wp.vec3i(fi, fj, fk), world, offset)
    mid = p - 0.5 * dt * sample_velocity(grid, u, v, w, world, p, slot, tile)
    back = p - dt * sample_velocity(grid, u, v, w, world, mid, slot, tile)
    dst[slot, fi, fj, fk] = decay * sample_face(grid, src, axis, world, local_index(grid, back) - offset, slot, tile)


@wp.kernel
def advect_cell_kernel(
    grid: SparseGrid,
    src: wp.array4d[float],
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    dt: float,
    decay: float,
    dst: wp.array4d[float],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    world = tile_world(tile)
    p = cell_position(grid, tile * TILE + wp.vec3i(i, j, k), world, OFFSET_C)
    mid = p - 0.5 * dt * sample_velocity(grid, u, v, w, world, p, slot, tile)
    back = p - dt * sample_velocity(grid, u, v, w, world, mid, slot, tile)
    dst[slot, i, j, k] = decay * sample_scalar(grid, src, world, local_index(grid, back) - OFFSET_C, slot, tile)


@wp.func
def emitter_falloff(p: wp.vec3, center: wp.vec3, radius: float) -> float:
    return wp.max(0.0, 1.0 - wp.length(p - center) / radius)


@wp.kernel
def emit_scalars_kernel(
    grid: SparseGrid,
    world_emitter_start: wp.array[wp.int32],
    world_emitters: wp.array[wp.int32],
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
    emitter_heat_rate: wp.array[float],
    dt: float,
    density: wp.array4d[float],
    temperature: wp.array4d[float],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    world = tile_world(tile)
    p = cell_position(grid, tile * TILE + wp.vec3i(i, j, k), world, OFFSET_C)
    add_density = float(0.0)
    add_heat = float(0.0)
    for c in range(world_emitter_start[world], world_emitter_start[world + 1]):
        e = world_emitters[c]
        f = emitter_falloff(p, emitter_position[e], emitter_radius[e])
        add_density += f * emitter_rate[e]
        add_heat += f * emitter_heat_rate[e]
    if add_density != 0.0 or add_heat != 0.0:
        density[slot, i, j, k] = wp.max(density[slot, i, j, k] + dt * add_density, 0.0)
        temperature[slot, i, j, k] = temperature[slot, i, j, k] + dt * add_heat


@wp.kernel
def emit_velocity_kernel(
    grid: SparseGrid,
    vel: wp.array4d[float],
    axis: int,
    world_emitter_start: wp.array[wp.int32],
    world_emitters: wp.array[wp.int32],
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
    emitter_velocity: wp.array[wp.vec3],
    coupling: float,
):
    """Pull the air near active emitters toward the emitter velocity."""
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, axis, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    world = tile_world(tile)
    p = cell_position(grid, tile * TILE + wp.vec3i(fi, fj, fk), world, axis_offset(axis))
    for c in range(world_emitter_start[world], world_emitter_start[world + 1]):
        e = world_emitters[c]
        if emitter_rate[e] <= 0.0:
            continue
        f = emitter_falloff(p, emitter_position[e], emitter_radius[e]) * coupling
        if f > 0.0:
            vel[slot, fi, fj, fk] = wp.lerp(vel[slot, fi, fj, fk], emitter_velocity[e][axis], wp.min(f, 1.0))


@wp.kernel
def buoyancy_kernel(
    grid: SparseGrid,
    density: wp.array4d[float],
    temperature: wp.array4d[float],
    density_weight: float,
    temperature_lift: float,
    dt: float,
    w: wp.array4d[float],
):
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, 2, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(fi, fj, fk)
    below = g - wp.vec3i(0, 0, 1)
    rho = 0.5 * (read_cell(grid, density, g, slot, tile, 0.0) + read_cell(grid, density, below, slot, tile, 0.0))
    temp = 0.5 * (
        read_cell(grid, temperature, g, slot, tile, 0.0) + read_cell(grid, temperature, below, slot, tile, 0.0)
    )
    w[slot, fi, fj, fk] = w[slot, fi, fj, fk] + dt * (temperature_lift * temp - density_weight * rho)


@wp.kernel
def center_velocity_kernel(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    out: wp.array4d[wp.vec3],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(i, j, k)
    out[slot, i, j, k] = 0.5 * wp.vec3(
        u[slot, i, j, k] + read_face(grid, u, 0, g + wp.vec3i(1, 0, 0), slot, tile),
        v[slot, i, j, k] + read_face(grid, v, 1, g + wp.vec3i(0, 1, 0), slot, tile),
        w[slot, i, j, k] + read_face(grid, w, 2, g + wp.vec3i(0, 0, 1), slot, tile),
    )


@wp.func
def read_vec(grid: SparseGrid, f: wp.array4d[wp.vec3], g: wp.vec3i, ctx_slot: int, ctx_tile: wp.vec3i) -> wp.vec3:
    t = tile_of(g)
    s = find_slot(grid, ctx_slot, ctx_tile, t)
    if s < 0:
        return wp.vec3(0.0)
    l = g - t * TILE
    return f[s, l[0], l[1], l[2]]


@wp.kernel
def curl_kernel(
    grid: SparseGrid,
    vel: wp.array4d[wp.vec3],
    curl: wp.array4d[wp.vec3],
    curl_mag: wp.array4d[float],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(i, j, k)
    s = 0.5 / grid.h
    dx = (
        read_vec(grid, vel, g + wp.vec3i(1, 0, 0), slot, tile) - read_vec(grid, vel, g - wp.vec3i(1, 0, 0), slot, tile)
    ) * s
    dy = (
        read_vec(grid, vel, g + wp.vec3i(0, 1, 0), slot, tile) - read_vec(grid, vel, g - wp.vec3i(0, 1, 0), slot, tile)
    ) * s
    dz = (
        read_vec(grid, vel, g + wp.vec3i(0, 0, 1), slot, tile) - read_vec(grid, vel, g - wp.vec3i(0, 0, 1), slot, tile)
    ) * s
    c = wp.vec3(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0])
    curl[slot, i, j, k] = c
    curl_mag[slot, i, j, k] = wp.length(c)


@wp.kernel
def confinement_kernel(
    grid: SparseGrid,
    curl: wp.array4d[wp.vec3],
    curl_mag: wp.array4d[float],
    density: wp.array4d[float],
    epsilon: float,
    force: wp.array4d[wp.vec3],
):
    """Vorticity confinement, weighted by smoke so clean air stays calm."""
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(i, j, k)
    m = curl_mag[slot, i, j, k]
    gx = read_cell(grid, curl_mag, g + wp.vec3i(1, 0, 0), slot, tile, m) - read_cell(
        grid, curl_mag, g - wp.vec3i(1, 0, 0), slot, tile, m
    )
    gy = read_cell(grid, curl_mag, g + wp.vec3i(0, 1, 0), slot, tile, m) - read_cell(
        grid, curl_mag, g - wp.vec3i(0, 1, 0), slot, tile, m
    )
    gz = read_cell(grid, curl_mag, g + wp.vec3i(0, 0, 1), slot, tile, m) - read_cell(
        grid, curl_mag, g - wp.vec3i(0, 0, 1), slot, tile, m
    )
    grad = wp.vec3(gx, gy, gz)
    g_len = wp.length(grad)
    if g_len < 1.0e-6:
        force[slot, i, j, k] = wp.vec3(0.0)
        return
    smoke = wp.min(density[slot, i, j, k], 1.0)
    force[slot, i, j, k] = smoke * epsilon * grid.h * wp.cross(grad / g_len, curl[slot, i, j, k])


@wp.kernel
def apply_cell_force_kernel(
    grid: SparseGrid,
    force: wp.array4d[wp.vec3],
    axis: int,
    dt: float,
    vel: wp.array4d[float],
):
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, axis, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(fi, fj, fk)
    f = read_vec(grid, force, g, slot, tile)[axis] + read_vec(grid, force, g - axis_step(axis), slot, tile)[axis]
    vel[slot, fi, fj, fk] = vel[slot, fi, fj, fk] + dt * 0.5 * f


# ---------------------------------------------------------------------------
# Pressure projection
# ---------------------------------------------------------------------------


@wp.kernel
def divergence_kernel(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    solid: wp.array4d[wp.int32],
    div: wp.array4d[float],
):
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    if solid[slot, i, j, k] != 0:
        div[slot, i, j, k] = 0.0
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(i, j, k)
    du = read_face(grid, u, 0, g + wp.vec3i(1, 0, 0), slot, tile) - u[slot, i, j, k]
    dv = read_face(grid, v, 1, g + wp.vec3i(0, 1, 0), slot, tile) - v[slot, i, j, k]
    dw = read_face(grid, w, 2, g + wp.vec3i(0, 0, 1), slot, tile) - w[slot, i, j, k]
    div[slot, i, j, k] = (du + dv + dw) / grid.h


@wp.func
def accumulate_neighbor(
    grid: SparseGrid,
    p: wp.array4d[float],
    solid: wp.array4d[wp.int32],
    g: wp.vec3i,
    ctx_slot: int,
    ctx_tile: wp.vec3i,
    acc: wp.vec2,
) -> wp.vec2:
    """Accumulate (pressure sum, neighbor count); solid neighbors are Neumann and skipped."""
    if is_solid(grid, solid, g, ctx_slot, ctx_tile):
        return acc
    return acc + wp.vec2(read_cell(grid, p, g, ctx_slot, ctx_tile, 0.0), 1.0)


@wp.kernel
def pressure_rbsor_kernel(
    grid: SparseGrid,
    div: wp.array4d[float],
    solid: wp.array4d[wp.int32],
    omega: float,
    parity: int,
    p: wp.array4d[float],
):
    """One red-black SOR sweep of the pressure Poisson equation."""
    slot, i, j, k = wp.tid()
    if not slot_valid(grid, slot):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(i, j, k)
    if ((g[0] + g[1] + g[2]) & 1) != parity:
        return
    if solid[slot, i, j, k] != 0:
        p[slot, i, j, k] = 0.0
        return
    acc = wp.vec2(0.0)
    acc = accumulate_neighbor(grid, p, solid, g - wp.vec3i(1, 0, 0), slot, tile, acc)
    acc = accumulate_neighbor(grid, p, solid, g + wp.vec3i(1, 0, 0), slot, tile, acc)
    acc = accumulate_neighbor(grid, p, solid, g - wp.vec3i(0, 1, 0), slot, tile, acc)
    acc = accumulate_neighbor(grid, p, solid, g + wp.vec3i(0, 1, 0), slot, tile, acc)
    acc = accumulate_neighbor(grid, p, solid, g - wp.vec3i(0, 0, 1), slot, tile, acc)
    acc = accumulate_neighbor(grid, p, solid, g + wp.vec3i(0, 0, 1), slot, tile, acc)
    if acc[1] == 0.0:
        return
    target = (acc[0] - grid.h * grid.h * div[slot, i, j, k]) / acc[1]
    p[slot, i, j, k] = p[slot, i, j, k] + omega * (target - p[slot, i, j, k])


@wp.kernel
def project_kernel(
    grid: SparseGrid,
    vel: wp.array4d[float],
    axis: int,
    p: wp.array4d[float],
    solid: wp.array4d[wp.int32],
):
    slot, fi, fj, fk = wp.tid()
    if not slot_valid(grid, slot) or not face_owned(grid, slot, axis, fi, fj, fk):
        return
    tile = grid.tile_coord[slot]
    g = tile * TILE + wp.vec3i(fi, fj, fk)
    g0 = g - axis_step(axis)
    if is_solid(grid, solid, g, slot, tile) or is_solid(grid, solid, g0, slot, tile):
        return  # solid faces keep the collider velocity
    dp = read_cell(grid, p, g, slot, tile, 0.0) - read_cell(grid, p, g0, slot, tile, 0.0)
    vel[slot, fi, fj, fk] = vel[slot, fi, fj, fk] - dp / grid.h


# ---------------------------------------------------------------------------
# Tracer particles
# ---------------------------------------------------------------------------


@wp.kernel
def emit_tracers_kernel(
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
    emitter_world: wp.array[wp.int32],
    world_tracer_start: wp.array[wp.int32],
    tracer_rate_scale: float,
    slots_per_emitter: int,
    dt: float,
    seed: wp.array[wp.int32],
    head: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_age: wp.array[float],
):
    tid = wp.tid()
    e = tid // slots_per_emitter
    world = wp.max(emitter_world[e], 0)
    begin = world_tracer_start[world]
    count = world_tracer_start[world + 1] - begin
    if count == 0:
        return
    expected = tracer_rate_scale * wp.max(emitter_rate[e], 0.0) * dt
    state = wp.rand_init(seed[0], tid)
    if wp.randf(state) * float(slots_per_emitter) >= expected:
        return
    slot = begin + wp.atomic_add(head, world, 1) % count
    r = emitter_radius[e]
    jitter = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
    particle_q[slot] = emitter_position[e] + r * jitter
    particle_qd[slot] = wp.vec3(0.0)
    particle_age[slot] = 0.0


@wp.kernel
def advance_seed_kernel(seed: wp.array[wp.int32], head: wp.array[wp.int32], world_tracer_start: wp.array[wp.int32]):
    world = wp.tid()
    if world == 0:
        seed[0] = seed[0] + 1
    head[world] = head[world] % wp.max(world_tracer_start[world + 1] - world_tracer_start[world], 1)


@wp.kernel
def advect_tracers_kernel(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    particle_world: wp.array[wp.int32],
    dt: float,
    lifetime: float,
    particle_q_in: wp.array[wp.vec3],
    particle_age_in: wp.array[float],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_age: wp.array[float],
):
    tid = wp.tid()
    age = particle_age_in[tid]
    p = particle_q_in[tid]
    if age >= lifetime:
        particle_q[tid] = p
        particle_qd[tid] = wp.vec3(0.0)
        particle_age[tid] = age
        return
    world = wp.max(particle_world[tid], 0)
    tile = point_tile(grid, world, p)
    slot = wp.volume_lookup_index(grid.volume, tile[0], tile[1], tile[2])
    vel = sample_velocity(grid, u, v, w, world, p, slot, tile)
    vel = sample_velocity(grid, u, v, w, world, p + 0.5 * dt * vel, slot, tile)
    p = p + dt * vel
    p = wp.vec3(p[0], p[1], wp.max(p[2], grid.origin[2]))
    particle_q[tid] = p
    particle_qd[tid] = vel
    particle_age[tid] = age + dt


@wp.kernel
def sample_points_kernel(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    density: wp.array4d[float],
    world: int,
    points: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    smoke: wp.array[float],
):
    tid = wp.tid()
    p = points[tid]
    tile = point_tile(grid, world, p)
    slot = wp.volume_lookup_index(grid.volume, tile[0], tile[1], tile[2])
    velocity[tid] = sample_velocity(grid, u, v, w, world, p, slot, tile)
    smoke[tid] = sample_scalar(grid, density, world, local_index(grid, p) - OFFSET_C, slot, tile)


# ---------------------------------------------------------------------------
# Render grains
# ---------------------------------------------------------------------------


@wp.func
def tracer_extent(base_radius: float, spread_rate: float, age: float) -> float:
    return base_radius + spread_rate * age


@wp.kernel
def sample_grains_kernel(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_age: wp.array[float],
    spread_rate: float,
    seed: int,
    grains: wp.array2d[wp.vec3],
):
    pid, g = wp.tid()
    state = wp.rand_init(seed, pid * grains.shape[1] + g)
    r = tracer_extent(particle_radius[pid], spread_rate, wp.min(particle_age[pid], 1.0e3))
    offset = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
    grains[pid, g] = particle_q[pid] + 2.0 * r * offset


@wp.kernel
def update_grains_kernel(
    grid: SparseGrid,
    u: wp.array4d[float],
    v: wp.array4d[float],
    w: wp.array4d[float],
    particle_world: wp.array[wp.int32],
    dt: float,
    lifetime: float,
    spread_rate: float,
    seed: int,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_age: wp.array[float],
    grains: wp.array2d[wp.vec3],
):
    pid, g = wp.tid()
    age = particle_age[pid]
    if age >= lifetime:
        return
    center = particle_q[pid]
    r = tracer_extent(particle_radius[pid], spread_rate, age)
    x = grains[pid, g]
    if age <= 1.5 * dt:
        # The tracer was just (re)emitted: scatter its grains around the new position.
        state = wp.rand_init(seed, pid * grains.shape[1] + g)
        x = center + 2.0 * r * wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
    else:
        world = wp.max(particle_world[pid], 0)
        tile = point_tile(grid, world, x)
        slot = wp.volume_lookup_index(grid.volume, tile[0], tile[1], tile[2])
        x = x + dt * sample_velocity(grid, u, v, w, world, x, slot, tile)
    # Keep the grain inside the tracer's puff.
    d = x - center
    d_len = wp.length(d)
    if d_len > r:
        x = center + d * (r / d_len)
    grains[pid, g] = wp.vec3(x[0], x[1], wp.max(x[2], grid.origin[2]))
