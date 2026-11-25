# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for :class:`~newton.solvers.SolverSmoke`.

The grid is a dense, axis-aligned, staggered (MAC) grid. Cell ``(i, j, k)``
covers ``lower + h * [i, i+1] x [j, j+1] x [k, k+1]``. Velocity components
live on the faces normal to their axis; scalars live at cell centers.
"""

import warp as wp

from ...geometry.kernels import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder, sdf_ellipsoid, sdf_sphere
from ...geometry.types import Axis, GeoType

# Sample offsets (in cells) of each staggered field from the grid lower corner.
OFFSET_U = wp.constant(wp.vec3(0.0, 0.5, 0.5))
OFFSET_V = wp.constant(wp.vec3(0.5, 0.0, 0.5))
OFFSET_W = wp.constant(wp.vec3(0.5, 0.5, 0.0))
OFFSET_C = wp.constant(wp.vec3(0.5, 0.5, 0.5))

AXIS_Z = wp.constant(int(Axis.Z))


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


@wp.func
def sample_grid(f: wp.array3d[float], g: wp.vec3) -> float:
    """Trilinearly sample ``f`` at index-space coordinate ``g``, clamped to the array."""
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    x = wp.clamp(g[0], 0.0, float(nx - 1))
    y = wp.clamp(g[1], 0.0, float(ny - 1))
    z = wp.clamp(g[2], 0.0, float(nz - 1))
    i = wp.min(int(x), nx - 2)
    j = wp.min(int(y), ny - 2)
    k = wp.min(int(z), nz - 2)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    c00 = wp.lerp(f[i, j, k], f[i + 1, j, k], fx)
    c10 = wp.lerp(f[i, j + 1, k], f[i + 1, j + 1, k], fx)
    c01 = wp.lerp(f[i, j, k + 1], f[i + 1, j, k + 1], fx)
    c11 = wp.lerp(f[i, j + 1, k + 1], f[i + 1, j + 1, k + 1], fx)
    return wp.lerp(wp.lerp(c00, c10, fy), wp.lerp(c01, c11, fy), fz)


@wp.func
def sample_velocity(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    lower: wp.vec3,
    inv_h: float,
    p: wp.vec3,
) -> wp.vec3:
    """Sample the staggered velocity field at world position ``p``."""
    q = (p - lower) * inv_h
    return wp.vec3(sample_grid(u, q - OFFSET_U), sample_grid(v, q - OFFSET_V), sample_grid(w, q - OFFSET_W))


@wp.func
def axis_offset(axis: int) -> wp.vec3:
    if axis == 0:
        return OFFSET_U
    if axis == 1:
        return OFFSET_V
    return OFFSET_W


@wp.func
def axis_step(axis: int) -> wp.vec3i:
    if axis == 0:
        return wp.vec3i(1, 0, 0)
    if axis == 1:
        return wp.vec3i(0, 1, 0)
    return wp.vec3i(0, 0, 1)


@wp.func
def in_grid(shape: wp.vec3i, i: int, j: int, k: int) -> bool:
    return i >= 0 and j >= 0 and k >= 0 and i < shape[0] and j < shape[1] and k < shape[2]


@wp.func
def is_solid(solid: wp.array3d[wp.int32], i: int, j: int, k: int) -> bool:
    """Below the grid is ground (solid); the sides and the top of the grid are open air."""
    if k < 0:
        return True
    if not in_grid(wp.vec3i(solid.shape[0], solid.shape[1], solid.shape[2]), i, j, k):
        return False
    return solid[i, j, k] != 0


@wp.func
def pressure_at(p: wp.array3d[float], i: int, j: int, k: int) -> float:
    """Pressure outside the grid is zero (open boundary)."""
    if not in_grid(wp.vec3i(p.shape[0], p.shape[1], p.shape[2]), i, j, k):
        return 0.0
    return p[i, j, k]


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
    lower: wp.vec3,
    h: float,
    collider_shapes: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    solid: wp.array3d[wp.int32],
    solid_velocity: wp.array3d[wp.vec3],
):
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + OFFSET_C) * h
    cell_reach = 0.87 * h  # half the cell diagonal
    for c in range(collider_shapes.shape[0]):
        s = collider_shapes[c]
        b = shape_body[s]
        X_wb = wp.transform_identity()
        if b >= 0:
            X_wb = body_q[b]
        X_ws = wp.transform_multiply(X_wb, shape_transform[s])
        if wp.length(p - wp.transform_get_translation(X_ws)) > shape_radius[s] + cell_reach:
            continue
        x_local = wp.transform_point(wp.transform_inverse(X_ws), p)
        if shape_signed_distance(shape_type[s], shape_scale[s], shape_source_ptr[s], x_local, cell_reach) < 0.0:
            vel = wp.vec3(0.0)
            if b >= 0:
                com = wp.transform_point(X_wb, body_com[b])
                qd = body_qd[b]
                vel = wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), p - com)
            solid[i, j, k] = 1
            solid_velocity[i, j, k] = vel
            return
    solid[i, j, k] = 0
    solid_velocity[i, j, k] = wp.vec3(0.0)


@wp.kernel
def enforce_solid_faces_kernel(
    vel: wp.array3d[float],
    axis: int,
    solid: wp.array3d[wp.int32],
    solid_velocity: wp.array3d[wp.vec3],
):
    """Set faces touching a solid cell to the solid's velocity (no-through-flow)."""
    i, j, k = wp.tid()
    s = axis_step(axis)
    i0 = i - s[0]
    j0 = j - s[1]
    k0 = k - s[2]
    if axis == 2 and k == 0:
        vel[i, j, k] = 0.0  # ground
        return
    s1 = is_solid(solid, i, j, k)
    s0 = is_solid(solid, i0, j0, k0)
    if not s0 and not s1:
        return
    total = wp.vec3(0.0)
    count = float(0.0)
    if s0:
        total += solid_velocity[i0, j0, k0]
        count += 1.0
    if s1:
        total += solid_velocity[i, j, k]
        count += 1.0
    vel[i, j, k] = total[axis] / count


# ---------------------------------------------------------------------------
# Advection, sources and forces
# ---------------------------------------------------------------------------


@wp.kernel
def advect_kernel(
    src: wp.array3d[float],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    lower: wp.vec3,
    h: float,
    offset: wp.vec3,
    dt: float,
    decay: float,
    dst: wp.array3d[float],
):
    """Semi-Lagrangian advection with an RK2 backtrace."""
    i, j, k = wp.tid()
    inv_h = 1.0 / h
    p = lower + (wp.vec3(float(i), float(j), float(k)) + offset) * h
    mid = p - 0.5 * dt * sample_velocity(u, v, w, lower, inv_h, p)
    back = p - dt * sample_velocity(u, v, w, lower, inv_h, mid)
    dst[i, j, k] = decay * sample_grid(src, (back - lower) * inv_h - offset)


@wp.func
def emitter_falloff(p: wp.vec3, center: wp.vec3, radius: float) -> float:
    return wp.max(0.0, 1.0 - wp.length(p - center) / radius)


@wp.kernel
def emit_scalars_kernel(
    lower: wp.vec3,
    h: float,
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
    emitter_heat_rate: wp.array[float],
    dt: float,
    density: wp.array3d[float],
    temperature: wp.array3d[float],
):
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + OFFSET_C) * h
    add_density = float(0.0)
    add_heat = float(0.0)
    for e in range(emitter_position.shape[0]):
        f = emitter_falloff(p, emitter_position[e], emitter_radius[e])
        add_density += f * emitter_rate[e]
        add_heat += f * emitter_heat_rate[e]
    if add_density != 0.0 or add_heat != 0.0:
        density[i, j, k] = wp.max(density[i, j, k] + dt * add_density, 0.0)
        temperature[i, j, k] = temperature[i, j, k] + dt * add_heat


@wp.kernel
def emit_velocity_kernel(
    vel: wp.array3d[float],
    axis: int,
    lower: wp.vec3,
    h: float,
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
    emitter_velocity: wp.array[wp.vec3],
    coupling: float,
):
    """Pull the air near active emitters toward the emitter velocity."""
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + axis_offset(axis)) * h
    for e in range(emitter_position.shape[0]):
        if emitter_rate[e] <= 0.0:
            continue
        f = emitter_falloff(p, emitter_position[e], emitter_radius[e]) * coupling
        if f > 0.0:
            vel[i, j, k] = wp.lerp(vel[i, j, k], emitter_velocity[e][axis], wp.min(f, 1.0))


@wp.kernel
def buoyancy_kernel(
    density: wp.array3d[float],
    temperature: wp.array3d[float],
    density_weight: float,
    temperature_lift: float,
    dt: float,
    w: wp.array3d[float],
):
    i, j, k = wp.tid()
    nz = density.shape[2]
    k0 = wp.max(k - 1, 0)
    k1 = wp.min(k, nz - 1)
    rho = 0.5 * (density[i, j, k0] + density[i, j, k1])
    temp = 0.5 * (temperature[i, j, k0] + temperature[i, j, k1])
    w[i, j, k] = w[i, j, k] + dt * (temperature_lift * temp - density_weight * rho)


@wp.kernel
def center_velocity_kernel(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    out: wp.array3d[wp.vec3],
):
    i, j, k = wp.tid()
    out[i, j, k] = 0.5 * wp.vec3(u[i, j, k] + u[i + 1, j, k], v[i, j, k] + v[i, j + 1, k], w[i, j, k] + w[i, j, k + 1])


@wp.func
def central_diff(f: wp.array3d[wp.vec3], i: int, j: int, k: int, axis: int, h: float) -> wp.vec3:
    s = axis_step(axis)
    lo = wp.vec3i(wp.max(i - s[0], 0), wp.max(j - s[1], 0), wp.max(k - s[2], 0))
    hi = wp.vec3i(wp.min(i + s[0], f.shape[0] - 1), wp.min(j + s[1], f.shape[1] - 1), wp.min(k + s[2], f.shape[2] - 1))
    span = float(hi[axis] - lo[axis]) * h
    if span == 0.0:
        return wp.vec3(0.0)
    return (f[hi[0], hi[1], hi[2]] - f[lo[0], lo[1], lo[2]]) / span


@wp.kernel
def curl_kernel(vel: wp.array3d[wp.vec3], h: float, curl: wp.array3d[wp.vec3], curl_mag: wp.array3d[float]):
    i, j, k = wp.tid()
    dx = central_diff(vel, i, j, k, 0, h)
    dy = central_diff(vel, i, j, k, 1, h)
    dz = central_diff(vel, i, j, k, 2, h)
    c = wp.vec3(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0])
    curl[i, j, k] = c
    curl_mag[i, j, k] = wp.length(c)


@wp.kernel
def confinement_kernel(
    curl: wp.array3d[wp.vec3],
    curl_mag: wp.array3d[float],
    density: wp.array3d[float],
    h: float,
    epsilon: float,
    force: wp.array3d[wp.vec3],
):
    """Vorticity confinement, weighted by smoke so clean air stays calm."""
    i, j, k = wp.tid()
    nx = curl.shape[0]
    ny = curl.shape[1]
    nz = curl.shape[2]
    g = wp.vec3(
        curl_mag[wp.min(i + 1, nx - 1), j, k] - curl_mag[wp.max(i - 1, 0), j, k],
        curl_mag[i, wp.min(j + 1, ny - 1), k] - curl_mag[i, wp.max(j - 1, 0), k],
        curl_mag[i, j, wp.min(k + 1, nz - 1)] - curl_mag[i, j, wp.max(k - 1, 0)],
    )
    g_len = wp.length(g)
    if g_len < 1.0e-6:
        force[i, j, k] = wp.vec3(0.0)
        return
    smoke = wp.min(density[i, j, k], 1.0)
    force[i, j, k] = smoke * epsilon * h * wp.cross(g / g_len, curl[i, j, k])


@wp.kernel
def apply_cell_force_kernel(force: wp.array3d[wp.vec3], axis: int, dt: float, vel: wp.array3d[float]):
    i, j, k = wp.tid()
    s = axis_step(axis)
    i0 = wp.max(i - s[0], 0)
    j0 = wp.max(j - s[1], 0)
    k0 = wp.max(k - s[2], 0)
    i1 = wp.min(i, force.shape[0] - 1)
    j1 = wp.min(j, force.shape[1] - 1)
    k1 = wp.min(k, force.shape[2] - 1)
    vel[i, j, k] = vel[i, j, k] + dt * 0.5 * (force[i0, j0, k0][axis] + force[i1, j1, k1][axis])


# ---------------------------------------------------------------------------
# Pressure projection
# ---------------------------------------------------------------------------


@wp.kernel
def divergence_kernel(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    solid: wp.array3d[wp.int32],
    h: float,
    div: wp.array3d[float],
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        div[i, j, k] = 0.0
        return
    div[i, j, k] = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) / h


@wp.func
def accumulate_neighbor(
    p: wp.array3d[float], solid: wp.array3d[wp.int32], i: int, j: int, k: int, acc: wp.vec2
) -> wp.vec2:
    """Accumulate (pressure sum, neighbor count); solid neighbors are Neumann and skipped."""
    if is_solid(solid, i, j, k):
        return acc
    return acc + wp.vec2(pressure_at(p, i, j, k), 1.0)


@wp.kernel
def pressure_rbsor_kernel(
    div: wp.array3d[float],
    solid: wp.array3d[wp.int32],
    h: float,
    omega: float,
    parity: int,
    p: wp.array3d[float],
):
    """One red-black SOR sweep of the pressure Poisson equation."""
    i, j, k = wp.tid()
    if (i + j + k) % 2 != parity:
        return
    if solid[i, j, k] != 0:
        p[i, j, k] = 0.0
        return
    acc = wp.vec2(0.0)
    acc = accumulate_neighbor(p, solid, i - 1, j, k, acc)
    acc = accumulate_neighbor(p, solid, i + 1, j, k, acc)
    acc = accumulate_neighbor(p, solid, i, j - 1, k, acc)
    acc = accumulate_neighbor(p, solid, i, j + 1, k, acc)
    acc = accumulate_neighbor(p, solid, i, j, k - 1, acc)
    acc = accumulate_neighbor(p, solid, i, j, k + 1, acc)
    if acc[1] == 0.0:
        return
    target = (acc[0] - h * h * div[i, j, k]) / acc[1]
    p[i, j, k] = p[i, j, k] + omega * (target - p[i, j, k])


@wp.kernel
def project_kernel(
    vel: wp.array3d[float],
    axis: int,
    p: wp.array3d[float],
    solid: wp.array3d[wp.int32],
    h: float,
):
    i, j, k = wp.tid()
    s = axis_step(axis)
    i0 = i - s[0]
    j0 = j - s[1]
    k0 = k - s[2]
    if is_solid(solid, i, j, k) or is_solid(solid, i0, j0, k0):
        return  # solid faces keep the collider velocity
    vel[i, j, k] = vel[i, j, k] - (pressure_at(p, i, j, k) - pressure_at(p, i0, j0, k0)) / h


# ---------------------------------------------------------------------------
# Tracer particles
# ---------------------------------------------------------------------------


@wp.kernel
def emit_tracers_kernel(
    emitter_position: wp.array[wp.vec3],
    emitter_radius: wp.array[float],
    emitter_rate: wp.array[float],
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
    expected = tracer_rate_scale * wp.max(emitter_rate[e], 0.0) * dt
    state = wp.rand_init(seed[0], tid)
    if wp.randf(state) * float(slots_per_emitter) >= expected:
        return
    slot = wp.atomic_add(head, 0, 1) % particle_q.shape[0]
    r = emitter_radius[e]
    jitter = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
    particle_q[slot] = emitter_position[e] + r * jitter
    particle_qd[slot] = wp.vec3(0.0)
    particle_age[slot] = 0.0


@wp.kernel
def advance_seed_kernel(seed: wp.array[wp.int32], head: wp.array[wp.int32], particle_count: int):
    seed[0] = seed[0] + 1
    head[0] = head[0] % wp.max(particle_count, 1)


@wp.kernel
def advect_tracers_kernel(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    lower: wp.vec3,
    upper: wp.vec3,
    h: float,
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
    inv_h = 1.0 / h
    vel = sample_velocity(u, v, w, lower, inv_h, p)
    mid = p + 0.5 * dt * vel
    vel = sample_velocity(u, v, w, lower, inv_h, mid)
    p = p + dt * vel
    p = wp.vec3(p[0], p[1], wp.max(p[2], lower[2]))
    outside = p[0] < lower[0] or p[1] < lower[1] or p[0] > upper[0] or p[1] > upper[1] or p[2] > upper[2]
    particle_q[tid] = p
    particle_qd[tid] = vel
    particle_age[tid] = wp.where(outside, lifetime, age + dt)


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
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    lower: wp.vec3,
    h: float,
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
        x = x + dt * sample_velocity(u, v, w, lower, 1.0 / h, x)
    # Keep the grain inside the tracer's puff.
    d = x - center
    d_len = wp.length(d)
    if d_len > r:
        x = center + d * (r / d_len)
    grains[pid, g] = wp.vec3(x[0], x[1], wp.max(x[2], lower[2]))
