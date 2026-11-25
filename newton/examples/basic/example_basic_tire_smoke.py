# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Basic Tire Smoke

A small car spins donuts while its rear tires emit smoke. The smoke lives on
an Eulerian grid (stable fluids on a staggered MAC grid): semi-Lagrangian
advection, buoyancy, vorticity confinement, and a red-black SOR pressure
projection. The car chassis is a moving solid obstacle, so it pushes the air
around and leaves a wake. Coupling is one way: the smoke never affects the car.

Smoke is emitted where a wheel's contact point slides over the ground, so the
``TireSmoke`` class works with any Newton state that has wheel bodies. It is
rendered with tracer particles carried by the grid velocity.

Success criteria:
- Rear tires emit smoke as the car spins; the smoke rises, curls, and drifts.
- The grid fields stay finite and the velocities stay bounded.

Runs:
- `python -m newton.examples basic_tire_smoke`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

# Sample offsets (in cells) of each staggered field relative to the grid lower corner.
OFFSET_U = wp.constant(wp.vec3(0.0, 0.5, 0.5))
OFFSET_V = wp.constant(wp.vec3(0.5, 0.0, 0.5))
OFFSET_W = wp.constant(wp.vec3(0.5, 0.5, 0.0))
OFFSET_C = wp.constant(wp.vec3(0.5, 0.5, 0.5))


@wp.func
def sample_grid(f: wp.array3d[float], g: wp.vec3) -> float:
    """Trilinearly sample ``f`` at index-space coordinate ``g`` (clamped to the array)."""
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
    q = (p - lower) * inv_h
    return wp.vec3(sample_grid(u, q - OFFSET_U), sample_grid(v, q - OFFSET_V), sample_grid(w, q - OFFSET_W))


@wp.func
def is_solid(solid: wp.array3d[wp.int32], i: int, j: int, k: int) -> bool:
    """Below the grid is ground (solid); the sides and top of the grid are open air."""
    if k < 0:
        return True
    if i < 0 or j < 0 or i >= solid.shape[0] or j >= solid.shape[1] or k >= solid.shape[2]:
        return False
    return solid[i, j, k] != 0


@wp.func
def pressure_at(p: wp.array3d[float], i: int, j: int, k: int) -> float:
    """Pressure outside the grid is zero (open boundary)."""
    if i < 0 or j < 0 or k < 0 or i >= p.shape[0] or j >= p.shape[1] or k >= p.shape[2]:
        return 0.0
    return p[i, j, k]


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


# ---------------------------------------------------------------------------
# Smoke kernels
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
    i, j, k = wp.tid()
    inv_h = 1.0 / h
    p = lower + (wp.vec3(float(i), float(j), float(k)) + offset) * h
    # RK2 backtrace
    vel = sample_velocity(u, v, w, lower, inv_h, p)
    mid = p - 0.5 * dt * vel
    vel = sample_velocity(u, v, w, lower, inv_h, mid)
    back = p - dt * vel
    dst[i, j, k] = decay * sample_grid(src, (back - lower) * inv_h - offset)


@wp.kernel
def mark_solid_kernel(
    lower: wp.vec3,
    h: float,
    obstacle_xform: wp.transform,
    half_extents: wp.vec3,
    solid: wp.array3d[wp.int32],
):
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + OFFSET_C) * h
    local = wp.transform_point(wp.transform_inverse(obstacle_xform), p)
    inside = (
        wp.abs(local[0]) < half_extents[0] and wp.abs(local[1]) < half_extents[1] and wp.abs(local[2]) < half_extents[2]
    )
    solid[i, j, k] = wp.where(inside, 1, 0)


@wp.kernel
def inject_scalars_kernel(
    lower: wp.vec3,
    h: float,
    src_pos: wp.array[wp.vec3],
    src_strength: wp.array[float],
    radius: float,
    density_rate: float,
    heat_rate: float,
    dt: float,
    density: wp.array3d[float],
    temperature: wp.array3d[float],
):
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + OFFSET_C) * h
    add = float(0.0)
    for s in range(src_pos.shape[0]):
        weight = wp.max(0.0, 1.0 - wp.length(p - src_pos[s]) / radius)
        add += weight * src_strength[s]
    if add > 0.0:
        density[i, j, k] = density[i, j, k] + dt * density_rate * add
        temperature[i, j, k] = temperature[i, j, k] + dt * heat_rate * add


@wp.kernel
def inject_velocity_kernel(
    vel: wp.array3d[float],
    axis: int,
    lower: wp.vec3,
    h: float,
    src_pos: wp.array[wp.vec3],
    src_strength: wp.array[float],
    src_vel: wp.array[wp.vec3],
    radius: float,
):
    i, j, k = wp.tid()
    p = lower + (wp.vec3(float(i), float(j), float(k)) + axis_offset(axis)) * h
    for s in range(src_pos.shape[0]):
        weight = wp.max(0.0, 1.0 - wp.length(p - src_pos[s]) / radius) * src_strength[s]
        if weight > 0.0:
            vel[i, j, k] = wp.lerp(vel[i, j, k], src_vel[s][axis], wp.min(weight, 1.0))


@wp.kernel
def buoyancy_kernel(
    density: wp.array3d[float],
    temperature: wp.array3d[float],
    weight: float,
    lift: float,
    dt: float,
    w: wp.array3d[float],
):
    i, j, k = wp.tid()
    nz = density.shape[2]
    k0 = wp.max(k - 1, 0)
    k1 = wp.min(k, nz - 1)
    rho = 0.5 * (density[i, j, k0] + density[i, j, k1])
    temp = 0.5 * (temperature[i, j, k0] + temperature[i, j, k1])
    w[i, j, k] = w[i, j, k] + dt * (lift * temp - weight * rho)


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
    n = wp.vec3i(f.shape[0], f.shape[1], f.shape[2])
    lo = wp.vec3i(wp.max(i - s[0], 0), wp.max(j - s[1], 0), wp.max(k - s[2], 0))
    hi = wp.vec3i(wp.min(i + s[0], n[0] - 1), wp.min(j + s[1], n[1] - 1), wp.min(k + s[2], n[2] - 1))
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
    i, j, k = wp.tid()
    nx = curl.shape[0]
    ny = curl.shape[1]
    nz = curl.shape[2]
    gx = curl_mag[wp.min(i + 1, nx - 1), j, k] - curl_mag[wp.max(i - 1, 0), j, k]
    gy = curl_mag[i, wp.min(j + 1, ny - 1), k] - curl_mag[i, wp.max(j - 1, 0), k]
    gz = curl_mag[i, j, wp.min(k + 1, nz - 1)] - curl_mag[i, j, wp.max(k - 1, 0)]
    g = wp.vec3(gx, gy, gz)
    g_len = wp.length(g)
    if g_len < 1.0e-6:
        force[i, j, k] = wp.vec3(0.0)
        return
    # Only stir where there is smoke, so clean air stays calm.
    smoke = wp.min(density[i, j, k], 1.0)
    force[i, j, k] = smoke * epsilon * h * wp.cross(g / g_len, curl[i, j, k])


@wp.kernel
def apply_cell_force_kernel(force: wp.array3d[wp.vec3], axis: int, dt: float, vel: wp.array3d[float]):
    i, j, k = wp.tid()
    s = axis_step(axis)
    n = wp.vec3i(force.shape[0], force.shape[1], force.shape[2])
    i0 = wp.max(i - s[0], 0)
    j0 = wp.max(j - s[1], 0)
    k0 = wp.max(k - s[2], 0)
    i1 = wp.min(i, n[0] - 1)
    j1 = wp.min(j, n[1] - 1)
    k1 = wp.min(k, n[2] - 1)
    f = 0.5 * (force[i0, j0, k0][axis] + force[i1, j1, k1][axis])
    vel[i, j, k] = vel[i, j, k] + dt * f


@wp.kernel
def enforce_solid_faces_kernel(
    vel: wp.array3d[float],
    axis: int,
    solid: wp.array3d[wp.int32],
    lower: wp.vec3,
    h: float,
    obstacle_com: wp.vec3,
    obstacle_lin: wp.vec3,
    obstacle_ang: wp.vec3,
):
    i, j, k = wp.tid()
    s = axis_step(axis)
    if axis == 2 and k == 0:
        vel[i, j, k] = 0.0  # ground
        return
    if not is_solid(solid, i, j, k) and not is_solid(solid, i - s[0], j - s[1], k - s[2]):
        return
    p = lower + (wp.vec3(float(i), float(j), float(k)) + axis_offset(axis)) * h
    vel[i, j, k] = (obstacle_lin + wp.cross(obstacle_ang, p - obstacle_com))[axis]


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
        return  # solid faces keep the velocity set by enforce_solid_faces_kernel
    vel[i, j, k] = vel[i, j, k] - (pressure_at(p, i, j, k) - pressure_at(p, i0, j0, k0)) / h


@wp.kernel
def wheel_sources_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    wheel_bodies: wp.array[wp.int32],
    wheel_radius: float,
    ground_height: float,
    contact_tolerance: float,
    slip_threshold: float,
    slip_full: float,
    kick: float,
    src_pos: wp.array[wp.vec3],
    src_strength: wp.array[float],
    src_vel: wp.array[wp.vec3],
):
    """Turn each wheel's contact-point slip into a smoke source."""
    tid = wp.tid()
    b = wheel_bodies[tid]
    X = body_q[b]
    center = wp.transform_get_translation(X)
    com = wp.transform_point(X, body_com[b])
    lin = wp.spatial_top(body_qd[b])
    ang = wp.spatial_bottom(body_qd[b])
    contact = wp.vec3(center[0], center[1], ground_height)
    v = lin + wp.cross(ang, contact - com)
    slip = wp.length(wp.vec3(v[0], v[1], 0.0))
    touching = center[2] - wheel_radius < ground_height + contact_tolerance
    strength = wp.clamp((slip - slip_threshold) / (slip_full - slip_threshold), 0.0, 1.0)
    src_strength[tid] = wp.where(touching, strength, 0.0)
    # The tire surface drags the smoke along with it and the hot air rises.
    src_pos[tid] = contact + wp.vec3(0.0, 0.0, 0.5 * wheel_radius)
    src_vel[tid] = kick * wp.vec3(v[0], v[1], 0.0) + wp.vec3(0.0, 0.0, 0.2 * slip * kick)


@wp.kernel
def emit_tracers_kernel(
    src_pos: wp.array[wp.vec3],
    src_strength: wp.array[float],
    per_source: int,
    head: int,
    radius: float,
    seed: int,
    pos: wp.array[wp.vec3],
    age: wp.array[float],
):
    tid = wp.tid()
    s = tid // per_source
    state = wp.rand_init(seed, tid)
    if wp.randf(state) >= src_strength[s]:
        return
    idx = (head + tid) % pos.shape[0]
    jitter = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
    pos[idx] = src_pos[s] + 2.0 * radius * jitter
    age[idx] = 0.0


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
    pos: wp.array[wp.vec3],
    age: wp.array[float],
):
    tid = wp.tid()
    if age[tid] >= lifetime:
        return
    inv_h = 1.0 / h
    p = pos[tid]
    vel = sample_velocity(u, v, w, lower, inv_h, p)
    mid = p + 0.5 * dt * vel
    p = p + dt * sample_velocity(u, v, w, lower, inv_h, mid)
    p = wp.vec3(p[0], p[1], wp.max(p[2], lower[2]))
    pos[tid] = p
    outside = p[0] < lower[0] or p[1] < lower[1] or p[0] > upper[0] or p[1] > upper[1] or p[2] > upper[2]
    age[tid] = wp.where(outside, lifetime, age[tid] + dt)


@wp.kernel
def tracer_style_kernel(
    age: wp.array[float],
    lifetime: float,
    radius_start: float,
    radius_end: float,
    max_opacity: float,
    pos: wp.array[wp.vec3],
    xforms: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[float],
):
    tid = wp.tid()
    t = age[tid] / lifetime
    if t >= 1.0:
        scales[tid] = wp.vec3(0.0)
        opacities[tid] = 0.0
        return
    xforms[tid] = wp.transform(pos[tid], wp.quat_identity())
    scales[tid] = wp.vec3(wp.lerp(radius_start, radius_end, wp.sqrt(t)))
    colors[tid] = wp.vec3(wp.lerp(0.55, 0.85, t))
    # Fade in quickly, then thin out as the puff grows.
    opacities[tid] = max_opacity * wp.min(t * 20.0, 1.0) * (1.0 - t) * (1.0 - t)


# ---------------------------------------------------------------------------
# Smoke solver
# ---------------------------------------------------------------------------


class TireSmoke:
    """Grid smoke driven by sliding wheels, coupled one way to a Newton state.

    The grid is axis aligned and Z up. The ground is the bottom face of the grid;
    the sides and the top are open. One box attached to a body (the chassis)
    acts as a moving obstacle.

    Args:
        lower: Lower corner of the grid [m]. Its z value is the ground height.
        upper: Upper corner of the grid [m].
        cell_size: Grid cell edge length [m].
        wheel_bodies: Body indices of the wheels that can emit smoke.
        wheel_radius: Wheel radius [m].
        obstacle_body: Body index of the obstacle box, or -1 for none.
        obstacle_half_extents: Half extents of the obstacle box in its body frame [m].
        max_tracers: Number of tracer particles used for rendering.
        device: Warp device.
    """

    def __init__(
        self,
        lower,
        upper,
        cell_size: float,
        wheel_bodies,
        wheel_radius: float,
        obstacle_body: int = -1,
        obstacle_half_extents=(0.0, 0.0, 0.0),
        max_tracers: int = 40000,
        device=None,
    ):
        self.device = wp.get_device(device)
        self.h = float(cell_size)
        self.lower = wp.vec3(*lower)
        dims = [max(2, round((upper[a] - lower[a]) / self.h)) for a in range(3)]
        self.dims = tuple(dims)
        self.upper = wp.vec3(*(lower[a] + dims[a] * self.h for a in range(3)))
        nx, ny, nz = self.dims

        # Tunables, in SI units where they have them.
        self.density_rate = 30.0  # smoke added per second at full slip
        self.heat_rate = 30.0  # heat added per second at full slip
        self.source_radius = 2.5 * self.h
        self.buoyancy_lift = 4.0  # [m/s^2 per unit temperature]
        self.buoyancy_weight = 0.05  # [m/s^2 per unit density]
        self.density_decay = 0.15  # [1/s]
        self.temperature_decay = 0.4  # [1/s]
        self.velocity_decay = 0.1  # [1/s]
        self.vorticity = 2.0
        self.pressure_iterations = 30
        self.sor_omega = 1.7
        self.slip_threshold = 0.3  # [m/s] below this a tire makes no smoke
        self.slip_full = 2.0  # [m/s] slip at which emission saturates
        self.kick = 0.3  # fraction of the slip velocity given to fresh smoke
        self.tracer_lifetime = 4.0  # [s]
        self.tracers_per_source = 24  # per step, at full slip
        self.tracer_radius = (0.5 * self.h, 3.0 * self.h)
        self.tracer_opacity = 0.12

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.u = zeros((nx + 1, ny, nz))
        self.v = zeros((nx, ny + 1, nz))
        self.w = zeros((nx, ny, nz + 1))
        self.u_tmp = zeros(self.u.shape)
        self.v_tmp = zeros(self.v.shape)
        self.w_tmp = zeros(self.w.shape)
        self.density = zeros(self.dims)
        self.temperature = zeros(self.dims)
        self.scalar_tmp = zeros(self.dims)
        self.pressure = zeros(self.dims)
        self.div = zeros(self.dims)
        self.solid = zeros(self.dims, dtype=wp.int32)
        self.center_vel = zeros(self.dims, dtype=wp.vec3)
        self.curl = zeros(self.dims, dtype=wp.vec3)
        self.curl_mag = zeros(self.dims)
        self.force = zeros(self.dims, dtype=wp.vec3)

        self.wheel_bodies = wp.array(wheel_bodies, dtype=wp.int32, device=self.device)
        self.wheel_radius = float(wheel_radius)
        n_src = len(wheel_bodies)
        self.src_pos = zeros(n_src, dtype=wp.vec3)
        self.src_strength = zeros(n_src)
        self.src_vel = zeros(n_src, dtype=wp.vec3)

        self.obstacle_body = int(obstacle_body)
        self.obstacle_half_extents = wp.vec3(*obstacle_half_extents)

        self.tracer_pos = wp.full(max_tracers, self.lower, dtype=wp.vec3, device=self.device)
        self.tracer_age = wp.full(max_tracers, self.tracer_lifetime, dtype=float, device=self.device)
        self.tracer_xforms = zeros(max_tracers, dtype=wp.transform)
        self.tracer_scales = zeros(max_tracers, dtype=wp.vec3)
        self.tracer_colors = zeros(max_tracers, dtype=wp.vec3)
        self.tracer_opacities = zeros(max_tracers)
        # Matte: (roughness, metallic, checker, texture) as the viewer expects.
        self.tracer_materials = wp.full(max_tracers, wp.vec4(1.0, 0.0, 0.0, 0.0), dtype=wp.vec4, device=self.device)
        self._puff_viewer = None
        self._tracer_head = 0
        self._seed = 0

    @property
    def cell_count(self) -> int:
        return self.dims[0] * self.dims[1] * self.dims[2]

    def _faces(self):
        return ((self.u, 0), (self.v, 1), (self.w, 2))

    def step(self, model: newton.Model, state: newton.State, dt: float):
        """Advance the smoke by ``dt`` using wheel and obstacle motion from ``state``."""
        h = self.h
        dev = self.device

        # Sources from wheel slip.
        wp.launch(
            wheel_sources_kernel,
            dim=len(self.wheel_bodies),
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                self.wheel_bodies,
                self.wheel_radius,
                float(self.lower[2]),
                0.25 * self.wheel_radius,
                self.slip_threshold,
                self.slip_full,
                self.kick,
            ],
            outputs=[self.src_pos, self.src_strength, self.src_vel],
            device=dev,
        )

        # Obstacle state (read back once per step: a single body).
        if self.obstacle_body >= 0:
            X = state.body_q[self.obstacle_body : self.obstacle_body + 1].numpy()[0]
            qd = state.body_qd[self.obstacle_body : self.obstacle_body + 1].numpy()[0]
            com_local = model.body_com[self.obstacle_body : self.obstacle_body + 1].numpy()[0]
            xform = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
            com = wp.transform_point(xform, wp.vec3(*com_local))
            lin, ang = wp.vec3(*qd[:3]), wp.vec3(*qd[3:])
            half = self.obstacle_half_extents
        else:
            xform, com, lin, ang, half = wp.transform_identity(), wp.vec3(), wp.vec3(), wp.vec3(), wp.vec3()
        wp.launch(
            mark_solid_kernel, dim=self.dims, inputs=[self.lower, h, xform, half], outputs=[self.solid], device=dev
        )

        # Advect velocity (all components read the old field) and scalars.
        vel_decay = math.exp(-self.velocity_decay * dt)
        for (src, axis), dst in zip(self._faces(), (self.u_tmp, self.v_tmp, self.w_tmp), strict=True):
            wp.launch(
                advect_kernel,
                dim=src.shape,
                inputs=[src, self.u, self.v, self.w, self.lower, h, axis_offset_host(axis), dt, vel_decay],
                outputs=[dst],
                device=dev,
            )
        self.u, self.u_tmp = self.u_tmp, self.u
        self.v, self.v_tmp = self.v_tmp, self.v
        self.w, self.w_tmp = self.w_tmp, self.w

        for name, decay in (("density", self.density_decay), ("temperature", self.temperature_decay)):
            src = getattr(self, name)
            wp.launch(
                advect_kernel,
                dim=self.dims,
                inputs=[src, self.u, self.v, self.w, self.lower, h, OFFSET_C, dt, math.exp(-decay * dt)],
                outputs=[self.scalar_tmp],
                device=dev,
            )
            setattr(self, name, self.scalar_tmp)
            self.scalar_tmp = src

        # Sources and forces.
        wp.launch(
            inject_scalars_kernel,
            dim=self.dims,
            inputs=[
                self.lower,
                h,
                self.src_pos,
                self.src_strength,
                self.source_radius,
                self.density_rate,
                self.heat_rate,
                dt,
            ],
            outputs=[self.density, self.temperature],
            device=dev,
        )
        for vel, axis in self._faces():
            wp.launch(
                inject_velocity_kernel,
                dim=vel.shape,
                inputs=[vel, axis, self.lower, h, self.src_pos, self.src_strength, self.src_vel, self.source_radius],
                device=dev,
            )
        wp.launch(
            buoyancy_kernel,
            dim=self.w.shape,
            inputs=[self.density, self.temperature, self.buoyancy_weight, self.buoyancy_lift, dt],
            outputs=[self.w],
            device=dev,
        )
        if self.vorticity > 0.0:
            wp.launch(
                center_velocity_kernel,
                dim=self.dims,
                inputs=[self.u, self.v, self.w],
                outputs=[self.center_vel],
                device=dev,
            )
            wp.launch(
                curl_kernel, dim=self.dims, inputs=[self.center_vel, h], outputs=[self.curl, self.curl_mag], device=dev
            )
            wp.launch(
                confinement_kernel,
                dim=self.dims,
                inputs=[self.curl, self.curl_mag, self.density, h, self.vorticity],
                outputs=[self.force],
                device=dev,
            )
            for vel, axis in self._faces():
                wp.launch(
                    apply_cell_force_kernel, dim=vel.shape, inputs=[self.force, axis, dt], outputs=[vel], device=dev
                )

        # Pressure projection.
        for vel, axis in self._faces():
            wp.launch(
                enforce_solid_faces_kernel,
                dim=vel.shape,
                inputs=[vel, axis, self.solid, self.lower, h, com, lin, ang],
                device=dev,
            )
        wp.launch(
            divergence_kernel,
            dim=self.dims,
            inputs=[self.u, self.v, self.w, self.solid, h],
            outputs=[self.div],
            device=dev,
        )
        for _ in range(self.pressure_iterations):
            for parity in (0, 1):
                wp.launch(
                    pressure_rbsor_kernel,
                    dim=self.dims,
                    inputs=[self.div, self.solid, h, self.sor_omega, parity],
                    outputs=[self.pressure],
                    device=dev,
                )
        for vel, axis in self._faces():
            wp.launch(project_kernel, dim=vel.shape, inputs=[vel, axis, self.pressure, self.solid, h], device=dev)

        # Tracers for rendering.
        n_emit = len(self.wheel_bodies) * self.tracers_per_source
        wp.launch(
            emit_tracers_kernel,
            dim=n_emit,
            inputs=[
                self.src_pos,
                self.src_strength,
                self.tracers_per_source,
                self._tracer_head,
                self.source_radius,
                self._seed,
            ],
            outputs=[self.tracer_pos, self.tracer_age],
            device=dev,
        )
        self._tracer_head = (self._tracer_head + n_emit) % len(self.tracer_pos)
        self._seed += 1
        wp.launch(
            advect_tracers_kernel,
            dim=len(self.tracer_pos),
            inputs=[self.u, self.v, self.w, self.lower, self.upper, h, dt, self.tracer_lifetime],
            outputs=[self.tracer_pos, self.tracer_age],
            device=dev,
        )

    def render(self, viewer):
        """Draw tracers as translucent puffs: instanced unit spheres with per-instance opacity."""
        if self._puff_viewer is not viewer:
            mesh = newton.Mesh.create_sphere(1.0, num_latitudes=8, num_longitudes=10, compute_inertia=False)
            viewer.log_mesh(
                "/smoke_puff",
                wp.array(mesh.vertices, dtype=wp.vec3, device=self.device),
                wp.array(mesh.indices, dtype=wp.int32, device=self.device),
                wp.array(mesh.normals, dtype=wp.vec3, device=self.device),
                hidden=True,
            )
            self._puff_viewer = viewer
        wp.launch(
            tracer_style_kernel,
            dim=len(self.tracer_pos),
            inputs=[self.tracer_age, self.tracer_lifetime, *self.tracer_radius, self.tracer_opacity, self.tracer_pos],
            outputs=[self.tracer_xforms, self.tracer_scales, self.tracer_colors, self.tracer_opacities],
            device=self.device,
        )
        viewer.log_instances(
            "/smoke",
            "/smoke_puff",
            self.tracer_xforms,
            self.tracer_scales,
            self.tracer_colors,
            self.tracer_materials,
            opacities=self.tracer_opacities,
        )


def axis_offset_host(axis: int) -> wp.vec3:
    return (OFFSET_U, OFFSET_V, OFFSET_W)[axis]


# ---------------------------------------------------------------------------
# Kinematic donut car
# ---------------------------------------------------------------------------


@wp.kernel
def donut_car_kernel(
    yaw: float,
    yaw_rate: float,
    pivot: wp.vec3,
    pivot_local: wp.vec3,
    wheel_local: wp.array[wp.vec3],
    wheel_spin: wp.array[float],
    wheel_driven: wp.array[wp.int32],
    rear_spin_rate: float,
    wheel_radius: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Chassis yaws about a point under its front axle; rear wheels spin up and slide.

    Body 0 is the chassis, bodies 1..4 are the wheels.
    """
    tid = wp.tid()
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    origin = pivot - wp.quat_rotate(rot, pivot_local)
    ang = wp.vec3(0.0, 0.0, yaw_rate)
    if tid == 0:
        body_q[0] = wp.transform(origin, rot)
        body_qd[0] = wp.spatial_vector(wp.cross(ang, origin - pivot), ang)
        return
    wi = tid - 1
    center = origin + wp.quat_rotate(rot, wheel_local[wi])
    lin = wp.cross(ang, center - pivot)
    axle = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
    if wheel_driven[wi] != 0:
        spin_rate = rear_spin_rate
    else:
        # Free wheels roll with the ground.
        spin_rate = wp.dot(lin, wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))) / wheel_radius
    spin = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wheel_spin[wi])
    body_q[tid] = wp.transform(center, rot * spin)
    body_qd[tid] = wp.spatial_vector(lin, ang + spin_rate * axle)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer

        # A small, RC-sized car.
        half = (0.2, 0.1, 0.035)
        ride_height = 0.08
        self.wheel_radius = 0.05
        wheel_x, wheel_y = 0.14, 0.125
        wheel_local = [
            (wheel_x, wheel_y, self.wheel_radius - ride_height),
            (wheel_x, -wheel_y, self.wheel_radius - ride_height),
            (-wheel_x, wheel_y, self.wheel_radius - ride_height),
            (-wheel_x, -wheel_y, self.wheel_radius - ride_height),
        ]
        driven = [0, 0, 1, 1]

        builder = newton.ModelBuilder()
        chassis = builder.add_body(xform=wp.transform((0.0, 0.0, ride_height)), is_kinematic=True, label="chassis")
        builder.add_shape_box(chassis, hx=half[0], hy=half[1], hz=half[2], color=(0.8, 0.15, 0.1))
        wheel_xform = wp.transform(q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi))
        wheels = []
        for i in range(4):
            b = builder.add_body(is_kinematic=True, label=f"wheel_{i}")
            builder.add_shape_cylinder(
                b, xform=wheel_xform, radius=self.wheel_radius, half_height=0.02, color=(0.1, 0.1, 0.1)
            )
            # A light marker so the spin is visible.
            builder.add_shape_box(
                b, xform=wp.transform((0.0, 0.0, 0.03)), hx=0.008, hy=0.021, hz=0.015, color=(0.9, 0.9, 0.9)
            )
            wheels.append(b)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.state = self.model.state()

        self.pivot = wp.vec3(0.0, 0.0, ride_height)
        self.pivot_local = wp.vec3(wheel_x, 0.0, 0.0)
        self.max_yaw_rate = 5.0  # [rad/s]
        self.rear_spin_rate = 6.0 / self.wheel_radius  # rear tire surface at 6 m/s
        self.yaw = 0.0
        self.wheel_local = wp.array(wheel_local, dtype=wp.vec3)
        self.wheel_driven = wp.array(driven, dtype=wp.int32)
        self.wheel_spin_host = np.zeros(4, dtype=np.float32)
        self.wheel_spin = wp.array(self.wheel_spin_host, dtype=float)

        self.smoke = TireSmoke(
            lower=(-1.5, -1.5, 0.0),
            upper=(1.5, 1.5, 1.2),
            cell_size=0.03,
            wheel_bodies=[wheels[2], wheels[3]],
            wheel_radius=self.wheel_radius,
            obstacle_body=chassis,
            obstacle_half_extents=half,
        )

        self._update_car(0.0, 0.0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-1.5, -1.5, 0.8), pitch=-20.0, yaw=45.0)

    def _update_car(self, yaw_rate: float, ramp: float):
        wp.launch(
            donut_car_kernel,
            dim=5,
            inputs=[
                self.yaw,
                yaw_rate,
                self.pivot,
                self.pivot_local,
                self.wheel_local,
                self.wheel_spin,
                self.wheel_driven,
                self.rear_spin_rate * ramp,
                self.wheel_radius,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
        )

    def step(self):
        dt = self.frame_dt
        ramp = min(self.sim_time / 1.0, 1.0)
        yaw_rate = self.max_yaw_rate * ramp
        self.yaw += yaw_rate * dt
        # Integrate wheel spin angles on the host from the velocities we just set.
        qd = self.state.body_qd.numpy()[1:5]
        q = self.state.body_q.numpy()[1:5]
        for i in range(4):
            axle = wp.quat_rotate(wp.quat(*q[i, 3:]), wp.vec3(0.0, 1.0, 0.0))
            spin_rate = float(np.dot(qd[i, 3:], np.array(axle)))
            self.wheel_spin_host[i] += spin_rate * dt
        self.wheel_spin.assign(self.wheel_spin_host)
        self._update_car(yaw_rate, ramp)

        self.smoke.step(self.model, self.state, dt)
        self.sim_time += dt

    def test_final(self):
        density = self.smoke.density.numpy()
        assert np.all(np.isfinite(density)), "smoke density is not finite"
        assert density.sum() > 1.0, "the tires emitted no smoke"
        for vel in (self.smoke.u, self.smoke.v, self.smoke.w):
            assert np.all(np.abs(vel.numpy()) < 50.0), "smoke velocity blew up"
        alive = np.count_nonzero(self.smoke.tracer_age.numpy() < self.smoke.tracer_lifetime)
        assert alive > 0, "no smoke tracers alive"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.smoke.render(self.viewer)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
