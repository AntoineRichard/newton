# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-wheel implicit impulse-budget tire solve.

Pure Warp math with no model/state knowledge. The tire's tangential action on
the wheel body is computed as an *impulse* over the substep, solved implicitly
in the slip velocities and projected onto the friction circle
``|p| <= mu * Fz * dt``. By construction no substep can apply more tangential
impulse than the contact can absorb, which removes the saturated-force sign
chatter that made explicit injection explode at high grip (see
``docs/superpowers/specs/2026-07-03-wheeled-vehicle-implicit-tire-core-design.md``).

Conventions: slip velocity ``u = v_contact - omega * r`` (tire force opposes
``u``); ``A`` is the slip-space Delassus (inverse effective mass) so that
``u_new = u + A @ p`` for a tire impulse ``p`` on the wheel body.
"""

from __future__ import annotations

import warp as wp

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def wheel_effective_mass(
    m_inv: float,
    i_inv_world: wp.mat33,
    offset: wp.vec3,
    t_fwd: wp.vec3,
    t_lat: wp.vec3,
) -> wp.vec3:
    """Tangential Delassus block ``W = J M^-1 J^T`` of the free wheel body.

    ``offset`` is the contact point relative to the body COM [m]; the returned
    ``(W11, W12, W22)`` maps a tangential impulse [N·s] at the contact to the
    contact-point velocity change [m/s] (1 = t_fwd, 2 = t_lat). The free-body
    block ignores joint constraints, which can only increase effective mass, so
    impulses computed against it are always absorbable — a stable-side error.
    """
    ru = wp.cross(offset, t_fwd)
    rv = wp.cross(offset, t_lat)
    w11 = m_inv + wp.dot(ru, i_inv_world * ru)
    w12 = wp.dot(ru, i_inv_world * rv)
    w22 = m_inv + wp.dot(rv, i_inv_world * rv)
    return wp.vec3(w11, w12, w22)


@wp.func
def solve_tire_impulse(
    u_long: float,
    u_lat: float,
    a11: float,
    a12: float,
    a22: float,
    k_long: float,
    k_lat: float,
    budget: float,
    budget_stick: float,
) -> vec6:
    """Implicit tire impulse with stick test and friction-circle projection.

    Args:
        u_long: Free longitudinal slip velocity [m/s].
        u_lat: Free lateral slip velocity [m/s].
        a11: Slip-space Delassus (1,1) [(m/s)/(N·s)] (includes spin mobility).
        a12: Slip-space Delassus (1,2).
        a22: Slip-space Delassus (2,2).
        k_long: Longitudinal secant impulse stiffness ``dt*C`` [N·s/(m/s)].
        k_lat: Lateral secant impulse stiffness [N·s/(m/s)].
        budget: Kinetic friction-circle impulse budget ``mu*Fz*dt`` [N·s].
        budget_stick: Static budget ``mu_s*Fz*dt`` [N·s].

    Returns:
        ``(p_long, p_lat, u_long_new, u_lat_new, stick, utilization)``:
        tire impulse on the wheel body [N·s], post-solve slip velocities [m/s],
        stick flag (1.0 when the stick solution was taken), and
        ``|p| / budget`` clamped to [0, 1].
    """
    if budget <= 0.0:
        return vec6(0.0, 0.0, u_long, u_lat, 0.0, 0.0)

    # Stick first: the impulse that zeroes the slip velocity, A p = -u.
    det_a = a11 * a22 - a12 * a12
    det_a = wp.max(det_a, 1.0e-12)
    ps1 = -(a22 * u_long - a12 * u_lat) / det_a
    ps2 = -(a11 * u_lat - a12 * u_long) / det_a
    ps_norm = wp.sqrt(ps1 * ps1 + ps2 * ps2)
    if ps_norm <= budget_stick:
        util = wp.min(ps_norm / budget, 1.0)
        return vec6(ps1, ps2, 0.0, 0.0, 1.0, util)

    # Slip: p = -K u_new, (I + A K) u_new = u, K = diag(k_long, k_lat).
    b11 = 1.0 + a11 * k_long
    b12 = a12 * k_lat
    b21 = a12 * k_long
    b22 = 1.0 + a22 * k_lat
    det_b = wp.max(b11 * b22 - b12 * b21, 1.0e-12)
    un1 = (b22 * u_long - b12 * u_lat) / det_b
    un2 = (b11 * u_lat - b21 * u_long) / det_b
    p1 = -k_long * un1
    p2 = -k_lat * un2
    p_norm = wp.sqrt(p1 * p1 + p2 * p2)
    if p_norm > budget:
        s = budget / wp.max(p_norm, 1.0e-12)
        p1 = p1 * s
        p2 = p2 * s
        # Recompute the post-impulse slip consistently with the clamped impulse.
        un1 = u_long + a11 * p1 + a12 * p2
        un2 = u_lat + a12 * p1 + a22 * p2
        p_norm = budget
    util = wp.min(p_norm / budget, 1.0)
    return vec6(p1, p2, un1, un2, 0.0, util)
