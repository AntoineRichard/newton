# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pluggable per-wheel tire-force models.

:func:`tire_force` is a Warp function selected per wheel by an integer model id
(matching :class:`newton.vehicles.TireModel`). It maps slip + normal load to a
longitudinal/lateral force pair in the contact tangent frame, with the force
opposing slip and saturating on the ``mu * Fz`` friction circle.

* ``BRUSH`` (default): elastic-bristle brush with intrinsic combined-slip
  saturation -- the longitudinal/lateral split rides the friction circle without
  a separate clip.
* ``LINEAR``: linear slip-to-force with a friction-circle clip.
"""

from __future__ import annotations

import warp as wp

TIRE_BRUSH = wp.constant(0)
TIRE_LINEAR = wp.constant(1)


@wp.func
def tire_force(
    model_id: int,
    kappa: float,
    alpha: float,
    fz: float,
    mu: float,
    c_long: float,
    c_lat: float,
) -> wp.vec2:
    """Tire force in the patch tangent frame.

    Args:
        model_id: Tire model selector (``TIRE_BRUSH`` or ``TIRE_LINEAR``).
        kappa: Longitudinal slip ``(omega*r - v_long) / max(|v_long|, v_ref)``
            (positive when driving forward).
        alpha: Slip angle [rad], ``atan2(v_lat, max(|v_long|, v_ref))``.
        fz: Normal load [N].
        mu: Friction coefficient.
        c_long: Longitudinal slip stiffness [N].
        c_lat: Lateral slip stiffness [N].

    Returns:
        ``(F_long, F_lat)`` [N]: longitudinal force (positive forward when
        driving) and lateral force (opposes the lateral slip).
    """
    if fz <= 0.0 or mu <= 0.0:
        return wp.vec2(0.0, 0.0)
    limit = mu * fz

    if model_id == TIRE_LINEAR:
        fx = c_long * kappa
        fy = -c_lat * alpha
        mag = wp.sqrt(fx * fx + fy * fy)
        if mag > limit and mag > 1.0e-9:
            s = limit / mag
            fx = fx * s
            fy = fy * s
        return wp.vec2(fx, fy)

    # BRUSH: theoretical slip, then parabolic-pressure brush saturation.
    inv = 1.0 / (1.0 + wp.abs(kappa))
    sx = kappa * inv
    sy = wp.tan(alpha) * inv
    flx = c_long * sx
    fly = c_lat * sy
    flin = wp.sqrt(flx * flx + fly * fly)
    if flin < 1.0e-9:
        return wp.vec2(0.0, 0.0)
    phi = flin / (3.0 * limit)
    if phi < 1.0:
        fmag = 3.0 * limit * phi * (1.0 - phi + phi * phi / 3.0)
    else:
        fmag = limit
    scale = fmag / flin
    return wp.vec2(scale * flx, -scale * fly)


@wp.kernel
def _eval_tire_kernel(
    model_id: wp.array[wp.int32],
    kappa: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    fz: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    c_long: wp.array[wp.float32],
    c_lat: wp.array[wp.float32],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = tire_force(model_id[i], kappa[i], alpha[i], fz[i], mu[i], c_long[i], c_lat[i])
