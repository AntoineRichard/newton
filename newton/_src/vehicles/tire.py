# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pluggable per-wheel tire-force models.

:func:`tire_force` is a Warp function selected per wheel by an integer model id
(matching :class:`newton.vehicles.TireModel`). It maps slip + normal load to a
longitudinal/lateral force pair and a self-aligning moment in the contact
tangent frame, with the force opposing slip and saturating on the ``mu * Fz``
friction circle.

* ``BRUSH`` (default): elastic-bristle brush with parabolic pressure. The force
  magnitude follows the textbook law ``F = mu*Fz*(1 - (1 - z)^3)`` with
  ``z = stiffness*slip / (3*mu*Fz)``, using the canonical theoretical slip
  ``sigma = slip / (1 + kappa)`` (guarded at lock-up). Combined slip is intrinsic.
* ``LINEAR``: linear slip-to-force with a friction-circle clip.
* ``PACEJKA``: core Magic Formula per axis (``D = mu*Fz``) with Pacejka cosine
  combined-slip weighting, using per-wheel parameter rows indexed by the
  ``MF_*`` constants.

The self-aligning moment ``Mz = -F_lat * t`` uses a pneumatic trail ``t`` that
collapses toward zero as the tire saturates, reproducing the rise-then-fall of
the aligning torque.
"""

from __future__ import annotations

import warp as wp

TIRE_BRUSH = wp.constant(0)
TIRE_LINEAR = wp.constant(1)
TIRE_PACEJKA = wp.constant(2)

# mf_params column layout. Order is public API: it must match the field order of
# newton.vehicles.PacejkaParams. Extending the parameter set means appending
# indices and bumping MF_PARAM_COUNT (never reordering).
MF_BX = 0
MF_CX = 1
MF_EX = 2
MF_BY = 3
MF_CY = 4
MF_EY = 5
MF_RBX1 = 6
MF_RBX2 = 7
MF_RBY1 = 8
MF_RBY2 = 9
MF_PARAM_COUNT = 10


@wp.func
def _mf_pure(s: float, b: float, c: float, e: float) -> float:
    # normalized Magic Formula: sin(C*atan(B*s - E*(B*s - atan(B*s)))), peak 1
    bs = b * s
    return wp.sin(c * wp.atan(bs - e * (bs - wp.atan(bs))))


@wp.func
def tire_force(
    model_id: int,
    kappa: float,
    alpha: float,
    fz: float,
    mu: float,
    c_long: float,
    c_lat: float,
    trail: float,
    mf: wp.array2d[wp.float32],
    w: int,
) -> wp.vec3:
    """Tire force and self-aligning moment in the patch tangent frame.

    Args:
        model_id: Tire model selector (``TIRE_BRUSH``, ``TIRE_LINEAR``, or
            ``TIRE_PACEJKA``).
        kappa: Longitudinal slip ratio ``(omega*r - v_long) / max(|v_long|, v_ref)``
            (positive when driving forward).
        alpha: Slip angle [rad], ``atan2(v_lat, max(|v_long|, v_ref))``.
        fz: Normal load [N].
        mu: Friction coefficient.
        c_long: Longitudinal slip stiffness per unit normal load [1/rad];
            the linear-regime slope is ``c_long * fz``.
        c_lat: Lateral slip stiffness per unit normal load [1/rad].
        trail: Pneumatic trail at low slip [m] for the self-aligning moment.
        mf: Per-wheel Magic Formula parameter rows, indexed by the ``MF_*``
            constants; only read by ``TIRE_PACEJKA``.
        w: Wheel index selecting the ``mf`` row.

    Returns:
        ``(F_long, F_lat, Mz)``: longitudinal force [N] (positive forward when
        driving), lateral force [N] (opposes the lateral slip), and self-aligning
        moment [N·m] about the contact normal (opposes the slip angle).
    """
    if fz <= 0.0 or mu <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    limit = mu * fz

    if model_id == TIRE_LINEAR:
        fx = c_long * fz * kappa
        fy = -c_lat * fz * alpha
        mag = wp.sqrt(fx * fx + fy * fy)
        if mag > limit and mag > 1.0e-9:
            s = limit / mag
            fx = fx * s
            fy = fy * s
        util = wp.sqrt(fx * fx + fy * fy) / limit
        mz = -fy * trail * wp.max(1.0 - util, 0.0)
        return wp.vec3(fx, fy, mz)

    if model_id == TIRE_PACEJKA:
        # Core Magic Formula per axis on the same theoretical slip as the brush
        # branch, with Pacejka cosine combined-slip weighting. D = mu*Fz, so each
        # axis is bounded by the friction limit; the combined magnitude is shaped
        # by the r-parameters (the impulse solver's budget projection enforces
        # the hard circle at the vehicle level).
        k = wp.max(kappa, -0.9999)
        inv = 1.0 / (1.0 + k)
        sx = k * inv
        sy = wp.tan(alpha) * inv
        fx0 = limit * _mf_pure(sx, mf[w, MF_BX], mf[w, MF_CX], mf[w, MF_EX])
        fy0 = limit * _mf_pure(sy, mf[w, MF_BY], mf[w, MF_CY], mf[w, MF_EY])
        bxa = mf[w, MF_RBX1] * wp.cos(wp.atan(mf[w, MF_RBX2] * sx))
        byk = mf[w, MF_RBY1] * wp.cos(wp.atan(mf[w, MF_RBY2] * sy))
        fx = wp.cos(wp.atan(bxa * sy)) * fx0
        fy = -wp.cos(wp.atan(byk * sx)) * fy0
        util = wp.min(wp.sqrt(fx * fx + fy * fy) / limit, 1.0)
        mz = -fy * trail * wp.max(1.0 - util, 0.0)
        return wp.vec3(fx, fy, mz)

    # BRUSH: canonical theoretical slip (guarded at lock), parabolic-pressure law.
    k = wp.max(kappa, -0.9999)  # 1 + kappa -> 0 at lock-up; guard the singularity
    inv = 1.0 / (1.0 + k)
    sx = k * inv
    sy = wp.tan(alpha) * inv
    flx = c_long * fz * sx
    fly = c_lat * fz * sy
    flin = wp.sqrt(flx * flx + fly * fly)
    if flin < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    phi = flin / (3.0 * limit)
    if phi < 1.0:
        fmag = 3.0 * limit * phi * (1.0 - phi + phi * phi / 3.0)
    else:
        fmag = limit
    scale = fmag / flin
    fx = scale * flx
    fy = -scale * fly
    util = fmag / limit  # in [0, 1]
    mz = -fy * trail * wp.max(1.0 - util, 0.0)
    return wp.vec3(fx, fy, mz)


@wp.kernel
def _eval_tire_kernel(
    model_id: wp.array[wp.int32],
    kappa: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    fz: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    c_long: wp.array[wp.float32],
    c_lat: wp.array[wp.float32],
    trail: wp.array[wp.float32],
    mf_params: wp.array2d[wp.float32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = tire_force(model_id[i], kappa[i], alpha[i], fz[i], mu[i], c_long[i], c_lat[i], trail[i], mf_params, i)
