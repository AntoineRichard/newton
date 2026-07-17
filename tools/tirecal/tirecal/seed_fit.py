# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Least-squares MF-to-brush curve fit used to seed the trajectory optimizer.

Standalone: numpy only. Both tire laws scale linearly with Fz (brush through
``phi = C*Fz*s / (3*mu*Fz)``, MF through ``D = mu*Fz``), so curves are fit per
unit normal load on a (kappa, alpha) grid. The optimizer is the package's own
CMA-ES on the curve MSE.
"""

from __future__ import annotations

import numpy as np

from .ga import StrategyCMAES

PARAM_NAMES = ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2")

DEFAULT_BOUNDS = {
    "bx": (2.0, 30.0),
    "cx": (1.0, 2.5),
    "ex": (-2.0, 1.0),
    "by": (2.0, 30.0),
    "cy": (1.0, 2.5),
    "ey": (-2.0, 1.0),
    "rbx1": (0.0, 30.0),
    "rbx2": (0.0, 30.0),
    "rby1": (0.0, 30.0),
    "rby2": (0.0, 30.0),
}


def bounds_arrays(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays in PARAM_NAMES order."""
    lo = np.array([bounds[n][0] for n in PARAM_NAMES], dtype=np.float64)
    hi = np.array([bounds[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lo, hi


def brush_forces(kappa, alpha, mu: float, c_long: float, c_lat: float):
    """Brush combined-slip forces per unit Fz (textbook parabolic-pressure law).

    Mirrors newton's kernel independently (same formulas as the golden-curve
    test reference in newton/tests/test_vehicles_tire.py).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    fx_lin = c_long * k * inv
    fy_lin = c_lat * np.tan(alpha) * inv
    f_lin = np.hypot(fx_lin, fy_lin)
    phi = f_lin / (3.0 * mu)
    f_mag = np.where(phi < 1.0, 3.0 * mu * phi * (1.0 - phi + phi * phi / 3.0), mu)
    scale = np.divide(f_mag, f_lin, out=np.zeros_like(f_lin), where=f_lin >= 1.0e-9)
    return scale * fx_lin, -scale * fy_lin


def mf_forces(theta, kappa, alpha, mu: float):
    """Core-MF + cosine combined-slip forces per unit Fz (mirrors the kernel)."""
    bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2 = np.asarray(theta, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    sx = k * inv
    sy = np.tan(alpha) * inv

    def mf(s, b, c, e):
        bs = b * s
        return np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))

    fx0 = mu * mf(sx, bx, cx, ex)
    fy0 = mu * mf(sy, by, cy, ey)
    gxa = np.cos(np.arctan(rbx1 * np.cos(np.arctan(rbx2 * sx)) * sy))
    gyk = np.cos(np.arctan(rby1 * np.cos(np.arctan(rby2 * sy)) * sx))
    return gxa * fx0, -gyk * fy0


def fit_to_brush(
    mu: float,
    c_long: float,
    c_lat: float,
    bounds: dict | None = None,
    seed: int = 0,
    iters: int = 300,
    population: int = 32,
) -> tuple[np.ndarray, float]:
    """Fit MF curves to the brush curves; returns (theta, rms_error / mu)."""
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    lo, hi = bounds_arrays(bounds)
    kg, ag = np.meshgrid(np.linspace(-0.8, 0.8, 33), np.linspace(-0.6, 0.6, 25))
    kappa, alpha = kg.ravel(), ag.ravel()
    fx_t, fy_t = brush_forces(kappa, alpha, mu, c_long, c_lat)
    es = StrategyCMAES(lo, hi, x0=0.5 * (lo + hi), population=population, seed=seed)
    for _ in range(iters):
        pop = es.ask()
        errs = np.empty(pop.shape[0])
        for i, th in enumerate(pop):
            fx, fy = mf_forces(th, kappa, alpha, mu)
            errs[i] = np.mean((fx - fx_t) ** 2 + (fy - fy_t) ** 2)
        es.tell(errs)
    return es.best_x.copy(), float(np.sqrt(es.best_f) / mu)
