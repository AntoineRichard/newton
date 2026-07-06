# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

from newton._src.vehicles.tire import TIRE_BRUSH, TIRE_LINEAR, _eval_tire_kernel


def _eval(model_id, kappa, alpha, fz, mu, c_long, c_lat, trail=0.0, device="cpu"):
    n = len(kappa)

    def arr(values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(
        _eval_tire_kernel,
        dim=n,
        inputs=[
            wp.array(np.full(n, int(model_id), dtype=np.int32), dtype=wp.int32, device=device),
            arr(kappa),
            arr(alpha),
            arr(fz),
            arr(mu),
            arr(c_long),
            arr(c_lat),
            arr(np.full(n, trail)),
            out,
        ],
        device=device,
    )
    return out.numpy()


def _brush_reference(kappa, alpha, fz, mu, c_long, c_lat, trail):
    """Independent NumPy implementation of the brush law (golden-curve reference).

    Canonical theoretical slip ``sigma = slip / (1 + kappa)`` (guarded at
    lock-up) feeding the parabolic-pressure magnitude law
    ``F = 3*mu*Fz*phi*(1 - phi + phi^2/3)`` with ``phi = F_lin / (3*mu*Fz)``,
    saturating at ``mu*Fz``. Written directly from the textbook formulas so a
    regression in the Warp kernel cannot hide in a shared implementation.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    limit = mu * fz
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    fx_lin = c_long * fz * k * inv
    fy_lin = c_lat * fz * np.tan(alpha) * inv
    f_lin = np.hypot(fx_lin, fy_lin)
    phi = f_lin / (3.0 * limit)
    f_mag = np.where(phi < 1.0, 3.0 * limit * phi * (1.0 - phi + phi * phi / 3.0), limit)
    scale = np.divide(f_mag, f_lin, out=np.zeros_like(f_lin), where=f_lin >= 1.0e-9)
    fx = scale * fx_lin
    fy = -scale * fy_lin
    util = f_mag / limit
    mz = -fy * trail * np.maximum(1.0 - util, 0.0)
    return fx, fy, mz


class TestBrushGoldenCurves(unittest.TestCase):
    """Compare the brush kernel against an independent NumPy reference on a slip grid."""

    def test_brush_matches_reference_over_slip_grid(self):
        fz, mu, c_long, c_lat, trail = 1500.0, 0.9, 15.0, 12.0, 0.03
        # Include kappa <= -1 (past lock-up) to exercise the -0.9999 singularity
        # guard, which the reference clamps identically.
        kappa_grid, alpha_grid = np.meshgrid(
            np.concatenate([[-5.0, -2.0, -1.0], np.linspace(-0.95, 1.5, 26)]),
            np.linspace(-1.0, 1.0, 21),
        )
        kappa = kappa_grid.ravel()
        alpha = alpha_grid.ravel()
        n = kappa.size

        forces = _eval(
            TIRE_BRUSH, kappa, alpha, np.full(n, fz), np.full(n, mu), np.full(n, c_long), np.full(n, c_lat), trail=trail
        )
        fx_ref, fy_ref, mz_ref = _brush_reference(kappa, alpha, fz, mu, c_long, c_lat, trail)

        limit = mu * fz
        tol = 2.0e-4 * limit  # float32 kernel vs float64 reference
        np.testing.assert_allclose(forces[:, 0], fx_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 1], fy_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 2], mz_ref, atol=tol * trail)

        # Golden-curve invariants: the force never leaves the friction circle,
        # and pure-slip curves are odd in their slip argument.
        mag = np.hypot(forces[:, 0], forces[:, 1])
        self.assertLessEqual(float(mag.max()), limit * (1.0 + 1.0e-4))

    def test_brush_pure_lateral_curve_is_odd(self):
        alpha = np.linspace(-0.8, 0.8, 17)
        n = alpha.size
        forces = _eval(
            TIRE_BRUSH, np.zeros(n), alpha, np.full(n, 800.0), np.full(n, 1.0), np.full(n, 20.0), np.full(n, 20.0)
        )
        np.testing.assert_allclose(forces[:, 1], -forces[::-1, 1], atol=1.0e-3)
        np.testing.assert_allclose(forces[:, 0], np.zeros(n), atol=1.0e-3)


class TestTireForce(unittest.TestCase):
    def test_zero_slip_zero_force(self):
        f = _eval(TIRE_BRUSH, [0.0], [0.0], [100.0], [1.0], [20.0], [20.0])[0]
        self.assertAlmostEqual(float(f[0]), 0.0, places=4)
        self.assertAlmostEqual(float(f[1]), 0.0, places=4)

    def test_longitudinal_saturates(self):
        f = _eval(TIRE_BRUSH, [10.0], [0.0], [100.0], [1.0], [20.0], [20.0])[0]
        self.assertAlmostEqual(float(f[0]), 100.0, delta=2.0)  # ~ mu*Fz, forward
        self.assertLess(abs(float(f[1])), 1.0e-3)

    def test_lateral_saturates(self):
        f = _eval(TIRE_BRUSH, [0.0], [0.8], [100.0], [1.0], [20.0], [20.0])[0]
        self.assertLess(abs(float(f[0])), 1.0e-3)
        self.assertAlmostEqual(abs(float(f[1])), 100.0, delta=2.0)
        self.assertLess(float(f[1]), 0.0)  # lateral force opposes positive slip angle

    def test_combined_slip_on_friction_circle(self):
        f = _eval(TIRE_BRUSH, [5.0], [0.5], [100.0], [1.0], [20.0], [20.0])[0]
        mag = math.hypot(float(f[0]), float(f[1]))
        self.assertLessEqual(mag, 100.0 * 1.001)
        self.assertGreater(mag, 90.0)  # near saturation under large combined slip

    def test_driving_sign(self):
        f = _eval(TIRE_BRUSH, [0.01], [0.0], [1000.0], [1.0], [20.0], [20.0])[0]
        self.assertGreater(float(f[0]), 0.0)  # positive slip -> forward force

    def test_linear_slope(self):
        # below saturation: F_long ~= c_long * kappa
        f = _eval(TIRE_LINEAR, [0.01], [0.0], [1000.0], [1.0], [20.0], [20.0])[0]
        self.assertAlmostEqual(float(f[0]), 200.0, delta=1.0)
        self.assertAlmostEqual(float(f[1]), 0.0, places=4)

    def test_linear_clips_to_circle(self):
        f = _eval(TIRE_LINEAR, [10.0], [0.5], [100.0], [1.0], [20.0], [20.0])[0]
        mag = math.hypot(float(f[0]), float(f[1]))
        self.assertAlmostEqual(mag, 100.0, delta=1.0)

    def test_zero_load_zero_force(self):
        f = _eval(TIRE_BRUSH, [5.0], [0.5], [0.0], [1.0], [20.0], [20.0])[0]
        self.assertEqual(float(f[0]), 0.0)
        self.assertEqual(float(f[1]), 0.0)

    def test_braking_force_sign(self):
        # negative slip (wheel slower than road) -> braking force, opposing motion
        f = _eval(TIRE_BRUSH, [-0.02], [0.0], [100.0], [1.0], [20.0], [20.0])[0]
        self.assertLess(float(f[0]), 0.0)

    def test_lockup_saturates(self):
        # beyond lock-up (kappa <= -1) the canonical (1+kappa) slip is guarded and
        # the wheel is fully sliding -> braking force at the friction limit
        f = _eval(TIRE_BRUSH, [-5.0], [0.0], [100.0], [1.0], [20.0], [20.0])[0]
        self.assertAlmostEqual(float(f[0]), -100.0, delta=2.0)

    def test_self_aligning_moment(self):
        # partial lateral slip -> Mz opposes the lateral force (restoring)
        f = _eval(TIRE_BRUSH, [0.0], [0.05], [100.0], [1.0], [20.0], [20.0], trail=0.02)[0]
        self.assertLess(float(f[1]), 0.0)  # lateral force
        self.assertGreater(float(f[2]), 0.05)  # aligning moment, opposite sign to F_lat
        self.assertEqual(float(f[2]) * float(f[1]) < 0.0, True)

    def test_aligning_moment_collapses_at_saturation(self):
        # at full saturation the pneumatic trail -> 0, so Mz -> 0 even though F_lat is large
        f = _eval(TIRE_BRUSH, [0.0], [0.8], [100.0], [1.0], [20.0], [20.0], trail=0.02)[0]
        self.assertLess(abs(float(f[2])), 0.02)

    def test_aligning_moment_zero_at_zero_slip(self):
        f = _eval(TIRE_BRUSH, [0.0], [0.0], [100.0], [1.0], [20.0], [20.0], trail=0.02)[0]
        self.assertEqual(float(f[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
