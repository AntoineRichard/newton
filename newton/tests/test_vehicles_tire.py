# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
import unittest

import numpy as np
import warp as wp

from newton._src.vehicles import tire as _tire
from newton._src.vehicles.tire import (
    TIRE_BRUSH,
    TIRE_LINEAR,
    TIRE_PACEJKA,
    _eval_stiffness_kernel,
    _eval_tire_kernel,
)
from newton.vehicles import PacejkaParams, TireModel


def _eval(model_id, kappa, alpha, fz, mu, c_long, c_lat, trail=0.0, mf=None, device="cpu"):
    n = len(kappa)

    def arr(values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    if mf is None:
        mf_arr = wp.zeros((n, 10), dtype=wp.float32, device=device)
    else:
        mf_arr = wp.array(np.tile(np.asarray(mf, dtype=np.float32), (n, 1)), dtype=wp.float32, device=device)
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
            mf_arr,
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


# Canonical MF test parameter vector, order (bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2)
_MF_THETA = (12.5, 1.6, 0.4, 14.0, 1.35, 0.3, 12.0, 10.0, 10.0, 8.0)


def _pacejka_reference(theta, kappa, alpha, fz, mu, trail):
    """Independent NumPy implementation of the core-MF + cosine-weighting law."""
    bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2 = theta
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    sx = k * inv
    sy = np.tan(alpha) * inv
    d = mu * fz

    def mf(s, b, c, e):
        bs = b * s
        return np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))

    fx0 = d * mf(sx, bx, cx, ex)
    fy0 = d * mf(sy, by, cy, ey)
    gxa = np.cos(np.arctan(rbx1 * np.cos(np.arctan(rbx2 * sx)) * sy))
    gyk = np.cos(np.arctan(rby1 * np.cos(np.arctan(rby2 * sy)) * sx))
    fx = gxa * fx0
    fy = -gyk * fy0
    util = np.minimum(np.hypot(fx, fy) / max(d, 1.0e-9), 1.0)
    mz = -fy * trail * np.maximum(1.0 - util, 0.0)
    return fx, fy, mz


class TestPacejkaGoldenCurves(unittest.TestCase):
    """Compare the Pacejka kernel branch against an independent NumPy reference."""

    def test_pacejka_matches_reference_over_slip_grid(self):
        fz, mu, trail = 1500.0, 0.9, 0.03
        # Physical slip range only. Unlike the brush law (which saturates to a flat
        # mu*Fz past lock-up), the Magic Formula stays on its falloff tail there, so
        # the float32 rounding of the -0.9999 guard is amplified by 1/(1+kappa) and
        # leaks into the force. The clamp path is covered separately by
        # test_pacejka_lockup_guard_clamps.
        kappa_grid, alpha_grid = np.meshgrid(
            np.linspace(-0.95, 1.5, 26),
            np.linspace(-1.0, 1.0, 21),
        )
        kappa = kappa_grid.ravel()
        alpha = alpha_grid.ravel()
        n = kappa.size
        forces = _eval(
            TIRE_PACEJKA,
            kappa,
            alpha,
            np.full(n, fz),
            np.full(n, mu),
            np.full(n, 20.0),
            np.full(n, 20.0),
            trail=trail,
            mf=_MF_THETA,
        )
        fx_ref, fy_ref, mz_ref = _pacejka_reference(_MF_THETA, kappa, alpha, fz, mu, trail)
        tol = 2.0e-4 * mu * fz
        np.testing.assert_allclose(forces[:, 0], fx_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 1], fy_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 2], mz_ref, atol=tol * trail)

    def test_pacejka_lockup_guard_clamps(self):
        # Past lock-up (kappa <= -1) the 1/(1+kappa) slip is guarded at kappa =
        # -0.9999, so every deeper slip collapses to the same clamp point. This
        # comparison is float32-stable (identical inputs to the kernel), unlike a
        # float64 reference on the amplified falloff tail.
        alpha = np.array([0.0, 0.3, -0.5])
        m = alpha.size
        ref = _eval(
            TIRE_PACEJKA,
            np.full(m, -0.9999),
            alpha,
            np.full(m, 1000.0),
            np.full(m, 1.0),
            np.full(m, 20.0),
            np.full(m, 20.0),
            mf=_MF_THETA,
        )
        for kappa in (-1.0, -2.0, -5.0):
            f = _eval(
                TIRE_PACEJKA,
                np.full(m, kappa),
                alpha,
                np.full(m, 1000.0),
                np.full(m, 1.0),
                np.full(m, 20.0),
                np.full(m, 20.0),
                mf=_MF_THETA,
            )
            np.testing.assert_allclose(f, ref, atol=1.0e-4 * 1.0 * 1000.0)
        # sliding wheel past lock-up brakes, and stays inside the friction circle
        self.assertLess(float(ref[0, 0]), 0.0)  # Fx < 0 at alpha = 0
        mag = np.hypot(ref[:, 0], ref[:, 1])
        self.assertLessEqual(float(mag.max()), 1000.0 * 1.001)

    def test_pacejka_zero_slip_zero_force(self):
        f = _eval(TIRE_PACEJKA, [0.0], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertAlmostEqual(float(f[0]), 0.0, places=3)
        self.assertAlmostEqual(float(f[1]), 0.0, places=3)

    def test_pacejka_zero_load_zero_force(self):
        f = _eval(TIRE_PACEJKA, [0.5], [0.3], [0.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertEqual(float(f[0]), 0.0)
        self.assertEqual(float(f[1]), 0.0)

    def test_pacejka_driving_and_braking_signs(self):
        fwd = _eval(TIRE_PACEJKA, [0.05], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        brk = _eval(TIRE_PACEJKA, [-0.05], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertGreater(float(fwd[0]), 0.0)
        self.assertLess(float(brk[0]), 0.0)

    def test_pacejka_lateral_opposes_slip(self):
        f = _eval(TIRE_PACEJKA, [0.0], [0.1], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertLess(float(f[1]), 0.0)

    def test_pacejka_peak_bounded_by_mu_fz(self):
        # pure-slip curves peak at D = mu*Fz (sin(C*atan(...)) <= 1)
        alpha = np.linspace(-1.0, 1.0, 41)
        n = alpha.size
        f = _eval(
            TIRE_PACEJKA,
            np.zeros(n),
            alpha,
            np.full(n, 1000.0),
            np.full(n, 1.0),
            np.full(n, 20.0),
            np.full(n, 20.0),
            mf=_MF_THETA,
        )
        self.assertLessEqual(float(np.abs(f[:, 1]).max()), 1000.0 * 1.001)

    def test_pacejka_combined_slip_reduces_forces(self):
        pure = _eval(TIRE_PACEJKA, [0.08], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        comb = _eval(TIRE_PACEJKA, [0.08], [0.15], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertLess(abs(float(comb[0])), abs(float(pure[0])))

    def test_pacejka_peak_then_falloff(self):
        # C > 1 gives a peak followed by falloff toward the asymptote
        kappa = np.linspace(0.0, 1.5, 121)
        n = kappa.size
        f = _eval(
            TIRE_PACEJKA,
            kappa,
            np.zeros(n),
            np.full(n, 1000.0),
            np.full(n, 1.0),
            np.full(n, 20.0),
            np.full(n, 20.0),
            mf=_MF_THETA,
        )
        fx = f[:, 0]
        i_peak = int(np.argmax(fx))
        self.assertGreater(i_peak, 0)
        self.assertLess(i_peak, n - 1)
        self.assertLess(float(fx[-1]), float(fx[i_peak]))


class TestTireSlipStiffness(unittest.TestCase):
    def _stiffness(self, model_id, fz, mu, c_long=20.0, c_lat=30.0, mf=None, device="cpu"):
        mf_arr = (
            wp.zeros((1, 10), dtype=wp.float32, device=device)
            if mf is None
            else wp.array(np.asarray(mf, dtype=np.float32).reshape(1, 10), dtype=wp.float32, device=device)
        )
        out = wp.zeros(1, dtype=wp.vec2, device=device)
        wp.launch(
            _eval_stiffness_kernel,
            dim=1,
            inputs=[int(model_id), float(fz), float(mu), float(c_long), float(c_lat), mf_arr, out],
            device=device,
        )
        return out.numpy()[0]

    def test_brush_and_linear_slopes_are_legacy(self):
        for model in (TIRE_BRUSH, TIRE_LINEAR):
            k = self._stiffness(model, fz=1500.0, mu=0.9)
            self.assertEqual(float(k[0]), 20.0 * 1500.0)
            self.assertEqual(float(k[1]), 30.0 * 1500.0)

    def test_pacejka_slope_is_bcd(self):
        k = self._stiffness(TIRE_PACEJKA, fz=1000.0, mu=0.8, mf=_MF_THETA)
        self.assertAlmostEqual(float(k[0]), 12.5 * 1.6 * 0.8 * 1000.0, delta=1.0)
        self.assertAlmostEqual(float(k[1]), 14.0 * 1.35 * 0.8 * 1000.0, delta=1.0)

    def test_pacejka_slope_matches_finite_difference(self):
        eps = 1.0e-4
        f = _eval(TIRE_PACEJKA, [eps], [0.0], [1000.0], [0.8], [20.0], [20.0], mf=_MF_THETA)[0]
        k = self._stiffness(TIRE_PACEJKA, fz=1000.0, mu=0.8, mf=_MF_THETA)
        self.assertAlmostEqual(float(f[0]) / eps, float(k[0]), delta=0.02 * float(k[0]))


class TestPacejkaParamsLayout(unittest.TestCase):
    def test_enum_value(self):
        self.assertEqual(int(TireModel.PACEJKA), 2)

    def test_field_order_matches_mf_constants(self):
        names = tuple(f.name for f in dataclasses.fields(PacejkaParams))
        self.assertEqual(names, ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2"))
        self.assertEqual(names, PacejkaParams.names())
        self.assertEqual(len(names), _tire.MF_PARAM_COUNT)
        order = (
            _tire.MF_BX,
            _tire.MF_CX,
            _tire.MF_EX,
            _tire.MF_BY,
            _tire.MF_CY,
            _tire.MF_EY,
            _tire.MF_RBX1,
            _tire.MF_RBX2,
            _tire.MF_RBY1,
            _tire.MF_RBY2,
        )
        self.assertEqual(order, tuple(range(_tire.MF_PARAM_COUNT)))

    def test_as_array_roundtrip(self):
        p = PacejkaParams(bx=1.0, cx=2.0, ex=3.0, by=4.0, cy=5.0, ey=6.0, rbx1=7.0, rbx2=8.0, rby1=9.0, rby2=10.0)
        np.testing.assert_array_equal(p.as_array(), np.arange(1.0, 11.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
