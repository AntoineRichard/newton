# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.vehicles.impulse import solve_tire_impulse, vec6, wheel_effective_mass
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _solve_kernel(
    inp: wp.array[vec6],
    budgets: wp.array[wp.vec2],
    stiff: wp.array[wp.vec4],
    uff: wp.array[wp.vec2],
    out: wp.array[vec6],
):
    i = wp.tid()
    v = inp[i]
    out[i] = solve_tire_impulse(
        v[0],
        v[1],
        v[2],
        v[3],
        v[4],
        stiff[i][0],
        stiff[i][1],
        stiff[i][2],
        stiff[i][3],
        uff[i][0],
        uff[i][1],
        budgets[i][0],
        budgets[i][1],
    )


@wp.kernel
def _effmass_kernel(out: wp.array[wp.vec3]):
    # 1 kg point mass, inertia I = identity*0.01, contact 0.05 m below COM
    i_inv = wp.mat33(100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    out[0] = wheel_effective_mass(1.0, i_inv, wp.vec3(0.0, 0.0, -0.05), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0))


def _solve(device, u, a, k, budget, budget_stick, k_stick=(1.0, 1.0), u_ff=(0.0, 0.0)):
    inp = wp.array([vec6(u[0], u[1], a[0], a[1], a[2], 0.0)], dtype=vec6, device=device)
    budgets = wp.array([wp.vec2(budget, budget_stick)], dtype=wp.vec2, device=device)
    stiff = wp.array([wp.vec4(k[0], k[1], k_stick[0], k_stick[1])], dtype=wp.vec4, device=device)
    uff = wp.array([wp.vec2(u_ff[0], u_ff[1])], dtype=wp.vec2, device=device)
    out = wp.zeros(1, dtype=vec6, device=device)
    wp.launch(_solve_kernel, dim=1, inputs=[inp, budgets, stiff, uff, out], device=device)
    return out.numpy()[0]


def test_effective_mass_positive_definite(test, device):
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_effmass_kernel, dim=1, inputs=[out], device=device)
    w11, w12, w22 = (float(x) for x in out.numpy()[0])
    # offset -0.05 z, t_fwd x: r x t_fwd = (0,-0.05,0) -> +0.05^2*100 = 0.25 rotational term
    test.assertAlmostEqual(w11, 1.0 + 0.25, places=5)
    test.assertAlmostEqual(w22, 1.0 + 0.25, places=5)
    test.assertAlmostEqual(w12, 0.0, places=6)
    test.assertGreater(w11 * w22 - w12 * w12, 0.0)


def test_stick_when_impulse_fits_budget(test, device):
    # tiny slip velocity, huge budget: stick. With the split-gain contract the
    # feedback gain K_stick is the local mass; choosing K_stick = A^-1 here
    # (k_stick = 1/a11 = 1) makes the stick a full deadbeat: p = -K_stick u and
    # u_new = u + A p = 0 exactly.
    r = _solve(device, (0.02, -0.01), (1.0, 0.0, 1.0), (100.0, 100.0), 10.0, 10.0, k_stick=(1.0, 1.0))
    p_long, p_lat, u1, u2, stick, _util = (float(x) for x in r)
    test.assertEqual(stick, 1.0)
    test.assertAlmostEqual(u1, 0.0, places=6)
    test.assertAlmostEqual(u2, 0.0, places=6)
    test.assertAlmostEqual(p_long, -0.02, places=5)
    test.assertAlmostEqual(p_lat, 0.01, places=5)


def test_stick_split_gain_partial_decay(test, device):
    # Local feedback gain below the coupled mass (k_stick = 0.25, A^-1 = 1):
    # p = -k_stick*u and the reported slip decays geometrically, u_new =
    # (1 - a11*k_stick)*u -- honest, not forced to zero.
    r = _solve(device, (0.1, 0.0), (1.0, 0.0, 1.0), (100.0, 100.0), 10.0, 10.0, k_stick=(0.25, 0.25))
    p_long, p_lat, u1, u2, stick, _util = (float(x) for x in r)
    test.assertEqual(stick, 1.0)
    test.assertAlmostEqual(p_long, -0.025, places=6)
    test.assertAlmostEqual(p_lat, 0.0, places=6)
    test.assertAlmostEqual(u1, 0.075, places=6)
    test.assertAlmostEqual(u2, 0.0, places=6)


def test_stick_feedforward_cancels_field(test, device):
    # At rest (u = 0) with a feedforward slip u_ff = dt*g_t, the stick impulse
    # is the pure coupled-mass feedforward p = -A^-1 u_ff regardless of the
    # feedback gain, and the post-solve total slip is exactly zero: this is the
    # slope-hold force, velocity-independent so it cannot ring.
    r = _solve(device, (0.0, 0.0), (0.5, 0.0, 0.5), (100.0, 100.0), 10.0, 10.0, k_stick=(0.1, 0.1), u_ff=(0.01, -0.02))
    p_long, p_lat, u1, u2, stick, _util = (float(x) for x in r)
    test.assertEqual(stick, 1.0)
    test.assertAlmostEqual(p_long, -0.02, places=6)  # -u_ff/a11
    test.assertAlmostEqual(p_lat, 0.04, places=6)
    test.assertAlmostEqual(u1, 0.0, places=6)
    test.assertAlmostEqual(u2, 0.0, places=6)


def test_slip_solve_reduces_slip_without_reversal(test, device):
    # stiff tire, big slip, budget too small to stick: implicit solve, no sign flip
    r = _solve(device, (2.0, 0.0), (1.0, 0.0, 1.0), (50.0, 50.0), 0.5, 0.5)
    p_long, _p_lat, u1, _u2, stick, util = (float(x) for x in r)
    test.assertEqual(stick, 0.0)
    test.assertLess(p_long, 0.0)  # opposes slip
    test.assertGreater(u1, 0.0)  # reduced but NOT reversed
    test.assertLess(u1, 2.0)
    test.assertAlmostEqual(util, 1.0, places=4)  # budget binds


def test_clamped_impulse_on_budget_boundary(test, device):
    r = _solve(device, (5.0, 5.0), (2.0, 0.1, 2.0), (30.0, 30.0), 0.3, 0.3)
    p_long, p_lat, u1, u2, _stick, _util = (float(x) for x in r)
    p_norm = np.hypot(p_long, p_lat)
    test.assertAlmostEqual(p_norm, 0.3, places=4)
    # clamped u+ must be consistent: u+ = u + A p
    test.assertAlmostEqual(u1, 5.0 + 2.0 * p_long + 0.1 * p_lat, places=4)
    test.assertAlmostEqual(u2, 5.0 + 0.1 * p_long + 2.0 * p_lat, places=4)


def test_passivity_random_inputs(test, device):
    # Two invariants over random draws:
    # 1. Budget: |p| <= max(budget, budget_stick) always (stick is bounded by the
    #    static budget, slip by the kinetic one) -- checked on ALL draws.
    # 2. Passivity: p . u_new <= 0 -- checked only on draws with u_ff = 0. The
    #    feedforward term is force-against-field (it cancels gravity's in-substep
    #    velocity), not damping, so p . u_new <= 0 is not a meaningful contract
    #    when u_ff != 0. The stick feedback gain is drawn below the local
    #    stability bound 1/lambda_max(A) (as the kernel guarantees with
    #    k_stick = m_wheel <= m_c), which is what makes the stick branch passive.
    rng = np.random.default_rng(42)
    for i in range(200):
        u = rng.uniform(-5.0, 5.0, 2)
        a11, a22 = rng.uniform(0.1, 20.0, 2)
        a12 = rng.uniform(-1.0, 1.0) * np.sqrt(a11 * a22) * 0.5
        k = rng.uniform(0.0, 100.0, 2)
        ks = rng.uniform(0.0, 1.0) / (max(a11, a22) + abs(a12))  # <= 1/lambda_max(A)
        budget = rng.uniform(0.01, 5.0)
        budget_stick = budget * rng.uniform(1.0, 1.5)
        with_ff = i % 2 == 1
        u_ff = rng.uniform(-0.5, 0.5, 2) if with_ff else (0.0, 0.0)
        r = _solve(device, u, (a11, a12, a22), k, budget, budget_stick, k_stick=(ks, ks), u_ff=u_ff)
        p = np.array([float(r[0]), float(r[1])])
        u_new = np.array([float(r[2]), float(r[3])])
        test.assertLessEqual(np.hypot(*p), max(budget, budget_stick) * (1.0 + 1.0e-4))
        if not with_ff:
            test.assertLessEqual(float(p @ u_new), 1.0e-5)


def test_zero_stiffness_zero_impulse(test, device):
    r = _solve(device, (1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0), 1.0, 0.0)
    test.assertAlmostEqual(float(r[0]), 0.0, places=6)
    test.assertAlmostEqual(float(r[1]), 0.0, places=6)


class TestVehiclesImpulse(unittest.TestCase):
    pass


for _name, _fn in (
    ("test_effective_mass_positive_definite", test_effective_mass_positive_definite),
    ("test_stick_when_impulse_fits_budget", test_stick_when_impulse_fits_budget),
    ("test_stick_split_gain_partial_decay", test_stick_split_gain_partial_decay),
    ("test_stick_feedforward_cancels_field", test_stick_feedforward_cancels_field),
    ("test_slip_solve_reduces_slip_without_reversal", test_slip_solve_reduces_slip_without_reversal),
    ("test_clamped_impulse_on_budget_boundary", test_clamped_impulse_on_budget_boundary),
    ("test_passivity_random_inputs", test_passivity_random_inputs),
    ("test_zero_stiffness_zero_impulse", test_zero_stiffness_zero_impulse),
):
    add_function_test(TestVehiclesImpulse, _name, _fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
