# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from tirecal import seed_fit


class TestSeedFit(unittest.TestCase):
    def test_brush_reference_saturates_at_mu(self):
        fx, fy = seed_fit.brush_forces(np.array([10.0]), np.array([0.0]), mu=1.0, c_long=20.0, c_lat=20.0)
        self.assertAlmostEqual(float(fx[0]), 1.0, delta=0.02)
        fx, fy = seed_fit.brush_forces(np.array([0.0]), np.array([0.8]), mu=1.0, c_long=20.0, c_lat=20.0)
        self.assertAlmostEqual(abs(float(fy[0])), 1.0, delta=0.02)

    def test_mf_self_recovery(self):
        # curves generated from a known theta must be re-fit to near-zero error
        theta_star = np.array([12.0, 1.5, 0.3, 13.0, 1.4, 0.2, 10.0, 8.0, 9.0, 7.0])
        kappa = np.linspace(-0.6, 0.6, 25)
        alpha = np.linspace(-0.5, 0.5, 21)
        kg, ag = np.meshgrid(kappa, alpha)

        def objective(pop):
            errs = np.empty(pop.shape[0])
            fx_t, fy_t = seed_fit.mf_forces(theta_star, kg.ravel(), ag.ravel(), mu=1.0)
            for i, th in enumerate(pop):
                fx, fy = seed_fit.mf_forces(th, kg.ravel(), ag.ravel(), mu=1.0)
                errs[i] = np.mean((fx - fx_t) ** 2 + (fy - fy_t) ** 2)
            return errs

        from tirecal.ga import StrategyCMAES  # noqa: PLC0415

        lo, hi = seed_fit.bounds_arrays(seed_fit.DEFAULT_BOUNDS)
        es = StrategyCMAES(lo, hi, x0=0.5 * (lo + hi), population=32, seed=0)
        for _ in range(200):
            es.tell(objective(es.ask()))
        self.assertLess(es.best_f, 1.0e-4)

    def test_fit_to_brush_matches_curves(self):
        theta, err = seed_fit.fit_to_brush(mu=2.0, c_long=20.0, c_lat=40.0, seed=0)
        self.assertEqual(theta.shape, (10,))
        # RMS curve mismatch below 5% of the friction limit
        self.assertLess(err, 0.05)
        lo, hi = seed_fit.bounds_arrays(seed_fit.DEFAULT_BOUNDS)
        self.assertTrue(np.all(theta >= lo) and np.all(theta <= hi))
