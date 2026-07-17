# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from tirecal.ga import StrategyGA


def _sphere(x):
    return np.sum(x * x, axis=1)


def _rastrigin(x):
    return 10.0 * x.shape[1] + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


class TestStrategyGA(unittest.TestCase):
    def _run(self, fn, n, population, generations, seed=0, x0=None):
        lo, hi = np.full(n, -5.0), np.full(n, 5.0)
        ga = StrategyGA(lo, hi, population=population, seed=seed, x0=x0)
        for _ in range(generations):
            ga.tell(fn(ga.ask()))
        return ga

    def test_sphere(self):
        ga = self._run(_sphere, n=8, population=96, generations=120)
        self.assertLess(ga.best_f, 0.05)

    def test_rastrigin_multimodal(self):
        ga = self._run(_rastrigin, n=5, population=192, generations=250)
        self.assertLess(ga.best_f, 3.0)

    def test_respects_bounds(self):
        lo, hi = np.array([0.0, 1.0]), np.array([1.0, 2.0])
        ga = StrategyGA(lo, hi, population=32, seed=1)
        for _ in range(5):
            x = ga.ask()
            self.assertTrue(np.all(x >= lo) and np.all(x <= hi))
            ga.tell(_sphere(x))

    def test_x0_seeding_starts_near(self):
        x0 = np.full(4, 2.0)
        ga = StrategyGA(np.full(4, -5.0), np.full(4, 5.0), population=64, seed=2, x0=x0, init_spread=0.05)
        x = ga.ask()
        self.assertLess(float(np.abs(x - x0).mean()), 1.0)

    def test_deterministic_under_seed(self):
        a = StrategyGA(np.zeros(3), np.ones(3), population=16, seed=7).ask()
        b = StrategyGA(np.zeros(3), np.ones(3), population=16, seed=7).ask()
        np.testing.assert_array_equal(a, b)
