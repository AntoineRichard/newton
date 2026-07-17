# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from tirecal.ga import StrategyCMAES


def _sphere(x):
    return np.sum(x * x, axis=1)


def _rosenbrock(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1)


class TestStrategyCMAES(unittest.TestCase):
    def _run(self, fn, n, x0, iters, population=None, seed=0):
        lo, hi = np.full(n, -5.0), np.full(n, 5.0)
        es = StrategyCMAES(lo, hi, x0=np.asarray(x0, dtype=float), population=population, seed=seed)
        for _ in range(iters):
            es.tell(fn(es.ask()))
        return es

    def test_sphere_high_precision(self):
        es = self._run(_sphere, n=10, x0=np.full(10, 2.0), iters=250, population=20)
        self.assertLess(es.best_f, 1.0e-6)

    def test_rosenbrock_valley(self):
        es = self._run(_rosenbrock, n=5, x0=np.zeros(5), iters=600, population=24)
        self.assertLess(es.best_f, 1.0e-2)

    def test_respects_bounds(self):
        lo, hi = np.array([0.0, 1.0]), np.array([1.0, 2.0])
        es = StrategyCMAES(lo, hi, x0=np.array([0.5, 1.5]), population=12, seed=1)
        for _ in range(10):
            x = es.ask()
            self.assertTrue(np.all(x >= lo) and np.all(x <= hi))
            es.tell(_sphere(x))

    def test_large_population_works(self):
        es = self._run(_sphere, n=6, x0=np.full(6, 1.0), iters=60, population=256)
        self.assertLess(es.best_f, 1.0e-4)

    def test_deterministic_under_seed(self):
        kw = {"x0": np.full(3, 0.5), "population": 16, "seed": 9}
        a = StrategyCMAES(np.zeros(3), np.ones(3), **kw).ask()
        b = StrategyCMAES(np.zeros(3), np.ones(3), **kw).ask()
        np.testing.assert_array_equal(a, b)
