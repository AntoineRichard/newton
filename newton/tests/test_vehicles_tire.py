# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

from newton._src.vehicles.tire import TIRE_BRUSH, TIRE_LINEAR, _eval_tire_kernel


def _eval(model_id, kappa, alpha, fz, mu, c_long, c_lat, device="cpu"):
    n = len(kappa)

    def arr(values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    out = wp.zeros(n, dtype=wp.vec2, device=device)
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
            out,
        ],
        device=device,
    )
    return out.numpy()


class TestTireForce(unittest.TestCase):
    def test_zero_slip_zero_force(self):
        f = _eval(TIRE_BRUSH, [0.0], [0.0], [100.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertAlmostEqual(float(f[0]), 0.0, places=4)
        self.assertAlmostEqual(float(f[1]), 0.0, places=4)

    def test_longitudinal_saturates(self):
        f = _eval(TIRE_BRUSH, [10.0], [0.0], [100.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertAlmostEqual(float(f[0]), 100.0, delta=2.0)  # ~ mu*Fz, forward
        self.assertLess(abs(float(f[1])), 1.0e-3)

    def test_lateral_saturates(self):
        f = _eval(TIRE_BRUSH, [0.0], [0.8], [100.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertLess(abs(float(f[0])), 1.0e-3)
        self.assertAlmostEqual(abs(float(f[1])), 100.0, delta=2.0)
        self.assertLess(float(f[1]), 0.0)  # lateral force opposes positive slip angle

    def test_combined_slip_on_friction_circle(self):
        f = _eval(TIRE_BRUSH, [5.0], [0.5], [100.0], [1.0], [2.0e4], [2.0e4])[0]
        mag = math.hypot(float(f[0]), float(f[1]))
        self.assertLessEqual(mag, 100.0 * 1.001)
        self.assertGreater(mag, 90.0)  # near saturation under large combined slip

    def test_driving_sign(self):
        f = _eval(TIRE_BRUSH, [0.01], [0.0], [1000.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertGreater(float(f[0]), 0.0)  # positive slip -> forward force

    def test_linear_slope(self):
        # below saturation: F_long ~= c_long * kappa
        f = _eval(TIRE_LINEAR, [0.01], [0.0], [1000.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertAlmostEqual(float(f[0]), 200.0, delta=1.0)
        self.assertAlmostEqual(float(f[1]), 0.0, places=4)

    def test_linear_clips_to_circle(self):
        f = _eval(TIRE_LINEAR, [10.0], [0.5], [100.0], [1.0], [2.0e4], [2.0e4])[0]
        mag = math.hypot(float(f[0]), float(f[1]))
        self.assertAlmostEqual(mag, 100.0, delta=1.0)

    def test_zero_load_zero_force(self):
        f = _eval(TIRE_BRUSH, [5.0], [0.5], [0.0], [1.0], [2.0e4], [2.0e4])[0]
        self.assertEqual(float(f[0]), 0.0)
        self.assertEqual(float(f[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
