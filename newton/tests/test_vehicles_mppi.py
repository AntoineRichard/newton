# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton.vehicles as nv


def _make(device=None, **overrides):
    kwargs = {"num_samples": 64, "horizon": 8, "dim": 2, "sigma": (0.3, 0.4), "seed": 11}
    kwargs.update(overrides)
    return nv.ControllerMPPI(config=nv.ControllerMPPI.Config(**kwargs), device=device)


class TestControllerMPPI(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(dim=2, sigma=(0.1,)))
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(num_samples=1))

    def test_sample_zero_is_nominal(self):
        planner = _make()
        nom = np.linspace(-0.5, 0.5, 8 * 2).astype(np.float32).reshape(8, 2)
        planner.nominal.assign(nom)
        planner.sample()
        samples = planner.samples.numpy()
        np.testing.assert_allclose(samples[0], nom, atol=1e-6)
        # other samples actually differ from the nominal
        self.assertGreater(float(np.abs(samples[1:] - nom).max()), 1e-3)

    def test_sample_respects_bounds(self):
        planner = _make(sigma=(5.0, 5.0))
        planner.sample()
        samples = planner.samples.numpy()
        self.assertLessEqual(float(samples.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(samples.min()), -1.0 - 1e-6)

    def test_successive_samples_differ(self):
        planner = _make()
        planner.sample()
        first = planner.samples.numpy().copy()
        planner.sample()
        self.assertGreater(float(np.abs(planner.samples.numpy() - first).max()), 1e-3)

    def test_seed_determinism(self):
        a, b = _make(), _make()
        a.sample()
        b.sample()
        np.testing.assert_allclose(a.samples.numpy(), b.samples.numpy())

    def test_update_moves_nominal_toward_low_cost_sample(self):
        planner = _make()
        planner.set_temperature(1e-3)  # winner-takes-all weights
        planner.sample()
        samples = planner.samples.numpy()
        costs = np.full(64, 1e3, dtype=np.float32)
        costs[5] = 0.0
        planner.update(wp.array(costs, dtype=wp.float32, device=planner.device))
        np.testing.assert_allclose(planner.nominal.numpy(), samples[5], atol=1e-3)

    def test_cpu_device_full_cycle(self):
        planner = _make(device="cpu")
        planner.sample()
        samples = planner.samples.numpy()
        self.assertTrue(np.isfinite(samples).all())
        np.testing.assert_allclose(samples[0], planner.nominal.numpy(), atol=1e-6)
        costs = wp.array(np.linspace(0.0, 10.0, 64, dtype=np.float32), dtype=wp.float32, device="cpu")
        planner.update(costs)
        planner.shift()
        self.assertTrue(np.isfinite(planner.nominal.numpy()).all())

    def test_update_validates_costs(self):
        planner = _make()
        with self.assertRaises(ValueError):
            planner.update(wp.zeros(8, dtype=wp.float32, device=planner.device))

    def test_shift_rolls_and_repeats_last(self):
        planner = _make()
        nom = np.arange(8 * 2, dtype=np.float32).reshape(8, 2) * 0.01
        planner.nominal.assign(nom)
        planner.shift()
        out = planner.nominal.numpy()
        np.testing.assert_allclose(out[:-1], nom[1:], atol=1e-6)
        np.testing.assert_allclose(out[-1], nom[-1], atol=1e-6)


if __name__ == "__main__":
    wp.init()
    unittest.main()
