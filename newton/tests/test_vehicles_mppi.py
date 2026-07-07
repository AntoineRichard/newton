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

    def test_sigma_horizon_schedule_default_is_flat(self):
        planner = _make()
        np.testing.assert_allclose(planner._sigma_schedule.numpy(), np.ones(8, dtype=np.float32))

    def test_sigma_horizon_factor_one_is_bit_identical(self):
        a, b = _make(), _make()
        a._set_sigma_horizon_factor(1.0)
        a.sample()
        b.sample()
        np.testing.assert_array_equal(a.samples.numpy(), b.samples.numpy())

    def test_sigma_horizon_schedule_shape(self):
        planner = _make()
        factor = 3.0
        planner._set_sigma_horizon_factor(factor)
        h = planner.config.horizon
        expected = factor ** (np.arange(h, dtype=np.float64) / (h - 1))
        np.testing.assert_allclose(planner._sigma_schedule.numpy(), expected, rtol=1e-6)
        # monotone increasing: calm near the executed step, exploratory far out
        sched = planner._sigma_schedule.numpy()
        self.assertTrue(np.all(np.diff(sched) > 0.0))
        self.assertAlmostEqual(float(sched[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sched[-1]), factor, places=5)

    def test_sigma_horizon_factor_grows_far_horizon_variance(self):
        # white noise (beta=0) and wide bounds so the per-step sample std
        # tracks sigma * schedule directly
        planner = _make(
            num_samples=4096, beta=0.0, sigma=(0.1, 0.1), bounds_lo=(-100.0, -100.0), bounds_hi=(100.0, 100.0)
        )
        planner._set_sigma_horizon_factor(3.0)
        planner.sample()
        std = planner.samples.numpy()[1:].std(axis=0)  # [H, A]
        for a in range(2):
            self.assertGreater(float(std[-1, a]), 2.0 * float(std[0, a]))
            # non-decreasing within sampling tolerance
            self.assertTrue(np.all(np.diff(std[:, a]) > -0.01 * float(std[0, a])))

    def test_sigma_horizon_factor_respects_bounds(self):
        planner = _make(sigma=(1.0, 1.0))
        planner._set_sigma_horizon_factor(4.0)
        planner.sample()
        samples = planner.samples.numpy()
        self.assertLessEqual(float(samples.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(samples.min()), -1.0 - 1e-6)
        np.testing.assert_allclose(samples[0], planner.nominal.numpy(), atol=1e-6)

    def test_knots_none_reproduces_per_step_shapes(self):
        planner = _make()  # _n_knots defaults to None
        self.assertFalse(planner._use_knots)
        self.assertEqual(planner.samples.shape, (64, 8, 2))
        self.assertEqual(planner.noise.shape, (64, 8, 2))
        # the decision variable is the nominal itself (no separate knot buffer)
        self.assertIs(planner._knots, planner.nominal)

    def _make_knots(self, n_knots, horizon=8, device=None, **overrides):
        kwargs = {"num_samples": 64, "horizon": horizon, "dim": 2, "sigma": (0.3, 0.4), "seed": 11}
        kwargs.update(overrides)
        return nv.ControllerMPPI(config=nv.ControllerMPPI.Config(**kwargs), device=device, _n_knots=n_knots)

    def test_knots_shapes(self):
        planner = self._make_knots(4)
        self.assertTrue(planner._use_knots)
        # samples/nominal stay at horizon resolution for the caller's rollout;
        # noise lives at knot resolution for the update
        self.assertEqual(planner.samples.shape, (64, 8, 2))
        self.assertEqual(planner.nominal.shape, (8, 2))
        self.assertEqual(planner.noise.shape, (64, 4, 2))
        self.assertEqual(planner._knots.shape, (4, 2))

    def test_knots_config_validation(self):
        with self.assertRaises(ValueError):
            self._make_knots(1)  # < 2 knots
        with self.assertRaises(ValueError):
            self._make_knots(16, horizon=8)  # more knots than horizon

    def test_knots_sample0_is_interpolated_nominal(self):
        planner = self._make_knots(4)
        knots = np.linspace(-0.5, 0.5, 4 * 2).astype(np.float32).reshape(4, 2)
        planner._knots.assign(knots)
        planner._sync_nominal()
        planner.sample()
        # sample 0 is the zero-noise plan == the interpolated nominal
        np.testing.assert_allclose(planner.samples.numpy()[0], planner.nominal.numpy(), atol=1e-6)

    def test_knots_interpolation_exact_at_knot_locations(self):
        # horizon 7, 4 knots -> knots land exactly on t = 0, 2, 4, 6
        planner = self._make_knots(4, horizon=7)
        knots = np.array([[0.1, -0.2], [0.4, 0.5], [-0.3, 0.2], [0.0, -0.6]], dtype=np.float32)
        planner._knots.assign(knots)
        planner._sync_nominal()
        nominal = planner.nominal.numpy()
        for j, t in enumerate((0, 2, 4, 6)):
            np.testing.assert_allclose(nominal[t], knots[j], atol=1e-6)
        # midpoint of adjacent knots is their linear average
        np.testing.assert_allclose(nominal[1], 0.5 * (knots[0] + knots[1]), atol=1e-6)

    def test_knots_bounds_respected(self):
        planner = self._make_knots(4, sigma=(5.0, 5.0))
        planner.sample()
        samples = planner.samples.numpy()
        self.assertLessEqual(float(samples.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(samples.min()), -1.0 - 1e-6)

    def test_knots_shift_advances_one_fine_step(self):
        # the warm-start must advance one executed control step, not one knot:
        # horizon 7, 4 knots -> knots at t = 0, 2, 4, 6, delta = 0.5 knots =
        # 1 fine step; shifted knot j equals the old spline at t_j + 1
        planner = self._make_knots(4, horizon=7)
        knots = np.array([[0.1, -0.2], [0.4, 0.5], [-0.3, 0.2], [0.0, -0.6]], dtype=np.float32)
        planner._knots.assign(knots)
        planner._sync_nominal()
        before = planner.nominal.numpy().copy()
        planner.shift()
        after = planner.nominal.numpy()
        for j in range(3):
            np.testing.assert_allclose(after[2 * j], before[2 * j + 1], atol=1e-6)
        np.testing.assert_allclose(after[6], before[6], atol=1e-6)  # last knot held

    def test_knots_full_cycle_stays_finite_and_bounded(self):
        planner = self._make_knots(4)
        planner.sample()
        costs = wp.array(np.linspace(0.0, 10.0, 64, dtype=np.float32), dtype=wp.float32, device=planner.device)
        planner.update(costs)
        planner.shift()
        nominal = planner.nominal.numpy()
        self.assertTrue(np.isfinite(nominal).all())
        self.assertLessEqual(float(nominal.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(nominal.min()), -1.0 - 1e-6)

    def test_knots_cpu_device_full_cycle(self):
        # mirror test_cpu_device_full_cycle with the knot control path enabled
        planner = self._make_knots(4, device="cpu")
        self.assertTrue(planner._use_knots)
        planner.sample()
        samples = planner.samples.numpy()
        self.assertEqual(samples.shape, (64, 8, 2))
        self.assertTrue(np.isfinite(samples).all())
        np.testing.assert_allclose(samples[0], planner.nominal.numpy(), atol=1e-6)
        costs = wp.array(np.linspace(0.0, 10.0, 64, dtype=np.float32), dtype=wp.float32, device="cpu")
        planner.update(costs)
        planner.shift()
        nominal = planner.nominal.numpy()
        self.assertTrue(np.isfinite(nominal).all())
        self.assertLessEqual(float(nominal.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(nominal.min()), -1.0 - 1e-6)

    # --- RA-MPPI zero-mean sample fraction (experiment A) ----------------

    def test_zero_mean_fraction_validation(self):
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(num_samples=8), _zero_mean_fraction=-0.1)
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(num_samples=8), _zero_mean_fraction=1.5)

    def test_zero_mean_fraction_default_is_flat(self):
        planner = _make()
        self.assertEqual(int(planner._zero_mean_count.numpy()[0]), 0)

    def test_zero_mean_fraction_count(self):
        planner = _make(num_samples=64)
        planner._set_zero_mean_fraction(0.3)
        self.assertEqual(int(planner._zero_mean_count.numpy()[0]), int(np.ceil(0.3 * 64)))

    def test_zero_mean_fraction_zero_nominal_is_bit_identical(self):
        # with an all-zero nominal, nominal + eps == eps, so zero-mean samples
        # equal ordinary samples: the sampler and update must be bit-identical
        a, b = _make(), _make()
        b._set_zero_mean_fraction(0.2)
        a.sample()
        b.sample()
        np.testing.assert_array_equal(a.samples.numpy(), b.samples.numpy())
        costs = np.linspace(0.0, 5.0, 64, dtype=np.float32)
        a.update(wp.array(costs, dtype=wp.float32, device=a.device))
        b.update(wp.array(costs, dtype=wp.float32, device=b.device))
        np.testing.assert_array_equal(a.nominal.numpy(), b.nominal.numpy())

    def test_zero_mean_samples_are_pure_noise(self):
        # with a large nominal offset, the first ceil(f*K) non-hero samples are
        # centered on zero (pure noise), the rest on the nominal
        planner = _make(num_samples=64, sigma=(0.05, 0.05))
        planner._set_zero_mean_fraction(0.25)
        count = int(planner._zero_mean_count.numpy()[0])
        nom = np.full((8, 2), 0.8, dtype=np.float32)
        planner.nominal.assign(nom)
        planner.sample()
        samples = planner.samples.numpy()
        # zero-mean samples sit near 0, ordinary samples near the 0.8 nominal
        self.assertLess(float(np.abs(samples[1 : count + 1]).mean()), 0.2)
        self.assertGreater(float(samples[count + 1 :].mean()), 0.6)

    def test_zero_mean_update_uses_deviation_from_nominal(self):
        # the stored noise is s - nominal even for zero-mean samples, so a
        # winner-take-all update lands the nominal exactly on the winning
        # sample (a naive u-not-(u-nominal) delta would miss when nominal != 0)
        planner = _make(num_samples=64, sigma=(0.05, 0.05))
        planner._set_zero_mean_fraction(1.0)  # every non-hero sample zero-mean
        planner.set_temperature(1e-3)
        planner.nominal.assign(np.full((8, 2), 0.5, dtype=np.float32))
        planner.sample()
        samples = planner.samples.numpy()
        costs = np.full(64, 1e3, dtype=np.float32)
        costs[7] = 0.0
        planner.update(wp.array(costs, dtype=wp.float32, device=planner.device))
        np.testing.assert_allclose(planner.nominal.numpy(), samples[7], atol=1e-3)

    # --- Tsallis / deformed-exponential weighting (experiment B) ---------

    def test_tsallis_q_validation(self):
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(num_samples=8), _tsallis_q=0.0)

    def test_tsallis_q_default_is_one(self):
        planner = _make()
        self.assertEqual(float(planner._tsallis_q.numpy()[0]), 1.0)

    def test_tsallis_q_one_is_bit_identical(self):
        a, b = _make(), _make()
        b._set_tsallis_q(1.0)  # explicit no-op
        a.sample()
        b.sample()
        costs = np.linspace(0.0, 5.0, 64, dtype=np.float32)
        a.update(wp.array(costs, dtype=wp.float32, device=a.device))
        b.update(wp.array(costs, dtype=wp.float32, device=b.device))
        np.testing.assert_array_equal(a.weights.numpy(), b.weights.numpy())
        np.testing.assert_array_equal(a.nominal.numpy(), b.nominal.numpy())

    def test_tsallis_q_shifts_ess(self):
        # with the q-exponential and the min-cost shift, q < 1 has compact
        # support (elite concentration -> lower ESS) while q > 1 has heavier
        # tails (more uniform averaging -> higher ESS) than the q = 1 softmax
        costs = np.linspace(0.0, 30.0, 64, dtype=np.float32)

        def ess_for(q):
            p = _make()
            p.set_temperature(10.0)
            if q is not None:
                p._set_tsallis_q(q)
            p.sample()
            p.update(wp.array(costs, dtype=wp.float32, device=p.device))
            return float(p.ess.numpy()[0])

        base = ess_for(None)
        self.assertLess(ess_for(0.5), base)  # elite concentration
        self.assertGreater(ess_for(2.0), base)  # heavier-tailed averaging

    def test_tsallis_q_matches_deformed_exponential(self):
        # q = 2: exp_2(x) = [1 - x]_+^{-1}, x = -(cost - min)/lambda, so the
        # (unnormalized) weight is 1 / (1 + (cost - min)/lambda)
        planner = _make(device="cpu")
        lam = 7.0
        planner.set_temperature(lam)
        planner._set_tsallis_q(2.0)
        planner.sample()
        costs = np.linspace(2.0, 40.0, 64, dtype=np.float32)
        planner.update(wp.array(costs, dtype=wp.float32, device="cpu"))
        expected = 1.0 / (1.0 + (costs - costs.min()) / lam)
        np.testing.assert_allclose(planner.weights.numpy(), expected, rtol=1e-5, atol=1e-6)

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
