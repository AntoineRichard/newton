# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import newton

    # The recorder drives the MPPI track example, which imports the external
    # track_gen package; skip (not error) when it is absent or incompatible.
    import newton.examples.vehicles.example_vehicle_mppi_track  # noqa: F401

    _HAVE_SIM = True
except Exception:
    _HAVE_SIM = False


@unittest.skipUnless(_HAVE_SIM, "newton with sim extra + track_gen required")
class TestReplayEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tirecal import format as tcformat  # noqa: PLC0415
        from tirecal import generate  # noqa: PLC0415

        cls._dir = tempfile.TemporaryDirectory()
        out = Path(cls._dir.name) / "session.h5"
        generate.main(["--out", str(out), "--laps", "1", "--track-seed0", "0", "--frames-max", "90", "--test"])
        cls.session = tcformat.read_session(out)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_brush_self_replay_is_faithful(self):
        from tirecal.newton_eval import EvalConfig, ReplayEvaluator  # noqa: PLC0415

        cfg = EvalConfig(window_s=1.0, stride_s=0.5, max_worlds=64, tire_model="brush")
        ev = ReplayEvaluator(self.session, cfg)
        self.assertGreaterEqual(ev.num_segments, 2)
        # Validation ladder step 2: the generator model replayed against its own
        # recording must be near-exact. 2 cm mean position error over 1 s.
        err_m = ev.mean_position_error(np.zeros(10, dtype=np.float64))
        self.assertLess(err_m, 0.02)

    def test_pacejka_orders_candidates_sensibly(self):
        from tirecal.newton_eval import EvalConfig, ReplayEvaluator  # noqa: PLC0415
        from tirecal.seed_fit import fit_to_brush  # noqa: PLC0415

        cfg = EvalConfig(window_s=1.0, stride_s=0.5, max_worlds=64, tire_model="pacejka")
        ev = ReplayEvaluator(self.session, cfg)
        tp = self.session.vehicle["tire_params"]
        seeded, _ = fit_to_brush(mu=tp["mu"], c_long=tp["c_long"], c_lat=tp["c_lat"], iters=120, population=24)
        absurd = np.array([2.0, 1.0, -2.0, 2.0, 1.0, -2.0, 30.0, 30.0, 30.0, 30.0])
        fit = ev.evaluate(np.stack([seeded, absurd]))
        self.assertEqual(fit.shape, (2,))
        self.assertTrue(np.isfinite(fit).all())
        # the curve-fit seed must beat a deliberately terrible tire
        self.assertLess(fit[0], fit[1])
