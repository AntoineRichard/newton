# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from tirecal import format as tcformat
from tirecal import tune
from tirecal.newton_eval import EvalConfig


class _QuadraticEvaluator:
    """Fake evaluator: fitness = ||theta - target||^2. No newton involved."""

    target = np.array([12.0, 1.5, 0.3, 13.0, 1.4, 0.2, 10.0, 8.0, 9.0, 7.0])

    def __init__(self, session, cfg):
        self.num_segments = 4
        self.window_steps = 10

    def evaluate(self, params):
        return np.sum((params - self.target) ** 2, axis=1)

    def mean_position_error(self, row):
        return float(np.sqrt(np.sum((row - self.target) ** 2)))


def _dummy_session():
    lap = tcformat.Lap(
        time=np.zeros(2),
        body_q=np.zeros((2, 7)),
        body_qd=np.zeros((2, 6)),
        wheel_omega=np.zeros((2, 4)),
        controls=np.zeros((1, 3)),
        control_names=("drive", "steer", "brake"),
        meta={"track_seed": 0},
    )
    return tcformat.Session(
        vehicle={"tire_model": "brush", "tire_params": {"c_long": 20.0, "c_lat": 40.0, "mu": 2.0}},
        laps=[lap, lap],
    )


class TestTune(unittest.TestCase):
    def _report(self, strategy):
        return tune.run_calibration(
            # 60 generations = the CLI's production default; at 40 the GA
            # deterministically lands at ~1.08 on seed 0 (verified seeds 0-7),
            # so the budget -- not the < 1.0 threshold -- was miscalibrated.
            _dummy_session(),
            strategy,
            generations=60,
            population=32,
            eval_cfg=EvalConfig(max_worlds=8),
            holdout_laps=1,
            seed=0,
            evaluator_factory=_QuadraticEvaluator,
        )

    def test_converges_with_both_strategies(self):
        for strategy in ("ga", "cmaes"):
            report = self._report(strategy)
            self.assertLess(report["best_fitness"], 1.0, msg=strategy)
            self.assertEqual(
                set(report["best_params"]), {"bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2"}
            )
            self.assertEqual(len(report["history"]), 60)
            self.assertIn("mean_position_error_m", report["holdout"])

    def test_history_is_monotone_best(self):
        report = self._report("cmaes")
        h = report["history"]
        self.assertTrue(all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
