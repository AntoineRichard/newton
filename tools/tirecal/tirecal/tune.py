# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibration CLI: fit Pacejka parameters to a recorded session.

Flow: curve-fit seed (standalone) -> ask/tell strategy (GA or CMA-ES) driving
the batched replay evaluator -> holdout scoring -> JSON report.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from . import format as tcformat
from . import seed_fit
from .ga import StrategyCMAES, StrategyGA


def _split_session(session, holdout_laps: int):
    if holdout_laps <= 0 or holdout_laps >= len(session.laps):
        return session, None
    train = copy.copy(session)
    hold = copy.copy(session)
    train.laps = session.laps[:-holdout_laps]
    hold.laps = session.laps[-holdout_laps:]
    return train, hold


def run_calibration(
    session, strategy_name, generations, population, eval_cfg, holdout_laps, seed, evaluator_factory=None
):
    if evaluator_factory is None:
        from .newton_eval import ReplayEvaluator  # noqa: PLC0415

        evaluator_factory = ReplayEvaluator
    train, hold = _split_session(session, holdout_laps)
    evaluator = evaluator_factory(train, eval_cfg)

    tp = session.vehicle["tire_params"]
    if session.vehicle.get("tire_model") == "brush":
        x0, curve_rms = seed_fit.fit_to_brush(mu=tp["mu"], c_long=tp["c_long"], c_lat=tp["c_lat"], seed=seed)
    else:
        lo, hi = seed_fit.bounds_arrays(seed_fit.DEFAULT_BOUNDS)
        x0, curve_rms = 0.5 * (lo + hi), float("nan")

    lo, hi = seed_fit.bounds_arrays(seed_fit.DEFAULT_BOUNDS)
    if strategy_name == "ga":
        strategy = StrategyGA(lo, hi, population=population, x0=x0, seed=seed)
    elif strategy_name == "cmaes":
        strategy = StrategyCMAES(lo, hi, x0=x0, population=population, seed=seed)
    else:
        raise ValueError(f"unknown strategy {strategy_name!r}")

    history = []
    for gen in range(generations):
        fitness = evaluator.evaluate(strategy.ask())
        strategy.tell(fitness)
        history.append(float(strategy.best_f))
        print(f"gen {gen:3d}  best {strategy.best_f:.6f}")

    report = {
        "best_params": dict(zip(seed_fit.PARAM_NAMES, (float(v) for v in strategy.best_x), strict=True)),
        "best_fitness": float(strategy.best_f),
        "history": history,
        "seed_fit": {
            "params": dict(zip(seed_fit.PARAM_NAMES, (float(v) for v in x0), strict=True)),
            "curve_rms": curve_rms,
        },
        "holdout": None,
        "config": {
            "strategy": strategy_name,
            "generations": generations,
            "population": population,
            "window_s": eval_cfg.window_s,
            "stride_s": eval_cfg.stride_s,
            "seed": seed,
            "holdout_laps": holdout_laps,
        },
    }
    if hold is not None:
        hold_eval = evaluator_factory(hold, eval_cfg)
        report["holdout"] = {
            "fitness": float(hold_eval.evaluate(strategy.best_x[None, :])[0]),
            "mean_position_error_m": float(hold_eval.mean_position_error(strategy.best_x)),
        }
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True)
    p.add_argument("--strategy", default="cmaes", choices=("ga", "cmaes"))
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--max-worlds", type=int, default=4096)
    p.add_argument("--holdout-laps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="report JSON path")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="replay the recorded generator model against its own recording and exit",
    )
    a = p.parse_args(argv)

    from .newton_eval import EvalConfig, ReplayEvaluator  # noqa: PLC0415

    session = tcformat.read_session(a.session)
    cfg = EvalConfig(window_s=a.window, stride_s=a.stride, max_worlds=a.max_worlds)
    if a.self_test:
        cfg.tire_model = session.vehicle["tire_model"]
        ev = ReplayEvaluator(session, cfg)
        err = ev.mean_position_error(np.zeros(10))
        print(f"self-test: {ev.num_segments} segments, mean position error {err * 100.0:.2f} cm")
        return
    report = run_calibration(session, a.strategy, a.generations, a.population, cfg, a.holdout_laps, a.seed)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {a.out}")


if __name__ == "__main__":
    main()
