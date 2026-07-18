# ABC-SMC Bayesian Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task. Work in a dedicated worktree branched from `antoiner/wheeled-vehicle-design` (suggested: `antoiner/tirecal-abc`).

**Goal:** Add `StrategyABCSMC` (posterior sampling via ABC sequential Monte Carlo) as a third strategy behind tirecal's ask/tell protocol, wire it into `tune.py` with a posterior report, and run it on the recorded brush session for a findings addendum.

**Architecture:** ABC-SMC maps directly onto the existing batch protocol: `ask()` proposes a particle population, the `ReplayEvaluator` fitness is the ABC distance, `tell()` shrinks the acceptance threshold to the q-quantile, importance-reweights survivors (uniform prior ⇒ weights ∝ 1/kernel-mixture density), and resamples + perturbs with a Gaussian kernel scaled to the survivors' weighted covariance. No evaluator changes.

**Tech Stack:** numpy only. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-17-pacejka-tire-ga-calibration-design.md`, Addendum 2026-07-18.

## Global Constraints

- All constraints from the 2026-07-17 plan apply (unittest only; PEP 604; Google docstrings; SPDX years unchanged on modified files; scoped pre-commit only; never commit `uv.lock`; imperative ~50-char commit subjects; benchmark thresholds loosenable at most 2x with a comment, never the algorithm).
- The ask/tell protocol is load-bearing and unchanged: `ask() -> (P, n)`, `tell(values (P,))`, `best_x`, `best_f`, lower is better.
- `ga.py` stays numpy-only. Existing `StrategyGA`/`StrategyCMAES` untouched.
- Commits unsigned (`--no-gpg-sign`) — gpg pinentry unavailable in this environment.

## Model assignment

| Task | Content | Model |
|---|---|---|
| A | `StrategyABCSMC` + benchmarks | Sonnet |
| B | `tune.py --strategy abc` + posterior report + tests | Opus |
| C | Posterior run on the brush session + findings addendum | Fable |

Fable reviews every task. Dependencies: A → B → C.

---

### Task A: `StrategyABCSMC` + benchmarks

**Files:**
- Modify: `tools/tirecal/tirecal/ga.py` (append one class)
- Create: `tools/tirecal/tests/test_abc.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (exact, Task B relies on it): `StrategyABCSMC(bounds_lo, bounds_hi, population=512, seed=0, q=0.5, kernel_scale=2.0)`; ask/tell/best protocol; plus `posterior_samples(n) -> np.ndarray (n, dim)` (weighted resample of current particles), `epsilon: float`, `epsilons: list[float]`, `acceptance_rates: list[float]`, `particles`, `weights`.

- [ ] **Step 1: failing tests** — `tools/tirecal/tests/test_abc.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from tirecal.ga import StrategyABCSMC


class TestStrategyABCSMC(unittest.TestCase):
    def _run(self, dist_fn, n, rounds, population=512, seed=0, lo=-5.0, hi=5.0):
        s = StrategyABCSMC(np.full(n, lo), np.full(n, hi), population=population, seed=seed)
        for _ in range(rounds):
            s.tell(dist_fn(s.ask()))
        return s

    def test_posterior_concentrates_unimodal(self):
        target = np.array([1.5, -2.0, 0.5, 3.0])
        s = self._run(lambda x: np.linalg.norm(x - target, axis=1), n=4, rounds=8)
        post = s.posterior_samples(500)
        self.assertEqual(post.shape, (500, 4))
        np.testing.assert_allclose(post.mean(axis=0), target, atol=0.3)
        self.assertLess(float(post.std(axis=0).max()), 0.6)

    def test_epsilon_schedule_monotone(self):
        s = self._run(lambda x: np.linalg.norm(x, axis=1), n=3, rounds=6)
        eps = s.epsilons
        self.assertEqual(len(eps), 6)
        self.assertTrue(all(eps[i + 1] <= eps[i] + 1e-12 for i in range(len(eps) - 1)))
        self.assertEqual(len(s.acceptance_rates), 6)

    def test_bimodal_posterior_keeps_both_modes(self):
        # the selling point vs CMA-ES/GA: two equally good basins must BOTH survive
        m1, m2 = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
        dist = lambda x: np.minimum(
            np.linalg.norm(x - m1, axis=1), np.linalg.norm(x - m2, axis=1)
        )
        s = self._run(dist, n=2, rounds=8, population=1024, seed=3)
        post = s.posterior_samples(1000)
        near1 = (np.linalg.norm(post - m1, axis=1) < 1.0).mean()
        near2 = (np.linalg.norm(post - m2, axis=1) < 1.0).mean()
        self.assertGreater(float(near1), 0.15)
        self.assertGreater(float(near2), 0.15)

    def test_respects_bounds_and_protocol(self):
        lo, hi = np.array([0.0, 1.0]), np.array([1.0, 2.0])
        s = StrategyABCSMC(lo, hi, population=64, seed=1)
        for _ in range(4):
            x = s.ask()
            self.assertEqual(x.shape, (64, 2))
            self.assertTrue(np.all(x >= lo) and np.all(x <= hi))
            s.tell(np.linalg.norm(x - 0.5, axis=1))
        self.assertIsNotNone(s.best_x)
        self.assertTrue(np.isfinite(s.best_f))

    def test_deterministic_under_seed(self):
        a = StrategyABCSMC(np.zeros(3), np.ones(3), population=32, seed=9).ask()
        b = StrategyABCSMC(np.zeros(3), np.ones(3), population=32, seed=9).ask()
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: verify RED** — `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_abc` → `ImportError: cannot import name 'StrategyABCSMC'`.

- [ ] **Step 3: append to `ga.py`:**

```python
class StrategyABCSMC:
    """ABC-SMC posterior sampler behind the ask/tell protocol.

    Approximate Bayesian computation with sequential Monte Carlo (Toni et
    al. 2009): each ``ask()`` proposes a particle population; ``tell()``
    receives simulation distances (lower better), adaptively shrinks the
    acceptance threshold to the ``q``-quantile (kept monotone), importance-
    reweights survivors, and builds the next population by weighted
    resampling plus a Gaussian perturbation kernel with covariance
    ``kernel_scale`` times the survivors' weighted covariance (Filippi et
    al. 2013). The prior is uniform over the bounds; bounds are enforced by
    clipping. ``posterior_samples`` draws from the current weighted
    particle set.
    """

    def __init__(self, bounds_lo, bounds_hi, population: int = 512, seed: int = 0,
                 q: float = 0.5, kernel_scale: float = 2.0):
        self.lo = np.asarray(bounds_lo, dtype=np.float64)
        self.hi = np.asarray(bounds_hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape or np.any(self.hi <= self.lo):
            raise ValueError("invalid bounds")
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q!r}")
        self.n = self.lo.size
        self.population = int(population)
        if self.population < 2 * (self.n + 1):
            raise ValueError("population too small for covariance estimation")
        self.rng = np.random.default_rng(seed)
        self.q = float(q)
        self.kernel_scale = float(kernel_scale)
        span = self.hi - self.lo
        self._pending = self.lo + self.rng.random((self.population, self.n)) * span
        self.particles = None
        self.weights = None
        self.epsilon = np.inf
        self.epsilons = []
        self.acceptance_rates = []
        self._gen_cov = None  # kernel covariance that generated _pending
        self.best_x = None
        self.best_f = np.inf

    def ask(self) -> np.ndarray:
        return self._pending.copy()

    def tell(self, distances) -> None:
        d = np.asarray(distances, dtype=np.float64)
        if d.shape != (self.population,):
            raise ValueError(f"expected distances shape ({self.population},), got {d.shape}")
        i_best = int(np.argmin(d))
        if d[i_best] < self.best_f:
            self.best_f = float(d[i_best])
            self.best_x = self._pending[i_best].copy()
        eps = min(float(np.quantile(d, self.q)), self.epsilon)
        accept = d <= eps
        if int(accept.sum()) < self.n + 2:  # keep enough survivors for a covariance
            accept = np.zeros_like(accept)
            accept[np.argsort(d)[: self.n + 2]] = True
        survivors = self._pending[accept]
        if self.particles is None or self._gen_cov is None:
            new_w = np.full(len(survivors), 1.0 / len(survivors))
        else:
            # uniform prior: w_i \propto 1 / sum_j W_j K(theta_i | theta_j)
            inv = np.linalg.inv(self._gen_cov)
            diff = survivors[:, None, :] - self.particles[None, :, :]
            quad = np.einsum("smi,ij,smj->sm", diff, inv, diff)
            dens = (self.weights[None, :] * np.exp(-0.5 * quad)).sum(axis=1)
            new_w = 1.0 / np.maximum(dens, 1.0e-300)
            new_w /= new_w.sum()
        self.particles = survivors
        self.weights = new_w
        self.epsilon = eps
        self.epsilons.append(eps)
        self.acceptance_rates.append(float(accept.mean()))
        # next proposals: weighted resample + Gaussian perturbation
        mean = new_w @ survivors
        centered = survivors - mean
        cov = (centered * new_w[:, None]).T @ centered
        cov = self.kernel_scale * cov + 1.0e-10 * np.eye(self.n)
        self._gen_cov = cov
        chol = np.linalg.cholesky(cov)
        idx = self.rng.choice(len(survivors), size=self.population, p=new_w)
        prop = survivors[idx] + self.rng.standard_normal((self.population, self.n)) @ chol.T
        self._pending = np.clip(prop, self.lo, self.hi)

    def posterior_samples(self, n: int) -> np.ndarray:
        """Draw ``n`` samples from the current weighted particle set."""
        if self.particles is None:
            raise RuntimeError("posterior_samples() before any tell()")
        idx = self.rng.choice(len(self.particles), size=int(n), p=self.weights)
        return self.particles[idx].copy()
```

- [ ] **Step 4: GREEN + full tirecal discovery**, scoped lint, commit `Add ABC-SMC posterior strategy behind ask/tell`.

---

### Task B: `tune.py --strategy abc` + posterior report

**Files:**
- Modify: `tools/tirecal/tirecal/tune.py`
- Modify: `tools/tirecal/tests/test_tune.py`

**Interfaces:**
- Consumes: Task A's class; existing `run_calibration` flow.
- Produces: `--strategy abc` CLI choice. When the strategy is ABC, the report dict gains a `posterior` key:
  `{"param_order": list(PARAM_NAMES), "mean": {name: v}, "std": {name: v}, "correlation": [[...]] (10x10, PARAM_NAMES order), "epsilon_schedule": [...], "acceptance_rates": [...], "samples": [[...]] (256 rows)}` — all values Python floats/lists (JSON-strict, no NaN). `best_params` stays the min-distance particle (`strategy.best_x`) for cross-strategy comparability; `history` stays per-generation `best_f` (monotone). ABC ignores the curve-fit seed (uniform prior) — `seed_fit` stays in the report for reference; add one line to the `--strategy` help documenting that.

- [ ] **Step 1: failing test** — extend `test_tune.py`:

```python
    def test_abc_strategy_produces_posterior(self):
        report = tune.run_calibration(
            _dummy_session(), "abc", generations=12, population=256,
            eval_cfg=EvalConfig(max_worlds=8), holdout_laps=1, seed=0,
            evaluator_factory=_QuadraticEvaluator,
        )
        post = report["posterior"]
        self.assertEqual(post["param_order"], list(
            ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2")))
        self.assertEqual(len(post["samples"]), 256)
        self.assertEqual(len(post["correlation"]), 10)
        self.assertEqual(len(post["epsilon_schedule"]), 12)
        # quadratic distance: posterior mean must approach the target
        target = _QuadraticEvaluator.target
        mean = np.array([post["mean"][k] for k in post["param_order"]])
        self.assertLess(float(np.abs(mean - target).max()), 1.5)
        self.assertLess(report["best_fitness"], 5.0)
        # JSON-strict: no NaN/inf anywhere in the posterior block
        import json
        json.dumps(post, allow_nan=False)

    def test_non_abc_reports_have_no_posterior(self):
        report = self._report("cmaes")
        self.assertNotIn("posterior", report)
```

Note: `_QuadraticEvaluator.target` values must lie inside `DEFAULT_BOUNDS` for the uniform prior to reach them — they do (verify; if any lies outside, adjust the assertion tolerance, not the bounds).

- [ ] **Step 2: RED** — `-k test_tune` → KeyError/ValueError on unknown strategy `abc`.

- [ ] **Step 3: implement** — in `run_calibration`: add the `abc` branch constructing `StrategyABCSMC(lo, hi, population=population, seed=seed)`; after the generation loop, if the strategy is an `StrategyABCSMC` instance, build the posterior block:

```python
    if isinstance(strategy, StrategyABCSMC):
        post = strategy.posterior_samples(256)
        corr = np.corrcoef(post, rowvar=False)
        report["posterior"] = {
            "param_order": list(seed_fit.PARAM_NAMES),
            "mean": {k: float(v) for k, v in zip(seed_fit.PARAM_NAMES, post.mean(axis=0), strict=True)},
            "std": {k: float(v) for k, v in zip(seed_fit.PARAM_NAMES, post.std(axis=0), strict=True)},
            "correlation": [[float(c) if np.isfinite(c) else 0.0 for c in row] for row in corr],
            "epsilon_schedule": [float(e) for e in strategy.epsilons],
            "acceptance_rates": [float(a) for a in strategy.acceptance_rates],
            "samples": [[float(v) for v in row] for row in post],
        }
```

(Import `StrategyABCSMC` alongside the others; add `"abc"` to the CLI choices and help text.)

- [ ] **Step 4: GREEN + full tirecal discovery**, scoped lint, commit `Wire ABC-SMC posterior reporting into tune CLI`.

---

### Task C: posterior run on the brush session + findings addendum

Runs-and-judgment (GPU). Data: the merged brush session in `tirecal-data/` (regenerate with `python -m tirecal.merge` if the merged file is absent — the per-family .h5 files are preserved).

- [ ] Run: `PYTHONPATH=tools/tirecal:<track_gen path> uv run --extra tirecal --extra sim python -m tirecal.tune --session <merged.h5> --strategy abc --population 256 --generations 10 --stride 2.0 --holdout-laps 4 --out tirecal-data/abc-posterior.json` (population×rounds sized to roughly the CMA-ES budget; stride 2.0 halves segments to compensate — adjust down if wall-clock exceeds ~40 min, and record what was run).
- [ ] Analysis: per-parameter posterior mean±std table; correlation highlights (|r| > 0.5 pairs); explicitly compare identifiability against the synthetic-recovery finding (expectation: bx/cx/by/cy tight, e*/r* wide). Holdout: `mean_position_error_m` of the min-distance particle (in the report) AND of the posterior mean (one extra evaluation via a small snippet) — compare both against the CMA-ES point fit (0.0672 m).
- [ ] Append an "ABC-SMC posterior (2026-07-18)" section to `docs/superpowers/specs/2026-07-17-pacejka-calibration-findings.md`: the run config, the table, identifiability conclusions, holdout comparison, and a note on posterior-conditioned domain randomization as the intended consumer. Commit `Add ABC-SMC posterior findings for the brush calibration`.

## Plan self-review notes

- Coverage: spec addendum scope items all mapped (strategy → A, tune/report → B, validation/findings → C; Tier-2 deferral needs no task).
- Type consistency: `StrategyABCSMC` ctor/report keys match between A and B; `posterior` block keys match the addendum's list.
- The weight-update kernel covariance is the one that GENERATED the current population (`_gen_cov` cached before `_pending` is rebuilt) — the standard ABC-SMC importance-weight subtlety; Task A's reviewer should verify this ordering explicitly.
