# Pacejka Tire Model + GA Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **REQUIRED:** All implementation work happens in a dedicated git worktree created via superpowers:using-git-worktrees, branched from `antoiner/wheeled-vehicle-design` (the vehicle stack lives there, not on `main`). Suggested branch: `antoiner/pacejka-tirecal`.

**Goal:** Add a Pacejka (Magic Formula) tire model to Newton's vehicle stack and a standalone `tools/tirecal` package that tunes its parameters to reproduce the brush model's behavior, by replaying recorded MPPI laps across thousands of replicated worlds under a GA / CMA-ES optimizer.

**Architecture:** Workstream A adds a `TIRE_PACEJKA` branch to the pluggable `tire_force` Warp func, an append-friendly `(wheel_count, MF_PARAM_COUNT)` parameter array, a `tire_slip_stiffness` func replacing the wheel kernel's hard-coded linearization slopes, and public config/API. Workstream B builds `tools/tirecal`: an HDF5 trajectory format (the future real2sim contract), ask/tell GA + CMA-ES strategies, a standalone MF-to-brush curve-fit seeder, an MPPI-example recorder, and a batched segment-replay evaluator that maps (candidate × segment) onto replicated worlds.

**Tech Stack:** Python, Warp kernels, NumPy, h5py (optional `tirecal` extra only), unittest, uv.

**Spec:** `docs/superpowers/specs/2026-07-17-pacejka-tire-ga-calibration-design.md` (approved).

## Global Constraints

- `newton/_src/` is internal: examples/docs must not import it; new public symbols go through `newton/vehicles.py`. The tirecal package uses only public API plus attribute access on public objects (`vehicles.dynamics.mf_params`, same pattern the MPPI example uses) — with one sanctioned exception: importing `_build_model` from the MPPI example module.
- No new required dependencies. h5py lives only in a new `tirecal` optional-dependency extra.
- All tests use `unittest`, never pytest. Never call `wp.synchronize()` before `.numpy()`.
- Warp array annotations use bracket syntax: `wp.array[wp.float32]`, `wp.array2d[wp.float32]`.
- Google-style docstrings; SI units noted in public docstrings; PEP 604 unions.
- New-file SPDX copyright year is 2026; never update years on modified files.
- Imperative-mood commits, ~50-char subject, body wraps at 72.
- User-facing changes get a CHANGELOG entry at a *random position* within the correct `[Unreleased]` category.
- Verify regression tests fail without the fix before committing (TDD steps below enforce this).
- Run `uvx pre-commit run -a` before each commit.
- All four tires are physically identical: one shared MF parameter vector.

## Model assignment & parallelism

| Task | Content | Model |
|---|---|---|
| 1 | Pacejka `tire_force` branch + param storage | Opus |
| 2 | `tire_slip_stiffness` + wheel-kernel integration | Opus |
| 3 | Public API (`TireModel.PACEJKA`, `PacejkaParams`, docs) | Sonnet |
| 4 | tirecal scaffold + `format.py` + SCHEMA + extra | Sonnet |
| 5 | `StrategyGA` + benchmarks | Sonnet |
| 6 | `StrategyCMAES` + benchmarks | Sonnet |
| 7 | `seed_fit.py` (standalone curve fit) | Sonnet |
| 8 | `generate.py` recorder + smoke test | Opus |
| 9 | `newton_eval.py` batched evaluator + brush self-test | **Fable** |
| 10 | `tune.py` CLI + report | Opus |
| 11 | Validation ladder runs + findings | **Fable** |

Fable reviews every task's diff (two-stage review per subagent-driven-development).

Dependencies: 1 → 2 → 3 (sequential). 4, 5, 6 are independent of everything (parallel with 1-3). 7 needs 6. 8 needs 4. 9 needs 3, 4, 8. 10 needs 5, 7, 9. 11 needs 10. Suggested waves: {1, 4, 5, 6} → {2, 7, 8} → {3} → {9} → {10} → {11}.

## File Structure

```
newton/_src/vehicles/tire.py          MODIFY  MF constants, _mf_pure, PACEJKA branch,
                                              tire_slip_stiffness, new tire_force signature
newton/_src/vehicles/wheel.py         MODIFY  mf_params array + kernel plumbing + slopes
newton/_src/vehicles/controller.py    MODIFY  TireModel.PACEJKA, PacejkaParams, config fill
newton/_src/vehicles/__init__.py      MODIFY  export PacejkaParams
newton/vehicles.py                    MODIFY  export PacejkaParams
newton/tests/test_vehicles_tire.py    MODIFY  Pacejka golden curves + slope tests
CHANGELOG.md                          MODIFY  Added entry
pyproject.toml                        MODIFY  tirecal extra
.gitignore                            MODIFY  tirecal-data/
tools/tirecal/README.md               CREATE  usage (PYTHONPATH invocation, CLIs)
tools/tirecal/SCHEMA.md               CREATE  format spec v1
tools/tirecal/tirecal/__init__.py     CREATE  package marker
tools/tirecal/tirecal/format.py       CREATE  HDF5 reader/writer (numpy+h5py only)
tools/tirecal/tirecal/ga.py           CREATE  StrategyGA, StrategyCMAES (numpy only)
tools/tirecal/tirecal/seed_fit.py     CREATE  MF-to-brush curve fit (numpy only)
tools/tirecal/tirecal/generate.py     CREATE  lap recorder CLI (imports newton)
tools/tirecal/tirecal/newton_eval.py  CREATE  batched replay evaluator (imports newton)
tools/tirecal/tirecal/tune.py         CREATE  calibration CLI
tools/tirecal/tirecal/closedloop.py   CREATE  final-gate closed-loop comparison script
tools/tirecal/tests/__init__.py       CREATE
tools/tirecal/tests/test_format.py    CREATE
tools/tirecal/tests/test_ga.py        CREATE
tools/tirecal/tests/test_cmaes.py     CREATE
tools/tirecal/tests/test_seed_fit.py  CREATE
tools/tirecal/tests/test_generate.py  CREATE  smoke (records a few frames)
tools/tirecal/tests/test_newton_eval.py CREATE self-test (brush ≈ 0)
tools/tirecal/tests/test_tune.py      CREATE  fake-evaluator convergence
```

Commands (from repo root):

```bash
# newton-side tests
uv run --extra dev -m newton.tests -k test_vehicles_tire
# tirecal tests (discovery puts tools/tirecal on sys.path)
uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v
# tirecal CLIs
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.generate --help
```

---

### Task 1: Pacejka branch in `tire.py` + `mf_params` storage

**Files:**
- Modify: `newton/_src/vehicles/tire.py`
- Modify: `newton/_src/vehicles/wheel.py` (signature plumbing only — slopes are Task 2)
- Test: `newton/tests/test_vehicles_tire.py`

**Interfaces:**
- Consumes: existing `tire_force`, `_eval_tire_kernel`, `WheelDynamics`, `_wheel_dynamics_kernel`.
- Produces (later tasks rely on these exact names):
  - `tire.TIRE_PACEJKA` (wp.constant, value 2)
  - `tire.MF_BX = 0`, `MF_CX = 1`, `MF_EX = 2`, `MF_BY = 3`, `MF_CY = 4`, `MF_EY = 5`, `MF_RBX1 = 6`, `MF_RBX2 = 7`, `MF_RBY1 = 8`, `MF_RBY2 = 9`, `MF_PARAM_COUNT = 10` (plain module-level ints except noted)
  - `tire_force(model_id: int, kappa: float, alpha: float, fz: float, mu: float, c_long: float, c_lat: float, trail: float, mf: wp.array2d[wp.float32], w: int) -> wp.vec3`
  - `WheelDynamics.mf_params: wp.array2d[wp.float32]` shape `(wheel_count, MF_PARAM_COUNT)`, zero-initialized
  - `_eval_tire_kernel` gains an `mf_params: wp.array2d[wp.float32]` input before `out`

- [ ] **Step 1: Write the failing tests** — in `newton/tests/test_vehicles_tire.py`, update the `_eval` helper for the new signature and add a Pacejka golden-curve class. Replace the current `_eval` with:

```python
def _eval(model_id, kappa, alpha, fz, mu, c_long, c_lat, trail=0.0, mf=None, device="cpu"):
    n = len(kappa)

    def arr(values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    if mf is None:
        mf_arr = wp.zeros((n, 10), dtype=wp.float32, device=device)
    else:
        mf_arr = wp.array(np.tile(np.asarray(mf, dtype=np.float32), (n, 1)), dtype=wp.float32, device=device)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
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
            arr(np.full(n, trail)),
            mf_arr,
            out,
        ],
        device=device,
    )
    return out.numpy()
```

Update the import line to also bring in the new symbols:

```python
from newton._src.vehicles.tire import TIRE_BRUSH, TIRE_LINEAR, TIRE_PACEJKA, _eval_tire_kernel
```

Append the reference implementation and test class:

```python
# Canonical MF test parameter vector, order (bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2)
_MF_THETA = (12.5, 1.6, 0.4, 14.0, 1.35, 0.3, 12.0, 10.0, 10.0, 8.0)


def _pacejka_reference(theta, kappa, alpha, fz, mu, trail):
    """Independent NumPy implementation of the core-MF + cosine-weighting law."""
    bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2 = theta
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    sx = k * inv
    sy = np.tan(alpha) * inv
    d = mu * fz

    def mf(s, b, c, e):
        bs = b * s
        return np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))

    fx0 = d * mf(sx, bx, cx, ex)
    fy0 = d * mf(sy, by, cy, ey)
    gxa = np.cos(np.arctan(rbx1 * np.cos(np.arctan(rbx2 * sx)) * sy))
    gyk = np.cos(np.arctan(rby1 * np.cos(np.arctan(rby2 * sy)) * sx))
    fx = gxa * fx0
    fy = -gyk * fy0
    util = np.minimum(np.hypot(fx, fy) / max(d, 1.0e-9), 1.0)
    mz = -fy * trail * np.maximum(1.0 - util, 0.0)
    return fx, fy, mz


class TestPacejkaGoldenCurves(unittest.TestCase):
    """Compare the Pacejka kernel branch against an independent NumPy reference."""

    def test_pacejka_matches_reference_over_slip_grid(self):
        fz, mu, trail = 1500.0, 0.9, 0.03
        kappa_grid, alpha_grid = np.meshgrid(
            np.concatenate([[-5.0, -2.0, -1.0], np.linspace(-0.95, 1.5, 26)]),
            np.linspace(-1.0, 1.0, 21),
        )
        kappa = kappa_grid.ravel()
        alpha = alpha_grid.ravel()
        n = kappa.size
        forces = _eval(
            TIRE_PACEJKA,
            kappa,
            alpha,
            np.full(n, fz),
            np.full(n, mu),
            np.full(n, 20.0),
            np.full(n, 20.0),
            trail=trail,
            mf=_MF_THETA,
        )
        fx_ref, fy_ref, mz_ref = _pacejka_reference(_MF_THETA, kappa, alpha, fz, mu, trail)
        tol = 2.0e-4 * mu * fz
        np.testing.assert_allclose(forces[:, 0], fx_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 1], fy_ref, atol=tol)
        np.testing.assert_allclose(forces[:, 2], mz_ref, atol=tol * trail)

    def test_pacejka_zero_slip_zero_force(self):
        f = _eval(TIRE_PACEJKA, [0.0], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertAlmostEqual(float(f[0]), 0.0, places=3)
        self.assertAlmostEqual(float(f[1]), 0.0, places=3)

    def test_pacejka_zero_load_zero_force(self):
        f = _eval(TIRE_PACEJKA, [0.5], [0.3], [0.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertEqual(float(f[0]), 0.0)
        self.assertEqual(float(f[1]), 0.0)

    def test_pacejka_driving_and_braking_signs(self):
        fwd = _eval(TIRE_PACEJKA, [0.05], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        brk = _eval(TIRE_PACEJKA, [-0.05], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertGreater(float(fwd[0]), 0.0)
        self.assertLess(float(brk[0]), 0.0)

    def test_pacejka_lateral_opposes_slip(self):
        f = _eval(TIRE_PACEJKA, [0.0], [0.1], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertLess(float(f[1]), 0.0)

    def test_pacejka_peak_bounded_by_mu_fz(self):
        # pure-slip curves peak at D = mu*Fz (sin(C*atan(...)) <= 1)
        alpha = np.linspace(-1.0, 1.0, 41)
        n = alpha.size
        f = _eval(
            TIRE_PACEJKA, np.zeros(n), alpha, np.full(n, 1000.0), np.full(n, 1.0),
            np.full(n, 20.0), np.full(n, 20.0), mf=_MF_THETA,
        )
        self.assertLessEqual(float(np.abs(f[:, 1]).max()), 1000.0 * 1.001)

    def test_pacejka_combined_slip_reduces_forces(self):
        pure = _eval(TIRE_PACEJKA, [0.08], [0.0], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        comb = _eval(TIRE_PACEJKA, [0.08], [0.15], [1000.0], [1.0], [20.0], [20.0], mf=_MF_THETA)[0]
        self.assertLess(abs(float(comb[0])), abs(float(pure[0])))

    def test_pacejka_peak_then_falloff(self):
        # C > 1 gives a peak followed by falloff toward the asymptote
        kappa = np.linspace(0.0, 1.5, 121)
        n = kappa.size
        f = _eval(
            TIRE_PACEJKA, kappa, np.zeros(n), np.full(n, 1000.0), np.full(n, 1.0),
            np.full(n, 20.0), np.full(n, 20.0), mf=_MF_THETA,
        )
        fx = f[:, 0]
        i_peak = int(np.argmax(fx))
        self.assertGreater(i_peak, 0)
        self.assertLess(i_peak, n - 1)
        self.assertLess(float(fx[-1]), float(fx[i_peak]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire`
Expected: FAIL — `ImportError: cannot import name 'TIRE_PACEJKA'` (and, once constants exist, launch-arity errors). The pre-existing brush/linear tests must fail *only* through the changed `_eval` helper arity, not through behavior changes.

- [ ] **Step 3: Implement in `tire.py`.** After the `TIRE_LINEAR` constant add:

```python
TIRE_PACEJKA = wp.constant(2)

# mf_params column layout. Order is public API: it must match the field order of
# newton.vehicles.PacejkaParams. Extending the parameter set means appending
# indices and bumping MF_PARAM_COUNT (never reordering).
MF_BX = 0
MF_CX = 1
MF_EX = 2
MF_BY = 3
MF_CY = 4
MF_EY = 5
MF_RBX1 = 6
MF_RBX2 = 7
MF_RBY1 = 8
MF_RBY2 = 9
MF_PARAM_COUNT = 10


@wp.func
def _mf_pure(s: float, b: float, c: float, e: float) -> float:
    # normalized Magic Formula: sin(C*atan(B*s - E*(B*s - atan(B*s)))), peak 1
    bs = b * s
    return wp.sin(c * wp.atan(bs - e * (bs - wp.atan(bs))))
```

Change `tire_force`'s signature to append `mf: wp.array2d[wp.float32], w: int` after `trail` (update the docstring: `mf` is the per-wheel Magic Formula parameter row, indexed by the `MF_*` constants and only read by `TIRE_PACEJKA`; `w` is the wheel index selecting the row). Before the brush branch, add:

```python
    if model_id == TIRE_PACEJKA:
        # Core Magic Formula per axis on the same theoretical slip as the brush
        # branch, with Pacejka cosine combined-slip weighting. D = mu*Fz, so each
        # axis is bounded by the friction limit; the combined magnitude is shaped
        # by the r-parameters (the impulse solver's budget projection enforces
        # the hard circle at the vehicle level).
        k = wp.max(kappa, -0.9999)
        inv = 1.0 / (1.0 + k)
        sx = k * inv
        sy = wp.tan(alpha) * inv
        fx0 = limit * _mf_pure(sx, mf[w, MF_BX], mf[w, MF_CX], mf[w, MF_EX])
        fy0 = limit * _mf_pure(sy, mf[w, MF_BY], mf[w, MF_CY], mf[w, MF_EY])
        bxa = mf[w, MF_RBX1] * wp.cos(wp.atan(mf[w, MF_RBX2] * sx))
        byk = mf[w, MF_RBY1] * wp.cos(wp.atan(mf[w, MF_RBY2] * sy))
        fx = wp.cos(wp.atan(bxa * sy)) * fx0
        fy = -wp.cos(wp.atan(byk * sx)) * fy0
        util = wp.min(wp.sqrt(fx * fx + fy * fy) / limit, 1.0)
        mz = -fy * trail * wp.max(1.0 - util, 0.0)
        return wp.vec3(fx, fy, mz)
```

Update `_eval_tire_kernel`: add `mf_params: wp.array2d[wp.float32],` before `out`, and call `tire_force(model_id[i], kappa[i], alpha[i], fz[i], mu[i], c_long[i], c_lat[i], trail[i], mf_params, i)`. Update the module docstring's model list to include `PACEJKA`.

- [ ] **Step 4: Plumb through `wheel.py`** (mechanical; behavior unchanged for brush/linear since they ignore `mf`):
  - In `WheelDynamics.__init__`, after `self.c_lat = z(wp.float32)` add:
    ```python
    self.mf_params = wp.zeros((n, MF_PARAM_COUNT), dtype=wp.float32, device=self.device)
    ```
    and change the tire import line to `from .tire import MF_PARAM_COUNT, tire_force`.
  - In `apply_wheel_dynamics`'s `wp.launch` inputs list, insert `dyn.mf_params,` directly after `dyn.c_lat,`.
  - In `_wheel_dynamics_kernel`'s parameters, insert `mf_params: wp.array2d[wp.float32],` directly after `c_lat: wp.array[wp.float32],`.
  - Change the operating-point call to:
    ```python
    f0 = tire_force(tire_model[w], kappa, alpha, fz, mu, c_long[w], c_lat[w], 0.0, mf_params, w)
    ```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire`
Expected: PASS, including all pre-existing brush/linear tests (regression gate).

Also run the neighboring suites that exercise the wheel kernel:
`uv run --extra dev -m newton.tests -k test_vehicles_wheel`
Expected: PASS unchanged.

- [ ] **Step 6: Lint and commit**

```bash
uvx pre-commit run -a
git add newton/_src/vehicles/tire.py newton/_src/vehicles/wheel.py newton/tests/test_vehicles_tire.py
git commit -m "Add Pacejka Magic Formula branch to tire_force"
```

---

### Task 2: `tire_slip_stiffness` and wheel-kernel linearization

**Files:**
- Modify: `newton/_src/vehicles/tire.py`
- Modify: `newton/_src/vehicles/wheel.py:309-311` (secant slope lines)
- Test: `newton/tests/test_vehicles_tire.py`

**Interfaces:**
- Consumes: Task 1's `MF_*` constants, `mf_params` plumbing.
- Produces: `tire_slip_stiffness(model_id: int, fz: float, mu: float, c_long: float, c_lat: float, mf: wp.array2d[wp.float32], w: int) -> wp.vec2` — zero-slip force-per-unit-slip slopes `(k_long [N/rad], k_lat [N/rad])`; brush/linear return `(c_long*fz, c_lat*fz)` exactly; Pacejka returns `(Bx*Cx*mu*fz, By*Cy*mu*fz)`. Plus `_eval_stiffness_kernel` for tests.

- [ ] **Step 1: Write the failing tests** — append to `test_vehicles_tire.py`:

```python
from newton._src.vehicles.tire import _eval_stiffness_kernel  # add to existing import


class TestTireSlipStiffness(unittest.TestCase):
    def _stiffness(self, model_id, fz, mu, c_long=20.0, c_lat=30.0, mf=None, device="cpu"):
        mf_arr = (
            wp.zeros((1, 10), dtype=wp.float32, device=device)
            if mf is None
            else wp.array(np.asarray(mf, dtype=np.float32).reshape(1, 10), dtype=wp.float32, device=device)
        )
        out = wp.zeros(1, dtype=wp.vec2, device=device)
        wp.launch(
            _eval_stiffness_kernel,
            dim=1,
            inputs=[int(model_id), float(fz), float(mu), float(c_long), float(c_lat), mf_arr, out],
            device=device,
        )
        return out.numpy()[0]

    def test_brush_and_linear_slopes_are_legacy(self):
        for model in (TIRE_BRUSH, TIRE_LINEAR):
            k = self._stiffness(model, fz=1500.0, mu=0.9)
            self.assertEqual(float(k[0]), 20.0 * 1500.0)
            self.assertEqual(float(k[1]), 30.0 * 1500.0)

    def test_pacejka_slope_is_bcd(self):
        k = self._stiffness(TIRE_PACEJKA, fz=1000.0, mu=0.8, mf=_MF_THETA)
        self.assertAlmostEqual(float(k[0]), 12.5 * 1.6 * 0.8 * 1000.0, delta=1.0)
        self.assertAlmostEqual(float(k[1]), 14.0 * 1.35 * 0.8 * 1000.0, delta=1.0)

    def test_pacejka_slope_matches_finite_difference(self):
        eps = 1.0e-4
        f = _eval(TIRE_PACEJKA, [eps], [0.0], [1000.0], [0.8], [20.0], [20.0], mf=_MF_THETA)[0]
        k = self._stiffness(TIRE_PACEJKA, fz=1000.0, mu=0.8, mf=_MF_THETA)
        self.assertAlmostEqual(float(f[0]) / eps, float(k[0]), delta=0.02 * float(k[0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire`
Expected: FAIL with `ImportError: cannot import name '_eval_stiffness_kernel'`.

- [ ] **Step 3: Implement in `tire.py`** (after `tire_force`):

```python
@wp.func
def tire_slip_stiffness(
    model_id: int,
    fz: float,
    mu: float,
    c_long: float,
    c_lat: float,
    mf: wp.array2d[wp.float32],
    w: int,
) -> wp.vec2:
    """Zero-slip force-per-unit-slip slopes ``(k_long, k_lat)`` [N/rad].

    The wheel impulse solve caps its secant stiffness at these linear-regime
    slopes. Brush and linear models slope as ``c * Fz``; the Magic Formula
    slopes as ``B * C * D`` with ``D = mu * Fz``.
    """
    if model_id == TIRE_PACEJKA:
        d = wp.max(mu, 0.0) * wp.max(fz, 0.0)
        return wp.vec2(mf[w, MF_BX] * mf[w, MF_CX] * d, mf[w, MF_BY] * mf[w, MF_CY] * d)
    return wp.vec2(c_long * fz, c_lat * fz)


@wp.kernel
def _eval_stiffness_kernel(
    model_id: int,
    fz: float,
    mu: float,
    c_long: float,
    c_lat: float,
    mf_params: wp.array2d[wp.float32],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = tire_slip_stiffness(model_id, fz, mu, c_long, c_lat, mf_params, i)
```

- [ ] **Step 4: Use it in `wheel.py`.** Change the import to `from .tire import MF_PARAM_COUNT, tire_force, tire_slip_stiffness` and replace lines 309-311:

```python
                # secant impulse stiffness dt*C, capped by the linear-regime slope
                k_lin_long = dt * c_long[w] * fz / ref
                k_lin_lat = dt * c_lat[w] * fz / ref
```

with:

```python
                # secant impulse stiffness dt*C, capped by the model's linear-regime slope
                ks = tire_slip_stiffness(tire_model[w], fz, mu, c_long[w], c_lat[w], mf_params, w)
                k_lin_long = dt * ks[0] / ref
                k_lin_lat = dt * ks[1] / ref
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire && uv run --extra dev -m newton.tests -k test_vehicles_wheel`
Expected: PASS. The wheel suite passing confirms brush behavior is bit-identical (`ks` reduces to the exact same products).

- [ ] **Step 6: Lint and commit**

```bash
uvx pre-commit run -a
git add newton/_src/vehicles/tire.py newton/_src/vehicles/wheel.py newton/tests/test_vehicles_tire.py
git commit -m "Add tire_slip_stiffness for model-aware impulse slopes"
```

---

### Task 3: Public API — `TireModel.PACEJKA`, `PacejkaParams`, config plumbing, docs

**Files:**
- Modify: `newton/_src/vehicles/controller.py`
- Modify: `newton/_src/vehicles/__init__.py`, `newton/vehicles.py`
- Modify: `CHANGELOG.md`
- Test: `newton/tests/test_vehicles_tire.py`

**Interfaces:**
- Consumes: Task 1's `MF_PARAM_COUNT` and layout, `WheelDynamics.mf_params`.
- Produces:
  - `newton.vehicles.TireModel.PACEJKA == 2`
  - `newton.vehicles.PacejkaParams` dataclass, fields in `mf_params` column order: `bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2` (all `float`), with `as_array() -> np.ndarray` (shape `(10,)`, float32) and `names() -> tuple[str, ...]` classmethod
  - `WheeledConfig.pacejka: PacejkaParams` (default factory)
  - `WheeledVehicles.PacejkaParams` class alias
  - Constructing `WheeledVehicles` fills `dynamics.mf_params` rows with `config.pacejka.as_array()`

- [ ] **Step 1: Write the failing tests** — append to `test_vehicles_tire.py`:

```python
import dataclasses

from newton.vehicles import PacejkaParams, TireModel
from newton._src.vehicles import tire as _tire


class TestPacejkaParamsLayout(unittest.TestCase):
    def test_enum_value(self):
        self.assertEqual(int(TireModel.PACEJKA), 2)

    def test_field_order_matches_mf_constants(self):
        names = tuple(f.name for f in dataclasses.fields(PacejkaParams))
        self.assertEqual(names, ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2"))
        self.assertEqual(names, PacejkaParams.names())
        self.assertEqual(len(names), _tire.MF_PARAM_COUNT)
        order = (_tire.MF_BX, _tire.MF_CX, _tire.MF_EX, _tire.MF_BY, _tire.MF_CY,
                 _tire.MF_EY, _tire.MF_RBX1, _tire.MF_RBX2, _tire.MF_RBY1, _tire.MF_RBY2)
        self.assertEqual(order, tuple(range(_tire.MF_PARAM_COUNT)))

    def test_as_array_roundtrip(self):
        p = PacejkaParams(bx=1.0, cx=2.0, ex=3.0, by=4.0, cy=5.0, ey=6.0,
                          rbx1=7.0, rbx2=8.0, rby1=9.0, rby2=10.0)
        np.testing.assert_array_equal(p.as_array(), np.arange(1.0, 11.0, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire`
Expected: FAIL with `ImportError: cannot import name 'PacejkaParams'`.

- [ ] **Step 3: Implement in `controller.py`.** Add `PACEJKA` to `TireModel`:

```python
    PACEJKA = 2
    """Core Magic Formula with cosine combined-slip weighting (parameters in :class:`PacejkaParams`)."""
```

Add after the `TireModel` class (module level; also change the dataclasses import line to `from dataclasses import dataclass, field`):

```python
@dataclass
class PacejkaParams:
    """Magic Formula shape parameters for :attr:`TireModel.PACEJKA` (dimensionless).

    Field order is public API: it defines the column layout of the per-wheel
    parameter table, so new parameters must be appended, never reordered. The
    peak force is ``D = mu * Fz`` (``mu`` stays a separate config/contact value),
    and the zero-slip slope per axis is ``B * C * D``. Defaults are placeholder
    shapes roughly matching the brush model; calibrated values come from the
    ``tools/tirecal`` pipeline.

    Args:
        bx: Longitudinal stiffness factor ``B_x``.
        cx: Longitudinal shape factor ``C_x`` (>1 gives peak-then-falloff).
        ex: Longitudinal curvature factor ``E_x`` (must be <= 1).
        by: Lateral stiffness factor ``B_y``.
        cy: Lateral shape factor ``C_y``.
        ey: Lateral curvature factor ``E_y`` (must be <= 1).
        rbx1: Combined-slip weight of lateral slip on ``F_x`` (magnitude).
        rbx2: Combined-slip weight shaping vs longitudinal slip.
        rby1: Combined-slip weight of longitudinal slip on ``F_y`` (magnitude).
        rby2: Combined-slip weight shaping vs lateral slip.
    """

    bx: float = 12.5
    cx: float = 1.6
    ex: float = 0.4
    by: float = 14.0
    cy: float = 1.35
    ey: float = 0.3
    rbx1: float = 12.0
    rbx2: float = 10.0
    rby1: float = 10.0
    rby2: float = 8.0

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Field names in table-column order."""
        return tuple(f.name for f in dataclasses_fields(cls))

    def as_array(self) -> np.ndarray:
        """Parameter vector in table-column order, dtype float32, shape [10]."""
        return np.array([getattr(self, n) for n in self.names()], dtype=np.float32)
```

(Import `fields as dataclasses_fields` from `dataclasses` at the top of the file: `from dataclasses import dataclass, field, fields as dataclasses_fields`.)

In `WheeledConfig`, add the field after `apply_reaction_torque` and its docstring `Args:` entry (`pacejka: Magic Formula parameters used by wheels with tire_model PACEJKA.`):

```python
    pacejka: PacejkaParams = field(default_factory=PacejkaParams)
```

In `WheeledVehicles`, add `PacejkaParams = PacejkaParams` alongside the existing `Config = WheeledConfig` aliases. In `_init_params`, after the `pneumatic_trail` line add:

```python
        d.mf_params.assign(np.tile(c.pacejka.as_array(), (max(self.wheel_count, 1), 1)))
```

- [ ] **Step 4: Export.** In `newton/_src/vehicles/__init__.py` add `PacejkaParams` to the controller import and `__all__` (inspect the file; it mirrors the public list). In `newton/vehicles.py` add `PacejkaParams,` to the `from ._src.vehicles import (...)` block (alphabetical position: after `DriveMode`) and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_tire`
Expected: PASS.

- [ ] **Step 6: Docs + changelog.** Run `uv run python docs/generate_api.py` (commit any generated changes). Insert into `CHANGELOG.md` `[Unreleased]` → `Added` at a **random position within the category** (pick one that is not first or last):

```markdown
- Add Pacejka (Magic Formula) tire model for wheeled vehicles: `TireModel.PACEJKA` with shape parameters in `PacejkaParams` (`WheeledConfig.pacejka`)
```

- [ ] **Step 7: Lint and commit**

```bash
uvx pre-commit run -a
git add newton/_src/vehicles/controller.py newton/_src/vehicles/__init__.py newton/vehicles.py CHANGELOG.md docs/ newton/tests/test_vehicles_tire.py
git commit -m "Expose TireModel.PACEJKA and PacejkaParams config"
```

---

### Task 4: tirecal scaffold, HDF5 format, SCHEMA.md, pyproject extra

**Files:**
- Create: `tools/tirecal/tirecal/__init__.py`, `tools/tirecal/tirecal/format.py`, `tools/tirecal/SCHEMA.md`, `tools/tirecal/README.md`, `tools/tirecal/tests/__init__.py`, `tools/tirecal/tests/test_format.py`
- Modify: `pyproject.toml` (optional-dependency extra), `.gitignore`

**Interfaces:**
- Consumes: nothing (standalone: numpy + h5py only; **must not import newton**).
- Produces (exact API later tasks use):
  - `tirecal.format.SCHEMA_VERSION = 1`
  - `tirecal.format.Lap` dataclass: `time [T+1]`, `body_q [T+1,7]`, `body_qd [T+1,6]`, `wheel_omega [T+1,W]`, `controls [T,A]`, `control_names: tuple[str, ...]`, `meta: dict`, `sim_state: dict[str, np.ndarray] | None`
  - `tirecal.format.Session` dataclass: `vehicle: dict`, `laps: list[Lap]`, `source: str`, `notes: str`
  - `tirecal.format.write_session(path, session)` / `read_session(path) -> Session`

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_format.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tirecal import format as tcformat


def _random_lap(rng, steps=20, wheels=4, actions=3):
    t1 = steps + 1
    return tcformat.Lap(
        time=np.arange(t1, dtype=np.float64) / 60.0,
        body_q=rng.standard_normal((t1, 7)).astype(np.float32),
        body_qd=rng.standard_normal((t1, 6)).astype(np.float32),
        wheel_omega=rng.standard_normal((t1, wheels)).astype(np.float32),
        controls=rng.standard_normal((steps, actions)).astype(np.float32),
        control_names=("drive", "steer", "brake"),
        meta={"sim_dt": 1.0 / 480.0, "sim_substeps": 8, "control_rate_hz": 60.0, "track_seed": 7},
        sim_state={
            "joint_q": rng.standard_normal((t1, 11)).astype(np.float32),
            "joint_qd": rng.standard_normal((t1, 10)).astype(np.float32),
            "body_q_full": rng.standard_normal((t1, 5, 7)).astype(np.float32),
            "body_qd_full": rng.standard_normal((t1, 5, 6)).astype(np.float32),
            "trans_long": rng.standard_normal((t1, wheels)).astype(np.float32),
            "trans_lat": rng.standard_normal((t1, wheels)).astype(np.float32),
            "patch_fz": rng.standard_normal((t1, wheels)).astype(np.float32),
        },
    )


class TestFormatRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        session = tcformat.Session(
            vehicle={"mass": 4.2, "tire_model": "brush", "tire_params": {"c_long": 20.0}},
            laps=[_random_lap(rng), _random_lap(rng, steps=33)],
            source="newton-sim",
            notes="unit test",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            back = tcformat.read_session(path)
        self.assertEqual(back.source, "newton-sim")
        self.assertEqual(back.vehicle["tire_params"]["c_long"], 20.0)
        self.assertEqual(len(back.laps), 2)
        for a, b in zip(session.laps, back.laps, strict=True):
            np.testing.assert_array_equal(a.body_q, b.body_q)
            np.testing.assert_array_equal(a.controls, b.controls)
            self.assertEqual(a.control_names, b.control_names)
            self.assertEqual(a.meta["track_seed"], b.meta["track_seed"])
            for k, v in a.sim_state.items():
                np.testing.assert_array_equal(v, b.sim_state[k])

    def test_optional_sim_state(self):
        rng = np.random.default_rng(1)
        lap = _random_lap(rng)
        lap.sim_state = None
        session = tcformat.Session(vehicle={}, laps=[lap])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            back = tcformat.read_session(path)
        self.assertIsNone(back.laps[0].sim_state)

    def test_rejects_newer_schema(self):
        import h5py

        rng = np.random.default_rng(2)
        session = tcformat.Session(vehicle={}, laps=[_random_lap(rng)])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            with h5py.File(path, "a") as f:
                f.attrs["schema_version"] = tcformat.SCHEMA_VERSION + 1
            with self.assertRaises(ValueError):
                tcformat.read_session(path)

    def test_format_module_is_standalone(self):
        import sys

        self.assertNotIn("newton", sys.modules.get("tirecal.format", tcformat).__dict__.get("__all_imports__", ()))
        # the real guard: importing tirecal.format must not pull in newton
        self.assertTrue(all(not m.startswith("newton") for m in sys.modules if "tirecal" in m))
```

- [ ] **Step 2: Add the extra and scaffold.** In `pyproject.toml` `[project.optional-dependencies]`, after the `onnx` block add:

```toml
# Tire-calibration tooling (tools/tirecal): trajectory recording and fitting
tirecal = [
    "h5py>=3.8.0",
]
```

Append `tirecal-data/` on its own line to `.gitignore`. Create `tools/tirecal/tirecal/__init__.py` and `tools/tirecal/tests/__init__.py`, each containing only the two SPDX lines (2026) and, for the package one, the docstring `"""tirecal: trajectory-based tire-model calibration tooling."""`.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v`
Expected: FAIL — `AttributeError`/`ImportError` on `tcformat.Lap`.

- [ ] **Step 4: Implement `tools/tirecal/tirecal/format.py`:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""HDF5 trajectory session format, schema v1. Standalone: numpy + h5py only.

See ``SCHEMA.md`` for field semantics, SI units, and frame conventions. This
module is the contract shared by simulator recordings and future real-vehicle
logs; it must never import newton.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

SCHEMA_VERSION = 1

_LAP_ARRAYS = ("time", "body_q", "body_qd", "wheel_omega", "controls")


@dataclass
class Lap:
    """One recorded lap at control rate.

    State datasets have T+1 rows (samples 0..T); ``controls`` has T rows, where
    ``controls[t]`` drives the transition from state sample ``t`` to ``t+1``.
    """

    time: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    wheel_omega: np.ndarray
    controls: np.ndarray
    control_names: tuple
    meta: dict = field(default_factory=dict)
    sim_state: dict | None = None


@dataclass
class Session:
    vehicle: dict
    laps: list
    source: str = "newton-sim"
    notes: str = ""


def write_session(path, session: Session) -> None:
    """Write a session to ``path``, overwriting any existing file."""
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["source"] = session.source
        f.attrs["notes"] = session.notes
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.create_group("vehicle").attrs["meta"] = json.dumps(session.vehicle)
        laps = f.create_group("laps")
        for i, lap in enumerate(session.laps):
            g = laps.create_group(f"{i:04d}")
            g.attrs["meta"] = json.dumps(lap.meta)
            g.attrs["control_names"] = json.dumps(list(lap.control_names))
            for name in _LAP_ARRAYS:
                g.create_dataset(name, data=np.asarray(getattr(lap, name)))
            if lap.sim_state is not None:
                s = g.create_group("sim_state")
                for k, v in lap.sim_state.items():
                    s.create_dataset(k, data=np.asarray(v))


def read_session(path) -> Session:
    """Read a session; raises ValueError on a newer schema version."""
    with h5py.File(path, "r") as f:
        version = int(f.attrs["schema_version"])
        if version > SCHEMA_VERSION:
            raise ValueError(f"session schema v{version} is newer than supported v{SCHEMA_VERSION}")
        vehicle = json.loads(f["vehicle"].attrs["meta"])
        laps = []
        for key in sorted(f["laps"]):
            g = f["laps"][key]
            sim_state = None
            if "sim_state" in g:
                sim_state = {k: g["sim_state"][k][...] for k in g["sim_state"]}
            laps.append(
                Lap(
                    **{name: g[name][...] for name in _LAP_ARRAYS},
                    control_names=tuple(json.loads(g.attrs["control_names"])),
                    meta=json.loads(g.attrs["meta"]),
                    sim_state=sim_state,
                )
            )
        return Session(
            vehicle=vehicle, laps=laps, source=str(f.attrs["source"]), notes=str(f.attrs["notes"])
        )
```

- [ ] **Step 5: Write `tools/tirecal/SCHEMA.md`** documenting exactly: root attrs (`schema_version`, `source` = `"newton-sim" | "real"`, `created` ISO-8601 UTC, `notes`); `/vehicle` attr `meta` (JSON: mass [kg], wheelbase [m], track width [m], wheel radius [m], generator tire model name + named parameter dict, plus for sim sources the per-world layout counts); `/laps/NNNN` with attr `meta` (JSON; required keys for sim sources: `sim_dt` [s], `sim_substeps`, `control_rate_hz`, `track_seed`, `track_generator`) and attr `control_names` (JSON list mapping control columns); datasets `time [T+1]` [s], `body_q [T+1,7]` chassis position + quaternion `(x,y,z,qx,qy,qz,qw)` world frame [m], `body_qd [T+1,6]` angular then linear velocity, world frame, Newton spatial convention [rad/s, m/s], `wheel_omega [T+1,W]` [rad/s], `controls [T,A]` normalized commands; the T/T+1 alignment rule (`controls[t]`: sample t → t+1); optional `sim_state/` group (sim sources only, never required by readers — real logs won't have it): `joint_q [T+1,D]`, `joint_qd [T+1,Dv]`, `body_q_full [T+1,B,7]`, `body_qd_full [T+1,B,6]`, `trans_long/trans_lat [T+1,W]` [m/s], `patch_fz [T+1,W]` [N]. State the versioning rule (readers reject newer majors; additions are backward-compatible) and that channels/params are name-keyed so richer sets never break readers.

- [ ] **Step 6: Write `tools/tirecal/README.md`** — one page: purpose, install (`uv sync --extra tirecal --extra sim`), invocation pattern (`PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.generate ...`), test command, pointer to SCHEMA.md and the spec.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v`
Expected: PASS (4 tests).

- [ ] **Step 8: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal pyproject.toml .gitignore uv.lock
git commit -m "Add tirecal package with HDF5 trajectory format"
```

(If `uv.lock` did not change because the extra resolves lazily, commit without it.)

---

### Task 5: `StrategyGA` (ask/tell) + benchmarks

**Files:**
- Create: `tools/tirecal/tirecal/ga.py`
- Test: `tools/tirecal/tests/test_ga.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces (exact API): `tirecal.ga.StrategyGA(bounds_lo, bounds_hi, population=64, x0=None, init_spread=0.25, seed=0, tournament_k=3, crossover_prob=0.9, eta_c=15.0, mutation_prob=None, eta_m=20.0, elite=2)` with `ask() -> np.ndarray (P, n)`, `tell(fitness: np.ndarray (P,))`, `best_x: np.ndarray`, `best_f: float`. Lower fitness is better. Task 6 adds `StrategyCMAES` to the same module; `tune.py` treats both through ask/tell only.

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_ga.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_ga`
Expected: FAIL — `ModuleNotFoundError: No module named 'tirecal.ga'`.

- [ ] **Step 3: Implement `tools/tirecal/tirecal/ga.py`:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask/tell black-box optimizer strategies. Standalone: numpy only.

Protocol: ``x = strategy.ask()`` returns a ``(population, n)`` candidate
matrix; ``strategy.tell(fitness)`` reports the ``(population,)`` fitness of
exactly that matrix (lower is better). ``best_x``/``best_f`` track the best
candidate ever told.
"""

from __future__ import annotations

import numpy as np


class StrategyGA:
    """Genetic algorithm: tournament selection, SBX crossover, polynomial
    mutation, elitism. Bounds are enforced by clipping."""

    def __init__(
        self,
        bounds_lo,
        bounds_hi,
        population: int = 64,
        x0=None,
        init_spread: float = 0.25,
        seed: int = 0,
        tournament_k: int = 3,
        crossover_prob: float = 0.9,
        eta_c: float = 15.0,
        mutation_prob: float | None = None,
        eta_m: float = 20.0,
        elite: int = 2,
    ):
        self.lo = np.asarray(bounds_lo, dtype=np.float64)
        self.hi = np.asarray(bounds_hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape or np.any(self.hi <= self.lo):
            raise ValueError("invalid bounds")
        self.n = self.lo.size
        self.population = int(population)
        self.rng = np.random.default_rng(seed)
        self.tournament_k = tournament_k
        self.crossover_prob = crossover_prob
        self.eta_c = eta_c
        self.mutation_prob = 1.0 / self.n if mutation_prob is None else mutation_prob
        self.eta_m = eta_m
        self.elite = int(elite)
        span = self.hi - self.lo
        if x0 is None:
            self._pop = self.lo + self.rng.random((self.population, self.n)) * span
        else:
            x0 = np.clip(np.asarray(x0, dtype=np.float64), self.lo, self.hi)
            self._pop = x0 + self.rng.standard_normal((self.population, self.n)) * init_spread * span
            self._pop[0] = x0  # keep the seed itself in the population
        self._pop = np.clip(self._pop, self.lo, self.hi)
        self._fit = None
        self._pending = self._pop.copy()
        self.best_x = None
        self.best_f = np.inf

    def ask(self) -> np.ndarray:
        return self._pending.copy()

    def tell(self, fitness) -> None:
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.shape != (self.population,):
            raise ValueError(f"expected fitness shape ({self.population},), got {fitness.shape}")
        pop = self._pending
        i_best = int(np.argmin(fitness))
        if fitness[i_best] < self.best_f:
            self.best_f = float(fitness[i_best])
            self.best_x = pop[i_best].copy()
        order = np.argsort(fitness)
        nxt = np.empty_like(pop)
        nxt[: self.elite] = pop[order[: self.elite]]
        i = self.elite
        while i < self.population:
            p1 = pop[self._tournament(fitness)]
            p2 = pop[self._tournament(fitness)]
            c1, c2 = self._sbx(p1, p2)
            nxt[i] = self._mutate(c1)
            if i + 1 < self.population:
                nxt[i + 1] = self._mutate(c2)
            i += 2
        self._pending = np.clip(nxt, self.lo, self.hi)

    def _tournament(self, fitness) -> int:
        idx = self.rng.integers(0, self.population, self.tournament_k)
        return int(idx[np.argmin(fitness[idx])])

    def _sbx(self, p1, p2):
        if self.rng.random() >= self.crossover_prob:
            return p1.copy(), p2.copy()
        u = self.rng.random(self.n)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0)),
        )
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        return c1, c2

    def _mutate(self, x):
        span = self.hi - self.lo
        mask = self.rng.random(self.n) < self.mutation_prob
        u = self.rng.random(self.n)
        delta = np.where(
            u < 0.5,
            (2.0 * u) ** (1.0 / (self.eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_m + 1.0)),
        )
        return np.where(mask, x + delta * span, x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_ga`
Expected: PASS. If a benchmark threshold is flaky across numpy versions, loosen it by at most 2x and note it in the test comment — do not change the algorithm to chase a threshold.

- [ ] **Step 5: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/ga.py tools/tirecal/tests/test_ga.py
git commit -m "Add ask/tell GA strategy with SBX and polynomial mutation"
```

---

### Task 6: `StrategyCMAES` + benchmarks

**Files:**
- Modify: `tools/tirecal/tirecal/ga.py` (append class)
- Test: `tools/tirecal/tests/test_cmaes.py`

**Interfaces:**
- Consumes: Task 5's module/protocol.
- Produces: `tirecal.ga.StrategyCMAES(bounds_lo, bounds_hi, x0, sigma0=0.25, population=None, seed=0)` — same ask/tell/best protocol. Internally normalized to the unit box; `sigma0` is in normalized units; `population` defaults to `4 + floor(3*ln(n))` but accepts large values (rank-mu update uses the top half).

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_cmaes.py`:

```python
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
        kw = dict(x0=np.full(3, 0.5), population=16, seed=9)
        a = StrategyCMAES(np.zeros(3), np.ones(3), **kw).ask()
        b = StrategyCMAES(np.zeros(3), np.ones(3), **kw).ask()
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_cmaes`
Expected: FAIL — `ImportError: cannot import name 'StrategyCMAES'`.

- [ ] **Step 3: Append to `tools/tirecal/tirecal/ga.py`:**

```python
class StrategyCMAES:
    """CMA-ES (rank-mu update, Hansen's default constants), self-contained.

    Operates internally in the unit box (coordinates normalized by the bounds),
    which makes one isotropic ``sigma0`` meaningful across parameters with very
    different ranges. Bounds are enforced by projection; the update uses the
    projected candidates.
    """

    def __init__(self, bounds_lo, bounds_hi, x0, sigma0: float = 0.25, population: int | None = None, seed: int = 0):
        self.lo = np.asarray(bounds_lo, dtype=np.float64)
        self.hi = np.asarray(bounds_hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape or np.any(self.hi <= self.lo):
            raise ValueError("invalid bounds")
        self.span = self.hi - self.lo
        n = self.lo.size
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.lam = int(population) if population else 4 + int(3.0 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w**2)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chin = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.m = np.clip((np.asarray(x0, dtype=np.float64) - self.lo) / self.span, 0.0, 1.0)
        self.sigma = float(sigma0)
        self.C = np.eye(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self._pending_y = None
        self._pending_x = None
        self.best_x = None
        self.best_f = np.inf

    def ask(self) -> np.ndarray:
        vals, vecs = np.linalg.eigh(self.C)
        vals = np.maximum(vals, 1.0e-20)
        self._sqrtC = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        self._invsqrtC = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ self._sqrtC.T
        x = np.clip(self.m + self.sigma * y, 0.0, 1.0)
        self._pending_y = (x - self.m) / self.sigma  # projected steps
        self._pending_x = x
        return self.lo + x * self.span

    def tell(self, fitness) -> None:
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.shape != (self.lam,):
            raise ValueError(f"expected fitness shape ({self.lam},), got {fitness.shape}")
        order = np.argsort(fitness)
        i_best = int(order[0])
        if fitness[i_best] < self.best_f:
            self.best_f = float(fitness[i_best])
            self.best_x = self.lo + self._pending_x[i_best] * self.span
        y_sel = self._pending_y[order[: self.mu]]
        y_bar = self.w @ y_sel
        self.m = np.clip(self.m + self.sigma * y_bar, 0.0, 1.0)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (
            self._invsqrtC @ y_bar
        )
        hsig = float(np.linalg.norm(self.ps) / np.sqrt(1.0 - (1.0 - self.cs) ** 2) / self.chin < 1.4 + 2.0 / (self.n + 1.0))
        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_bar
        rank_mu = (y_sel * self.w[:, None]).T @ y_sel
        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1.0))
        self.sigma = float(np.clip(self.sigma, 1.0e-12, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_cmaes`
Expected: PASS (same flakiness rule as Task 5).

- [ ] **Step 5: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/ga.py tools/tirecal/tests/test_cmaes.py
git commit -m "Add CMA-ES strategy behind the ask/tell protocol"
```

---

### Task 7: `seed_fit.py` — standalone MF-to-brush curve fit

**Files:**
- Create: `tools/tirecal/tirecal/seed_fit.py`
- Test: `tools/tirecal/tests/test_seed_fit.py`

**Interfaces:**
- Consumes: `tirecal.ga.StrategyCMAES`.
- Produces:
  - `tirecal.seed_fit.PARAM_NAMES = ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2")`
  - `tirecal.seed_fit.DEFAULT_BOUNDS: dict[str, tuple[float, float]]` — `b*`: (2, 30), `c*`: (1.0, 2.5), `e*`: (-2.0, 1.0), `r*1`: (0.0, 30.0), `r*2`: (0.0, 30.0)
  - `tirecal.seed_fit.bounds_arrays(bounds: dict) -> tuple[np.ndarray, np.ndarray]` (in `PARAM_NAMES` order)
  - `tirecal.seed_fit.brush_forces(kappa, alpha, mu, c_long, c_lat) -> (fx, fy)` per unit Fz
  - `tirecal.seed_fit.mf_forces(theta, kappa, alpha, mu) -> (fx, fy)` per unit Fz
  - `tirecal.seed_fit.fit_to_brush(mu, c_long, c_lat, bounds=None, seed=0, iters=300, population=32) -> tuple[np.ndarray, float]` — (theta in `PARAM_NAMES` order, RMS curve error as a fraction of `mu`)

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_seed_fit.py`:

```python
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

        from tirecal.ga import StrategyCMAES

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_seed_fit`
Expected: FAIL — `ModuleNotFoundError: No module named 'tirecal.seed_fit'`.

- [ ] **Step 3: Implement `tools/tirecal/tirecal/seed_fit.py`:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Least-squares MF-to-brush curve fit used to seed the trajectory optimizer.

Standalone: numpy only. Both tire laws scale linearly with Fz (brush through
``phi = C*Fz*s / (3*mu*Fz)``, MF through ``D = mu*Fz``), so curves are fit per
unit normal load on a (kappa, alpha) grid. The optimizer is the package's own
CMA-ES on the curve MSE.
"""

from __future__ import annotations

import numpy as np

from .ga import StrategyCMAES

PARAM_NAMES = ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2")

DEFAULT_BOUNDS = {
    "bx": (2.0, 30.0),
    "cx": (1.0, 2.5),
    "ex": (-2.0, 1.0),
    "by": (2.0, 30.0),
    "cy": (1.0, 2.5),
    "ey": (-2.0, 1.0),
    "rbx1": (0.0, 30.0),
    "rbx2": (0.0, 30.0),
    "rby1": (0.0, 30.0),
    "rby2": (0.0, 30.0),
}


def bounds_arrays(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays in PARAM_NAMES order."""
    lo = np.array([bounds[n][0] for n in PARAM_NAMES], dtype=np.float64)
    hi = np.array([bounds[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lo, hi


def brush_forces(kappa, alpha, mu: float, c_long: float, c_lat: float):
    """Brush combined-slip forces per unit Fz (textbook parabolic-pressure law).

    Mirrors newton's kernel independently (same formulas as the golden-curve
    test reference in newton/tests/test_vehicles_tire.py).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    fx_lin = c_long * k * inv
    fy_lin = c_lat * np.tan(alpha) * inv
    f_lin = np.hypot(fx_lin, fy_lin)
    phi = f_lin / (3.0 * mu)
    f_mag = np.where(phi < 1.0, 3.0 * mu * phi * (1.0 - phi + phi * phi / 3.0), mu)
    scale = np.divide(f_mag, f_lin, out=np.zeros_like(f_lin), where=f_lin >= 1.0e-9)
    return scale * fx_lin, -scale * fy_lin


def mf_forces(theta, kappa, alpha, mu: float):
    """Core-MF + cosine combined-slip forces per unit Fz (mirrors the kernel)."""
    bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1, rby2 = np.asarray(theta, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.maximum(kappa, -0.9999)
    inv = 1.0 / (1.0 + k)
    sx = k * inv
    sy = np.tan(alpha) * inv

    def mf(s, b, c, e):
        bs = b * s
        return np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))

    fx0 = mu * mf(sx, bx, cx, ex)
    fy0 = mu * mf(sy, by, cy, ey)
    gxa = np.cos(np.arctan(rbx1 * np.cos(np.arctan(rbx2 * sx)) * sy))
    gyk = np.cos(np.arctan(rby1 * np.cos(np.arctan(rby2 * sy)) * sx))
    return gxa * fx0, -gyk * fy0


def fit_to_brush(
    mu: float,
    c_long: float,
    c_lat: float,
    bounds: dict | None = None,
    seed: int = 0,
    iters: int = 300,
    population: int = 32,
) -> tuple[np.ndarray, float]:
    """Fit MF curves to the brush curves; returns (theta, rms_error / mu)."""
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    lo, hi = bounds_arrays(bounds)
    kg, ag = np.meshgrid(np.linspace(-0.8, 0.8, 33), np.linspace(-0.6, 0.6, 25))
    kappa, alpha = kg.ravel(), ag.ravel()
    fx_t, fy_t = brush_forces(kappa, alpha, mu, c_long, c_lat)
    es = StrategyCMAES(lo, hi, x0=0.5 * (lo + hi), population=population, seed=seed)
    for _ in range(iters):
        pop = es.ask()
        errs = np.empty(pop.shape[0])
        for i, th in enumerate(pop):
            fx, fy = mf_forces(th, kappa, alpha, mu)
            errs[i] = np.mean((fx - fx_t) ** 2 + (fy - fy_t) ** 2)
        es.tell(errs)
    return es.best_x.copy(), float(np.sqrt(es.best_f) / mu)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_seed_fit`
Expected: PASS. Note: the brush model has no falloff past the peak while MF with C>1 does; a ~5% RMS bound is the achievable regime, not a bug.

- [ ] **Step 5: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/seed_fit.py tools/tirecal/tests/test_seed_fit.py
git commit -m "Add standalone MF-to-brush curve-fit seeder"
```

---

### Task 8: `generate.py` — MPPI lap recorder

**Files:**
- Create: `tools/tirecal/tirecal/generate.py`
- Test: `tools/tirecal/tests/test_generate.py`

**Interfaces:**
- Consumes: `tirecal.format` (Task 4); `newton.examples.vehicles.example_vehicle_mppi_track.Example` and `newton.viewer.ViewerNull` (existing).
- Produces:
  - CLI: `python -m tirecal.generate --out F.h5 --laps N --track-seed0 S [--num-samples K] [--frames-max F] [--brake-mode M] [--track-generator G] [--tire-model brush|pacejka] [--test]`
  - `tirecal.generate.record_lap(track_seed, num_samples, frames_max, brake_mode, track_generator, tire_model, test_mode) -> tuple[format.Lap, dict]` — the dict is the vehicle metadata (identical across laps; caller keeps the first)
  - Vehicle meta keys used by Task 9 (exact): `config` (the `WheeledConfig` numbers used, as a flat dict), `tire_model` (str), `tire_params` (named dict: brush → `{c_long, c_lat, mu}`; pacejka → PacejkaParams-named dict), `frame_dt`, `sim_substeps`, `bodies_per_world`, `dofs_per_world`, `vel_dofs_per_world`, `wheels_per_world`, `chassis_index`
  - Lap `meta` keys: `sim_dt`, `sim_substeps`, `control_rate_hz`, `track_seed`, `track_generator`, `frames`, `lap_complete` (bool), `track_len`

- [ ] **Step 1: Write the smoke test** — `tools/tirecal/tests/test_generate.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import newton  # noqa: F401

    _HAVE_SIM = True
except Exception:
    _HAVE_SIM = False


@unittest.skipUnless(_HAVE_SIM, "newton with sim extra required")
class TestGenerateSmoke(unittest.TestCase):
    def test_record_short_run(self):
        from tirecal import format as tcformat
        from tirecal import generate

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "smoke.h5"
            generate.main(
                [
                    "--out", str(out),
                    "--laps", "1",
                    "--track-seed0", "0",
                    "--frames-max", "20",
                    "--test",
                ]
            )
            session = tcformat.read_session(out)
        self.assertEqual(len(session.laps), 1)
        lap = session.laps[0]
        t = lap.controls.shape[0]
        self.assertGreaterEqual(t, 5)
        self.assertEqual(lap.body_q.shape, (t + 1, 7))
        self.assertEqual(lap.body_qd.shape, (t + 1, 6))
        self.assertEqual(lap.wheel_omega.shape[0], t + 1)
        self.assertEqual(lap.control_names, ("drive", "steer", "brake"))
        for key in ("joint_q", "joint_qd", "body_q_full", "body_qd_full", "trans_long", "trans_lat", "patch_fz"):
            self.assertIn(key, lap.sim_state)
            self.assertEqual(lap.sim_state[key].shape[0], t + 1)
        self.assertTrue(np.isfinite(lap.body_q).all())
        self.assertEqual(session.vehicle["tire_model"], "brush")
        self.assertIn("c_long", session.vehicle["tire_params"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_generate`
Expected: FAIL — `ModuleNotFoundError: No module named 'tirecal.generate'`.

- [ ] **Step 3: Implement `tools/tirecal/tirecal/generate.py`:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record brush-model MPPI laps into a tirecal session file.

Drives the existing MPPI track example headless (hero world only is recorded)
across a range of track seeds and writes laps through :mod:`tirecal.format`.
This module imports newton; the format it writes does not require it.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from . import format as tcformat

CONTROL_NAMES = ("drive", "steer", "brake")


def _example_args(parser, num_samples, track_seed, brake_mode, track_generator, test_mode):
    args = parser.parse_args([])
    args.num_samples = num_samples
    args.track_seed = track_seed
    args.brake_mode = brake_mode
    args.track_generator = track_generator
    args.test = test_mode
    return args


def _capture_row(ex):
    """Hero-slice state row (device -> host); returns dict of 1-sample arrays."""
    dyn, patch = ex.vehicles.dynamics, ex.vehicles.patch
    body_q_full = ex.state_0.body_q.numpy()[: ex.bodies_per_world].copy()
    body_qd_full = ex.state_0.body_qd.numpy()[: ex.bodies_per_world].copy()
    return {
        "joint_q": ex.state_0.joint_q.numpy()[: ex.dofs_per_world].copy(),
        "joint_qd": ex.state_0.joint_qd.numpy()[: ex.vel_dofs_per_world].copy(),
        "body_q_full": body_q_full,
        "body_qd_full": body_qd_full,
        "trans_long": dyn.trans_long.numpy()[: ex.wheels_per_world].copy(),
        "trans_lat": dyn.trans_lat.numpy()[: ex.wheels_per_world].copy(),
        "patch_fz": patch.fz.numpy()[: ex.wheels_per_world].copy(),
        "wheel_omega": dyn.omega.numpy()[: ex.wheels_per_world].copy(),
        "chassis_q": body_q_full[ex._chassis0].copy(),
        "chassis_qd": body_qd_full[ex._chassis0].copy(),
    }


def _override_tire_model(ex, tire_model: str):
    """Post-construction tire-model override (the example hardcodes brush)."""
    import newton.vehicles as nv

    if tire_model == "brush":
        return
    dyn = ex.vehicles.dynamics
    host = dyn.tire_model.numpy()
    host[...] = int(nv.TireModel.PACEJKA)
    dyn.tire_model.assign(host)
    dyn.mf_params.assign(
        np.tile(ex.vehicles.config.pacejka.as_array(), (max(dyn.wheel_count, 1), 1))
    )


def record_lap(track_seed, num_samples, frames_max, brake_mode, track_generator, tire_model, test_mode):
    import newton
    from newton.examples.vehicles.example_vehicle_mppi_track import Example

    viewer = newton.viewer.ViewerNull(num_frames=frames_max)
    args = _example_args(Example.create_parser(), num_samples, track_seed, brake_mode, track_generator, test_mode)
    ex = Example(viewer, args)
    _override_tire_model(ex, tire_model)

    rows = [_capture_row(ex)]
    controls = []
    lap_complete = False
    for _ in range(frames_max):
        ex.step()
        cmd = ex.vehicles.commands
        controls.append(
            [float(cmd.drive.numpy()[0]), float(cmd.steer.numpy()[0]), float(cmd.brake.numpy()[0])]
        )
        rows.append(_capture_row(ex))
        if float(ex.total_s.numpy()[0]) >= ex.track_len:
            lap_complete = True
            break

    steps = len(controls)
    stack = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    cfg = dataclasses.asdict(ex.vehicles.config)
    pacejka = cfg.pop("pacejka")
    lap = tcformat.Lap(
        time=np.arange(steps + 1, dtype=np.float64) * ex.frame_dt,
        body_q=stack["chassis_q"],
        body_qd=stack["chassis_qd"],
        wheel_omega=stack["wheel_omega"],
        controls=np.asarray(controls, dtype=np.float32),
        control_names=CONTROL_NAMES,
        meta={
            "sim_dt": ex.sim_dt,
            "sim_substeps": ex.sim_substeps,
            "control_rate_hz": ex.fps,
            "track_seed": int(ex.track_seed),
            "track_generator": track_generator,
            "frames": steps,
            "lap_complete": lap_complete,
            "track_len": float(ex.track_len),
        },
        sim_state={
            k: stack[k]
            for k in ("joint_q", "joint_qd", "body_q_full", "body_qd_full", "trans_long", "trans_lat", "patch_fz")
        },
    )
    tire_params = (
        {"c_long": cfg["longitudinal_stiffness"], "c_lat": cfg["lateral_stiffness"], "mu": cfg["friction"]}
        if tire_model == "brush"
        else pacejka
    )
    vehicle = {
        "config": cfg,
        "pacejka_defaults": pacejka,
        "tire_model": tire_model,
        "tire_params": tire_params,
        "frame_dt": ex.frame_dt,
        "sim_substeps": ex.sim_substeps,
        "bodies_per_world": ex.bodies_per_world,
        "dofs_per_world": ex.dofs_per_world,
        "vel_dofs_per_world": ex.vel_dofs_per_world,
        "wheels_per_world": ex.wheels_per_world,
        "chassis_index": int(ex._chassis0),
    }
    return lap, vehicle


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output session .h5 path")
    p.add_argument("--laps", type=int, default=24, help="number of laps (one track seed each)")
    p.add_argument("--track-seed0", type=int, default=0, help="first track seed; seed s records lap s")
    p.add_argument("--num-samples", type=int, default=1024, help="MPPI samples (planner quality)")
    p.add_argument("--frames-max", type=int, default=3600, help="frame cap per lap (60 fps)")
    p.add_argument("--brake-mode", default="none", choices=("none", "channel", "esc"))
    p.add_argument("--track-generator", default="bezier")
    p.add_argument("--tire-model", default="brush", choices=("brush", "pacejka"),
                   help="pacejka records with PacejkaParams defaults (synthetic-recovery mode)")
    p.add_argument("--test", action="store_true", help="tiny planner (32 worlds) for smoke tests")
    a = p.parse_args(argv)

    laps, vehicle = [], None
    for i in range(a.laps):
        lap, veh = record_lap(
            a.track_seed0 + i, a.num_samples, a.frames_max, a.brake_mode, a.track_generator, a.tire_model, a.test
        )
        vehicle = vehicle or veh
        laps.append(lap)
        print(f"lap {i}: seed={lap.meta['track_seed']} frames={lap.meta['frames']} complete={lap.meta['lap_complete']}")
    tcformat.write_session(a.out, tcformat.Session(vehicle=vehicle, laps=laps))
    print(f"wrote {len(laps)} laps to {a.out}")


if __name__ == "__main__":
    main()
```

Implementation notes for the worker:
- `Example.create_parser().parse_args([])` supplies every example default (horizon, rollout-substeps, cost weights); we override only what varies. `args.test=True` shrinks to 32 worlds / horizon 8 for the smoke test.
- Recording reads happen between frames on the host; this is offline tooling, sync cost is fine.
- If `Example.__init__` requires attributes not present on the parsed args in the current tree, set them the same way (`args.<name> = default`) rather than editing the example.
- One lap at 1024 samples takes GPU minutes; the smoke test must stay on `--test` + small `--frames-max`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_generate`
Expected: PASS (CPU-capable; ~1-3 min).

- [ ] **Step 5: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/generate.py tools/tirecal/tests/test_generate.py
git commit -m "Add MPPI lap recorder writing tirecal sessions"
```

---

### Task 9: `newton_eval.py` — batched segment-replay evaluator + brush self-test

**Files:**
- Create: `tools/tirecal/tirecal/newton_eval.py`
- Test: `tools/tirecal/tests/test_newton_eval.py`

**Interfaces:**
- Consumes: Task 4 `format`, Task 8 session files (or in-test recordings), Tasks 1-3 (`TireModel.PACEJKA`, `mf_params`), example's `_build_model`.
- Produces:
  - `tirecal.newton_eval.EvalConfig` dataclass: `window_s=2.0`, `stride_s=1.0`, `max_worlds=4096`, `w_pos=1.0`, `w_yaw=2.0`, `w_vel=0.3`, `w_yawrate=0.1`, `tire_model="pacejka"`
  - `tirecal.newton_eval.ReplayEvaluator(session, cfg)` with:
    - `evaluate(params: np.ndarray (P, 10)) -> np.ndarray (P,)` — mean weighted segment error per candidate (lower better)
    - `mean_position_error(params_row: np.ndarray (10,)) -> float` — mean per-step chassis position error [m] of one candidate over all segments (interpretable metric for thresholds)
    - `num_segments: int`, `window_steps: int`
  - Self-test path: `EvalConfig(tire_model="brush")` makes `evaluate` ignore the candidate rows and run the recorded brush parameters.

**Design notes (read before coding):**
- World topology: reuse `_build_model(num_worlds)` from `newton.examples.vehicles.example_vehicle_mppi_track` (cars collocated at origin, cross-world collision filtered, ground plane mu=1.0). Track geometry is irrelevant to replay — the ground is flat everywhere; controls encode the lap.
- `num_worlds = candidates_per_chunk * num_segments` where `candidates_per_chunk = max(1, cfg.max_worlds // num_segments)`; chunk the population when `P > candidates_per_chunk`, padding the last chunk by repeating row 0 (padded fitness discarded).
- Vehicle config: reconstruct `WheeledConfig(**session.vehicle["config"])` (add `pacejka=PacejkaParams(**session.vehicle["pacejka_defaults"])`), then set `tire_model` per `cfg.tire_model`. Solver/pipeline mirror the example exactly: `SolverMuJoCo(model, use_mujoco_contacts=False, njmax=192, nconmax=48)`, `CollisionPipeline(model, rigid_contact_max=max(1024, 48 * num_worlds))`, and the example's `_substep` sequence verbatim (clear_forces → update_controls → collide → vehicles.apply → solver.step → solver.update_contacts → latch_loads → swap).
- Segments: for each lap with `frames >= window_steps`, window starts at `t0 = 0, stride_steps, 2*stride_steps, ...` while `t0 + window_steps <= frames`. Only laps with `sim_state` participate (raise ValueError otherwise: v1 requires exact resets).
- Per-segment device arrays (built once): reset states at `t0` (`joint_q [S,D]`, `joint_qd [S,Dv]`, `body_q_full [S,B,7]` stored flat as `wp.transform [S*B]`, `body_qd_full` flat as `wp.spatial_vector [S*B]`, `wheel_omega/trans_long/trans_lat/patch_fz [S,W]`), controls `[S, T, 3]` float32 flat, references from samples `t0+1 .. t0+T`: `ref_pos [S*T] wp.vec2` (chassis xy), `ref_yaw [S*T]`, `ref_vel [S*T] wp.vec2` (planar linear velocity — `body_qd` rows 3:5 under Newton's angular-then-linear spatial convention; *verify against `_gather_car_pose` in the example and follow whatever it reads*), `ref_yawrate [S*T]` (`body_qd` row 2).
- Scatter kernels mirror the example's `_broadcast_slice_*` but indexed by `seg_of_world[w]` instead of broadcasting one slice; write per-world slices for joint_q, joint_qd, body_q, body_qd, and per-wheel arrays. Command application kernel indexes `ctrl[seg_of_world[world], t, {0,1,2}]` into `commands.drive/steer/brake[world]`. Error kernel accumulates, per world at step `t`: `w_pos*|p - ref|^2 + w_yaw*wrap(yaw - ref)^2 + w_vel*|v - ref|^2 + w_yawrate*(wz - ref)^2` into `cost[world]`, and `|p - ref|` into `pos_err[world]` (wrap = `atan2(sin, cos)`).
- Per generation: write `mf_params` host-side (`np.repeat(chunk_params, wheels_per_world * num_segments axis-mapping)` — wheel `i` belongs to world `i // wheels_per_world`, world `w` runs candidate `w // num_segments`), scatter resets, zero `cost`/`pos_err`, zero `commands.brake`, then for `t in range(window_steps)`: apply commands, run `sim_substeps` substeps of `frame_dt / sim_substeps`, accumulate error. Fitness per candidate = mean of its segment costs / window_steps.
- Brush self-test path (`tire_model="brush"`): identical machinery, `dynamics.tire_model` stays BRUSH, candidate rows unused.

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_newton_eval.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import newton  # noqa: F401

    _HAVE_SIM = True
except Exception:
    _HAVE_SIM = False


@unittest.skipUnless(_HAVE_SIM, "newton with sim extra required")
class TestReplayEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tirecal import format as tcformat
        from tirecal import generate

        cls._dir = tempfile.TemporaryDirectory()
        out = Path(cls._dir.name) / "session.h5"
        generate.main(
            ["--out", str(out), "--laps", "1", "--track-seed0", "0", "--frames-max", "90", "--test"]
        )
        cls.session = tcformat.read_session(out)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_brush_self_replay_is_faithful(self):
        from tirecal.newton_eval import EvalConfig, ReplayEvaluator

        cfg = EvalConfig(window_s=1.0, stride_s=0.5, max_worlds=64, tire_model="brush")
        ev = ReplayEvaluator(self.session, cfg)
        self.assertGreaterEqual(ev.num_segments, 2)
        # Validation ladder step 2: the generator model replayed against its own
        # recording must be near-exact. 2 cm mean position error over 1 s.
        err_m = ev.mean_position_error(np.zeros(10, dtype=np.float64))
        self.assertLess(err_m, 0.02)

    def test_pacejka_orders_candidates_sensibly(self):
        from tirecal.newton_eval import EvalConfig, ReplayEvaluator
        from tirecal.seed_fit import fit_to_brush

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_newton_eval`
Expected: FAIL — `ModuleNotFoundError: No module named 'tirecal.newton_eval'`.

- [ ] **Step 3: Implement `tools/tirecal/tirecal/newton_eval.py`** per the design notes. Skeleton (the worker fills the kernels/mechanics following the example's `_broadcast_slice_*` and `_substep` patterns exactly; keep every public name as specified):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched segment-replay evaluator: (candidate x segment) -> replicated worlds.

The only module the optimization loop needs newton for. Loads a recorded
session, slices laps into fixed windows, resets each world to a recorded
window start, replays the recorded controls open-loop under candidate Magic
Formula parameters, and accumulates weighted state error against the
recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class EvalConfig:
    """Replay/loss configuration. Weights multiply squared per-step errors."""

    window_s: float = 2.0
    stride_s: float = 1.0
    max_worlds: int = 4096
    w_pos: float = 1.0
    w_yaw: float = 2.0
    w_vel: float = 0.3
    w_yawrate: float = 0.1
    tire_model: str = "pacejka"  # "brush" runs the recorded generator params (self-test)


class ReplayEvaluator:
    def __init__(self, session, cfg: EvalConfig):
        ...

    @property
    def num_segments(self) -> int: ...

    @property
    def window_steps(self) -> int: ...

    def evaluate(self, params: np.ndarray) -> np.ndarray: ...

    def mean_position_error(self, params_row: np.ndarray) -> float: ...
```

Hard requirements for the worker:
- Verify the `body_qd` velocity layout empirically before writing the reference extraction: print one recorded `body_qd` row while the car drives straight and check which components carry the forward speed; the MPPI example's `_gather_car_pose` kernel is ground truth. Do not guess.
- All per-step work must be device-side (kernels), with host sync only at generation boundaries (`cost.numpy()` once per chunk).
- `wheels_per_world` comes from session metadata; assert `dynamics.mf_params.shape == (num_worlds * wheels_per_world, 10)` before assigning.
- Raise `ValueError` with a clear message if no lap has `sim_state` or no lap is long enough for one window.
- Zero `trans_long/trans_lat`? No — scatter the *recorded* values; the recording carries them exactly for this reason.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_newton_eval`
Expected: PASS. The self-test threshold (2 cm) is the deliverable of this task: if replay is not faithful, debug the reset path (compare a replayed trajectory step-by-step against the recording for segment 0 — divergence at step 0 means a bad scatter; growing divergence means missing state, e.g. `patch_fz` or transient slip).

- [ ] **Step 5: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/newton_eval.py tools/tirecal/tests/test_newton_eval.py
git commit -m "Add batched segment-replay evaluator with brush self-test"
```

---

### Task 10: `tune.py` CLI + report

**Files:**
- Create: `tools/tirecal/tirecal/tune.py`
- Test: `tools/tirecal/tests/test_tune.py`

**Interfaces:**
- Consumes: `format`, `ga.StrategyGA`, `ga.StrategyCMAES`, `seed_fit`, `newton_eval.ReplayEvaluator`.
- Produces:
  - CLI: `python -m tirecal.tune --session F.h5 --strategy {ga,cmaes} [--population P] [--generations G] [--window S] [--stride S] [--max-worlds W] [--holdout-laps N] [--seed N] --out report.json`, plus `--self-test` (runs the brush self-replay and prints/records the mean position error, no optimization)
  - `tirecal.tune.run_calibration(session, strategy_name, generations, population, eval_cfg, holdout_laps, seed, evaluator_factory=None) -> dict` — the report dict; `evaluator_factory(session, cfg)` defaults to `ReplayEvaluator` and exists so tests can inject a fake
  - Report JSON keys: `best_params` (name → value, `seed_fit.PARAM_NAMES`), `best_fitness`, `history` (list of per-generation best), `holdout` (`{fitness, mean_position_error_m}` or null when no holdout laps), `seed_fit` (`{params, curve_rms}`), `config` (echo of settings)

- [ ] **Step 1: Write the failing test** — `tools/tirecal/tests/test_tune.py`:

```python
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
        time=np.zeros(2), body_q=np.zeros((2, 7)), body_qd=np.zeros((2, 6)),
        wheel_omega=np.zeros((2, 4)), controls=np.zeros((1, 3)),
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
            _dummy_session(), strategy, generations=40, population=32,
            eval_cfg=EvalConfig(max_worlds=8), holdout_laps=1, seed=0,
            evaluator_factory=_QuadraticEvaluator,
        )

    def test_converges_with_both_strategies(self):
        for strategy in ("ga", "cmaes"):
            report = self._report(strategy)
            self.assertLess(report["best_fitness"], 1.0, msg=strategy)
            self.assertEqual(set(report["best_params"]), set(
                ("bx", "cx", "ex", "by", "cy", "ey", "rbx1", "rbx2", "rby1", "rby2")))
            self.assertEqual(len(report["history"]), 40)
            self.assertIn("mean_position_error_m", report["holdout"])

    def test_history_is_monotone_best(self):
        report = self._report("cmaes")
        h = report["history"]
        self.assertTrue(all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev --extra tirecal python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_tune`
Expected: FAIL — `ModuleNotFoundError: No module named 'tirecal.tune'`. (The test imports `newton_eval.EvalConfig`; that only needs the dataclass, which imports warp — if warp import in a no-sim environment is an issue, move `EvalConfig` usage in the test behind the same `_HAVE_SIM` guard used elsewhere; prefer keeping the fake-evaluator test sim-free by constructing `EvalConfig` only if importable, else a plain namespace.)

- [ ] **Step 3: Implement `tools/tirecal/tirecal/tune.py`:**

```python
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


def run_calibration(session, strategy_name, generations, population, eval_cfg,
                    holdout_laps, seed, evaluator_factory=None):
    if evaluator_factory is None:
        from .newton_eval import ReplayEvaluator

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
            "strategy": strategy_name, "generations": generations, "population": population,
            "window_s": eval_cfg.window_s, "stride_s": eval_cfg.stride_s, "seed": seed,
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
    p.add_argument("--self-test", action="store_true",
                   help="replay the recorded generator model against its own recording and exit")
    a = p.parse_args(argv)

    from .newton_eval import EvalConfig, ReplayEvaluator

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v -k test_tune`
Expected: PASS.

- [ ] **Step 5: Run the full tirecal + newton suites**

```bash
uv run --extra dev --extra tirecal --extra sim python -m unittest discover -s tools/tirecal -t tools/tirecal -v
uv run --extra dev -m newton.tests -k test_vehicles
```
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/tune.py tools/tirecal/tests/test_tune.py
git commit -m "Add tune CLI wiring seed fit, strategies, and evaluator"
```

---

### Task 11: Validation ladder + closed-loop gate + findings

**Files:**
- Create: `tools/tirecal/tirecal/closedloop.py`
- Create: `docs/superpowers/specs/2026-07-17-pacejka-calibration-findings.md` (results)

This task is runs + judgment, executed on GPU. Thresholds come from the spec: held-out mean 2-D position error < 0.2 m and mean heading error < 5 deg over 2 s windows; closed-loop lap time within 3% of brush baseline, no off-track excursions.

- [ ] **Step 1: Implement `closedloop.py`** — final-gate comparison script:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Final validation gate: closed-loop MPPI laps, brush vs tuned Pacejka.

Runs the MPPI track example headless twice on the same track seed -- once on
brush, once on Pacejka with parameters from a tune report -- and compares lap
progress and off-track counts.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def run_laps(tire_model, mf_params, track_seed, frames, num_samples):
    import newton
    from newton.examples.vehicles.example_vehicle_mppi_track import Example

    from .generate import _example_args, _override_tire_model

    viewer = newton.viewer.ViewerNull(num_frames=frames)
    args = _example_args(Example.create_parser(), num_samples, track_seed, "none", "bezier", False)
    ex = Example(viewer, args)
    if tire_model == "pacejka":
        import newton.vehicles as nv

        ex.vehicles.config.pacejka = nv.PacejkaParams(**mf_params)
        _override_tire_model(ex, "pacejka")
    oob_frames = 0
    for _ in range(frames):
        ex.step()
        oob_frames += int(ex.contact.oob.numpy()[0])
    meters = float(ex.total_s.numpy()[0])
    return {"meters": meters, "laps": meters / ex.track_len, "oob_frames": oob_frames}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, help="tune report JSON with best_params")
    p.add_argument("--track-seed", type=int, default=100, help="use a seed outside the training set")
    p.add_argument("--frames", type=int, default=3600)
    p.add_argument("--num-samples", type=int, default=1024)
    a = p.parse_args(argv)
    with open(a.report) as f:
        best = json.load(f)["best_params"]
    brush = run_laps("brush", None, a.track_seed, a.frames, a.num_samples)
    pacejka = run_laps("pacejka", best, a.track_seed, a.frames, a.num_samples)
    ratio = pacejka["meters"] / max(brush["meters"], 1e-9)
    print(f"brush:   {brush}")
    print(f"pacejka: {pacejka}")
    print(f"distance ratio (pacejka/brush): {ratio:.3f}  (gate: within 3% -> >= 0.97)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record the training set** (GPU, ~30-60 min):

```bash
mkdir -p tirecal-data
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.generate \
  --out tirecal-data/brush24.h5 --laps 24 --track-seed0 0
```
Expected: 24 laps, most with `complete=True`. Exclude nothing yet — incomplete laps still contribute segments.

- [ ] **Step 3: Ladder step 2 — self-test:**

```bash
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.tune \
  --session tirecal-data/brush24.h5 --self-test
```
Expected: mean position error < 2 cm. If not, stop and debug replay fidelity before any optimization (this number is also the noise floor that calibrates the 0.2 m acceptance threshold).

- [ ] **Step 4: Ladder step 3 — synthetic recovery, both strategies head-to-head:**

```bash
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.generate \
  --out tirecal-data/pacejka-synth8.h5 --laps 8 --track-seed0 50 --tire-model pacejka
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.tune \
  --session tirecal-data/pacejka-synth8.h5 --strategy cmaes --generations 60 --population 64 \
  --holdout-laps 2 --out tirecal-data/synth-cmaes.json
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.tune \
  --session tirecal-data/pacejka-synth8.h5 --strategy ga --generations 60 --population 64 \
  --holdout-laps 2 --out tirecal-data/synth-ga.json
```
Expected: both recover holdout mean position error near the self-test noise floor; recovered `best_params` close to `PacejkaParams` defaults (θ*) up to identifiability (compare curves via `seed_fit.mf_forces`, not raw parameters). **Record which strategy wins; that becomes the documented default** (update `--strategy` default in `tune.py` if GA wins).

- [ ] **Step 5: Ladder step 4 — real fit:**

```bash
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.tune \
  --session tirecal-data/brush24.h5 --strategy <winner> --generations 80 --population 128 \
  --holdout-laps 4 --out tirecal-data/brush-fit.json
```
Acceptance: holdout `mean_position_error_m` < 0.2 (and inspect heading via a one-off check against the 5-degree bar; add a `mean_heading_error` accessor to the evaluator if needed — small follow-up, not a redesign).

- [ ] **Step 6: Ladder step 5 — closed-loop gate:**

```bash
PYTHONPATH=tools/tirecal uv run --extra tirecal --extra sim python -m tirecal.closedloop \
  --report tirecal-data/brush-fit.json --track-seed 100
```
Acceptance: distance ratio ≥ 0.97 and comparable `oob_frames`.

- [ ] **Step 7: Write findings** to `docs/superpowers/specs/2026-07-17-pacejka-calibration-findings.md`: self-test noise floor, strategy head-to-head result and chosen default, final holdout numbers, closed-loop numbers, the calibrated `PacejkaParams` values, and any threshold revisions (with reasons). If the calibrated parameters are good, update `PacejkaParams` defaults in `controller.py` to the calibrated values (one commit, with the findings doc as justification).

- [ ] **Step 8: Lint and commit**

```bash
uvx pre-commit run -a
git add tools/tirecal/tirecal/closedloop.py docs/superpowers/specs/2026-07-17-pacejka-calibration-findings.md
git commit -m "Add closed-loop validation gate and calibration findings"
```

---

## Plan self-review notes

- Spec coverage: Part 1 → Tasks 1-3; Part 2 → Tasks 4, 8; Part 3 → Tasks 5-7, 9, 10; validation ladder → Task 9 (step 2 as unit test), Task 11 (steps 2-5). Staffing table implements the spec's model-assignment section. CMA-ES addendum → Task 6 + Task 11 step 4.
- Deviation from spec (already synced into the spec): `seed_fit.py` is standalone (numpy brush reference + CMA-ES) instead of using `_eval_tire_kernel`.
- Format detail beyond spec: the `sim_state/` optional group and the T/T+1 controls alignment — required for exact segment resets; documented in SCHEMA.md and consistent with the spec's "optional groups never required by readers" rule.
- Type consistency spot-checks: `tire_force(..., mf, w)` arity matches between tire.py, wheel.py, `_eval_tire_kernel`, and tests; `PacejkaParams` field order == `MF_*` == `seed_fit.PARAM_NAMES`; `Session.vehicle["tire_params"]` keys consumed by `tune.py` match what `generate.py` writes; `EvalConfig`/`ReplayEvaluator` names match between Tasks 9 and 10.
