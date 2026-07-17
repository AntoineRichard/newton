# Pacejka Tire Model + GA Calibration Tool — Design

Date: 2026-07-17
Status: Approved (design review complete)

## Goal

Add a Pacejka (Magic Formula) tire model to Newton's vehicle stack, and build a
calibration tool that tunes its parameters so the vehicle behaves like the
existing brush model. Calibration is trajectory-based: record MPPI laps driven
on the brush model, then use Newton's world-level parallelism to evaluate
thousands of candidate parameter sets simultaneously inside a genetic
algorithm. The recording format and optimizer are designed so the same pipeline
later serves real2sim calibration against real-vehicle logs.

## Decisions made during design review

- **MF variant:** core Magic Formula per axis + Pacejka cosine combined-slip
  weighting functions (not minimal friction-ellipse, not load-dependent
  PAC2002).
- **Tires are identical:** one shared parameter set for all four wheels; the GA
  searches one vector, not per-wheel sets.
- **Replay mode:** segmented replay (short windows reset to recorded state),
  not full-lap open-loop, not per-step teacher forcing.
- **Container:** HDF5 (h5py as an optional dependency of the tooling only).
- **Layout:** standalone `tools/tirecal/` package. The reusable core
  (`format.py`, `ga.py`) never imports `newton`; only the sim-facing modules
  (`generate.py`, `seed_fit.py`, `newton_eval.py`) do.
- **Optimizer:** pluggable ask/tell strategy interface with two v1
  implementations — GA and a plain-numpy CMA-ES — both seeded by a
  least-squares curve fit; the default strategy is chosen empirically in the
  synthetic-recovery validation step.
- **Acceptance:** held-out segment error is the working metric; a closed-loop
  MPPI lap on the tuned parameters is a one-time final validation gate only.

## Part 1 — Pacejka model in Newton

### Location and selection

- New branch `TIRE_PACEJKA = wp.constant(2)` in
  `newton/_src/vehicles/tire.py`, selected per wheel by the existing integer
  model id.
- Public API: `TireModel.PACEJKA = 2` (in the enum currently defined in
  `newton/_src/vehicles/controller.py`, exposed via `newton.vehicles`).

### Math

Slip inputs are identical to the brush branch: theoretical slip
`sx = kappa / (1 + kappa)`, `sy = tan(alpha) / (1 + kappa)`, with the same
lock-up guard (`kappa >= -0.9999`). Both models therefore see the same slip
quantities and calibration compares like for like.

Pure slip, per axis (`s` = `sx` or `sy`):

```
F0(s) = D * sin(C * atan(B*s - E*(B*s - atan(B*s))))     with D = mu * Fz
```

Combined slip via cosine weighting functions:

```
Fx = Gxa(alpha_s) * Fx0(sx)      Gxa = cos(Cxa * atan(Bxa * sy))
Fy = Gyk(kappa_s) * Fy0(sy)      Gyk = cos(Cyk * atan(Byk * sx))
Bxa = rbx1 * cos(atan(rbx2 * sx))
Byk = rby1 * cos(atan(rby2 * sy))
```

(`Cxa`, `Cyk` are folded into the r-parameters for v1; if more shape freedom is
needed they become explicit parameters — the storage scheme below makes that a
pure append.)

Aligning moment reuses the existing collapsing-trail form shared by the brush
and linear branches: `Mz = -Fy * trail * max(1 - util, 0)` with
`util = |F| / (mu * Fz)`. Same `pneumatic_trail` parameter, no new Mz params.

`mu` remains a separate input so `mu_override` and per-contact friction seeding
keep working; all MF shape parameters are dimensionless. Sign conventions match
the brush branch exactly (`Fx` positive driving forward, `Fy` opposing lateral
slip). `Fz <= 0` or `mu <= 0` returns zero, as today.

### Parameter storage (extensibility requirement)

- `WheelDynamics` gains one 2-D array `mf_params: wp.array2d[float32]` of
  shape `(wheel_count, MF_PARAM_COUNT)` instead of scalar fields.
- Named index constants in `tire.py`: `MF_BX, MF_CX, MF_EX, MF_BY, MF_CY,
  MF_EY, MF_RBX1, MF_RBX2, MF_RBY1, MF_RBY2` (`MF_PARAM_COUNT = 10` in v1).
  Enriching the parameter set later means appending indices; the GA already
  treats the row as a flat vector.
- `WheeledConfig` gains a nested `PacejkaParams` dataclass (prefix-first
  naming, nested-class convention) with defaults that roughly match the brush
  shape; the controller writes it into the arrays like every other parameter.
- The Python-side name→index mapping is exported so tooling can address
  parameters by name.

### Implicit-solve integration

`wheel.py` linearizes the tire for the implicit impulse solve using
`c_long * Fz` / `c_lat * Fz` slopes — a brush/linear assumption. Add a
companion Warp func `tire_slip_stiffness(model_id, ...)` in `tire.py`
returning the per-model zero-slip slopes (Pacejka: `B * C * D`), and use it in
the wheel kernel in place of the hard-coded products. Brush/linear behavior is
unchanged (their slopes are exactly `c * Fz`).

### Tests (unittest, extending `test_vehicles_tire.py`)

- Zero-slip slope equals `B*C*D` per axis (finite difference vs analytic).
- Peak force ≤ `mu*Fz`; `sin(C·atan(...))` shape produces peak-then-falloff
  for `C > 1`.
- Combined slip: adding lateral slip reduces `|Fx|` and vice versa;
  `G ∈ (0, 1]`.
- Degenerate inputs: `Fz <= 0`, `mu <= 0`, lock-up guard.
- `tire_slip_stiffness` matches brush/linear legacy slopes bit-for-bit.
- Existing brush/linear tests unchanged (regression).

### Housekeeping

- `docs/generate_api.py` run for new public symbols; CHANGELOG entry under
  `Added` ("Add Pacejka (Magic Formula) tire model `TireModel.PACEJKA` ...").

## Part 2 — Trajectory format and recorder

### Format module

`tools/tirecal/tirecal/format.py` — reader/writer importing only numpy and
h5py. **This module must not import newton.** It is the contract for both sim
recordings and future real-vehicle logs. `tools/tirecal/SCHEMA.md` documents
fields, SI units, and frame conventions.

One HDF5 file per recording session, schema v1:

```
/                 attrs: schema_version=1, source ("newton-sim" | "real"),
                  created (ISO timestamp), notes
/vehicle          attrs: JSON blob — mass, wheelbase, track width, wheel
                  radius, wheel positions, generator tire model name and
                  named parameter dict
/laps/NNN         attrs: track seed, sim dt, substeps per control step,
                  control rate, lap time
    time          [T]    control-step timestamps [s]
    body_q        [T,7]  chassis position + quaternion, world frame
    body_qd       [T,6]  chassis angular + linear velocity, world frame
                         (Newton spatial convention)
    wheel_omega   [T,W]  analytical wheel spin [rad/s]
    controls      [T,A]  executed command vector; column names stored as an
                         attribute (name -> index map)
    wheels/       OPTIONAL per-wheel diagnostics: kappa, alpha, f_long,
                  f_lat, fz, contact flag — present in sim recordings,
                  never required by the reader/evaluator
```

Design rules:

- **Named, not positional:** tire parameters and control columns are stored as
  name-keyed maps so richer parameter sets or a real car's different channel
  list never break the reader.
- **Recorded at control rate**, with `sim dt` and substep count in attrs so
  the evaluator reproduces the exact integration scheme. Segment resets occur
  at control-step boundaries, where exact states exist.
- Schema versioned; readers reject files with a newer major version.

### Recorder / generator

`tools/tirecal/tirecal/generate.py` — CLI, the generation-side script that
touches Newton. It drives the existing MPPI track example
(`newton/examples/vehicles/example_vehicle_mppi_track.py`, hero world only)
across 20–40 track seeds on the brush model and writes laps through the format
writer. The example itself gains only a minimal hook (a per-control-step
callback exposing hero state + executed command), not recording logic.

## Part 3 — Calibration tool (`tools/tirecal/`)

### Package layout

```
tools/tirecal/
  SCHEMA.md            format spec (fields, units, frames)
  tirecal/
    format.py          standalone: numpy + h5py only
    ga.py              standalone: numpy only — ask/tell strategies (GA, CMA-ES)
    seed_fit.py        least-squares MF-to-brush curve fit (uses newton's
                       _eval_tire_kernel; newton-importing, used once at init)
    newton_eval.py     batched evaluator (newton-importing); the only newton
                       dependency of the optimization loop itself
    generate.py        CLI: record laps (newton-importing)
    tune.py            CLI: run calibration (loads laps, seeds, runs GA
                       against the evaluator, writes result params + report)
  tests/               unittest
```

Registered as a `tirecal` optional-dependency extra (h5py) in
`pyproject.toml`. No new required dependencies anywhere.

### Optimizer interface (`ga.py`)

Ask/tell strategy protocol:

- `ask() -> np.ndarray (P, n_params)` — candidate matrix
- `tell(fitness: np.ndarray (P,))`
- bounds per parameter, provided by the caller (name-keyed, converted to the
  flat vector layout).

Two v1 strategies behind this protocol, both pure numpy:

- **GA:** tournament selection, SBX crossover, polynomial mutation, elitism.
- **CMA-ES:** own implementation (~150 lines, no `pycma` dependency), rank-mu
  covariance update, initialized from the curve-fit seed as the distribution
  mean. Well suited to the ~10-dimensional continuous search; its covariance
  update is the principled version of the large-population pseudo-gradient
  idea.

`tune.py` selects via `--strategy {ga,cmaes}`. The synthetic-recovery
validation step runs both head-to-head and fixes the default. Further
strategies can be added later without touching the evaluator. The optimizer
module knows nothing about tires or Newton — it is tested on standard
benchmark functions.

### Batched evaluator (`newton_eval.py`)

Black-box fitness: `(P, n_params)` matrix in, `(P,)` fitness out.

- Loads laps via `format.py`; slices each into segments of ~1–3 s windows
  (window length and stride are config; default 50% overlap).
- Builds one replicated-world model exactly like the MPPI example: identical
  cars collocated at the origin, cross-world collision filtered,
  `num_worlds = min(P * S, max_worlds)`; generations are chunked when
  `P * S` exceeds the world budget (`max_worlds` config, default a few
  thousand, sized to GPU memory).
- Per generation: write each candidate's MF vector into the per-wheel
  `mf_params` rows of its worlds (all four wheels share the vector), hard-reset
  each world to its segment's recorded start state (`body_q`, `body_qd`,
  `wheel_omega`; transient-slip states zeroed), replay recorded controls
  open-loop for the window, accumulate per-step weighted error on device:
  2-D position, heading, planar velocity, yaw rate (weights are config).
- Candidate fitness = mean over its segments. Train/held-out split is by lap.

### Seeding (`seed_fit.py`)

Least-squares fit of the MF pure-slip curves to the brush curves over a
(kappa, alpha, Fz) grid evaluated with the existing `_eval_tire_kernel`.
The result initializes the GA population center so the search budget is spent
on what curve matching cannot see: combined-slip transients through the
implicit impulse solver.

### Validation ladder

1. **Unit:** format round-trip; GA and CMA-ES each solve Rastrigin/sphere
   within budget.
2. **Self-test:** evaluate the brush model against its own recording —
   fitness ≈ 0. Proves reset/replay fidelity independent of any tire change.
3. **Synthetic recovery:** record laps with known Pacejka parameters θ*;
   run GA and CMA-ES head-to-head — each must recover θ* (within tolerance).
   Proves end-to-end signal and fixes the default strategy.
4. **Real run:** fit to brush recordings; accept when held-out-lap segment
   error is below the initial thresholds: mean 2-D position error < 0.2 m and
   mean heading error < 5 deg over 2 s windows (thresholds are config and may
   be revised once step 3 establishes the noise floor).
5. **Final gate (one-time):** MPPI closed-loop lap on tuned Pacejka; lap time
   within 3% of the brush baseline and no off-track excursions.

## Implementation staffing

Two parallel workstreams executed by subagents:

- **Workstream A — Pacejka model in Newton** (tire.py branch, param storage,
  slip-stiffness func, config plumbing, tests, docs).
- **Workstream B — tirecal package** (format + SCHEMA, GA, recorder hook +
  generate CLI, evaluator, tune CLI, tests).

Model assignment by task complexity: Sonnet for mechanical/well-specified
tasks (format module, GA and CMA-ES strategies on benchmarks, docs
plumbing), Opus for the tire
model math and recorder integration, Opus or Fable for the evaluator
reset/replay correctness (the hardest piece). Fable reviews all work.

## Out of scope (explicitly)

- Load-dependent MF parameters (PAC2002-style) — the append-only parameter
  layout leaves room.
- Per-wheel parameter sets — tires are physically identical.
- Real2sim calibration itself — this project only guarantees the format and
  evaluator interfaces don't preclude it.
- MF aligning-moment formula — existing collapsing-trail Mz is retained.
