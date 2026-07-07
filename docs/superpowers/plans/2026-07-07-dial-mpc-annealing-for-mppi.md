# DIAL-MPC-Style Annealing for the MPPI Controller — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate, and where it pays off, adopt the sampling improvements from DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC) into `newton.vehicles.ControllerMPPI`, so the wheeled-racing planner explores better and jitters less on the existing `example_vehicle_mppi_track` benchmark. Each ingredient runs an explicit equal-compute ablation against today's single-pass MPPI as evidence to report (see Decisions: not an auto-reject gate — the deployment path is a lighter rollout model); ingredients that win on neither the equal-compute nor the above-budget arm are recorded as rejected, not merged.

**Architecture:** `ControllerMPPI` today is a single-pass MPPI: one `sample()` (nominal + AR(1)-smoothed Gaussian noise, fixed per-channel `sigma`), the caller rolls out `K` Newton worlds over the horizon and accumulates costs, then one softmax-weighted `update()` and a `shift()`. DIAL-MPC keeps that MPPI update as its inner step but wraps it in an outer "diffusion" loop that re-samples/re-rolls the horizon `N_diffuse` times with an annealed covariance, uses a horizon-dependent noise schedule (more noise far in the horizon, less near the executed step), and parameterizes the control as a coarse spline. We port the cheap, structurally-free ingredients first (horizon-annealed noise), then the medium one (spline knots), and treat the expensive outer loop as an experiment gated hard on compute-matched wins. See the source: `newton/_src/vehicles/mppi.py` and the example `newton/examples/vehicles/example_vehicle_mppi_track.py`.

**Tech Stack:** Python, NVIDIA Warp kernels, Newton `SolverMuJoCo` rollouts, `newton.vehicles.WheeledVehicles`, `track_gen`, `unittest`.

## Global Constraints

- Tests use `unittest`, never pytest. Run with `uv run --extra dev -m newton.tests -k <pattern>`.
- `newton/_src/` is internal; the example imports only `newton`, `newton.examples`, `newton.vehicles`.
- Never call `wp.synchronize()` before `.numpy()` on a Warp array.
- All new planner methods must remain pure `wp.launch` (CUDA-graph capturable): no host sync, no Python-side per-sample loops that break capture. Anything tunable at runtime lives in `[·]` device arrays (mirror `set_temperature`/`set_beta`).
- PEP 604 unions (`x | None`); Warp array annotations use bracket syntax (`wp.array[wp.float32]`, `wp.array3d[float]`).
- Google-style docstrings; SI units where physical; Sphinx cross-refs without `newton._src`.
- New files: SPDX header year 2026 (`Copyright (c) 2026 The Newton Developers`, `Apache-2.0`). Never change the year on existing files.
- **Do not break the public API.** `ControllerMPPI.Config` fields and the `sample()`/`update()/shift()` cycle are shipped. New behavior must be **opt-in and default-off** (new Config fields whose defaults reproduce today's single-pass behavior bit-for-bit), or it is a breaking change requiring deprecation first.
- Commit style: imperative subject ≤ ~50 chars, body wraps at 72 explaining what and why, no `feat:` prefixes. End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Work stays on branch `antoiner/wheeled-vehicle-design`.
- Before each commit: `uvx pre-commit run -a`. Run `uv run docs/generate_api.py` when public API symbols change.
- CHANGELOG entry (random position in the right `[Unreleased]` category) only if user-facing behavior changes.

---

## Background

### Our MPPI as it stands (Phase 1)

`newton/_src/vehicles/mppi.py` — `ControllerMPPI`:

- **Control parameterization:** per-step. `nominal` is `[H, A]` (default `H = 32`, `A = 2`: drive, steer). No spline; every horizon step is an independent decision variable.
- **Sampling distribution / schedule (`_sample_sequences`):** sample 0 is the zero-noise nominal; samples `1..K-1` are `nominal + n_t`, where `n_t` is AR(1)-smoothed colored noise `n_t = b·n_{t-1} + sqrt(1 - b²)·(sigma_a · N(0,1))`, per-channel `beta` (drive 0.85, steer 0.6 in the example) and per-channel `sigma` (0.35, 0.45). **`sigma` is constant across the horizon** — there is no horizon-dependent or iteration-dependent noise schedule. Noise is clamped to per-channel bounds and the post-clamp delta is stored so the update respects bounds. An RNG device counter advances per `sample()` so streams never replay.
- **Samples / horizon:** `K = num_samples` (example uses `num_worlds`, default 1024; test 32), `H = 32` (test 8).
- **Temperature / weighting (`update`):** softmax over `exp(-(cost - min_cost)/temperature)`; example `temperature = 15.0` (tuned so ESS lands in ~5–20% of `K`; at 0.05 it was argmin-degenerate). `ess` is exposed as a diagnostic. Reductions are deterministic single-thread O(K) loops (no atomics) so results are bit-stable under CUDA-graph replay.
- **Warm-starting:** `shift()` rolls the nominal forward one step, repeating the last row.
- **Rollout mechanism:** the simulator *is* the model. `K` rc_car worlds are collocated at the origin (cross-world collision filtered), broadcast from a hero snapshot each frame, then rolled out. **One plan update = `H · rollout_substeps` solver substeps across all `K` worlds** (default `32 · 4 = 128` substeps × 1024 worlds), all inside a single CUDA graph, plus `sim_substeps = 8` for the real executed frame. Rollouts are the dominant cost by far.
- **Costs (`_accumulate_cost`):** centerline arc-length progress; one-sided over-reference-speed penalty (speed profile precomputed by curvature + accel/brake passes); terminal over-speed; steer magnitude; command-rate smoothness; time-decayed wall-kill; graded wall proximity; light clearance reward; recovery mode for rollouts that start OOB. A strong `EXEC_COUPLING = 25×` rate term ties the plan's first command to the previously executed command.
- **Known weaknesses (from code + spec comments):**
  - Straight-line steering **jitter/wobble** — the dominant failure mode; mitigated with speed-scaled steering, the exec-coupling term, and a clearance reward that breaks cost ties between centered-straight and swerving samples. Still a smoothness prior bolted on via costs, not the sampler.
  - **Softmax collapse** at low temperature (ESS → 1); needs manual temperature tuning per track/speed.
  - Noise **accumulates freely in the saturated steering region** at speed (hence the speed-scaled command).
  - Fixed horizon-wide `sigma` cannot spend exploration where it helps (far horizon) while staying calm where it executes (t = 0).
  - Classic sampling-MPC risks not directly addressed: **local minima around racing-line/obstacle-side choices** and sample inefficiency at long horizons.

### DIAL-MPC summary (Phase 2)

Xue, Pan, Yi, Qu, Shi, *Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing*, arXiv:2409.15610 (2024); project `lecar-lab.github.io/dial-mpc`, code `github.com/LeCAR-Lab/dial-mpc`. DIAL-MPC keeps the MPPI softmax update (`update_method: mppi`) as an inner step and adds:

1. **Dual-loop diffusion-style annealing.** An outer loop of `N_diffuse` refinement iterations per control step re-samples, re-rolls, and re-updates the control sequence with a **shrinking** noise level — analogous to denoising in diffusion. It couples this with an **inner horizon schedule**: noise grows for horizon steps farther from the current one. Combined isotropic covariance (Eq. 7):
   `Σ^i_{t+h} = exp( -(N-i)/(β₁·N) − (H-h)/(β₂·H) ) · I`
   where `i` indexes the outer diffusion iteration (noise anneals down as `i → N`) and `h` indexes the horizon offset (noise grows with `h`). `β₁`, `β₂` are annealing-rate ("temperature") parameters.
2. **Spline control parameterization.** The control is represented by `Hnode` coarse knots (default 5) interpolated up to `Hsample` fine steps (default 25) — lower effective dimensionality and built-in smoothness.
3. **Training-free, torque-level.** Reference config: `Nsample = 2048`, `Hsample = 25`, `Hnode = 5`, `Ndiffuse = 4` (`Ndiffuse_init = 10` on the first step), `temp_sample = 0.05`, `horizon_diffuse_factor = 1.0`, `traj_diffuse_factor = 0.5`, 50 Hz on an RTX 4090.
4. **Reduces to MPPI when `N_diffuse = 1`** with a flat schedule — the outer loop and horizon annealing vanish, leaving one Gaussian perturbation + softmax update.

Reported: ~13.4× lower tracking error vs. MPPI and +50% over an RL baseline on climbing, real quadruped jumping — **in contact-rich, ~12–19-DoF torque-level locomotion.** The paper reports no isolating ablation (annealing vs. simply more total samples) and does not quantify the compute multiplier; `N_diffuse` iterations multiply rollout cost roughly `N_diffuse×`.

---

## Assessment (Phase 3 — adversarial)

### Ingredient mapping onto our controller

| DIAL-MPC ingredient | Status in `ControllerMPPI` | Fit for 3-input wheeled racing |
| --- | --- | --- |
| Inner MPPI softmax update | **Present** (identical form) | — |
| Warm-start / receding-horizon shift | **Present** (`shift()`) | — |
| Colored/smoothed noise | **Present** (AR(1) `beta`), stronger than DIAL's white-per-knot noise | — |
| **Horizon-dependent noise schedule** (more noise far, less near t=0) | **Missing** — `sigma` is horizon-flat | **Good fit, ~free.** We execute only `nominal[0]`; calming near-term noise directly attacks the jitter weakness while keeping far-horizon exploration. Single pass, ~zero added compute. |
| **Spline control parameterization** (`Hnode` knots) | **Missing** — per-step | **Plausible.** A cleaner smoothness prior than AR(1)+rate-cost+exec-coupling; shrinks the effective decision dimension (already tiny: 2–3). Modest upside, modest cost, some retuning. |
| **Outer diffusion loop** (`N_diffuse` anneal iterations) | **Missing** | **Weak fit / expensive.** Their gains come from a highly non-convex, ~12–19-DoF contact landscape. Ours is 2–3 inputs where 1024 samples already densely cover the action space each frame. The outer loop multiplies the *dominant* cost (`H·rollout_substeps·K` Newton substeps) by `N_diffuse`. |

### Honest cost/benefit

- **Rollouts are the budget.** One plan update is already `~128` solver substeps × 1024 worlds in a CUDA graph. `N_diffuse = 4` makes that `~4×`. There is no free lunch: the fair question is not "does annealing beat vanilla MPPI?" but "does `N_diffuse` iterations of `K/N_diffuse` samples beat one pass of `K` samples **at equal total rollouts**?" DIAL's own paper does not answer this.
- **Low-dimensional action space undercuts the headline motivation.** Diffusion-style annealing exists to escape local minima and tame variance in high-dimensional non-convex sampling. With `A = 2–3` and dense sampling, MPPI's landscape is far tamer; the marginal value of iterative denoising is expected to be small. Where local minima *do* bite us (inside vs. outside line through a corner; which side of an obstacle), a few outer iterations with annealed noise *could* help — but that is exactly what the equal-compute ablation must prove.
- **The near-free ingredient likely captures most of the realizable benefit.** The horizon-annealed schedule targets our actual documented failure mode (near-term jitter) at ~zero compute and no CUDA-graph disruption. Spline is a bounded experiment. The outer loop is the costliest experiment but gets a fair trial (Decision 1): rollout cost on today's Newton backend is not the deployment constraint.

### The equal-compute comparison (applies to every step)

Benchmark = `example_vehicle_mppi_track` at a **fixed total rollout budget** `B = N_diffuse · K · H · rollout_substeps`, fixed frame count. Track protocol (per Decision 4): tune on the ONE user-picked track from the candidate grid, then validate every keep/reject decision on 4 other tracks. Report per track:
- **Lap time / distance-per-N-frames** (primary racing metric; hero lap odometer `total_s`).
- **Mean tracking cost** and **best/mean sample cost**.
- **Control smoothness**: RMS of executed `Δdrive`, `Δsteer` between frames (jitter proxy) and steering reversal count.
- **Robustness**: fraction of seeds finishing without a wall-kill of the hero.
- **Compute**: wall-clock per `step()` and effective steps/s (must stay real-time-capable, i.e. ≥ 60 Hz-equivalent headroom that today's config has).

Keep/reject rule (per Decisions 1–2): smoothness ranks **at least equal** to lap distance. A step is **kept** if, at equal `B`, it improves smoothness without a meaningful lap-distance or robustness loss (smoother-but-marginally-shorter = keep), or improves lap distance without regressing smoothness/robustness. A smoothness regression = reject. The equal-compute comparison is **evidence to report**, not an auto-reject gate — for Task 4 in particular, an above-budget win is still interesting given the lighter-rollout deployment path. Rejected steps are recorded (mirror the existing "rejected findings" commits on this branch).

---

## Plan

### Task 1: Benchmark harness + baseline numbers (no behavior change)

Establishes the equal-compute gate before touching the sampler, so every later step has a fixed yardstick.

**Files:**
- Create: `newton/_src/vehicles/_mppi_bench.py` *(internal helper; not public API)* or a scratch script under the session scratchpad — a headless driver that runs `example_vehicle_mppi_track` over a seed sweep and emits the metrics above as JSON.
- Reference only: `newton/examples/vehicles/example_vehicle_mppi_track.py` (metrics already exposed: `total_s`, `costs`, `ess`, executed `nominal[0]`).

**Interfaces:** produces a `bench(tracks, frames, config_overrides) -> dict` returning per-track and aggregate lap distance, mean/best/mean cost, executed-command RMS jitter, steering-reversal count, hero-kill fraction, and steps/s.

- [x] **Step 1:** Add executed-command logging to the harness (differences of `planner.nominal[0]` across frames) and a wall-clock timer around `step()`. Do **not** modify the example's control logic. *(Done: `newton/_src/vehicles/_mppi_bench.py`; logs executed `u_prev` and times `step()`. The only example change is a track-generation `--track-param` pass-through, not control logic.)*
- [x] **Step 2 (track protocol, Decision 4):** baseline on the tuning track — **picked: `hull s4 displacement=0.35`** (tight hairpin cluster with long straights) — plus the 4 validation tracks, default `--num-samples`/`--horizon`. Record baseline JSON; commit the numbers as the reference in the plan's results section (below) or a committed `docs/superpowers/notes/` file. *(Done: numbers in `docs/superpowers/notes/2026-07-07-mppi-task1-baseline-and-braking.md`, JSON committed alongside; all 5 tracks reproduce at attempt 0.)*
- [x] **Step 2b (under-braking diagnosis, user request 2026-07-07):** on the tuning track, diagnose why the car does not brake harder into hairpins. Suspects, in order: (a) action-space encoding — does the planner sample a real brake channel, or does negative drive merely cut torque (coasting) rather than engage `brake_target`? (b) AR(1)/exec-coupling smoothing damping fast brake onset; (c) per-channel sampling σ too small on the brake axis to ever explore threshold braking; (d) horizon too short for braking's corner-exit payoff to beat straight-line progress cost; (e) cost terms (speed/progress reward shaping) actively penalizing deceleration. Instrument executed brake commands vs. distance-to-corner; report findings before any annealing work, since a broken brake channel would confound every experiment in Tasks 2-4. *(Done: **suspect (a) confirmed** — no brake channel sampled; negative drive engages a 1 N·m motor servo that brakes no harder than coasting (3.6 vs 3.9 m/s²), while the unreachable 20 N·m friction brake gives 18.7 m/s² (5.2×). Fix behind default-off `--brake-mode {none, channel, esc}`. Per the user's regen-braking correction, `brake_target` models the ESC's regen/drag brake (there is no friction brake on the real car; `brake_max_torque` = ESC brake-current limit, to be calibrated against the real VXL); the recommended `esc` mode is transmitter-faithful — negative drive engages the ESC brake, keeping the 2-D action space — and brakes 3× harder into the measured hairpin (21.6 vs 7.2 m/s² entry decel) at equal lap pace to the 3-channel variant. Default stays `none`. See notes file.)*
- [x] **Acceptance:** reproducible baseline numbers on all 5 tracks; harness runs headless (`--viewer null`) and CUDA-graph path intact.

**Commit:** `Add MPPI racing benchmark harness for annealing experiments`

---

### Task 2: Horizon-annealed noise schedule (single pass, opt-in, default-off)

The cheap win: make `sigma` depend on the horizon step `t` — less noise near `t = 0` (executed), more far out — without adding any rollout passes.

**Files:**
- Modify: `newton/_src/vehicles/mppi.py` (`_sample_sequences`, `Config`, `__init__`, a `set_*` for runtime tuning).
- Modify: `newton/tests/test_vehicles_mppi.py`.

**Design:**
- New knob `sigma_horizon_factor: float = 1.0` (1.0 ⇒ flat ⇒ **bit-identical to today**). Per Decision 3 it stays **example-level/private** (internal `_src` field or `set_*` method, not a public `Config` field) until proven; promotion to the public `Config` happens only on keep. Effective per-step std: `sigma_t = sigma_a · f(t)` with a monotone schedule, e.g. `f(t) = exp((H-1-t)/(H) · ln(sigma_horizon_factor))` so the near-term step is calmest and the far step scales by `sigma_horizon_factor`. Store the precomputed `[H, A]` (or `[H]`) schedule in a device array so it stays graph-capturable and runtime-tunable.
- The AR(1) `beta` smoothing stays; the schedule multiplies `eps` per step.

- [ ] **Step 1 (TDD):** Add failing tests — with `sigma_horizon_factor = 1.0`, samples are unchanged vs. the current implementation (regression lock, seed-matched); with `factor > 1.0`, per-step sample variance increases monotonically with `t` (measure across many samples); bounds still respected; sample 0 still the nominal.
- [ ] **Step 2:** Implement the schedule (private surface); verify the `factor = 1.0` regression test passes bit-for-bit.
- [ ] **Step 3:** Sweep the factor on the tuning track (e.g. {1.0, 1.5, 2.0, 3.0}) at fixed `B`; pick the best; validate on the 4 validation tracks.
- [ ] **Acceptance (per Decision 2):** at equal compute, the best factor reduces executed-command RMS jitter and/or steering-reversal count without regressing hero-kill fraction; a small lap-distance cost is acceptable if smoothness clearly improves. If no factor helps, record rejected and leave the knob defaulting to 1.0 (no-op) or revert. Only on keep: promote to public `Config`, update `docs/generate_api.py` output, CHANGELOG under `Added`.

**Commit (if kept):** `Add horizon-annealed noise schedule to ControllerMPPI`

---

### Task 3: Spline (knot) control parameterization (opt-in, default-off)

Represent the plan by `n_knots` coarse control knots interpolated to the `H` fine steps, à la DIAL's `Hnode`.

**Files:**
- Modify: `newton/_src/vehicles/mppi.py` (nominal/knots representation, sample/update/shift over knots, interpolation kernel).
- Modify: `newton/tests/test_vehicles_mppi.py`.

**Design:**
- New knob `n_knots: int | None = None` (`None` ⇒ per-step, **today's behavior**); per Decision 3 it stays example-level/private until proven, promoted to the public `Config` only on keep. When set, the planner's decision variable is `[n_knots, A]`; `samples`/`noise` are sampled at knots, then a linear (or Catmull-Rom) interpolation kernel expands to `[K, H, A]` for the caller's rollout. `update()` averages noise at knot resolution; `shift()` shifts knots.
- Keep the interpolation kernel graph-capturable; expose the expanded `[K, H, A]` `samples` unchanged so the example's rollout loop is untouched.
- Interaction: with knots, the AR(1) `beta` and the Task-2 horizon schedule act at knot resolution (document this).

- [ ] **Step 1 (TDD):** Failing tests — `n_knots = None` reproduces current shapes/behavior; with knots, `samples` has shape `[K, H, A]`, sample 0 equals the interpolated nominal, interpolation is exact at knot locations, bounds respected after interpolation+clamp.
- [ ] **Step 2:** Implement knots + interpolation (private surface); verify `None` path is bit-identical.
- [ ] **Step 3:** Benchmark `n_knots ∈ {4, 6, 8, H}` at fixed `B` on the tuning track; compare against the Task-2 winner; validate on the 4 validation tracks.
- [ ] **Step 4 (removal ablation, Decision 5):** with the best `n_knots`, empirically test REMOVING the existing smoothness machinery — AR(1) `beta` (set to 0), the `w_rate` cost, and `EXEC_COUPLING` — individually and together. If splines subsume some of it, prefer the cleaner config (remove); if removal regresses, keep stacking and record why.
- [ ] **Acceptance (per Decision 2):** at equal compute, some `n_knots` beats the Task-2 config on smoothness without regressing robustness; a small lap-distance cost is acceptable. Record the Step-4 removal-ablation outcome either way. If knots do not beat Task 2, record rejected; keep `n_knots` defaulting to `None`.

**Commit (if kept):** `Add spline-knot control parameterization to ControllerMPPI`

---

### Task 4: Outer diffusion-annealing loop (`N_diffuse`) — fair trial

The expensive ingredient, given a **fair trial** (Decision 1): GPU budget is flexible for experiments, and the deployment path is a lighter rollout model (kinematic/dynamic bicycle model or a small learned dynamics net), so `N_diffuse > 1` is not ruled out by today's Newton-rollout cost. Wrap the sample→rollout→update cycle in an outer loop with per-iteration annealed `sigma`. Run the equal-compute ablation as evidence to report alongside an above-budget run — not as an auto-reject gate. Follow-up note: a bicycle-model rollout backend is plausible future work that would enable much larger annealing budgets.

**Files:**
- Modify: `newton/_src/vehicles/mppi.py` — add per-iteration noise-scale control (a `set_diffuse_scale(i)` writing a device scalar consumed by `_sample_sequences`). The **loop itself lives in the example/harness**, not the planner, so the planner stays a primitive and CUDA-graph capture of the multi-iteration cycle is explicit.
- Modify: `newton/examples/vehicles/example_vehicle_mppi_track.py` — optional `--n-diffuse` (default 1 ⇒ today). For `i` in `1..N_diffuse`: set the annealed scale (`exp(-(N-i)/(β₁·N))`), `sample()`, rollout, `update()` (no `shift()` between inner iterations; the nominal is refined in place). One `shift()` at the end.
- Modify: tests as needed for the new `set_*` and the `N_diffuse = 1` no-op.

**Design / correctness notes:**
- Between inner iterations, re-broadcast the hero snapshot and re-zero cost/dead buffers (the rollout mutates world state) — the example already snapshots/restores; the outer loop must restore before each re-rollout.
- CUDA graph: the whole `N_diffuse`-iteration cycle must be captured once; `N_diffuse` is fixed at capture time (a Python-level `range`, unrolled into the graph). The annealing scale is a device array so it stays tunable.
- Equal-compute: compare `(N_diffuse = D, K = K0/D)` vs. `(N_diffuse = 1, K = K0)` for `D ∈ {2, 4}` and, separately, the Task-2 horizon schedule inside each inner iteration (DIAL uses both).

- [ ] **Step 1 (TDD):** Failing tests — `set_diffuse_scale(1.0)` + `N_diffuse = 1` reproduces single-pass behavior bit-for-bit; the annealed scale correctly shrinks sampled noise variance across iterations.
- [ ] **Step 2:** Implement `set_diffuse_scale`; wire the outer loop + per-iteration restore in the example behind `--n-diffuse`.
- [ ] **Step 3:** Verify CUDA-graph capture of the full cycle; measure steps/s (`N_diffuse` cuts steps/s ~`N_diffuse×` at equal `K`; the equal-compute arm reduces `K` to hold `B`, the above-budget arm keeps `K`).
- [ ] **Step 4 (evidence):** run both arms on the tuning track, validate on the 4 validation tracks: (a) equal-compute `(N_diffuse = D, K = K0/D)` vs. `(1, K0)`; (b) above-budget `(N_diffuse = D, K = K0)`. Report both — the equal-compute ablation is evidence, not an auto-reject gate.
- [ ] **Acceptance (per Decisions 1–2):** keep if either arm beats single-pass MPPI on smoothness or lap distance/robustness (e.g. escaping a bad racing-line local minimum) without a smoothness regression, weighing smoothness at least equal to lap distance. The above-budget arm counts because deployment targets a lighter rollout backend. If neither arm wins, record rejected with the numbers and do not merge the outer loop; keep the near-free Task 2 (and Task 3 if kept) as the shipped improvement.

**Commit (if kept):** `Add optional diffusion-annealing outer loop to MPPI rollout`
**Commit (if rejected):** `Record DIAL-MPC outer-loop annealing findings as rejected`

---

### Task 5: Consolidate, document, changelog

- [ ] Fold the kept ingredients' final tuned values into the example's `ControllerMPPI.Config` and cost weights; note interactions (horizon schedule ↔ exec-coupling ↔ knots) in comments.
- [ ] Update `ControllerMPPI` docstring to describe any new Config fields and their default-off semantics.
- [ ] `uv run docs/generate_api.py`; add CHANGELOG entries for kept, user-facing changes.
- [ ] Record the full ablation table (baseline vs. each kept/rejected step at equal `B`) in `docs/superpowers/specs/` or `notes/`.

---

## Results (fill in during execution)

| Config | Total rollouts B | Lap dist (mean ± sd) | Mean cost | Δcmd RMS (jitter) | Steer reversals | Hero-kill frac | Steps/s | Kept? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline single-pass MPPI | — | | | | | | | ref |
| + horizon-annealed sigma | = | | | | | | | ? |
| + spline knots | = | | | | | | | ? |
| DIAL outer loop (D=2) | = | | | | | | | ? |
| DIAL outer loop (D=4) | = | | | | | | | ? |

---

## Decisions (2026-07-07)

1. **Compute ceiling: flexible for experiments.** GPU budget is not a hard wall during experimentation; the practical deployment path is a lighter rollout model (kinematic/dynamic bicycle model or a small learned dynamics net), so `N_diffuse > 1` is **not** ruled out by rollout cost. Task 4 is a **fair trial**, not an expected reject. The equal-compute ablation stays, but as **evidence to report**, not an auto-reject gate. Follow-up (out of scope here): a bicycle-model rollout backend is plausible future work that would enable much larger annealing budgets.
2. **Metric ranking: smoothness ranks at least equal to lap distance.** Steering jitter on straights costs top speed, so a smoother-but-marginally-shorter result is a **KEEP**. Acceptance criteria across all tasks: a smoothness regression = reject; a smoothness win at small lap-distance cost = keep. Robustness (hero-kill fraction) must not regress.
3. **New knobs stay example-level/private until proven.** No new public `ControllerMPPI.Config` fields during experimentation — new knobs live in the example / internal (`_src`) surface. Once an ingredient is proven kept, expose it on the public `Config` (then run `docs/generate_api.py` and add the CHANGELOG entry).
4. **Track protocol: one picked track first, then 4-track validation.** Initial tuning happens on ONE user-picked "interesting" track selected from a generated candidate grid (16 candidates rendered for the user to choose from). Each kept/rejected decision is then validated on 4 other tracks. This replaces the blanket ≥8-seed sweep as the primary protocol; a wider seed sweep remains optional supporting evidence.
5. **Spline vs. existing smoothness machinery: empirically test removal.** Once spline knots are in, explicitly ablate REMOVING the current machinery (AR(1) `beta`, rate cost, exec-coupling) rather than only stacking on top — see the added ablation step in Task 3.
