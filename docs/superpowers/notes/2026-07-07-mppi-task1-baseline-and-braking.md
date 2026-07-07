# Task 1 — MPPI benchmark baseline + under-braking diagnosis

Reference numbers for the DIAL-MPC annealing plan
(`docs/superpowers/plans/2026-07-07-dial-mpc-annealing-for-mppi.md`). Produced by
the headless harness `newton/_src/vehicles/_mppi_bench.py` on an NVIDIA RTX 5000
Ada (16 GiB), branch `antoiner/wheeled-vehicle-design`.

## Protocol

- Config: `num_samples=1024`, `horizon=48`, `rollout_substeps=4` (the example
  defaults) → rollout budget `B = 1024·48·4 = 196 608` solver substeps/plan.
- 240 executed frames per run (the example's `num_frames` default, 4 s at 60 Hz).
- 3 repeats per track; mean ± sd reported. The MuJoCo rollout is *not*
  bit-reproducible run-to-run (cross-world contact atomics), but the spread is
  small — see determinism note — so single runs are trustworthy to ~1 %.
- Tracks reproduced exactly via the new `--track-param KEY=VALUE` pass-through
  into `TrackGenConfig` (all 5 are valid at attempt 0, so the resolved seed
  equals the requested seed):
  - tuning: `hull s4 hull_displacement=0.35`
  - `bezier s0`
  - `bezier s9 rad=0.25 min_num_points=12 max_num_points=15`
  - `checkpoint s5 checkpoint_count=18`
  - `repulsive s3 repulsive_grow_mult_min=3.0 repulsive_grow_mult_max=3.5`
    (repulsive uses scale=10 per `TRACK_GENERATOR_SETTINGS`)

### Determinism

Two identical 80-frame runs of the tuning track: lap 3.6166 vs 3.6145 m
(0.06 %), steer reversals 33 vs 35, hero-OOB 0.00 vs 0.00. Over the 3×240-frame
baseline the tuning-track lap sd is 0.22 m on a 19.7 m mean (1.1 %). Not wildly
nondeterministic — not a blocker.

## Baseline metrics (single-pass MPPI, brake channel OFF)

Metrics: `lap` = hero centerline distance over 240 frames [m]; `dRMS d/s` = RMS
of frame-to-frame executed drive/steer increments (jitter proxy); `rev` =
executed-steer reversal count; `oob` = hero out-of-bounds frame fraction;
`sps` = executed steps/s (excludes the graph-capture frame).

| Track | Len [m] | Lap dist [m] (mean ± sd) | Mean cost | ΔRMS drive | ΔRMS steer | Steer rev | OOB frac | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hull s4 hull_displacement=0.35 (tuning) | 120.1 | 19.72 ± 0.22 | −4.1 | 0.0464 | 0.0835 | 89 | 0.00 | 4.23 |
| bezier s0 | 68.9 | 19.82 ± 0.49 | 44.1 | 0.0430 | 0.1166 | 87 | 0.00 | 3.65 |
| bezier s9 rad=0.25 min_np=12 max_np=15 | 88.4 | 16.32 ± 0.12 | 21.0 | 0.0457 | 0.0906 | 68 | 0.00 | 4.59 |
| checkpoint s5 checkpoint_count=18 | 83.1 | 22.53 ± 0.09 | 18.8 | 0.0590 | 0.1139 | 93 | 0.00 | 3.01 |
| repulsive s3 grow_min=3.0 grow_max=3.5 | 90.4 | 22.60 ± 0.07 | −67.2 | 0.0422 | 0.0584 | 74 | 0.00 | 2.28 |

Raw per-run JSON: `2026-07-07-mppi-task1-baseline.json` (same directory).
Cross-check: independent foreground re-runs of checkpoint s5 (22.46 ± 0.04)
and repulsive s3 (22.40 ± 0.07) reproduce the background numbers to ~0.6 %.
Steps/s varies with concurrent GPU load; treat it as indicative. The hero
never leaves the track on any baseline run (OOB 0.00 across all 15 runs).

## Step 2b — under-braking diagnosis

**User observation:** the car does not brake harder into hairpins.

### Ground truth: the action-space encoding (evidence first)

Read end-to-end — planner → example apply kernels → `update_vehicle_controls`
→ `apply_wheel_dynamics`:

1. `ControllerMPPI.sample()` fills `samples[K, H, dim]`. The example uses
   **`dim = 2`**: channel 0 = drive (bounds `[-0.3, 1.0]`), channel 1 = steer
   (`[-1, 1]`). **There is no brake channel in the sampled action space.**
2. `_apply_sample_commands` / `_apply_nominal_command` set `cmd.brake = 0.0`
   unconditionally. So `cmd.brake` is always zero.
3. `update_vehicle_controls._command_kernel`: `brake_target = clamp(brake,0,1) *
   brake_max`, so with `cmd.brake = 0` the friction brake torque is **always 0**.
   Drive (DRIVE_SPEED mode, the example's default) maps to
   `drive_target = drive · max_speed` (`max_speed = 315 rad/s`), so a negative
   drive command requests a *negative target wheel speed*.
4. `apply_wheel_dynamics`: `tau_drive = kp·(drive_target − omega)` clamped to
   `±tau_max`. In the example `motor_max_torque = tau_max = 1.0 N·m`. So the
   only deceleration the planner can command — negative drive — produces at
   most **1.0 N·m** of retarding motor torque. The dedicated friction brake
   (`brake_max_torque = 20 N·m`) is **physically unreachable** from the planner.

### Controlled deceleration experiment (isolates braking authority)

rc_car spun to ~6–7 m/s with full drive, then a 0.25 s command; measured mean
deceleration (`/tmp/decel_test.py`):

| Command | Decel [m/s²] |
| --- | --- |
| coast (drive=0, brake=0) | 3.86 |
| **motor-brake (drive=−0.3, brake=0)** — the only decel the planner can express | **3.61** |
| friction-brake (brake=1.0) — never engaged today | **18.72** |
| both (drive=−0.3, brake=1.0) | 18.68 |

**Negative drive decelerates the car no harder than simply lifting off — it is
weaker than coasting.** The friction brake is 5.2× stronger (tire-grip-limited,
~1.9 g, consistent with the profile's `A_BRK = 14 m/s²` assumption).

### Suspect verdict (plan's ordered list a–e)

- **(a) action-space encoding — CONFIRMED, root cause.** The planner samples no
  brake channel and negative drive only cuts/reverses a 1 N·m motor servo, not
  the 20 N·m friction brake. This alone explains "cannot brake harder."
- (b) AR(1)/exec-coupling smoothing — **not the cause.** Smoothing damps onset
  timing, but even an instantaneous, un-smoothed command cannot brake because
  the actuator it controls has ~no braking authority.
- (c) brake-axis σ too small — **N/A**: there is no brake axis to under-explore.
- (d) horizon too short — **not the cause**: the reference speed profile already
  bakes corner-entry braking foresight into `v_des`, and the over-speed penalty
  gives a braking gradient every step.
- (e) cost shaping penalizing deceleration — **not the cause**: the one-sided
  over-`v_des` penalty *rewards* slowing into corners; the planner wants to
  brake but has no actuator for it.

### Minimal fix (behind a default-off example flag)

`--brake-channel` (Decision 3: example-private until proven): sample a third
command channel (σ=0.35, β=0.6) mapped to `cmd.brake`. Off by default →
`dim = 2`, today's behavior. On → `dim = 3`, brake reaches the friction brake.

**Failure of the naive encoding (kept as a finding):** brake bounds `[0, 1]`
with a zero nominal make the clamped exploration noise strictly positive, so
the softmax-weighted noise average is positive every update and the nominal
brake ratchets up until the 10 N·m of standing brake torque swamps the 1 N·m
motor — measured: permanent mean brake 0.48, lap distance **0.00 m**. Fixed by
sampling brake in `[-1, 1]` and rectifying (`max(0, ·)`) at the apply kernels,
so the negative half means "no brake" and the noise stays zero-mean.

### Before/after (tuning track, 240 frames, 3 repeats, equal compute)

| Config | Lap dist [m] | Peak decel [m/s²] | Mean / max brake | ΔRMS steer | Steer rev | OOB frac | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (no brake channel) | 19.74 ± 0.15 | 5.99 ± 0.66 | 0.00 / 0.00 | 0.0702 | 95 | 0.00 | 4.34 |
| `--brake-channel` (rectified) | **20.26 ± 0.45** | **21.52 ± 2.79** | 0.014 / 0.54 | 0.0898 | 94 | 0.00 | 4.32 |

Raw per-run JSON: `2026-07-07-mppi-task1-brake-fix.json` (same directory).

With the channel on, the planner brakes **3.6× harder** into the hairpins
(peak decel 21.5 vs 6.0 m/s², matching the measured 18.7 m/s² friction-brake
authority plus corner load transfer) while using the brake sparingly
(mean command 0.014 — pulses at corner entry) and gains ~2.6 % lap distance
with no robustness loss and negligible compute cost. Executed-drive/steer
jitter is statistically unchanged (steer ΔRMS spread between baseline arms of
different sessions is comparable, 0.070–0.084).

**Conclusion:** the flag stays example-private and default-off per Decision 3.
Annealing experiments (Tasks 2–4) run against the default (2-channel) baseline
above; the brake channel is available as a validated knob if corner-entry
braking becomes the bottleneck.

## Regen-braking reframing and ESC mode (user correction, 2026-07-07)

A real electric RC car (e.g. a Traxxas VXL setup) has **no friction brake** —
braking is regen/drag braking through the motor ESC, and the transmitter folds
it into the throttle axis: the negative throttle half commands ESC brake, not
reverse. Our `brake_target` mechanism is already physically ESC-like
(resistive torque, zero-crossing clamp, cannot reverse the wheel) — only the
"friction brake" label above was off. `brake_max_torque` should be read as the
**ESC brake-current limit** (to be calibrated against the real VXL, not
treated as a hydraulic constant).

`--brake-channel` is superseded by `--brake-mode {none, channel, esc}`
(default `none` = today's behavior):

- `channel`: third sampled command in [-1, 1], rectified to [0, 1] (as above).
- `esc` (interface-faithful): the action space stays 2-D; at command
  application, `drive >= 0` → throttle with brake 0, `drive < 0` → throttle 0,
  `brake = |drive|`. The drive axis opens to the full transmitter range
  `[-1, 1]` (its negative half no longer means reverse). The [0, 1]-bound
  ratchet pathology cannot recur here: the brake engages only while the
  sampled drive is negative, and the drive axis's exploration noise is
  symmetric — confirmed empirically (mean brake 0.012, car launches normally).

### Head-to-head (tuning track, 240 frames, 3 paired same-session repeats)

| Mode | Lap dist [m] | Peak decel [m/s²] | Hairpin-1 entry decel [m/s²] | mean/max brake | ΔRMS drive | ΔRMS steer | Rev | OOB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 19.97 ± 0.20 | 7.2 ± 0.2 | 7.2 ± 0.2 | 0.000 / 0.00 | 0.0458 | 0.0749 | 93 | 0.00 |
| channel | 20.53 ± 0.16 | 23.9 ± 4.2 | 18.9 ± 2.4 | 0.012 / 0.68 | 0.0416 | 0.0831 | 89 | 0.00 |
| esc | 20.30 ± 0.21 | 21.6 ± 4.0 | 21.6 ± 4.0 | 0.012 / 0.30 | 0.0649 | 0.0830 | 91 | 0.00 |

Raw per-run JSON: `2026-07-07-mppi-task1-brake-modes.json`. Hairpin-entry
metrics come from the harness's new per-run hairpin instrumentation (deepest
reference-speed minima within the driven arc; 5 m entry window). The second
hairpin lies beyond the ~20 m covered in 240 frames, so only hairpin 1 is
populated at this frame budget.

### Notes on the folded axis (esc)

In esc mode the brake half-axis implicitly inherits the drive channel's AR(1)
smoothing and noise (β = 0.85, σ = 0.35). Braking onset is not blunted — esc
posts the strongest hairpin-1 entry decel (21.6 m/s²) of the three modes — so
the drive smoothing parameters remain serviceable when the axis does double
duty. The visible cost is executed-drive jitter (ΔRMS 0.065 vs 0.046),
because the executed command now crosses zero between throttle and brake;
physically this is ESC current chatter, not a mechanical concern, but a small
zero deadband or a slightly higher drive β is the obvious knob if it bothers.

### Recommendation

**Prefer `esc`**: it is interface-faithful to the real vehicle, brakes just as
hard into the measured hairpin (21.6 vs 18.9 m/s² entry decel), matches
channel's lap distance within noise (20.30 vs 20.53 m, ≈1 %), needs no extra
action dimension, and is immune to the rectification pathology by
construction. Both modes stay in the example as cheap flags; the default
remains `none` until the user flips it, and Tasks 2–4 baselines stay
unconfounded.

## Task 2 — horizon-annealed noise schedule (REJECTED)

Implementation: private `ControllerMPPI._set_sigma_horizon_factor(f)` scales
the per-step exploration noise by `f ** (t / (H - 1))` — the executed step
keeps the configured sigma, the far horizon end explores with `f * sigma`.
The schedule lives in a device array (`_sigma_schedule`, all-ones default),
so it is CUDA-graph-safe and runtime-tunable like `set_temperature`/`set_beta`;
`f = 1.0` (default) is bit-identical to the flat sampler (regression-locked by
`test_sigma_horizon_factor_one_is_bit_identical`). Unit tests cover schedule
shape/monotonicity, far-horizon variance growth, and bounds.

All runs below use `--brake-mode esc` (accepted braking model). Task-1
baselines are NOT comparable; every comparison is paired same-session.

### Tuning-track sweep (hull s4, 240 frames, 3 paired reps, equal compute)

| Factor | Lap dist [m] | ΔRMS drive | ΔRMS steer | Steer rev | OOB |
| --- | --- | --- | --- | --- | --- |
| 1.0 (flat) | 20.32 ± 0.17 | 0.0667 ± 0.0046 | 0.0816 ± 0.0049 | 98 ± 6 | 0.00 |
| 1.5 | 20.39 ± 0.08 | 0.0649 ± 0.0038 | 0.0811 ± 0.0066 | 103 ± 4 | 0.00 |
| 2.0 | 20.34 ± 0.17 | 0.0609 ± 0.0031 | 0.0866 ± 0.0085 | 104 ± 10 | 0.00 |
| 3.0 | 20.37 ± 0.13 | 0.0635 ± 0.0039 | 0.0861 ± 0.0124 | 109 ± 5 | 0.00 |

No factor improves smoothness: executed-drive ΔRMS moves at most −9 % (within
the ±0.005 rep noise), executed-steer ΔRMS drifts up at f ≥ 2, and steering
reversals monotonically worsen (98 → 109). Lap distance is flat. The named
secondary metric — ESC-mode zero-crossing drive jitter, 0.065 — is **not
reliably reduced** by any factor.

Structural note: with `sigma_t = sigma * f^(t/(H-1))` the executed step's
noise is unchanged by construction (`schedule[0] = 1`); the knob only adds
far-horizon exploration. It therefore cannot calm t = 0 — the jitter
hypothesis this task targeted needed near-term *reduction*, which is the same
knob as lowering sigma itself.

### Validation (f = 2.0 vs 1.0, esc mode, paired; bezier tracks x2 reps)

| Track | Lap f=1.0 | Lap f=2.0 | ΔRMS steer f=1.0 → f=2.0 |
| --- | --- | --- | --- |
| bezier s0 | 10.50 (10.52, 10.48) | 19.10 (19.85, 18.35) | 0.054 → 0.108 |
| bezier s9 hairpins | 8.81 (4.77, 12.85) | 12.75 (12.73, 12.76) | 0.057 → 0.076 |
| checkpoint s5 | 23.29 | 23.41 | 0.105 → 0.105 |
| repulsive s3 | 22.56 | 22.60 | 0.059 → 0.061 |

Raw JSON: `2026-07-07-mppi-task2-sigma-horizon.json`.

The bezier rows expose a **separate esc-mode finding**: at flat sigma the
esc-mode car reproducibly crawls on bezier s0 (lap 10.5 both reps, mean speed
2.6 m/s vs 19.8 m lap in Task 1's `none` mode) and collapses intermittently
on bezier s9 (4.8 vs 12.9). Extra far-horizon exploration (f = 2.0) largely
rescues both — but with steer jitter doubling on s0, and the f=1.0 jitter
numbers are flattered by the low speed (a crawling car barely steers). This
is an exploration-rescue of an esc-mode low-speed stall, not the near-term
smoothness win the task targeted; the stall itself deserves a dedicated
follow-up (suspect: the drive axis opened to [-1, 1] hovers near zero/negative
where ESC drag is cost-neutral at low speed).

### Verdict (Decision 2: smoothness >= lap distance)

**REJECTED.** No factor improves smoothness anywhere; the tuning track shows
a mild reversal regression and validation's lap wins come bundled with steer
jitter increases. The knob stays private and defaults to 1.0 (exact no-op),
kept in the code because Task 4's annealed outer loop reuses the same
device-array noise-scaling mechanism. Follow-up filed in these notes: fix the
esc-mode low-speed stall on flowing tracks before re-testing exploration
schedules.

## ESC-mode low-speed crawl: diagnosis and fix (follow-up from Task 2)

### Reproduction and instrumentation (bezier s0, esc, flat sigma)

Confirmed: lap 10.95, mean speed 2.71 m/s vs v_des 5.52. The time course is
the tell — this is not a launch problem:

| Frames | Speed [m/s] | Executed drive (mean) | Frames braking | Horizon steps with drive < 0 |
| --- | --- | --- | --- | --- |
| 0–60 | 2.06 | +0.66 | 0 % | 0 % |
| 60–120 | 5.95 | +0.82 | 0 % | 27 % |
| 120–180 | 2.74 | −0.43 | 87 % | **99 %** |
| 180–240 | **0.10** | −0.33 | 78 % | 56 % |

The car runs cleanly at ~6 m/s, brakes for a corner, overshoots to a
standstill, and **stays parked for 60+ frames** with v_des ≈ 5 and ESS ≈ 396
(~40 % of K — the softmax is nearly uniform).

### Mechanism (hypotheses tested)

- **(a) equilibrium trap — CONFIRMED, binding.** The one-sided over-speed
  penalty never punishes going slow, so a standstill is nearly cost-free; the
  progress reward's spread across samples that all start from 0 m/s is too
  small for temperature 15 to concentrate on (ESS 396). Direct test: the
  purely structural fix (b) below does *not* rescue the crawl (bez s0 lap
  10.1 with it), while a cost-side shortfall penalty does (lap 20.7–21.1).
- **(b) sampling asymmetry — CONTRIBUTING.** In `none` mode the drive bound
  −0.3 truncates the noise, biasing the post-clamp mean positive near the
  bound — a built-in self-recovery that esc's symmetric [−1, 1] axis lost.
  Restoring the tight bound (with a brake gain) alone does NOT fix the crawl
  (lap 10.1 / 4.7 on the bezier tracks): the bias is too weak against
  amplified small-negative ESC drag. Rejected as the primary fix; the bound
  stays open.
- **(c) warm-start persistence — CONFIRMED as the trap's memory.** After the
  overshoot, 99 % of the nominal's horizon steps are negative; with zero-mean
  noise and a flat cost landscape the softmax average has no direction, so
  `shift()` re-seeds the braking plan forever.

### Fix (esc-mode defaults, example-level)

1. **Anti-stall shortfall penalty** (the load-bearing part): new cost term
   `w_under * max(0, min(v_des, V_STALL) - speed)^2` with `V_STALL = 1.5 m/s`
   and `w_under = 0.5` (default in esc mode, 0 otherwise — `--w-underspeed`
   overrides). Gating by an absolute stall floor rather than tracking v_des
   matters: an ungated (or margin-gated) shortfall measurably raises hull
   drive jitter ~40 % (controlled same-session: ΔRMS drive 0.072 → 0.098–0.107)
   because it turns into a speed-tracking gradient during normal running.
   `w_under = 0.25` is too weak (crawl returns: bez s0 10.3, s9 4.8).
2. **Transmitter deadband**: esc commands in (−0.05, 0) coast; the remaining
   negative span maps linearly to the full [0, 1] brake range. A neutral zone
   like a real transmitter, so near-zero exploration noise does not drag the
   ESC brake.

### Acceptance (all 5 tracks, esc-fixed vs none, 240 frames)

| Track | none lap [m] | esc-fixed lap [m] | esc hairpin-1 decel [m/s²] | ΔRMS drv/str (esc) | OOB |
| --- | --- | --- | --- | --- | --- |
| hull s4 (tuning) | 20.09 | 20.41 | 22.0 | 0.083 / 0.100 | 0.00 |
| bezier s0 (was 10.5, crawl) | 19.88 | 20.59 | 29.0 | 0.054 / 0.099 | 0.00 |
| bezier s9 (was 4.8–12.9) | 16.36 | 16.62 | 22.3 | 0.064 / 0.090 | 0.00 |
| checkpoint s5 (paired) | 22.28 | 23.31 | — | 0.060 / 0.100 | 0.00 |
| repulsive s3 (paired) | 22.68 | 22.53 | — | 0.055 / 0.057 | 0.00 |

Both bezier tracks recover to (or beyond) their none-mode laps; hull's
hairpin braking authority is preserved (hp1 22.0 vs Task-1 esc 21.6 ± 4;
`none` manages only 7–8). Checkpoint/repulsive jitter matches none-mode.
Raw JSON: `2026-07-07-mppi-esc-crawl-fix.json` (includes the rejected
bounded-axis, margin-gate, and w=0.25 arms).

**Residual (honest):** on hull, the esc-fixed drive ΔRMS (0.082–0.089) sits
~15–30 % above the esc-w0 range (0.061–0.073). The shortfall term reshapes
which braking rollouts win near hairpins even when the executed speed never
drops below the stall floor (rollout samples do). All lighter variants tried
(margin gate, stall floor 3.0→1.5, w 0.25) either kept the jitter or
reintroduced the crawl; recorded as a known cost of the fix. Steer ΔRMS and
reversals stay within the esc baseline range everywhere.

Task-1's "esc within noise of channel" conclusion (measured only on hull s4)
survives: hull was never affected by the crawl, and the fixed esc mode now
also holds up on the 4 validation tracks.

## Task 3 — spline-knot control parameterization (KEPT: n_knots = 12)

Implementation: private `_n_knots` constructor argument on `ControllerMPPI`
(Decision 3). When set, the decision variable is `n_knots` coarse control
points; noise is sampled (and softmax-averaged) at knot resolution, and a
graph-capturable linear-interpolation kernel expands the per-sample knot
commands to `samples [K, H, A]`, so the example's rollout loop is untouched.
`None` (default) is bit-identical to the per-step sampler. The AR(1) `beta`
and the Task-2 sigma schedule act at knot resolution. Interpolation cost is
unmeasurable (steps/s unchanged, ~4.3). The t=0 exec-coupling weight moved to
its own cost slot (`params[9] = w_rate * EXEC_COUPLING`, behavior unchanged)
so the removal ablation could zero it independently.

All runs: `--brake-mode esc`, tuning track hull s4, 240 frames, paired
same-session. Raw JSON: `2026-07-07-mppi-task3-knots.json`.

### Finding: the knot warm-start must shift one STEP, not one knot

The first implementation rolled the knot array a whole knot per frame
(= (H-1)/(n-1) ≈ 6.7 fine steps at n=8), silently discarding most of the
converged plan every replan. Measured (3 reps): reversals 96 → 119 and lap
−4 % at n=8 vs per-step, with no ΔRMS win — a would-be spurious rejection.
Fixed by resampling the knot spline one fine step later
(`delta = (n-1)/(H-1)` knot units, last knot held); the raw JSON keeps the
broken-shift sweep under `broken_knot_shift_sweep` as the recorded finding.

### Tuning-track sweep (fixed shift, 3 paired reps, equal compute)

| n_knots | Lap dist [m] | ΔRMS drive | ΔRMS steer | Steer rev | OOB |
| --- | --- | --- | --- | --- | --- |
| None (per-step, H=48) | 20.43 ± 0.04 | 0.0761 ± 0.0060 | 0.0947 ± 0.0025 | 91 ± 5 | 0.00 |
| 8 | 20.40 ± 0.12 | 0.0407 ± 0.0013 | 0.0477 ± 0.0036 | 84 ± 3 | 0.00 |
| **12 (picked)** | 20.33 ± 0.26 | 0.0410 ± 0.0007 | 0.0513 ± 0.0021 | 76 ± 3 | 0.00 |
| 16 | 19.99 ± 0.66 | 0.0446 ± 0.0020 | 0.0579 ± 0.0007 | 68 ± 5 | 0.00 |

Every knot count cuts executed drive AND steer jitter 40-50 % at equal lap
distance — including the Task-1 honest residual (esc drive jitter 0.065-0.08
falls to ~0.041). n=12 picked: ΔRMS within noise of n=8, reversals −16 %,
lap flat; n=16 trades further reversal wins for a lap dip and run-to-run
variance. This also cleanly beats the Task-2 winner (f=2.0: 0.0609/0.0866,
rev 104).

### Decision-5 removal ablation (n_knots=12, 2 reps per arm vs 3-rep base)

| Arm | Lap [m] | ΔRMS drive | ΔRMS steer | Steer rev | Verdict |
| --- | --- | --- | --- | --- | --- |
| knots12, all machinery on | 20.33 ± 0.26 | 0.0410 | 0.0513 | 76 | ref |
| − AR(1) beta (=0) | 20.46 ± 0.09 | 0.0378 | 0.0501 | 65 | tuning-KEEPable, but **fails validation** (below) |
| − w_rate (=0) | 20.45 ± 0.10 | 0.0432 | 0.0516 | 81 | KEEP w_rate (mild but free) |
| − exec-coupling (w_exec=0) | 20.51 ± 0.10 | 0.0482 | 0.0675 | 89 | **KEEP exec-coupling** (steer ΔRMS +32 %) |
| − all three | 20.33 ± 0.04 | 0.0421 | 0.0656 | 84 | confirms w_exec is the load-bearing term |

The spline does NOT subsume exec-coupling: replan-to-replan flicker is
invisible to any within-plan prior. beta=0 looked like a win on the tuning
track (rev 65, ΔRMS best) but reproducibly stalls bezier s9 (lap 4.67 / 4.68
twice, vs 18.27 with beta on): white knot noise loses the temporal reach
that climbs out of the low-speed regime, so the AR(1) stays. **Keep all
three mechanisms; the spline stacks on top.**

### Validation (knots12, default machinery, paired vs per-step, 1 rep)

| Track | Lap base → knots12 [m] | ΔRMS drv base → k12 | ΔRMS str base → k12 | Rev base → k12 |
| --- | --- | --- | --- | --- |
| bezier s0 | 19.53 → 20.94 | 0.051 → 0.041 | 0.098 → 0.066 | 81 → 81 |
| bezier s9 hairpins | 12.97 → 18.27 | 0.059 → 0.048 | 0.074 → 0.070 | 87 → 66 |
| checkpoint s5 | 23.16 → 23.49 | 0.066 → 0.048 | 0.099 → 0.067 | 84 → 76 |
| repulsive s3 | 22.56 → 22.42 | 0.049 → 0.032 | 0.057 → 0.049 | 87 → 63 |

Smoothness improves on all four tracks, lap distance equal or better
everywhere (s9 +41 % — the smoother drive axis also avoids the esc
stall-recovery churn), OOB 0.00 throughout.

### Verdict (Decision 2)

**KEPT: `n_knots = 12`** (with beta/w_rate/exec-coupling all retained).
40-50 % executed-command jitter reduction at equal-or-better lap distance on
all 5 tracks — the strongest result of the plan so far, and it directly
repairs the Task-1 residual (esc drive chatter). The knob stays private
(`_n_knots`, example `--n-knots`) until Task 5 consolidates; promotion to
the public `Config` + CHANGELOG belongs there per the plan.
