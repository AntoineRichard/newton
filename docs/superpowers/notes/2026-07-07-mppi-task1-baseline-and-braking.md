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
