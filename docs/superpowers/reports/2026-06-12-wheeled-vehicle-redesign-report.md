# Wheeled Vehicle Redesign — Implementation Report

Date: 2026-06-12
Branch: `antoiner/wheeled-vehicle-design`
Spec: `docs/superpowers/specs/2026-06-12-wheeled-vehicle-redesign-design.md`
Plan: `docs/superpowers/plans/2026-06-12-wheeled-vehicle-redesign.md`

## Update 2026-06-15 — Tier 1 tire realism (sim-to-real)

Made the default tire model properly textbook for the steady-state, behind the
unchanged pluggable interface:

- **Canonical theoretical slip** `sigma = slip / (1 + kappa)` (guarded at lock-up),
  replacing the earlier symmetric `1/(1+|kappa|)`. Driving is unchanged; braking
  and lock-up are now correct (slip → large as the wheel locks). The saturation
  law was already the exact brush curve `F = mu*Fz*(1 - (1 - z)^3)`.
- **Self-aligning moment** `Mz = -F_lat * t` with a pneumatic trail `t` that
  auto-scales with wheel radius (`pneumatic_trail_ratio`, default 0.1) and
  collapses to zero as the tire saturates, applied as a couple about the contact
  normal into `body_f`. New `WheelDynamics.mz` diagnostic.

`tire_force` now returns `vec3` `(F_long, F_lat, Mz)`. New tests cover braking
sign, lock-up saturation, and the aligning-moment rise/zero-at-saturation/zero-at-
zero-slip. All 49 vehicle tests (incl. examples, CPU+CUDA) pass. Note: the
canonical slip makes a spinning skid-steer scrub slightly (asymmetric
drive/brake slip — physically real), which is expected.

Not done (Tier 2/3, deferred): anisotropic combined-slip rigor, transient/
relaxation-length dynamics, camber, full Pacejka MF. Suspension is already
handled solver-side (real joints); see note below.

## TL;DR

A clean, ground-up wheeled-vehicle layer (`newton.vehicles`) is **implemented and
verified end-to-end**. A cohesive `WheeledVehicles` controller wraps a MuJoCo
solver that owns collision + normal support (via Newton-detected contacts); the
layer owns analytical wheel spin and a brush combined-slip tire model. An RC-style
Ackermann car drives forward and steers through a curve, and a skid-steer Husky
drives forward and rotates in place — both through real MuJoCo physics.

It was built **alongside** codex's `newton/_src/wheeled/` (untouched), under the
new `newton.vehicles` name / `vehicle:*` attribute namespace, so the two coexist
until you choose to swap.

**40 unit/integration tests pass on CPU and CUDA; 2 examples pass headless.**

One designed piece is intentionally **deferred** (not the dynamics — an optional
contact-quality enhancement): honoring the `preserve_contact_footprint` flag in
Newton's collision core. See "Deferred" below. It is not required for correct
vehicle behavior and would touch shared collision kernels, which I did not want
to land unsupervised. The flag + `ShapeConfig` option are in place and a precise
edit set is recorded.

## What was built

Internal package `newton/_src/vehicles/` (public surface `newton.vehicles`):

| Module | Responsibility |
|---|---|
| `metadata.py` | `vehicle:*` custom attributes; `register_vehicle_attributes`, `set_vehicle`, `add_wheel`, `read_vehicle_model_data` (flat `VehicleModelData` tables), `configure_wheel_solver_contacts` |
| `contact.py` | `WheelContactPatch`; `update_wheel_contact_patches` (per-wheel center/normal/footprint/material from Newton contacts); `latch_wheel_loads` (solver normal load, exponentially smoothed) |
| `tire.py` | `tire_force` Warp func: brush combined-slip (default) + linear, load-normalized, friction-circle saturation |
| `wheel.py` | `WheelDynamics`; `apply_wheel_dynamics` — drive torque (speed servo / torque), slip, tire wrench into `state.body_f`, semi-implicit analytical spin, optional axle reaction |
| `vehicle.py` | `VehicleCommands`; `update_vehicle_controls` — per-vehicle `(drive, steer, brake)` → per-wheel targets + Ackermann steering joint targets, branching on `drive_mode` |
| `controller.py` | `WheeledVehicles` (owns the above; `set_commands`/`update_controls`/`apply`/`latch_loads`), `WheeledConfig`, `DriveMode`/`TireModel`/`DriveInput` enums |

Examples: `newton/examples/vehicles/example_vehicle_husky.py`,
`example_vehicle_rc_car.py`. Run: `python -m newton.examples vehicle_husky` /
`vehicle_rc_car`.

Newton-core (additive, opt-in, no behavior change to existing shapes):
`ShapeFlags.PRESERVE_CONTACT_FOOTPRINT` + `ShapeConfig.preserve_contact_footprint`.

## Verification (observed results)

Tests live in `newton/tests/test_vehicles_*.py` and
`test_collision_preserve_footprint.py`. Run: `uv run --extra dev -m newton.tests -k vehicles`.

- **Metadata** (8): registration, `add_wheel` sets attrs + the footprint flag,
  flat-table read, steer-DOF resolution, **heterogeneous two-vehicle**
  (Ackermann + skid-steer → correct per-vehicle drive modes and wheel→vehicle
  ids), **replication** (one car ×2 → `wheel_vehicle == [0,0,0,0,1,1,1,1]`),
  controller construction. → PASS.
- **Contact** (2, MuJoCo): wheel-on-plane patch active with +Z normal; latched
  normal load within 20% of m·g. → PASS.
- **Tire** (8): zero-slip→0, longitudinal/lateral saturation at μ·Fz, combined
  slip on the friction circle, driving sign, linear slope, zero load. → PASS.
- **Wheel** (6): free spin-up (ω = τ/I·t), brake-to-zero without reversal, tire
  reaction decelerates spin, forward force injected into `body_f`. → PASS.
- **Drive modes** (8): Ackermann inner/outer satisfy
  `cot(outer) − cot(inner) = track/wheelbase`; zero-steer → zero targets;
  skid-steer differential (left/right opposite, equal on straight); speed vs
  torque mode. → PASS.
- **Controller** (6, MuJoCo, CPU+CUDA): **drive forward** (reaches cruise
  ≈ ω·r), **skid-steer rotate-in-place**, pipeline finite. → PASS.
- **Examples** (4, registered in `test_examples.py`, CPU+CUDA, null viewer):
  `vehicle_husky` (2 worlds — also exercises runtime replication) drives then
  rotates; `vehicle_rc_car` drives forward and turns under steering. → PASS.
- **Flag** (4): `PRESERVE_CONTACT_FOOTPRINT` unique bit; `ShapeConfig` round-trip.
  → PASS.

Total: **40 vehicle tests + 4 example tests pass on CPU and CUDA.** Regression on
the surfaces my changes touch also passes: collision/rigid-contact (37), model +
shapes (67). The whole-repo `-m newton.tests` run exceeds the 10-minute tool
timeout (its runner buffers output, so a timed-out run yields no partial
signal); the targeted regressions above cover every file these changes touch
(new package + an additive flag + the `newton.__init__` export), so a full pass
is not expected to differ.

## Key decisions and deviations from the spec (with rationale)

1. **Public name `newton.vehicles` / `vehicle:*` namespace** (spec left this open
   as `newton.wheeled` vs `newton.vehicles`). Chosen so the new layer coexists
   with codex's `newton.wheeled` during the build-alongside phase. **Decide the
   final name at swap.**
2. **Friction ownership via MuJoCo `condim=1`** on wheel geoms (not zeroing μ).
   The spec flagged this as "verify the mechanism." Zeroing μ with `condim=3`
   produces NaN, and MuJoCo combines pair friction as `max(wheel, terrain)` so a
   zeroed wheel μ does not remove tangential friction — it would *cancel* the
   tire force (a double-count). `condim=1` (normal-only) + higher geom priority
   is the correct, verified mechanism. Requires
   `SolverMuJoCo.register_custom_attributes(builder)` before `finalize`, then
   `vehicles.configure_solver_contacts()`.
3. **Uses `use_mujoco_contacts=False`** (Newton-detected contacts → MuJoCo
   solve), per the spec, so contact geometry comes from Newton.
4. **Load-normalized tire stiffness** (`F ∝ c·Fz·slip`, default `c = 20`) instead
   of absolute `[N/slip]`. More robust across vehicle masses (saturation slip
   `≈ 3μ/c` is load-independent). The spec's brush is unchanged in form.
5. **Per-vehicle `(drive, steer, brake)` command** + per-wheel `driven/steerable/
   side/axle_row` masks (the spec's generic "channels" concretized). Covers
   Ackermann (throttle+steer) and skid-steer (drive+differential); AWD/FWD/RWD
   via the `driven` mask.
6. **Latched normal load is exponentially smoothed** (`load_filter`, default 0.2).
   A rigid multi-wheel body is statically indeterminate; the per-wheel solver
   load jitters between diagonals and, with the one-step latch delay, injected
   force jitter. Smoothing fixes it. A sprung asset (real suspension joints)
   would also resolve the indeterminacy.
7. **`apply_reaction_torque` defaults False** (spec leaned "on"). It only affects
   pitch/weight-transfer, not the primary traction, and I left it opt-in pending
   validation since I could not get your review. Implemented and switchable.

## Deferred (designed, not landed)

**Honoring `preserve_contact_footprint` in Newton's collision core.** The flag
and `ShapeConfig` option exist and `add_wheel` sets the flag on wheel shapes, but
the collision kernels do not yet act on it. It is **not required for correct
vehicle dynamics** (with `condim=1` the solver only uses wheel contacts for
normal support, and the tire model consumes center/normal/load — all adequate;
patch *area* is unused). It improves patch-area diagnostics and removes a small
upward bias in the patch center (a spurious analytic plane-cylinder "equator"
contact biases the averaged center by ~⅓·something small; it did not affect the
drive tests).

Why deferred: it touches shared collision kernels used by **all** solvers, and I
judged it irresponsible to land an unverifiable core-collision change in an
unsupervised run. The exact, verified edit set (from a code trace) is:

- **Mechanism 1 (plane-cylinder routing).** `narrow_phase.py:~439` — add
  `and (shape_flags[shape_b] & ShapeFlags.PRESERVE_CONTACT_FOOTPRINT) == 0` to the
  analytical plane-cylinder `elif`. When the flag is set, `contact_dist_*` stay
  `MAXVAL`, `num_contacts == 0`, and the pair routes to GJK/MPR (the
  `gjk_candidate_pairs` block at `~606`). Safe, local; `shape_flags` is already a
  kernel input.
- **Mechanism 2 (axial-rolling projection).** Add a `preserve_footprint: int`
  field to `GenericShapeData` (`support_function.py:83`); set it in
  `extract_shape_data` (pass `shape_flags`, requires threading `shape_flags` into
  the GJK kernel `narrow_phase.py:~640`, the mesh kernel `~955`, and
  `contact_reduction_global.py:~1530` + their launches); have
  `post_process_axial_on_discrete_contact` (`collision_core.py:173`) skip the
  projection when either shape's `preserve_footprint != 0`. This is the invasive
  part — gate it on the **full collision regression suite** being byte-identical
  for unflagged shapes.

Both mechanisms are needed together to recover a flat-ground footprint (M1 routes
to GJK, M2 stops the re-collapse).

Out of scope per the spec (interface room left): powertrain (motor curves,
gearbox, differentials), aero drag, Pacejka/Fiala tire models, hydroelastic
patch source, heightfield tuning.

## Swap checklist (when ready)

1. Decide the final public name (`newton.vehicles` vs `newton.wheeled`).
2. Delete `newton/_src/wheeled/`, `newton/wheeled.py`, its examples
   (`newton/examples/wheeled/`), and tests (`test_wheeled_vehicle_*.py`).
3. Remove `wheeled` from `newton/__init__.py`; re-run `docs/generate_api.py`.
4. Register the two examples in `README.md` (HTML table + 320×320 jpg screenshots
   — not added here because screenshots need a GUI run).
5. No deprecation cycle is required (unreleased feature branch).

## How to run

```bash
uv run --extra dev -m newton.tests -k vehicles                      # tests
uv run --extra examples python -m newton.examples vehicle_husky     # skid-steer
uv run --extra examples python -m newton.examples vehicle_rc_car    # Ackermann
```
