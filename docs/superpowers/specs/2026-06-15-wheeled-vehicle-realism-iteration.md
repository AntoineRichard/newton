# Wheeled Vehicle Realism Iteration (newton.vehicles)

Status: in progress
Date: 2026-06-15
Driver: **sim-to-real RL for a specific ground robot** (RC car / Clearpath AGV).
Layer: `newton.vehicles` (the ground-up redesign).

Related docs:
- Design: `docs/superpowers/specs/2026-06-12-wheeled-vehicle-redesign-design.md`
- Plan: `docs/superpowers/plans/2026-06-12-wheeled-vehicle-redesign.md`
- Report: `docs/superpowers/reports/2026-06-12-wheeled-vehicle-redesign-report.md`

## Purpose

Track realism improvements to the working `newton.vehicles` layer, prioritized
for sim-to-real of ground robots (not fidelity for its own sake). Realism work
should be anchored to a validation target; without reference data we are tuning
blind.

## Done in this iteration

### Tier 1 — canonical brush tire + self-aligning moment (landed)

Commit: "Make default tire model canonical brush with self-aligning moment".

- **Canonical theoretical slip** `sigma = slip / (1 + kappa)` (guarded at lock-up)
  replacing the symmetric `1/(1+|kappa|)`. Driving unchanged; braking and lock-up
  now correct. The saturation law was already the exact brush curve
  `F = mu*Fz*(1 - (1 - z)^3)`.
- **Self-aligning moment** `Mz = -F_lat * t`, pneumatic trail `t` auto-scaled by
  wheel radius (`WheeledConfig.pneumatic_trail_ratio`, default 0.1), collapsing to
  zero at saturation, applied as a couple about the contact normal into `body_f`.
  `tire_force` now returns `vec3 (F_long, F_lat, Mz)`; added `WheelDynamics.mz`.
- Tests: braking sign, lock-up saturation, aligning-moment rise/zero-at-saturation/
  zero-at-zero-slip. All 49 vehicle tests pass (CPU + CUDA).
- Known consequence: the canonical slip makes a spinning skid-steer scrub slightly
  (asymmetric drive/brake slip — physically real).

## Proposed work

### Item A — Wheel-contact gap + radial compliance (high value, cheap)

**Motivation.** A real tire is radially compliant; letting the wheel sink a few
mm both represents that and geometrically widens the cylinder-plane footprint
(chord `~ 2*sqrt(2*R*d)` at sink depth `d`). More importantly, the default
positive contact gap injects a spurious analytic plane-cylinder "margin/equator"
contact that biases the patch center far up the wheel.

**Empirical evidence** (wheel-on-plane, R=0.2 m, condim=1, settled):

| wheel `ke` | `gap`   | sink   | patch center z | contacts | area    |
|------------|---------|--------|----------------|----------|---------|
| 2500 (def) | default | 0.2 mm | **66.5 mm**    | 3        | 200 cm² (fake) |
| 100        | default | 2.9 mm | 63.8 mm        | 3        | 200 cm² |
| 30         | default | 5.2 mm | 61.5 mm        | 3        | 200 cm² |
| 2500       | **0.0** | 0.2 mm | **-0.2 mm**    | 2        | 0 cm²   |

**Findings.**
- `gap = 0` removes the spurious contact (3→2) and puts the patch center at
  ground level (66 mm → ~0 mm) with a stable, honest patch. This is the main
  accuracy win and is a per-shape config change, not a collision-core change.
- Softer `ke` adds real sink (radial compliance); on its own it only nudges the
  center because the equator contact still dominates. The win is `gap = 0`;
  softness is the realism dial on top. (`gap = 0` + softer `ke` together should
  give an accurate center *and* a real fore-aft footprint — confirm.)

**Scope.**
- Make `configure_wheel_solver_contacts` (and `WheeledVehicles.configure_solver_contacts`)
  set wheel-ground contact `gap ~= 0` in addition to `condim = 1`.
- Expose a tunable wheel-ground radial stiffness/softness (via shape `ke` /
  MuJoCo `solref`) so sink depth can be matched to a real tire's radial
  deflection under load. Default: keep stiff (minimal sink) unless the user opts
  into compliance; document tuning to the target tire.
- Re-verify the examples (changing the gap shifts contact behavior slightly).

**Exit criteria.**
- Flat-ground patch center within ~5 mm of the ground (vs 66 mm now).
- `vehicle_husky` and `vehicle_rc_car` still drive/steer/rotate stably with the
  new gap.
- Sink depth configurable; with softening the footprint area is non-degenerate.

**Risks.** `gap = 0` reduces the broad-phase detection margin; fast vehicles or
large `dt` may need a small positive gap (CCD). Keep it tunable.

### Item B — Tier 2: sprung-suspension validation via `rc_car.usda`

**Decision:** use the existing `newton/examples/assets/wheeled/rc_car.usda`
(authored suspension + steering) rather than an in-code sprung car. It matches
the sim-to-real workflow (the real robot is a USD asset) and is the genuine test.
The only mild edge of an in-code car is "no USD dependency in the test" — not
worth losing the real validation.

**Context:** suspension is already handled solver-side (real prismatic joints);
`newton.vehicles` is agnostic to it and needs no changes — it reads the wheel
body pose and applies tire forces, and the suspension joint transmits to the
chassis. This path is currently **unexercised** (examples are rigid single-body),
so it is real validation, not new capability.

**Scope.**
- Load `rc_car.usda`; map its wheel shape/body, prismatic-suspension, and
  revolute-steering labels (see `assets/wheeled/manifest.json`) → annotate with
  `add_wheel` (set `steer_joint` on the fronts; leave suspension joints to the
  solver), `set_vehicle(drive_mode=ACKERMANN, ...)`, apply `gap~=0`/`condim`.
- Add a `vehicle_rc_car_sprung` example (or load-asset variant) + a test
  (`usd_required`).

**Exit criteria.**
- The sprung car drives + steers stably on flat ground (and a ramp) through
  `newton.vehicles`, with visible suspension travel.
- The load-smoothing band-aid can be turned down (`WheeledConfig.load_filter`
  toward 1.0 = off) without instability — confirming the sprung load
  distribution removes the rigid-body indeterminacy. Update the default if so.

### Item C — Collision-core `preserve_contact_footprint` (deferred / mostly obviated)

The scoped Newton collision fix (honor `ShapeFlags.PRESERVE_CONTACT_FOOTPRINT`)
is **largely obviated for the patch center** by Item A's `gap = 0`. It remains
the only way to get a true 2-D footprint *area* on non-flat terrain (boxes,
ramps, meshes), which the current tire model does not use. Keep deferred; the
exact traced edit set is in the redesign report (Mechanism 1 at
`narrow_phase.py:~439`; Mechanism 2 via a `GenericShapeData.preserve_footprint`
field read in `post_process_axial_on_discrete_contact`). Revisit only if a future
tire model needs contact area on non-flat terrain.

## Out of scope (Tier 3, future)

Anisotropic combined-slip rigor (exact unequal long/lat brush), transient /
relaxation-length tire dynamics, camber thrust, full Pacejka Magic Formula
(needs coefficient data RC/AGV users rarely have). Powertrain (motor curves,
gearbox, differentials), aero drag.

## Open questions

- **Validation target.** Sim-to-real needs reference data for the specific robot
  (measured accel/braking distance, steady-state cornering radius vs speed, or a
  trusted reference sim). Which robot, and is reference data available? This
  determines which fidelity gaps actually matter and the tire/suspension tuning.
- Default wheel-ground softness (Item A): keep stiff by default, or ship a modest
  compliance default? Depends on the validation target.
- `apply_reaction_torque` (drivetrain stator reaction) is implemented but default
  off and unvalidated — turn on once there is a validation target.
