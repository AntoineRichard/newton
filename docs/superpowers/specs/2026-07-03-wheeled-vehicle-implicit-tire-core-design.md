# Wheeled-Vehicle Implicit Impulse-Budget Tire Core — Design

- **Date:** 2026-07-03
- **Status:** Approved design, pending implementation plan
- **Supersedes:** the explicit tire-force injection core of `newton/_src/vehicles/`
  (`wheel.py` apply kernel + `contact.py` load latch band-aids). Everything else in
  the `newton.vehicles` layer is retained.
- **Related:** `2026-06-12-wheeled-vehicle-redesign-design.md` (layer architecture,
  still valid), `2026-06-16-relaxation-length-tire-design.md` (rejected; its outcome
  section is the evidence base for this design), commit `f918cebbd`.

## 1. Problem statement

The `newton.vehicles` layer simulates wheeled robots (Ackermann RC car — Traxxas
Slash class — and differential/skid-steer Husky) as a thin tire-model layer over a
wrapped rigid-body solver (MuJoCo Warp with Newton contacts). It works well at
μ ≈ 1 but **explodes at high friction (μ ≳ 2) in low-speed regimes and during hard
braking**: the car hops, rolls, and diverges.

The prior effort's own experiments established the failure is structural, not a
tuning problem:

- μ = 3 does **not** converge as dt shrinks — more substeps do not help.
- Applying the tire wrench to the chassis instead of the wheel body fixes the hop
  (μ = 3 bounce metric 15 → 0.9) but breaks cornering, because it bypasses the
  steering-joint compliance that damps yaw.
- Adding relaxation-length transient slip made it worse (phase lag amplifies the
  oscillation) and was reverted.
- Making wheel inertia realistic made it worse (the unrealistically high default
  spin inertia was masking the instability).

**Root cause.** The tire force is a very stiff function of contact-point velocity,
evaluated at the start of the substep and injected as a constant explicit external
force into `state.body_f` of the **light wheel body** (0.18 kg on the rc_car), with
no implicit coupling to the body it pushes. Near zero slip velocity the force
saturates at ±μ·Fz; the impulse μ·Fz·dt far exceeds the wheel body's tangential
momentum, so the slip velocity overshoots and flips sign each substep, and the
resulting ±μ·Fz square wave pumps the suspension roll/hop mode. Higher μ means a
larger square wave. Secondary defects compound it:

- The shipped low-speed lateral cap (`lat_cap = fz/9.81 · |v_lat|/dt`) is the right
  idea — an impulse budget — computed against the wrong mass (chassis share `Fz/g`
  instead of the wheel body's contact effective mass) and never applied
  longitudinally, which is why hard braking still explodes.
- Normal load `Fz` is latched from the previous substep and low-pass filtered, so
  normal and tangential forces are never consistent within a step; on landing the
  tire fires with the last airborne load.
- There is **no static-friction regime**: near zero slip the model is a pure
  damper, so a parked car creeps sideways on a camber and brakes cannot hold on a
  hill.
- At brake lock-up (κ → −1) the canonical-slip guard produces `1/(1+κ) ≈ 10⁴`; the
  saturated force *magnitude* is fine but its *direction* is a ratio of two
  amplified near-noise quantities, injecting alternating lateral kicks exactly when
  load is highest.

## 2. Goals and non-goals

### Goals

1. **Stability by construction at high grip.** No substep may apply more tangential
   impulse than the friction circle allows: ‖p_t‖ ≤ μ·Fz·dt. Acceptance envelope:
   μ ∈ [0, 2.5] (soft RC compounds on high-grip surfaces).
2. **True static friction.** A stationary vehicle holds position on a slope with
   brakes applied; no lateral creep at rest.
3. **Robust lock-up braking.** Hard braking from top speed stops the vehicle
   monotonically with no lateral kick or explosion at any μ in the envelope.
4. **Preserve validated behavior at μ ≈ 1**: steered launch without spin-out,
   drift-free straight line, correct Ackermann cornering, skid-steer rotate in
   place.
5. **Remove the band-aids** (lateral cap, load filter) rather than retune them.
6. Keep the layer solver-agnostic, batched (thousands of vehicles), and free of
   per-wheel Python in the runtime loop.

### Non-goals

- Full Pacejka Magic Formula, camber thrust, anisotropic combined slip, powertrain
  (motor curves, differentials), aero drag — unchanged future work.
- Moving tangential friction inside the wrapped solver's contact solve
  (condim = 3 with slip-modulated μ). Considered and set aside: it ties the layer
  to MuJoCo and makes slip-curve shaping and the self-aligning moment harder to
  control. The interface changes here do not preclude it later.
- True 2-D contact-patch area on arbitrary terrain (the tire model does not use
  area; the analytic plane-cylinder footprint is kept as a diagnostic).

## 3. What is retained unchanged

The audits (2026-07-03) judged these sound; they are load-bearing and must not be
rewritten:

- **USD assets:** `newton/examples/assets/wheeled/rc_car.usda` (4.0 kg, exact solid
  inertias, prismatic spring/damper suspension 800/30, revolute steering ±35°,
  free axles), `husky.usda` (80 kg skid-steer), `manifest.json`, and the test
  sublayer variants.
- **Metadata layer** (`_src/vehicles/metadata.py`): `vehicle:*` custom attributes
  with custom frequencies and `references=` remapping (replication-safe ids), flat
  device tables built once at finalize.
- **Controller/API layer** (`vehicle.py`, `controller.py`, `newton/vehicles.py`):
  `WheeledVehicles`, normalized `(drive, steer, brake)` commands, heterogeneous
  drive modes in one batched kernel, Ackermann geometry.
- **Brush tire curve** (`_src/vehicles/tire.py`): canonical theoretical slip,
  parabolic-pressure brush magnitude `F = 3·μ·Fz·φ·(1 − φ + φ²/3)`, intrinsic
  combined slip, load-normalized stiffness, pneumatic-trail self-aligning moment.
  Reused as the force law inside the new solve.
- **Contact configuration** (`configure_wheel_solver_contacts`): `condim = 1`
  (solver owns normal support only), geom priority, `gap = 0` patch centering,
  optional radial compliance.
- **Axle-lock joint surgery** (`configure_wheel_axle_joints`): revolute → fixed
  conversion with DOF remapping. (Long-term it should become a first-class
  `ModelBuilder` operation; out of scope here.)
- **Per-substep pipeline shape:** `update_controls → model.collide → apply →
  solver.step → solver.update_contacts → latch_loads`.
- **Physics regression tests** in `newton/tests/test_vehicles_*.py`, adapted where
  they lock in band-aid behavior (see §7).
- **Core-engine changes:** hydroelastic-plane support, narrow-phase footprint
  toggles, USD `references` customData parsing.

## 4. The new core: per-wheel implicit impulse-budget solve

Replaces the explicit force injection and one-sided semi-implicit spin update in
`_src/vehicles/wheel.py` (`apply_wheel_dynamics`, roughly lines 209–293).

### 4.1 State and frames

Per wheel, at the geometric ground-contact point `c = wheel_center − r·n` (kept —
the solver's averaged patch centroid caused a spurious yaw torque):

- Contact tangent frame `(t_fwd, t_lat, n)` from the wheel forward direction
  projected onto the patch plane.
- Slip velocities `u = (u_long, u_lat)` where `u_long = v_c·t_fwd − ω·r` and
  `u_lat = v_c·t_lat`, with `v_c` the wheel-body twist evaluated at `c`.
- Analytical spin state `ω` with inertia `I` (unchanged representation; axle
  joints remain locked).

### 4.2 Effective mass

Build the contact-frame effective mass of the wheel body from `body_inv_mass` and
`body_inv_inertia` (world-frame, at the contact point):

```
W  = J M⁻¹ Jᵀ            # 2×2 tangential Delassus block of the free wheel body
Mw = W⁻¹                 # contact effective mass, 2×2
```

plus the scalar spin inertia `I` coupled to `u_long` through the wheel radius.

The free-body Delassus ignores the suspension and steering joint constraints.
Constraints can only *increase* effective mass, so `Mw` is a lower bound: the solve
computes impulses the wheel body can definitely absorb. The error is one substep of
slight under-grip — always on the stable side. This is a deliberate accuracy/
stability trade and must be documented in the kernel.

### 4.3 The solve

Per wheel, per substep, solve the linearized implicit system in
`x = (u_long⁺, u_lat⁺, ω⁺)`:

1. Evaluate the brush curve at the current slip state to get the operating-point
   force `F₀` and the local tangent stiffness `K = ∂F/∂u` (2×2, diagonal in the
   isotropic case, from the load-normalized brush stiffnesses; in saturation
   `K → 0`).
2. Form the 3×3 linear system for end-of-substep velocities:

   ```
   [ Mw + dt·K        coupling(r) ] [ u⁺ ]   [ Mw·u  + dt·(F₀ − K·u) terms ]
   [ coupling(r)      I + dt·Kᵣ   ] [ ω⁺ ] = [ I·ω + dt·(τ_drive − τ_resist) ]
   ```

   where the coupling row/column carries `F_long·r` between spin and `u_long`,
   `Kᵣ = K_long·r²`, and `τ_resist` collects brake, rolling resistance, and spin
   damping with the existing zero-crossing clamps.
3. **Stick test first (static friction):** compute the impulse `p_stick` that
   drives `u⁺ = 0` (and, when brakes lock the wheel, `ω⁺·r = v_c·t_fwd`). If
   `‖p_stick‖ ≤ μ·Fz·dt` (using a static μ, default equal to kinetic μ), take the
   stick solution and skip steps 4–5. Stick/slip is decided by the circle test,
   not a sign test — this is what eliminates the square wave.
4. Otherwise convert the slip solution to a tangential impulse
   `p_t = Mw·(u⁺ − u) − dt·(external tangential terms)` — equivalently, the
   impulse the tire must apply to produce `u⁺`.
5. **Project onto the friction circle:** if `‖p_t‖ > μ·Fz·dt`, scale `p_t` to the
   boundary and recompute `u⁺, ω⁺` consistently with the clamped impulse (one
   re-solve of the linear system with the force pinned at the boundary direction).
6. Apply `F = p_t/dt` at `c` on the **wheel body** (atomic add into
   `state.body_f`), preserving the steering-compliance path. Advance `ω ← ω⁺`.
7. Self-aligning moment `Mz = −F_lat·t·max(1 − utilization, 0)` computed from the
   *resolved* lateral force, applied about `n` as today.

At lock-up the direction of `p_t` comes from the velocity-level solve, so the
κ → −1 direction chatter cannot occur; the canonical-slip form survives only inside
the brush curve evaluation for the force *magnitude* shaping.

### 4.4 Normal load

- `latch_wheel_loads` keeps latching the solver-reported normal force (it is the
  only Fz source consistent with `condim = 1`), but:
  - the exponential `load_filter` smoothing is **removed** (default and option) —
    it existed to hide load-latch jitter on unsprung rigid bodies; sprung assets
    made loads determinate, and the impulse budget makes residual jitter benign;
  - `fz` **decays to zero when the patch is inactive** (airborne), so a landing
    cannot fire the tire with a stale airborne load;
  - `fallback_load` behavior on the first step is unchanged.
- One-substep staleness of Fz is accepted: because the tangential impulse is
  bounded by μ·Fz·dt, a stale Fz misestimates grip for one substep but cannot
  destabilize.

### 4.5 Removed and demoted mechanisms

- **Removed:** the low-speed lateral anti-overshoot cap (`lat_cap = fz/9.81·…`)
  and its config knob; the `load_filter` config knob. Their regression tests are
  replaced by the stronger acceptance tests in §7.
- **Demoted:** `min_reference_speed` survives only as the regularization inside
  the brush slip normalization (κ, α well-defined at rest); stability no longer
  depends on its value.
- **Optional, default off:** relaxation-length transient slip returns as an
  implicit state *inside* the solve (first-order lag on the slip input to the
  brush curve, integrated implicitly within the same 3×3 system). The 2026-06-16
  rejection showed relaxation is unsafe as a filter feeding an explicit loop; as
  part of an implicit loop it adds transient realism without the phase-lag hazard.
  Ship disabled (`relaxation_length_ratio = 0.0`) until validated against the
  acceptance suite.

### 4.6 Diagnostics

Existing per-wheel diagnostics (`kappa`, `alpha`, `f_long`, `f_lat`, `mz`,
`normal_load_used`) are kept. Added:

- `stick` flag (stick solution taken this substep),
- `impulse_utilization = ‖p_t‖ / (μ·Fz·dt)` (1.0 when the budget binds),

so tuning sessions can see saturation and stick/slip transitions directly in the
examples' telemetry HUD and live-tuning sliders (both retained).

## 5. Cleanup: retire the first-generation module

- **Delete** `newton/_src/wheeled/` (~4,860 LOC), `newton/wheeled.py`,
  `docs/api/newton_wheeled.rst`, the `newton/examples/wheeled/` examples, and the
  `newton/tests/test_wheeled_vehicle_*.py` suite. The branch is unreleased; no
  deprecation cycle is required. Update `README.md` example registrations
  accordingly.
- **Port into `_src/vehicles/` before deletion:**
  1. Manifest-ingestion and USD wheel-prim auto-detection helpers
     (`_src/wheeled/metadata.py`) — the new module currently only supports direct
     annotation.
  2. The analytic plane-cylinder footprint (`_src/wheeled/contact_patch.py`,
     sink-depth chord math) as a diagnostic patch-area source.
  3. The Fiala golden-curve tests (validated against an independent Python
     reference) and the ripple-terrain patch-stability sweep, retargeted at the
     new module. Porting the Fiala *model* itself is optional and out of scope.
- **Remove** the `PRESERVE_CONTACT_FOOTPRINT` shape flag and
  `ShapeConfig.preserve_contact_footprint`: it is set but read by no kernel
  (dead weight), and `gap = 0` obviated its patch-centering purpose. The
  narrow-phase diagnostic toggles (`enable_plane_cylinder_contact_collapse`,
  `enable_axial_contact_projection`) stay, and gain at least one non-vehicle test.
- Re-run `docs/generate_api.py`; update `CHANGELOG.md`.

## 6. Public API impact

- No new public modules. `newton.vehicles.WheeledVehicles` keeps its surface.
- `WheeledConfig` (or the nested `Config`): remove `load_filter` and the lateral
  cap knob; add `static_mu_scale` (default 1.0) and keep
  `relaxation_length_ratio` (default 0.0). Since the branch is unreleased, removal
  without deprecation is acceptable; if any of these shipped in a release, follow
  the standard deprecation policy instead.
- Docstrings follow the SI-unit convention; the new impulse-budget semantics are
  documented on the tire-model section of the vehicles docs page.

## 7. Validation and acceptance

All tests run on rc_car **and** husky, CPU and CUDA, `unittest`, at the examples'
timestep (60 fps × 8 substeps, dt = 1/480 s).

**Regime map (the scenarios that killed the last attempt), at μ ∈ {0.5, 1, 2, 2.5}:**

1. *Low-speed steer reversals:* aggressive full-lock steering reversals at
   < 1 m/s for 5 s. Assert max wheel vertical speed below a small bound (no hop),
   chassis roll bounded, all states finite.
2. *Hard braking from top speed:* full brake from ≥ 90% top speed. Assert speed
   decreases monotonically (small tolerance), lateral velocity and yaw rate stay
   bounded, vehicle stops and stays stopped.
3. *Slope hold (new capability):* place the vehicle at rest on a 15° incline with
   full brakes. Assert drift < 1 cm over 5 s (static friction holds; no creep).
4. *Steered launch:* full throttle + steering from rest; judge steady-state yaw
   rate (not peak) matches the Ackermann prediction within tolerance; no
   spin-out.
5. *Straight-line drive:* full throttle straight; lateral drift below the existing
   test's bound.

**Invariants (per-substep, checked in `test_post_step` of the examples and unit
tests):**

- ‖tangential impulse‖ ≤ μ·Fz·dt always (exact, by construction — assert it).
- The tire impulse never increases the contact-frame tangential kinetic energy of
  the wheel body (passivity check).
- Stick flag ⇒ slip velocity ≈ 0 next evaluation.

**Retained regressions:** golden brush-curve tests, friction-circle tests,
self-aligning-moment tests, force-at-ground-contact (no spurious yaw), Ackermann
identity, latched load ≈ m·g, gap-zero patch centering, heterogeneous
metadata/replication tests. The two band-aid regression tests
(`test_low_speed_lateral_force_capped_against_overshoot`, load-filter behavior)
are superseded by regime-map tests 1–3.

**Examples:** `vehicle_rc_car` and `vehicle_husky` keep their follow camera,
telemetry HUD, and live-tuning UI; rc_car's default μ is raised to a realistic
soft-compound value (target 2.0) once the acceptance suite passes at 2.5. Each
implements `test_final()` (and `test_post_step()` for the invariants) and stays
registered in the example test suite and `README.md`.

## 8. Risks and mitigations

- *Free-body effective mass is too soft when suspension is near its travel limit
  or steering drives are stiff* → transient under-grip. Mitigation: acceptance
  test 4 (steered launch) bounds the behavioral impact; if needed later, augment
  `Mw` with a joint-space correction — explicitly out of scope for v1.
- *Boundary re-solve (clamped-impulse case) introduces a second linear solve per
  wheel* → negligible cost (3×3, closed form) but must be branch-light for warp
  divergence; both paths are straight-line algebra.
- *Fz staleness on the very first contact substep* (fallback load) → bounded by
  the budget; test 1 starts from rest to cover it.
- *Deleting the old module breaks external users* → none exist (unreleased
  branch); `README.md`/docs updated in the same change.

## 9. Implementation phasing (for the plan)

1. New solve kernel behind the existing `apply` entry point, band-aids removed,
   unit tests for the 3×3 solve (stick, slip, clamp, lock-up cases) — TDD.
2. Load-latch changes (decay on inactive, remove filter) + invariant checks.
3. Regime-map acceptance suite; tune defaults (rc_car μ target 2.0).
4. Optional implicit relaxation length (default off) + its tests.
5. Cleanup phase: ports from `_src/wheeled/`, deletion, flag removal, docs,
   CHANGELOG, README.
