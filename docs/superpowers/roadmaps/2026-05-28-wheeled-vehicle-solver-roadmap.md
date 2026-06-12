# Wheeled Vehicle Solver Roadmap

## Purpose

Newton needs a wheeled-vehicle simulation layer that can run thousands of
vehicles in parallel without Python, NumPy, Torch, or per-vehicle runtime loops.
The first users are RC cars and small AGVs such as Clearpath Husky or Jackal.

The solver should be a thin layer over existing rigid-body solvers. It should
own wheel-contact interpretation, tire and drive force generation, drive-mode
logic, motor and differential models, and diagnostics, while leaving rigid-body
integration, articulation constraints, collision detection, and ordinary body
collisions to Newton and the wrapped solver.

## Constraints

- Keep runtime work in Warp kernels and flat arrays.
- Support heterogeneous vehicles in the same model.
- Expose user-facing APIs only through public modules.
- Start with MuJoCo Warp as the wrapped rigid solver.
- Avoid adding required or optional dependencies.
- Use Newton's collision pipeline as the primary source of wheel-ground contact
  geometry. Do not build a raycast-based wheel-contact path in this roadmap.
- Keep normal-contact support and tire-friction ownership explicit. The rigid
  solver may own normal support, while the wheeled layer should be able to own
  wheel-specific longitudinal and lateral friction without double-counting
  solver friction.
- Keep tire models pluggable. Pacejka, Brush, Fiala, and simpler empirical
  models should be implementation choices behind a shared interface, not baked
  into the solver architecture.

## Current Work

The wheeled branch has working Phase 00 assets, wheel metadata loading, Newton
contact-patch extraction, basic drive/braking helpers, tire-force helpers,
vehicle command mapping, terrain-contact probes, analytical wheel moment
dynamics, hydroelastic/SDF contact studies, and an opt-in Fiala lateral tire
model. The controllable RC-car and Husky examples run through the public
`newton.wheeled` API while leaving suspension, steering, rigid integration, and
normal contact support with the main solver.

The active implementation focus is now Phase 7 powertrain and tire-model
quality, plus Phase 8 hardening. The immediate cleanup work is to keep roadmap
status, public API docs, README examples, changelog entries, and focused
regression tests aligned with the current branch before adding motor curves,
differentials, stronger calibration guidance, or additional tire formulations.

The local files
`docs/superpowers/specs/2026-03-12-wheel-ground-contact-design.md` and
`docs/superpowers/plans/2026-03-12-wheel-ground-contact.md` assume an existing
`newton/_src/solvers/car` implementation. That implementation is not present in
this checkout, so this roadmap starts from the current Newton solver structure
and the public `newton.wheeled` surface.

## Phase 00: Simplified USDA Fixtures

Plan: `docs/superpowers/plans/2026-06-02-wheeled-vehicle-phase-00-fixtures.md`

Goal: create simple, deterministic USDA reference vehicles before relying on
real robot assets.

Tasks:

- Create `newton/examples/assets/wheeled/husky.usda` as a box chassis with four
  cylinder wheels, two per side, no suspension, and no steering.
- Use Clearpath Husky A300-inspired values for the default Husky fixture:
  mass `80.0` kg, wheelbase `0.512` m, track width `0.566` m, wheel radius
  `0.1625` m, and approximate wheel width `0.13` m. Note that legacy Husky A200
  is lighter at `50.0` kg if a smaller fixture is needed later.
- Create `newton/examples/assets/wheeled/rc_car.usda` as a box chassis with four
  wheels, passive spring/damper suspension joints on all wheels, and front
  steering joints with centering drives and servo-like velocity limits.
- Use F1TENTH/RC-inspired starting values for the RC-car fixture: mass `4.0` kg,
  wheel radius `0.055` m, wheel width `0.045` m, wheelbase `0.324` m, and track
  width `0.296` m. Use `0.047` m center ground clearance from Traxxas Slash 4X4
  Ultimate-style specs, plus `0.05` m approximate suspension travel for the
  simplified fixture. Record in the manifest that these spacing values follow
  Traxxas Slash/F1TENTH-style references rather than the rough initial design
  note values of `0.40` m wheelbase and `0.20` m track width.
- Give all bodies, wheel collision shapes, suspension joints, and steering
  joints stable labels for the Phase 0 manifest/intake step.

Exit criteria:

- Both generated USDA files can be loaded into `ModelBuilder`.
- The fixtures expose wheel body, wheel shape, suspension, and steering labels
  without using high-poly meshes or real-robot asset dependencies.

## Phase 0: Fixture Intake And Inspection

Plan: `docs/superpowers/plans/2026-06-01-wheeled-vehicle-phase-0-assets.md`

Report: `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`

Goal: load and inspect the Phase 00 reference fixtures before tuning the
vehicle model.

Tasks:

- Add a manifest for the generated Ackermann RC-car and skid-steer Husky
  fixtures.
- Identify body, wheel, suspension, and steering labels in each fixture.
- Decide how fixture-authored metadata maps to Newton custom attributes.

Exit criteria:

- The generated assets can be loaded into `ModelBuilder`.
- Wheel bodies and relevant joints can be identified without hard-coded Python
  object references in the runtime path.

## Phase 1A: Wheel Metadata Loading

Plan: `docs/superpowers/plans/2026-06-02-wheeled-vehicle-phase-1a-metadata-loading.md`

Goal: prove that the wheeled layer can load wheel metadata from the Phase 0
fixtures and build flat wheel tables without doing any contact or collision
interpretation.

Scope:

- Add the minimal wheeled-vehicle metadata loader or solver wrapper needed to
  consume a `ModelBuilder`/`Model` plus wheel metadata.
- Register initial `wheeled:*` custom attributes for wheel metadata, including
  wheel shape/body identity, radius, width, and vehicle id. Use custom
  `wheeled:vehicle` and `wheeled:wheel` frequencies so `vehicle_id`,
  `wheel_id`, and `wheel_body_id` remain unique when template builders are
  replicated.
- Use the Phase 0 manifest labels as the runtime annotation source, and add
  separate metadata-bearing USDA test fixtures to exercise direct `wheeled:*`
  custom-attribute import.
- Resolve wheel body labels and wheel shape labels into model indices for the
  runtime annotation path, and validate equivalent indices from authored USDA
  attributes.
- Build flat wheel arrays at initialization for single-world and multi-world
  models.
- Leave steering and suspension joints to ordinary simulator dynamics and later
  control phases rather than adding joint fields to the wheeled metadata tables.
- Add diagnostics that report the wheel-to-shape, wheel-to-body, and
  wheel-to-vehicle mappings, including per-vehicle wheel counts.

Out of scope:

- Contact generation, contact grouping, contact patch estimation, or terrain
  material lookup.
- Raycast terrain or raycast wheel contact.
- Any dedicated tire model, including Pacejka, Brush, or Fiala.
- Steering commands, drive commands, motor curves, or differentials.
- Hydroelastic tire-ground contacts.
- Final USD schema polish.

Exit criteria:

- The RC-car and Husky fixtures load through both wheeled metadata paths:
  runtime annotation after import, and direct import from pre-authored USDA
  `wheeled:*` attributes.
- Wheel shapes and wheel bodies are resolved from the Phase 0 manifest labels
  without hard-coded runtime object references, and the pre-authored USDA path
  produces equivalent wheel identity and dimension tables.
- The metadata path works for single-world and multi-world model construction.
- The implementation contains no runtime collision/contact assumptions.
- Tests fail before implementation and pass after implementation.

## Phase 1B: Newton Contact Patch Wrapper

Spec: `docs/superpowers/specs/2026-06-03-wheeled-vehicle-phase-1b-contact-patch-wrapper.md`

Plan: `docs/superpowers/plans/2026-06-03-wheeled-vehicle-phase-1b-contact-patch-wrapper.md`

Goal: use the Phase 1A wheel tables to identify wheel-ground contacts from
Newton's collision pipeline and estimate per-wheel contact state on a flat
reference scene.

Scope:

- Keep wheel collision enabled for wheel-ground contact generation.
- Start with `SolverMuJoCo` using Newton-generated contacts
  (`use_mujoco_contacts=False`) so the contact geometry comes from Newton.
- Group active Newton contacts by Phase 1A wheel shape index and counterpart
  terrain shape.
- Estimate per-wheel contact location, normal, patch extents, and patch area
  from the contact cloud.
- Read terrain shape and material fields, including friction coefficients, as
  seeds for later wheel friction models.
- Delegate normal support to the wrapped rigid solver; do not add a separate
  analytical plane-support kernel.
- Test single-world and multi-world behavior.

Out of scope:

- Wheel metadata loading beyond the Phase 1A contract.
- Raycast terrain or raycast wheel contact.
- Any dedicated tire model, including Pacejka, Brush, or Fiala.
- Steering or drive modes.
- Motor power curves and differentials.
- Hydroelastic tire-ground contacts.
- USD schema polish.

Exit criteria:

- A wheel-ground fixture produces stable contact location, normal, terrain
  shape, material, and patch-area diagnostics from Newton contacts.
- A simple wheeled body remains supported by the wrapped rigid solver using
  Newton-generated contacts.
- The runtime contact grouping and patch estimation path contains no Python
  loops over wheels, vehicles, or contacts.
- Tests fail before implementation and pass after implementation.

## Phase 2: Basic Wheel Drive And Braking

Spec: `docs/superpowers/specs/2026-06-03-wheeled-vehicle-phase-2-basic-drive-braking.md`

Plan: `docs/superpowers/plans/2026-06-03-wheeled-vehicle-phase-2-basic-drive-braking.md`

Goal: add minimal longitudinal behavior using the Phase 1B contact patch state.

Scope:

- Add per-wheel torque or target angular-speed control inputs.
- Track wheel angular velocity if it is not already represented by joints.
- Estimate normal load from reported solver contact forces when available, with
  a penetration/stiffness fallback when a solver cannot report forces.
- Apply longitudinal traction and braking forces at the estimated contact patch
  center.
- Use simple friction limits derived from wheel normal force and material
  friction coefficients.
- Make wheel-pair solver friction handling explicit so the solver and wheeled
  layer do not both apply the same tire friction.
- Provide examples that simulate the Phase 00 assets, including one example that
  drives joints through MuJoCo contact friction and one example that applies the
  wheeled force helper at the contact patch.

Exit criteria:

- A vehicle accelerates forward and brakes on the flat reference scene.
- A skid-steer vehicle can rotate in place from opposite wheel commands.
- Forces remain batched and device-side.
- The contact-patch force example keeps solver wheel friction explicit and
  applies matching axle reaction torque so contact-patch forces do not create
  meaningless wheel spin.

## Phase 3: Tire Model Interface And First Tire Force Model

Spec: `docs/superpowers/specs/2026-06-09-wheeled-vehicle-phase-3-tire-model-interface.md`

Plan: `docs/superpowers/plans/2026-06-09-wheeled-vehicle-phase-3-tire-model-interface.md`

Goal: replace the simple longitudinal-only force law with a modular tire-model
layer that produces longitudinal and lateral tire forces from Phase 1B contact
patches without committing the solver to Pacejka, Brush, Fiala, or any other
single formulation.

Scope:

- Draft the Phase 3 spec and plan before implementation.
- Define a tire-model input/output contract over flat wheel arrays. The inputs
  should include patch activity, patch center, patch normal, patch area,
  material friction seed, normal load, wheel radius, forward/axle axes,
  analytical wheel angular speed, contact-point velocity, longitudinal slip, and
  lateral slip or slip angle.
- Add separate `WheelTireControl` and `WheelTireState` objects so the Phase 2
  simple drive API remains intact. Keep public names exported through
  `newton.wheeled` and avoid importing `newton._src` from examples or docs.
- Implement a saturated linear velocity-slip tire model behind the shared
  interface before adding tuned Pacejka coefficients. Pacejka, Brush, and Fiala
  should remain candidate implementations behind the interface.
- Support combined-slip limiting so longitudinal and lateral forces share the
  available `mu * normal_load` budget instead of clipping independently.
- Apply tire forces at the Phase 1B patch center through `State.body_f`. Wheel
  spin is analytical for this phase; physical wheel spin joints should be
  locked, omitted, or visual-only.
- Treat suspension and steering as ordinary main-solver joints. The tire layer
  consumes the resulting wheel body pose and does not own those dynamics.
- Keep normal support with the wrapped rigid solver. Phase 3 still owns
  wheel-specific tire friction; tests and examples should set wheel-ground
  contacts to normal-only where supported so solver friction is not the hidden
  tire model.
- Keep runtime work batched and device-side. Model selection may be configured at
  setup time, but the simulation loop must not iterate over vehicles or wheels
  in Python.

Out of scope:

- Suspension dynamics, suspension control, Ackermann steering command logic,
  skid-steer command mapping, or drive-mode abstractions beyond per-wheel test
  commands.
- Motor power curves, gearboxes, differentials, AWD/FWD/RWD policy, and battery
  models.
- Non-flat terrain validation beyond simple flat/slope fixtures needed to test
  tire-force directions.
- Hydroelastic contact implementation.
- Tuning Pacejka, Brush, or Fiala to match a specific physical tire. Phase 3 may
  evaluate their data requirements, but it should not make any of them the only
  tire path.

Exit criteria:

- A tire-model API can compute per-wheel longitudinal and lateral forces from
  `WheelContactPatchState` and wheel kinematics in flat arrays.
- Unit tests cover inactive patches, zero/invalid normal load, pure
  longitudinal slip, pure lateral slip, combined-slip saturation, material
  friction seeding, replicated metadata, and force/wrench accumulation at an
  off-COM patch point.
- A flat-scene example demonstrates forward/back motion with the new tire-model
  interface and shows lateral tire force behavior without relying on full solver
  wheel friction or unlocked physical wheel spin.
- The implementation leaves room for Pacejka, Brush, Fiala, and simpler
  empirical models without changing the solver wrapper contract.

## Phase 4: Vehicle Geometry And Drive Mapping

Plan: `docs/superpowers/plans/2026-06-10-wheeled-vehicle-phase-4-vehicle-geometry-drive-mapping.md`

Goal: provide common vehicle-geometry/layout and actuator-mapping abstractions
once the wheeled layer can generate lateral tire forces.

Scope:

- Add Ackermann-style vehicle geometry mapping for assets whose front steering
  joints are ordinary simulator joints.
- Add skid-steer side-based drive command mapping.
- Add per-wheel driven and steerable flags or command masks without folding
  steering/suspension topology into Phase 1A wheel metadata.
- Keep geometry/layout kernels modular so different vehicle shapes can coexist.
- Use the Phase 3 tire-model interface for lateral and longitudinal force
  generation rather than adding steering-specific tire force code.

Exit criteria:

- An Ackermann vehicle responds to normalized drive/steering commands on the
  flat reference scene.
- A skid-steer vehicle responds to normalized left/right drive commands.
- Heterogeneous vehicles can run in one model without host-side branching in
  the runtime loop.

## Phase 5: Newton Collision Terrain Contact

Spec: `docs/superpowers/specs/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md`

Plan: `docs/superpowers/plans/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md`

Reports:

- `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-gap-line-patch-generation.md`
- `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-contact-observability.md`

Goal: validate the contact patch estimator and tire-model inputs on non-flat
terrain using Newton's collision pipeline.

Scope:

- Use Newton collision contacts for mesh and primitive terrain
  where feasible.
- Use wheel-terrain `shape_gap = 0.0` as the baseline setup to test, and keep
  margin behavior visible as a separate variable.
- Store per-wheel contact shape, normal, point cloud, patch extents, area
  estimate, and material data.
- Check whether cylinder tire contacts are point-like, line-like, or area-like,
  and identify whether any line-like behavior comes from geometry, contact
  reduction, or forced axial rolling stabilization.
- Add an optional regular contact-patch mode that replaces active
  wheel-cylinder/analytic-plane contact-cloud extents with the closed-form
  cylinder-plane footprint for flat or locally planar terrain comparisons.
- If forced cylinder contact line alignment hurts wheel patch quality, design a
  scoped wheel-terrain opt-out rather than changing all cylinder contacts.
- Study contact reduction, contact matching, deterministic sorting, and solver
  force reporting for wheel-terrain pairs.
- Compare Newton-generated contacts converted into MuJoCo Warp against
  MuJoCo-native contacts only as an observability and stability study.
- Identify jitter, sparse-contact, and performance limits after the first tire
  model exists, so any contact-quality issues can be measured through tire-force
  outputs.

Exit criteria:

- Wheel contact state works on representative non-planar terrain without
  raycasts.
- The regular contact-patch path can opt into an analytical cylinder-plane
  footprint and compare it against the raw rigid contact cloud.
- Contact normal, patch estimate, and material data feed the tire model.
- Performance remains suitable for thousands of vehicles.

## Phase 5A: Cylinder Contact Projection Diagnostics

Report: `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5a-cylinder-contact-projection-diagnostics.md`

Goal: determine whether axial rolling stabilization or the analytical
plane-cylinder primitive path is hiding useful wheel-cylinder contact spread
before changing the tire model or collision defaults.

Scope:

- Add an internal diagnostic path that compares ordinary cylinder contacts
  against contacts generated without axial rolling projection for wheel-cylinder
  versus box, mesh, ridge, and jump terrain cases.
- Treat plane-cylinder contacts separately by comparing the analytical
  plane-cylinder primitive path against a GJK/MPR diagnostic path with
  plane-cylinder contact collapse disabled.
- Record pre/post projection differences in contact count, patch center,
  tangent extents, patch area, normal stability, and tire-force output
  stability.
- Keep the default collision path unchanged while the diagnostic runs. If the
  no-projection or no-collapse path improves wheel patch quality, design a scoped
  wheel-terrain contact mode rather than changing all cylinder/cone contacts.
- Avoid adding raw and projected contacts to the core public `Contacts`
  structure unless the diagnostic proves that downstream users need both.

Exit criteria:

- A focused report shows whether axial rolling projection materially affects
  wheel patch quality on representative discrete terrain.
- Tests cover at least one primitive obstacle, one triangle-mesh terrain case,
  and one flat plane-cylinder case with ordinary and diagnostic contact
  generation.
- The roadmap records whether the next action is no change, a wheeled-only
  diagnostic helper, or a scoped wheel-terrain collision mode.

## Phase 5B: Simulated Wheel Moment Dynamics

Goal: simulate wheel rotational moments analytically now that physical wheel
spin joints are locked, omitted, or visual-only.

Scope:

- Add per-wheel analytical rotational state for angular speed, angular
  acceleration, drive torque, brake torque, tire reaction torque, and wheel
  inertia.
- Integrate wheel angular speed device-side from
  `inertia * angular_acceleration = drive_torque - brake_torque -
  longitudinal_tire_force * radius - damping_or_rolling_resistance`.
- Feed the integrated analytical wheel angular speed into the tire model instead
  of treating `wheel_angular_speed` only as an externally prescribed command.
- Preserve a direct analytical speed mode for diagnostics and simple examples,
  but make torque-driven wheel moments the realistic vehicle path.
- Account for equal-and-opposite axle/body reaction moments needed when tire
  forces are applied at the patch while the visible wheel body is not physically
  spinning.
- Keep the implementation batched and device-side; do not introduce per-vehicle
  Python control loops.

Out of scope:

- Detailed motor curves, gearboxes, differentials, and battery models; those
  stay in Phase 7.
- Visual wheel spin synchronization; visual-only wheel rotation can be added
  later on top of the analytical state.
- Re-enabling physical axle spin as the default vehicle model.

Exit criteria:

- Torque and brake commands evolve per-wheel analytical angular speed from wheel
  inertia and tire reaction torque.
- Locked or fixed wheel bodies can drive vehicles without losing the rotational
  torque balance that physical spinning wheels would normally carry.
- Tests cover free spin-up, braking, tire-force reaction torque, inactive
  contacts, and replicated multi-vehicle batches.

## Phase 6: Hydroelastic/SDF Contact Patch Study

Goal: evaluate hydroelastic/SDF contacts as a higher-quality patch source for
non-flat terrain and compliant tire-ground pairs.

Current study outcome:

- Initial report:
  `docs/superpowers/reports/2026-06-11-wheeled-vehicle-phase-6-hydroelastic-sdf-contact-study.md`.
- Hydroelastic/SDF contacts are promising as an optional patch source. In a
  cylinder-over-volumetric-box probe, the hydroelastic surface produced a
  sink-depth-dependent patch area, while the rigid cylinder path remained a
  sparse line-like contact set.
- Keep rigid-contact patch extraction as the default path. Hydroelastic finite
  terrain still requires volumetric SDF shapes and CUDA. Phase 6A adds analytic
  infinite-plane support for plane + SDF hydroelastic pairs, but heightfields
  remain unsupported as hydroelastic shapes.
- Use `gap=0` as the first tire-patch study setting so the wheel can sink into
  the terrain before a patch is measured. Positive gaps can generate shallow
  margin contacts at exact touch.

Phase 6A: Analytic Infinite-Plane Hydro Contacts

Goal: make flat-ground hydroelastic probes possible without modeling the ground
as a volumetric box.

Scope:

- Allow hydroelastic planes to remain analytic, with no texture SDF allocation.
- Route hydroelastic plane + SDF-backed shape pairs through the hydroelastic
  pipeline while keeping the finite shape as the sampled voxel domain.
- Generate contact surfaces by sampling the plane signed distance analytically
  during iso-voxel refinement and marching cubes.
- Keep plane-plane and heightfield hydroelastic contacts out of scope.

Exit criteria:

- A hydroelastic plane-cylinder test produces a finite 2D patch footprint,
  routes exactly one pair through the hydroelastic pipeline, and compares the
  measured extents and area against the closed-form cylinder-plane footprint.
- Existing SDF-SDF hydroelastic contact paths continue to use the same textured
  shape ordering and reduction flow.

Scope:

- Study Newton hydroelastic/SDF contact surfaces and their area-weighted
  internal data.
- Compare rigid contact-cloud area estimates against hydroelastic contact
  surfaces on slopes, uneven terrain, and curved wheel surfaces.
- Decide whether tire-ground pairs should opt into hydroelastic geometry, keep a
  rigid-contact estimator, or support both behind one wheel-contact interface.
- Keep the study optional and dependency-free.
- Prefer volumetric terrain fixtures for this phase: thin boxes, ramp boxes,
  obstacle boxes, and watertight non-flat meshes. Do not spend this phase on
  heightfield support.

Follow-up tasks:

- Build a non-example probe/test utility for hydroelastic wheel-over-terrain
  contact measurements.
- Extract hydro surface patch aggregates: centroid, area, projected extents,
  average normal, depth range, and area-weighted normal-load proxy.
- Compare hydroelastic patch aggregates against the rigid patch estimator on
  flat analytic planes, volumetric terrain, bumps/steps, ramps, and watertight
  non-flat meshes.
- Verify hydroelastic stiffness tuning guidance before exposing it to users; the
  code path and docs should agree on the effective stiffness formula.

Exit criteria:

- The roadmap records whether hydroelastic contacts improve wheel patch quality
  enough to justify the added setup cost.
- Any follow-up implementation plan keeps hydroelastic support optional and
  compatible with the rigid contact path.

## Phase 7: Powertrain And Higher-Fidelity Tire Models

Goal: move from the first tire-force model to richer vehicle behavior without
making a specific tire formulation exclusive.

Scope:

- Add configurable motor curves.
- Add differential modules and AWD/FWD/RWD variants.
- Add aerodynamic drag.
- Add additional tire-model implementations, likely including Brush/Fiala and
  Pacejka-style empirical curves where the parameter data is available.
- Add calibration docs that explain which parameters are physical, which are
  empirical, and how material friction seeds map into each model.

Exit criteria:

- RC-car and AGV reference scenarios produce plausible motion under configured
  powertrain and tire choices.
- Tire-model choice is configurable without changing the solver wrapper.
- Tire and drive modules can be tested independently.

### Phase 7A: Brush/Fiala Lateral Tire Model

Status: implemented as an opt-in tire-model selector on
`WheelTireControl.tire_model`. The saturated-linear model remains the default.

Notes:

- `configure_wheel_tire_control(tire_model=...)` accepts model ids and aliases
  such as `"linear"`, `"saturated_linear"`, `"fiala"`, and `"brush"`.
- The first Fiala branch models lateral force only. Longitudinal force still uses
  the existing saturated velocity-slip model, and the combined force vector still
  passes through the existing Coulomb force-circle limiter.
- `lateral_stiffness` is model-dependent: saturated-linear uses [N/(m/s)], while
  Fiala uses cornering stiffness [N/rad]. Calibration docs should make this unit
  change explicit before broader user-facing examples rely on tuned values.
- `wheeled_car_control --tire-model fiala` provides the first controllable
  Ackermann-car demo path for the Fiala lateral model.

## Phase 8: Public API, Examples, And Docs Hardening

Goal: make the wheeled layer robust and understandable for Newton users.

Scope:

- Add public exports through stable modules as each phase introduces public
  behavior.
- Add examples following the Newton `Example` class format.
- Register examples in `README.md`.
- Add API docs and changelog entries for user-facing behavior.
- Run `docs/generate_api.py` when public symbols are added.

Exit criteria:

- Users can configure wheeled vehicles without importing `newton._src`.
- Examples include final-state validation and screenshots.
- Documentation explains supported assumptions and limitations.

## Open Questions

- Which tire model should follow the first Fiala lateral branch: longitudinal
  brush behavior, combined-slip Fiala, or a minimal Pacejka-style curve with
  placeholder coefficients?
- How should `WheelContactPatchState.patch_area` influence the first tire model,
  if at all, before hydroelastic contacts are studied?
- How should terrain material fields seed tire parameters beyond scalar `mu`?
- Should tire and powertrain forces keep a `body_f` merge contract or move to
  additional namespaced diagnostics buffers?
- Should Phase 3 apply tire forces only to wheel bodies and rely on main-solver
  suspension/steering constraints to transmit reactions, or should a later phase
  study force distribution to chassis bodies for conditioning?
- Should wheel-pair solver friction be set to a tiny value by default in tire
  examples, or should examples expose this explicitly through setup code?
- Should patch area come from the geometric contact cloud, force-weighted
  contacts, penetration-weighted contacts, or hydroelastic contact surfaces
  when available?
- What force-reporting path should be required for MuJoCo Warp when Newton
  contacts are converted into MuJoCo contacts?
