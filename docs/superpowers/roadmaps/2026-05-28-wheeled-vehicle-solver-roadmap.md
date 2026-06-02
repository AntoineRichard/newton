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

Phase 1 needs a fresh spec and implementation plan based on Newton collision
contacts and contact patch estimation. The previous Phase 1 spec was removed so
this work can restart from the revised collision-first direction.

The local files
`docs/superpowers/specs/2026-03-12-wheel-ground-contact-design.md` and
`docs/superpowers/plans/2026-03-12-wheel-ground-contact.md` assume an existing
`newton/_src/solvers/car` implementation. That implementation is not present in
this checkout, so this roadmap starts from the current Newton solver structure.

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
  wheels, simple suspension joints on all wheels, and front steering joints.
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

## Phase 1: Newton Contact Patch Wrapper

Goal: prove that the wheeled layer can identify wheel-ground contacts from
Newton's collision pipeline and estimate per-wheel contact state on a flat
reference scene.

Scope:

- Add a public wheeled-vehicle solver wrapper over a rigid solver.
- Register `wheeled:*` custom attributes for wheel metadata.
- Build flat wheel arrays at solver initialization.
- Keep wheel collision enabled for wheel-ground contact generation.
- Start with `SolverMuJoCo` using Newton-generated contacts
  (`use_mujoco_contacts=False`) so the contact geometry comes from Newton.
- Group active Newton contacts by wheel shape and counterpart terrain shape.
- Estimate per-wheel contact location, normal, patch extents, and patch area
  from the contact cloud.
- Read terrain shape and material fields, including friction coefficients, as
  seeds for later wheel friction models.
- Delegate normal support to the wrapped rigid solver; do not add a separate
  analytical plane-support kernel.
- Test single-world and multi-world behavior.

Out of scope:

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

Goal: add minimal longitudinal behavior using the Phase 1 contact patch state.

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

Exit criteria:

- A vehicle accelerates forward and brakes on the flat reference scene.
- A skid-steer vehicle can rotate in place from opposite wheel commands.
- Forces remain batched and device-side.

## Phase 3: Steering And Drive Modes

Goal: provide common control abstractions without locking the data model to one
vehicle topology.

Scope:

- Add Ackermann-style front steering.
- Add skid-steer differential commands.
- Add per-wheel driven and steerable flags.
- Keep drive-mode kernels modular so vehicles with different modes can coexist.

Exit criteria:

- An Ackermann vehicle follows steering commands.
- A skid-steer vehicle follows differential commands.
- Heterogeneous vehicles can run in one model without host-side branching in
  the runtime loop.

## Phase 4: Newton Collision Terrain Contact

Goal: validate the contact patch estimator on non-flat terrain using Newton's
collision pipeline.

Scope:

- Use Newton collision contacts for mesh, heightfield, and primitive terrain
  where feasible.
- Store per-wheel contact shape, normal, point cloud, patch extents, area
  estimate, and material data.
- Study contact reduction, contact matching, deterministic sorting, and solver
  force reporting for wheel-terrain pairs.
- Compare Newton-generated contacts converted into MuJoCo Warp against
  MuJoCo-native contacts only as an observability and stability study.
- Identify jitter, sparse-contact, and performance limits before adding more
  tire-model complexity.

Exit criteria:

- Wheel contact state works on representative non-planar terrain without
  raycasts.
- Contact normal, patch estimate, and material data feed the wheel model.
- Performance remains suitable for thousands of vehicles.

## Phase 5: Hydroelastic Contact Patch Study

Goal: evaluate hydroelastic contacts as a higher-quality patch source for
non-flat terrain and compliant tire-ground pairs.

Scope:

- Study Newton hydroelastic/SDF contact surfaces and their area-weighted
  internal data.
- Compare rigid contact-cloud area estimates against hydroelastic contact
  surfaces on slopes, uneven terrain, and curved wheel surfaces.
- Decide whether tire-ground pairs should opt into hydroelastic geometry, keep a
  rigid-contact estimator, or support both behind one wheel-contact interface.
- Keep the study optional and dependency-free.

Exit criteria:

- The roadmap records whether hydroelastic contacts improve wheel patch quality
  enough to justify the added setup cost.
- Any follow-up implementation plan keeps hydroelastic support optional and
  compatible with the rigid contact path.

## Phase 6: Tire Model Interface And Powertrain Models

Goal: move from simple traction to realistic vehicle behavior without making a
specific tire formulation exclusive.

Scope:

- Define a modular tire-model interface for normal, longitudinal, and lateral
  contact state.
- Implement one first tire model behind that interface.
- Evaluate Pacejka, Brush, Fiala, and simpler empirical models as candidates.
- Add configurable motor curves.
- Add differential modules and AWD/FWD/RWD variants.
- Add aerodynamic drag.

Exit criteria:

- RC-car and AGV reference scenarios produce plausible motion.
- Tire-model choice is configurable without changing the solver wrapper.
- Tire and drive modules can be tested independently.

## Phase 7: Public API, Examples, And Docs

Goal: make the solver usable by Newton users.

Scope:

- Add public exports through stable modules.
- Add examples following the Newton `Example` class format.
- Register examples in `README.md`.
- Add API docs and changelog entries for user-facing behavior.
- Run `docs/generate_api.py` when public symbols are added.

Exit criteria:

- Users can configure wheeled vehicles without importing `newton._src`.
- Examples include final-state validation and screenshots.
- Documentation explains supported assumptions and limitations.

## Open Questions

- For later phases, should tire and powertrain forces keep a `body_f` merge
  contract or move to additional namespaced diagnostics buffers?
- For vehicles with explicit suspension, should support forces continue to act
  on wheel bodies only or split across wheel and chassis bodies?
- How should suspension joints and Newton contact damping share normal-contact
  behavior?
- Which material fields should seed shared tire-model parameters?
- Should wheel-pair solver friction be set low/zero before applying custom tire
  friction, or should the wheeled layer initially reuse solver friction until a
  tire model is active?
- Should patch area come from the geometric contact cloud, force-weighted
  contacts, penetration-weighted contacts, or hydroelastic contact surfaces
  when available?
- What force-reporting path should be required for MuJoCo Warp when Newton
  contacts are converted into MuJoCo contacts?
