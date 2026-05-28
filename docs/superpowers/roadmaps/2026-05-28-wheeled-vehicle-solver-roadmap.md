# Wheeled Vehicle Solver Roadmap

## Purpose

Newton needs a wheeled-vehicle simulation layer that can run thousands of
vehicles in parallel without Python, NumPy, Torch, or per-vehicle runtime loops.
The first users are RC cars and small AGVs such as Clearpath Husky or Jackal.

The solver should be a thin layer over existing rigid-body solvers. It should
own wheel-ground force generation, drive-mode logic, motor and differential
models, and terrain sampling, while leaving rigid-body integration,
articulation constraints, and ordinary body collisions to the wrapped solver.

## Constraints

- Keep runtime work in Warp kernels and flat arrays.
- Support heterogeneous vehicles in the same model.
- Expose user-facing APIs only through public modules.
- Start with MuJoCo Warp as the wrapped rigid solver.
- Avoid adding required or optional dependencies.
- Keep wheel collision/friction separate from Newton rigid contacts until the
  interaction model is validated.

## Current Work

Phase 1 is specified in
`docs/superpowers/specs/2026-05-28-wheeled-vehicle-phase-1-design.md`.

The local files
`docs/superpowers/specs/2026-03-12-wheel-ground-contact-design.md` and
`docs/superpowers/plans/2026-03-12-wheel-ground-contact.md` assume an existing
`newton/_src/solvers/car` implementation. That implementation is not present in
this checkout, so this roadmap starts from the current Newton solver structure.

## Phase 0: Reference Assets

Goal: load and inspect representative assets before tuning the vehicle model.

Tasks:

- Add or document an Ackermann RC-car USD asset.
- Add or document a skid-steer AGV USD asset.
- Identify body, wheel, suspension, and steering joints in each asset.
- Decide how asset-authored metadata maps to Newton custom attributes.

Exit criteria:

- The assets can be loaded into `ModelBuilder`.
- Wheel bodies and relevant joints can be identified without hard-coded Python
  object references in the runtime path.

## Phase 1: Plane Wheel Contact Wrapper

Goal: prove the wrapper architecture and wheel support forces on a flat plane.

Scope:

- Add a public wheeled-vehicle solver wrapper over a rigid solver.
- Register `wheeled:*` custom attributes for wheel metadata.
- Build flat wheel arrays at solver initialization.
- Disable or ignore wheel shape collision for wheel-ground support.
- Compute vertical plane contact forces in Warp kernels.
- Accumulate per-body spatial wrenches and delegate to `SolverMuJoCo`.
- Test single-world and multi-world behavior.

Out of scope:

- Raycast terrain.
- Pacejka tire forces.
- Steering or drive modes.
- Motor power curves and differentials.
- USD schema polish.

Exit criteria:

- A simple multi-wheel vehicle remains supported above an XY plane under
  gravity.
- The runtime step path contains no Python loops over wheels or vehicles.
- Tests fail before implementation and pass after implementation.

## Phase 2: Basic Wheel Drive And Braking

Goal: add minimal longitudinal behavior while keeping the tire model simple.

Scope:

- Add per-wheel torque or target angular-speed control inputs.
- Track wheel angular velocity if it is not already represented by joints.
- Apply longitudinal traction and braking forces at the contact point.
- Use simple friction limits derived from wheel normal force and material
  friction coefficients.

Exit criteria:

- A vehicle accelerates forward and brakes on the plane.
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

## Phase 4: Raycast Terrain Contact

Goal: replace the plane-only contact query with terrain queries.

Scope:

- Use Newton raycast utilities for shape and terrain intersection.
- Store hit distance, normal, and shape/material data per wheel.
- Support mesh, heightfield, and primitive terrain where feasible.
- Compare direct raycast contact against possible Newton collision-pipeline
  alternatives for jitter, stability, and performance.

Exit criteria:

- Wheel contact works on non-planar terrain.
- Contact normal and material data feed the tire model.
- Performance remains suitable for thousands of vehicles.

## Phase 5: Tire And Powertrain Models

Goal: move from simple traction to realistic vehicle behavior.

Scope:

- Add lateral tire forces.
- Evaluate Pacejka or a simpler brush model as the first tire model.
- Add configurable motor curves.
- Add differential modules and AWD/FWD/RWD variants.
- Add aerodynamic drag.

Exit criteria:

- RC-car and AGV reference scenarios produce plausible motion.
- Tire and drive modules can be tested independently.

## Phase 6: Public API, Examples, And Docs

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

- For later phases, should tire and powertrain forces keep the phase-1
  `body_f` merge contract or move to additional namespaced diagnostics buffers?
- For vehicles with explicit suspension, should support forces continue to act
  on wheel bodies only or split across wheel and chassis bodies?
- How should suspension joints and analytical wheel contact share damping?
- Which material fields should seed tire friction parameters?
- When raycasts arrive, should terrain shape filtering reuse Newton collision
  groups or a separate `wheeled:*` mask?

