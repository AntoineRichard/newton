# Wheeled Vehicle Phase 1 Design

## Goal

Add the first wheeled-vehicle solver slice: a thin wrapper over an existing
rigid solver that observes Newton-generated wheel-ground contacts and estimates
per-wheel contact patch state. This phase proves the data model, batching,
contact grouping, material lookup, and solver wrapping before adding tire,
steering, drive, terrain, or hydroelastic complexity.

Phase 1 uses a flat reference scene for validation, but the contact source is
Newton's collision pipeline, not a custom plane query or raycast.

## Non-Goals

- Do not implement raycast terrain or raycast wheel contact.
- Do not implement an analytical plane-support force kernel.
- Do not implement any dedicated tire model, including Pacejka, Brush, or
  Fiala, or lateral/longitudinal tire forces.
- Do not implement Ackermann, skid-steer, motor curves, or differentials.
- Do not implement hydroelastic wheel-ground contacts. Hydroelastic contacts are
  a later study for improving patch quality on non-flat terrain.
- Do not add required or optional dependencies.
- Do not rely on imports from `newton._src` in examples or docs.
- Do not remove existing public API symbols.

## Public Shape

Add a public solver symbol named `SolverWheeledVehicle` in `newton.solvers`.
The class wraps another rigid solver instance or constructs one from the model.
Phase 1 should support `SolverMuJoCo` first, using Newton-generated contacts
with `use_mujoco_contacts=False`.

Example target usage:

```python
import newton
from newton.solvers import SolverMuJoCo, SolverWheeledVehicle

builder = newton.ModelBuilder()
SolverWheeledVehicle.register_custom_attributes(builder)

# Asset or builder code marks wheel shapes/bodies with wheeled:* attributes.
model = builder.finalize()
rigid_solver = SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=False)
solver = SolverWheeledVehicle(model, rigid_solver)
```

The final API can be adjusted during implementation if existing solver
construction patterns make another form cleaner, but examples and docs must use
public imports.

## Data Model

Register attributes in the `wheeled` namespace.

Per-shape model attributes:

- `wheeled:is_wheel`: `int32`, default `0`. Marks a shape as a wheel source.
- `wheeled:wheel_radius`: `float32`, default `0.0` [m]. If zero, the solver
  may infer radius from sphere, capsule, or cylinder shape scale.
- `wheeled:wheel_width`: `float32`, default `0.0` [m]. If zero, the solver may
  infer width from cylinder or capsule shape scale when available.

Control/state attributes are not required for phase 1. Later phases can add
namespaced control arrays for drive commands once the contact path is validated.

Phase 1 should not add suspension attributes. Assets with suspension should use
ordinary Newton joints and solver contact response. Later phases can decide
whether wheel-specific suspension tuning needs a namespaced API.

## Solver Initialization

At initialization, `SolverWheeledVehicle` scans model arrays once on the host to
build flat device arrays:

- `wheel_shape`: source shape index for each wheel.
- `wheel_body`: body associated with the wheel shape.
- `wheel_radius`: wheel radius [m].
- `wheel_width`: wheel width [m].
- `wheel_world`: world index for each wheel.
- `wheel_axis_local`: local wheel spin axis used to build the contact patch
  tangent frame.

This one-time setup may use Python and NumPy because it is outside the runtime
step path. The runtime `step()` and contact update paths must launch Warp
kernels over flat arrays.

Validation rules:

- A wheel shape must be attached to a dynamic or articulated body.
- A wheel radius must be positive after explicit value or inference.
- Wheel collision must be enabled for Phase 1 contact generation. Construction
  should warn or fail clearly if a marked wheel shape is a site or otherwise
  cannot participate in rigid contacts.
- If no wheels are marked, construction should fail with a clear `ValueError`.
- If the wheel spin axis cannot be inferred from the shape, construction should
  fail clearly or require an explicit future axis attribute before supporting
  that shape type.

## Runtime Flow

`SolverWheeledVehicle.step(state_in, state_out, control, contacts, dt)`:

1. Expect `contacts` to contain Newton-generated contacts for the current
   `state_in`. For the MuJoCo target, the wrapped solver should be configured
   with `use_mujoco_contacts=False`.
2. Zero internal per-wheel contact-state scratch arrays.
3. Launch a contact grouping kernel over `contacts.rigid_contact_count`.
4. Accumulate wheel-terrain contact samples into per-wheel scratch buffers.
5. Derive per-wheel contact location, normal, patch extents, patch area, and
   terrain material seed data.
6. Delegate normal support and integration to the wrapped rigid solver's
   `step()`.
7. Forward `update_contacts(contacts, state=None)` to the wrapped solver so
   later phases can use solver-reported contact forces when available.

Phase 1 does not add forces to `state_in.body_f`. Newton contacts and the
wrapped solver provide normal support. Tire forces are added in later phases
once the friction-ownership contract is explicit.

## Contact Patch Estimator

The contact grouping kernel should consider a rigid contact active for wheel
`w` when either `rigid_contact_shape0` or `rigid_contact_shape1` matches
`wheel_shape[w]`.

For each active wheel contact:

1. Transform `rigid_contact_point0` and `rigid_contact_point1` from body-local
   coordinates to world coordinates using the contacted bodies' transforms.
2. Compute the sample point as the midpoint between the two world-space surface
   points.
3. Orient the contact normal consistently so it points from terrain toward the
   wheel.
4. Build a tangent frame from the wheel spin axis and contact normal:
   longitudinal/rolling direction, lateral direction, and normal.
5. Project sample points into the tangent frame.
6. Accumulate sample count, weighted sample center, normal sum, tangent-frame
   min/max extents, counterpart terrain shape, and material values.

Patch location should be the weighted center of contact samples. Phase 1 can use
penetration-weighted samples when a penetration estimate is available, otherwise
it should fall back to an unweighted average. Later phases can replace this with
force weighting after solver force reporting is validated.

Patch area should be a conservative estimate from tangent-frame extents. A
simple bounding rectangle or covariance-based ellipse is acceptable for Phase 1
as long as the tests define the chosen estimator. Degenerate cases with fewer
than two meaningful samples should report a small or zero area explicitly rather
than inventing a high-confidence patch.

Material seed data should record the counterpart terrain shape and the relevant
shape material values, especially `shape_material_mu`, `shape_material_ke`, and
`shape_material_kd`. Phase 1 should not permanently map these values into a tire
model; it only makes the data available to later phases.

## Wrapped Solver Contract

Phase 1 targets `SolverMuJoCo`/MuJoCo Warp. For Newton-generated contacts, use
`SolverMuJoCo(model, use_mujoco_contacts=False)` and call `model.collide()` or a
configured `CollisionPipeline` before the wrapper step.

The wrapper should forward:

- `notify_model_changed(flags)` to the wrapped solver and refresh wheel arrays
  when shape, body, or model properties change.
- `update_contacts(contacts, state=None)` to the wrapped solver.
- `device` and `model` behavior through `SolverBase`.

The design should keep the wrapper generic enough to wrap other rigid solvers
that consume Newton `Contacts` and can provide normal support. Solver-specific
force reporting should be optional in Phase 1 and required only by later phases
that need normal load for tire forces.

## Solver Friction Ownership

Phase 1 observes contacts and does not apply tire friction. This avoids hiding a
friction double-counting problem behind early implementation choices.

Later phases must explicitly decide how wheel-pair solver friction is handled
before adding custom longitudinal or lateral tire forces. Candidate policies are:

- keep solver friction active until a tire model is enabled;
- reduce or zero solver friction for wheel-terrain pairs and apply wheeled tire
  friction separately;
- use solver friction for normal stabilization only where the wrapped solver
  exposes that distinction.

The Phase 1 implementation should make enough material and contact diagnostics
available to evaluate these policies.

## Hydroelastic Follow-Up

Hydroelastic contacts are not part of Phase 1. They should be studied later as a
quality improvement for non-flat terrain and compliant tire-ground pairs.

The later hydroelastic study should compare rigid contact-cloud patch estimates
against hydroelastic contact surface area and center-of-pressure data. The study
should decide whether hydroelastic support is worth the added setup cost and, if
so, keep it optional behind the same wheel-contact interface.

## Testing

Use `unittest`, not pytest.

Tests must be written before implementation and verified red first.

Kernel tests:

- Non-wheel contacts do not affect wheel contact state.
- A contact involving a marked wheel increments only that wheel's contact count.
- Contact points are transformed from body-local to world coordinates correctly.
- Contact normal is oriented from terrain toward wheel consistently regardless
  of shape order.
- Multiple samples produce the expected center and tangent-frame patch extents.
- Degenerate one-sample contacts report an explicit low-confidence or zero-area
  patch.
- Terrain shape and material values are captured for the contacted counterpart.

Solver tests:

- A simple wheel or wheeled body on a flat plane produces contact patch
  diagnostics through Newton contacts.
- The same fixture remains supported by `SolverMuJoCo` using
  `use_mujoco_contacts=False` and Newton-generated contacts.
- Two worlds with different vehicle positions run in one model without indexing
  cross-talk.
- Construction fails clearly when no wheels are marked.
- Construction fails or warns clearly for invalid wheel radius or non-colliding
  wheel shape configuration.

The solver-level tests should run through `uv run --extra dev -m newton.tests
-k test_wheeled_vehicle` once implemented.

## Error Handling

Raise `ValueError` for invalid static configuration:

- no marked wheels
- marked wheel shape has no attached body
- non-positive inferred radius
- unsupported wrapped solver if the wrapper cannot consume Newton contacts with
  it
- unsupported wheel shape when radius, width, or spin axis cannot be inferred

Warnings are acceptable for configuration that is suspicious but recoverable:

- wheel material friction may conflict with later custom tire friction
- contact force reporting is unavailable from the wrapped solver in Phase 1

## Documentation And Follow-Up

When public symbols are added:

- update `newton/solvers.py` and solver package exports
- add an `[Unreleased]` changelog entry under `Added`
- run `docs/generate_api.py`
- keep examples and docs on public imports only

The next implementation plan should cover only Phase 1. It should not include
drive modes, tire-model interfaces, tire forces, raycasts, hydroelastic contact
support, or reference asset work except for minimal test fixtures.
