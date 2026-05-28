# Wheeled Vehicle Phase 1 Design

## Goal

Add the first wheeled-vehicle solver slice: a thin wrapper over an existing
rigid solver that computes wheel support forces against a fixed XY ground plane.
This phase proves the data model, batching, solver wrapping, and force
application path before adding tire, steering, drive, or terrain complexity.

## Non-Goals

- Do not implement Pacejka or lateral/longitudinal tire forces.
- Do not implement raycast terrain.
- Do not implement Ackermann, skid-steer, motor curves, or differentials.
- Do not add required or optional dependencies.
- Do not rely on imports from `newton._src` in examples or docs.
- Do not remove existing public API symbols.

## Public Shape

Add a public solver symbol named `SolverWheeledVehicle` in `newton.solvers`.
The class wraps another rigid solver instance or constructs one from the model.
Phase 1 should support `SolverMuJoCo` first.

Example target usage:

```python
import newton
from newton.solvers import SolverMuJoCo, SolverWheeledVehicle

builder = newton.ModelBuilder()
SolverWheeledVehicle.register_custom_attributes(builder)

# Asset or builder code marks wheel shapes/bodies with wheeled:* attributes.
model = builder.finalize()
rigid_solver = SolverMuJoCo(model, use_mujoco_cpu=False)
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
- `wheeled:suspension_rest_length`: `float32`, default `0.0` [m].
- `wheeled:suspension_ke`: `float32`, default `2500.0` [N/m].
- `wheeled:suspension_kd`: `float32`, default `100.0` [N*s/m].

Per-world model attributes:

- `wheeled:ground_altitude`: `float32`, default `0.0` [m]. Plane height along
  world Z for phase 1.

Control/state attributes are not required for phase 1. Later phases can add
namespaced control arrays for drive commands once the force path is validated.

## Solver Initialization

At initialization, `SolverWheeledVehicle` scans model arrays once on the host to
build flat device arrays:

- `wheel_shape`: source shape index for each wheel.
- `wheel_body`: body receiving the contact wrench.
- `wheel_radius`: wheel radius [m].
- `wheel_rest_length`: suspension rest length [m].
- `wheel_ke`: normal stiffness [N/m].
- `wheel_kd`: normal damping [N*s/m].
- `wheel_world`: world index for each wheel.

This one-time setup may use Python and NumPy because it is outside the runtime
step path. The runtime `step()` path must launch Warp kernels over these arrays.

Validation rules:

- A wheel shape must be attached to a dynamic or articulated body.
- A wheel radius must be positive after explicit value or inference.
- Wheel shape collisions should be disabled by user configuration or ignored by
  the wheeled support model. Phase 1 should warn, not mutate model collision
  flags silently, if a marked wheel shape is still collidable.
- If no wheels are marked, construction should fail with a clear `ValueError`.

## Runtime Flow

`SolverWheeledVehicle.step(state_in, state_out, control, contacts, dt)`:

1. Zero an internal `body_f_wheeled` scratch array.
2. Launch a wheel plane-contact kernel with one thread per wheel.
3. Accumulate spatial wrenches into `body_f_wheeled`.
4. Merge `body_f_wheeled` into `state_in.body_f`.
5. Delegate to the wrapped rigid solver's `step()`.

The wrapper must not clear user-provided forces. It only adds its wheel support
forces to `state_in.body_f`, matching Newton's existing force accumulation
convention. Users remain responsible for calling `state.clear_forces()` at the
same point in the simulation loop where they would clear ordinary body forces.
The scratch array prevents stale wheel forces from accumulating inside the
wrapper across substeps.

## Plane Contact Model

Phase 1 uses a fixed plane with normal `+Z` and altitude
`model.wheeled.ground_altitude[world]`.

For each wheel:

```text
wheel_center = body pose transformed by wheel shape local transform
bottom_height = wheel_center.z - wheel_radius
compression = ground_altitude + suspension_rest_length - bottom_height
normal_speed = dot(contact point velocity, +Z)
normal_force = wheel_ke * compression + wheel_kd * max(-normal_speed, 0)
```

If `compression <= 0`, no force is applied.

The force is applied at the wheel contact point as a spatial wrench on
`wheel_body`, in world frame and referenced to that body's center of mass. This
means the kernel must add both:

- linear force `F = normal_force * +Z` [N]
- torque `tau = cross(contact_point - body_com_world, F)` [N*m]

This is more important than a center-of-mass-only upward force because the
wrapper needs to support pitch and roll from per-wheel compression.

## Wrapped Solver Contract

Phase 1 targets `SolverMuJoCo`/MuJoCo Warp. The design should keep the wrapper
generic enough to wrap other rigid solvers that honor `state.body_f`.

The wrapper should forward:

- `notify_model_changed(flags)` to the wrapped solver and refresh wheel arrays
  when shape, body, or model properties change.
- `update_contacts(contacts, state=None)` to the wrapped solver.
- `device` and `model` behavior through `SolverBase`.

## Testing

Use `unittest`, not pytest.

Tests must be written before implementation and verified red first.

Kernel tests:

- Wheel above plane produces zero force.
- Wheel exactly touching the plane produces zero force.
- Wheel below plane produces upward force proportional to compression.
- Downward velocity adds damping.
- Upward velocity does not add damping.
- Off-center wheel force produces torque on the body.

Solver tests:

- A simple four-wheel body under gravity remains supported above the plane.
- Two worlds with different vehicle positions run in one model without indexing
  cross-talk.
- Construction fails clearly when no wheels are marked.
- Construction fails or warns clearly for invalid wheel radius.

The solver-level tests should run through `uv run --extra dev -m newton.tests
-k test_wheeled_vehicle` once implemented.

## Error Handling

Raise `ValueError` for invalid static configuration:

- no marked wheels
- marked wheel shape has no attached body
- non-positive inferred radius
- unsupported wrapped solver if the wrapper cannot apply body forces to it

Warnings are acceptable for configuration that is suspicious but recoverable:

- marked wheel shape remains collidable
- suspension rest length is zero

## Documentation And Follow-Up

When public symbols are added:

- update `newton/solvers.py` and solver package exports
- add an `[Unreleased]` changelog entry under `Added`
- run `docs/generate_api.py`
- keep examples and docs on public imports only

The next implementation plan should cover only phase 1. It should not include
drive modes, tire forces, raycasts, or reference asset work except for minimal
test fixtures.

