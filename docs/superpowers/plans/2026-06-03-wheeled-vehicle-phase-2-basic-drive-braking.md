# Wheeled Vehicle Phase 2 Basic Drive And Braking Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add basic longitudinal drive and braking force application on top of
Phase 1B wheel contact patches while keeping normal support with the wrapped
rigid solver.

**Done when:** A flat reference fixture can accelerate forward, brake, and show
skid-steer-style yaw from opposite wheel commands using device-side wheeled
force kernels. The implementation must make normal-load and solver-friction
ownership explicit.

**Scope:** Phase 2 applies simple Coulomb-limited longitudinal wheel forces. It
does not add steering commands, drive modes, differentials, motor curves,
raycasts, or full tire models.

---

## Inputs

Spec:

- `docs/superpowers/specs/2026-06-03-wheeled-vehicle-phase-2-basic-drive-braking.md`

Implementation inputs:

- `newton.wheeled.WheeledModelMetadata`
- `newton.wheeled.WheelContactPatchState`
- `newton.State.body_f`
- `Model.body_com`
- `Model.shape_body`
- `State.body_q`
- `State.body_qd`
- `WheelContactPatchState.center`
- `WheelContactPatchState.normal`
- `WheelContactPatchState.normal_force`
- `WheelContactPatchState.friction_mu_seed`

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/_src/wheeled/drive.py` | Create | Drive/brake control/state objects, kernels, and update helpers |
| `newton/_src/wheeled/__init__.py` | Modify | Internal exports |
| `newton/wheeled.py` | Modify | Public Phase 2 import surface |
| `newton/tests/test_wheeled_vehicle_drive.py` | Create | Unit and integration tests for force application |
| `docs/api/newton_wheeled.rst` | Regenerate | Public API docs after new symbols |
| `CHANGELOG.md` | Modify | Public API entry under `Added` |

Do not modify Phase 00 USDA fixtures in the first implementation unless a test
proves the simple geometry cannot exercise drive/brake behavior.

## Task 00: Execution Order And Failing API Tests

**Files:**

- Create: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Write public import tests**

Assert these names are importable from `newton.wheeled`:

```python
from newton.wheeled import (
    WheelDriveControl,
    WheelDriveState,
    apply_wheel_drive_forces,
    update_wheel_drive_normal_loads,
)
```

Name adjustments are acceptable if they stay prefix-first and explicit.

- [ ] **Step 2: Write allocation/default tests**

Create a small Phase 1A wheeled model and assert:

- drive control/state arrays are sized to `wheeled_metadata.wheel_count`;
- `forward_axis_body` defaults to `+X`;
- `axle_axis_body` defaults to `+Y`;
- drive/brake commands default to zero;
- friction override defaults to a disabled sentinel such as `-1`;
- fallback normal load defaults to zero.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive
```

Expected before implementation: import failure.

## Task 01: Drive Control And State Objects

**Files:**

- Create: `newton/_src/wheeled/drive.py`
- Modify: `newton/_src/wheeled/__init__.py`
- Modify: `newton/wheeled.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Implement `WheelDriveControl`**

Allocate per-wheel arrays on the model device:

- `enabled`
- `drive_torque`
- `brake_torque`
- `target_speed`
- `target_speed_gain`
- `friction_mu`
- `fallback_normal_load`
- `forward_axis_body`
- `axle_axis_body`

Use SI units in public docstrings. Bind the object to the `WheeledModelMetadata`
used at construction, matching the Phase 1B patch-state validation pattern.

- [ ] **Step 2: Implement `WheelDriveState`**

Allocate per-wheel diagnostic arrays:

- `normal_load`
- `previous_normal_load`
- `longitudinal_direction`
- `wheel_angular_speed`
- `longitudinal_speed`
- `slip_speed`
- `requested_force`
- `applied_force`
- `friction_limit`

Add `clear()` for diagnostics that should reset each step, but preserve
`previous_normal_load` unless explicitly requested.

- [ ] **Step 3: Export public names**

Re-export through `newton/wheeled.py` and internal `newton/_src/wheeled/__init__.py`.

- [ ] **Step 4: Verify default tests pass**

Run the focused drive test command.

## Task 02: Longitudinal Direction And Wheel Kinematics

**Files:**

- Modify: `newton/_src/wheeled/drive.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Add direction projection kernel tests**

Create synthetic patch normals and wheel body poses. Assert that body-frame
`forward_axis_body` is transformed to world frame, projected onto the tangent
plane orthogonal to the patch normal, and normalized.

- [ ] **Step 2: Add wheel speed diagnostics**

Compute:

- wheel COM world position from `state.body_q` and `model.body_com`;
- patch-point velocity from body linear/angular velocity and patch offset;
- wheel angular speed by projecting body angular velocity onto `axle_axis_body`;
- slip speed as `longitudinal_speed - angular_speed * wheel_radius`.

Use `WheeledModelMetadata.wheel_radius` for radius.

- [ ] **Step 3: Verify kinematic diagnostics**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive
```

## Task 03: Force Computation And Wrench Application

**Files:**

- Modify: `newton/_src/wheeled/drive.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Implement `apply_wheel_drive_forces()`**

The helper should accept:

- `model`
- `state`
- `wheeled_metadata`
- `patch_state`
- `drive_control`
- `drive_state`

It should launch kernels only. `patch_state` is the contact geometry source;
`state.body_f` is the destination for external wheel-body wrenches.

- [ ] **Step 2: Add inactive and no-load tests**

Assert inactive patches, disabled wheels, zero normal load, or degenerate
longitudinal direction do not apply forces.

- [ ] **Step 3: Add friction-limit tests**

Assert drive torque converts to `drive_torque / radius`, then clips to
`mu * normal_load` using either explicit friction override or patch material
seed.

- [ ] **Step 4: Add braking tests**

Assert brake torque opposes current longitudinal/slip velocity and does not
accelerate a stopped wheel from rest.

- [ ] **Step 5: Add wrench tests**

Use a patch point offset from wheel COM and assert `State.body_f` receives both:

- linear force [N];
- torque `cross(patch_center - wheel_com_world, force_world)` [N*m].

Run focused tests.

## Task 04: Normal-Load Latching

**Files:**

- Modify: `newton/_src/wheeled/drive.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Implement `update_wheel_drive_normal_loads()`**

Copy usable `patch_state.normal_force` values into
`drive_state.previous_normal_load`. If `patch_state.normal_force` is zero or the
patch is inactive, keep or clear according to an explicit policy documented in
the function docstring.

- [ ] **Step 2: Test fallback load behavior**

Assert `fallback_normal_load` is used for force computation when no latched
solver normal load is available.

- [ ] **Step 3: Test latched load behavior**

Populate Phase 1B normal-force diagnostics synthetically and assert the latched
load becomes the load used by the next drive-force call.

Run focused tests.

## Task 05: Flat-Scene Integration Tests

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_drive.py`

- [ ] **Step 1: Add forward acceleration test**

Build a minimal flat fixture with wheel shapes and low/zero solver friction for
wheel-ground contact. Use `WheelDriveControl.friction_mu` and a fallback normal
load to make wheeled traction the explicit friction owner. Assert the vehicle or
wheel body gains forward velocity under drive torque.

- [ ] **Step 2: Add braking test**

Start with forward velocity, apply brake torque, and assert longitudinal velocity
decreases without reversing from rest in a single step.

- [ ] **Step 3: Add skid-steer-style opposite command test**

Use left/right wheel force commands with opposite signs on a simple chassis or
paired wheel bodies. Assert yaw/angular velocity changes in the expected
direction. Keep this as a flat reference behavior test, not a full drive-mode
implementation.

- [ ] **Step 4: Add wrapped solver flow test**

Exercise this order with `SolverMuJoCo(use_mujoco_contacts=False)` when MuJoCo is
available:

```python
state.clear_forces()
model.collide(state, contacts)
update_wheel_contact_patches(model, state, contacts, wheeled_metadata, patch_state)
apply_wheel_drive_forces(model, state, wheeled_metadata, patch_state, drive_control, drive_state)
solver.step(state, next_state, control, contacts, dt)
```

Skip gracefully when MuJoCo dependencies are unavailable.

Run focused tests.

## Task 06: API Docs, Regression Tests, And Commit

**Files:**

- Modify: `docs/api/newton_wheeled.rst`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Regenerate API docs**

Run:

```bash
uv run docs/generate_api.py
```

- [ ] **Step 2: Run focused regression tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
```

- [ ] **Step 3: Run pre-commit**

Run:

```bash
uvx pre-commit run -a
```

- [ ] **Step 4: Commit Phase 2 implementation**

Stage only Phase 2 files and commit:

```bash
git add CHANGELOG.md docs/api/newton_wheeled.rst   newton/wheeled.py newton/_src/wheeled/__init__.py   newton/_src/wheeled/drive.py   newton/tests/test_wheeled_vehicle_drive.py
git commit -m "Add wheeled drive and braking"
```

Suggested commit body:

```text
Add a basic longitudinal wheeled force layer that consumes Phase 1B contact
patch diagnostics and applies clipped drive/brake wrenches to wheel bodies. The
implementation keeps normal support with the wrapped solver and makes solver
friction ownership explicit for flat reference tests.
```

## Open Questions

- The first implementation should use configurable body-frame forward and axle
  axes. Automatic axis inference from wheel cylinder geometry can follow once
  the simplified fixtures have stable conventions.
- The penetration/stiffness normal-load fallback needs a contact-distance audit
  before implementation. Until then, use explicit fallback normal loads for
  tests that cannot rely on solver-reported contact forces.
- Applying only contact-patch traction is enough for basic vehicle motion tests,
  but motor torque balance around the axle may need a later motor/drivetrain
  phase if wheel spin dynamics become important.
