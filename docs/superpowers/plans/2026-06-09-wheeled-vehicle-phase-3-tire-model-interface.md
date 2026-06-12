# Wheeled Vehicle Phase 3 Tire Model Interface Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a modular tire-model API that computes longitudinal and lateral
tire forces from Phase 1B contact patches, main-solver wheel body kinematics,
and analytical wheel angular speed.

**Done when:** A flat reference fixture can move from analytical wheel-speed
inputs through the new tire-model force path, pure lateral slip is resisted by
the tire model, and combined longitudinal/lateral slip is clipped by one shared
friction limit. The implementation must keep normal support, suspension, and
steering with the wrapped rigid solver, and make wheel-pair solver friction
ownership explicit.

**Scope:** Phase 3 adds `WheelTireControl`, `WheelTireState`,
`apply_wheel_tire_forces()`, and `update_wheel_tire_normal_loads()`, plus tests
and one example. It does not add steering modes, suspension control, powertrain
modules, differentials, raycasts, hydroelastic contacts, or Pacejka/Brush/Fiala
implementations. Physical wheel-body spin should be locked, omitted, or visual
only for the tire-model path.

---

## Inputs

Spec:

- `docs/superpowers/specs/2026-06-09-wheeled-vehicle-phase-3-tire-model-interface.md`

Implementation inputs:

- `newton.wheeled.WheeledModelMetadata`
- `newton.wheeled.WheelContactPatchState`
- `newton.State.body_f`
- `Model.body_com`
- `State.body_q`
- `State.body_qd`
- `WheelTireControl.wheel_angular_speed`
- `WheelContactPatchState.active`
- `WheelContactPatchState.center`
- `WheelContactPatchState.normal`
- `WheelContactPatchState.patch_area`
- `WheelContactPatchState.normal_force`
- `WheelContactPatchState.friction_mu_seed`

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/_src/wheeled/tire.py` | Create | Tire control/state objects, slip diagnostics, tire-force kernels, normal-load latch |
| `newton/_src/wheeled/__init__.py` | Modify | Internal exports |
| `newton/wheeled.py` | Modify | Public Phase 3 import surface |
| `newton/tests/test_wheeled_vehicle_tire.py` | Create | Unit and integration tests for tire forces |
| `newton/examples/wheeled/example_wheeled_tire_drive.py` | Create | Flat-scene tire-model example using Phase 00 assets |
| `newton/tests/test_examples.py` | Modify | Register the new example smoke test |
| `README.md` | Modify | Register the new example command and screenshot entry |
| `docs/api/newton_wheeled.rst` | Regenerate | Public API docs after new symbols |
| `CHANGELOG.md` | Modify | Public API and example entries under `Added` |

Keep the Phase 2 `drive.py` API intact. Internal helper sharing is acceptable,
but examples and docs must import only from public modules.

## Task 00: Failing API And Allocation Tests

**Files:**

- Create: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Write public import tests**

Assert these names are importable from `newton.wheeled`:

```python
from newton.wheeled import (
    WheelTireControl,
    WheelTireState,
    apply_wheel_tire_forces,
    update_wheel_tire_normal_loads,
)
```

Expected before implementation: import failure.

- [ ] **Step 2: Write allocation/default tests**

Create a small Phase 1A wheeled model and assert:

- tire control/state arrays are sized to `wheeled_metadata.wheel_count`;
- `enabled` defaults to true;
- `wheel_angular_speed` defaults to zero;
- `friction_mu` defaults to a disabled sentinel such as `-1`;
- `fallback_normal_load` defaults to zero;
- `forward_axis_body` defaults to `+X`;
- `axle_axis_body` defaults to `+Y`;
- longitudinal and lateral stiffness defaults are positive and finite;
- `min_reference_speed` defaults to a positive speed floor;
- all tire-state diagnostics default to zero except persistent latches when
  explicitly preserved.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_tire
```

## Task 01: Tire Control And State Objects

**Files:**

- Create: `newton/_src/wheeled/tire.py`
- Modify: `newton/_src/wheeled/__init__.py`
- Modify: `newton/wheeled.py`
- Modify: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Implement `WheelTireControl`**

Allocate per-wheel arrays on the model device:

- `enabled`
- `wheel_angular_speed`
- `friction_mu`
- `fallback_normal_load`
- `forward_axis_body`
- `axle_axis_body`
- `longitudinal_stiffness`
- `lateral_stiffness`
- `min_reference_speed`

Use SI units in public docstrings. Bind the object to the
`WheeledModelMetadata` used at construction, matching the Phase 1B/Phase 2
validation pattern.

- [ ] **Step 2: Implement `WheelTireState`**

Allocate per-wheel diagnostic arrays:

- `normal_load`
- `previous_normal_load`
- `longitudinal_direction`
- `lateral_direction`
- `wheel_angular_speed`
- `longitudinal_speed`
- `lateral_speed`
- `longitudinal_slip_speed`
- `longitudinal_slip_ratio`
- `lateral_slip_angle`
- `requested_longitudinal_force`
- `requested_lateral_force`
- `applied_longitudinal_force`
- `applied_lateral_force`
- `friction_limit`
- `combined_slip_scale`

Add `clear(clear_previous_normal_load=False)`.

- [ ] **Step 3: Export public names**

Re-export through `newton/wheeled.py` and internal
`newton/_src/wheeled/__init__.py`.

- [ ] **Step 4: Verify allocation/default tests pass**

Run the focused tire test command.

## Task 02: Tire Directions And Slip Diagnostics

**Files:**

- Modify: `newton/_src/wheeled/tire.py`
- Modify: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Add direction tests**

Create synthetic patch normals and wheel body poses. Assert:

- body-frame `forward_axis_body` is transformed to world frame;
- the forward axis is projected onto the contact tangent plane;
- `longitudinal_direction` is normalized;
- `lateral_direction = normalize(cross(normal, longitudinal_direction))`;
- steering/suspension poses are naturally represented by the wheel body pose;
- degenerate direction cases apply no tire force.

- [ ] **Step 2: Add kinematic diagnostic tests**

Compute and assert:

- wheel COM world position from `state.body_q` and `model.body_com`;
- patch-point velocity from body linear/angular velocity and patch offset;
- `longitudinal_speed`;
- `lateral_speed`;
- analytical `wheel_angular_speed` copied from `WheelTireControl`;
- `longitudinal_slip_speed = wheel_angular_speed * radius - longitudinal_speed`;
- regularized `longitudinal_slip_ratio`;
- regularized `lateral_slip_angle`.

- [ ] **Step 3: Verify diagnostics**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_tire
```

## Task 03: Saturated Linear Tire Force Model

**Files:**

- Modify: `newton/_src/wheeled/tire.py`
- Modify: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Implement `apply_wheel_tire_forces()`**

The helper should accept:

- `model`
- `state`
- `wheeled_metadata`
- `patch_state`
- `tire_control`
- `tire_state`

It should launch kernels only. `patch_state` is the contact geometry source;
`state.body_f` is the destination for external wheel-body wrenches.

- [ ] **Step 2: Add inactive and invalid-input tests**

Assert inactive patches, disabled wheels, zero normal load, invalid radius,
zero/negative friction limit, or degenerate directions do not apply forces.

- [ ] **Step 3: Add pure longitudinal slip tests**

Assert positive `wheel_angular_speed * radius - longitudinal_speed` produces a
positive longitudinal force and negative slip produces a negative longitudinal
force. `wheel_angular_speed` must come from `WheelTireControl`, not a physical
spin joint.

- [ ] **Step 4: Add pure lateral slip tests**

Assert positive lateral patch velocity produces negative lateral force, and
negative lateral patch velocity produces positive lateral force.

- [ ] **Step 5: Add combined-slip limit tests**

Assert the requested force vector is scaled to satisfy `mu * normal_load` while
preserving direction. Cover both explicit `friction_mu` and material
`friction_mu_seed`.

Run focused tests.

## Task 04: Wrench Application And Locked-Spin Behavior

**Files:**

- Modify: `newton/_src/wheeled/tire.py`
- Modify: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Add world-frame wrench tests**

Use a patch point offset from wheel COM and assert `State.body_f` receives:

- linear force [N];
- torque `cross(patch_center - wheel_com_world, force_world)` [N*m].

- [ ] **Step 2: Add analytical-spin source test**

Set body angular velocity about the axle to a nonzero value while
`WheelTireControl.wheel_angular_speed` is zero, and assert longitudinal tire slip
uses the analytical input rather than physical body spin. Then set analytical
wheel speed nonzero and assert slip/force changes accordingly.

- [ ] **Step 3: Keep Phase 2 behavior separate**

Do not change `apply_wheel_drive_forces()` in this task unless a shared internal
helper can be added without changing public behavior. If shared helpers are
added, rerun the existing Phase 2 drive tests.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_tire
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive
```

## Task 05: Normal-Load Latching

**Files:**

- Modify: `newton/_src/wheeled/tire.py`
- Modify: `newton/tests/test_wheeled_vehicle_tire.py`

- [ ] **Step 1: Implement `update_wheel_tire_normal_loads()`**

Copy usable positive `patch_state.normal_force` values into
`tire_state.previous_normal_load`. If the patch is inactive or force is
nonpositive, keep or clear the latch according to the explicit
`clear_inactive` argument.

- [ ] **Step 2: Test fallback load behavior**

Assert `fallback_normal_load` is used for force computation when no latched
solver normal load is available.

- [ ] **Step 3: Test latched load behavior**

Populate Phase 1B normal-force diagnostics synthetically and assert the latched
load becomes the load used by the next tire-force call.

Run focused tests.

## Task 06: Flat-Scene Integration Example

**Files:**

- Create: `newton/examples/wheeled/example_wheeled_tire_drive.py`
- Modify: `newton/tests/test_examples.py`
- Modify: `README.md`

- [ ] **Step 1: Build the example from Phase 00 assets**

Use the same RC-car and Husky fixtures and metadata path as the existing wheeled
examples. Replicate to `32` worlds by default. Lock or omit physical wheel spin
joints in the tire-model path. Keep suspension and steering joints, if present,
as ordinary main-solver joints.

- [ ] **Step 2: Use explicit normal-only solver support**

Use `SolverMuJoCo(use_mujoco_contacts=False)` with Newton-generated contacts,
an elliptic cone, and MuJoCo normal-only wheel-ground contacts. Configure wheel
shapes with `mujoco:condim = 1` and higher `mujoco:geom_priority` than terrain
so MuJoCo contact parameter mixing does not promote wheel contacts back to
frictional constraints. Provide tire friction through
`WheelTireControl.friction_mu`.

- [ ] **Step 3: Drive slip through analytical wheel speed**

Assign simple per-wheel `WheelTireControl.wheel_angular_speed` values to create
slip. Do not apply axle torques through physical wheel spin joints and do not
convert drive torque directly to contact force in the tire example.

- [ ] **Step 4: Apply tire forces from contact patches**

Use this flow:

```python
state.clear_forces()
viewer.apply_forces(state)
update tire_control.wheel_angular_speed
model.collide(state, contacts)
update_wheel_contact_patches(model, state, contacts, wheeled_metadata, patch_state)
apply_wheel_tire_forces(model, state, wheeled_metadata, patch_state, tire_control, tire_state)
solver.step(state, next_state, control, contacts, dt)
```

- [ ] **Step 5: Add example validation**

Implement `test_final()` and, if useful, `test_post_step()` to assert finite
state, bounded velocity, and measurable forward/back motion in short headless
runs.

- [ ] **Step 6: Register docs/tests**

Register the example in `README.md` and `newton/tests/test_examples.py`.

Run:

```bash
uv run --extra dev -m newton.examples wheeled_tire_drive --viewer null --num-frames 120 --world-count 2 --test --quiet
uv run --extra dev -m newton.examples wheeled_tire_drive --viewer null --num-frames 60 --world-count 32 --test --quiet
uv run --extra dev -m newton.tests -k wheeled_tire_drive
```

## Task 07: API Docs, Regression Tests, And Commit Prep

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
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_tire
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
uv run --extra dev -m newton.tests -k wheeled_tire_drive
```

- [ ] **Step 3: Run pre-commit**

Run:

```bash
uvx pre-commit run -a
```

- [ ] **Step 4: Update implementation report if requested**

If this phase is implemented by an agent, add a short report under
`docs/superpowers/reports/` describing the API, tire model, example behavior,
test commands, and any remaining stability caveats.
