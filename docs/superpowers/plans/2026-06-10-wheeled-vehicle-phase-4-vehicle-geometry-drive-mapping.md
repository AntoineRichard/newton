# Wheeled Vehicle Phase 4 Vehicle Geometry And Drive Mapping Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add vehicle-geometry/layout mappings for Ackermann and skid-steer
assets on top of the Phase 3 tire-control path without turning the wheeled layer
into a closed-loop vehicle controller. Ackermann and skid-steer describe the
vehicle shape and wheel topology: where the wheels are, which wheels steer, and
how side/front/rear wheel roles distribute actuator commands. The new layer
should convert normalized user or robot commands into simulation-side
motor/steering actuator inputs, then into per-wheel analytical wheel speeds and,
for Ackermann assets, steering joint targets owned by the main rigid solver.

**Done when:** The simplified RC car responds to normalized drive/steering
commands, the simplified Husky responds to normalized left/right skid-steer
commands including in-place rotation, and a user-drivable example can switch
between driving the car and the Husky. Runtime command mapping must remain
batched and device-side; setup-time manifest parsing and validation may run on
the host.

**Scope:** Phase 4 adds a vehicle geometry/layout layer, normalized command
buffers, minimal actuator-style drive/steering mapping, and one drivable
example. It does not add chassis-speed/yaw-rate controllers,
path-following controllers, suspension control, powertrain dynamics, gearboxes,
calibrated differentials, battery models, Brush/Fiala/Pacejka tire models,
raycasts, or hydroelastic contacts. Suspension and steering joint dynamics remain ordinary main-solver
dynamics; Phase 4 only writes best-effort steering joint targets where a vehicle
layout marks a wheel as steerable.

---

## Inputs

Roadmap:

- `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`

Implementation inputs:

- `newton.wheeled.WheeledModelMetadata`
- `newton.wheeled.WheelTireControl`
- `newton.Control.joint_target_pos`
- `newton.Control.joint_target_vel`
- Newton actuator patterns in `newton/_src/actuators/`
- `newton/examples/assets/wheeled/manifest.json`
- Phase 00 fixture reference dimensions when a built-in geometry helper needs
  them: wheelbase, track width, steering limit, and steering joint labels.

## Design Notes

Keep Phase 1A wheel metadata limited to wheel identity, wheel body/shape
indices, wheel dimensions, and vehicle ids. Phase 4 should introduce a separate
vehicle-geometry/layout concept for topology, wheel roles, command distribution,
and actuator wiring. This avoids folding steering, suspension, or vehicle-shape
fields into the wheel metadata tables.

Follow the spirit of Newton's actuator design: small composable pieces should
map command inputs through optional scaling/curves, clamping, delays, and
transmission/distribution before scattering into simulation arrays. Phase 4
should provide simulation tools that consume commands; it should not promise
precise chassis velocity, yaw-rate, or path tracking. A robot policy may send
normalized values such as `[-1, 1]`, and the wheeled layer maps those values
through configured motor and steering actuator models.

Ackermann and skid-steer should be treated as geometry/layout choices, not
controller types. Ackermann means the vehicle has steerable front wheel roles;
when the built-in helper expands one steering command into inner/outer steering
joint targets, it needs wheelbase and track width. Skid-steer means the vehicle
has left/right drive groups that distribute drive actuator commands by side.

Use a flat layout that can represent heterogeneous vehicle shapes in one model:

- per-vehicle geometry kind: Ackermann or skid-steer;
- per-wheel vehicle id, driven flag, steerable flag, drive channel, and steering
  channel;
- per-wheel steering joint DOF index, using `-1` when the wheel is not
  steerable;
- optional left/right and front/rear role arrays for built-in Ackermann and
  skid-steer helpers. These roles may be stored internally as signs for kernels,
  but the public concept should be role/channel assignment rather than a sign
  convention.

Do not duplicate wheel radius in this layout. Wheel radius already belongs to
`WheeledModelMetadata`; motor-style commands can map directly to wheel angular
speed [rad/s], and any future linear-speed convenience mapper can read the
metadata radius when needed.

The runtime update helper should write actuator outputs, not solved vehicle
motion:

- `WheelTireControl.wheel_angular_speed` for driven wheels after normalized
  motor commands pass through scale/curve/distribution/clamping;
- `newton.Control.joint_target_pos` for steerable Ackermann joints after
  normalized steering commands pass through steering geometry and limits;
- optional `newton.Control.joint_target_vel` entries only if the selected solver
  or fixture drive configuration benefits from velocity targets.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/_src/wheeled/vehicle.py` | Create | Vehicle geometry/layout, normalized command buffers, actuator-style drive/steering mapping kernels |
| `newton/_src/wheeled/__init__.py` | Modify | Internal exports |
| `newton/wheeled.py` | Modify | Public Phase 4 import surface |
| `newton/tests/test_wheeled_vehicle_drive_modes.py` | Create | Unit and integration tests for geometry layout and command mapping |
| `newton/examples/wheeled/example_wheeled_car_control.py` | Create | User-drivable Ackermann RC car demo using Phase 00 assets |
| `newton/examples/wheeled/example_wheeled_husky_control.py` | Create | User-drivable skid-steer Husky demo using Phase 00 assets |
| `newton/tests/test_examples.py` | Modify | Register the new example smoke test |
| `README.md` | Modify | Register the new example command and screenshot entry |
| `docs/api/newton_wheeled.rst` | Regenerate | Public API docs after new symbols |
| `CHANGELOG.md` | Modify | Public API and example entries under `Added` |

Candidate public symbols:

- `WheeledVehicleLayout`
- `WheeledVehicleControl`
- `WheeledVehicleState`
- `WheeledMotorConfig`
- `WheeledSteeringConfig`
- `build_wheeled_vehicle_layout()`
- `configure_wheeled_vehicle_control()`
- `update_wheeled_vehicle_controls()`

Keep final naming prefix-first and consistent with the existing
`newton.wheeled` API before implementation.

## Task 00: Failing API And Layout Tests

**Files:**

- Create: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Write public import tests**

Assert the Phase 4 names are importable from `newton.wheeled`:

```python
from newton.wheeled import (
    WheeledVehicleControl,
    WheeledVehicleLayout,
    WheeledVehicleState,
    build_wheeled_vehicle_layout,
    configure_wheeled_vehicle_control,
    update_wheeled_vehicle_controls,
)
```

Expected before implementation: import failure.

- [x] **Step 2: Write allocation/default tests**

Build a small synthetic two-vehicle wheeled model or use the Phase 00 fixtures.
Assert:

- layout arrays are sized to `vehicle_count` and `wheel_count`;
- vehicle command arrays default to disabled or zero command;
- per-wheel output diagnostics default to zero;
- steering DOF indices are `-1` for non-steerable wheels;
- Ackermann and skid-steer vehicle geometries can coexist in one layout.

- [x] **Step 3: Write validation tests**

Assert failures for:

- missing steering joint labels on an Ackermann steerable wheel;
- mismatched wheel role/channel counts;
- missing or non-positive wheelbase/track width only when the built-in Ackermann
  steering expansion helper is requested;
- steering joint labels that do not resolve to one-DOF revolute joints;
- layout wheel count that does not match `WheeledModelMetadata`.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive_modes
```

## Task 01: Vehicle Layout Loading

**Files:**

- Create: `newton/_src/wheeled/vehicle.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Define layout and role data**

Implement `WheeledVehicleLayout` with host-visible counts and device arrays for:

Core layout data:

- vehicle geometry kind;
- wheel vehicle id;
- wheel driven flag;
- wheel steerable flag;
- wheel drive channel, such as shared, left, right, or disabled;
- wheel steering channel, such as none, front-left, front-right, or explicit
  per-joint target channel;
- wheel steering joint DOF index.

Optional helper data:

- vehicle wheelbase [m], only needed for built-in Ackermann steering expansion;
- vehicle track width [m], only needed for built-in Ackermann steering expansion
  or future twist-to-skid-steer convenience helpers;
- vehicle steering limit [rad], only needed when normalized steering commands
  are expanded to physical steering joint targets.

Use nested enum-like constants or small nested classes for geometry kinds,
drive channels, and steering channels if that fits Newton style better than
module-level constants.

- [x] **Step 2: Build from explicit role inputs**

Add `build_wheeled_vehicle_layout()` support for explicit role arrays first.
This keeps unit tests independent of manifest parsing and makes the public API
usable for generated assets.

- [x] **Step 3: Build from the Phase 00 manifest**

Add manifest-assisted construction for the RC car and Husky fixtures. The helper
may interpret `vehicle_type` as a Phase 4 geometry/layout hint, but it must not
add that field to `WheeledAssetMetadata` or the Phase 1A metadata contract.
Resolve steering joint labels to `joint_qd_start`/`joint_q_start` DOF indices
after model finalization. Use fixture wheel labels to assign front/rear and
left/right roles for the built-in helpers, but keep explicit role inputs
available so users are not forced into label parsing.

Validate the expected Phase 00 roles:

- RC car: Ackermann geometry, front wheels steerable, all wheels driven for the
  first pass unless the manifest explicitly marks a subset;
- Husky: skid-steer geometry, no steerable wheels, all wheels driven.

- [x] **Step 4: Verify layout tests pass**

Run the focused drive-mode tests.

## Task 02: Vehicle Command, Actuator Config, And State Objects

**Files:**

- Modify: `newton/_src/wheeled/vehicle.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Implement `WheeledVehicleControl` as a channel buffer**

Allocate generic normalized command channels. These are robot/user actuator
commands, not solved vehicle states and not geometry-specific fields:

- `enabled`, shape `(vehicle_count,)`;
- `drive_command`, shape `(drive_channel_count,)`, normalized `[-1, 1]`;
- `steering_command`, shape `(steering_channel_count,)`, normalized `[-1, 1]`;
- optional `brake_command`, shape `(brake_channel_count,)`, normalized `[0, 1]`,
  only if the first implementation needs a separate braking path.

The base control object should not expose `ackermann_drive`,
`left_drive_command`, `right_drive_command`, or other geometry-specific fields.
Ackermann and skid-steer helpers may provide named convenience functions or thin
facades that write the relevant generic channels, but the shared runtime update
should consume channel arrays plus `WheeledVehicleLayout`.

Do not expose `linear_speed`, `yaw_rate`, or `steering_angle` as the primary
Phase 4 command contract. Convenience mappers from twist-like commands may be a
future helper, but the core API should model the commands a robot sends to motor
and steering actuators.

- [x] **Step 2: Implement minimal actuator-style configuration**

Add configuration data for the first simple open-loop actuator maps:

- drive channel command clamp, default `[-1, 1]`;
- maximum wheel angular speed [rad/s] or per-channel motor speed scale;
- optional motor response curve hook, default linear;
- steering channel command clamp, default `[-1, 1]`;
- steering limit [rad] from layout or explicit config;
- optional steering response curve hook, default linear.

Keep this modular in the same spirit as Newton actuators: command input,
optional curve/delay/clamp, then transmission/distribution into simulation
arrays. The first implementation can be stateless and linear, but the shape
should not preclude motor curves or differential modules later.

- [x] **Step 3: Implement `WheeledVehicleState` diagnostics**

Allocate diagnostics that report what the simulation-side actuator mapping
actually wrote:

- per-wheel normalized drive command after channel lookup and clamping;
- per-wheel target angular speed [rad/s];
- per-wheel target steering angle [rad];
- per-drive-channel clipped command;
- per-steering-channel clipped command;
- per-steering-channel steering angle target [rad].

Avoid diagnostics named as if the vehicle achieved a chassis speed or yaw rate.
Those belong to downstream state estimation, not this command-mapping layer.

- [x] **Step 4: Implement `configure_wheeled_vehicle_control()`**

Match `configure_wheel_tire_control()` behavior for generic channel buffers:
scalar command values broadcast across selected channels, arrays must match the
selected drive or steering channel count, values are clamped in the runtime
update path, and invalid shapes raise `ValueError`. Geometry-specific helper
functions may translate user-friendly Ackermann or skid-steer commands into
these generic channel writes.

- [x] **Step 5: Verify control/state tests pass**

Run the focused drive-mode tests.

## Task 03: Skid-Steer Geometry Mapping

**Files:**

- Modify: `newton/_src/wheeled/vehicle.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Write skid-steer actuator-map tests**

For a skid-steer vehicle whose layout maps left and right wheels to two drive
channels and max wheel speed `omega_max`, assert:

- channel commands are clamped to `[-1, 1]`;
- left wheels receive `drive_command[left_channel] * omega_max`;
- right wheels receive `drive_command[right_channel] * omega_max`;
- equal channel commands request forward or reverse wheel motion;
- opposite channel commands request in-place rotation;
- per-wheel diagnostics report the clipped normalized command and angular-speed
  target.

- [x] **Step 2: Implement skid-steer kernel path**

`update_wheeled_vehicle_controls()` should launch a batched kernel over wheels
or vehicles and write `WheelTireControl.wheel_angular_speed` for driven
skid-steer wheels. It should not loop over vehicles or wheels in the simulation
step.

- [x] **Step 3: Add disabled and non-driven tests**

Assert disabled vehicles and non-driven wheels receive zero target wheel speed.

## Task 04: Ackermann Geometry Mapping

**Files:**

- Modify: `newton/_src/wheeled/vehicle.py`
- Modify: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Write steering actuator-map tests**

For wheelbase `L`, track width `T`, steering command `u`, and steering limit
`delta_max`, assert:

- `u` is clamped to `[-1, 1]`;
- center steering target is `u * delta_max` for the first linear steering map;
- near-zero steering gives equal left/right steering targets;
- nonzero steering computes inner and outer front steering targets using
  Ackermann geometry;
- steering targets are written only for steerable wheels;
- non-steerable wheels keep steering DOF index `-1` and are not written.

This is a best-effort steering target, not a guarantee that the physical joint
reaches that angle within a timestep. The fixture-authored steering drive and
main solver own the tracking dynamics.

- [x] **Step 2: Write Ackermann drive-map tests**

Assert driven wheel angular-speed targets are generated from the drive channel
selected by the layout, motor speed scale, and optional steering geometry
distribution. For the first implementation, use an open-loop linear motor map:

```python
center_wheel_speed = clamp(drive_command[channel], -1.0, 1.0) * max_wheel_angular_speed
```

If left/right speed scaling is included for steering, derive a yaw-rate-like
geometric scale from the steering target only as a distribution hint. Do not name
or test it as an achieved vehicle yaw-rate controller.

- [x] **Step 3: Implement Ackermann kernel path**

`update_wheeled_vehicle_controls()` should write tire wheel speeds and steering
joint targets in one or more kernels. Steering joint targets go through
`newton.Control.joint_target_pos`; steering dynamics remain with the main
solver and the fixture-authored drives.

- [x] **Step 4: Add clamp and sign tests**

Cover left turns, right turns, reverse motor commands, and steering limit
clipping.

## Task 05: Integrated Tire-Control Flow

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_drive_modes.py`

- [x] **Step 1: Build a mixed RC/Husky model**

Use the Phase 00 assets, register MuJoCo and wheeled attributes, apply the
wheeled manifest, and call `configure_wheel_axle_joints()` before finalization
for the analytical tire path.

- [x] **Step 2: Verify runtime update writes the expected destinations**

After one `update_wheeled_vehicle_controls()` call, assert:

- RC-car steering target DOFs are nonzero for a steering command;
- Husky has no steering target writes;
- RC-car and Husky wheel angular speed commands match their normalized command
  mapping and vehicle geometry;
- tire-control arrays can be consumed by `apply_wheel_tire_forces()` without
  changing the Phase 3 tire API.

- [x] **Step 3: Verify heterogeneous layout behavior**

Replicate the mixed template to multiple worlds and assert each replicated
vehicle receives the correct geometry kind, wheel roles, and command outputs
without host-side branching in the runtime update.

## Task 06: User-Drivable Example

**Files:**

- Create: `newton/examples/wheeled/example_wheeled_car_control.py`
- Create: `newton/examples/wheeled/example_wheeled_husky_control.py`
- Modify: `newton/tests/test_examples.py`
- Modify: `README.md`

- [x] **Step 1: Create the example from the tire-drive baseline**

Start from `example_wheeled_tire_drive.py` so the example uses:

- Phase 00 assets;
- `SolverMuJoCo` with Newton-generated contacts;
- normal-only wheel contacts via `configure_mujoco_wheel_contacts()`;
- fixed axle joints via `configure_wheel_axle_joints()`;
- Phase 3 tire forces via `apply_wheel_tire_forces()`;
- Phase 4 command mapping via `update_wheeled_vehicle_controls()`.

- [x] **Step 2: Add drive controls**

Expose GUI controls that work in the existing example UI:

- selected vehicle: RC car or Husky;
- scripted command cycle for automated tests;
- manual drive command slider `[-1, 1]` for the RC car Ackermann helper;
- manual steering command slider `[-1, 1]` for the RC car Ackermann helper;
- manual left and right drive command sliders `[-1, 1]` for the Husky
  skid-steer helper;
- drive scale.

If the viewer exposes stable keyboard input, optional keyboard bindings may be
added, but sliders/radio buttons are sufficient for this phase.

- [x] **Step 3: Keep automated tests deterministic**

Default test mode should run scripted commands, not depend on interactive input.
`test_final()` should check finite body state and a weak behavioral assertion:

- RC car translates forward and changes yaw or lateral position under steering
  command;
- Husky rotates in place under opposite side drive commands.

Use tolerances loose enough for MuJoCo/contact variability but strong enough to
catch a disconnected command path.

- [x] **Step 4: Register the example**

Add the example to `newton/tests/test_examples.py`, `README.md`, and the example
screenshot table. Reuse an existing wheeled screenshot only as a temporary
placeholder if no new screenshot is generated in the implementation pass.

Run:

```bash
uv run --extra dev -m newton.examples wheeled_car_control --viewer null --test --num-frames 180 --world-count 2 --device cpu --quiet
uv run --extra dev -m newton.examples wheeled_husky_control --viewer null --test --num-frames 180 --world-count 2 --device cpu --quiet
uv run --extra dev -m newton.tests -k wheeled_car_control
uv run --extra dev -m newton.tests -k wheeled_husky_control
```

## Task 07: Public API, Docs, And Changelog

**Files:**

- Modify: `newton/_src/wheeled/__init__.py`
- Modify: `newton/wheeled.py`
- Modify: `CHANGELOG.md`
- Regenerate: `docs/api/newton_wheeled.rst`

- [x] **Step 1: Export public names**

Expose Phase 4 names through `newton.wheeled`. Examples and docs must not import
from `newton._src`.

- [x] **Step 2: Regenerate API docs**

Run:

```bash
uv run docs/generate_api.py
```

or the repository-standard equivalent if this script requires a different entry
point in the current environment.

- [x] **Step 3: Add changelog entries**

Add public API and example entries under `CHANGELOG.md` `[Unreleased]` / `Added`.
Use user-facing language and avoid implementation-only details.

- [x] **Step 4: Run focused verification**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive_modes
uv run --extra dev -m newton.tests -k wheeled_car_control
uv run --extra dev -m newton.tests -k wheeled_husky_control
uv run --extra dev -m newton.examples wheeled_car_control --viewer null --test --num-frames 180 --world-count 2 --device cpu --quiet
uv run --extra dev -m newton.examples wheeled_husky_control --viewer null --test --num-frames 180 --world-count 2 --device cpu --quiet
git diff --check
```

## Out Of Scope

- Making physical wheel spin part of the tire model path.
- Visual wheel spin.
- Suspension control or suspension-specific helpers.
- Better tire models beyond the saturated-linear Phase 3 baseline.
- Powertrain modules, motor maps, gearboxes, and differentials.
- Non-flat terrain validation beyond keeping the existing flat reference scene
  stable.
- Hydroelastic contact implementation or tuning.

## Exit Criteria

- Public Phase 4 APIs map normalized drive/steering commands to per-wheel tire
  controls and steering joint targets.
- Ackermann and skid-steer geometry/actuator mappings work for the Phase 00 RC
  car and Husky assets.
- A single mixed model can contain both modes and be replicated across worlds.
- The runtime command update is batched and device-side.
- A user-drivable example lets users select and command the RC car or Husky.
- Focused tests and the new example smoke test pass.
