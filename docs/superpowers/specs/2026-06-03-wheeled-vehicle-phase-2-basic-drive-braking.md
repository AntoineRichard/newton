# Wheeled Vehicle Phase 2 Basic Drive And Braking Spec

## Purpose

Phase 2 adds the first force-producing wheeled behavior. It consumes Phase 1A
wheel metadata and the Phase 1B `WheelContactPatchState`, then applies simple
longitudinal traction and braking wrenches to wheel bodies.

Normal support remains owned by the wrapped rigid solver. The wheeled layer owns
only wheel-specific longitudinal drive/brake forces in this phase.

## Goals

- Add a reusable per-wheel drive/brake control surface.
- Apply longitudinal force at the Phase 1B contact patch center through
  `State.body_f` as a world-frame external wrench on the wheel body.
- Use Phase 1B patch activity, support normal, contact center, material friction
  seed, and normal-load diagnostics to gate and limit force application.
- Keep force computation and wrench application in Warp kernels and flat arrays.
- Make solver-friction ownership explicit so solver friction and wheeled traction
  are not silently double-counted.
- Demonstrate basic forward acceleration, braking, and skid-steer-style opposite
  wheel commands on a flat reference scene.

## Non-Goals

- Do not implement Pacejka, Brush, Fiala, or another full tire model.
- Do not add steering, Ackermann geometry, steering commands, or suspension
  control.
- Do not add motor power curves, differentials, gearboxes, or battery models.
- Do not replace Phase 1B contact patch reduction or add raycasts.
- Do not require new USD schema work for the first implementation.
- Do not make hydroelastic terrain behavior part of this phase.

## Public API Shape

Keep the API readable but narrow. Candidate public names:

- `WheelDriveControl`
- `WheelDriveState`
- `apply_wheel_drive_forces()`

`WheelDriveControl` should own user-writable per-wheel command/config arrays:

| Field | Meaning |
| --- | --- |
| `drive_torque` | Requested drive torque at the wheel [N*m] |
| `brake_torque` | Requested braking torque magnitude [N*m] |
| `target_speed` | Optional target longitudinal speed [m/s], disabled by default |
| `target_speed_gain` | Optional proportional gain from speed error to force [N/(m/s)] |
| `friction_mu` | Optional tire friction override; negative means use patch material seed |
| `forward_axis_body` | Wheel forward axis in wheel body frame |
| `axle_axis_body` | Wheel axle/spin axis in wheel body frame |
| `enabled` | Whether drive/brake commands are active for the wheel |

`WheelDriveState` should own per-wheel diagnostics and latched values:

| Field | Meaning |
| --- | --- |
| `normal_load` | Normal load used for the current force solve [N] |
| `previous_normal_load` | Normal load reported after the previous solver step [N] |
| `longitudinal_direction` | World-space tangent direction used for force application |
| `wheel_angular_speed` | Wheel angular speed around the configured axle [rad/s] |
| `longitudinal_speed` | Wheel body contact-point speed along the longitudinal direction [m/s] |
| `slip_speed` | `longitudinal_speed - wheel_angular_speed * radius` [m/s] |
| `requested_force` | Unclipped longitudinal force request [N] |
| `applied_force` | Clipped longitudinal force applied at the patch [N] |
| `friction_limit` | `mu * normal_load` force limit [N] |

Public names should be exported through `newton/wheeled.py`; implementation
stays under `newton/_src/wheeled/`.

## Direction Convention

Phase 2 should not infer steering or suspension topology. The first
implementation should use configurable per-wheel body-frame axes:

- `forward_axis_body` defaults to body-frame `+X`.
- `axle_axis_body` defaults to body-frame `+Y`.

At runtime, transform `forward_axis_body` by the wheel body pose, then project it
onto the tangent plane orthogonal to `WheelContactPatchState.normal`. Normalize
that projection to obtain the longitudinal direction. If the projected direction
is degenerate, leave the wheel inactive for drive/brake force application.

The axle axis is used only for wheel angular-speed diagnostics and slip-speed
estimation. Steering joints, if present in the asset, remain ordinary simulator
joints; the wheel body pose should already include their effect.

## Normal Load Source

Phase 1B can report `normal_force`, but solver contact force reporting is often
available only after a solver step via `solver.update_contacts()`. Phase 2
should therefore use a latched normal-load model:

1. Before the solver step, drive force computation uses `WheelDriveState.previous_normal_load`.
2. If the previous load is unavailable or zero, use an explicit fallback normal
   load if configured.
3. After the solver step, when `contacts.force` is available, update Phase 1B
   patch diagnostics and copy `WheelContactPatchState.normal_force` into
   `WheelDriveState.previous_normal_load` for the next step.

A penetration/stiffness fallback is still desirable, but it should be introduced
only after the implementation verifies the contact-distance convention from
Newton contacts. The first Phase 2 implementation may support a per-wheel
fallback normal-load override so force tests do not depend on solver force
reporting.

## Force Model

For each active wheel patch:

1. Choose friction coefficient `mu`: use `WheelDriveControl.friction_mu` when it
   is nonnegative, otherwise use `WheelContactPatchState.friction_mu_seed`.
2. Choose normal load from `WheelDriveState.previous_normal_load` or fallback.
3. Compute friction limit `mu * normal_load`.
4. Convert drive torque to force with `drive_torque / wheel_radius`.
5. Convert brake torque to force magnitude with `brake_torque / wheel_radius`
   and apply it opposite the current longitudinal/slip velocity. If velocity is
   near zero, braking should not create a new accelerating force.
6. Optionally add target-speed force from speed error when enabled.
7. Clip the resulting force to the friction limit.
8. Apply the clipped force at the patch center to the wheel body through
   `State.body_f`.

The wrench applied to the wheel body should use the body COM as reference:

```text
force_world = applied_force * longitudinal_direction
torque_world = cross(patch_center - wheel_com_world, force_world)
body_f[wheel_body] += spatial_vector(force_world, torque_world)
```

Use `model.body_com` and `state.body_q` to compute `wheel_com_world`.

## Solver Friction Ownership

The wrapped rigid solver still owns normal contact support. Phase 2 owns
longitudinal wheel drive/braking force. Tests and examples must avoid silently
using full solver friction and wheeled traction for the same wheel-ground pair.

Initial acceptable policies:

- Use low or zero solver friction in Phase 2 drive/brake tests and provide the
  wheeled layer friction coefficient through `WheelDriveControl.friction_mu`.
- Or use solver friction only for non-wheel contacts while explicitly documenting
  any wheel-ground solver-friction behavior that remains enabled.

Do not hide this behind implicit defaults.

## Expected Step Order

A typical simulation step should be documented as:

```text
state.clear_forces()
model.collide(state, contacts)
update_wheel_contact_patches(model, state, contacts, wheeled_metadata, patch_state)
apply_wheel_drive_forces(model, state, wheeled_metadata, patch_state, drive_control, drive_state)
solver.step(state, next_state, control, contacts, dt)
solver.update_contacts(contacts, next_state)      # when supported/requested
update_wheel_contact_patches(model, next_state, contacts, wheeled_metadata, patch_state)
update wheel drive normal-load latch from patch_state.normal_force
```

The post-step force update is optional when a solver cannot report contact
forces; in that case the configured fallback load remains the source.

## Tests

Add `unittest` coverage for:

- Public imports and allocation of drive control/state arrays.
- Inactive wheel patches applying zero force.
- Drive torque converting to longitudinal force and clipping to `mu * normal_load`.
- Brake torque opposing current longitudinal/slip velocity without accelerating a
  stopped wheel.
- Correct world-frame wrench accumulation into `State.body_f`, including torque
  from an off-COM patch point.
- Direction projection onto the contact tangent plane.
- Normal-load latching from Phase 1B `normal_force` diagnostics.
- A simple flat-scene vehicle accelerating forward with solver wheel-ground
  friction handled explicitly.
- A skid-steer-style fixture rotating or yawing from opposite left/right wheel
  force commands.

## Acceptance Criteria

- Basic drive/brake force computation is batched and device-side.
- Longitudinal wheel forces are applied at Phase 1B patch centers as external
  wheel-body wrenches.
- Force limits use explicit normal-load and friction sources.
- Solver friction ownership is documented and covered by tests.
- A flat-scene fixture accelerates forward and brakes without adding a raycast
  contact path.
- Opposite wheel commands produce a skid-steer-style turning response on a flat
  reference scene.
