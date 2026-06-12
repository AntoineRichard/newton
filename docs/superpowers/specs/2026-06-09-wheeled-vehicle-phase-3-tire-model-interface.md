# Wheeled Vehicle Phase 3 Tire Model Interface Spec

## Purpose

Phase 3 introduces the first tire-model layer. It consumes Phase 1A wheel
metadata, Phase 1B contact patches, wheel-body pose/velocity from the main
solver, and analytical wheel spin state, then applies combined longitudinal and
lateral tire forces at the contact patch.

Normal support, suspension motion, steering motion, and articulation constraints
remain owned by the wrapped rigid solver. The wheeled layer owns wheel-specific
tire friction for wheel-ground pairs. The implementation should not commit the
solver architecture to Pacejka, Brush, Fiala, the PhysX vehicle snippet shape, or
any other single tire or vehicle formulation.

## Goals

- Add a public tire-model control/state surface separate from the Phase 2 simple
  drive/brake helper.
- Use wheel body pose from the main solver, including any steering or suspension
  joint motion already solved by the articulation system.
- Use analytical per-wheel angular speed for tire slip. Physical wheel-body spin
  joints should be locked, omitted, or treated as visual-only for this tire path.
- Compute longitudinal and lateral tire directions, wheel contact-point
  velocity, longitudinal slip, and lateral slip diagnostics in Warp kernels over
  flat wheel arrays.
- Implement one first tire model: a saturated linear velocity-slip model with
  combined-slip friction limiting.
- Apply tire forces at the Phase 1B contact patch center through
  `State.body_f` as a world-frame external wrench on the wheel body.
- Keep solver-friction ownership explicit so the wrapped rigid solver does not
  silently provide the tire model.
- Keep the design open for later Pacejka, Brush, Fiala, and empirical tire
  models behind the same interface.

## Non-Goals

- Do not assume four wheels, Ackermann steering, skid-steer topology, driven
  axle groups, or any fixed vehicle layout.
- Do not implement suspension dynamics, suspension control, steering dynamics,
  steering commands, Ackermann steering commands, or skid-steer command mapping.
  These remain ordinary main-solver joints and controls.
- Do not use physical wheel spin joints as the tire model's source of wheel
  angular speed.
- Do not add motor curves, differentials, gearboxes, battery models, or
  drive-mode abstractions.
- Do not replace Phase 1B contact patch reduction or add raycasts.
- Do not tune a Pacejka, Brush, or Fiala model to a specific real tire in this
  phase.
- Do not add hydroelastic contact implementation.
- Do not remove or rename the Phase 2 `WheelDriveControl`,
  `WheelDriveState`, or `apply_wheel_drive_forces()` API.

## Phase 1B And Phase 2 Handoff

Phase 3 consumes:

- `WheeledModelMetadata` flat wheel arrays for wheel body index, wheel shape
  index, wheel radius, wheel width, and vehicle id.
- `WheelContactPatchState` patch activity, center, normal, patch area, material
  friction seed, and normal-force diagnostics.
- Wheel body transforms and velocities from the main solver. These poses already
  include suspension and steering joint motion when the asset has those joints.
- Existing Phase 2 conventions for wheel body-frame forward and axle axes, used
  as kinematic directions rather than unlocked spin-joint state.

Phase 3 should not rediscover wheels from labels or manifests at runtime.
Runtime-annotated and pre-authored USDA metadata must continue to produce flat
wheel ids after builder replication.

The Phase 2 drive/brake helper remains a simpler force helper. Phase 3 should
not deprecate it. New examples may use the tire-model API for more physical
contact-patch behavior while keeping Phase 2 tests as regression coverage for
the simple helper.

## Public API Shape

Add public names exported through `newton/wheeled.py`:

- `WheelTireControl`
- `WheelTireState`
- `apply_wheel_tire_forces()`
- `update_wheel_tire_normal_loads()`

Implementation should live under `newton/_src/wheeled/`, likely in a new
`tire.py` module, with internal exports updated through
`newton/_src/wheeled/__init__.py`.

`WheelTireControl` should own user-writable per-wheel configuration/input arrays:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether tire forces are active for the wheel |
| `wheel_angular_speed` | Analytical wheel angular speed input [rad/s] |
| `friction_mu` | Optional tire friction override; negative means use patch material seed |
| `fallback_normal_load` | Explicit normal load used when no solver load is latched [N] |
| `forward_axis_body` | Wheel forward axis in wheel body frame |
| `axle_axis_body` | Wheel axle/spin axis in wheel body frame |
| `longitudinal_stiffness` | Linear longitudinal slip-speed stiffness [N/(m/s)] |
| `lateral_stiffness` | Linear lateral slip-speed stiffness [N/(m/s)] |
| `min_reference_speed` | Speed floor for slip-ratio and slip-angle diagnostics [m/s] |

The first public control object should not own motor torque, brake torque, gear
ratio, differential state, suspension control, or steering control. Those are
main-solver or powertrain concerns. In Phase 3 examples, simple analytical wheel
angular-speed inputs may be assigned directly to create slip while physical wheel
spin remains locked or omitted.

`WheelTireState` should own per-wheel diagnostics and latched values:

| Field | Meaning |
| --- | --- |
| `normal_load` | Normal load used for the current tire solve [N] |
| `previous_normal_load` | Normal load reported after the previous solver step [N] |
| `longitudinal_direction` | World-space tangent direction used for longitudinal force |
| `lateral_direction` | World-space tangent direction used for lateral force |
| `wheel_angular_speed` | Analytical wheel angular speed used by the solve [rad/s] |
| `longitudinal_speed` | Contact-point speed along the longitudinal direction [m/s] |
| `lateral_speed` | Contact-point speed along the lateral direction [m/s] |
| `longitudinal_slip_speed` | `wheel_angular_speed * radius - longitudinal_speed` [m/s] |
| `longitudinal_slip_ratio` | Regularized longitudinal slip ratio |
| `lateral_slip_angle` | Regularized lateral slip angle [rad] |
| `requested_longitudinal_force` | Unclipped longitudinal tire force [N] |
| `requested_lateral_force` | Unclipped lateral tire force [N] |
| `applied_longitudinal_force` | Combined-slip-clipped longitudinal tire force [N] |
| `applied_lateral_force` | Combined-slip-clipped lateral tire force [N] |
| `friction_limit` | `mu * normal_load` total tire force limit [N] |
| `combined_slip_scale` | Scale applied to requested force vector to satisfy the limit |

Add `WheelTireState.clear(clear_previous_normal_load=False)` following the Phase
2 state clear convention.

## Direction And Slip Convention

Use configurable body-frame axes:

- `forward_axis_body` defaults to body-frame `+X`.
- `axle_axis_body` defaults to body-frame `+Y`.

For each active patch:

1. Transform `forward_axis_body` into world space using the wheel body pose from
   the main solver. This pose may already include suspension and steering joint
   motion.
2. Project it onto the tangent plane orthogonal to
   `WheelContactPatchState.normal`.
3. Normalize the projection to form `longitudinal_direction`.
4. Compute `lateral_direction = normalize(cross(normal, longitudinal_direction))`.
5. Compute patch-point velocity from body linear/angular velocity and the patch
   offset from wheel COM. This is rigid-body motion of the wheel carrier/body,
   not spin-joint angular velocity.
6. Compute `longitudinal_speed = dot(patch_velocity, longitudinal_direction)`.
7. Compute `lateral_speed = dot(patch_velocity, lateral_direction)`.
8. Read analytical `wheel_angular_speed` from `WheelTireControl` and copy it to
   `WheelTireState` diagnostics.
9. Compute `longitudinal_slip_speed = wheel_angular_speed * radius - longitudinal_speed`.

Positive `longitudinal_slip_speed` means the wheel surface is trying to drive the
vehicle forward along `longitudinal_direction`, so the first tire model should
request positive longitudinal force.

For diagnostics:

```text
reference_speed = max(abs(longitudinal_speed), min_reference_speed)
longitudinal_slip_ratio = longitudinal_slip_speed / reference_speed
lateral_slip_angle = atan2(lateral_speed, reference_speed)
```

The first model may use slip speeds directly for force computation to avoid
low-speed singularities. Slip ratio and slip angle are still useful diagnostics
and prepare the interface for Fiala, Brush, and Pacejka variants.

## First Tire Model

The first implementation should be a saturated linear velocity-slip model:

```text
requested_longitudinal_force = longitudinal_stiffness * longitudinal_slip_speed
requested_lateral_force = -lateral_stiffness * lateral_speed
```

The lateral force opposes lateral patch velocity. The longitudinal force follows
the sign convention above: positive wheel surface speed relative to the patch
requests forward traction.

Use a friction circle for combined-slip limiting:

```text
limit = mu * normal_load
requested_norm = length(vec2(requested_longitudinal_force, requested_lateral_force))
combined_slip_scale = min(1, limit / requested_norm)
applied_longitudinal_force = requested_longitudinal_force * combined_slip_scale
applied_lateral_force = requested_lateral_force * combined_slip_scale
```

If `requested_norm` is near zero, keep scale at `1` and apply zero force. Future
models may replace the circle with a friction ellipse or a model-specific
combined-slip law without changing the public patch/kinematics inputs.

## Normal Load And Material Source

Use the same ownership model as Phase 2:

1. Use `WheelTireState.previous_normal_load` when it is positive.
2. Fall back to `WheelTireControl.fallback_normal_load` when configured.
3. Otherwise do not apply tire force.

Use `WheelTireControl.friction_mu` when nonnegative; otherwise use
`WheelContactPatchState.friction_mu_seed`. Clamp negative material seeds to zero.

Add `update_wheel_tire_normal_loads(patch_state, tire_state,
clear_inactive=False)` to latch positive `patch_state.normal_force` diagnostics.

## Wrench And Locked Wheel Spin Ownership

Apply the combined tire force at the Phase 1B patch center:

```text
force_world = longitudinal_direction * applied_longitudinal_force
            + lateral_direction * applied_lateral_force
torque_world = cross(patch_center - wheel_com_world, force_world)
body_f[wheel_body] += spatial_vector(force_world, torque_world)
```

The tire layer should not apply motor/brake torque to physical axle joints and
should not use physical wheel-body spin as tire state. If an asset has wheel spin
joints, the tire-model path should lock them, omit them, or keep them visual-only.
Suspension and steering joints remain active ordinary solver joints.

The patch wrench should still be applied at the patch center. Any resulting
reaction through locked spin, steering, or suspension constraints is handled by
the main solver. Later phases may study whether distributing tire forces to
wheel and chassis bodies improves conditioning, but Phase 3 should keep a single
clear body-force contract.

## Solver Friction Ownership

The wrapped rigid solver owns normal contact support, suspension constraints, and
steering constraints. Phase 3 owns wheel-specific longitudinal and lateral tire
friction. Tests and examples should use normal-only wheel-ground solver contacts
where supported so solver friction is not the hidden tire model.

For MuJoCo Warp examples using Newton-generated contacts, prefer normal-only
wheel-ground support through `mujoco:condim = 1` on wheel shapes. Because MuJoCo
contact parameter mixing uses the maximum `condim` for equal-priority geoms, set
the wheel geom priority high enough that wheel contact settings win over the
terrain shape. Keep a tiny nonzero material-friction floor only as a fallback
for solver paths that cannot use `condim=1`.

## Expected Step Order

A typical simulation step should be documented as:

```text
state.clear_forces()
viewer.apply_forces(state)
update analytical wheel angular-speed inputs, if the example has a power input
model.collide(state, contacts)
update_wheel_contact_patches(model, state, contacts, wheeled_metadata, patch_state)
apply_wheel_tire_forces(model, state, wheeled_metadata, patch_state, tire_control, tire_state)
solver.step(state, next_state, control, contacts, dt)
solver.update_contacts(contacts, next_state)      # when supported/requested
update_wheel_contact_patches(model, next_state, contacts, wheeled_metadata, patch_state)
update_wheel_tire_normal_loads(patch_state, tire_state)
```

The post-step force update remains optional when a solver cannot report contact
forces; in that case the configured fallback load remains the source.

## Tests

Add `unittest` coverage for:

- Public imports and allocation of tire control/state arrays sized by
  `WheeledModelMetadata.wheel_count`.
- Default control values, including enabled wheels, zero analytical wheel speed,
  negative friction override, positive stiffness defaults, `+X` forward axis,
  and `+Y` axle axis.
- Inactive patches, disabled wheels, zero normal load, invalid radius, or
  degenerate directions applying zero force.
- Longitudinal/lateral direction construction from wheel pose and patch normal,
  including poses produced by steering/suspension joints.
- Patch-point speed, analytical wheel angular speed, longitudinal slip speed,
  slip ratio, and lateral slip angle diagnostics.
- Tire slip using `WheelTireControl.wheel_angular_speed` rather than physical
  body angular velocity about the axle.
- Pure longitudinal slip producing a force with the expected sign.
- Pure lateral slip producing a force opposite lateral velocity.
- Combined-slip scaling preserving force direction while respecting
  `mu * normal_load`.
- Material friction seeding and explicit friction override behavior.
- Correct world-frame wrench accumulation into `State.body_f`, including the
  patch torque about wheel COM.
- Normal-load latching from Phase 1B `normal_force` diagnostics.
- Replicated metadata whose tire control/state arrays remain indexed by flat
  wheel id.
- A simple flat-scene fixture where analytical wheel angular speed creates slip
  and the tire model moves the vehicle without relying on full solver wheel
  friction or unlocked wheel spin joints.

## Acceptance Criteria

- Tire force computation is batched, device-side, and indexed by flat wheel id.
- The first tire model computes both longitudinal and lateral forces from wheel
  body kinematics, analytical wheel speed, and Phase 1B contact patches.
- Combined-slip limiting uses explicit normal-load and friction sources.
- Solver friction ownership is documented and covered by tests.
- Wheel-body spin joints are not required for tire slip and should be locked,
  omitted, or visual-only in the tire-model path.
- Suspension and steering remain owned by the main solver.
- The public API leaves room for Pacejka, Brush, Fiala, and simpler empirical
  models without changing the solver wrapper contract.
