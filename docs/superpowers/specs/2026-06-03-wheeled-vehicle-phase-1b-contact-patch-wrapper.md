# Wheeled Vehicle Phase 1B Contact Patch Wrapper Spec

## Purpose

Phase 1B adds the first contact-facing wheeled layer. It consumes the Phase 1A
wheel metadata tables and Newton's rigid contact buffers, groups active contacts
by wheel, and reports per-wheel contact patch diagnostics.

This phase does not add a raycast path. Contact geometry comes from Newton's
collision engine, and normal support remains owned by the wrapped rigid solver.

## Goals

- Identify active wheel-ground contacts from Newton rigid contact pairs using
  `WheeledModelMetadata.wheel_shape_indices`.
- Normalize contact orientation so downstream code can reason in terms of
  support on the wheel independent of shape-pair ordering.
- Estimate per-wheel contact center, normal, patch extents, patch area, terrain
  shape, and friction seed values on a flat reference scene.
- Keep the runtime grouping and patch estimation path in Warp kernels and flat
  arrays, with no Python loops over worlds, wheels, vehicles, or contacts.
- Verify the path with `SolverMuJoCo(use_mujoco_contacts=False)` so Newton
  generates the contact geometry consumed by the wrapped solver.

## Non-Goals

- Do not implement raycast terrain or raycast wheel contact.
- Do not apply tire forces or implement Pacejka, Brush, Fiala, or another tire
  model in this phase.
- Do not add steering, drive, brake, motor, differential, or suspension control
  behavior.
- Do not change the Phase 1A metadata schema beyond using the existing wheel
  shape/body arrays.
- Do not replace Newton's collision engine or create a separate analytical
  plane-support kernel.
- Do not implement hydroelastic wheel-terrain handling. Keep that as a later
  study for non-flat terrain quality.

## Inputs

Phase 1B depends on:

- `newton.wheeled.WheeledModelMetadata`
- `newton.Contacts`
- `Model.shape_body`
- `State.body_q`
- `Model.shape_material_mu`
- Optional `Contacts.force` when contact force reporting has been requested and
  the wrapped solver has populated it.

Relevant rigid contact fields:

- `rigid_contact_count`
- `rigid_contact_shape0`
- `rigid_contact_shape1`
- `rigid_contact_point0`
- `rigid_contact_point1`
- `rigid_contact_normal`
- `rigid_contact_margin0`
- `rigid_contact_margin1`

`rigid_contact_normal` points from shape 0 toward shape 1. `rigid_contact_point0`
and `rigid_contact_point1` are body-frame contact points on the respective
shapes.

## Contact Interpretation

A rigid contact is a wheel contact when either contact shape is present in the
Phase 1A wheel shape table.

For each active wheel contact:

- `wheel_shape_index` is the shape from the contact pair that maps to a wheel.
- `terrain_shape_index` is the other shape in the pair. The name is a diagnostic
  convention; the shape can be any non-wheel counterpart supported by Newton.
- `wheel_body_index` is `Model.shape_body[wheel_shape_index]`.
- The wheel contact point is the contact point associated with the wheel shape,
  transformed from body frame into world frame with `State.body_q`.
- The counterpart contact point is transformed the same way for diagnostics and
  patch-center estimation.
- The support normal acting on the wheel is:
  - `-rigid_contact_normal` when the wheel is `shape0`;
  - `rigid_contact_normal` when the wheel is `shape1`.

This orientation rule is part of the Phase 1B contract and should be tested
directly with synthetic shape ordering.

## Output Data

Add a contact-patch data object exposed through `newton.wheeled`. The exact
class name can be chosen during implementation, but the public surface should be
prefix-first and explicit, for example `WheelContactPatchState`.

The object should own flat per-wheel arrays:

| Field | Meaning |
| --- | --- |
| `active` | Whether the wheel has at least one active rigid contact |
| `contact_count` | Number of active rigid contacts grouped into the wheel patch |
| `terrain_shape_index` | Counterpart shape chosen for the wheel patch, or `-1` |
| `center` | Estimated world-space contact patch center [m] |
| `normal` | Estimated support normal acting on the wheel |
| `patch_u_extent` | Contact-cloud extent along the first patch tangent [m] |
| `patch_v_extent` | Contact-cloud extent along the second patch tangent [m] |
| `patch_area` | Estimated contact patch area [m^2] |
| `friction_mu_seed` | Friction coefficient seed from the counterpart shape |
| `normal_force` | Optional normal-force diagnostic [N], `0` if unavailable |

The arrays should be allocated on the model device. Host-side diagnostics can be
provided for tests and debugging, but the runtime update path should remain
device-side.

## Patch Estimation

For a flat reference scene, Phase 1B should use a deterministic contact-cloud
estimate:

1. Accumulate wheel contact points and support normals for each wheel.
2. Normalize the average support normal.
3. Build a stable tangent basis orthogonal to the normal.
4. Project grouped contact points into the tangent basis.
5. Compute min/max extents in both tangent directions.
6. Report patch area as `patch_u_extent * patch_v_extent`.

For a single contact, patch extents and area may be zero. A nonzero fallback can
be added later if a tire model needs it, but this phase should avoid inventing a
contact area that Newton did not provide.

If one wheel reports contacts against multiple counterpart shapes in the same
update, keep deterministic behavior. The first implementation may choose the
counterpart with the largest contact count and report the aggregate patch over
all wheel contacts, as long as this rule is documented and tested.

## Material And Force Diagnostics

`friction_mu_seed` should initially come from `Model.shape_material_mu` on the
counterpart shape. Combining wheel and terrain material values should be left to
the later tire/friction phase unless an existing solver convention is required
for correct diagnostics.

`normal_force` is diagnostic-only in Phase 1B. If `Contacts.force` is allocated
and the wrapped solver has populated it, convert the reported linear force
into force-on-wheel before projection. `Contacts.force` reports force on body0
by body1, so use it directly when the wheel is `shape0` and negate it when the
wheel is `shape1`. Project that force-on-wheel onto the support normal and
accumulate a nonnegative normal-force diagnostic per wheel. If force reporting
is unavailable, leave `normal_force` at `0` and do not synthesize a
penetration/stiffness fallback yet.

## Solver Relationship

The wrapper should support this flow:

1. Build or load a wheeled model.
2. Build Phase 1A wheeled metadata.
3. Create Newton contacts from the model or a `CollisionPipeline`.
4. Call `model.collide(state, contacts)`.
5. Update the Phase 1B wheel contact patch state from those contacts.
6. Step `SolverMuJoCo` with `use_mujoco_contacts=False` using the same Newton
   contact buffers.

Phase 1B may read contact force diagnostics after a solver step only when the
solver explicitly supports `update_contacts()`.

## Tests

Add `unittest` coverage for:

- Synthetic contacts that verify wheel shape lookup, shape0/shape1 ordering, and
  support-normal orientation.
- A simple flat wheel-ground fixture that verifies contact grouping, terrain
  shape reporting, contact center, normal, and patch-area diagnostics.
- Material lookup from counterpart `shape_material_mu`.
- Optional force diagnostics when `Contacts.force` is requested and populated by
  a solver that supports contact force reporting.
- `SolverMuJoCo(use_mujoco_contacts=False)` stepping with Newton-generated
  contacts while the wheeled layer reads the same contact buffers.
- Single-world and multi-world model construction.

Tests should fail before implementation and pass after implementation.

## Acceptance Criteria

- Phase 1B can update per-wheel patch diagnostics from Newton contacts generated
  by the collision pipeline.
- Contact grouping uses the Phase 1A wheel shape arrays and does not rediscover
  wheels from labels or manifests at runtime.
- The support normal is correct regardless of whether the wheel is shape 0 or
  shape 1 in the contact pair.
- Material and optional normal-force diagnostics are reported without applying
  any tire forces.
- A simple wheeled body remains supported by the wrapped rigid solver using
  Newton-generated contacts.
- The runtime update path contains no Python loops over contacts, wheels,
  vehicles, or worlds.

## Follow-Up Study

Open a later hydroelastic-contact study once the rigid-contact patch wrapper is
working. That study should focus on contact quality and patch estimation on
non-flat terrain, not on replacing the Phase 1B rigid-contact path.
