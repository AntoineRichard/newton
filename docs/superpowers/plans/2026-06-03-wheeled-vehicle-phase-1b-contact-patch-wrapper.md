# Wheeled Vehicle Phase 1B Contact Patch Wrapper Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Newton-contact wrapper that consumes Phase 1A wheel metadata,
groups active rigid contacts by wheel shape, and reports per-wheel contact patch
diagnostics while leaving normal support to the wrapped rigid solver.

**Done when:** A simple wheeled fixture produces stable per-wheel contact center,
support normal, patch extents, patch area, terrain shape, material friction seed,
and optional normal-force diagnostics from Newton contacts. The same contact
buffers can be stepped by `SolverMuJoCo(use_mujoco_contacts=False)`.

**Scope:** Phase 1B is contact interpretation only. It does not apply tire
forces, add raycasts, command steering or drives, modify suspension dynamics, or
change the Phase 1A wheel metadata contract.

---

## Inputs

Use the Phase 1A outputs and existing Newton contact APIs:

- `newton.wheeled.WheeledModelMetadata`
- `newton.Contacts`
- `Model.shape_body`
- `State.body_q`
- `Model.shape_material_mu`
- Optional `Contacts.force`
- `SolverMuJoCo(use_mujoco_contacts=False)`

Spec:

- `docs/superpowers/specs/2026-06-03-wheeled-vehicle-phase-1b-contact-patch-wrapper.md`

Relevant handoff from Phase 1A:

- Consume `WheeledModelMetadata.wheel_shape_indices` and related flat arrays.
- Do not rediscover wheels from labels or manifests at runtime.
- Do not add raycast fallbacks.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/_src/wheeled/contact_patch.py` | Create | Internal contact-patch data object, kernels, and update helper |
| `newton/_src/wheeled/__init__.py` | Modify | Internal exports |
| `newton/wheeled.py` | Modify | Public Phase 1B import surface |
| `newton/tests/test_wheeled_vehicle_contact_patch.py` | Create | Unit tests for grouping, orientation, patch estimates, materials, force diagnostics, and solver flow |
| `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md` | Modify | Link the Phase 1B spec and plan |

Keep examples/docs from importing `newton._src`. Public user-facing symbols must
be re-exported through `newton/wheeled.py`.

## Task 00: Contact API Audit And Failing Test Skeleton

**Files:**

- Create: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Confirm rigid contact fields and conventions**

Record the fields used by the implementation in the test module comments or
assertions:

- `rigid_contact_shape0`
- `rigid_contact_shape1`
- `rigid_contact_point0`
- `rigid_contact_point1`
- `rigid_contact_normal`
- `rigid_contact_count`

The implementation must treat `rigid_contact_normal` as shape0-to-shape1 and
convert it into a support normal acting on the wheel.

- [ ] **Step 2: Write failing public import tests**

Assert that the planned public API can be imported from `newton.wheeled`.
Candidate public names:

```python
from newton.wheeled import (
    WheelContactPatchState,
    update_wheel_contact_patches,
)
```

Keep names prefix-first if adjusted during implementation.

- [ ] **Step 3: Write failing synthetic grouping tests**

Create a small builder with two wheel shapes and one terrain shape, build Phase
1A metadata, then populate a `Contacts` object with synthetic shape pairs.

Assert:

- contacts where the wheel is `shape0` and contacts where the wheel is `shape1`
  both group into the correct wheel;
- support normal is flipped only when the wheel is `shape0`;
- non-wheel/non-wheel contacts are ignored;
- `contact_count`, `active`, and `terrain_shape_index` are deterministic.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

Expected before implementation: import failure or missing contact-patch helpers.

## Task 01: Public Data Surface

**Files:**

- Create: `newton/_src/wheeled/contact_patch.py`
- Modify: `newton/_src/wheeled/__init__.py`
- Modify: `newton/wheeled.py`
- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Add `WheelContactPatchState`**

Create a public, readable entrypoint for the wheel-indexed reduction of Newton
`Contacts`. This object is not a collision object and should not replace or wrap
`Contacts`; it is a reusable per-wheel output buffer sized from Phase 1A
metadata. It should own flat per-wheel arrays on the model device:

- `active`
- `contact_count`
- `terrain_shape_index`
- `center`
- `normal`
- `patch_u_extent`
- `patch_v_extent`
- `patch_area`
- `friction_mu_seed`
- `normal_force`

Use SI units in public docstrings. Keep the object responsible for allocation,
clearing, and validation of its own wheel count, not for discovering wheel
metadata. Any temporary arrays needed for reductions may be internal, but avoid
exposing scratch fields as public API.

- [ ] **Step 2: Add `update_wheel_contact_patches()`**

Add a helper that accepts:

- `model`
- `state`
- `contacts`
- `wheeled_metadata`
- `patch_state`

The helper should treat `contacts` as the source of truth and `patch_state` as
the destination reduction. It should launch kernels only. Host-side convenience
diagnostics can be added separately for tests, but the update path must not loop
over contacts or wheels in Python.

- [ ] **Step 3: Export through public modules**

Re-export the Phase 1B public names from `newton/wheeled.py`. Do not import
`newton._src` from examples or tests except where tests intentionally inspect
internal behavior.

- [ ] **Step 4: Verify import tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

The import tests should now pass, while grouping and patch tests may still fail.

## Task 02: Wheel Shape Lookup And Contact Grouping

**Files:**

- Modify: `newton/_src/wheeled/contact_patch.py`
- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Build a shape-to-wheel lookup**

Create a device lookup array sized to `model.shape_count`, initialized to `-1`,
then filled from `WheeledModelMetadata.wheel_shape_indices`. This prevents
runtime label or manifest lookups.

- [ ] **Step 2: Implement grouping kernel**

Launch one kernel over active rigid contacts. For each contact:

- read `shape0` and `shape1`;
- look up each shape in the shape-to-wheel table;
- ignore non-wheel/non-wheel contacts;
- assign the wheel id, terrain shape id, wheel contact point, counterpart point,
  and support-normal sign for wheel contacts.

If both shapes are wheels, keep deterministic behavior. The first implementation
may ignore wheel-wheel contacts or choose the lower wheel id, as long as tests
document the choice.

- [ ] **Step 3: Accumulate per-wheel counts and first terrain shape**

Use device-side atomics to update per-wheel contact counts and active flags. For
the reported terrain shape, use a deterministic rule such as minimum counterpart
shape id, or add a second reduction pass if the largest-contact-count terrain
rule from the spec is implemented immediately. Avoid order-dependent
first-contact writes.

- [ ] **Step 4: Verify synthetic grouping tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

The synthetic grouping and orientation tests should pass.

## Task 03: Patch Center, Normal, Extents, And Area

**Files:**

- Modify: `newton/_src/wheeled/contact_patch.py`
- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Transform body-frame contact points to world frame**

Use `Model.shape_body` and `State.body_q` to transform the wheel and counterpart
contact points into world coordinates. Prefer the wheel point for per-wheel
contact cloud accumulation, and keep the counterpart point available if it makes
the center estimate more stable.

- [ ] **Step 2: Accumulate center and support normal**

Accumulate contact points and support normals per wheel, then normalize by
`contact_count`. For inactive wheels, write deterministic defaults:

- `active=False`
- `contact_count=0`
- `terrain_shape_index=-1`
- zero vectors for `center` and `normal`
- zero extents, area, material, and normal force

- [ ] **Step 3: Compute deterministic tangent extents**

Build a stable tangent basis from the averaged support normal. Project grouped
contact points onto that basis, compute min/max values, and write:

- `patch_u_extent`
- `patch_v_extent`
- `patch_area`

For one contact, area may be zero.

- [ ] **Step 4: Add flat fixture tests**

Create a simple wheel-on-plane or cylinder-on-plane fixture using `ModelBuilder`.
Run Newton collision, update patches, and assert:

- at least one wheel contact is active;
- support normal is approximately upward for a flat ground plane;
- patch center is near the wheel-ground contact region;
- patch area is finite and nonnegative.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

## Task 04: Material And Optional Force Diagnostics

**Files:**

- Modify: `newton/_src/wheeled/contact_patch.py`
- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Add friction seed lookup**

Read `Model.shape_material_mu` from the terrain/counterpart shape and write it
to `friction_mu_seed`. Keep this diagnostic separate from any tire-force
calculation.

- [ ] **Step 2: Add optional normal-force accumulation**

If `Contacts.force` is allocated and has been populated by a solver, convert the
reported linear force into force-on-wheel before projection. `Contacts.force`
reports force on body0 by body1, so use it directly when the wheel is `shape0`
and negate it when the wheel is `shape1`. Project force-on-wheel onto the
support normal and accumulate a nonnegative value per wheel. If `Contacts.force`
is not allocated, leave `normal_force` at zero.

Do not add a penetration/stiffness fallback in Phase 1B.

- [ ] **Step 3: Test material and force behavior**

Add tests for:

- material seed follows the counterpart shape;
- missing force buffers leave `normal_force` at zero;
- populated force buffers produce deterministic accumulated normal force in a
  synthetic contact test.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

## Task 05: Wrapped Solver Flow

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`

- [ ] **Step 1: Add MuJoCo Newton-contact flow test**

Create a simple wheeled body on a ground plane and step:

```python
model.collide(state_0, contacts)
update_wheel_contact_patches(model, state_0, contacts, wheeled_metadata, patch_state)
solver.step(state_0, state_1, control, contacts, dt)
```

Use `SolverMuJoCo(use_mujoco_contacts=False)` so Newton contacts feed the solver.

- [ ] **Step 2: Assert normal support remains solver-owned**

After several steps, assert the body remains above the ground. Do not apply any
wheeled-layer normal or tire force in this test.

- [ ] **Step 3: Add optional force-reporting path only if supported**

If the selected solver supports `update_contacts()` with `Contacts.force`, add a
small assertion that force diagnostics can be read after stepping. If solver
support is unavailable or unstable, keep force reporting covered by synthetic
tests and document the limitation.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
```

## Task 06: Multi-World Verification And Regression Suite

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_contact_patch.py`
- Modify: `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`

- [ ] **Step 1: Add multi-world coverage**

Build a multi-world model from the same simple fixture and verify:

- shape-to-wheel lookup remains correct after model finalization;
- contacts from multiple worlds group into the expected flat wheel ids;
- inactive wheels remain deterministic.

- [ ] **Step 2: Run focused regression tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_contact_patch
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
```

- [ ] **Step 3: Run formatting and linting**

Run:

```bash
uvx pre-commit run -a
```

- [ ] **Step 4: Commit Phase 1B**

Stage only files changed for Phase 1B:

```bash
git add newton/wheeled.py newton/_src/wheeled/__init__.py   newton/_src/wheeled/contact_patch.py   newton/tests/test_wheeled_vehicle_contact_patch.py   docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md
git commit -m "Add wheeled contact patch wrapper"
```

Suggested commit body:

```text
Group Newton rigid contacts by Phase 1A wheel metadata and report per-wheel
contact patch diagnostics for the wheeled solver roadmap. The new path uses
Newton-generated contacts, leaves normal support to the wrapped solver, and
keeps tire-force application for later phases.
```

## Open Questions

- Whether `Contacts.force` should be part of the normal Phase 1B update or only
  a post-solver diagnostic call depends on solver support. Keep the initial API
  tolerant of unavailable force buffers.
- Contact capacity and atomic contention should be inspected once tests cover
  many worlds. The first implementation can optimize only after correctness is
  established.
- Combining wheel and terrain material friction values belongs in the tire-force
  phase unless an existing solver convention needs to be mirrored for
  diagnostics.
