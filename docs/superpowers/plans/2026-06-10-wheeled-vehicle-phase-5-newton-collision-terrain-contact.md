# Wheeled Vehicle Phase 5 Newton Collision Terrain Contact Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Validate Newton collision contacts as the wheel-terrain patch source
on non-flat terrain, with explicit contact-gap setup and cylinder tire contact
quality checks. The phase should determine whether the existing rigid contact
cloud is sufficient for tire inputs, where it degenerates, and which setup
conventions examples/tests must enforce.

**Done when:** Wheel contact patches are validated on a small set of
primitive and mesh terrain cases; gap-zero wheel contact setup is tested;
cylinder wheel contacts are classified as point-like, line-like, or area-like;
any cylinder contact line-alignment source is identified with an explicit
opt-out decision; and tire-force examples can consume the terrain patches
without non-finite state or obvious contact instability.

**Scope:** Phase 5 is terrain contact validation and contact-quality hardening.
It does not add raycasts, hydroelastic contacts, Brush/Fiala/Pacejka tire models,
powertrain modules, physical wheel spin, or a new vehicle controller.

---

## Inputs

Roadmap:

- `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`

Spec:

- `docs/superpowers/specs/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md`

Implementation inputs:

- `newton.wheeled.WheeledModelMetadata`
- `newton.wheeled.WheelContactPatchState`
- `newton.wheeled.WheelTireControl`
- `newton.wheeled.WheeledVehicleLayout`
- `Model.shape_gap`, `Model.shape_margin`, and material arrays
- Newton collision contacts from `Model.collide()`
- Phase 00 RC car and Husky fixtures

## Design Notes

Start with tests and instrumentation before changing the contact estimator. The
current estimator already reports contact count, terrain shape, patch center,
normal, extents, area, material friction seed, and normal force. Phase 5 should
first answer whether those fields are reliable on terrain.

Treat the solver guidance about contact gap as a hypothesis to test:

- wheel and terrain `shape_gap` should be `0.0` in the baseline terrain-contact
  setup;
- tests should prove the intended values are used;
- compare against a nonzero gap only in a controlled sweep;
- keep `shape_margin` visible as a separate variable.

Treat cylinder contacts as a contact-manifold quality question. Cylinder-plane
contact may be line-like by construction. That is acceptable if it is stable and
if downstream tire code has enough center/normal/extents information. If it is
not stable, document whether the likely fix is contact setup, patch estimation,
fixture geometry, or a later hydroelastic path.

Also audit whether line-like contact clouds are being forced by implementation
heuristics. The current collision code has a plane-cylinder analytical helper
and an axial-shape rolling stabilization post-process. Phase 5 should determine
whether wheel-terrain contacts are affected by either path before adding an API
or changing defaults.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `docs/superpowers/specs/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md` | Create | Phase 5 contact-quality spec |
| `docs/superpowers/plans/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md` | Create | Executable Phase 5 plan |
| `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md` | Modify | Mark Phase 4 complete and link Phase 5 spec/plan |
| `newton/tests/test_wheeled_vehicle_terrain_contact.py` | Create | Gap, terrain, and cylinder contact-quality tests |
| `newton/_src/wheeled/contact_patch.py` | Modify if needed | Optional contact quality diagnostics or estimator hardening |
| `newton/_src/wheeled/mujoco.py` | Modify if needed | Optional helper coverage for normal-only wheel contact setup |
| `newton/_src/geometry/collision_primitive.py` | Modify only if justified | Optional scoped plane-cylinder contact behavior |
| `newton/_src/geometry/collision_core.py` | Modify only if justified | Optional scoped axial rolling-stabilization opt-out |
| `newton/examples/wheeled/example_wheeled_terrain_contact.py` | Create if useful | Debug/validation example for non-flat terrain patches |
| `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-gap-line-patch-generation.md` | Create | Gap, line, and patch generation findings |
| `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-contact-observability.md` | Create | MuJoCo contact-source observability findings |
| `README.md` | Modify if example added | Register terrain-contact example |
| `CHANGELOG.md` | Modify if public behavior changes | User-facing changes only |

## Task 00: Roadmap And Spec Alignment

**Files:**

- Modify: `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`
- Create: `docs/superpowers/specs/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md`
- Create: `docs/superpowers/plans/2026-06-10-wheeled-vehicle-phase-5-newton-collision-terrain-contact.md`

- [x] **Step 1: Update current roadmap state**

Replace stale text saying Phase 4 is next. Record that Phase 4 command mapping
exists and that Phase 5 is the next validation step.

- [x] **Step 2: Link Phase 5 spec and plan**

Add spec and plan links under the Phase 5 roadmap heading.

- [x] **Step 3: Record contact-gap and cylinder notes**

Ensure the roadmap mentions:

- wheel-terrain `shape_gap = 0.0` as the baseline setup to test;
- cylinder tire contacts must be checked for line-like or point-like manifolds;
- any forced cylinder contact line alignment must be audited before deciding
  whether to add a scoped opt-out.

## Task 01: Contact Gap Audit And Test Fixture

**Files:**

- Create: `newton/tests/test_wheeled_vehicle_terrain_contact.py`

- [x] **Step 1: Build a minimal wheel-terrain contact fixture**

Create a deterministic test builder with one cylinder wheel shape and one terrain
shape. Register wheeled metadata, build a model, place the wheel in shallow
contact, call `Model.collide()`, and update `WheelContactPatchState`.

- [x] **Step 2: Assert baseline gap-zero setup**

Set wheel and terrain contact gaps to `0.0` before finalization or through the
supported model setup path. Assert the resulting model has the intended wheel and
terrain `shape_gap` values.

- [x] **Step 3: Compare against nonzero gap**

Add a small controlled gap sweep only for diagnostics. Record how contact count,
center, normal, extents, and area change. The test should not assume nonzero gap
is better; it should make any gap-dependent behavior visible.

## Task 02: Cylinder Contact-Manifold Quality Tests

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_terrain_contact.py`
- Modify if needed: `newton/_src/wheeled/contact_patch.py`

- [x] **Step 1: Audit cylinder contact generation paths**

Inspect and summarize the source of cylinder wheel contact points for the test
cases:

- analytical plane-cylinder contact generation;
- GJK/MPR cylinder contact manifolds for primitive or mesh terrain;
- axial-shape rolling stabilization projection for axial-vs-discrete pairs;
- contact sorting/reduction before `WheelContactPatchState` consumes contacts.

Do not change collision behavior in this step. The output should identify
whether line-like contacts are geometric, heuristic, or caused by reduction.

- [x] **Step 2: Classify flat cylinder-plane contacts**

For a cylinder wheel on a flat plane, record contact count, patch extents, patch
area, and extent ratio. Classify the result as point-like, line-like, or
area-like using deterministic thresholds local to the test.

- [x] **Step 3: Detect forced line alignment**

Add a diagnostic that measures whether raw contact points are rank-one in the
patch tangent plane. Use a best-fit-line residual or equivalent extent-ratio
metric, and record whether the alignment appears before or after patch
reduction.

- [x] **Step 4: Check frame-to-frame stability**

Run several low-speed frames and assert finite patch center, normal, extents,
and area. Measure jitter in patch center and normal. Keep thresholds loose until
observed behavior is understood.

- [x] **Step 5: Decide whether estimator or collision changes are needed**

If cylinder contacts are stable but line-like, keep the estimator and document
that area may be near zero for rigid cylinder-plane contact. If unstable, add a
small estimator hardening change with focused tests.

Potential hardening options, only if tests justify them:

- deterministic contact sorting before reduction;
- area floor derived from tire width and penetration/contact-count diagnostics;
- separate longitudinal/lateral extent diagnostics aligned with wheel axes;
- fixture geometry adjustment such as a rounded tire proxy, documented as an
  asset choice rather than a solver assumption.

Potential collision options, only if tests show a forced line projection harms
wheel-terrain patches:

- per-shape or per-pair contact mode for wheel-terrain cylinder contacts;
- public `newton.wheeled` setup helper that opts wheel-terrain pairs out of
  axial rolling projection without exposing internal modules;
- alternate cylinder contact sampling for wheel shapes while preserving the
  existing default for ordinary cylinder contacts.

## Task 03: Primitive And Mesh Terrain Cases

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_terrain_contact.py`

- [x] **Step 1: Add inclined plane/ramp case**

Assert patch normals track the terrain normal and tire forces remain finite while
a wheel or simple vehicle rests/drives on a slope.

- [x] **Step 2: Add curb or low box ridge case**

Assert contact state remains finite when contacts transition between flat ground
and a sharp primitive feature.

- [x] **Step 3: Add triangle mesh ripple and jump cases**

Use small deterministic mesh terrain with a shallow bump/ripple and a jump-like
ramp profile. Assert contact shape/material data is populated and patch
diagnostics remain finite.

## Task 04: Vehicle-Level Terrain Validation

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_terrain_contact.py`
- Optional create: `newton/examples/wheeled/example_wheeled_terrain_contact.py`

- [x] **Step 1: Run RC car on a simple slope or ramp**

Use the Phase 4 car command mapping and Phase 3 tire path. Assert finite state,
active wheel patches, and plausible terrain normals.

- [x] **Step 2: Run Husky on a simple slope or ridge**

Use direct left/right skid-steer commands. Assert finite state, active wheel
patches, and stable patch diagnostics.

- [x] **Step 3: Verify multi-world batching**

Replicate a terrain-contact scene and assert patch diagnostics work across
multiple worlds without Python runtime loops over vehicles or wheels.

## Task 05: MuJoCo Contact Observability Study

**Files:**

- Modify: `newton/tests/test_wheeled_vehicle_terrain_contact.py`
- Create: `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-gap-line-patch-generation.md`
- Create: `docs/superpowers/reports/2026-06-10-wheeled-vehicle-phase-5-contact-observability.md`

- [x] **Step 1: Compare Newton-generated contacts converted to MuJoCo Warp**

Run the normal Phase 3/4 path with `use_mujoco_contacts=False`. Record contact
counts, normals, patch extents, and tire-force stability.

- [x] **Step 2: Compare MuJoCo-native contacts where feasible**

Run a matching scene with MuJoCo-native contacts only as an observability study.
Do not change the architecture based on this without a concrete finding.

- [x] **Step 3: Record findings**

Document whether converted Newton contacts expose enough data for the tire path,
and whether MuJoCo-native contacts reveal a gap, margin, or cylinder-manifold
issue worth addressing.

## Task 06: Docs, Examples, And Verification

**Files:**

- Modify: `README.md` only if a new example is added
- Modify: `CHANGELOG.md` only if public behavior changes or a public example is added

- [x] **Step 1: Add user-facing docs only for user-facing changes**

If Phase 5 only adds tests and internal validation, keep changelog changes out.
If it adds a public helper or example, add a concise changelog entry.

- [x] **Step 2: Run focused verification**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_terrain_contact
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_tire
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_drive_modes
git diff --check
```

If a terrain-contact example is added, also run:

```bash
uv run --extra dev -m newton.examples wheeled_terrain_contact --viewer null --test --num-frames 180 --world-count 2 --device cpu --quiet
uv run --extra dev -m newton.tests -k wheeled_terrain_contact
```

## Out Of Scope

- Raycast contacts.
- Hydroelastic contacts beyond recording follow-up questions for Phase 6.
- New tire force laws.
- Powertrain modules.
- Physical wheel spin.
- Replacing cylinder wheel fixtures without measured evidence.

## Exit Criteria

- Gap-zero wheel-terrain setup is tested and documented.
- Cylinder tire contact manifolds are measured and classified.
- Any forced cylinder contact line alignment is identified and either accepted
  as stable, handled in patch estimation, or given a scoped opt-out design.
- Wheel contact patches remain finite and usable on representative non-flat
  terrain.
- Terrain shape and material friction seed data feed the tire path.
- Vehicle-level RC car and Husky terrain tests remain stable.
- Any contact-quality limitations are documented with a concrete follow-up path.
