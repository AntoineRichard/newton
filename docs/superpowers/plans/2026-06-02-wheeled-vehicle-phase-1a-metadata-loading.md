# Wheeled Vehicle Phase 1A Metadata Loading Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load wheeled-vehicle metadata from the Phase 0 fixture manifest, register initial `wheeled:*` custom attributes, and build deterministic flat wheel tables without doing contact or collision interpretation.

**Done when:** The RC-car and Husky Phase 00 fixtures can be loaded through a Phase 1A metadata path that resolves wheel body labels, wheel shape labels, and optional suspension/steering joint labels into model indices; finalizes shape/body custom attributes; and exposes flat arrays or diagnostics suitable for Phase 1B contact grouping.

**Scope:** Phase 1A is metadata and indexing only. It does not create contact buffers, group contacts, estimate contact patches, apply forces, build tire models, command steering/drives, or change collision behavior.

---

## Inputs

Use the Phase 0 outputs as the source of fixture labels and dimensions:

- `newton/examples/assets/wheeled/manifest.json`
- `newton/examples/assets/wheeled/rc_car.usda`
- `newton/examples/assets/wheeled/husky.usda`
- `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md`

Relevant local APIs and patterns:

- `ModelBuilder.add_custom_attribute()` and `custom_attributes={...}` on `add_shape_*()` and `add_body()` already support namespaced model attributes.
- `Model.body_label`, `Model.shape_label`, and `Model.joint_label` preserve imported labels after finalization.
- `Model.shape_body` maps shape indices to body indices.
- `newton/_src/utils/wheeled_asset_inspection.py` can be reused as a test utility but should not become the runtime Phase 1A metadata path.

## Metadata Contract

Use these initial `wheeled:*` attributes:

| Attribute | Frequency | Type | Meaning |
| --- | --- | --- | --- |
| `wheeled:is_wheel` | `SHAPE` | `wp.bool` | Shape participates as a wheel in wheeled metadata tables |
| `wheeled:wheel_id` | `SHAPE` | `wp.int32` | Flat wheel index for wheel shapes, `-1` for non-wheel shapes |
| `wheeled:vehicle_id` | `SHAPE` | `wp.int32` | Vehicle index owning the wheel shape, `-1` for non-wheel shapes |
| `wheeled:wheel_radius` | `SHAPE` | `wp.float32` | Wheel radius [m] |
| `wheeled:wheel_width` | `SHAPE` | `wp.float32` | Wheel width [m] |
| `wheeled:is_steerable` | `SHAPE` | `wp.bool` | Wheel has an associated steering joint |
| `wheeled:is_driven` | `SHAPE` | `wp.bool` | Wheel is available for later drive/brake commands |
| `wheeled:is_wheel_body` | `BODY` | `wp.bool` | Body owns at least one wheel shape |
| `wheeled:wheel_body_id` | `BODY` | `wp.int32` | Flat wheel index for wheel bodies, `-1` for non-wheel bodies |

Store optional joint mappings in the Phase 1A wheel table rather than as joint custom attributes in this first step. This keeps the custom attribute surface small and avoids guessing future command/control layouts.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `newton/wheeled.py` | Create | Public import surface for Phase 1A metadata helpers |
| `newton/_src/wheeled/metadata.py` | Create | Internal dataclasses, manifest parsing, label resolution, wheel table construction |
| `newton/_src/wheeled/__init__.py` | Create | Internal package marker and exports |
| `newton/tests/test_wheeled_vehicle_metadata.py` | Create | Unit tests for custom attribute registration, manifest loading, label resolution, and wheel table diagnostics |
| `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md` | Modify | Add the Phase 1A plan link |
| `docs/superpowers/reports/2026-06-01-wheeled-vehicle-phase-0-assets.md` | Read | Source metadata decision notes |

Do not modify `newton/examples/assets/wheeled/*.usda` in Phase 1A unless a test proves the manifest labels no longer resolve.

## Task 1: Public Surface And Attribute Registration

**Files:**
- Create: `newton/tests/test_wheeled_vehicle_metadata.py`
- Create: `newton/_src/wheeled/metadata.py`
- Create: `newton/_src/wheeled/__init__.py`
- Create: `newton/wheeled.py`

- [ ] **Step 1: Write failing public import and registration tests**

Create tests that assert:

- `import newton.wheeled` succeeds.
- `newton.wheeled.register_wheeled_custom_attributes(builder)` registers the metadata contract above.
- A simple manually-created builder can add wheel and non-wheel shapes with `custom_attributes={"wheeled:is_wheel": True, ...}` and finalize a model whose `model.wheeled` namespace arrays contain expected values.
- Default values on non-wheel shapes/bodies are `False` for bool attributes and `-1` for id attributes.

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
```

Expected before implementation: import failure or missing registration helper.

- [ ] **Step 2: Implement `register_wheeled_custom_attributes()`**

Implement the registration helper using `ModelBuilder.CustomAttribute` and public `newton.Model.AttributeFrequency` values. Register explicit `default=-1` values for id attributes and rely on dtype defaults only for booleans and physical scalars. Keep the helper side-effect free except for registering attributes on the passed builder.

Recommended public exports in `newton/wheeled.py`:

```python
from ._src.wheeled.metadata import (
    WheeledAssetMetadata,
    WheeledModelMetadata,
    WheelMetadata,
    build_wheeled_metadata,
    load_wheeled_manifest,
    register_wheeled_custom_attributes,
)
```

Keep docstrings Google-style and use SI units for physical fields.

- [ ] **Step 3: Verify registration tests pass**

Run the focused test command again.

- [ ] **Step 4: Commit the registration surface**

Run `uvx pre-commit run -a`, then commit:

```bash
git add newton/wheeled.py newton/_src/wheeled/__init__.py \
  newton/_src/wheeled/metadata.py newton/tests/test_wheeled_vehicle_metadata.py
git commit -m "Add wheeled metadata attributes"
```

Commit body:

```text
Add the initial public wheeled metadata surface and custom attribute
registration helper. The registered attributes mark wheel shapes and bodies,
record flat wheel and vehicle ids, and store basic wheel dimensions for later
contact grouping.
```

## Task 2: Manifest Loading And Validation

**Files:**
- Modify: `newton/_src/wheeled/metadata.py`
- Modify: `newton/tests/test_wheeled_vehicle_metadata.py`

- [ ] **Step 1: Add failing manifest loader tests**

Test `load_wheeled_manifest(path)` with the Phase 0 manifest:

- returns two `WheeledAssetMetadata` entries named `rc_car` and `husky`;
- preserves `vehicle_type`;
- exposes wheel body, wheel shape, suspension joint, and steering joint labels;
- reads `wheel_radius_m` and `wheel_width_m` into per-asset defaults;
- rejects duplicate asset names, missing files, and mismatched wheel body/shape list lengths.

Use temporary manifests for negative tests rather than editing the real manifest.

- [ ] **Step 2: Implement typed manifest parsing**

Recommended dataclasses:

```python
@dataclass(frozen=True)
class WheeledAssetMetadata:
    name: str
    file: Path
    vehicle_type: str
    wheel_body_labels: tuple[str, ...]
    wheel_shape_labels: tuple[str, ...]
    suspension_joint_labels: tuple[str, ...]
    steering_joint_labels: tuple[str, ...]
    wheel_radius: float
    wheel_width: float
```

Validation should raise `ValueError` with the asset name and failing key. Keep parsing stdlib-only.

- [ ] **Step 3: Verify manifest loader tests pass**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
```

- [ ] **Step 4: Commit manifest loading**

Run `uvx pre-commit run -a`, then commit:

```bash
git add newton/_src/wheeled/metadata.py newton/tests/test_wheeled_vehicle_metadata.py
git commit -m "Load wheeled fixture metadata"
```

Commit body:

```text
Parse the Phase 0 wheeled fixture manifest into typed metadata objects and
validate the label and dimension contract before any model-index resolution.
This keeps fixture intake separate from collision and contact logic.
```

## Task 3: Builder Annotation From Manifest

**Files:**
- Modify: `newton/_src/wheeled/metadata.py`
- Modify: `newton/tests/test_wheeled_vehicle_metadata.py`

- [ ] **Step 1: Add failing builder annotation tests**

Load `rc_car.usda` and `husky.usda` into a `ModelBuilder`, call `register_wheeled_custom_attributes(builder)`, and then call an annotation helper such as `apply_wheeled_manifest_metadata(builder, asset_metadata, vehicle_id=0)`. Tests should assert before finalization that:

- all manifest wheel shape labels resolve to shape indices;
- all manifest wheel body labels resolve to body indices;
- shape custom attributes are populated for wheel shapes only;
- body custom attributes are populated for wheel bodies only;
- missing labels raise `ValueError` with the missing label and label kind;
- RC front wheels are steerable and Husky wheels are not steerable.

Use `SchemaResolverPhysx()` when loading USDA so the builder state matches Phase 0 validation.

- [ ] **Step 2: Implement annotation helper**

Recommended function:

```python
def apply_wheeled_manifest_metadata(
    builder: newton.ModelBuilder,
    asset: WheeledAssetMetadata,
    vehicle_id: int,
    *,
    driven_by_default: bool = True,
) -> list[WheelMetadata]:
    ...
```

Implementation notes:

- Build `dict[str, int]` lookup maps from `builder.body_label`, `builder.shape_label`, and `builder.joint_label`.
- Use `builder.shape_body[shape_index]` to verify each wheel shape is attached to the corresponding wheel body.
- Populate builder custom attribute storage through the same paths used by existing custom attribute support. If direct post-import mutation is awkward, prefer a focused helper that appends values to registered custom attribute storage rather than re-importing assets.
- Keep optional steering/suspension joint ids in returned `WheelMetadata`; do not require joint custom attributes yet.

Recommended `WheelMetadata` fields:

```python
@dataclass(frozen=True)
class WheelMetadata:
    wheel_id: int
    vehicle_id: int
    body_index: int
    shape_index: int
    steering_joint_index: int
    suspension_joint_index: int
    radius: float
    width: float
    is_steerable: bool
    is_driven: bool
```

Use `-1` for missing optional joint indices.

- [ ] **Step 3: Verify annotation tests pass**

Run the focused tests.

- [ ] **Step 4: Commit builder annotation**

Run `uvx pre-commit run -a`, then commit:

```bash
git add newton/_src/wheeled/metadata.py newton/tests/test_wheeled_vehicle_metadata.py
git commit -m "Annotate wheeled fixture builders"
```

Commit body:

```text
Resolve Phase 0 manifest labels against imported fixture builders and annotate
wheel shapes and bodies with wheeled metadata attributes. The returned wheel
metadata records optional steering and suspension joints without adding contact
logic.
```

## Task 4: Flat Wheel Table Construction

**Files:**
- Modify: `newton/_src/wheeled/metadata.py`
- Modify: `newton/tests/test_wheeled_vehicle_metadata.py`

- [ ] **Step 1: Add failing wheel table tests**

Add tests for `build_wheeled_metadata(model, wheel_metadata)` or equivalent that assert:

- RC car produces four wheels with deterministic ordering matching the manifest;
- Husky produces four wheels with deterministic ordering matching the manifest;
- shape/body index arrays match model labels;
- radius and width arrays match manifest dimensions;
- RC front wheel entries are steerable with valid steering joint indices;
- Husky entries have `-1` steering and suspension joint indices;
- diagnostics can be converted to JSON-compatible dictionaries for reports/tests.

- [ ] **Step 2: Implement `WheeledModelMetadata`**

Recommended dataclass:

```python
@dataclass(frozen=True)
class WheeledModelMetadata:
    wheel_count: int
    vehicle_count: int
    wheel_shape_indices: tuple[int, ...]
    wheel_body_indices: tuple[int, ...]
    steering_joint_indices: tuple[int, ...]
    suspension_joint_indices: tuple[int, ...]
    wheel_radius: tuple[float, ...]
    wheel_width: tuple[float, ...]
    is_steerable: tuple[bool, ...]
    is_driven: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        ...
```

This object can stay host-side in Phase 1A. Do not introduce Warp kernels until Phase 1B needs device-side contact grouping.

- [ ] **Step 3: Verify wheel table tests pass**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
```

- [ ] **Step 4: Commit flat wheel tables**

Run `uvx pre-commit run -a`, then commit:

```bash
git add newton/_src/wheeled/metadata.py newton/tests/test_wheeled_vehicle_metadata.py
git commit -m "Build wheeled metadata tables"
```

Commit body:

```text
Build deterministic host-side wheel metadata tables from annotated fixture
models. The tables expose wheel shape, body, steering, suspension, and dimension
fields needed by Phase 1B contact grouping.
```

## Task 5: Multi-World Metadata Checks

**Files:**
- Modify: `newton/_src/wheeled/metadata.py`
- Modify: `newton/tests/test_wheeled_vehicle_metadata.py`

- [ ] **Step 1: Add failing multi-world tests**

Build a model with two loaded fixture instances, for example two RC cars or one RC car plus one Husky with separate root transforms. Assert:

- `vehicle_id` values distinguish fixture instances;
- `wheel_id` values are globally flat and deterministic;
- all shape/body indices are unique across instances;
- diagnostics include per-vehicle wheel counts.

If `ModelBuilder.add_usd()` label collisions make duplicated assets ambiguous, document that finding in the test and use one RC car plus one Husky for Phase 1A.

- [ ] **Step 2: Extend metadata builder for multiple assets**

Recommended function:

```python
def apply_wheeled_manifest(
    builder: newton.ModelBuilder,
    manifest_path: str | Path,
    *,
    asset_names: Sequence[str] | None = None,
) -> list[WheelMetadata]:
    ...
```

This can be a convenience around `load_wheeled_manifest()` and `apply_wheeled_manifest_metadata()` for tests and examples. Avoid runtime assumptions about vehicles per world; only build flat metadata for whatever labels are present.

- [ ] **Step 3: Verify multi-world tests pass**

Run the focused metadata tests.

- [ ] **Step 4: Commit multi-world metadata checks**

Run `uvx pre-commit run -a`, then commit:

```bash
git add newton/_src/wheeled/metadata.py newton/tests/test_wheeled_vehicle_metadata.py
git commit -m "Support multi-vehicle wheeled metadata"
```

Commit body:

```text
Extend wheeled metadata loading to multiple fixture instances and verify flat
wheel ids, vehicle ids, and resolved shape/body indices remain deterministic.
This prepares Phase 1B contact grouping without adding contact logic.
```

## Task 6: Roadmap And API Docs

**Files:**
- Modify: `docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`
- Possibly modify generated API docs if public symbols require it

- [ ] **Step 1: Confirm the roadmap plan link**

Confirm the Phase 1A section links this plan immediately below the heading:

```markdown
Plan: `docs/superpowers/plans/2026-06-02-wheeled-vehicle-phase-1a-metadata-loading.md`
```

- [ ] **Step 2: Generate API documentation if required**

Because `newton/wheeled.py` introduces public symbols, run:

```bash
uv run docs/generate_api.py
```

If the docs generator changes files, include those changes in the commit. If it fails for an environment reason, record the failure in the final handoff.

- [ ] **Step 3: Final verification**

Run:

```bash
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_metadata
uv run --extra dev -m newton.tests -k test_wheeled_vehicle_assets
uvx pre-commit run -a
```

Expected: focused metadata tests pass, existing Phase 0 asset tests still pass, and pre-commit exits 0.

- [ ] **Step 4: Commit roadmap/API docs**

Commit message:

```bash
git commit -m "Document wheeled metadata loading"
```

Commit body:

```text
Link the Phase 1A metadata-loading plan from the roadmap and refresh public API
documentation for the new wheeled metadata helpers.
```

## Handoff To Phase 1B

Phase 1B should consume `WheeledModelMetadata.wheel_shape_indices` and related arrays for contact grouping. It should not rediscover wheel labels, reread the manifest at runtime, or add raycast fallback paths. The open Phase 1B questions are how Newton contacts expose shape pairs, normals, contact points, and solver/material data, and how to keep contact grouping device-side for many vehicles.
