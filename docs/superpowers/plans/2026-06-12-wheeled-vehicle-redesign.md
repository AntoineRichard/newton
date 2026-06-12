# Wheeled Vehicle Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, faithful wheeled-vehicle layer for Newton: the wrapped MuJoCo solver owns collision (Newton-detected) + normal support, while a cohesive `WheeledVehicles` controller owns analytical wheel spin and a brush combined-slip tire model, supporting heterogeneous vehicles in one model.

**Architecture:** Co-simulation split. `use_mujoco_contacts=False` so Newton's collision pipeline detects contacts (with a new scoped `preserve_contact_footprint` shape flag that stops cylinder footprints collapsing to a line), MuJoCo solves them. The controller injects tire wrenches into `state.body_f` and integrates wheel spin analytically. Everything runs as batched Warp kernels over flat device arrays; vehicle heterogeneity is data-driven via a per-vehicle `drive_mode` enum.

**Tech Stack:** Python, NVIDIA Warp (`warp-lang`), `mujoco-warp`, Newton internal APIs (`newton._src`), unittest.

**Source spec:** `docs/superpowers/specs/2026-06-12-wheeled-vehicle-redesign-design.md`

---

## Conventions for every task

- **Run tests** with: `uv run --extra dev -m newton.tests -k <pattern>`
  (the runner discovers `newton/tests/test_*.py`; it runs each test on `cpu` and `cuda:0`).
- **Lint/format** before each commit: `uvx pre-commit run -a` (or at least on touched files).
- **Commit** with `git -c commit.gpgsign=false commit` (GPG signing has no TTY here),
  imperative subject ≤ ~50 chars, body wrapped at 72, trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Never commit to `main`.** Stay on `antoiner/wheeled-vehicle-design`.
- **Warp array annotations**: `wp.array[T]`, not `wp.array(dtype=T)`. PEP 604 unions.
- **No Python loops over wheels/vehicles/contacts in runtime methods** — kernels only.
- **TDD**: write the failing test, see it fail, implement minimally, see it pass, commit.
- A device kernel "unit test" launches the kernel (or a thin wrapper kernel around a
  `@wp.func`) on synthetic flat arrays and asserts on `.numpy()` of the outputs.

## File structure

**Newton core (Phase 1 — separable, landed first):**
- Modify `newton/_src/geometry/flags.py` — add `ShapeFlags.PRESERVE_CONTACT_FOOTPRINT`.
- Modify `newton/_src/sim/builder.py` — `ShapeConfig.preserve_contact_footprint`; OR the flag into `shape_flags` at shape creation.
- Modify `newton/_src/geometry/narrow_phase.py` — per-shape branch around the plane-cylinder routing (~line 439); thread `shape_flags` to post-process.
- Modify `newton/_src/geometry/collision_core.py` — `post_process_axial_on_discrete_contact` skips axial projection for flagged shapes.
- Create `newton/tests/test_collision_preserve_footprint.py`.

**Vehicle package (Phases 2–7 — new, parallel to `newton/_src/wheeled/`):**
- `newton/_src/vehicles/__init__.py` — internal aggregation.
- `newton/_src/vehicles/metadata.py` — `vehicle:*` custom attributes; flat tables; `register_vehicle_attributes`, `add_wheel`, `configure_wheel_solver_contacts`.
- `newton/_src/vehicles/contact.py` — `WheelContactPatch`, `update_wheel_contact_patches`, `latch_wheel_loads`.
- `newton/_src/vehicles/tire.py` — tire interface (`@wp.func` dispatch), brush + linear models, `WheelTireParams`.
- `newton/_src/vehicles/wheel.py` — analytical spin integration + tire wrench injection.
- `newton/_src/vehicles/vehicle.py` — per-vehicle layout, drive modes, command mapping.
- `newton/_src/vehicles/controller.py` — `WheeledVehicles`, `WheeledConfig`, nested enums.
- `newton/vehicles.py` — public re-exports.
- Tests: `newton/tests/test_vehicles_metadata.py`, `test_vehicles_contact.py`, `test_vehicles_tire.py`, `test_vehicles_wheel.py`, `test_vehicles_drive.py`, `test_vehicles_controller.py`, `test_vehicles_integration.py`.

**Examples / docs (Phase 8):**
- `newton/examples/vehicles/example_vehicle_rc_car.py`, `example_vehicle_husky.py`, `example_vehicle_fleet.py`.
- Reuse assets `newton/examples/assets/wheeled/rc_car.usda`, `husky.usda` (verify; create minimal USDA if missing).
- `docs/api/newton_vehicles.rst`, `CHANGELOG.md`, `README.md` registration, run `docs/generate_api.py`.

## Data model (flat device arrays owned by `WheeledVehicles`)

Per wheel (length `W` = total wheels):
`shape_index:int32, body_index:int32, vehicle_index:int32, radius:f32, width:f32,`
`forward_axis:vec3, axle_axis:vec3 (body frame), driven:bool, steerable:bool,`
`side:int32 (-1 L / 0 C / +1 R), axle_row:int32 (0 front / 1 rear),`
`steer_dof:int32 (-1 if none), tire_model:int32, drive_input:int32 (0 speed / 1 torque),`
`inertia:f32, damping:f32, rolling_resistance:f32, kp:f32, tau_max:f32, max_speed:f32,`
`long_stiffness:f32, lat_stiffness:f32, mu_override:f32 (<0 → use seed),`
plus state `omega:f32, drive_target:f32, fz_latched:f32` and diagnostics
`slip_ratio:f32, slip_angle:f32, f_long:f32, f_lat:f32, normal_load:f32`.

Per vehicle (length `V`): `drive_mode:int32, wheelbase:f32, track_width:f32, steer_limit:f32, enabled:bool`.

Per-vehicle command buffers (length `V`): `cmd_drive:f32 [-1,1], cmd_steer:f32 [-1,1], cmd_brake:f32 [0,1]`.

Enums (nested on `WheeledVehicles`): `DriveMode.{GENERIC=0, ACKERMANN=1, SKID_STEER=2}`,
`TireModel.{BRUSH=0, LINEAR=1}` (PACEJKA/FIALA reserved), `DriveInput.{SPEED=0, TORQUE=1}`.

> Refinement vs the design doc: the doc's generic "channels" are concretized to a per-vehicle
> `(drive, steer, brake)` triple + per-wheel `side`/`axle_row` masks. This covers Ackermann
> (throttle+steer) and skid-steer (drive+differential) and AWD/FWD/RWD (via the `driven` mask).

---

## Phase 1 — Newton collision: scoped `preserve_contact_footprint`

Land first; independent and separately reviewable. Verified hook points:
`narrow_phase.py:439` (plane-cylinder routing), `collision_core.py:173`
(`post_process_axial_on_discrete_contact`), flag default `True` for both global switches.

### Task 1.1: Add the shape flag

**Files:** Modify `newton/_src/geometry/flags.py`; Test `newton/tests/test_collision_preserve_footprint.py`.

- [ ] **Step 1 — Read the current flags.** Open `newton/_src/geometry/flags.py`, find `class ShapeFlags` and the highest used bit. Pick the next free bit.
- [ ] **Step 2 — Write the failing test** (`test_collision_preserve_footprint.py`):

```python
import unittest
import newton
from newton._src.geometry.flags import ShapeFlags

class TestPreserveContactFootprintFlag(unittest.TestCase):
    def test_flag_exists_and_is_unique(self):
        self.assertTrue(hasattr(ShapeFlags, "PRESERVE_CONTACT_FOOTPRINT"))
        bit = int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT)
        self.assertNotEqual(bit, 0)
        # power of two (single bit) and not colliding with other flags
        self.assertEqual(bit & (bit - 1), 0)
```

- [ ] **Step 3 — Run, expect fail:** `uv run --extra dev -m newton.tests -k test_flag_exists_and_is_unique` → FAIL (AttributeError).
- [ ] **Step 4 — Add the flag** to `ShapeFlags` (next free bit), with a docstring: "Skip cylinder/cone contact-footprint collapse and axial-rolling projection for contacts involving this shape, preserving the full manifold (wheel-terrain patch)."
- [ ] **Step 5 — Run, expect pass.** Commit: `Add ShapeFlags.PRESERVE_CONTACT_FOOTPRINT`.

### Task 1.2: Expose `ShapeConfig.preserve_contact_footprint`

**Files:** Modify `newton/_src/sim/builder.py` (`ShapeConfig` ~line 231 and the place that assembles `shape_flags` per shape).

- [ ] **Step 1 — Failing test** (add to the same test file):

```python
def test_shape_config_sets_flag(self):
    builder = newton.ModelBuilder()
    b = builder.add_body()
    cfg = newton.ModelBuilder.ShapeConfig(preserve_contact_footprint=True)
    s = builder.add_shape_cylinder(b, radius=0.1, half_height=0.05, cfg=cfg)
    model = builder.finalize()
    flags = model.shape_flags.numpy()
    from newton._src.geometry.flags import ShapeFlags
    self.assertTrue(int(flags[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

def test_shape_config_default_off(self):
    builder = newton.ModelBuilder()
    b = builder.add_body()
    s = builder.add_shape_cylinder(b, radius=0.1, half_height=0.05)
    model = builder.finalize()
    from newton._src.geometry.flags import ShapeFlags
    self.assertFalse(int(model.shape_flags.numpy()[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))
```

(Confirm the exact `add_shape_cylinder` signature — adjust `half_height`/`half_width` arg names to the real API while implementing.)

- [ ] **Step 2 — Run, expect fail.**
- [ ] **Step 3 — Implement:** add `preserve_contact_footprint: bool = False` to `ShapeConfig` with a Google docstring; in the shape-finalization path that builds `shape_flags`, OR in the flag when the config bool is set. Follow the existing pattern used by other `ShapeConfig` bools that map to `ShapeFlags`.
- [ ] **Step 4 — Run, expect pass.** Commit: `Expose ShapeConfig.preserve_contact_footprint`.

### Task 1.3: Plumb the flag into plane-cylinder routing

**Files:** Modify `newton/_src/geometry/narrow_phase.py` (~439, `narrow_phase_primitive_kernel`).

- [ ] **Step 1 — Read** the kernel around line 439. Understand how `enable_plane_cylinder_contact_collapse` (a `wp.static(...)` compile-time bool) gates calling `collide_plane_cylinder` vs routing to GJK/MPR. Identify how `shape_flags` is (or can be) made available in this kernel (it is a per-shape array on the model passed to collision).
- [ ] **Step 2 — Failing integration test** (this is the real behavioral test; write it now, it will stay failing until 1.4 too):

```python
import numpy as np
import warp as wp
import newton

def _patch_extents(model, contacts, wheel_shape):
    # gather contact points on the wheel shape, project out the normal, measure spread
    cnt = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()[:cnt]
    s1 = contacts.rigid_contact_shape1.numpy()[:cnt]
    p0 = contacts.rigid_contact_point0.numpy()[:cnt]
    p1 = contacts.rigid_contact_point1.numpy()[:cnt]
    n  = contacts.rigid_contact_normal.numpy()[:cnt]
    pts = []
    for i in range(cnt):
        if s0[i] == wheel_shape: pts.append(p0[i])
        elif s1[i] == wheel_shape: pts.append(p1[i])
    if len(pts) < 2: return 0.0, len(pts)
    pts = np.array(pts); pts -= pts.mean(0)
    # spread magnitude (max pairwise distance proxy)
    return float(np.linalg.norm(pts, axis=1).max()), len(pts)

class TestPreserveFootprintContacts(unittest.TestCase):
    def _build(self, preserve):
        builder = newton.ModelBuilder()
        # ground plane (static)
        builder.add_ground_plane()
        # a wheel cylinder resting on the plane, axis along Y (rolling pose)
        b = builder.add_body(xform=wp.transform((0.0, 0.0, 0.099), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(preserve_contact_footprint=preserve)
        # orient cylinder so its axis is horizontal (a real wheel)
        s = builder.add_shape_cylinder(b, radius=0.1, half_height=0.05, cfg=cfg,
                                       xform=wp.transform((0,0,0), wp.quat_from_axis_angle(wp.vec3(1,0,0), np.pi/2)))
        model = builder.finalize()
        return model, s

    def test_unflagged_collapses_flagged_spreads(self):
        for device in newton.tests.get_test_devices() if hasattr(newton.tests,'get_test_devices') else ['cuda:0']:
            with wp.ScopedDevice(device):
                m_off, s_off = self._build(False)
                st = m_off.state(); c = m_off.contacts(); m_off.collide(st, c)
                spread_off, _ = _patch_extents(m_off, c, s_off)

                m_on, s_on = self._build(True)
                st2 = m_on.state(); c2 = m_on.contacts(); m_on.collide(st2, c2)
                spread_on, n_on = _patch_extents(m_on, c2, s_on)

                self.assertGreater(spread_on, spread_off + 1e-3)
                self.assertGreaterEqual(n_on, 2)
```

(Adjust device enumeration to the project's actual `newton.tests` helper while implementing; if none, use `["cpu","cuda:0"]`.)

- [ ] **Step 3 — Run, expect fail** (spreads equal: both collapse).
- [ ] **Step 4 — Implement the per-shape branch:** at ~439, when the pair is plane-cylinder and `enable_plane_cylinder_contact_collapse` is statically on, additionally read `preserve = (shape_flags[shape_cyl] & ShapeFlags.PRESERVE_CONTACT_FOOTPRINT) != 0` at runtime and, when set, route to the GJK/MPR manifold path instead of `collide_plane_cylinder`. Mirror exactly what the global-off path does, but gated per shape. Ensure `shape_flags` is a kernel input here (add to the kernel/launch signature if not already present, following how other per-shape arrays are passed).
- [ ] **Step 5 — Run.** The flagged spread should now exceed unflagged, but may still be reduced by axial projection (Task 1.4). If the test passes already (routing alone is enough), great; otherwise it still fails on `spread_on` and 1.4 finishes it.
- [ ] **Step 6 — Commit:** `Route plane-cylinder per-shape to full manifold when flagged`.

### Task 1.4: Skip axial projection for flagged shapes

**Files:** Modify `newton/_src/geometry/collision_core.py` (`post_process_axial_on_discrete_contact` ~173) and any kernel(s) in `collision_core.py`/`narrow_phase.py` that call it, to pass `shape_flags` + the two shape indices.

- [ ] **Step 1 — Read** `post_process_axial_on_discrete_contact` (173–275) and its call sites (the `post_process_contact` selection at `narrow_phase.py:1565`, and `create_solve_convex_*` in `collision_core.py`). Determine how the post-process receives shape ids.
- [ ] **Step 2 — Reuse the 1.3 test** (`test_unflagged_collapses_flagged_spreads`) as the failing target.
- [ ] **Step 3 — Implement:** make `post_process_axial_on_discrete_contact` accept `shape_flags` and the discrete/axial shape indices; early-return (no projection) when either shape has `PRESERVE_CONTACT_FOOTPRINT`. Thread `shape_flags` and shape ids through the calling kernels' signatures. Keep the non-flagged path identical.
- [ ] **Step 4 — Run, expect pass** for `test_unflagged_collapses_flagged_spreads`.
- [ ] **Step 5 — Regression:** add and run:

```python
def test_unflagged_matches_baseline(self):
    # two identical unflagged cylinders → contact count + normal identical to current behavior
    # (snapshot current contact count for a plane-cylinder; assert it is unchanged)
```

Run the broader collision test module to ensure nothing regressed:
`uv run --extra dev -m newton.tests -k collision` → all PASS.
- [ ] **Step 6 — Commit:** `Skip axial contact projection for footprint-preserving shapes`.

### Task 1.5: Phase-1 verification gate

- [ ] Run `uv run --extra dev -m newton.tests -k "collision or contact"` and confirm no regressions.
- [ ] Run `uvx pre-commit run -a`; fix issues; commit any formatting.

---

## Phase 2 — Vehicle metadata and flat tables

### Task 2.1: Package skeleton + public module

**Files:** Create `newton/_src/vehicles/__init__.py`, `newton/vehicles.py`; Test `newton/tests/test_vehicles_metadata.py`.

- [ ] **Step 1 — Failing test:**

```python
import unittest
class TestVehiclesImport(unittest.TestCase):
    def test_public_import(self):
        import newton.vehicles as nv
        self.assertTrue(hasattr(nv, "WheeledVehicles"))
        self.assertTrue(hasattr(nv, "register_vehicle_attributes"))
```

- [ ] **Step 2 — Run, expect fail** (ModuleNotFound / missing attr).
- [ ] **Step 3 — Create** the package with a placeholder `WheeledVehicles` class (`pass`) and `register_vehicle_attributes` stub in `metadata.py`; re-export from `_src/vehicles/__init__.py` and `newton/vehicles.py` (copyright header with year 2026, SPDX Apache-2.0 — match a sibling file's header exactly).
- [ ] **Step 4 — Run, expect pass.** Commit: `Scaffold newton.vehicles package`.

### Task 2.2: Register `vehicle:*` custom attributes

**Files:** `newton/_src/vehicles/metadata.py`.

Attributes (namespace `vehicle`), per the data model. Frequencies: SHAPE for wheel-shape attrs, BODY for wheel-body attrs, custom `vehicle:vehicle` and `vehicle:wheel` for per-vehicle/per-wheel rows; `references=` set for all index attrs (`"shape"`, `"body"`, `"vehicle:wheel"`, `"vehicle:vehicle"`).

- [ ] **Step 1 — Failing test:**

```python
import newton
from newton.vehicles import register_vehicle_attributes
def test_register_and_finalize(self):
    builder = newton.ModelBuilder()
    register_vehicle_attributes(builder)
    b = builder.add_body()
    builder.add_shape_cylinder(b, radius=0.1, half_height=0.05)
    model = builder.finalize()
    ns = getattr(model, "vehicle")
    self.assertTrue(hasattr(ns, "is_wheel"))
    self.assertEqual(len(ns.is_wheel.numpy()), model.shape_count)
```

- [ ] **Step 2 — Run, expect fail.**
- [ ] **Step 3 — Implement** `register_vehicle_attributes(builder)` using `builder.add_custom_frequency(CustomFrequency(name="vehicle", namespace="vehicle"))`, `...(name="wheel", namespace="vehicle")`, then `builder.add_custom_attribute(CustomAttribute(...))` for each attribute in the data model. Import `CustomAttribute`, `CustomFrequency`, `Model` from the internal sim package. Use `Model.AttributeFrequency.SHAPE/BODY` and the string `"vehicle:wheel"`/`"vehicle:vehicle"` for custom ones. Set sensible defaults (`is_wheel=False`, ids=`-1`, axes default `+X`/`+Y`).
- [ ] **Step 4 — Run, expect pass.** Commit: `Register vehicle:* custom attributes`.

### Task 2.3: `add_wheel` build helper

**Files:** `newton/_src/vehicles/metadata.py`.

`add_wheel(builder, *, shape, body, vehicle_id, wheel_id, radius, width, driven=True, steerable=False, side=0, axle_row=0, steer_dof=-1, forward_axis=(1,0,0), axle_axis=(0,1,0))` — sets the `vehicle:*` attribute values for that shape/body/wheel-row, and sets `preserve_contact_footprint=True` on the shape (via `model`/builder shape flags) and records it for `configure_wheel_solver_contacts`.

- [ ] **Step 1 — Failing test:** build one wheel via `add_wheel`, finalize, assert `model.vehicle.is_wheel[shape]==True`, `wheel_radius[shape]≈0.1`, `wheel_id[shape]==0`, and the shape has `PRESERVE_CONTACT_FOOTPRINT` set.
- [ ] **Step 2 — Run, expect fail.**
- [ ] **Step 3 — Implement.** Use `builder.set_custom_attribute`/the documented setter (confirm the exact setter name in `builder.py` while implementing; the verification showed values are stored per builder). Set the shape flag through `ShapeConfig` at shape-creation time if possible, else via a builder shape-flags setter.
- [ ] **Step 4 — Run, expect pass.** Commit: `Add add_wheel build helper`.

### Task 2.4: Read finalized model → flat device tables

**Files:** `newton/_src/vehicles/metadata.py` — `read_vehicle_model_data(model) -> VehicleModelData` (a small dataclass of host counts + the device `wp.array`s, sliced/copied from `model.vehicle.*`, plus derived per-vehicle wheel counts).

- [ ] **Step 1 — Failing test:** build a 4-wheel Ackermann vehicle; `read_vehicle_model_data(model)`; assert `wheel_count==4`, `vehicle_count==1`, `wheel_body_index` matches, arrays are `wp.array` on the model device.
- [ ] **Step 2 — Run, expect fail. Step 3 — Implement** (gather the `vehicle:*` arrays; build any derived arrays with a small kernel or host prep at construction — host prep is fine here since it is one-time setup, not the runtime path). **Step 4 — pass. Commit:** `Read finalized vehicle metadata into flat tables`.

### Task 2.5: Heterogeneity + replication test

- [ ] **Step 1 — Test:** build an Ackermann sub-builder and a skid-steer sub-builder, merge both via `add_world`/`add_builder` into one model; assert `vehicle_count==2`, wheel ids `0..N-1` unique and contiguous, each wheel's `vehicle_index` correct, `drive_mode` distinct per vehicle. Replicate the Ackermann builder ×4 and assert ids stay correct.
- [ ] **Step 2 — Run.** If `references=` remapping is right, it passes; otherwise fix the `references` fields in 2.2. **Commit:** `Test heterogeneous + replicated vehicle metadata`.

---

## Phase 3 — Contact patch extraction and load latching

### Task 3.1: `WheelContactPatch` allocation

**Files:** `newton/_src/vehicles/contact.py`; Test `newton/tests/test_vehicles_contact.py`.

`WheelContactPatch` holds device arrays length `W`: `active:bool, contact_count:int32, terrain_shape:int32, center:vec3, normal:vec3, tangent_extent:vec2, area:f32, normal_load:f32, friction_seed:f32`. Constructor takes `wheel_count` + device.

- [ ] TDD: allocate, assert shapes/dtypes/zeroed. Commit: `Add WheelContactPatch state`.

### Task 3.2: `update_wheel_contact_patches`

**Files:** `contact.py`. Batched kernels (no Python loops): (a) clear; (b) iterate contacts, for each contact whose shape0/shape1 is a wheel, atomic-accumulate point, normal (sign-corrected: `-normal` if wheel is shape0 else `+normal`), count, and record terrain shape + `friction_seed = model.shape_material_mu[terrain]`; (c) finalize center/normal; (d) project points to tangent plane for extents (second pass over contacts or a reduction).

Signature: `update_wheel_contact_patches(model, contacts, model_data, patch)`.

- [ ] **Step 1 — Failing test:** build a single wheel cylinder (flagged) on a ground plane, `use_mujoco_contacts=False`; create `SolverMuJoCo(model, use_mujoco_contacts=False)`; one substep (`clear_forces`, `collide`, `solver.step`); call `update_wheel_contact_patches`; assert `patch.active[0]==True`, `normal[0]≈(0,0,1)` within 1e-2, `center[0].z≈0` (ground), center x,y ≈ wheel x,y.
- [ ] **Step 2 — fail. Step 3 — implement kernels. Step 4 — pass. Commit:** `Add wheel contact patch extraction`.

### Task 3.3: `latch_wheel_loads`

**Files:** `contact.py`. Requires `model.request_contact_attributes("force")` at build (document in controller). Kernel: for each wheel contact, `f = wp.spatial_top(contacts.force[i])`, sign-correct, `proj = dot(f, support_normal)`, `if proj>0: atomic_add(fz, wheel, proj)`.

- [ ] **Step 1 — Test:** wheel of known mass `m` resting on plane, settle ~20 substeps, `solver.update_contacts(contacts, state)`, `latch_wheel_loads`, assert `fz[0] ≈ m*9.81` within 10%.
- [ ] **Step 2 — fail. Step 3 — implement. Step 4 — pass. Commit:** `Add wheel normal-load latching from solver contacts`.

---

## Phase 4 — Tire model

### Task 4.1: Tire params + brush `@wp.func`

**Files:** `newton/_src/vehicles/tire.py`; Test `newton/tests/test_vehicles_tire.py`.

`@wp.func tire_force(model_id:int, kappa:float, alpha:float, fz:float, mu:float, c_long:float, c_lat:float) -> wp.vec2` returning `(F_long, F_lat)` in the patch frame. Dispatch on `model_id` (BRUSH/LINEAR).

Brush (combined slip): build theoretical slip `sx = -kappa/(1+kappa)` (guard `1+kappa>eps`), `sy = -tan(alpha)/(1+kappa)`; `s = sqrt(sx*sx+sy*sy)`; with combined slip stiffness `C` (use `c_long` for sx, `c_lat` for sy via scaled components — for v1 use isotropic `C=c_long`, and scale lateral by `c_lat/c_long`), and `theta = 1.0` lumped: `f = mu*fz*g(C*s/(mu*fz))` where `g(z)= z - z*z/3 + z*z*z/27` for `z<3` else `1`. Direction `(sx,sy)/s`. Cap magnitude at `mu*fz`. Linear model: `F_long=clamp(c_long*kappa,±mu*fz_proj)`, `F_lat=clamp(-c_lat*alpha,...)`, then friction-circle scale.

Test via a wrapper kernel that calls the `@wp.func` over an array of synthetic inputs.

- [ ] **Step 1 — Tests:**
  - `test_zero_slip_zero_force`: kappa=0, alpha=0 → (0,0).
  - `test_longitudinal_saturates`: large kappa → |F_long| ≈ mu*fz (within 2%), F_lat≈0.
  - `test_lateral_saturates`: large alpha → |F_lat| ≈ mu*fz, F_long≈0.
  - `test_combined_on_friction_ellipse`: kappa,alpha both large → `sqrt(F_long²+F_lat²) ≤ mu*fz*(1+1e-3)`.
  - `test_linear_slope`: LINEAR small kappa → F_long ≈ c_long*kappa.
- [ ] **Step 2 — fail. Step 3 — implement. Step 4 — pass. Commit:** `Add brush + linear tire-force functions`.

---

## Phase 5 — Analytical wheel spin + tire wrench injection

### Task 5.1: spin integration + slip computation + injection

**Files:** `newton/_src/vehicles/wheel.py`; Test `newton/tests/test_vehicles_wheel.py`.

`apply_wheel_dynamics(model, state, model_data, patch, params_state, dt)` — one kernel per wheel:
1. If `not patch.active[w]`: zero diagnostics, optionally still integrate free spin (drive/brake/damping) and continue.
2. Compute world axes: `R = transform of wheel body`; `fwd = normalize(project(R*forward_axis, onto plane ⟂ patch.normal))`; `lat = cross(patch.normal, fwd)`.
3. Contact-point velocity: `v = body_qd linear + cross(body_qd angular, patch.center - body_com_world)` (build from `state.body_qd[body]`, careful with `spatial_top`=linear, `spatial_bottom`=angular). `v_long=dot(v,fwd)`, `v_lat=dot(v,lat)`.
4. `omega` (analytical) from `params_state.omega[w]`. `kappa = (omega*r - v_long)/max(|v_long|, v_ref)`; `alpha = atan2(v_lat, max(|v_long|, v_ref))`.
5. `fz = max(patch.normal_load[w] or fz_latched, fallback)`; `mu = mu_override>=0 ? mu_override : patch.friction_seed`.
6. `(F_long, F_lat) = tire_force(tire_model, kappa, alpha, fz, mu, c_long, c_lat)`.
7. Wrench: `F = F_long*fwd + F_lat*lat`; `tau = cross(patch.center - body_com_world, F)`; `atomic_add(state.body_f[body], spatial_vector(F, tau))`.
8. Reaction torque (if enabled): add `-(drive_reaction about axle)` to chassis — for v1 apply `+`/`-` axle reaction on the wheel body about `lat`/axle axis (document approximation).
9. Spin update (semi-implicit in F_long(omega)): `tau_net = tau_drive - tau_brake*sign(omega) - F_long*r - damping*omega - rolling_resistance*sign(omega)`; treat `dF_long/domega ≈ (c_long*r)/v_ref` as implicit term: `omega += dt*tau_net/(I + dt*(c_long*r*r)/v_ref)`; zero-crossing clamp for brake.
10. Write diagnostics.

`tau_drive` derives from `drive_target` and `drive_input`: SPEED → `clamp(kp*(omega_target-omega), ±tau_max)`; TORQUE → `drive_target` (clamped to ±tau_max).

- [ ] **Step 1 — Tests** (synthetic: bypass solver, set arrays directly):
  - `test_free_spin_up`: inactive patch, drive torque `T`, `I`, dt → after N steps `omega ≈ T/I * N*dt` (no tire load).
  - `test_brake_to_zero_no_reverse`: spinning wheel, brake torque, no drive → omega → 0 and stays ≥ 0 (no sign flip).
  - `test_tire_reaction_decelerates`: active patch with positive slip and load → F_long>0 and omega decreases relative to no-load case.
  - `test_force_injection_direction`: active patch, prescribe omega>v_long/r (driving) → `state.body_f[body]` linear component along +fwd.
- [ ] **Step 2 — fail. Step 3 — implement. Step 4 — pass. Commit:** `Add analytical wheel spin and tire wrench injection`.

---

## Phase 6 — Vehicle layout, drive modes, command mapping

### Task 6.1: command mapping kernel

**Files:** `newton/_src/vehicles/vehicle.py`; Test `newton/tests/test_vehicles_drive.py`.

`update_vehicle_controls(model, control, model_data, cmd)` — one kernel per wheel:
- Read `vehicle = vehicle_index[w]`, `mode = drive_mode[vehicle]`, `d=cmd_drive[vehicle]`, `s=cmd_steer[vehicle]`, `brk=cmd_brake[vehicle]`.
- Drive target (if `driven[w]`):
  - GENERIC/ACKERMANN: `base = d`.
  - SKID_STEER: `base = d + side[w]*s` (side −1 left/+1 right) → clamp [-1,1].
  - SPEED mode: `omega_target[w] = base*max_speed[w]/r`; TORQUE mode: `drive_target[w]=base*tau_max[w]`.
  - Brake adds `brake_target` consumed by spin (set `tau_brake = brk*tau_brake_max`).
- Steering (if `steerable[w]` and `mode==ACKERMANN` and `steer_dof[w]>=0`):
  - center angle `delta = s*steer_limit[vehicle]`.
  - per-wheel Ackermann: with wheelbase `L`, half-track `t=track/2`, signed by `side`:
    `if |delta|<eps: target=delta` else `R = L/tan(delta); target = atan2(L, R - side*t)` with sign handling so inner wheel turns more. Write `control.joint_target_pos[steer_dof[w]] = target`.
- [ ] **Step 1 — Tests:**
  - `test_ackermann_inner_outer`: V=1 Ackermann, set steer=+1, drive=0; read the two steer DOFs; assert `cot(outer)-cot(inner) ≈ track/wheelbase` within 5%, and inner angle magnitude > outer.
  - `test_ackermann_zero_steer`: steer=0 → both steer targets ≈ 0.
  - `test_skid_steer_differential`: skid-steer, drive=0, steer=+1 → left wheel target speed negative, right positive (spin in place); drive=1,steer=0 → both equal positive.
  - `test_speed_vs_torque_mode`: SPEED writes `omega_target`; TORQUE writes `drive_target`.
- [ ] **Step 2 — fail. Step 3 — implement. Step 4 — pass. Commit:** `Add vehicle command mapping and drive modes`.

---

## Phase 7 — `WheeledVehicles` controller

### Task 7.1: controller assembly

**Files:** `newton/_src/vehicles/controller.py`; Test `newton/tests/test_vehicles_controller.py`.

`WheeledVehicles(model, *, config=WheeledConfig())`: calls `read_vehicle_model_data`, allocates `WheelContactPatch` + per-wheel param/state/diagnostic arrays (initialized from `config` defaults; per-wheel overridable), command buffers. Nested enums. Methods:
- `set_commands(*, drive=None, steer=None, brake=None)` — accept scalar (broadcast to all vehicles) or array-like length `V`; copy to device.
- `update_controls(control)` — launch `update_vehicle_controls`.
- `apply(state, contacts, dt)` — `update_wheel_contact_patches` → `apply_wheel_dynamics`.
- `latch_loads(contacts)` — `latch_wheel_loads`.
- Read-only properties returning the diagnostic arrays.
- `WheeledConfig`: default tire model + stiffnesses + mu, inertia, damping, rolling_resistance, kp, tau_max, max_speed, reaction torque toggle, fallback Fz, `min_reference_speed`.

`WheeledVehicles.register_attributes(builder)` (static) → `register_vehicle_attributes`.

- [ ] **Step 1 — Test (object wiring):** build a 4-wheel Ackermann model, construct `WheeledVehicles`, `set_commands(drive=1.0)`, run a single `update_controls/collide/apply/step/update_contacts/latch_loads` cycle without error; assert diagnostics arrays are populated and finite.
- [ ] **Step 2 — fail. Step 3 — implement. Step 4 — pass. Commit:** `Add WheeledVehicles controller object`.

### Task 7.2: friction-ownership build helper

**Files:** `metadata.py` — `configure_wheel_solver_contacts(model, model_data, *, condim=1, priority=...)`.

- [ ] **Step 1 — Read** `newton/_src/solvers/mujoco/solver_mujoco.py` for how per-geom `condim`/priority are set (and whether they apply in the `use_mujoco_contacts=False` path; if not, set converted-contact tangential friction to ~0 on wheel shapes via `model.shape_material_mu`). **Step 2 — Test:** after the helper, wheel shapes have `condim==1` (or `shape_material_mu≈0` in the fallback). **Step 3 — implement. Step 4 — pass. Commit:** `Add wheel solver-contact friction configuration`.

---

## Phase 8 — Examples, docs, changelog

### Task 8.1: RC-car example (Ackermann)

**Files:** `newton/examples/vehicles/example_vehicle_rc_car.py`. Reuse `assets/wheeled/rc_car.usda` (verify labels; if the asset lacks `vehicle:*` attrs, annotate at load via `add_wheel`). Follow the `Example` class format (`__init__(viewer,args)`, `step`, `render`, `test_final`, optional `test_post_step`). Loop order: `clear_forces` → `update_controls(control)` → `collide` → `apply(state,contacts,dt)` → `solver.step` → `update_contacts` → `latch_loads` → swap.

- [ ] **test_final:** after driving `drive=1` for ~2 s sim, chassis world-x advanced > 0.3 m and is finite; with `steer` set, yaw changed sign-appropriately. Run: `uv run --extra examples -m newton.examples vehicle_rc_car --num-frames 200` (headless). Commit: `Add RC car Ackermann example`.

### Task 8.2: Husky example (skid-steer)

- [ ] As above; `test_final`: `drive=1,steer=0` → moves forward; `drive=0,steer=1` → yaw rotates with near-zero net translation. Commit: `Add Husky skid-steer example`.

### Task 8.3: Heterogeneous fleet example + integration test

**Files:** `example_vehicle_fleet.py`; `newton/tests/test_vehicles_integration.py`.

- [ ] Build one model with an RC car + a Husky (+ replicate each ×K). Single controller. `test_final`/integration test: both vehicle types respond correctly in one batched model; no per-vehicle host branching (assert by construction — single kernel launches). Commit: `Add heterogeneous fleet example and integration test`.

### Task 8.4: Docs + changelog + API gen

- [ ] Create `docs/api/newton_vehicles.rst` (mirror `newton_wheeled.rst`). Add `CHANGELOG.md` `[Unreleased] / Added` entry: "Add `newton.vehicles` wheeled-vehicle layer with brush tire model and heterogeneous drive modes." Register the three examples in `README.md` with the run command. Run `uv run python docs/generate_api.py`. Commit: `Document newton.vehicles public API`.

---

## Phase 9 — Final verification gate

- [ ] `uv run --extra dev -m newton.tests -k "vehicles or collision or preserve_footprint"` → all PASS (capture output).
- [ ] `uv run --extra examples -m newton.examples vehicle_rc_car --num-frames 200`, `vehicle_husky`, `vehicle_fleet` → run clean, `test_final` passes.
- [ ] `uvx pre-commit run -a` → clean.
- [ ] Write `docs/superpowers/reports/2026-06-12-wheeled-vehicle-redesign-report.md`: what was built, every test + its observed result, known gaps, and the swap checklist (delete `newton/_src/wheeled/`, decide final public name, deprecation N/A on unreleased branch).
- [ ] Final commit + push the branch.

---

## Self-review notes (author check against the spec)

- Spec §6 collision fix → Phase 1 (1.1–1.5). ✓
- Spec §5 heterogeneity / custom attrs / replication → Phase 2 (2.2–2.5). ✓
- Spec §7 contact patch + §8 load latching → Phase 3. ✓
- Spec §9.1–9.2 tire interface + brush → Phase 4. ✓
- Spec §9.3–9.4 analytical spin + injection (body_f `[lin,ang]`) → Phase 5. ✓
- Spec §10 drive modes + Ackermann/skid-steer + command semantics → Phase 6. ✓
- Spec §11 suspension/steering as solver joints → honored (Phase 6 writes only steer targets; Phase 8 assets carry suspension joints). ✓
- Spec §12 public API surface → Phase 7. ✓
- Spec §8 friction ownership / `use_mujoco_contacts=False` / `request_contact_attributes("force")` → Phase 3.3 + 7.2 + example loop. ✓
- Spec §15 testing → tests in every phase + Phase 9. ✓
- Spec §16 rollout (build alongside) → new `newton/_src/vehicles/`; swap checklist in Phase 9 report. ✓
- Open items deferred per spec §17 (powertrain, drag, Pacejka/Fiala, hydroelastic) — not in plan. ✓
