# Wheeled Vehicle Layer — Clean Redesign Design

Status: proposed
Date: 2026-06-12
Supersedes (on swap): `newton/_src/wheeled/` and
`docs/superpowers/roadmaps/2026-05-28-wheeled-vehicle-solver-roadmap.md`

## 1. Context and motivation

Newton needs a wheeled-vehicle simulation layer for RC cars and small AGVs
(Clearpath Husky/Jackal) that runs thousands of vehicles in parallel with no
Python/NumPy/Torch in the runtime path and supports heterogeneous vehicles in a
single model.

A prior agent-built attempt exists on this branch (`newton/_src/wheeled/`,
~4,860 LOC across 8 modules, ~17 public classes, 30+ free functions, 14
documented "phases"). It works, but it became iterative and drifted from a
faithful vehicle model. The root cause was a mid-flight architectural pivot:
the original intent was analytical/raycast single-point wheel contact with the
wheeled layer owning friction, but the implementation pivoted to extracting
contact patches from Newton's general collision pipeline. That decision created
the bulk of the churn:

- Rigid cylinder-on-plane contacts **collapse to a line / zero area** (the
  analytical plane-cylinder helper and a global axial-rolling projection both
  flatten the footprint).
- Rigid contacts are **insensitive to sink depth** (no compliance information).
- Hydroelastic — the only area-capable path — **cannot handle planes or
  heightfields, is CUDA-only, and yields 600–2600 contacts/wheel**.
- **Friction ownership was never settled** — left an "explicit but
  unconfigured" hybrid with a standing double-counting risk.
- Three overlapping force paths (`drive`/`tire`/`moment`) duplicated slip,
  load, and clip logic.

This redesign keeps what was genuinely good (delegating suspension + steering to
real solver joints; the `body_f` injection pattern; flat-array discipline;
custom-attribute metadata; a pluggable tire interface) and fixes the contact
problem at its source rather than working around it.

## 2. Decisions (locked with the requester)

1. **The wrapped solver owns collision and normal support.** Wheels remain
   collision shapes; the solver provides normal force. This yields stabler
   physics and a single source of truth for contact.
2. **The line-collapse is fixed in Newton's collision code**, scoped to an
   opt-in per-shape mode so default cylinder/cone contacts are unchanged.
3. **The wheeled layer owns the tire model** (longitudinal + lateral, combined
   slip), consuming the now-correct contact patch. Solver *tangential* friction
   on wheel-ground pairs is disabled so nothing is double-counted.
4. **Wheel spin is analytical**, not a physical solver DOF. The wheeled layer
   integrates `I·dω/dt = τ_drive − τ_brake − F_long·r − resistance` and feeds ω
   back into the tire slip.
5. **Suspension and steering stay as real solver joints** (asset-authored
   prismatic/revolute joints; the layer reads the resulting wheel pose and
   writes steering targets).
6. **Public surface is a cohesive controller object** (`WheeledVehicles`) that
   preserves full heterogeneity (mixed vehicle topologies in one model).
7. **Default tire model is a brush / combined-slip model** behind a pluggable,
   per-wheel-selectable interface.
8. **Built fresh alongside** the existing implementation; swapped in and the old
   code deleted once at parity.

## 3. Goals, non-goals, constraints

Goals:

- Faithful longitudinal + lateral tire behavior with combined-slip saturation.
- Heterogeneous vehicles (Ackermann, skid-steer, generic N-wheel), mixed and
  replicated in one model, with no host-side branching in the runtime path.
- One clean object as the public surface.
- A scoped, reusable improvement to Newton's rolling-shape contact generation.

Non-goals (v1 — interface room left, not built):

- Powertrain: motor curves, gearbox, differentials, AWD/FWD/RWD policy, battery.
- Quadratic aerodynamic drag (vehicle-level force; trivial to add later).
- Additional tire formulations (Pacejka Magic Formula, Fiala) beyond stubs.
- Hydroelastic/SDF patch source.
- A raycast wheel-contact path (we are solver-collision-based).
- Heightfield-specific tuning.

Constraints (from `AGENTS.md` and the roadmap):

- Runtime work in Warp kernels over flat arrays; no Python loops over
  wheels/vehicles/contacts in the step path.
- Examples/docs import only public modules, never `newton._src`.
- No new required or optional dependencies.
- Public API follows prefix-first naming, PEP 604 unions, Google docstrings,
  SI units, `wp.array[...]` annotations.

## 4. Architecture overview

The layer is a **co-simulation** split:

```
                   ┌─────────────────────── per substep ───────────────────────┐
 set_commands(...) │ clear_forces                                               │
 (per frame)       │ vehicles.update_controls(control)   # cmd→steer targets    │
                   │                                     #     +per-wheel drive  │
                   │ model.collide(state, contacts)      # SOLVER owns geometry  │
                   │ vehicles.apply(state, contacts, dt) # patch→tire→spin→body_f│
                   │ solver.step(state, next, control, contacts, dt)            │
                   │ solver.update_contacts(contacts, next)  # populate forces   │
                   │ vehicles.latch_loads(contacts)      # Fz for next step      │
                   │ swap(state, next)                                          │
                   └────────────────────────────────────────────────────────────┘
```

The wrapped rigid solver (MuJoCo Warp first) owns chassis integration,
suspension + steering joints, normal contact support, and the contact *solve*.
`WheeledVehicles` owns the wheel-rotational subsystem (analytical spin) and the
tangential tire forces it injects into `state.body_f`.

**Contact-source requirement.** The MuJoCo solver runs with
`use_mujoco_contacts=False`, so contact *detection* uses Newton's collision
pipeline — where the `preserve_contact_footprint` fix (Section 6) lives — and
the Newton contacts are converted into the MuJoCo solve. MuJoCo still owns
normal support and integration. With the default `use_mujoco_contacts=True`,
MuJoCo's own narrow phase runs and the footprint fix would never execute; the
design therefore mandates the Newton-contacts path. This matches the prior
Phase 1B finding.

### 4.1 The `WheeledVehicles` object

Built from a finalized `Model`; reads `model.wheeled.*` attributes into flat
device arrays; allocates per-wheel/per-vehicle state and diagnostics. All arrays
are sized to totals across every vehicle and wheel in the model.

```python
vehicles = newton.wheeled.WheeledVehicles(model, config=WheeledConfig(...))

# per frame (or when commands change): normalized [-1, 1] per channel
vehicles.set_commands(drive=..., steer=..., brake=...)

# per substep:
vehicles.update_controls(control)        # writes Control.joint_target_pos
                                          #   for steerable joints; resolves
                                          #   per-wheel drive targets internally
model.collide(state, contacts)
vehicles.apply(state, contacts, dt)       # patches → tire forces → spin → body_f
solver.step(state, next_state, control, contacts, dt)
solver.update_contacts(contacts, next_state)
vehicles.latch_loads(contacts)            # latch solver normal force as next Fz
```

Method responsibilities (each a thin host wrapper around one or more batched
Warp kernels; none iterate in Python):

| Method | Reads | Writes | Kernel work |
|---|---|---|---|
| `set_commands` | host/array inputs | command buffers | copy into device arrays |
| `update_controls(control)` | command buffers, layout | `control.joint_target_pos`, internal per-wheel drive targets | per-vehicle drive-mode branch (Ackermann/skid/generic) |
| `apply(state, contacts, dt)` | `contacts`, `state`, latched Fz | per-wheel patch/tire/spin diagnostics, `state.body_f` | patch extraction → tire force → spin integration → wrench scatter-add |
| `latch_loads(contacts)` | `contacts.force` | per-wheel `Fz` for next step | project contact force on support normal, atomic-add per wheel |

The four-phase split mirrors the natural substep boundaries (before collide /
after collide / after solve) and keeps the loop explicit. A thin
`SolverWheeled` wrapper that drives the object inside `step()` may be added
later, but the object is the primary surface.

## 5. Heterogeneity model

Heterogeneity is the central requirement and is **data-driven**, never
host-branched:

- Every per-wheel/per-vehicle quantity is a flat device array indexed by global
  wheel-id / vehicle-id. Replication/merge correctness comes from Newton's
  custom-attribute `references=` remapping (verified: `add_builder`/`add_world`
  offset both the attribute's frequency dimension and any index values stored in
  it, preserving `-1` sentinels).
- Each vehicle carries a `drive_mode` enum in a device array. The command-mapping
  kernel reads it per-vehicle and branches (`GENERIC` / `ACKERMANN` /
  `SKID_STEER`). Adding a topology = a new enum case + kernel branch.
- Each wheel carries a `tire_model` enum, so tire formulations can also differ
  per wheel within one batched kernel.
- Multiple heterogeneous sub-builders merge into one model; the verification
  confirmed `model.wheeled.vehicle_id` / `wheel_id` stay correct per instance.

### 5.1 Metadata schema (`wheeled:*` custom attributes)

Registered via `WheeledVehicles.register_attributes(builder)` before
finalization. Two custom frequencies (`wheeled:vehicle`, `wheeled:wheel`) give
stable flat ids across replication.

| Attribute | Frequency | dtype | Meaning |
|---|---|---|---|
| `wheeled:is_wheel` | SHAPE | bool | shape is a wheel |
| `wheeled:wheel_id` | SHAPE | int32 | flat wheel id (`references="wheeled:wheel"`) |
| `wheeled:vehicle_id` | SHAPE | int32 | flat vehicle id (`references="wheeled:vehicle"`) |
| `wheeled:wheel_radius` | SHAPE | float32 | wheel radius [m] |
| `wheeled:wheel_width` | SHAPE | float32 | wheel width [m] |
| `wheeled:is_wheel_body` | BODY | bool | body carries a wheel shape |
| `wheeled:wheel_body_id` | BODY | int32 | flat wheel id (`references="wheeled:wheel"`) |
| `wheeled:drive_mode` | `wheeled:vehicle` | int32 | per-vehicle `DriveMode` |
| `wheeled:wheelbase` | `wheeled:vehicle` | float32 | [m], Ackermann |
| `wheeled:track_width` | `wheeled:vehicle` | float32 | [m] |
| `wheeled:steer_limit` | `wheeled:vehicle` | float32 | [rad] |
| `wheeled:wheel_driven` | `wheeled:wheel` | bool | receives drive command |
| `wheeled:wheel_steerable` | `wheeled:wheel` | bool | receives steering command |
| `wheeled:drive_channel` | `wheeled:wheel` | int32 | drive channel index |
| `wheeled:steer_channel` | `wheeled:wheel` | int32 | steering channel index |
| `wheeled:steer_joint` | `wheeled:wheel` | int32 | steering joint index, or `-1` |
| `wheeled:forward_axis` | `wheeled:wheel` | vec3 | wheel forward axis in body frame |
| `wheeled:axle_axis` | `wheeled:wheel` | vec3 | wheel spin axis in body frame |

Two ingestion paths (both produce identical device tables):

1. **Direct builder annotation** — a focused helper marks a shape as a wheel with
   its dimensions and role after `add_shape_*`/`add_joint_*`. The helper sets
   `wheeled:*` attributes *and* the `preserve_contact_footprint` shape flag and
   the normal-only friction config (Section 8) on the wheel shape.
2. **Pre-authored USDA** — assets carrying `wheeled:*` attributes import
   directly.

Steering joint indices are recorded (to write targets); suspension joints are
not recorded (nothing to do — pure solver dynamics).

## 6. Newton core change — scoped `preserve_contact_footprint` mode

This is the only change outside the wheeled package. It is a per-shape opt-in;
default behavior for every existing shape is byte-for-byte unchanged. It takes
effect only on the Newton contact-detection path (`use_mujoco_contacts=False`,
Section 4); under MuJoCo-native collision it is inert.

New flag: `ShapeFlags.PRESERVE_CONTACT_FOOTPRINT` (`newton/_src/geometry/flags.py`)
surfaced as `ShapeConfig.preserve_contact_footprint: bool = False`
(`newton/_src/sim/builder.py`), stored in the existing
`model.shape_flags` array (no new array).

For a contact pair where either shape has the flag set:

1. **Bypass the analytical plane-cylinder collapse** — in
   `narrow_phase_primitive_kernel` (`narrow_phase.py:439`), route the
   plane-cylinder pair to the GJK/MPR manifold path instead of
   `collide_plane_cylinder` (`collision_primitive.py:516`). This is the same
   routing the global `enable_plane_cylinder_contact_collapse=False` already
   produces, now scoped per-shape.
2. **Skip axial rolling projection** — in `post_process_axial_on_discrete_contact`
   (`collision_core.py:173`), pass `shape_flags` through and skip the projection
   when the flag is set (the global `enable_axial_contact_projection=False` path
   already exists via `post_process_minkowski_only`; this scopes it per-shape).

Deterministic contact reduction (`contact_reduction.py`) needs no change — it
already preserves spatial-extreme (footprint-spanning) points.

Verified outcome from prior diagnostics: with both mechanisms off for a
wheel-cylinder, a flat-plane footprint of ~6.3e-3 m² appears that is otherwise
collapsed to ~0; ramp/jump cases gain ~50× footprint area.

Regression coverage: a primitive plane-cylinder pair, a triangle-mesh terrain
pair, and a box/ramp pair, each asserting (a) flagged shape → multi-point
footprint with non-degenerate tangent extents, and (b) unflagged shape →
identical contacts to current `main`.

## 7. Contact patch extraction (`contact.py`)

After `collide()`, one batched kernel groups contacts by wheel shape id and
produces a per-wheel `WheelContactPatch` (flat device arrays):

| Field | dtype | Units | Meaning |
|---|---|---|---|
| `active` | bool | — | wheel has ≥1 active contact |
| `contact_count` | int32 | — | contacts grouped to this wheel |
| `terrain_shape` | int32 | — | counterpart shape, or `-1` |
| `center` | vec3 | m | penetration-weighted patch center (world) |
| `normal` | vec3 | — | unit support normal on the wheel |
| `tangent_extent` | vec2 | m | footprint extents along tangent axes |
| `area` | float32 | m² | estimated patch area |
| `normal_load` | float32 | N | latched from `contacts.force` (Section 8) |
| `friction_seed` | float32 | — | terrain `shape_material_mu` |

Support-normal orientation flips by whether the wheel is shape0 or shape1
(`−normal` vs `+normal`). The patch is a per-step derived view; it never mutates
`contacts`.

For the brush model (Section 9) the essential inputs are `center`, `normal`, and
`normal_load`; `area`/`tangent_extent`/contact length `2a` are available for
contact-length-aware tire variants and diagnostics.

## 8. Friction ownership and normal-load read-back

Explicit, no hidden defaults:

- A build-time helper sets wheel shapes to MuJoCo `condim=1` (normal-only) with
  high geom priority, so the solver provides normal support while the tire model
  owns all tangential force. The exact mechanism in the `use_mujoco_contacts=False`
  path (per-geom `condim`/priority vs. zeroing converted-contact tangential
  friction) must be verified against `solver_mujoco` during implementation;
  whichever holds, the contract is normal-only on wheel-ground pairs, set
  explicitly. For solvers that cannot express normal-only, a tiny configurable
  tangential-μ fallback is documented (not silently applied).
- The model is built with `request_contact_attributes("force")`. After
  `solver.step()`, `solver.update_contacts(contacts, state)` populates
  `contacts.force` (works whether or not `use_mujoco_contacts`).
- `latch_loads(contacts)` projects each wheel contact's linear force
  (`wp.spatial_top(contacts.force[i])`, sign-corrected for shape0/shape1) onto
  the support normal and atomic-adds per wheel → `Fz` used next step.
- Fallback when force reporting is unavailable: a configured per-wheel static
  load (e.g. vehicle mass share). There is no penetration field exposed on
  `Contacts`, so a penetration·stiffness fallback is explicitly out of scope.

## 9. Tire model (`tire.py`) and analytical spin (`wheel.py`)

### 9.1 Tire-model interface

A Warp-callable contract selected per wheel by a `tire_model` enum:

```
tire_force(kappa, alpha, Fz, mu, params) -> (F_long, F_lat)   # patch frame
```

Inputs: longitudinal slip `kappa`, slip angle `alpha`, normal load `Fz`,
friction `mu`, and a per-wheel parameter block. Output: longitudinal and lateral
force in the contact tangent frame. v1 implements `BRUSH` (default) and `LINEAR`;
`PACEJKA`/`FIALA` are reserved enum values with stubs.

### 9.2 Brush / combined-slip default

Physically grounded, few parameters (slip stiffness + μ), graceful saturation:

- Slip quantities, low-speed regularized (reference speed floored to
  `min_reference_speed` so a braked/stopped wheel holds rather than oscillates):
  longitudinal slip from analytical `ω·r` vs contact-point longitudinal speed;
  slip angle from lateral vs longitudinal speed; theoretical slip vector
  `σ = (σ_x, σ_y)`.
- Combined brush force: `F = μ·Fz · g(θ·|σ|) · (−σ̂)`, where `g` rises with slope
  = bristle stiffness and saturates to 1, and `θ = (2/3)·c_p·a²/(μ·Fz)`.
  Longitudinal/lateral split falls out of `σ̂`; **combined slip is intrinsic**
  (no separate friction-circle clip). A `μ·Fz` magnitude cap is retained as a
  safety net.

### 9.3 Analytical wheel spin

One coherent path (replacing codex's `drive`+`tire`+`moment`). Per wheel:

```
I·dω/dt = τ_drive − τ_brake·sign(ω) − F_long·r − c_damp·ω − τ_rolling(ω)
```

- Integrated **semi-implicitly** in the `F_long(ω)` coupling for stability near
  lock-up/spin-up (slip↔spin is a stiff feedback loop).
- Brake torque uses a zero-crossing clamp so it cannot reverse the wheel.
- `τ_drive` comes from the command mapping (Section 10), in either a
  torque-limited speed-servo mode or a direct torque mode.

### 9.4 Force injection (verified convention)

The tire wrench is applied at the patch center to the wheel body, scatter-added:

```
F_world  = F_long * long_dir + F_lat * lat_dir
tau_world = cross(center - wheel_com_world, F_world)
state.body_f[wheel_body] += wp.spatial_vector(F_world, tau_world)   # [lin(3), ang(3)]
```

`body_f` is `[linear(3), angular(3)]`, world frame, referenced to COM. The solver
transmits the wrench through suspension/steering joints to the chassis. The
motor's **axle reaction torque** on the chassis (about the axle axis) is applied
as an optional body torque (default on) so weight-transfer/squat effects are
represented even though the wheel body does not physically spin.

## 10. Vehicle layer — drive modes and command mapping (`vehicle.py`)

`update_controls(control)` runs one batched kernel over wheels, branching on the
per-vehicle `drive_mode`:

- **GENERIC** — each driven wheel pulls its `drive_channel` directly to a
  per-wheel drive target.
- **ACKERMANN** — `drive_channel` → driven-wheel target; `steer_channel` → center
  angle (`u·steer_limit`) → per-wheel inner/outer angles from `wheelbase`/`track`
  → `control.joint_target_pos[steer_joint_dof]`.
- **SKID_STEER** — left/right `drive_channel`s → side wheel targets (opposite →
  spin-in-place); no steering.

Command semantics: a normalized drive command maps to a **target wheel speed
realized through a torque-limited servo** in the spin integrator —
`τ_drive = clamp(Kp·(ω_target − ω), ±τ_motor_max)`. Actual motion emerges from
the tire/slip dynamics, not a kinematic override. A pure torque mode is
first-class (`DriveInput.TORQUE`); powertrain curves slot in here later.

Steering joints use the solver's existing PD/servo actuator; we write
`joint_target_pos`. DOF indices come from `model.joint_q_start`/`joint_qd_start`.

## 11. Suspension and steering

Pure solver dynamics. Suspension prismatic joints (spring/damper in the asset)
are integrated entirely by the wrapped solver; the layer only reads the
resulting wheel body pose (which already includes suspension + steering effect)
when computing contact-point velocity and tire-axis directions. The layer writes
steering targets and otherwise does not touch these joints.

## 12. Public API surface (`newton.wheeled`)

- `WheeledVehicles` — controller object.
  - `WheeledVehicles(model, *, config=None)`
  - `set_commands(*, drive=None, steer=None, brake=None)`
  - `update_controls(control)`
  - `apply(state, contacts, dt)`
  - `latch_loads(contacts)`
  - Read-only diagnostics: `patch`, `wheel_state` (ω, slip, forces, Fz),
    `vehicle_state` (resolved targets) — flat device arrays for `.numpy()`.
  - `register_attributes(builder)` (static) — register `wheeled:*` attributes.
  - Nested enums (prefix-first, self-contained): `WheeledVehicles.DriveMode`,
    `WheeledVehicles.TireModel`, `WheeledVehicles.DriveInput`.
- `WheeledConfig` — global options: default tire model and params, fallback Fz,
  reaction-torque toggle, integration options.
- Build-time helpers (operate on `ModelBuilder`/`Model`):
  - `add_wheel(builder, shape, *, vehicle, radius, width, role, ...)` — annotate a
    wheel and set its `preserve_contact_footprint` flag + normal-only friction.
  - `configure_wheel_solver_contacts(model, ...)` — apply `condim=1`/priority.

Examples and docs import only from `newton.wheeled`. `newton/wheeled.py`
re-exports from the internal package. No symbol references `newton._src`.

## 13. Units and conventions

| Quantity | Units |
|---|---|
| length, radius, width, extent, center | m |
| normal / axis vectors | unit (dimensionless) |
| force, friction limit | N |
| torque | N·m |
| linear speed | m/s |
| angular speed ω, slip speed | rad/s |
| slip ratio | dimensionless |
| slip angle, steering angle/limit | rad |
| slip/cornering stiffness | model-dependent (documented per model) |
| μ | dimensionless |
| `body_f` | `[linear(3) N, angular(3) N·m]`, world frame at COM |

## 14. Module layout (internal, built fresh)

Developed in a parallel internal package alongside `newton/_src/wheeled/`:

| Module | Responsibility |
|---|---|
| `metadata.py` | register `wheeled:*` attributes; read finalized `Model` → flat tables; build-time `add_wheel`/friction helpers |
| `contact.py` | per-wheel patch extraction from `contacts`; `latch_loads` |
| `tire.py` | tire-model interface + brush (default) + linear; Pacejka/Fiala stubs |
| `wheel.py` | analytical spin integration + tire wrench injection + chassis reaction torque |
| `vehicle.py` | per-vehicle layout, drive modes, command mapping, steering targets |
| `controller.py` | `WheeledVehicles`, `WheeledConfig`; owns arrays; orchestrates kernels |

Six focused modules vs the prior eight, with the three force paths collapsed
into `tire.py` + `wheel.py`. The collision-mode change lives in
`newton/_src/geometry/` (Section 6), not here.

## 15. Testing strategy

Unit (device kernels, flat arrays, multi-vehicle batches):

- Metadata: single/multi-world build; heterogeneous merge (Ackermann + skid-steer
  in one model) yields correct per-instance ids; USDA and direct-annotation paths
  agree.
- Collision mode: flagged plane-cylinder / mesh / ramp pairs produce non-degenerate
  tangent extents; unflagged pairs match `main` exactly.
- Contact patch: inactive wheel; single vs multi-contact; sign of support normal
  for shape0/shape1; load latched from `contacts.force` matches a known mg case.
- Brush model: zero slip → zero force; pure longitudinal; pure lateral; combined
  slip saturates on the friction ellipse; μ·Fz cap; low-speed hold (no oscillation).
- Spin integration: free spin-up; brake to zero without reversal; tire reaction
  torque balance; rolling resistance.
- Command mapping: Ackermann inner/outer angle geometry; skid-steer spin-in-place;
  generic per-wheel; torque-limited speed servo vs torque mode.

Integration / examples (`Example` format, `test_final` + `test_post_step`):

- RC car (Ackermann) accelerates, steers, and brakes on flat ground and a ramp.
- Husky (skid-steer) drives straight and rotates in place.
- A heterogeneous batch (RC car + Husky, replicated) runs stably with no
  host-side per-vehicle branching.

Each regression test must fail before its implementation and pass after.

## 16. Rollout — build alongside, then swap

1. Build the new package in a parallel internal location with its own tests and
   examples; leave `newton/_src/wheeled/` untouched and working.
2. Land the scoped `preserve_contact_footprint` collision mode first (independent,
   reusable, separately reviewable).
3. Reach parity: both reference vehicles drive/steer/brake stably on flat + simple
   terrain; heterogeneous batch runs; performance acceptable at thousands of
   vehicles.
4. Swap: repoint the public module to the new implementation; delete
   `newton/_src/wheeled/`, its examples and tests; run `docs/generate_api.py`;
   update `README.md` and `CHANGELOG.md`. As an unreleased feature branch, no
   deprecation cycle is required.

## 17. Open questions

- Final public name at swap: keep `newton.wheeled`, or rename to
  `newton.vehicles`. Leaning `newton.wheeled` (matches the branch and intent).
- Brush parameterization: expose bristle stiffness `c_p` + contact length `2a`
  directly, or expose lumped longitudinal/lateral slip stiffnesses and derive
  `c_p` internally. Leaning lumped stiffnesses for ergonomics.
- Whether `apply()` should optionally fold `latch_loads()` using the previous
  step's already-populated `contacts.force`, to reduce the loop to three calls.
- Reaction-torque default: on (faithful) vs off (simpler) — leaning on.
