# Wheeled Free Fall Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a wheeled example that drops a single RC car without terrain while steering and drive commands exercise analytical wheel moments.

**Architecture:** Create a focused `example_wheeled_free_fall.py` alongside the other wheeled examples, reusing the RC car manifest, vehicle command mapping, and wheel moment pipeline from `example_wheeled_terrain_contact.py`. The example omits ground/course geometry, follows the tracked chassis with the camera, and restores the model to the initial state when the chassis falls below `z = -100 m`.

**Tech Stack:** Newton examples framework, Warp kernels and arrays, MuJoCo solver bridge, Newton wheeled public API, `unittest` example subprocess registration.

---

### Task 1: Register the Expected Example Test

**Files:**
- Modify: `newton/tests/test_examples.py`

- [ ] **Step 1: Write the failing test registration**

Add this block in `TestWheeledExamples`, after `example_wheeled_terrain_contact`:

```python
add_example_test(
    TestWheeledExamples,
    name="wheeled.example_wheeled_free_fall",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 180, "world-count": 1},
    use_viewer=True,
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_examples.TestWheeledExamples.test_wheeled_example_wheeled_free_fall
```

Expected: fail during registration or subprocess launch because `newton/examples/wheeled/example_wheeled_free_fall.py` does not exist yet.

### Task 2: Add the Free-Fall Example

**Files:**
- Create: `newton/examples/wheeled/example_wheeled_free_fall.py`

- [ ] **Step 1: Implement the example**

Create the example by adapting the terrain-contact example with these concrete changes:

```python
RESET_HEIGHT = -100.0
TIRE_FRICTION_MU = 1.0
TIRE_FALLBACK_NORMAL_LOAD = 14.0
TIRE_LONGITUDINAL_STIFFNESS = 70.0
TIRE_LATERAL_STIFFNESS = 60.0
MAX_WHEEL_ANGULAR_SPEED = 28.0
MAX_WHEEL_DRIVE_TORQUE = 0.65
COAST_BRAKE_TORQUE = 0.05
WHEEL_INERTIA = 0.01
WHEEL_ANGULAR_DAMPING = 0.008
WHEEL_ROLLING_RESISTANCE_TORQUE = 0.01
```

The scene must call `scene.replicate(world, self.world_count)` and must not call `scene.add_ground_plane()` or add terrain shapes. Keep `newton.wheeled.configure_mujoco_wheel_contacts(self.model, self.wheeled_metadata)` so the same normal-only wheel contact setup is exercised if geometry is later added, while the free-fall scene has no contact geometry.

The step loop must:

```python
self._update_vehicle_commands()
for _ in range(self.sim_substeps):
    self.state_0.clear_forces()
    self.control.clear()
    self.viewer.apply_forces(self.state_0)
    newton.wheeled.update_wheeled_vehicle_controls(...)
    wp.launch(_update_moment_drive_commands, ...)
    self.model.collide(self.state_0, self.contacts)
    newton.wheeled.update_wheel_contact_patches(...)
    newton.wheeled.apply_wheel_tire_forces(...)
    newton.wheeled.update_wheel_moments(...)
    self.solver.step(...)
    self.state_0, self.state_1 = self.state_1, self.state_0
self._reset_if_too_low()
self.sim_time += self.frame_dt
```

The reset helper must restore `body_q`, `body_qd`, `joint_q`, and `joint_qd` from snapshots captured after `eval_fk`, then clear force/control/contact state and zero transient wheeled state arrays that have `zero_()` or `fill_()` methods. It must increment `self._reset_count` so `test_final()` can verify the reset path when the test runs long enough.

- [ ] **Step 2: Run the example test**

Run:

```bash
uv run --extra dev -m newton.examples wheeled_free_fall --viewer null --test --quiet --num-frames 180 --world-count 1
```

Expected: pass, with nonzero moment wheel speed and bounded body state.

### Task 3: Document and Verify Discovery

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README gallery entry**

Insert `wheeled_free_fall` in the wheeled examples table, reusing the existing wheeled screenshot URL and linking to:

```text
newton/examples/wheeled/example_wheeled_free_fall.py
```

- [ ] **Step 2: Verify short-name discovery**

Run:

```bash
uv run --extra dev -m newton.examples wheeled_free_fall --viewer null --test --quiet --num-frames 10 --world-count 1
```

Expected: pass; this proves `python -m newton.examples wheeled_free_fall` resolves via automatic example discovery.

### Task 4: Final Verification

**Files:**
- Modify: `newton/tests/test_examples.py`
- Create: `newton/examples/wheeled/example_wheeled_free_fall.py`
- Modify: `README.md`

- [ ] **Step 1: Run targeted example test**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_examples.TestWheeledExamples.test_wheeled_example_wheeled_free_fall
```

Expected: pass.

- [ ] **Step 2: Run formatting/linting**

Run:

```bash
uvx pre-commit run -a
```

Expected: pass, or report any unrelated pre-existing failures separately.
