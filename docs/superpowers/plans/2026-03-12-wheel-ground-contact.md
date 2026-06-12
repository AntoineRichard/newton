# Wheel Ground Contact via Repulsion Force — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace wheel collision shapes with a ground repulsion force kernel so the car solver's Pacejka tire model is the sole source of friction.

**Architecture:** New Warp kernel `eval_ground_contact_kernel` applies a penalty force when wheels penetrate the ground plane. Wheel collision shapes become visual-only (`as_site=True`). A box collision shape is added to the chassis for body-level collisions.

**Tech Stack:** Warp (GPU kernels), Newton physics engine (custom attributes, ModelBuilder)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `newton/_src/solvers/car/kernels_ground.py` | Create | Ground contact penalty force kernel |
| `newton/tests/test_car_ground.py` | Create | Unit tests for ground contact kernel |
| `newton/_src/solvers/car/solver_car.py` | Modify | Launch ground kernel in `step()`, read new custom attrs |
| `newton/_src/solvers/car/car_builder.py` | Modify | Visual-only wheel shapes, chassis box, new custom attrs, `ground_altitude` stored |
| `newton/_src/solvers/car/car_descriptor.py` | Modify | New `ground_ke`/`ground_kd` params, pass `ground_altitude` as custom attr |
| `newton/examples/car/example_car_circles.py` | Modify | Remove collision pipeline, adapt spawn heights |

---

## Chunk 1: Ground contact kernel + tests

### Task 1: Ground contact kernel

**Files:**
- Create: `newton/_src/solvers/car/kernels_ground.py`
- Create: `newton/tests/test_car_ground.py`

- [ ] **Step 1: Write the ground contact kernel**

Create `newton/_src/solvers/car/kernels_ground.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernel for wheel-ground contact via penalty force."""

import warp as wp


@wp.kernel
def eval_ground_contact_kernel(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    wheel_body_idx: wp.array(dtype=wp.int32),
    wheel_radius: wp.array(dtype=float),
    ground_altitude: float,
    ground_ke: float,
    ground_kd: float,
    # output
    body_f: wp.array(dtype=wp.spatial_vectorf),
):
    """Apply upward penalty force when a wheel penetrates the ground plane.

    One thread per wheel. Writes to ``body_f`` via atomic add on the wheel body.

    Args:
        body_q: Body transforms, shape ``(n_bodies,)``.
        body_qd: Body spatial velocities, shape ``(n_bodies,)``.
        wheel_body_idx: Wheel body index, shape ``(n_wheels,)``.
        wheel_radius: Per-wheel radius [m], shape ``(n_wheels,)``.
        ground_altitude: Ground plane Y coordinate [m].
        ground_ke: Ground contact stiffness [N/m].
        ground_kd: Ground contact damping [N*s/m].
        body_f: Output spatial forces, shape ``(n_bodies,)``.
    """
    tid = wp.tid()
    wb = wheel_body_idx[tid]
    wheel_pos = wp.transform_get_translation(body_q[wb])
    wheel_vel = wp.spatial_top(body_qd[wb])
    penetration = (ground_altitude + wheel_radius[tid]) - wheel_pos[1]
    if penetration > 0.0:
        damping_vel = wp.max(-wheel_vel[1], 0.0)
        force_y = ground_ke * penetration + ground_kd * damping_vel
        wp.atomic_add(body_f, wb, wp.spatial_vectorf(0.0, force_y, 0.0, 0.0, 0.0, 0.0))
```

- [ ] **Step 2: Write failing tests**

Create `newton/tests/test_car_ground.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.car.kernels_ground import eval_ground_contact_kernel


class TestGroundContact(unittest.TestCase):
    def _run_kernel(self, wheel_y, wheel_vy=0.0, radius=0.3, altitude=0.0, ke=2500.0, kd=100.0):
        """Helper: single wheel, returns body_f array."""
        body_q = wp.zeros(1, dtype=wp.transformf)
        body_qd = wp.zeros(1, dtype=wp.spatial_vectorf)

        q_np = body_q.numpy()
        q_np[0] = [0.0, wheel_y, 0.0, 0.0, 0.0, 0.0, 1.0]
        body_q = wp.array(q_np, dtype=wp.transformf)

        qd_np = body_qd.numpy()
        qd_np[0] = [0.0, wheel_vy, 0.0, 0.0, 0.0, 0.0]
        body_qd = wp.array(qd_np, dtype=wp.spatial_vectorf)

        wheel_idx = wp.array([0], dtype=wp.int32)
        wheel_r = wp.array([radius], dtype=float)
        body_f = wp.zeros(1, dtype=wp.spatial_vectorf)

        wp.launch(
            eval_ground_contact_kernel,
            dim=1,
            inputs=[body_q, body_qd, wheel_idx, wheel_r, altitude, ke, kd],
            outputs=[body_f],
        )
        return body_f.numpy()

    def test_wheel_above_ground_no_force(self):
        """Wheel center at y=0.5, radius=0.3, ground at 0 => bottom at 0.2, no contact."""
        bf = self._run_kernel(wheel_y=0.5, radius=0.3, altitude=0.0)
        np.testing.assert_allclose(bf[0][1], 0.0, atol=1e-6)

    def test_wheel_at_ground_no_force(self):
        """Wheel center at y=radius exactly touching ground => penetration=0, no force."""
        bf = self._run_kernel(wheel_y=0.3, radius=0.3, altitude=0.0)
        np.testing.assert_allclose(bf[0][1], 0.0, atol=1e-6)

    def test_wheel_below_ground_repulsion(self):
        """Wheel center at y=0.2, radius=0.3, ground at 0 => penetration=0.1, upward force."""
        bf = self._run_kernel(wheel_y=0.2, radius=0.3, altitude=0.0, ke=2500.0, kd=0.0)
        expected = 2500.0 * 0.1  # 250 N
        np.testing.assert_allclose(bf[0][1], expected, rtol=0.01)

    def test_damping_when_moving_down(self):
        """Downward velocity adds damping force."""
        bf = self._run_kernel(wheel_y=0.2, wheel_vy=-1.0, radius=0.3, ke=0.0, kd=100.0)
        expected = 100.0 * 1.0  # 100 N (damping only)
        np.testing.assert_allclose(bf[0][1], expected, rtol=0.01)

    def test_no_damping_when_moving_up(self):
        """Upward velocity => no damping contribution (clamped to 0)."""
        bf = self._run_kernel(wheel_y=0.2, wheel_vy=1.0, radius=0.3, ke=0.0, kd=100.0)
        np.testing.assert_allclose(bf[0][1], 0.0, atol=1e-6)

    def test_ground_altitude_offset(self):
        """Ground at altitude=1.0, wheel at y=1.2, radius=0.3 => penetration=0.1."""
        bf = self._run_kernel(wheel_y=1.2, radius=0.3, altitude=1.0, ke=2500.0, kd=0.0)
        expected = 2500.0 * 0.1
        np.testing.assert_allclose(bf[0][1], expected, rtol=0.01)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run --extra dev -m pytest newton/tests/test_car_ground.py -v`
Expected: 6 PASS

- [ ] **Step 4: Commit**

```bash
git add newton/_src/solvers/car/kernels_ground.py newton/tests/test_car_ground.py
git commit -m "Add ground contact penalty force kernel with tests"
```

---

## Chunk 2: Wire kernel into solver + builder changes

### Task 2: Register ground custom attributes

**Files:**
- Modify: `newton/_src/solvers/car/solver_car.py:509-547`
- Modify: `newton/_src/solvers/car/car_builder.py:315-434`

- [ ] **Step 1: Add custom attribute registrations**

In `solver_car.py`, inside `register_custom_attributes()`, after the existing per-world attributes (after line 522), add:

```python
_add("ground_altitude", Model.AttributeFrequency.WORLD, wp.float32, 0.0)
_add("ground_ke", Model.AttributeFrequency.WORLD, wp.float32, 2500.0)
_add("ground_kd", Model.AttributeFrequency.WORLD, wp.float32, 100.0)
```

- [ ] **Step 2: Write ground attributes in CarBuilder.build()**

In `car_builder.py`, inside the `build()` method's world attributes dict (around line 413), add the three new keys:

```python
f"{_NAMESPACE}:ground_altitude": self._ground_altitude,
f"{_NAMESPACE}:ground_ke": self._ground_ke,
f"{_NAMESPACE}:ground_kd": self._ground_kd,
```

Store the values from `add_suspension()` or a new method. The simplest approach: add `ground_altitude`, `ground_ke`, `ground_kd` parameters to `CarBuilder.__init__` or track them as instance state set by `add_ground()`.

Update `add_ground()` to store the altitude instead of (or in addition to) creating a ground plane:

```python
def add_ground(self, altitude: float = 0.0, ke: float = 2500.0, kd: float = 100.0) -> int:
    self._ground_altitude = altitude
    self._ground_ke = ke
    self._ground_kd = kd
    return self._builder.add_ground_plane(height=altitude)
```

Initialize defaults in `__init__`:
```python
self._ground_altitude: float = 0.0
self._ground_ke: float = 2500.0
self._ground_kd: float = 100.0
```

- [ ] **Step 3: Read ground attributes in SolverCarDynamics.__init__()**

In `solver_car.py`, in `__init__()`, after reading per-car attributes (around line 161), add:

```python
self._ground_altitude = float(car.ground_altitude.numpy()[0])
self._ground_ke = float(car.ground_ke.numpy()[0])
self._ground_kd = float(car.ground_kd.numpy()[0])
```

- [ ] **Step 4: Launch ground kernel in step()**

In `solver_car.py`, in `step()`, add as step 0 (before step 1 "Extract per-wheel state"), importing the kernel at the top of the file:

```python
from .kernels_ground import eval_ground_contact_kernel
```

In `step()`, before the existing step 1:

```python
# 0. Ground contact penalty force on wheel bodies
wp.launch(
    eval_ground_contact_kernel,
    dim=n_wheels,
    inputs=[
        state_in.body_q,
        state_in.body_qd,
        self._wheel_body_idx,
        self._wheel_radius,
        self._ground_altitude,
        self._ground_ke,
        self._ground_kd,
    ],
    outputs=[state_in.body_f],
)
```

- [ ] **Step 5: Run all car tests**

Run: `uv run --extra dev -m pytest newton/tests/ -k car -v`
Expected: All pass (ground kernel now wired in but default altitude=0.0 matches previous behavior)

- [ ] **Step 6: Commit**

```bash
git add newton/_src/solvers/car/solver_car.py newton/_src/solvers/car/car_builder.py
git commit -m "Wire ground contact kernel into car solver and builder"
```

### Task 3: Visual-only wheel shapes + chassis box

**Files:**
- Modify: `newton/_src/solvers/car/car_builder.py:162-179`
- Modify: `newton/_src/solvers/car/car_builder.py:315-362`

- [ ] **Step 1: Make wheel collision shapes visual-only**

In `car_builder.py`, change `add_wheel_collision()` to use `as_site=True`:

```python
def add_wheel_collision(self, shape: str = "sphere") -> None:
    if not self._wheel_bodies:
        raise ValueError("No wheels added — call add_wheel() first")
    if shape != "sphere":
        raise ValueError(f"Unsupported collision shape: {shape!r}. Only 'sphere' is supported.")

    for wheel_body, radius in zip(self._wheel_bodies, self._wheel_radii, strict=True):
        self._builder.add_shape_sphere(body=wheel_body, radius=radius, as_site=True)
```

- [ ] **Step 2: Add chassis box collision in build()**

In `car_builder.py`, inside `build()`, after creating joints and before writing custom attributes, add a chassis collision box. Derive dimensions from wheel positions:

```python
# Add chassis collision box
if self._wheel_positions:
    import numpy as _np
    pos_arr = _np.array([[p[0], p[1], p[2]] for p in self._wheel_positions])
    half_x = (pos_arr[:, 0].max() - pos_arr[:, 0].min()) / 2.0 + 0.1  # lateral + margin
    half_z = (pos_arr[:, 2].max() - pos_arr[:, 2].min()) / 2.0 + 0.1  # longitudinal + margin
    half_y = 0.25  # reasonable default chassis height
    self._builder.add_shape_box(
        body=self._chassis_body,
        hx=half_x,
        hy=half_y,
        hz=half_z,
        xform=wp.transform(wp.vec3(0.0, 0.3, 0.0), wp.quat_identity()),
    )
```

- [ ] **Step 3: Run all car tests**

Run: `uv run --extra dev -m pytest newton/tests/ -k car -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add newton/_src/solvers/car/car_builder.py
git commit -m "Make wheel shapes visual-only, add chassis collision box"
```

### Task 4: Update CarDescriptor and example

**Files:**
- Modify: `newton/_src/solvers/car/car_descriptor.py:91-190`
- Modify: `newton/examples/car/example_car_circles.py`

- [ ] **Step 1: Add ground_ke/ground_kd to CarDescriptor**

In `car_descriptor.py`, add fields after `ground_altitude`:

```python
ground_ke: float = 2500.0
"""Ground contact stiffness [N/m]."""
ground_kd: float = 100.0
"""Ground contact damping [N*s/m]."""
```

Update the `build()` method to pass them to `add_ground()`:

```python
if self.ground_altitude is not None:
    car.add_ground(altitude=self.ground_altitude, ke=self.ground_ke, kd=self.ground_kd)
```

- [ ] **Step 2: Update example_car_circles.py**

The example currently creates a collision pipeline for wheel-ground contacts. With visual-only wheels, the collision pipeline is still needed for the chassis box, but the example should work without changes since the collision pipeline handles whatever collidable shapes exist.

Verify the example still passes its `test_final()` by running:

Run: `uv run --extra dev -m pytest newton/tests/ -k car_circles -v`

If the car no longer moves (because ground contact force isn't strong enough), tune `ground_ke` in the descriptor. The default `ke=2500` may need to be higher — try `50000.0` to match suspension stiffness order of magnitude.

- [ ] **Step 3: Run full car test suite**

Run: `uv run --extra dev -m pytest newton/tests/ -k car -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add newton/_src/solvers/car/car_descriptor.py newton/examples/car/example_car_circles.py
git commit -m "Add ground contact params to CarDescriptor, update example"
```

---

## Tuning Notes

The `ground_ke` value needs to be stiff enough that the wheel doesn't sink noticeably into the ground under the car's weight. A 1500 kg car on 4 wheels = ~3750 N per wheel. With `ke=2500`, that's 1.5m of penetration — way too much. Starting value should be closer to `ground_ke=50000.0` (7.5 cm penetration under static load) or higher. Adjust during integration testing.
