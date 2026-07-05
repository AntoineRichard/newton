# MPPI RC-car Track Racing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable `newton.vehicles.ControllerMPPI` planner plus an example that races the RC car around a track_gen-generated bezier track using N+1 collocated simulated worlds as the MPPI rollout model.

**Architecture:** One model with `num_samples` worlds replicated at spacing (0,0,0) — world 0 is the rendered hero and runs the zero-noise nominal sample; worlds 1..K-1 are particles. Each frame: snapshot hero slice → broadcast to all worlds → sample noisy command sequences → rollout the horizon accumulating progress reward and OOB kills → MPPI update → restore hero → hero executes `nominal[0]` for one real frame. Track signals come from track_gen utilities at E = K envs sharing identical-seed geometry.

**Tech Stack:** Warp kernels + CUDA graph capture, `newton.vehicles.WheeledVehicles`, `SolverMuJoCo`, track_gen (`TrackGenerator`, `CheckpointSampler`, `ProgressTracker`, `CollisionChecker`, `PropSampler`), ViewerGL instanced rendering.

Spec: `docs/superpowers/specs/2026-07-05-mppi-track-design.md`.

## Global Constraints

- `newton/_src/` is internal; example imports only `newton`, `newton.examples`, `newton.vehicles`.
- unittest, never pytest. Test runs: `uv run --extra dev -m newton.tests -k <pattern>`.
- No new required dependencies: track_gen is imported only by the example, with an install-hint `ImportError`. Dev env already has `uv pip install -e ../track_gen`.
- PEP 604 unions; `wp.array[dtype]` bracket annotations; Google docstrings; SI units in public docstrings; Sphinx cross-refs without `newton._src`.
- SPDX year 2026 on new files.
- Prefix-first naming (`ControllerMPPI`); nested `Config` dataclass with field docstrings below fields.
- Imperative commit subjects ~50 chars; body wraps at 72; `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `uvx pre-commit run -a` before each commit.
- CHANGELOG entries at random positions within the correct `[Unreleased]` category.
- Branch: `antoiner/mppi-track-example` (already created).

### Verified external facts (do not re-derive)

- `WheeledVehicles.commands.drive/.steer/.brake` are `wp.array[wp.float32]` of length `vehicle_count` (`newton/_src/vehicles/vehicle.py:41-43`) — write per-vehicle from kernels; do NOT use `set_commands` (host sync).
- Stateful per-wheel arrays needing hero resync: `vehicles.dynamics.omega`, `.trans_long`, `.trans_lat` (`wheel.py:79,83-84`), `vehicles.patch.fz` (`contact.py:92`).
- `ModelBuilder.replicate(builder, count, spacing=(0,0,0))` collocates worlds; cross-world collision filtered by `shape_world`; ground plane added on the scene builder after `replicate` is global (world -1). Pattern: `newton/examples/wheeled/example_wheeled_car_control.py:113-114`.
- `newton.ShapeFlags.VISIBLE = 1 << 0` (`newton/_src/geometry/flags.py:23`); ViewerGL reads `model.shape_flags.numpy()` in `set_model` (`viewer.py:1600`), so clearing the bit for particle-world shapes before `viewer.set_model` hides them.
- track_gen `PerEnvSeededRNG(seeds=wp.array-of-equal-int32, num_envs=E)` ⇒ identical tracks in all envs (no env-index folding). Scalar int seeds ⇒ per-env different (`rng_utils.py:41-44`).
- `TrackGenerator` is fixed-batch; to retry a failed seed, rebuild `PerEnvSeededRNG` + `TrackGenerator` with seed+1 (init-time only, pre-capture).
- `Track` buffers are flat `[E*N_max]` vec2f, env e at `[e*N_max : e*N_max+count[e]]`; `track.valid`/`track.count` are `[E]` int32.
- `ProgressTracker` internal state to snapshot/broadcast: `_prev_pos` (`wp.array2d[float]`, [E,2]), `_next`, `_laps`, `_progress` ([E] int32) (`track_gen/_src/progress.py:254-258`). `ProgressEvents` fields: `passed`, `checkpoint_passed`, `next_checkpoint`, `laps`, `progress`, `wrong_way`, `wrong_checkpoint`, `dist_to_next`.
- `CollisionChecker(track, max_boxes=1, method="segments")` + `bind_inputs(position, yaw, half_extents)` + `query()`; buffers `[E]` (max_boxes=1); `contact.oob` [E] int32.
- `PropSampler(track, spacing, boundary="inner"|"outer", mode="points")`; `sample()` → `PropSet` with `position` (vec2f), `yaw` (f32), `count` [E].
- Viewer: `log_shapes(name, geo_type, geo_scale, xforms, colors=None, ..., geo_src=<newton.Mesh>)` (`viewer.py:787`); `log_lines(name, starts, ends, colors)` (`viewer.py:1138`); `gui(ui)` receives raw imgui (`plot_lines`, `progress_bar` available).
- Render-only mesh from USD: `Usd.Stage.Open` → `UsdGeom.Mesh` points/indices → `newton.Mesh(vertices, indices)`; `.finalize()`; pass as `geo_src` (pattern: `example_basic_viewer.py:43-48`).
- CUDA graph: `with wp.ScopedCapture() as cap: ...; wp.capture_launch(cap.graph)` (`example_robot_h1.py:96-124`); call `track_gen.set_capturing(True)` before capturing track_gen kernels.
- MuJoCo solver state of record is `joint_q`/`joint_qd` (broadcast BOTH joint- and body-space state to be safe).

---

### Task 1: `ControllerMPPI` planner + unit tests + public export

**Files:**
- Create: `newton/_src/vehicles/mppi.py`
- Modify: `newton/_src/vehicles/__init__.py` (add export)
- Modify: `newton/vehicles.py` (public export + `__all__`)
- Test: `newton/tests/test_vehicles_mppi.py`
- Run: `docs/generate_api.py`

**Interfaces:**
- Produces: `newton.vehicles.ControllerMPPI` with nested `Config` dataclass; attributes `nominal: wp.array2d[float]` [H, A], `samples: wp.array3d[float]` [K, H, A], `sigma/bounds_lo/bounds_hi: wp.array[float]` [A]; methods `sample()`, `update(costs: wp.array[float])`, `shift()`, `set_temperature(float)`, `set_beta(float)`. All methods are pure `wp.launch` (graph-capturable); temperature/beta live in `[1]` device arrays so they stay tunable under capture.

- [ ] **Step 1: Write the failing tests** — `newton/tests/test_vehicles_mppi.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton.vehicles as nv


def _make(**overrides):
    kwargs = dict(num_samples=64, horizon=8, dim=2, sigma=(0.3, 0.4), seed=11)
    kwargs.update(overrides)
    return nv.ControllerMPPI(config=nv.ControllerMPPI.Config(**kwargs))


class TestControllerMPPI(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(dim=2, sigma=(0.1,)))
        with self.assertRaises(ValueError):
            nv.ControllerMPPI(config=nv.ControllerMPPI.Config(num_samples=1))

    def test_sample_zero_is_nominal(self):
        planner = _make()
        nom = np.linspace(-0.5, 0.5, 8 * 2).astype(np.float32).reshape(8, 2)
        planner.nominal.assign(nom)
        planner.sample()
        samples = planner.samples.numpy()
        np.testing.assert_allclose(samples[0], nom, atol=1e-6)
        # other samples actually differ from the nominal
        self.assertGreater(float(np.abs(samples[1:] - nom).max()), 1e-3)

    def test_sample_respects_bounds(self):
        planner = _make(sigma=(5.0, 5.0))
        planner.sample()
        samples = planner.samples.numpy()
        self.assertLessEqual(float(samples.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(samples.min()), -1.0 - 1e-6)

    def test_successive_samples_differ(self):
        planner = _make()
        planner.sample()
        first = planner.samples.numpy().copy()
        planner.sample()
        self.assertGreater(float(np.abs(planner.samples.numpy() - first).max()), 1e-3)

    def test_seed_determinism(self):
        a, b = _make(), _make()
        a.sample()
        b.sample()
        np.testing.assert_allclose(a.samples.numpy(), b.samples.numpy())

    def test_update_moves_nominal_toward_low_cost_sample(self):
        planner = _make()
        planner.set_temperature(1e-3)  # winner-takes-all weights
        planner.sample()
        samples = planner.samples.numpy()
        costs = np.full(64, 1e3, dtype=np.float32)
        costs[5] = 0.0
        planner.update(wp.array(costs, dtype=wp.float32, device=planner.device))
        np.testing.assert_allclose(planner.nominal.numpy(), samples[5], atol=1e-3)

    def test_shift_rolls_and_repeats_last(self):
        planner = _make()
        nom = np.arange(8 * 2, dtype=np.float32).reshape(8, 2) * 0.01
        planner.nominal.assign(nom)
        planner.shift()
        out = planner.nominal.numpy()
        np.testing.assert_allclose(out[:-1], nom[1:], atol=1e-6)
        np.testing.assert_allclose(out[-1], nom[-1], atol=1e-6)


if __name__ == "__main__":
    wp.init()
    unittest.main()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_mppi`
Expected: errors — `AttributeError: module 'newton.vehicles' has no attribute 'ControllerMPPI'`.

- [ ] **Step 3: Implement `newton/_src/vehicles/mppi.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sampling-based MPPI planner over batched simulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@wp.kernel
def _sample_sequences(
    nominal: wp.array2d[float],
    sigma: wp.array[float],
    bounds_lo: wp.array[float],
    bounds_hi: wp.array[float],
    beta: wp.array[float],
    seed: int,
    counter: wp.array[wp.int32],
    noise: wp.array3d[float],
    samples: wp.array3d[float],
):
    k, a = wp.tid()
    horizon = nominal.shape[0]
    if k == 0:
        # sample 0 is always the zero-noise nominal
        for t in range(horizon):
            noise[0, t, a] = 0.0
            samples[0, t, a] = nominal[t, a]
        return
    state = wp.rand_init(seed + counter[0], k * nominal.shape[1] + a)
    b = beta[0]
    scale = wp.sqrt(wp.max(0.0, 1.0 - b * b))
    n = float(0.0)
    for t in range(horizon):
        eps = sigma[a] * wp.randn(state)
        if t == 0:
            n = eps
        else:
            n = b * n + scale * eps
        s = wp.clamp(nominal[t, a] + n, bounds_lo[a], bounds_hi[a])
        samples[k, t, a] = s
        # effective (post-clamp) noise so update() respects the bounds
        noise[k, t, a] = s - nominal[t, a]


@wp.kernel
def _advance_counter(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


@wp.kernel
def _min_cost(costs: wp.array[float], out: wp.array[float]):
    m = costs[0]
    for k in range(1, costs.shape[0]):
        m = wp.min(m, costs[k])
    out[0] = m


@wp.kernel
def _softmax_weights(
    costs: wp.array[float],
    min_cost: wp.array[float],
    temperature: wp.array[float],
    weights: wp.array[float],
):
    k = wp.tid()
    weights[k] = wp.exp(-(costs[k] - min_cost[0]) / wp.max(temperature[0], 1.0e-6))


@wp.kernel
def _sum_weights(weights: wp.array[float], out: wp.array[float]):
    s = float(0.0)
    for k in range(weights.shape[0]):
        s += weights[k]
    out[0] = s


@wp.kernel
def _update_nominal(
    noise: wp.array3d[float],
    weights: wp.array[float],
    weight_sum: wp.array[float],
    bounds_lo: wp.array[float],
    bounds_hi: wp.array[float],
    nominal: wp.array2d[float],
):
    t, a = wp.tid()
    acc = float(0.0)
    for k in range(weights.shape[0]):
        acc += weights[k] * noise[k, t, a]
    nominal[t, a] = wp.clamp(
        nominal[t, a] + acc / wp.max(weight_sum[0], 1.0e-9), bounds_lo[a], bounds_hi[a]
    )


@wp.kernel
def _shift_nominal(nominal: wp.array2d[float]):
    a = wp.tid()
    for t in range(nominal.shape[0] - 1):
        nominal[t, a] = nominal[t + 1, a]


class ControllerMPPI:
    """Model Predictive Path Integral planner over externally simulated rollouts.

    The planner owns the nominal command sequence and the sampled candidate
    sequences; the caller owns the rollouts and the per-sample cost array.
    One replan cycle is::

        planner.sample()          # fill planner.samples [K, H, A]
        costs = rollout(...)      # caller: simulate sample k, accumulate cost[k]
        planner.update(costs)     # softmax-weighted update of the nominal
        command = planner.nominal # execute row 0, then
        planner.shift()           # warm-start the next cycle

    All methods launch Warp kernels only (no host synchronization), so the
    full cycle can be recorded into a CUDA graph. Temperature and noise
    smoothing live in device arrays and remain adjustable while captured.
    """

    @dataclass
    class Config:
        num_samples: int = 1024
        """Number of sampled command sequences K. Sample 0 is always the zero-noise nominal."""
        horizon: int = 32
        """Planning horizon H in control steps."""
        dim: int = 2
        """Number of command channels A per step."""
        sigma: tuple[float, ...] = (0.3, 0.4)
        """Per-channel exploration noise standard deviation, length ``dim``."""
        temperature: float = 0.05
        """Softmax temperature; lower concentrates the update on the best samples."""
        beta: float = 0.7
        """Per-step noise smoothing in [0, 1); 0 is white noise."""
        bounds_lo: tuple[float, ...] = (-1.0, -1.0)
        """Per-channel lower command bounds, length ``dim``."""
        bounds_hi: tuple[float, ...] = (1.0, 1.0)
        """Per-channel upper command bounds, length ``dim``."""
        seed: int = 0
        """Base RNG seed; resampling advances an internal device counter."""

    def __init__(self, config: Config | None = None, device=None):
        cfg = config if config is not None else ControllerMPPI.Config()
        if cfg.num_samples < 2:
            raise ValueError("num_samples must be >= 2 (sample 0 is the nominal)")
        if cfg.horizon < 2:
            raise ValueError("horizon must be >= 2")
        for name in ("sigma", "bounds_lo", "bounds_hi"):
            if len(getattr(cfg, name)) != cfg.dim:
                raise ValueError(f"{name} must have length dim={cfg.dim}")
        if not 0.0 <= cfg.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.config = cfg
        self.device = wp.get_device(device)
        k, h, a = cfg.num_samples, cfg.horizon, cfg.dim
        with wp.ScopedDevice(self.device):
            self.nominal = wp.zeros((h, a), dtype=wp.float32)
            self.samples = wp.zeros((k, h, a), dtype=wp.float32)
            self.noise = wp.zeros((k, h, a), dtype=wp.float32)
            self.weights = wp.zeros(k, dtype=wp.float32)
            self.sigma = wp.array(cfg.sigma, dtype=wp.float32)
            self.bounds_lo = wp.array(cfg.bounds_lo, dtype=wp.float32)
            self.bounds_hi = wp.array(cfg.bounds_hi, dtype=wp.float32)
            self._temperature = wp.array([cfg.temperature], dtype=wp.float32)
            self._beta = wp.array([cfg.beta], dtype=wp.float32)
            self._counter = wp.zeros(1, dtype=wp.int32)
            self._min_cost = wp.zeros(1, dtype=wp.float32)
            self._weight_sum = wp.zeros(1, dtype=wp.float32)

    def set_temperature(self, value: float) -> None:
        """Sets the softmax temperature (safe while a CUDA graph is captured)."""
        self._temperature.fill_(float(value))

    def set_beta(self, value: float) -> None:
        """Sets the noise smoothing factor (safe while a CUDA graph is captured)."""
        self._beta.fill_(float(value))

    def sample(self) -> None:
        """Fills :attr:`samples` with clamped nominal + smoothed Gaussian noise."""
        cfg = self.config
        wp.launch(
            _sample_sequences,
            dim=(cfg.num_samples, cfg.dim),
            inputs=[self.nominal, self.sigma, self.bounds_lo, self.bounds_hi,
                    self._beta, cfg.seed, self._counter],
            outputs=[self.noise, self.samples],
            device=self.device,
        )
        wp.launch(_advance_counter, dim=1, inputs=[self._counter], device=self.device)

    def update(self, costs: wp.array[float]) -> None:
        """Applies the MPPI softmax-weighted noise average to the nominal.

        Args:
            costs: Per-sample accumulated rollout costs, shape [num_samples].
        """
        cfg = self.config
        wp.launch(_min_cost, dim=1, inputs=[costs, self._min_cost], device=self.device)
        wp.launch(_softmax_weights, dim=cfg.num_samples,
                  inputs=[costs, self._min_cost, self._temperature, self.weights],
                  device=self.device)
        wp.launch(_sum_weights, dim=1, inputs=[self.weights, self._weight_sum],
                  device=self.device)
        wp.launch(_update_nominal, dim=(cfg.horizon, cfg.dim),
                  inputs=[self.noise, self.weights, self._weight_sum,
                          self.bounds_lo, self.bounds_hi, self.nominal],
                  device=self.device)

    def shift(self) -> None:
        """Rolls the nominal one step forward, repeating the final row."""
        wp.launch(_shift_nominal, dim=self.config.dim, inputs=[self.nominal],
                  device=self.device)
```

- [ ] **Step 4: Export.** In `newton/_src/vehicles/__init__.py` add `from .mppi import ControllerMPPI` next to the existing imports and extend its `__all__` if present. In `newton/vehicles.py` add `ControllerMPPI` to the `from ._src.vehicles import (...)` block and to `__all__` (alphabetical position).

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_mppi`
Expected: all 7 tests PASS.

- [ ] **Step 6: API docs + lint + commit**

Run: `uv run docs/generate_api.py`, then `uvx pre-commit run -a`.

```bash
git add newton/_src/vehicles/mppi.py newton/_src/vehicles/__init__.py newton/vehicles.py newton/tests/test_vehicles_mppi.py docs/
git commit -m "Add ControllerMPPI sampling-based planner to newton.vehicles"
```

---

### Task 2: `cone.usda` asset

**Files:**
- Create: `newton/examples/assets/cone.usda` (generated, committed)
- Scratch: generation script in the session scratchpad (not committed)

**Interfaces:**
- Produces: a triangles-only `UsdGeom.Mesh` at prim path `/cone` (defaultPrim `cone`), Z-up, metersPerUnit 1, ~0.20 m tall orange sports cone: 0.15 m square base plate (0.012 m thick) + truncated cone shell (base r 0.055 m, top r 0.014 m), `displayColor` (1.0, 0.35, 0.05).

- [ ] **Step 1: Write the generation script** to the scratchpad and run it with `uv run python <script>`:

```python
import numpy as np

SEGS = 24
R_BASE, R_TOP = 0.055, 0.014
PLATE_HALF, PLATE_H = 0.075, 0.012
H_TOTAL = 0.20

pts, tris = [], []

def add(p):
    pts.append(tuple(round(c, 5) for c in p))
    return len(pts) - 1

# base plate box (8 verts, 12 tris)
for z in (0.0, PLATE_H):
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        add((sx * PLATE_HALF, sy * PLATE_HALF, z))
quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
for a, b, c, d in quads:
    tris += [(a, b, c), (a, c, d)]

# frustum shell + top cap
ang = np.linspace(0.0, 2.0 * np.pi, SEGS, endpoint=False)
bot = [add((R_BASE * np.cos(t), R_BASE * np.sin(t), PLATE_H)) for t in ang]
top = [add((R_TOP * np.cos(t), R_TOP * np.sin(t), H_TOTAL)) for t in ang]
apex = add((0.0, 0.0, H_TOTAL))
for i in range(SEGS):
    j = (i + 1) % SEGS
    tris += [(bot[i], bot[j], top[j]), (bot[i], top[j], top[i])]
    tris += [(top[i], top[j], apex)]

counts = ", ".join("3" for _ in tris)
indices = ", ".join(str(i) for t in tris for i in t)
points = ", ".join(f"({p[0]}, {p[1]}, {p[2]})" for p in pts)

usda = f'''#usda 1.0
(
    defaultPrim = "cone"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "cone"
{{
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{indices}]
    point3f[] points = [{points}]
    color3f[] primvars:displayColor = [(1.0, 0.35, 0.05)]
    uniform token subdivisionScheme = "none"
}}
'''
with open("newton/examples/assets/cone.usda", "w") as f:
    f.write(usda)
print(f"{len(pts)} points, {len(tris)} tris")
```

Expected output: `81 points, 84 tris`.

- [ ] **Step 2: Verify the asset loads as a render mesh**

Run (from repo root):
```bash
uv run python - <<'EOF'
from pxr import Usd, UsdGeom
import numpy as np
import newton, newton.examples
stage = Usd.Stage.Open(newton.examples.get_asset("cone.usda"))
mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cone"))
v = np.array(mesh.GetPointsAttr().Get()); i = np.array(mesh.GetFaceVertexIndicesAttr().Get())
m = newton.Mesh(v, i); m.finalize()
print("ok", v.shape, i.shape, float(v[:, 2].max()))
EOF
```
Expected: `ok (81, 3) (252,) 0.2`.

- [ ] **Step 3: Commit**

```bash
git add newton/examples/assets/cone.usda
git commit -m "Add cone.usda prop asset for track visualization"
```

---

### Task 3: Example skeleton — replicated worlds, track, cones, camera

**Files:**
- Create: `newton/examples/vehicles/example_vehicle_mppi_track.py`

**Interfaces:**
- Consumes: rc_car build recipe from `example_vehicle_rc_car._build` (copied, not imported — examples are standalone); track_gen APIs per "Verified external facts".
- Produces: `Example` class runnable headless (`--viewer null`) with the hero driving a constant open-loop command; attributes used by Task 4: `self.model`, `self.vehicles`, `self.solver`, `self.state_0/1`, `self.control`, `self.contacts`, `self.num_worlds`, `self.chassis` (`wp.array[wp.int32]` per-world chassis body ids), `self.track`, `self.tracker`, `self.checker`, `self.car_pos` (`wp.array[wp.vec2f]` [E], bound to tracker + checker), `self.car_yaw`, `self.bodies_per_world`, `self.dofs_per_world`, `self.vel_dofs_per_world`.

- [ ] **Step 1: Write the skeleton.** Full file (Task 4 extends it; the `step()` here is temporary):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle MPPI Track
#
# Races the rc_car fixture around a procedurally generated closed track
# using an MPPI controller whose rollout model is the simulator itself:
# num-samples replicated worlds are collocated at the origin (cross-world
# collision is filtered), world 0 is the rendered hero executing the
# optimized command and worlds 1..K-1 evaluate noise-perturbed command
# sequences every frame. Track generation, out-of-bounds tests, and
# checkpoint progress come from the track_gen package.
#
# Command: python -m newton.examples vehicle_mppi_track --viewer gl
#
###########################################################################

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.vehicles as nv

try:
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    import track_gen
    from track_gen.checkpoints import CheckpointSampler
    from track_gen.collision import CollisionChecker
    from track_gen.progress import ProgressTracker
    from track_gen.props import PropSampler
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "This example requires the track_gen package: pip install -e <path-to-track_gen>"
    ) from exc

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))

TRACK_HALF_WIDTH = 0.5  # [m]
TRACK_SCALE = 10.0      # calibrated in Task 5 for a ~20 m footprint
TRACK_N_MAX = 512
CONE_SPACING = 0.5      # [m]
CHECKPOINT_SPACING = 1.0  # [m]
CAR_HALF_EXTENTS = (0.29, 0.15)  # oriented OOB box [m] (Slash-class rc car)
MAX_TRACK_ATTEMPTS = 32


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.kernel
def _gather_car_pose(
    body_q: wp.array[wp.transform],
    chassis: wp.array[wp.int32],
    pos: wp.array[wp.vec2f],
    yaw: wp.array[float],
):
    e = wp.tid()
    tf = body_q[chassis[e]]
    p = wp.transform_get_translation(tf)
    pos[e] = wp.vec2f(p[0], p[1])
    yaw[e] = _quat_yaw(wp.transform_get_rotation(tf))


def _build_model(num_worlds):
    manifest = json.loads((_ASSET_DIR / "manifest.json").read_text())
    asset = next(a for a in manifest["assets"] if a["name"] == "rc_car")
    rd = asset["reference_dimensions"]

    car = newton.ModelBuilder()
    nv.register_vehicle_attributes(car)
    newton.solvers.SolverMuJoCo.register_custom_attributes(car)
    car.add_usd(str(_ASSET_DIR / asset["file"]))
    nv.configure_wheel_axle_joints(car, axle_joint_labels=asset["axle_joint_labels"])

    joint_by_label = {label: i for i, label in enumerate(car.joint_label)}
    shape_by_label = {label: i for i, label in enumerate(car.shape_label)}
    nv.set_vehicle(
        car,
        0,
        drive_mode=int(nv.DriveMode.ACKERMANN),
        wheelbase=rd["wheelbase_m"],
        track_width=rd["track_width_m"],
        steer_limit=math.radians(rd["steering_limit_deg"]),
    )
    steering = asset["steering_joint_labels"]
    for wheel_id, (body_label, shape_label) in enumerate(
        zip(asset["wheel_body_labels"], asset["wheel_shape_labels"], strict=True)
    ):
        name = body_label.split("/")[-1]
        front = "front" in name
        left = "left" in name
        steer_joint = joint_by_label[steering[0 if left else 1]] if front else -1
        nv.add_wheel(
            car,
            shape=shape_by_label[shape_label],
            vehicle_id=0,
            wheel_id=wheel_id,
            radius=rd["wheel_radius_m"],
            width=rd["wheel_width_m"],
            driven=True,
            steerable=front,
            side=(-1 if left else 1),
            axle_row=(0 if front else 1),
            steer_joint=steer_joint,
        )

    scene = newton.ModelBuilder()
    nv.register_vehicle_attributes(scene)
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(car, num_worlds, spacing=(0.0, 0.0, 0.0))
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    scene.add_ground_plane(cfg=terrain_cfg)
    model = scene.finalize()

    # hide every shape outside world 0 so the viewer draws a single car
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    flags[worlds >= 1] &= ~int(newton.ShapeFlags.VISIBLE)
    model.shape_flags.assign(flags)

    joint_type = model.joint_type.numpy()
    joint_child = model.joint_child.numpy()
    free_children = joint_child[joint_type == int(newton.JointType.FREE)]
    if len(free_children) != num_worlds:
        raise RuntimeError(f"expected {num_worlds} free joints, found {len(free_children)}")
    return model, np.sort(free_children).astype(np.int32)


def _generate_track(num_envs, seed, device):
    """Generates one bezier track shared by all envs; retries invalid seeds."""
    for attempt in range(MAX_TRACK_ATTEMPTS):
        seeds = wp.array(
            np.full(num_envs, seed + attempt, dtype=np.int32), dtype=wp.int32, device=device
        )
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=str(device))
        config = TrackGenConfig(
            num_envs=num_envs,
            generator="bezier",
            scale=TRACK_SCALE,
            half_width=TRACK_HALF_WIDTH,
            N_max=TRACK_N_MAX,
            device=str(device),
        )
        track = TrackGenerator(config, rng).generate()
        if bool(track.valid.numpy()[0]):
            return track, seed + attempt
    raise RuntimeError(f"no valid track after {MAX_TRACK_ATTEMPTS} attempts from seed {seed}")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self._test_mode = getattr(args, "test", False)

        self.num_worlds = 32 if self._test_mode else args.num_samples
        self.model, chassis_ids = _build_model(self.num_worlds)
        self.chassis = wp.array(chassis_ids, dtype=wp.int32, device=self.model.device)
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.dofs_per_world = self.model.joint_coord_count // self.num_worlds
        self.vel_dofs_per_world = self.model.joint_dof_count // self.num_worlds

        self.vehicles = nv.WheeledVehicles(
            self.model,
            config=nv.WheeledConfig(
                max_wheel_speed=315.0,
                motor_max_torque=1.0,
                angular_damping=0.0005,
                friction=2.0,
                longitudinal_stiffness=20.0,
                lateral_stiffness=40.0,
            ),
        )
        self.vehicles.configure_solver_contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            njmax=64 * self.num_worlds,
            nconmax=32 * self.num_worlds,
        )
        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # --- track, checkpoints, progress, collision --------------------
        self.track, self.track_seed = _generate_track(
            self.num_worlds, args.track_seed, self.model.device
        )
        self._spawn_on_track()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.car_pos = wp.zeros(self.num_worlds, dtype=wp.vec2f, device=self.model.device)
        self.car_yaw = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_half_extents = wp.array(
            np.tile(np.array(CAR_HALF_EXTENTS, dtype=np.float32), (self.num_worlds, 1)),
            dtype=wp.vec2f,
            device=self.model.device,
        )
        self.sampler = CheckpointSampler(self.track, spacing=CHECKPOINT_SPACING)
        self.checkpoints = self.sampler.sample()
        self.tracker = ProgressTracker(self.checkpoints, position=self.car_pos)
        self.checker = CollisionChecker(self.track, max_boxes=1, method="segments")
        self.checker.bind_inputs(
            position=self.car_pos, yaw=self.car_yaw, half_extents=self.car_half_extents
        )

        self._init_track_render()
        self.follow_camera = True
        self.viewer.set_model(self.model)
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

    # --- track helpers ---------------------------------------------------

    def _env0_polyline(self, flat_vec2):
        count = int(self.track.count.numpy()[0])
        return flat_vec2.numpy()[:count]

    def _spawn_on_track(self):
        center = self._env0_polyline(self.track.center)
        p0, p1 = center[0], center[1]
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        joint_q = self.model.joint_q.numpy()
        joint_type = self.model.joint_type.numpy()
        q_start = self.model.joint_q_start.numpy()
        for j in np.flatnonzero(joint_type == int(newton.JointType.FREE)):
            qs = int(q_start[j])
            joint_q[qs + 0] = p0[0]
            joint_q[qs + 1] = p0[1]
            # keep the authored spawn height joint_q[qs + 2]
            joint_q[qs + 3 : qs + 7] = [q[0], q[1], q[2], q[3]]
        self.model.joint_q.assign(joint_q)

    def _init_track_render(self):
        device = self.model.device
        inner = self._env0_polyline(self.track.inner)
        outer = self._env0_polyline(self.track.outer)

        def _loop_lines(poly, z):
            pts = np.column_stack([poly, np.full(len(poly), z, dtype=np.float32)])
            starts = pts
            ends = np.roll(pts, -1, axis=0)
            return (
                wp.array(starts, dtype=wp.vec3, device=device),
                wp.array(ends, dtype=wp.vec3, device=device),
            )

        self._boundary_lines = [_loop_lines(inner, 0.01), _loop_lines(outer, 0.01)]

        # cone poses along both boundaries (env 0 only)
        xforms = []
        for boundary in ("inner", "outer"):
            props = PropSampler(
                self.track, spacing=CONE_SPACING, boundary=boundary, mode="points"
            ).sample()
            n = int(props.count.numpy()[0])
            pos = props.position.numpy()[:n]
            yaw = props.yaw.numpy()[:n]
            for p, y in zip(pos, yaw, strict=True):
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(y))
                xforms.append(wp.transform(wp.vec3(float(p[0]), float(p[1]), 0.0), q))
        self._cone_xforms = wp.array(xforms, dtype=wp.transform, device=device)
        self._cone_mesh = self._load_cone_mesh()

    @staticmethod
    def _load_cone_mesh():
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(newton.examples.get_asset("cone.usda"))
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cone"))
        vertices = np.array(usd_mesh.GetPointsAttr().Get())
        indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        mesh = newton.Mesh(vertices, indices)
        mesh.finalize()
        return mesh

    # --- per-frame -------------------------------------------------------

    def step(self):
        # temporary open-loop drive; replaced by MPPI in the next commit
        self.vehicles.set_commands(drive=0.3, steer=0.2, brake=0.0)
        for _ in range(self.sim_substeps):
            self._substep(self.sim_dt)
        wp.launch(
            _gather_car_pose,
            dim=self.num_worlds,
            inputs=[self.state_0.body_q, self.chassis, self.car_pos, self.car_yaw],
            device=self.model.device,
        )
        self.tracker.update()
        self.checker.query()
        self.sim_time += self.frame_dt

    def _substep(self, dt):
        self.state_0.clear_forces()
        self.vehicles.update_controls(self.control)
        self.model.collide(self.state_0, self.contacts)
        self.vehicles.apply(self.state_0, self.contacts, dt)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.vehicles.latch_loads(self.contacts)
        self.state_0, self.state_1 = self.state_1, self.state_0

    # --- rendering -------------------------------------------------------

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        tf = self.state_0.body_q.numpy()[int(self.chassis.numpy()[0])]
        x, y, z, w = tf[3], tf[4], tf[5], tf[6]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = tf[:3] - 2.8 * forward + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.viewer.set_camera(
            pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])),
            pitch=-18.0,
            yaw=math.degrees(yaw),
        )

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes(
            "/track/cones",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self._cone_xforms,
            geo_src=self._cone_mesh,
        )
        for i, (starts, ends) in enumerate(self._boundary_lines):
            self.viewer.log_lines(f"/track/boundary_{i}", starts, ends, (0.35, 0.35, 0.4))
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("non-finite body poses")

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("non-finite final poses")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-samples", type=int, default=1024,
                            help="MPPI samples K (= simulated worlds; sample 0 is the hero)")
        parser.add_argument("--horizon", type=int, default=32,
                            help="MPPI planning horizon in control steps")
        parser.add_argument("--rollout-substeps", type=int, default=4,
                            help="solver substeps per rollout control step")
        parser.add_argument("--track-seed", type=int, default=0,
                            help="base seed for track generation")
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
```

- [ ] **Step 2: Smoke-run headless**

Run: `uv run -m newton.examples vehicle_mppi_track --viewer null --num-frames 30 --num-samples 8`
Expected: clean exit; no exceptions. If `log_shapes`/`log_lines` are unavailable on the null viewer, guard the calls with `hasattr` checks (verify against `newton/_src/viewer/viewer_null.py`).

- [ ] **Step 3: Visual check** (GL, brief): `uv run -m newton.examples vehicle_mppi_track --viewer gl --num-samples 8 --num-frames 200` — expect ONE car visible on a cone-lined closed track, driving an open-loop arc; camera follows.

- [ ] **Step 4: Commit**

```bash
git add newton/examples/vehicles/example_vehicle_mppi_track.py
git commit -m "Add MPPI track example skeleton (worlds, track, cones)"
```

---

### Task 4: MPPI integration — snapshot/broadcast/restore, rollout, cost, graph, UI

**Files:**
- Modify: `newton/examples/vehicles/example_vehicle_mppi_track.py`

**Interfaces:**
- Consumes: `nv.ControllerMPPI` (Task 1 signature), Task 3 attributes.
- Produces: complete example; `test_final` requires hero progress ≥ 2 checkpoints.

- [ ] **Step 1: Add kernels** (module level, after `_gather_car_pose`):

```python
@wp.kernel
def _broadcast_slice_tf(snap: wp.array[wp.transform], n_per: int, dst: wp.array[wp.transform]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_sv(
    snap: wp.array[wp.spatial_vector], n_per: int, dst: wp.array[wp.spatial_vector]
):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_f32(snap: wp.array[float], n_per: int, dst: wp.array[float]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_env_i32(snap: wp.array[wp.int32], dst: wp.array[wp.int32]):
    dst[wp.tid()] = snap[0]


@wp.kernel
def _restore_env0_i32(snap: wp.array[wp.int32], dst: wp.array[wp.int32]):
    dst[0] = snap[0]


@wp.kernel
def _snap_prev_pos(src: wp.array2d[float], snap: wp.array2d[float]):
    snap[0, 0] = src[0, 0]
    snap[0, 1] = src[0, 1]


@wp.kernel
def _broadcast_prev_pos(snap: wp.array2d[float], dst: wp.array2d[float]):
    e = wp.tid()
    dst[e, 0] = snap[0, 0]
    dst[e, 1] = snap[0, 1]


@wp.kernel
def _restore_env0_prev_pos(snap: wp.array2d[float], dst: wp.array2d[float]):
    dst[0, 0] = snap[0, 0]
    dst[0, 1] = snap[0, 1]


@wp.kernel
def _apply_sample_commands(
    samples: wp.array3d[float],
    t: int,
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    drive[v] = samples[v, t, 0]
    steer[v] = samples[v, t, 1]
    brake[v] = 0.0


@wp.kernel
def _apply_nominal_command(
    nominal: wp.array2d[float],
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    drive[v] = nominal[0, 0]
    steer[v] = nominal[0, 1]
    brake[v] = 0.0


@wp.kernel
def _zero_plan_buffers(costs: wp.array[float], dead: wp.array[wp.int32]):
    e = wp.tid()
    costs[e] = 0.0
    dead[e] = 0


@wp.kernel
def _accumulate_cost(
    dist_to_next: wp.array[float],
    passed: wp.array[wp.int32],
    oob: wp.array[wp.int32],
    samples: wp.array3d[float],
    t: int,
    params: wp.array[float],  # [w_progress, w_pass, w_steer, kill_penalty]
    dist_prev: wp.array[float],
    dead: wp.array[wp.int32],
    costs: wp.array[float],
):
    e = wp.tid()
    if dead[e] == 1:
        return
    if oob[e] == 1:
        dead[e] = 1
        costs[e] = costs[e] + params[3]
        return
    d = dist_to_next[e]
    progress = dist_prev[e] - d
    if progress != progress or progress > 1.0e3 or progress < -1.0e3:  # NaN/garbage guard
        progress = 0.0
    steer = samples[e, t, 1]
    costs[e] = (
        costs[e]
        - params[0] * progress
        - params[1] * float(passed[e])
        + params[2] * steer * steer
    )
    dist_prev[e] = d


@wp.kernel
def _record_ribbon(
    body_q: wp.array[wp.transform],
    chassis0: int,
    t: int,
    ribbon: wp.array[wp.vec3],
):
    p = wp.transform_get_translation(body_q[chassis0])
    ribbon[t] = wp.vec3(p[0], p[1], p[2] + 0.05)
```

- [ ] **Step 2: Extend `__init__`** (after the collision-checker setup, replacing nothing):

```python
        # --- MPPI planner and plan-cycle buffers -------------------------
        horizon = 8 if self._test_mode else args.horizon
        self.rollout_substeps = 2 if self._test_mode else args.rollout_substeps
        total_substeps = horizon * self.rollout_substeps + self.sim_substeps
        if total_substeps % 2 != 0:
            raise ValueError(
                "horizon * rollout_substeps + 8 must be even so state buffers "
                "return to their starting roles each frame (CUDA graph replay)"
            )
        self.planner = nv.ControllerMPPI(
            config=nv.ControllerMPPI.Config(
                num_samples=self.num_worlds,
                horizon=horizon,
                dim=2,
                sigma=(0.35, 0.45),
                temperature=0.05,
                beta=0.7,
                bounds_lo=(-0.3, -1.0),  # limited reverse, full steering
                bounds_hi=(1.0, 1.0),
            ),
            device=self.model.device,
        )
        device = self.model.device
        E = self.num_worlds
        self.costs = wp.zeros(E, dtype=wp.float32, device=device)
        self.dead = wp.zeros(E, dtype=wp.int32, device=device)
        self.dist_prev = wp.zeros(E, dtype=wp.float32, device=device)
        # [w_progress, w_pass, w_steer, kill_penalty]
        self.cost_params = wp.array(
            [30.0, 30.0, 0.05, 200.0], dtype=wp.float32, device=device
        )
        self.ribbon = wp.zeros(horizon, dtype=wp.vec3, device=device)

        # hero-slice snapshots (world 0 leads every per-world array)
        self.snap_joint_q = wp.zeros(self.dofs_per_world, dtype=wp.float32, device=device)
        self.snap_joint_qd = wp.zeros(self.vel_dofs_per_world, dtype=wp.float32, device=device)
        self.snap_body_q = wp.zeros(self.bodies_per_world, dtype=wp.transform, device=device)
        self.snap_body_qd = wp.zeros(
            self.bodies_per_world, dtype=wp.spatial_vector, device=device
        )
        wheels_per_world = self.vehicles.dynamics.omega.shape[0] // E
        self.wheels_per_world = wheels_per_world
        self.snap_omega = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_long = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_lat = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_fz = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_prev_pos = wp.zeros((1, 2), dtype=wp.float32, device=device)
        self.snap_next = wp.zeros(1, dtype=wp.int32, device=device)
        self.snap_laps = wp.zeros(1, dtype=wp.int32, device=device)
        self.snap_progress = wp.zeros(1, dtype=wp.int32, device=device)

        self.graph = None
        self._telemetry = {"speed": 0.0, "laps": 0, "progress": 0, "dist": 0.0,
                           "alive": 1.0, "best_cost": 0.0, "mean_cost": 0.0,
                           "drive": 0.0, "steer": 0.0, "hero_oob": 0}
        self.ui_temperature = self.planner.config.temperature
        self.ui_sigma_drive, self.ui_sigma_steer = self.planner.config.sigma
```

- [ ] **Step 3: Replace the temporary `step()`** with the plan-execute cycle and add helpers:

```python
    def _snapshot_hero(self):
        wp.copy(self.snap_joint_q, self.state_0.joint_q, count=self.dofs_per_world)
        wp.copy(self.snap_joint_qd, self.state_0.joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.snap_body_q, self.state_0.body_q, count=self.bodies_per_world)
        wp.copy(self.snap_body_qd, self.state_0.body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(self.snap_omega, dyn.omega, count=self.wheels_per_world)
        wp.copy(self.snap_trans_long, dyn.trans_long, count=self.wheels_per_world)
        wp.copy(self.snap_trans_lat, dyn.trans_lat, count=self.wheels_per_world)
        wp.copy(self.snap_fz, patch.fz, count=self.wheels_per_world)
        dev = self.model.device
        wp.launch(_snap_prev_pos, dim=1,
                  inputs=[self.tracker._prev_pos, self.snap_prev_pos], device=dev)
        wp.copy(self.snap_next, self.tracker._next, count=1)
        wp.copy(self.snap_laps, self.tracker._laps, count=1)
        wp.copy(self.snap_progress, self.tracker._progress, count=1)

    def _broadcast_hero(self):
        dev = self.model.device
        E = self.num_worlds
        wp.launch(_broadcast_slice_f32, dim=(E, self.dofs_per_world),
                  inputs=[self.snap_joint_q, self.dofs_per_world, self.state_0.joint_q],
                  device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, self.vel_dofs_per_world),
                  inputs=[self.snap_joint_qd, self.vel_dofs_per_world, self.state_0.joint_qd],
                  device=dev)
        wp.launch(_broadcast_slice_tf, dim=(E, self.bodies_per_world),
                  inputs=[self.snap_body_q, self.bodies_per_world, self.state_0.body_q],
                  device=dev)
        wp.launch(_broadcast_slice_sv, dim=(E, self.bodies_per_world),
                  inputs=[self.snap_body_qd, self.bodies_per_world, self.state_0.body_qd],
                  device=dev)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        n = self.wheels_per_world
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_omega, n, dyn.omega], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_long, n, dyn.trans_long], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_lat, n, dyn.trans_lat], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_fz, n, patch.fz], device=dev)
        wp.launch(_broadcast_prev_pos, dim=E,
                  inputs=[self.snap_prev_pos, self.tracker._prev_pos], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_next, self.tracker._next], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_laps, self.tracker._laps], device=dev)
        wp.launch(_broadcast_env_i32, dim=E, inputs=[self.snap_progress, self.tracker._progress], device=dev)

    def _restore_hero(self):
        dev = self.model.device
        wp.copy(self.state_0.joint_q, self.snap_joint_q, count=self.dofs_per_world)
        wp.copy(self.state_0.joint_qd, self.snap_joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.state_0.body_q, self.snap_body_q, count=self.bodies_per_world)
        wp.copy(self.state_0.body_qd, self.snap_body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(dyn.omega, self.snap_omega, count=self.wheels_per_world)
        wp.copy(dyn.trans_long, self.snap_trans_long, count=self.wheels_per_world)
        wp.copy(dyn.trans_lat, self.snap_trans_lat, count=self.wheels_per_world)
        wp.copy(patch.fz, self.snap_fz, count=self.wheels_per_world)
        wp.launch(_restore_env0_prev_pos, dim=1,
                  inputs=[self.snap_prev_pos, self.tracker._prev_pos], device=dev)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_next, self.tracker._next], device=dev)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_laps, self.tracker._laps], device=dev)
        wp.launch(_restore_env0_i32, dim=1, inputs=[self.snap_progress, self.tracker._progress], device=dev)

    def _gather_and_track(self):
        wp.launch(_gather_car_pose, dim=self.num_worlds,
                  inputs=[self.state_0.body_q, self.chassis, self.car_pos, self.car_yaw],
                  device=self.model.device)
        self.tracker.update()
        self.checker.query()

    def _plan_and_execute(self):
        cmd = self.vehicles.commands
        dev = self.model.device
        horizon = self.planner.config.horizon
        rollout_dt = self.frame_dt / self.rollout_substeps

        self._snapshot_hero()
        self._broadcast_hero()
        self.planner.sample()
        wp.launch(_zero_plan_buffers, dim=self.num_worlds,
                  inputs=[self.costs, self.dead], device=dev)
        self._gather_and_track()
        wp.copy(self.dist_prev, self.tracker.events.dist_to_next, count=self.num_worlds)

        for t in range(horizon):
            wp.launch(_apply_sample_commands, dim=self.num_worlds,
                      inputs=[self.planner.samples, t, cmd.drive, cmd.steer, cmd.brake],
                      device=dev)
            for _ in range(self.rollout_substeps):
                self._substep(rollout_dt)
            self._gather_and_track()
            wp.launch(_accumulate_cost, dim=self.num_worlds,
                      inputs=[self.tracker.events.dist_to_next, self.tracker.events.passed,
                              self.checker.contact.oob, self.planner.samples, t,
                              self.cost_params, self.dist_prev, self.dead, self.costs],
                      device=dev)
            wp.launch(_record_ribbon, dim=1,
                      inputs=[self.state_0.body_q, int(self.chassis.numpy()[0]), t, self.ribbon],
                      device=dev)

        self.planner.update(self.costs)
        self._restore_hero()
        wp.launch(_apply_nominal_command, dim=self.num_worlds,
                  inputs=[self.planner.nominal, cmd.drive, cmd.steer, cmd.brake],
                  device=dev)
        for _ in range(self.sim_substeps):
            self._substep(self.sim_dt)
        self.planner.shift()
        self._gather_and_track()

    def step(self):
        if self.graph is None and self.model.device.is_cuda:
            track_gen.set_capturing(True)
            self._chassis0 = int(self.chassis.numpy()[0])  # host read before capture
            with wp.ScopedCapture() as capture:
                self._plan_and_execute()
            self.graph = capture.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._plan_and_execute()
        self.sim_time += self.frame_dt
        self._update_telemetry()
```

Note the `int(self.chassis.numpy()[0])` host read inside `_plan_and_execute` (ribbon kernel input) — hoist it: compute `self._chassis0` once in `__init__` right after `self.chassis` is created, and use `self._chassis0` in the `_record_ribbon` launch. No `.numpy()` calls may occur inside the captured function.

- [ ] **Step 4: Telemetry, UI, ribbon render, test hooks** (replace the Task 3 stubs):

```python
    def _update_telemetry(self):
        t = self._telemetry
        qd = self.state_0.body_qd.numpy()[self._chassis0]
        t["speed"] = float(np.linalg.norm(qd[3:5]))  # check spatial layout: linear part
        events = self.tracker.events
        t["laps"] = int(events.laps.numpy()[0])
        t["progress"] = int(events.progress.numpy()[0])
        t["dist"] = float(events.dist_to_next.numpy()[0])
        t["hero_oob"] = int(self.checker.contact.oob.numpy()[0])
        dead = self.dead.numpy()
        costs = self.costs.numpy()
        t["alive"] = 1.0 - float(dead.mean())
        t["best_cost"] = float(costs.min())
        t["mean_cost"] = float(costs.mean())
        nominal = self.planner.nominal.numpy()
        t["drive"], t["steer"] = float(nominal[0, 0]), float(nominal[0, 1])
        self._nominal_plan = nominal

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        ui.separator()
        ui.text("Controller output")
        t = self._telemetry
        ui.text(f"Drive: {t['drive']:+.2f}   Steer: {t['steer']:+.2f}")
        ui.plot_lines("drive plan", np.ascontiguousarray(self._nominal_plan[:, 0]))
        ui.plot_lines("steer plan", np.ascontiguousarray(self._nominal_plan[:, 1]))
        ui.separator()
        ui.text("Race")
        ui.text(f"Speed: {t['speed']:.2f} m/s")
        ui.text(f"Laps: {t['laps']}   Checkpoints: {t['progress']}")
        ui.text(f"Dist to next: {t['dist']:.2f} m   OOB: {t['hero_oob']}")
        ui.separator()
        ui.text("Planner")
        ui.text(f"Alive: {100.0 * t['alive']:.0f}%")
        ui.text(f"Cost best/mean: {t['best_cost']:.1f} / {t['mean_cost']:.1f}")
        changed_t, self.ui_temperature = ui.slider_float(
            "Temperature", self.ui_temperature, 0.005, 0.5)
        if changed_t:
            self.planner.set_temperature(self.ui_temperature)
        changed_d, self.ui_sigma_drive = ui.slider_float(
            "Sigma drive", self.ui_sigma_drive, 0.05, 1.0)
        changed_s, self.ui_sigma_steer = ui.slider_float(
            "Sigma steer", self.ui_sigma_steer, 0.05, 1.0)
        if changed_d or changed_s:
            self.planner.sigma.assign(
                np.array([self.ui_sigma_drive, self.ui_sigma_steer], dtype=np.float32))
```

In `render()`, after the boundary lines, add the plan ribbon:

```python
        horizon = self.planner.config.horizon
        self.viewer.log_lines(
            "/mppi/plan", self.ribbon[: horizon - 1], self.ribbon[1:horizon], (0.1, 0.9, 0.3)
        )
```

Replace the test hooks:

```python
    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()[: self.bodies_per_world]).all():
            raise ValueError("non-finite hero poses")
        if not np.isfinite(self.costs.numpy()).all():
            raise ValueError("non-finite MPPI costs")

    def test_final(self):
        hero_q = self.state_0.body_q.numpy()[self._chassis0]
        if not np.isfinite(hero_q).all():
            raise ValueError("non-finite hero pose")
        progress = int(self.tracker.events.progress.numpy()[0])
        if progress < 2:
            raise ValueError(f"hero passed only {progress} checkpoints")
```

- [ ] **Step 5: Verify telemetry speed indexing.** `body_qd` is `wp.spatial_vector`; check Newton's convention (angular-first vs linear-first) in `newton/_src/sim/model.py` docs for `body_qd` and fix the `qd[...]` slice accordingly (rc_car example uses `qd[:2]` for planar speed — mirror whatever it does).

- [ ] **Step 6: Headless run**

Run: `uv run -m newton.examples vehicle_mppi_track --viewer null --num-frames 120 --num-samples 128 --horizon 16`
Expected: completes; print/inspect telemetry progress > 0 (add a temporary print or run under `--test`; remove prints before commit).

- [ ] **Step 7: Test-mode run** (exercises `test_post_step`/`test_final`):

Run: `uv run -m newton.examples vehicle_mppi_track --viewer null --test --num-frames 240`
Expected: passes `test_final` (progress ≥ 2). If progress is too slow, raise `--num-frames` for test defaults or lower `CHECKPOINT_SPACING`; do not weaken the assertion below 2.

- [ ] **Step 8: Check the example-test harness dependency skip.** Find how `newton/tests` wraps examples (search `test_examples`); register/verify this example is skipped when `track_gen` is missing (follow the existing optional-dependency pattern, e.g. torch examples). Run: `uv run --extra dev -m newton.tests -k vehicle_mppi` and confirm it runs (or cleanly skips on machines without track_gen).

- [ ] **Step 9: Commit**

```bash
git add newton/examples/vehicles/example_vehicle_mppi_track.py newton/tests/
git commit -m "Wire MPPI planner into vehicle track example"
```

---

### Task 5: Calibration and validation

**Files:**
- Modify: `newton/examples/vehicles/example_vehicle_mppi_track.py` (constants only)
- Scratch: calibration scripts in the scratchpad (not committed)

- [ ] **Step 1: Track scale calibration.** Scratch script: generate E=64 tracks (distinct seeds) at `half_width=0.5` sweeping `scale` in {8, 10, 12}; report per-env bounding-box extents (`track.center` min/max per env) and `track.length`. Pick the scale whose mean footprint ≈ 20 m; verify `max(count) < N_MAX` with margin (else raise `TRACK_N_MAX`). Update `TRACK_SCALE`/`TRACK_N_MAX` in the example.

- [ ] **Step 2: Validity yield.** Scratch script: with the chosen config, generate E=1024 tracks with distinct seeds; report `valid.mean()`. Required: ≥ 0.999. If lower, raise `relax_iters` (e.g. 200) in `TrackGenConfig` inside `_generate_track` and re-measure. Record the measured yield in the commit message.

- [ ] **Step 3: Racing quality.** Run `uv run -m newton.examples vehicle_mppi_track --viewer gl` (defaults: 1024 samples, horizon 32). Watch for: hero completes laps without leaving the band; if it cuts corners into OOB, raise `kill_penalty`/`w_progress`; if it crawls, raise `sigma` drive or lower `temperature`; if it oscillates, raise `w_steer` or `beta`. Iterate on the defaults in the file (cost_params, sigma, temperature) until the car laps reliably from `--track-seed 0..4`. Record chosen values.

- [ ] **Step 4: Perf sanity.** Report wall-clock FPS at defaults on the 4090 (informational; no hard requirement). If < 2 FPS, lower default `--horizon` to 24 or `--rollout-substeps` to 2 and re-verify racing quality.

- [ ] **Step 5: Full test suite**

Run: `uv run --extra dev -m newton.tests -k "vehicles or vehicle"`
Expected: all pass, including existing vehicle tests (no regressions from the `newton/vehicles.py` export change).

- [ ] **Step 6: Commit calibrated constants**

```bash
git add newton/examples/vehicles/example_vehicle_mppi_track.py
git commit -m "Calibrate MPPI track example defaults"
```

---

### Task 6: Registration, changelog, screenshot, final review

**Files:**
- Modify: `newton/examples/README.md` (register example)
- Modify: `CHANGELOG.md` (two Added entries)
- Create: screenshot jpg (follow the path convention used by existing example entries in `newton/examples/README.md`)

- [ ] **Step 1: CHANGELOG.** Insert at random positions within `[Unreleased] > Added`:
  - `Add `ControllerMPPI` sampling-based MPC planner to `newton.vehicles``
  - `Add `vehicle_mppi_track` example racing the RC car on procedurally generated tracks (requires the `track_gen` package)`

- [ ] **Step 2: README registration.** Add the example row following the existing format, command `python -m newton.examples vehicle_mppi_track`, with a 320x320 jpg screenshot. Capture: run with `--viewer gl`, grab a frame showing the car mid-track with cones (check ViewerGL for a screenshot/save-frame facility; otherwise capture the window and crop to 320x320).

- [ ] **Step 3: Lint + full tests**

Run: `uvx pre-commit run -a` then `uv run --extra dev -m newton.tests`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add newton/examples/README.md CHANGELOG.md <screenshot path>
git commit -m "Register MPPI track example in README and changelog"
```

- [ ] **Step 5: Request code review** (superpowers:requesting-code-review) before merging/PR.

---

## Self-review notes

- Spec coverage: planner (Task 1), cone asset (Task 2), example skeleton + track + viz (Task 3), MPPI cycle + UI + ribbon + tests (Task 4), 20 m scale + 100% validity + racing quality (Task 5), README/CHANGELOG/screenshot (Task 6). Follow camera and hidden particle worlds are in Task 3.
- Known risk points called out inline: null-viewer `log_shapes` availability (T3 S2), `body_qd` spatial layout (T4 S5), example-test dependency skip mechanism (T4 S8), imgui `plot_lines` numpy signature (verify at T4 S6; fall back to `ui.text` sparklines if unavailable).
- Type consistency: planner attribute names (`samples`, `nominal`, `sigma`, `set_temperature`) match between Task 1 and Task 4; `tracker._prev_pos/_next/_laps/_progress` names verified against track_gen source.
