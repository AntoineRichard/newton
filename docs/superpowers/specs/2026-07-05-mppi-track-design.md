# MPPI RC-car track-racing example — design

Date: 2026-07-05
Status: approved (approach A; user: "Go with A", one model/one solver)

## Goal

An MPPI (Model Predictive Path Integral) controller that races the Newton RC
car around a procedurally generated closed track from the sibling `track_gen`
package. The simulator itself is the rollout model: N particle cars sample
noisy command sequences in parallel worlds while one hero car executes the
optimized command. Deliverables:

1. A reusable, sim-agnostic MPPI planner: `newton.vehicles.ControllerMPPI`.
2. An example: `newton/examples/vehicles/example_vehicle_mppi_track.py`.
3. A cone prop asset `newton/examples/assets/cone.usda` for track visualization.
4. Unit tests for the planner; example `test_final`/`test_post_step` hooks.

## Architecture (approach A — one model, N+1 worlds)

- One `ModelBuilder`: rc_car template (built as in
  `example_vehicle_rc_car._build`, via `newton.vehicles` + manifest), then
  `scene.replicate(car, world_count=N+1, spacing=(0, 0, 0))` with a global
  ground plane. All worlds are collocated at the origin; cross-world collision
  is filtered by `shape_world`, so overlapped cars never interact.
- World 0 = hero (rendered). Worlds 1..N = MPPI particles (shapes hidden so
  the viewer draws a single car; hiding via shape visibility flags after
  build, or two template variants if flags are immutable post-finalize).
- One `SolverMuJoCo` (`use_mujoco_contacts=False`, `njmax`/`nconmax` scaled to
  world count), one `WheeledVehicles` layer (auto-discovers N+1 vehicles; the
  `references=` attribute remapping keeps vehicle/wheel ids correct under
  replication), one CUDA graph for the whole per-frame pipeline on CUDA.

### Per-frame pipeline (replan every frame)

1. **Snapshot** hero slice: `body_q`/`body_qd` (world-0 bodies), wheel
   `dyn.omega`, `dyn.trans_long`, `dyn.trans_lat`, `patch.fz` (world-0
   wheels), and env-0 `ProgressTracker` state.
2. **Broadcast** the snapshot to all N+1 worlds/envs (scatter kernels; wheel
   and body layouts are homogeneous contiguous per-world slices).
3. **Sample**: `planner.sample()` fills `samples [K, H, A]` =
   clamp(nominal + smoothed noise); sample 0 is zero-noise (the nominal).
   K = N+1 (world k runs sample k; the hero world rolls out the nominal).
4. **Rollout** H steps. Each step, per world k: write `samples[k, t]` into the
   `WheeledVehicles` command arrays (device kernel, not `set_commands`),
   substep the solver (`--rollout-substeps`, default 4), update track signals
   (`ProgressTracker.update()`, `CollisionChecker.query()`), accumulate cost.
   Record the hero-world chassis position per step for the trajectory ribbon.
5. **Update**: `planner.update(costs)` — MPPI softmax-weighted noise average
   into the nominal; then `planner.shift()`.
6. **Restore** the hero slice from the snapshot (same scatter kernels,
   world 0 only) including env-0 tracker state.
7. **Execute**: write `nominal[0]` to the hero's commands, step one real frame
   (8 substeps, matching the rc_car example). Particle worlds step along with
   garbage state — harmless, they are re-broadcast next frame.

No independent per-world stepping is ever needed, hence one model and one
solver.

### Cost function (per particle k, accumulated over the horizon)

- Progress reward: `-w_prog * (dist_to_next[t-1] - dist_to_next[t])` and
  `-w_pass * checkpoint_passed[t]` (negated: MPPI minimizes cost).
- Wall kill: on first `oob[k] == 1`, set a persistent dead flag; add a large
  terminal penalty and stop accruing progress reward (frozen cost).
- Small control regularization (steer magnitude / command rate) for smoothness.
- Weights are constants in the example, tunable via the UI panel.

## Track (track_gen)

- `TrackGenConfig(num_envs=E, generator="bezier", relax_enable=True,
  half_width=0.5, scale=S, device=...)` with `E = N+1` and an identical-seed
  `wp.array` (`PerEnvSeededRNG` folds no env index, so equal seeds ⇒ identical
  tracks) so every car/env shares one geometry while owning its own collision
  box and progress state (ProgressTracker is strictly one-agent-per-env).
- `S` calibrated during implementation so the track footprint is ~20 m
  (defaults span ~1–2 units; expect S ≈ 10; verify via `Track.length` and
  bump `N_max` if `perimeter/spacing` approaches it).
- Validity: generate, check `track.valid[0]`; on failure retry with seed+1 (all
  envs bumped identically) up to a bounded number of attempts. Dev-time
  calibration: measure yield over ≥1024 distinct seeds at the chosen config
  and require ≥99.9% (raise `relax_iters` if short) before locking defaults.
- Utilities wired directly (not the `Course` facade, since we must
  snapshot/broadcast tracker state anyway): `CheckpointSampler` (spacing
  ~1.0 m) + `ProgressTracker` (positions bound to a `[E]` vec2 buffer filled
  from chassis `body_q` each step) + `CollisionChecker(method="segments",
  max_boxes=1)` with per-car oriented boxes (car half-extents, chassis yaw).
- Car spawn: centerline point 0, oriented along the tangent, all worlds.

## Planner API (`newton/_src/vehicles/mppi.py`)

```python
class ControllerMPPI:
    @dataclass
    class Config:
        num_samples: int = 1024   # K, sample 0 is the zero-noise nominal
        horizon: int = 32         # H steps
        dim: int = 2              # A action channels (example: drive, steer)
        sigma: tuple[float, ...] = (0.3, 0.4)      # per-channel noise std
        temperature: float = 0.05                  # softmax lambda
        beta: float = 0.7         # per-step noise smoothing (0 = white noise)
        bounds_lo: tuple[float, ...] = (-1.0, -1.0)
        bounds_hi: tuple[float, ...] = (1.0, 1.0)
        seed: int = 0

    samples: wp.array3d[float]    # [K, H, A], filled by sample()
    nominal: wp.array2d[float]    # [H, A]

    def sample(self) -> None
    def update(self, costs: wp.array[float]) -> None   # length K
    def shift(self) -> None
```

- All methods are pure `wp.launch` (CUDA-graph capturable): RNG state advances
  via a device counter, min-cost reduction and weighted averaging are
  deterministic loop-over-K kernels (H·A threads), no atomics, no host sync.
- Sim-agnostic: the consumer owns rollouts and the cost array; docstring shows
  the sample → rollout/cost → update → shift cycle.
- Exported from `newton/vehicles.py`; registered in `docs/generate_api.py`
  output; CHANGELOG entry under Added.

## Visualization

- `cone.usda`: hand-authored ~20 cm tall orange sports cone (cone + thin
  square base), authored in metres, Z-up. Loaded via
  `Usd.Stage.Open(get_asset(...))` → `newton.Mesh` (render-only, never in the
  physics model).
- `PropSampler(track, spacing=0.5, boundary="inner"|"outer", mode="points")`
  on env 0; poses → one `wp.array[wp.transform]`; a single instanced
  `viewer.log_shapes("/track/cones", GeoType.MESH, ..., geo_src=cone_mesh)`.
- Nominal-rollout ribbon via `viewer.log_lines` from the recorded hero
  positions; recolored by plan cost.
- Follow camera identical in spirit to `example_vehicle_rc_car`.
- UI panel (controller outputs, not manual controls): throttle/steer/brake of
  the executed command, speed, laps + checkpoint progress, best/mean sample
  cost, alive-particle fraction, and sliders for temperature/sigma/cost
  weights.

## Testing

- `newton/tests/test_vehicle_mppi.py` (unittest, no track_gen import): sample
  bounds respected; sample 0 equals nominal; update moves nominal toward a
  synthetic low-cost sample; shift semantics; seeded determinism; CPU device
  support.
- Example implements `test_post_step` (finite states) and `test_final` (hero
  advanced ≥ threshold checkpoints, never flagged OOB-dead, finite pose).
  Under `--test`: small K (e.g. 64), short horizon, fixed seed. If track_gen
  is missing the example raises `ImportError` with an install hint at import
  time; the test wrapper skips it cleanly.
- Manual verification: run with the GL viewer on CUDA; confirm laps complete.

## Accepted cuts (post-implementation)

- The nominal-plan ribbon renders in a fixed green rather than recolored by
  plan cost (cosmetic; cost is shown in the UI panel).
- Rollouts additionally kill particles whose measured backward travel exceeds
  a 1 m budget (reverse stays a brake/back-out tool), and rollouts that start
  out-of-bounds switch to a penetration-depth recovery cost instead of kills —
  both added beyond this spec at user request.

## Out of scope

- Physical cone/wall collision (cones are decorative; walls act through cost).
- Multi-lap timing/leaderboards, opponent cars, texture/road rendering.
- Promoting MPPI beyond `newton.vehicles` (revisit if a second consumer shows
  up).
