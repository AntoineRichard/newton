# Relaxation-Length (Transient) Tire — Design

Status: Scoped (approved 2026-06-16); not yet implemented
Date: 2026-06-16
Layer: `newton.vehicles`
Related:
- Realism iteration: `docs/superpowers/specs/2026-06-15-wheeled-vehicle-realism-iteration.md` (Tier 3 item, now scoped)
- Touches `newton/_src/vehicles/wheel.py`, `controller.py`; tests in `newton/tests/test_vehicles_wheel.py`

## Purpose

Give the tire a transient (relaxation-length) lateral response so it builds
lateral force gradually as it rolls, instead of reacting instantly to the
current slip. This is both:

1. **More realistic** — a real tire develops side force over a characteristic
   rolling distance (the relaxation length `σ`); it does not switch instantly.
2. **A stabilizer** — today the lateral force is an explicit function of the
   instantaneous slip, applied to light wheel bodies through a stiff
   articulation. At high grip (high `mu`/stiffness) the saturated force can flip
   direction step-to-step and pump a roll/hop instability. Lagging the slip
   low-passes the force, so high `mu` stops being an explicit-integration
   stiffness problem.

## Background / current state

- `tire_force()` (`tire.py`) maps slip (`kappa`, `alpha`) + load to a force; it
  is a pure function of the slip and is **not changed** by this work.
- `_wheel_dynamics_kernel` (`wheel.py`) computes the instantaneous `alpha =
  atan2(v_lat, ref)` and `kappa`, calls `tire_force`, applies an anti-overshoot
  lateral cap, and injects the wrench. `ref = max(|v_long|, min_ref)` is the
  regularized rolling reference speed already in scope.
- An anti-overshoot lateral cap (committed) limits `|f_lat|` so it cannot reverse
  the contact's lateral velocity in a substep. Relaxation will make this rarely
  bind; we keep it as a cheap backstop.

## Approach (chosen): relax the slip angle

Carry a per-wheel **transient slip angle** `alpha'` that lags the instantaneous
`alpha`, and feed `alpha'` to `tire_force` in place of `alpha`. The first-order
relaxation ODE (Pacejka transient model) is

```
σ · d(alpha')/dt + V · alpha' = V · alpha
```

where `V` is the rolling speed and `σ` the relaxation length. Integrated
**implicitly** for unconditional stability:

```
alpha' ← (σ · alpha'_prev + dt · V · alpha) / (σ + dt · V)
```

Use `V = ref` (the existing regularized reference speed). Properties:

- **Stable for any `σ`, `dt`, `V`** (implicit first-order) — this is the point:
  high `mu` no longer needs a smaller `dt`.
- **Low speed is handled for free**: as `V → 0` the update → `alpha' =
  alpha'_prev` (the tire holds its deflection near standstill); no division
  blow-up, which is the usual failure mode of relaxation models.
- **Steady state is unchanged**: at constant slip `alpha' → alpha`, so the force
  equals today's brush force. Top speed, steady cornering radius, braking, and
  skid-steer steady behavior are unchanged — only the *transient* response lags.
- **`σ = 0` degenerates to the current model**: the update gives `alpha' =
  alpha`. This is the library default (off), so existing vehicles/tests are
  byte-for-byte unaffected; the rc_car opts in.

Rejected alternatives:
- **Filter the output force** — lagging an already friction-circle-saturated
  force is awkward and couples poorly with the circle clip.
- **Explicit carcass-deflection spring** — mathematically equivalent to slip
  relaxation but adds redundant state.

## Decisions (locked)

1. **Lateral only.** Relax `alpha` (lateral). Longitudinal slip already gets
   implicit damping from the semi-implicit spin coupling (`denom`), and relaxing
   `kappa` interacts with that loop for little gain. Longitudinal relaxation
   (`kappa'`) is a noted follow-up, not in this work.
2. **Radius-scaled `σ`.** `σ = relaxation_length_ratio · wheel_radius`, computed
   per wheel at construction (mirrors how `pneumatic_trail` is radius-scaled).
   One `σ` per wheel. Default ratio chosen below.
3. **Keep the anti-overshoot cap** as a backstop (cheap; rarely binds with
   relaxation; guards `σ → 0` / edge cases).
4. **Library default off (`relaxation_length_ratio = 0.0`); rc_car opts in
   (`= 1.0`).** Keeps husky and all existing tests unchanged; the rc_car (the
   sim-to-real target) gets the transient model.

## Design / changes (all in `newton._src.vehicles`)

- `controller.py` — `WheeledConfig`:
  - Add `relaxation_length_ratio: float = 0.0` (σ as a fraction of wheel radius;
    0 disables, preserving current behavior).
  - In `WheeledVehicles._init_params`, set the per-wheel `σ` array from
    `radius * relaxation_length_ratio` (like `pneumatic_trail`).
- `wheel.py` — `WheelDynamics`:
  - Add per-wheel state `trans_alpha: wp.array[float]` (transient slip angle,
    persists across steps like `omega`), zero-initialized.
  - Add parameter `relax_length: wp.array[float]` (per-wheel `σ`).
  - In `_wheel_dynamics_kernel`, after computing `alpha` and before `tire_force`:
    ```
    sigma = relax_length[w]
    v_ref = ref
    alpha_used = (sigma * trans_alpha[w] + dt * v_ref * alpha) / (sigma + dt * v_ref)
    trans_alpha[w] = alpha_used
    ```
    (When `sigma == 0` this reduces to `alpha_used = alpha`.) Pass `alpha_used`
    to `tire_force` instead of `alpha`. The diagnostic `out_alpha` reports the
    instantaneous `alpha` (the actual slip), not the lagged value.
  - Add `relax_length` and `trans_alpha` to the `apply_wheel_dynamics` launch and
    kernel signature.
  - Reset/handling: `trans_alpha` is only updated for active wheels; when a wheel
    is inactive (airborne) it holds its last value and relaxes to the new slip on
    re-contact (acceptable; documented).
- `example_vehicle_rc_car.py`: set `relaxation_length_ratio = 1.0` in the config;
  optionally add a "Relaxation length" slider (ratio 0–3) to the Handling panel.

## Testing

- **Step response (unit, `wheel.py`)**: at constant `V` with a step in lateral
  velocity, `f_lat` must rise *gradually*, not instantly — after rolling one
  relaxation length (time `σ/V`) the lagged slip/force reaches ≈63% (1 − 1/e) of
  the steady value, and approaches steady later. Fails with `σ = 0` (instant).
- **High-grip stability (integration)**: the hop scenario (high `c_lat`, `mu`)
  with the **anti-overshoot cap disabled** stays stable (bounded wheel vertical
  velocity) with relaxation on, demonstrating relaxation alone is a stabilizer.
- **Steady-state unchanged (regression)**: straight-line top speed, steady
  cornering, and skid-steer rotation match the no-relaxation results within a
  small tolerance (only transients differ).
- **Low-speed sanity**: near-standstill steering with relaxation on produces no
  NaN and no hop.
- All parametrized CPU + CUDA via `get_test_devices()`.

## Exit criteria

- With `relaxation_length_ratio = 0`, every existing vehicle test is unchanged.
- With relaxation on, the rc_car stays stable at `mu = 2–3` and `sim_substeps =
  8` with the anti-overshoot cap **off** (relaxation carries the stability).
- Measured lateral-force step-response time constant ≈ `σ / V`.
- rc_car straight-line, top speed, steady cornering, and husky skid-steer are
  unchanged (steady state); only transient turn-in response is lagged.

## Out of scope (follow-ups)

- Longitudinal slip relaxation (`kappa'`) and its interaction with the spin
  coupling.
- Speed/load-dependent `σ` (real `σ` grows with load and falls with speed).
- Combined-slip relaxation cross-coupling.
- Measuring/fitting `σ` from the real vehicle (sim-to-real calibration).
