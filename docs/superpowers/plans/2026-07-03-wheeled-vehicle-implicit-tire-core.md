# Implicit Impulse-Budget Tire Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the explicit tire-force injection core of `newton/_src/vehicles/` with a per-wheel implicit impulse-budget solve that is stable by construction up to μ = 2.5, then retire the first-generation `newton/_src/wheeled/` module.

**Architecture:** Per wheel and substep, a small implicit solve in slip velocities `(u_long, u_lat)` against the wheel body's contact-frame effective mass (Delassus block from `body_inv_mass`/`body_inv_inertia`) plus the analytical spin inertia. A stick test runs first (impulse that zeroes slip, checked against the static friction circle); otherwise the slip system is solved with a secant tire stiffness and the resulting impulse is projected onto `‖p‖ ≤ μ·Fz·dt`. The two shipped band-aids (low-speed lateral cap, load-filter smoothing) are deleted. See the spec: `docs/superpowers/specs/2026-07-03-wheeled-vehicle-implicit-tire-core-design.md`.

**Tech Stack:** Python, NVIDIA Warp kernels, Newton `ModelBuilder`/`SolverMuJoCo`, `unittest`.

## Global Constraints

- Tests use `unittest`, never pytest. Run with `uv run --extra dev -m newton.tests -k <pattern>`.
- Never call `wp.synchronize()` before `.numpy()` on a Warp array.
- PEP 604 unions (`x | None`); Warp array annotations use bracket syntax (`wp.array[wp.float32]`).
- Google-style docstrings; SI units in public docstrings (`[N]`, `[m/s]`, `[N·s]`).
- New files: SPDX header year 2026 (`Copyright (c) 2026 The Newton Developers`, `Apache-2.0`). Never change the year on existing files.
- Commit style: imperative subject ≤ ~50 chars, body wraps at 72 explaining what and why, no `feat:` prefixes. End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Work stays on branch `antoiner/wheeled-vehicle-design`.
- Before each commit: `uvx pre-commit run -a` (fix what it flags on your files).
- MuJoCo-dependent tests must keep the existing `except ImportError: raise unittest.SkipTest` pattern.
- The examples must not import from `newton._src` (public modules only).

## File Structure

| File | Role |
|---|---|
| `newton/_src/vehicles/impulse.py` (new) | Pure Warp-func math: effective mass, implicit slip solve, circle projection. No model/state knowledge. |
| `newton/_src/vehicles/wheel.py` (rewrite core) | Kernel orchestration: frames, slip, drive/brake, calls into `impulse.py`, applies wrench, spin update. |
| `newton/_src/vehicles/contact.py` (modify) | Load latch: direct latch, no smoothing; zero when airborne. Analytic plane footprint diagnostic (ported). |
| `newton/_src/vehicles/controller.py` (modify) | Config: drop `load_filter`, add `static_mu_scale`. |
| `newton/_src/vehicles/metadata.py` (modify) | Manifest ingestion + USD auto-detect (ported); drop `PRESERVE_CONTACT_FOOTPRINT` usage. |
| `newton/tests/test_vehicles_impulse.py` (new) | Unit tests for the solve funcs (stick/slip/clamp/locked/passivity). |
| `newton/tests/test_vehicles_stability.py` (new) | Regime-map acceptance suite (sprung fixture, μ sweep). |
| Deletions | `newton/_src/wheeled/`, `newton/wheeled.py`, `newton/examples/wheeled/`, `newton/tests/test_wheeled_vehicle_*.py`, `docs/api/newton_wheeled.rst`, `PRESERVE_CONTACT_FOOTPRINT`. |

---

### Task 1: Implicit impulse solve math (`impulse.py`)

**Files:**
- Create: `newton/_src/vehicles/impulse.py`
- Test: `newton/tests/test_vehicles_impulse.py`

**Interfaces:**
- Produces: `wheel_effective_mass(m_inv: float, i_inv_world: wp.mat33, offset: wp.vec3, t_fwd: wp.vec3, t_lat: wp.vec3) -> wp.vec3` returning `(W11, W12, W22)`.
- Produces: `solve_tire_impulse(u_long: float, u_lat: float, a11: float, a12: float, a22: float, k_long: float, k_lat: float, budget: float, budget_stick: float) -> vec6` returning `(p_long, p_lat, u_long_new, u_lat_new, stick, utilization)`. `k_*` are impulse-per-slip-velocity secant stiffnesses (`dt·C`) [N·s/(m/s)]; `budget = μ·Fz·dt` [N·s].
- Produces: `vec6 = wp.types.vector(length=6, dtype=wp.float32)` (module-level type alias used by tests and `wheel.py`).

- [ ] **Step 1: Write the failing tests**

Create `newton/tests/test_vehicles_impulse.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.vehicles.impulse import solve_tire_impulse, vec6, wheel_effective_mass
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _solve_kernel(
    inp: wp.array[vec6],
    budgets: wp.array[wp.vec2],
    stiff: wp.array[wp.vec2],
    out: wp.array[vec6],
):
    i = wp.tid()
    v = inp[i]
    out[i] = solve_tire_impulse(
        v[0], v[1], v[2], v[3], v[4], stiff[i][0], stiff[i][1], budgets[i][0], budgets[i][1]
    )


@wp.kernel
def _effmass_kernel(out: wp.array[wp.vec3]):
    # 1 kg point mass, inertia I = identity*0.01, contact 0.05 m below COM
    i_inv = wp.mat33(100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    out[0] = wheel_effective_mass(
        1.0, i_inv, wp.vec3(0.0, 0.0, -0.05), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)
    )


def _solve(device, u, a, k, budget, budget_stick):
    inp = wp.array([vec6(u[0], u[1], a[0], a[1], a[2], 0.0)], dtype=vec6, device=device)
    budgets = wp.array([wp.vec2(budget, budget_stick)], dtype=wp.vec2, device=device)
    stiff = wp.array([wp.vec2(k[0], k[1])], dtype=wp.vec2, device=device)
    out = wp.zeros(1, dtype=vec6, device=device)
    wp.launch(_solve_kernel, dim=1, inputs=[inp, budgets, stiff, out], device=device)
    return out.numpy()[0]


def test_effective_mass_positive_definite(test, device):
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_effmass_kernel, dim=1, inputs=[out], device=device)
    w11, w12, w22 = (float(x) for x in out.numpy()[0])
    # offset -0.05 z, t_fwd x: r x t_fwd = (0,-0.05,0) -> +0.05^2*100 = 0.25 rotational term
    test.assertAlmostEqual(w11, 1.0 + 0.25, places=5)
    test.assertAlmostEqual(w22, 1.0 + 0.25, places=5)
    test.assertAlmostEqual(w12, 0.0, places=6)
    test.assertGreater(w11 * w22 - w12 * w12, 0.0)


def test_stick_when_impulse_fits_budget(test, device):
    # tiny slip velocity, huge budget: stick, slip zeroed exactly
    r = _solve(device, (0.02, -0.01), (1.0, 0.0, 1.0), (100.0, 100.0), 10.0, 10.0)
    p_long, p_lat, u1, u2, stick, util = (float(x) for x in r)
    test.assertEqual(stick, 1.0)
    test.assertAlmostEqual(u1, 0.0, places=6)
    test.assertAlmostEqual(u2, 0.0, places=6)
    # p_stick = -A^-1 u
    test.assertAlmostEqual(p_long, -0.02, places=5)
    test.assertAlmostEqual(p_lat, 0.01, places=5)


def test_slip_solve_reduces_slip_without_reversal(test, device):
    # stiff tire, big slip, budget too small to stick: implicit solve, no sign flip
    r = _solve(device, (2.0, 0.0), (1.0, 0.0, 1.0), (50.0, 50.0), 0.5, 0.5)
    p_long, p_lat, u1, u2, stick, util = (float(x) for x in r)
    test.assertEqual(stick, 0.0)
    test.assertLess(p_long, 0.0)  # opposes slip
    test.assertGreater(u1, 0.0)  # reduced but NOT reversed
    test.assertLess(u1, 2.0)
    test.assertAlmostEqual(util, 1.0, places=4)  # budget binds


def test_clamped_impulse_on_budget_boundary(test, device):
    r = _solve(device, (5.0, 5.0), (2.0, 0.1, 2.0), (30.0, 30.0), 0.3, 0.3)
    p_long, p_lat, u1, u2, stick, util = (float(x) for x in r)
    p_norm = np.hypot(p_long, p_lat)
    test.assertAlmostEqual(p_norm, 0.3, places=4)
    # clamped u+ must be consistent: u+ = u + A p
    test.assertAlmostEqual(u1, 5.0 + 2.0 * p_long + 0.1 * p_lat, places=4)
    test.assertAlmostEqual(u2, 5.0 + 0.1 * p_long + 2.0 * p_lat, places=4)


def test_passivity_random_inputs(test, device):
    # impulse never feeds energy into the slip state: p . u_new <= 0
    rng = np.random.default_rng(42)
    for _ in range(200):
        u = rng.uniform(-5.0, 5.0, 2)
        a11, a22 = rng.uniform(0.1, 20.0, 2)
        a12 = rng.uniform(-1.0, 1.0) * np.sqrt(a11 * a22) * 0.5
        k = rng.uniform(0.0, 100.0, 2)
        budget = rng.uniform(0.01, 5.0)
        r = _solve(device, u, (a11, a12, a22), k, budget, budget)
        p = np.array([float(r[0]), float(r[1])])
        u_new = np.array([float(r[2]), float(r[3])])
        test.assertLessEqual(float(p @ u_new), 1.0e-5)
        test.assertLessEqual(np.hypot(*p), budget * (1.0 + 1.0e-4))


def test_zero_stiffness_zero_impulse(test, device):
    r = _solve(device, (1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0), 1.0, 0.0)
    test.assertAlmostEqual(float(r[0]), 0.0, places=6)
    test.assertAlmostEqual(float(r[1]), 0.0, places=6)


class TestVehiclesImpulse(unittest.TestCase):
    pass


for _name, _fn in (
    ("test_effective_mass_positive_definite", test_effective_mass_positive_definite),
    ("test_stick_when_impulse_fits_budget", test_stick_when_impulse_fits_budget),
    ("test_slip_solve_reduces_slip_without_reversal", test_slip_solve_reduces_slip_without_reversal),
    ("test_clamped_impulse_on_budget_boundary", test_clamped_impulse_on_budget_boundary),
    ("test_passivity_random_inputs", test_passivity_random_inputs),
    ("test_zero_stiffness_zero_impulse", test_zero_stiffness_zero_impulse),
):
    add_function_test(TestVehiclesImpulse, _name, _fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
```

Note on `test_stick_when_impulse_fits_budget`: with `A = I` and `u = (0.02, −0.01)`, `p_stick = −u`; the stick budget of 10 N·s comfortably contains it. Note on `test_slip_solve_reduces_slip_without_reversal`: unclamped `p = −k·u⁺` with `(1 + a·k)u⁺ = u` gives `u⁺ = 2/51 ≈ 0.039`, `|p| ≈ 1.96 > 0.5`, so the budget binds and `u⁺ = 2 − 0.5 = 1.5 > 0` — slip reduced, never reversed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_impulse`
Expected: FAIL/ERROR with `ModuleNotFoundError: ... newton._src.vehicles.impulse`.

- [ ] **Step 3: Write the implementation**

Create `newton/_src/vehicles/impulse.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-wheel implicit impulse-budget tire solve.

Pure Warp math with no model/state knowledge. The tire's tangential action on
the wheel body is computed as an *impulse* over the substep, solved implicitly
in the slip velocities and projected onto the friction circle
``|p| <= mu * Fz * dt``. By construction no substep can apply more tangential
impulse than the contact can absorb, which removes the saturated-force sign
chatter that made explicit injection explode at high grip (see
``docs/superpowers/specs/2026-07-03-wheeled-vehicle-implicit-tire-core-design.md``).

Conventions: slip velocity ``u = v_contact - omega * r`` (tire force opposes
``u``); ``A`` is the slip-space Delassus (inverse effective mass) so that
``u_new = u + A @ p`` for a tire impulse ``p`` on the wheel body.
"""

from __future__ import annotations

import warp as wp

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def wheel_effective_mass(
    m_inv: float,
    i_inv_world: wp.mat33,
    offset: wp.vec3,
    t_fwd: wp.vec3,
    t_lat: wp.vec3,
) -> wp.vec3:
    """Tangential Delassus block ``W = J M^-1 J^T`` of the free wheel body.

    ``offset`` is the contact point relative to the body COM [m]; the returned
    ``(W11, W12, W22)`` maps a tangential impulse [N·s] at the contact to the
    contact-point velocity change [m/s] (1 = t_fwd, 2 = t_lat). The free-body
    block ignores joint constraints, which can only increase effective mass, so
    impulses computed against it are always absorbable — a stable-side error.
    """
    ru = wp.cross(offset, t_fwd)
    rv = wp.cross(offset, t_lat)
    w11 = m_inv + wp.dot(ru, i_inv_world * ru)
    w12 = wp.dot(ru, i_inv_world * rv)
    w22 = m_inv + wp.dot(rv, i_inv_world * rv)
    return wp.vec3(w11, w12, w22)


@wp.func
def solve_tire_impulse(
    u_long: float,
    u_lat: float,
    a11: float,
    a12: float,
    a22: float,
    k_long: float,
    k_lat: float,
    budget: float,
    budget_stick: float,
) -> vec6:
    """Implicit tire impulse with stick test and friction-circle projection.

    Args:
        u_long: Free longitudinal slip velocity [m/s].
        u_lat: Free lateral slip velocity [m/s].
        a11: Slip-space Delassus (1,1) [(m/s)/(N·s)] (includes spin mobility).
        a12: Slip-space Delassus (1,2).
        a22: Slip-space Delassus (2,2).
        k_long: Longitudinal secant impulse stiffness ``dt*C`` [N·s/(m/s)].
        k_lat: Lateral secant impulse stiffness [N·s/(m/s)].
        budget: Kinetic friction-circle impulse budget ``mu*Fz*dt`` [N·s].
        budget_stick: Static budget ``mu_s*Fz*dt`` [N·s].

    Returns:
        ``(p_long, p_lat, u_long_new, u_lat_new, stick, utilization)``:
        tire impulse on the wheel body [N·s], post-solve slip velocities [m/s],
        stick flag (1.0 when the stick solution was taken), and
        ``|p| / budget`` clamped to [0, 1].
    """
    if budget <= 0.0:
        return vec6(0.0, 0.0, u_long, u_lat, 0.0, 0.0)

    # Stick first: the impulse that zeroes the slip velocity, A p = -u.
    det_a = a11 * a22 - a12 * a12
    det_a = wp.max(det_a, 1.0e-12)
    ps1 = -(a22 * u_long - a12 * u_lat) / det_a
    ps2 = -(a11 * u_lat - a12 * u_long) / det_a
    ps_norm = wp.sqrt(ps1 * ps1 + ps2 * ps2)
    if ps_norm <= budget_stick:
        util = wp.min(ps_norm / budget, 1.0)
        return vec6(ps1, ps2, 0.0, 0.0, 1.0, util)

    # Slip: p = -K u_new, (I + A K) u_new = u, K = diag(k_long, k_lat).
    b11 = 1.0 + a11 * k_long
    b12 = a12 * k_lat
    b21 = a12 * k_long
    b22 = 1.0 + a22 * k_lat
    det_b = wp.max(b11 * b22 - b12 * b21, 1.0e-12)
    un1 = (b22 * u_long - b12 * u_lat) / det_b
    un2 = (b11 * u_lat - b21 * u_long) / det_b
    p1 = -k_long * un1
    p2 = -k_lat * un2
    p_norm = wp.sqrt(p1 * p1 + p2 * p2)
    if p_norm > budget:
        s = budget / wp.max(p_norm, 1.0e-12)
        p1 = p1 * s
        p2 = p2 * s
        # Recompute the post-impulse slip consistently with the clamped impulse.
        un1 = u_long + a11 * p1 + a12 * p2
        un2 = u_lat + a12 * p1 + a22 * p2
        p_norm = budget
    util = wp.min(p_norm / budget, 1.0)
    return vec6(p1, p2, un1, un2, 0.0, util)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_impulse`
Expected: all 6 tests PASS (per device).

If `test_passivity_random_inputs` fails on the clamped branch: the clamped `p` keeps the direction of the unclamped `p = −K u⁺`, and `p·u_new = p·u + pᵀAp`. If a rare random draw violates the 1e-5 bound, tighten the projection by scaling toward the stick direction instead: replace the clamp block's direction with `(ps1, ps2)/ps_norm * budget` (the stick impulse direction, which always opposes `u`), and recompute `un1/un2` the same way. Re-run.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/vehicles/impulse.py newton/tests/test_vehicles_impulse.py
git commit  # subject: "Add implicit impulse-budget tire solve math"
```

---

### Task 2: Rewire the wheel-dynamics kernel onto the implicit solve

**Files:**
- Modify: `newton/_src/vehicles/wheel.py` (replace the active-contact branch, `wheel.py:209-293`)
- Modify: `newton/tests/test_vehicles_wheel.py` (replace the cap regression test)

**Interfaces:**
- Consumes: `wheel_effective_mass`, `solve_tire_impulse`, `vec6` from Task 1; `tire_force` from `tire.py` (unchanged).
- Produces: `WheelDynamics` gains arrays `stick: wp.array[wp.int32]`, `impulse_utilization: wp.array[wp.float32]`, and parameter `static_mu: wp.array[wp.float32]`. `apply_wheel_dynamics(model, state, data, patch, dyn, dt)` keeps its signature but the kernel now needs `model.body_inv_mass` and `model.body_inv_inertia`. The low-speed lateral cap is gone.

- [ ] **Step 1: Write the failing tests**

In `newton/tests/test_vehicles_wheel.py`, delete `test_low_speed_lateral_force_capped_against_overshoot` (lines 133-157) and its `add_function_test` registration, and add these two tests (mirror the file's existing fixture style — it builds a one-wheel rig and calls `apply_wheel_dynamics` directly; reuse its helpers):

```python
def test_low_speed_impulse_never_reverses_slip(test, device):
    """At high mu and tiny lateral velocity the tire must stick or reduce slip,
    never reverse it. This replaces the deleted lat_cap band-aid regression."""
    # Build the file's standard one-wheel fixture with a small lateral velocity
    # and a very high friction override (mu=3), fz latched to 40 N, dt=1/240.
    # After apply_wheel_dynamics:
    #   - dyn.stick[0] == 1, or
    #   - the applied lateral impulse |f_lat*dt| <= effective-mass * |v_lat| bound
    # and in both cases |f_lat| * dt <= mu * fz * dt.
    ...


def test_hard_brake_lockup_no_lateral_kick(test, device):
    """A locked wheel sliding straight must produce force opposing motion with
    negligible lateral component (no direction chatter at kappa -> -1)."""
    # One-wheel fixture moving at v_long=5 m/s, omega=0, brake_target large,
    # mu=2.5. After apply_wheel_dynamics: f_long < 0, |f_lat| < 0.05*|f_long|.
    ...
```

Write these as real code against the existing fixture in that file (read `test_vehicles_wheel.py:1-130` first and follow `test_tire_reaction_and_injection`'s setup pattern exactly — same builder, same latched-fz injection via `patch.fz.assign`, same manual kernel invocation). The assertions above are the contract; the fixture plumbing comes from the file.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_wheel`
Expected: the two new tests FAIL (no `dyn.stick` array; lockup produces lateral chatter), existing tests still pass.

- [ ] **Step 3: Implement the kernel rewrite**

In `newton/_src/vehicles/wheel.py`:

1. Add imports: `from .impulse import solve_tire_impulse, wheel_effective_mass`.
2. In `WheelDynamics.__init__`, add:

```python
        self.static_mu_scale = z(wp.float32)   # parameters section
        # diagnostics section
        self.stick = z(wp.int32)
        self.impulse_utilization = z(wp.float32)
```

3. In `apply_wheel_dynamics`, zero the new diagnostics (`dyn.stick.zero_()`, `dyn.impulse_utilization.zero_()`) and extend the launch inputs with `model.body_inv_mass`, `model.body_inv_inertia`, `dyn.static_mu_scale`, `dyn.brake_max` is already there via `brake_target`; append outputs `dyn.stick`, `dyn.impulse_utilization`.
4. Replace the kernel body from the slip computation down to the spin update (keep the frame/offset/`v_contact` code and both explanatory comment blocks at `wheel.py:222-238` verbatim). New physics section:

```python
            v_long = wp.dot(v_contact, fwd_t)
            v_lat = wp.dot(v_contact, lat_t)
            ref = wp.max(wp.abs(v_long), wp.max(min_ref[w], 1.0e-4))

            fz = fz_latched[w]
            if fz <= 0.0:
                fz = fallback_load[w]
            if fz > 0.0:
                mu = mu_override[w]
                if mu < 0.0:
                    mu = friction_seed[w]
                mu = wp.max(mu, 0.0)

                # --- free velocities (drive torque advances spin; brake handled below)
                inv_i = 1.0 / wp.max(inertia[w], 1.0e-9)
                omega_free = om + dt * (tau_drive - damping[w] * om) * inv_i
                # brake locks the wheel this substep if its impulse capacity
                # exceeds the free spin momentum (conservative: ignores the
                # tire's own spin-up torque)
                locked = brake_target[w] * dt * inv_i >= wp.abs(omega_free)

                # --- slip state and operating-point tire force (for the secant)
                if locked:
                    u_long = v_long   # wheel surface is stationary: slip = ground speed
                else:
                    u_long = v_long - omega_free * r
                u_lat = v_lat
                kappa = -u_long / ref
                alpha = wp.atan2(u_lat, ref)
                f0 = tire_force(tire_model[w], kappa, alpha, fz, mu, c_long[w], c_lat[w], 0.0)

                # secant impulse stiffness dt*C, capped by the linear-regime slope
                k_lin_long = dt * c_long[w] * fz / ref
                k_lin_lat = dt * c_lat[w] * fz / ref
                k_long = wp.min(dt * wp.abs(f0[0]) / wp.max(wp.abs(u_long), 1.0e-6), k_lin_long)
                k_lat = wp.min(dt * wp.abs(f0[1]) / wp.max(wp.abs(u_lat), 1.0e-6), k_lin_lat)

                # --- slip-space effective mass: free-body Delassus + spin mobility
                rot_m = wp.quat_to_matrix(rot)
                i_inv_w = rot_m * body_inv_inertia[body] * wp.transpose(rot_m)
                dela = wheel_effective_mass(body_inv_mass[body], i_inv_w, offset, fwd_t, lat_t)
                a11 = dela[0]
                a12 = dela[1]
                a22 = dela[2]
                if not locked:
                    a11 = a11 + r * r * inv_i  # spinning wheel adds slip mobility

                budget = mu * fz * dt
                sol = solve_tire_impulse(
                    u_long, u_lat, a11, a12, a22, k_long, k_lat, budget, static_mu_scale[w] * budget
                )
                f_long = sol[0] / dt
                f_lat = sol[1] / dt

                # self-aligning moment from the *resolved* lateral force
                util = sol[5]
                mz = -f_lat * pneumatic_trail[w] * wp.max(1.0 - util, 0.0)

                force_world = fwd_t * f_long + lat_t * f_lat
                torque_world = wp.cross(offset, force_world) + mz * n
                wp.atomic_add(body_f, body, wp.spatial_vector(force_world, torque_world))

                # --- spin update from the resolved impulse
                if locked:
                    om_new = 0.0
                else:
                    om_new = omega_free - sol[0] * r * inv_i
                    # residual resistive torques (brake below capacity + rolling)
                    # brake toward zero without reversing
                    resist = (brake_target[w] + rolling_resistance[w]) * dt * inv_i
                    if om_new > 0.0:
                        om_new = wp.max(om_new - resist, 0.0)
                    elif om_new < 0.0:
                        om_new = wp.min(om_new + resist, 0.0)
                omega[w] = om_new

                out_kappa[w] = kappa
                out_alpha[w] = alpha
                out_f_long[w] = f_long
                out_f_lat[w] = f_lat
                out_mz[w] = mz
                out_normal_load[w] = fz
                out_stick[w] = wp.int32(sol[4])
                out_utilization[w] = util
```

5. Restructure the inactive path: when there is no contact (or `fz <= 0`), the spin still integrates from drive/brake exactly as the current lines 283-293 do (free spin-up and brake-to-zero tests must keep passing). Move that block into the `else` paths. Delete the old `denom` semi-implicit machinery (`wheel.py:210`, `:275`, `:283-285`) and the entire `lat_cap` block (`wheel.py:260-270`) — the solve replaces both.
6. Keep the drivetrain reaction-torque block (`wheel.py:295-305`) unchanged.
7. Update the kernel signature accordingly (new inputs `body_inv_mass: wp.array[wp.float32]`, `body_inv_inertia: wp.array[wp.mat33]`, `static_mu_scale: wp.array[wp.float32]`; new outputs `out_stick: wp.array[wp.int32]`, `out_utilization: wp.array[wp.float32]`). Update the module docstring: step 3 now reads "solve the implicit tire impulse against the contact effective mass, project onto the friction circle, and accumulate the wrench".

- [ ] **Step 4: Run the wheel and impulse tests**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_wheel` and `-k test_vehicles_impulse`
Expected: all PASS. `test_free_spin_up`, `test_brake_to_zero_no_reverse`, `test_tire_reaction_and_injection`, `test_force_applied_at_ground_contact_not_biased_patch` must pass unmodified (if `test_tire_reaction_and_injection` asserts exact force values from the old explicit path, relax it to direction/saturation assertions and note why in the test docstring).

- [ ] **Step 5: Run the full vehicle suite**

Run: `uv run --extra dev -m newton.tests -k test_vehicles`
Expected: PASS, including `test_steered_launch_does_not_spin_out` and `test_drive_forward`.

- [ ] **Step 6: Commit**

```bash
git add newton/_src/vehicles/wheel.py newton/tests/test_vehicles_wheel.py
git commit  # subject: "Solve tire impulse implicitly against contact effective mass"
# body: explain the stick test, budget clamp, locked-wheel branch, and that
# the lat_cap band-aid is deleted because the budget supersedes it.
```

---

### Task 3: Load latch — direct, decaying, unfiltered

**Files:**
- Modify: `newton/_src/vehicles/contact.py:209-246` and `contact.py:442-445`
- Modify: `newton/_src/vehicles/controller.py` (config + `latch_loads`)
- Modify: `newton/examples/vehicles/example_vehicle_rc_car.py:120` (drop `load_filter=1.0` and its comment)
- Test: `newton/tests/test_vehicles_contact.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `latch_wheel_loads(model, contacts, data, patch)` — the `alpha` parameter is removed. `WheeledConfig` loses `load_filter` and gains `static_mu_scale: float = 1.0`. `WheeledVehicles.latch_loads(contacts)` unchanged externally.

- [ ] **Step 1: Write the failing test**

Add to `newton/tests/test_vehicles_contact.py` (follow the file's existing fixture pattern):

```python
def test_latched_load_zeroes_when_airborne(test, device):
    """fz must not persist across contact loss: a landing wheel starts from the
    fresh measured load, not a stale airborne latch."""
    # Use the file's standard fixture. Drive one latch cycle on the ground so
    # patch.fz > 0, then teleport the body 1 m up, collide + step + latch again.
    # Assert patch.fz.numpy()[0] == 0.0.
    ...
```

Write it as real code against the file's fixture (read `test_vehicles_contact.py:1-100`; `test_normal_load_matches_weight` shows the latch cycle).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_contact`
Expected: new test FAILS (stale fz persists — `_blend_loads` with measured 0 decays but does not zero, and with the default alpha of 0.2 retains 80%).

- [ ] **Step 3: Implement**

In `contact.py`:
- Change the signature to `def latch_wheel_loads(model, contacts, data: VehicleModelData, patch: WheelContactPatch) -> None:` and rewrite the docstring: the measured per-wheel load is latched directly (`fz = measured`); a wheel with no solver-reported normal force this step gets `fz = 0`, so airborne wheels decay immediately and the tire never fires with a stale load. Remove the smoothing rationale paragraph.
- Replace the `_blend_loads` launch with `wp.copy(patch.fz, patch.normal_load)` and delete the `_blend_loads` kernel.

In `controller.py`:
- Delete `load_filter: float = 0.2` from `WheeledConfig` and its docstring mention; add `static_mu_scale: float = 1.0` with docstring: "Static friction budget as a multiple of the kinetic ``mu`` (stick engages when the stopping impulse fits ``static_mu_scale * mu * Fz * dt``)."
- In `_init_params`, add `fill(d.static_mu_scale, c.static_mu_scale)`.
- `latch_loads` becomes `latch_wheel_loads(self.model, contacts, self.data, self.patch)`.

In `example_vehicle_rc_car.py`, remove the `load_filter=1.0` argument and its "band-aid is off" comment lines.

- [ ] **Step 4: Run the tests**

Run: `uv run --extra dev -m newton.tests -k test_vehicles`
Expected: PASS. If `test_normal_load_matches_weight` becomes flaky on the rigid 4-wheel fixture (statically indeterminate per-wheel loads now unsmoothed), average `patch.fz` over the last 10 steps in the test instead of reading one step — the physical total is unchanged.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/vehicles/contact.py newton/_src/vehicles/controller.py \
        newton/examples/vehicles/example_vehicle_rc_car.py newton/tests/test_vehicles_contact.py
git commit  # subject: "Latch wheel loads directly and zero them when airborne"
```

---

### Task 4: Regime-map acceptance suite

**Files:**
- Create: `newton/tests/test_vehicles_stability.py`

**Interfaces:**
- Consumes: the full `newton.vehicles` public API; the fixture patterns of `test_vehicles_controller.py` (`_build_car`, `_steer_front_axle`, `_drive`).
- Produces: the acceptance gate for the spec (§7). No library code.

- [ ] **Step 1: Write the sprung fixture and scenario harness**

Create `newton/tests/test_vehicles_stability.py`. Contents, in order:

1. A **sprung** car builder `_build_sprung_car(device, mu)`: chassis box body (2.9 kg) plus four *separate* wheel bodies (0.18 kg each, cylinder shapes, radius 0.055, matching the rc_car scale) connected by prismatic Z suspension joints with drive stiffness 800 and damping 30 and ±0.025 m limits — this reproduces the light-articulated-wheel-body structure that drove the historic instability (a rigid fixture cannot). Model it on the values in `newton/examples/assets/wheeled/rc_car.usda:273-333`; use `builder.add_joint_prismatic(parent=chassis, child=wheel_body, axis=(0,0,1), ...)` with the builder's joint-drive arguments (read `newton/_src/sim/builder.py`'s `add_joint_prismatic` signature first and mirror how `rc_car.usda`'s drives import). Terrain `ShapeConfig.mu = mu`; vehicle config `friction=-1.0` so the terrain seed drives the sweep. Register attributes and wheels exactly as `test_vehicles_controller.py:_build_car` does, with `nv.add_wheel(..., shape=<wheel cylinder>)` per wheel.
2. A step helper identical in structure to `test_vehicles_controller.py:_drive`'s inner `run`, but recording per-step: `state.body_qd` of all wheel bodies (max |vertical velocity|), chassis roll (from `body_q` quaternion), `vehicles.dynamics.impulse_utilization.numpy().max()`, and finiteness.
3. `MU_SWEEP = (0.5, 1.0, 2.0, 2.5)` and five scenario functions, each looping over `MU_SWEEP`:

```python
def test_low_speed_steer_reversals(test, device):
    # settle 0.5 s; then 5 s at drive=0.15 flipping baked front steering
    # (+/- 25 deg via the _steer_front_axle pattern) every 0.5 s.
    # Assert per mu: max wheel |v_z| < 1.0 m/s, |chassis roll| < 0.35 rad,
    # all states finite, and max impulse_utilization <= 1.0 + 1e-4.

def test_hard_brake_from_top_speed(test, device):
    # settle; full drive 3 s; then brake=1.0, drive=0. Record chassis speed
    # per step. Assert: speed non-increasing within a 0.05 m/s tolerance,
    # |v_lat| < 0.3 m/s and |yaw rate| < 1.0 rad/s throughout braking,
    # final speed < 0.05 m/s and it stays there for 1 s.

def test_slope_hold_static_friction(test, device):
    # Tilt gravity 15 deg (builder.gravity = (g*sin15, 0, -g*cos15)) with a
    # flat ground plane -- equivalent to a 15 deg incline. brake=1.0 from the
    # start. After 5 s: chassis displacement < 0.01 m and dyn.stick all 1
    # for the last 100 steps.

def test_steered_launch_bounded_yaw(test, device):
    # Same protocol as test_vehicles_controller.test_steered_launch_does_not_spin_out
    # (traction-sized motor, baked 25 deg steer, steady-state yaw judged over
    # the second half) but swept over MU_SWEEP on the sprung fixture.
    # Assert steady yaw rate < 6.0 rad/s and travel > 0.3 m at every mu.

def test_straight_line_drift_free(test, device):
    # Full drive 3 s straight. Assert |y| < 0.15 * x and x > 0.5 m.
```

Write all five in full, plus a shared `_run_scenario(...)` helper so the pipeline loop appears once. `dt = 1.0/240.0` (matching the controller tests). Register with `add_function_test(..., devices=get_test_devices())` in a `TestWheeledVehiclesStability(unittest.TestCase)` class.

- [ ] **Step 2: Run the suite**

Run: `uv run --extra dev -m newton.tests -k test_vehicles_stability`
Expected: PASS at all four μ values. This is the spec's acceptance gate. If a scenario fails at μ = 2.0/2.5:
- hop/roll in scenario 1 → the solve is not being hit (check `stick` diagnostics; verify the budget uses the *current* `fz`), or the secant cap `k_lin_*` is being bypassed;
- lateral kick in scenario 2 → confirm the locked-wheel branch engages (`omega == 0`) and `u_long = v_long` there;
- creep in scenario 3 → the stick branch isn't taken: check `budget_stick` plumbing and that brake lock precedes the stick test.
Debug with the diagnostics, do not re-add caps or filters.

- [ ] **Step 3: Verify the suite actually gates (mutation check)**

Temporarily set `budget = 1.0e6` (disable the clamp) in `impulse.py:solve_tire_impulse`, rerun scenario 1 at μ = 2.5, and confirm it FAILS. Revert. This proves the acceptance suite would have caught the historic explosion.

- [ ] **Step 4: Commit**

```bash
git add newton/tests/test_vehicles_stability.py
git commit  # subject: "Add regime-map stability acceptance suite"
# body: cite the spec's mu envelope and the five scenarios; mention the
# mutation check result.
```

---

### Task 5: Example defaults and per-step invariants

**Files:**
- Modify: `newton/examples/vehicles/example_vehicle_rc_car.py`
- Modify: `newton/examples/vehicles/example_vehicle_husky.py`

**Interfaces:**
- Consumes: `dynamics.stick`, `dynamics.impulse_utilization` from Task 2.

- [ ] **Step 1: Update the rc_car example**

- Raise the default tire friction from `friction=1.0` to `friction=2.0` and rewrite the comment block at `example_vehicle_rc_car.py:121-128`: soft-compound RC tires reach μ ≈ 2; the impulse-budget core is validated to 2.5 (point at the stability suite).
- Update the "Tire mu" UI slider bounds at `:183` to `0.2, 2.5` and the `tune_friction` default at `:162` accordingly.
- Add `stick` count and max `impulse_utilization` to the telemetry HUD (read both arrays where the HUD already reads `dyn.kappa`/`dyn.f_lat`).
- Extend `test_post_step()` (add it if the example only has `test_final()`): assert all body states finite and `vehicles.dynamics.impulse_utilization.numpy().max() <= 1.0 + 1e-4`.

- [ ] **Step 2: Update the husky example**

Same `test_post_step()` invariant additions; leave husky's friction defaults alone (skid-steer scrub at high μ is a tuning question, not a stability one — the suite already covers its stability).

- [ ] **Step 3: Run the example tests**

Run: `uv run --extra dev -m newton.tests -k test_examples 2>&1 | grep -i vehicle` (or the specific registered names in `newton/tests/test_examples.py`)
Expected: both vehicle example tests PASS.

- [ ] **Step 4: Commit**

```bash
git add newton/examples/vehicles/
git commit  # subject: "Raise rc_car tire mu to 2.0 on the implicit core"
```

---

### Task 6: Optional implicit relaxation length (default off)

**Files:**
- Modify: `newton/_src/vehicles/impulse.py`, `newton/_src/vehicles/wheel.py`, `newton/_src/vehicles/controller.py`
- Test: `newton/tests/test_vehicles_impulse.py`

**Interfaces:**
- Produces: `solve_tire_impulse_relaxed(u_long, u_lat, s_long, s_lat, beta, a11, a12, a22, k_long, k_lat, budget, budget_stick) -> vec6` plus new `WheelDynamics` state arrays `trans_long`, `trans_lat` and config `relaxation_length_ratio: float = 0.0`.

- [ ] **Step 1: Write the failing tests**

Add to `test_vehicles_impulse.py`:

```python
def test_relaxed_solve_reduces_to_instant_at_beta_one(test, device):
    # beta = 1 (sigma = 0) must reproduce solve_tire_impulse exactly.
    ...

def test_relaxed_solve_is_passive(test, device):
    # random sweep as test_passivity_random_inputs but with beta in (0, 1]
    # and random transient state s; assert p . u_new <= 1e-5 and |p| <= budget.
    ...
```

Write both fully, reusing `_solve`-style launch plumbing with the extended signature.

- [ ] **Step 2: Verify they fail** (`ImportError` on the new func).

- [ ] **Step 3: Implement**

In `impulse.py`, add `solve_tire_impulse_relaxed`. The transient slip `s⁺ = (1−β)s + β u⁺` with `β = dt·V/(σ + dt·V)` feeds the force: `p = −K s⁺ = −(K β) u⁺ − K(1−β)s`. So the slip system becomes `(I + A K β) u⁺ = u + A p₀` with constant `p₀ = −K(1−β)s`; solve, form `p = p₀ − Kβ u⁺`, clamp to the budget as before (recompute `u⁺ = u + A p` on clamp). Stick test unchanged (stick zeroes `u⁺`, and `s⁺ = (1−β)s`). Return `vec6` as before; the caller updates `s⁺` from the returned `u⁺`.

In `wheel.py`: add `trans_long`/`trans_lat` state arrays and a `relaxation_ratio` parameter array to `WheelDynamics`; in the kernel compute `sigma = relaxation_ratio[w] * r`, `beta = dt * ref / wp.max(sigma + dt * ref, 1.0e-9)` and call the relaxed solve; after the solve store `s⁺ = (1.0 - beta) * s + beta * u_new` componentwise. With the default ratio 0, `beta = 1` and behavior is bit-identical to Task 2.

In `controller.py`: add `relaxation_length_ratio: float = 0.0` to `WheeledConfig` with a docstring noting it ships off pending validation (spec §4.5), and `fill` it in `_init_params`.

- [ ] **Step 4: Run** `uv run --extra dev -m newton.tests -k "test_vehicles_impulse or test_vehicles_stability"`
Expected: PASS — including the whole stability suite at ratio 0 (bit-identical path).

- [ ] **Step 5: Commit** — subject: "Add implicit relaxation-length transient slip (default off)".

---

### Task 7: Port manifest ingestion and USD wheel auto-detection

**Files:**
- Modify: `newton/_src/vehicles/metadata.py`
- Source (read-only): `newton/_src/wheeled/metadata.py:176-360` and `:593-608`
- Test: `newton/tests/test_vehicles_metadata.py`
- Reference asset: `newton/examples/assets/wheeled/manifest.json`

**Interfaces:**
- Produces: `load_vehicle_manifest(path: str | Path) -> tuple[VehicleAssetMetadata, ...]` and `apply_vehicle_manifest(builder: ModelBuilder, asset: VehicleAssetMetadata, ...) -> None` in `newton/_src/vehicles/metadata.py`, re-exported from `newton/vehicles.py`.

- [ ] **Step 1: Write failing tests** — in `test_vehicles_metadata.py`, add a test that loads `newton/examples/assets/wheeled/manifest.json` via `load_vehicle_manifest` and asserts the rc_car entry's wheel count (4), radius (0.055), and drive mode; and a test that `apply_vehicle_manifest` on a builder that imported `rc_car.usda` produces `read_vehicle_model_data` tables with `wheel_count == 4` and correct `wheel_id` ordering. Model both on the existing old-module tests in `newton/tests/test_wheeled_vehicle_metadata.py` (they are being deleted in Task 9 — this is their transplant).

- [ ] **Step 2: Verify they fail** (functions do not exist).

- [ ] **Step 3: Port** — copy `load_wheeled_manifest` (`_src/wheeled/metadata.py:176`), `apply_wheeled_manifest_metadata` (`:254`), `apply_wheeled_manifest` (`:339`), the `_usd_is_vehicle_prim`/`_usd_is_wheel_prim`/`_usd_has_true_attribute` helpers (`:593-608`), and the `_require_*` validators they use, into `_src/vehicles/metadata.py`. Rename `wheeled` → `vehicle` throughout (`WheeledAssetMetadata` → `VehicleAssetMetadata`, attribute namespace stays `vehicle:*`), and retarget the metadata-stamping calls at the new module's `set_vehicle`/`add_wheel` (`_src/vehicles/metadata.py:196`/`:237`) instead of the old `_set_builder_attr` machinery. Export the two public names from `newton/vehicles.py`.

- [ ] **Step 4: Run** `uv run --extra dev -m newton.tests -k test_vehicles_metadata` — PASS.
- [ ] **Step 5: Commit** — subject: "Port manifest ingestion to newton.vehicles metadata".

---

### Task 8: Port the analytic plane footprint and golden-curve tests

**Files:**
- Modify: `newton/_src/vehicles/contact.py`
- Source (read-only): `newton/_src/wheeled/contact_patch.py:404-500` (analytic plane patch kernel), `newton/tests/test_wheeled_vehicle_tire.py:600-680` (independent-reference golden curves), `newton/tests/test_wheeled_vehicle_terrain_contact.py:272-282` (narrow-phase toggles) and `:770-801` (ripple sweep)
- Test: `newton/tests/test_vehicles_tire.py`, `newton/tests/test_vehicles_contact.py`, `newton/tests/test_narrow_phase.py`

- [ ] **Step 1: Port the analytic footprint** — add an `enable_analytic_plane_patches: bool = False` keyword to `update_wheel_contact_patches` and port `_apply_analytic_plane_wheel_contact_patches` (sink-depth chord math) from the old module, adapted to the `WheelContactPatch` arrays. Diagnostic only: it overwrites `tangent_extent`/`area` for wheel-on-plane pairs, never the force path.
- [ ] **Step 2: Transplant tests** — (a) into `test_vehicles_tire.py`: a golden-curve test comparing `_eval_tire_kernel` brush output against an independent NumPy implementation of the brush law over a κ/α grid (port the *methodology* of `test_wheeled_vehicle_tire.py:634-668`; the Fiala model itself is not ported — spec §5); (b) into `test_vehicles_contact.py`: the analytic plane chord/area check and the ripple-terrain patch-stability sweep; (c) into `test_narrow_phase.py`: a minimal non-vehicle test that `enable_plane_cylinder_contact_collapse=False` + `enable_axial_contact_projection=False` yields >2 cylinder-plane contacts (ports `test_wheeled_vehicle_terrain_contact.py:272-282`, satisfying spec §5's "non-vehicle test for the toggles").
- [ ] **Step 3: Run** `uv run --extra dev -m newton.tests -k "test_vehicles_tire or test_vehicles_contact or test_narrow_phase"` — PASS.
- [ ] **Step 4: Commit** — subject: "Port analytic footprint and golden-curve tests to vehicles".

---

### Task 9: Delete the first-generation wheeled module

**Files:**
- Delete: `newton/_src/wheeled/`, `newton/wheeled.py`, `newton/examples/wheeled/`, `newton/tests/test_wheeled_vehicle_*.py`, `newton/tests/test_wheeled_vehicle_assets.py`, `docs/api/newton_wheeled.rst`
- Modify: `newton/__init__.py`, `docs/api/_toctree.rst`, `docs/api/newton.rst`, `README.md`, `newton/tests/test_examples.py`, `newton/examples/assets/wheeled/manifest.json` consumers

- [ ] **Step 1: Find every reference first**

Run: `grep -rn "newton.wheeled\|_src.wheeled\|newton:wheeled" --include="*.py" --include="*.rst" --include="*.md" --include="*.usda" newton/ docs/ scripts/ README.md`
Handle each hit: test-asset `.usda` files under `newton/tests/assets/wheeled/` stamp the old `newton:wheeled:*` attribute namespace — restamp them to the surviving `vehicle:*` names used by `_src/vehicles/metadata.py:register_vehicle_attributes` (Task 7's tests exercise them). `scripts/inspect_wheeled_assets.py` and `newton/_src/utils/wheeled_asset_inspection.py`: delete them too if they import the old module (they are Phase-0 one-shot tools); keep only if import-clean.

- [ ] **Step 2: Delete**

```bash
git rm -r newton/_src/wheeled newton/wheeled.py newton/examples/wheeled \
          docs/api/newton_wheeled.rst
git rm newton/tests/test_wheeled_vehicle_*.py newton/tests/test_wheeled_vehicle_assets.py
```

Remove the `newton.wheeled` export from `newton/__init__.py` and its `_toctree.rst`/`newton.rst` entries. In `README.md`, replace the old `example_wheeled_*` registrations (`README.md:249-276`) with `python -m newton.examples vehicle_rc_car` and `vehicle_husky` entries; reuse/replace the screenshot at `docs/images/examples/example_wheeled_drive.jpg` by capturing a 320×320 jpg from `uv run -m newton.examples vehicle_rc_car` (viewer screenshot; if headless, keep the existing image renamed and flag it in the PR description for manual recapture). Remove old-example registrations from `newton/tests/test_examples.py`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run --extra dev -m newton.tests`
Expected: PASS — no import errors, no orphaned registrations.

- [ ] **Step 4: Commit** — subject: "Delete first-generation newton.wheeled module"; body notes the swap checklist from the redesign report and that ports landed in Tasks 7-8.

---

### Task 10: Remove the dead PRESERVE_CONTACT_FOOTPRINT flag

**Files:**
- Modify: `newton/_src/geometry/flags.py:38`, `newton/_src/sim/builder.py:308,423,432`, `newton/_src/vehicles/metadata.py:303`
- Delete: `newton/tests/test_collision_preserve_footprint.py`
- Modify: `newton/tests/test_vehicles_contact.py`, `newton/tests/test_vehicles_metadata.py` (drop flag assertions)

- [ ] **Step 1: Verify it is dead** — `grep -rn "PRESERVE_CONTACT_FOOTPRINT\|preserve_contact_footprint" newton/` and confirm no collision kernel reads it (only flag definition, builder plumbing, metadata setter, tests).
- [ ] **Step 2: Remove** the enum member, the `ShapeConfig.preserve_contact_footprint` field and its docstring, the flag-set in `configure_wheel_solver_contacts`, and the dedicated test file; scrub remaining assertions.
- [ ] **Step 3: Run** `uv run --extra dev -m newton.tests -k "test_vehicles or collision"` — PASS.
- [ ] **Step 4: Commit** — subject: "Remove unwired PRESERVE_CONTACT_FOOTPRINT flag"; body cites the spec (set-but-never-read; `gap=0` obviated it).

---

### Task 11: Docs, CHANGELOG, final verification

**Files:**
- Modify: `CHANGELOG.md`, `docs/` (via `docs/generate_api.py`)

- [ ] **Step 1: CHANGELOG** — in `[Unreleased]`, at random positions within each category: under `Changed`: "Change the wheeled-vehicle tire core to an implicit impulse-budget solve, stable at high friction (μ ≤ 2.5); remove the `WheeledConfig.load_filter` option in favor of direct load latching." Under `Added`: "Add static friction (slope holding), stick/utilization diagnostics, vehicle manifest loading, and an optional relaxation-length transient to `newton.vehicles`." Under `Removed`: "Remove the experimental `newton.wheeled` module in favor of `newton.vehicles`."
- [ ] **Step 2: API docs** — run `uv run docs/generate_api.py`; confirm `newton_vehicles.rst` picks up the manifest functions and `newton_wheeled.rst` is gone from the toctree.
- [ ] **Step 3: Full gate**

```bash
uvx pre-commit run -a
uv run --extra dev -m newton.tests
```
Expected: clean lint, full suite PASS (CPU; run CUDA too if available).

- [ ] **Step 4: Verify the regression tests fail without the fix** — already covered by Task 4 Step 3's mutation check; re-confirm it is documented in that commit message.
- [ ] **Step 5: Commit** — subject: "Update changelog and API docs for the implicit tire core".

---

## Self-review notes

- Spec §4.1-4.3 → Tasks 1-2; §4.4 → Task 3; §4.5 → Tasks 2 (cap removal), 3 (filter removal), 6 (relaxation); §4.6 → Tasks 2, 5; §5 → Tasks 7-10; §6 → Tasks 3, 6, 11; §7 → Tasks 4-5; §9 phasing preserved.
- Tasks 2 Step 1, 3 Step 1, 4 Step 1, 6 Step 1, 7 Step 1 direct the implementer to write full test code against existing in-repo fixtures (exact files and line ranges given); the contracts (assertions, tolerances, protocols) are fully specified here and the fixtures already exist — read them before writing.
- Type consistency: `vec6`, `solve_tire_impulse`, `wheel_effective_mass`, `static_mu_scale`, `impulse_utilization`, `stick`, `trans_long/trans_lat`, `load_vehicle_manifest`/`apply_vehicle_manifest` are used with identical names across tasks.
