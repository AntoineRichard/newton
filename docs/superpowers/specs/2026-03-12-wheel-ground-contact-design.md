# Wheel Ground Contact via Repulsion Force

**Goal:** Replace wheel collision shapes with a simple ground repulsion force kernel so the car solver's tire model is the sole source of lateral/longitudinal friction, eliminating the conflict with XPBD contact friction that prevents the car from moving.

**Architecture:** Remove wheel shapes from the collision pipeline (visual-only), add a chassis box for body collisions, and introduce a lightweight ground contact kernel that applies a penalty force when wheels penetrate the ground plane.

## Problem

The car solver computes lateral and longitudinal tire forces via a Pacejka tire model. The wheel bodies also have sphere collision shapes that participate in the XPBD collision pipeline, which applies its own Coulomb friction at the contact point. These two friction sources fight each other — XPBD's contact friction is strong enough to prevent the car from moving.

## Design

### 1. Visual-only wheel shapes

`CarBuilder.add_wheel_collision()` creates sphere shapes with `as_site=True`. This preserves visual rendering of wheels in debug views but removes them from the collision pipeline entirely. No friction, no contact forces from XPBD on wheels.

### 2. Chassis collision box

Add a box collision shape to the chassis body so the car can still collide with walls, other cars, and obstacles. Dimensions auto-derived from the wheel positions bounding box (width from lateral wheel spread, length from front-rear spread, height a reasonable default like 0.5m). This is added during `CarBuilder.build()`.

### 3. Ground contact kernel

New Warp kernel `eval_ground_contact_kernel` in a new file `kernels_ground.py`:

- **Dimensionality:** One thread per wheel.
- **Inputs:** `body_q`, `body_qd`, `wheel_body_idx`, `wheel_radius`, `ground_altitude`, `ground_ke`, `ground_kd`.
- **Output:** `body_f` (atomic add).
- **Logic:**
  ```
  wheel_pos = transform_get_translation(body_q[wheel_body_idx[tid]])
  wheel_vel = spatial_top(body_qd[wheel_body_idx[tid]])
  penetration = (ground_altitude + wheel_radius[tid]) - wheel_pos[1]
  if penetration > 0:
      force_y = ground_ke * penetration + ground_kd * max(-wheel_vel[1], 0.0)
      atomic_add(body_f, wheel_body_idx[tid], spatial_vectorf(0, force_y, 0, 0, 0, 0))
  ```
- `ground_ke` and `ground_kd` default to `2500.0` and `100.0` respectively (matching `ShapeConfig` defaults). Stored as car custom attributes for tunability.

### 4. Solver integration

The ground contact kernel runs in `SolverCarDynamics.step()` after state extraction (step 1) and before suspension/tire/assembly forces. It writes to `body_f` on wheel bodies. XPBD integrates all forces in `body_f` naturally.

No changes to suspension, tire, assembly, or drivetrain kernels. They already derive behavior from body positions and the analytical wheel omega.

### 5. Custom attributes

New per-world car custom attributes:
- `car:ground_altitude` (float, default 0.0) — already exists in `CarDescriptor`
- `car:ground_ke` (float, default 2500.0) — ground contact stiffness [N/m]
- `car:ground_kd` (float, default 100.0) — ground contact damping [N*s/m]

### 6. CarDescriptor changes

- `wheel_collision_shape` parameter: when set to `"sphere"`, creates visual-only site shapes (was creating collidable spheres)
- New `chassis_collision_shape` parameter (default `"box"`): adds a box collision shape to chassis during `build()`
- `ground_altitude` already exists, now also used by the ground contact kernel

## Files changed

- **Create:** `newton/_src/solvers/car/kernels_ground.py` — new ground contact kernel
- **Modify:** `newton/_src/solvers/car/car_builder.py` — visual-only wheel shapes, chassis box, new custom attributes
- **Modify:** `newton/_src/solvers/car/solver_car.py` — launch ground contact kernel in step(), read new custom attributes
- **Modify:** `newton/examples/car/example_car_circles.py` — may need minor adjustments for chassis box
- **Create:** `newton/tests/test_car_ground.py` — tests for ground contact kernel

## Testing

- Unit test: wheel above ground → zero force
- Unit test: wheel at ground → zero force (just touching)
- Unit test: wheel below ground → positive upward force proportional to penetration
- Unit test: damping term active when wheel moving downward
- Integration test: car settles to equilibrium above ground plane
- Existing car tests should continue to pass
