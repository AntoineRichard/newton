# Wheeled Vehicle Phase 00 Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create deterministic low-poly USDA reference vehicles for the wheeled-vehicle roadmap: one skid-steer Husky-like AGV and one Ackermann RC-car-like fixture.

**Done when:** `newton/examples/assets/wheeled/husky.usda` and `newton/examples/assets/wheeled/rc_car.usda` exist, are text-readable USDA files, and can be loaded by `newton.ModelBuilder.add_usd()` when USD support is available.

**Scope:** Phase 00 creates fixture assets only. It does not create the Phase 0 manifest, inspection helper, tests, report, solver wrapper, contact kernels, tire models, commanded drive controls, examples, or public API symbols.

---

## Inputs

Reference checks for fixture dimensions:

- Clearpath Husky A300 manual: https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a300/user_manual_husky/
- Clearpath Husky A300/A200 comparison: https://clearpathrobotics.com/husky-spec-comparison/
- F1TENTH build docs: https://f1tenth.readthedocs.io/en/stable/getting_started/build_car/index.html
- Traxxas Slash 4X4 Ultimate specs: https://traxxas.com/slash-4x4-vxl-ultimate-68277-4

Use these fixture dimensions:

| Fixture | Quantity | Value | Note |
| --- | --- | ---: | --- |
| Husky | mass | `80.0` kg | Clearpath Husky A300 reference; legacy A200 is `50.0` kg |
| Husky | wheelbase | `0.512` m | front/rear wheel center spacing |
| Husky | track width | `0.566` m | left/right wheel center spacing |
| Husky | wheel radius | `0.1625` m | Clearpath A300 effective tire radius |
| Husky | wheel width | `0.13` m | simple fixture approximation from external width vs track |
| Husky | chassis size | `0.99 x 0.55 x 0.25` m | simplified box, not a vendor mesh |
| RC car | mass | `4.0` kg | F1TENTH-style car with autonomy payload, intentionally approximate |
| RC car | wheel radius | `0.055` m | close to Traxxas Slash/F1TENTH tire diameter `109.5` mm |
| RC car | wheel width | `0.045` m | simple tire-width approximation |
| RC car | wheelbase | `0.324` m | Traxxas Slash/F1TENTH-style reference spacing |
| RC car | track width | `0.296` m | Traxxas Slash/F1TENTH-style reference spacing |
| RC car | center ground clearance | `0.047` m | Traxxas Slash 4X4 Ultimate-style spec, `47` mm |
| RC car | suspension travel | `0.05` m | simplified approximate wheel travel for the fixture, not a vendor spec |
| RC car | suspension drive stiffness | `800.0` N/m | gives roughly `9-12` mm static sag for `0.74-1.0` kg per corner |
| RC car | suspension drive damping | `30.0` N*s/m | about half critical damping for a `1.0` kg corner with `800.0` N/m stiffness |
| RC car | steering drive stiffness | `3.0` N*m/rad | modest centering servo stiffness for front steering links |
| RC car | steering drive damping | `0.2` N*m*s/rad | damped steering response without making the joint rigid |
| RC car | steering drive torque limit | `2.0` N*m | enough to reach the `35` deg steering limit at the chosen stiffness |
| RC car | chassis size | `0.45 x 0.14 x 0.06` m | simplified box, not a vendor mesh |

## Authoring Conventions

- Author ASCII USDA directly. Do not add a generator script unless hand-authored text becomes error-prone.
- Use `metersPerUnit = 1`, `kilogramsPerUnit = 1`, `upAxis = "Z"`, and gravity along negative Z.
- Use X forward, Y left, Z up. Put the nominal ground plane at `z = 0` but do not include terrain geometry in either vehicle asset.
- Use one root prim per asset: `husky` and `rc_car`. Set `defaultPrim` to that root and apply `PhysicsArticulationRootAPI` to the root vehicle Xform.
- Put rigid bodies under `RigidBodies` and joints under `Joints` for predictable inspection.
- Give every body, collision shape, suspension joint, steering joint, and axle joint a stable semantic name. Later Phase 0 inspection may decide which of those labels enter the manifest.
- Use primitive collision geometry only: `Cube` for chassis bodies and `Cylinder` for wheels. Do not use meshes or vendor assets.
- Give all collision primitives `PhysicsCollisionAPI`. Use `PhysicsMeshCollisionAPI` only for `Cube` primitives, matching existing simple USDA assets in this repo.
- Use wheel cylinders with `axis = "Y"`, radius equal to wheel radius, and height equal to wheel width.
- Add wheel axle joints even when the vehicle has no steering or suspension. Otherwise imported wheels can become free bodies instead of constrained wheels.
- Add passive linear spring/damper drives on RC-car suspension joints and passive angular centering drives on front steering joints. Do not add axle drives, rear steering joints, motors, tire parameters, custom `wheeled:*` attributes, or ground-contact material fields in Phase 00.

## Task Steps

- [ ] **Step 1: Create the wheeled asset directory**

Create `newton/examples/assets/wheeled/`.

Run:

```bash
mkdir -p newton/examples/assets/wheeled
```

- [ ] **Step 2: Author `husky.usda`**

Create `newton/examples/assets/wheeled/husky.usda` as a low-poly skid-steer fixture.

Required root structure:

```text
/husky
/husky/RigidBodies/husky_chassis
/husky/RigidBodies/husky_front_left_wheel
/husky/RigidBodies/husky_rear_left_wheel
/husky/RigidBodies/husky_front_right_wheel
/husky/RigidBodies/husky_rear_right_wheel
/husky/Joints/husky_front_left_axle
/husky/Joints/husky_rear_left_axle
/husky/Joints/husky_front_right_axle
/husky/Joints/husky_rear_right_axle
```

Place wheel body origins at the wheel centers:

| Wheel | X [m] | Y [m] | Z [m] |
| --- | ---: | ---: | ---: |
| front left | `0.256` | `0.283` | `0.1625` |
| rear left | `-0.256` | `0.283` | `0.1625` |
| front right | `0.256` | `-0.283` | `0.1625` |
| rear right | `-0.256` | `-0.283` | `0.1625` |

Use these body and joint rules:

- Chassis is one collidable cube labeled `husky_chassis` with size `0.99 x 0.55 x 0.25` m.
- Wheels are four collidable cylinders with radius `0.1625` m, width `0.13` m, and axis `Y`.
- Assign mass as `72.0` kg on the chassis and `2.0` kg on each wheel so the total is `80.0` kg.
- Add one `PhysicsRevoluteJoint` axle per wheel, body0=`husky_chassis`, body1=the wheel body, axis=`Y`.
- Do not add suspension joints, steering joints, drives, or motors.

- [ ] **Step 3: Author `rc_car.usda`**

Create `newton/examples/assets/wheeled/rc_car.usda` as a low-poly Ackermann-style RC fixture.

Required root structure:

```text
/rc_car
/rc_car/RigidBodies/rc_car_chassis
/rc_car/RigidBodies/rc_front_left_suspension_link
/rc_car/RigidBodies/rc_rear_left_suspension_link
/rc_car/RigidBodies/rc_front_right_suspension_link
/rc_car/RigidBodies/rc_rear_right_suspension_link
/rc_car/RigidBodies/rc_front_left_steering_link
/rc_car/RigidBodies/rc_front_right_steering_link
/rc_car/RigidBodies/rc_front_left_wheel
/rc_car/RigidBodies/rc_rear_left_wheel
/rc_car/RigidBodies/rc_front_right_wheel
/rc_car/RigidBodies/rc_rear_right_wheel
/rc_car/Joints/rc_front_left_suspension
/rc_car/Joints/rc_rear_left_suspension
/rc_car/Joints/rc_front_right_suspension
/rc_car/Joints/rc_rear_right_suspension
/rc_car/Joints/rc_front_left_steering
/rc_car/Joints/rc_front_right_steering
/rc_car/Joints/rc_front_left_axle
/rc_car/Joints/rc_rear_left_axle
/rc_car/Joints/rc_front_right_axle
/rc_car/Joints/rc_rear_right_axle
```

Use this nominal layout:

- Chassis center Z is `0.077` m, derived from `0.047` m center ground clearance plus half of the `0.06` m chassis height.
- Wheel centers use `z = 0.055` m so the wheel bottoms touch the nominal `z = 0` ground plane.
- Front wheel centers use `x = 0.162` m; rear wheel centers use `x = -0.162` m.
- Left wheel centers use `y = 0.148` m; right wheel centers use `y = -0.148` m.

Use these body and joint rules:

- Chassis is one collidable cube labeled `rc_car_chassis` with size `0.45 x 0.14 x 0.06` m.
- Wheels are four collidable cylinders with radius `0.055` m, width `0.045` m, and axis `Y`.
- Suspension and steering links are rigid bodies with mass properties but no collision geometry.
- Assign mass as `2.96` kg on the chassis, `0.18` kg on each wheel, `0.07` kg on each suspension link, and `0.02` kg on each front steering link so the total is `4.0` kg.
- Add one `PhysicsPrismaticJoint` suspension per wheel, body0=`rc_car_chassis`, body1=the matching suspension link, axis=`Z`, lower limit `-0.025` m, upper limit `0.025` m, and `PhysicsDriveAPI:linear` with `stiffness = 800.0` N/m, `damping = 30.0` N*s/m, `targetPosition = 0.0`, `targetVelocity = 0.0`, `maxForce = inf`, and `type = "force"`.
- Add front-only `PhysicsRevoluteJoint` steering joints, body0=front suspension link, body1=front steering link, axis=`Z`, lower limit `-35` deg, upper limit `35` deg, and `PhysicsDriveAPI:angular` with imported Newton targets `stiffness = 3.0` N*m/rad and `damping = 0.2` N*m*s/rad. Author the USDA angular gains as `0.0523599` and `0.00349066` respectively because the importer converts angular drive gains from per-degree USD units to per-radian Newton units. Use `targetPosition = 0.0`, `targetVelocity = 0.0`, `maxForce = 2.0` N*m, and `type = "force"`.
- Add one `PhysicsRevoluteJoint` axle per wheel, axis=`Y`. Front axle body0 is the matching steering link; rear axle body0 is the matching suspension link; body1 is the wheel body.
- Do not add rear steering joints, axle drives, motors, differential state, or tire parameters.

- [ ] **Step 4: Verify USDA files are non-empty and text-readable**

Run:

```bash
python - <<'PY'
from pathlib import Path

paths = [
    Path('newton/examples/assets/wheeled/rc_car.usda'),
    Path('newton/examples/assets/wheeled/husky.usda'),
]
for path in paths:
    assert path.exists(), path
    text = path.read_text()
    assert text.startswith('#usda'), path
    assert 'metersPerUnit = 1' in text, path
    assert 'upAxis = "Z"' in text, path
    assert 'PhysicsCollisionAPI' in text, path
PY
```

- [ ] **Step 5: Verify import through `ModelBuilder.add_usd()` when USD is available**

Run:

```bash
uv run --extra dev - <<'PY'
from pathlib import Path

import newton
from newton.tests.unittest_utils import USD_AVAILABLE

if not USD_AVAILABLE:
    print('USD unavailable; skipped add_usd fixture load check')
    raise SystemExit(0)

checks = {
    'husky.usda': {'min_bodies': 5, 'min_joints': 4, 'min_shapes': 5},
    'rc_car.usda': {'min_bodies': 11, 'min_joints': 10, 'min_shapes': 5},
}
asset_dir = Path('newton/examples/assets/wheeled')
for filename, expected in checks.items():
    builder = newton.ModelBuilder()
    builder.add_usd(str(asset_dir / filename), floating=False, enable_self_collisions=False)
    model = builder.finalize()
    assert model.body_count >= expected['min_bodies'], (filename, model.body_count)
    assert model.joint_count >= expected['min_joints'], (filename, model.joint_count)
    assert model.shape_count >= expected['min_shapes'], (filename, model.shape_count)
    print(filename, model.body_count, model.joint_count, model.shape_count)

builder = newton.ModelBuilder()
builder.add_usd(str(asset_dir / 'rc_car.usda'), floating=False, enable_self_collisions=False)
expected_gains = {
    '/rc_car/Joints/rc_front_left_suspension': (800.0, 30.0),
    '/rc_car/Joints/rc_rear_left_suspension': (800.0, 30.0),
    '/rc_car/Joints/rc_front_right_suspension': (800.0, 30.0),
    '/rc_car/Joints/rc_rear_right_suspension': (800.0, 30.0),
    '/rc_car/Joints/rc_front_left_steering': (3.0, 0.2),
    '/rc_car/Joints/rc_front_right_steering': (3.0, 0.2),
}
seen = set()
for joint_index, label in enumerate(builder.joint_label):
    if label in expected_gains:
        dof = builder.joint_qd_start[joint_index]
        expected_ke, expected_kd = expected_gains[label]
        assert abs(builder.joint_target_ke[dof] - expected_ke) < 1.0e-5, (label, builder.joint_target_ke[dof])
        assert abs(builder.joint_target_kd[dof] - expected_kd) < 1.0e-5, (label, builder.joint_target_kd[dof])
        seen.add(label)
assert seen == set(expected_gains), seen
print('rc_car suspension and steering drive gains verified:', len(seen), 'joints')
PY
```

Expected: both files import; all four RC suspension joints report `800.0` N/m target stiffness and `30.0` N*s/m target damping; and both front steering joints report `3.0` N*m/rad target stiffness and `0.2` N*m*s/rad target damping. If USD support is unavailable, the text-readability check remains the Phase 00 verification and Phase 0 tests must mark USD-dependent checks as skipped.

- [ ] **Step 6: Commit the simplified fixture assets**

Run:

```bash
git add newton/examples/assets/wheeled/rc_car.usda \
  newton/examples/assets/wheeled/husky.usda
git commit -m "Add simplified wheeled USDA fixtures"
```

Commit body:

```text
Add deterministic low-poly USDA fixtures for the wheeled vehicle roadmap. The
Husky-like fixture provides constrained axle wheels without suspension or
steering, while the RC-car fixture provides spring-damper suspended wheels with
front steering centering drives and axle joints. These fixtures replace the
earlier dependency on user-provided real robot assets.
```

## Handoff To Phase 0

After this task, Phase 0 should add the manifest, inspection helper, tests, and report. Phase 0 should inspect the actual imported labels instead of assuming that USD prim names and Newton labels are identical.
