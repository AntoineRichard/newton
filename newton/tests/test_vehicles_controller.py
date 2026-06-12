# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.vehicles as nv
from newton.tests.unittest_utils import add_function_test, get_test_devices

_R = 0.08  # wheel radius
# long/low/wide layout so traction does not induce a wheelie on the rigid test car
_CORNERS = ((0.22, 0.16), (0.22, -0.16), (-0.22, 0.16), (-0.22, -0.16))


def _build_car(device, drive_mode):
    """A single free rigid body carrying four wheel cylinders (no suspension)."""
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    builder.add_ground_plane(cfg=terrain_cfg)

    car = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _R), wp.quat_identity()))
    chassis_cfg = newton.ModelBuilder.ShapeConfig()
    chassis_cfg.has_shape_collision = False
    builder.add_shape_box(car, xform=wp.transform(), hx=0.15, hy=0.1, hz=0.03, cfg=chassis_cfg)

    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    nv.set_vehicle(builder, 0, drive_mode=int(drive_mode), wheelbase=0.44, track_width=0.32, steer_limit=0.5)
    for i, (x, y) in enumerate(_CORNERS):
        s = builder.add_shape_cylinder(car, xform=wp.transform(wp.vec3(x, y, 0.0), axis_q), radius=_R, half_height=0.03)
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=0,
            wheel_id=i,
            radius=_R,
            width=0.06,
            driven=True,
            side=(-1 if y > 0 else 1),
            axle_row=(0 if x > 0 else 1),
        )
    model = builder.finalize(device=device)
    return model, car


def _heading(quat):
    # world-frame heading angle of the body +X axis
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    hx = 1.0 - 2.0 * (y * y + z * z)
    hy = 2.0 * (x * y + z * w)
    return math.atan2(hy, hx)


def _drive(model, car, vehicles, *, drive, steer, steps, device, settle=60):
    try:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=128, nconmax=64)
    except ImportError as exc:
        raise unittest.SkipTest(f"MuJoCo not available: {exc}") from exc
    contacts = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    dt = 1.0 / 240.0

    def run(n):
        nonlocal state_0, state_1
        for _ in range(n):
            state_0.clear_forces()
            vehicles.update_controls(control)
            model.collide(state_0, contacts)
            vehicles.apply(state_0, contacts, dt)
            solver.step(state_0, state_1, control, contacts, dt)
            solver.update_contacts(contacts, state_0)
            vehicles.latch_loads(contacts)
            state_0, state_1 = state_1, state_0

    # let the car settle onto its wheels, then apply the command
    vehicles.set_commands(drive=0.0, steer=0.0)
    run(settle)
    vehicles.set_commands(drive=drive, steer=steer)
    run(steps)
    return state_0


def test_drive_forward(test, device):
    model, car = _build_car(device, nv.DriveMode.GENERIC)
    vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=20.0))
    vehicles.configure_solver_contacts()  # tire owns tangential force
    state = _drive(model, car, vehicles, drive=1.0, steer=0.0, steps=240, device=device)
    q = state.body_q.numpy()[car]
    test.assertGreater(float(q[0]), 0.3, "car should drive forward in +X")
    test.assertLess(abs(float(q[1])), 0.2, "car should not drift sideways much")
    test.assertGreater(float(q[2]), _R * 0.5, "car should stay supported above the ground")


def test_skid_steer_rotates_in_place(test, device):
    model, car = _build_car(device, nv.DriveMode.SKID_STEER)
    vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=20.0))
    vehicles.configure_solver_contacts()
    state = _drive(model, car, vehicles, drive=0.0, steer=1.0, steps=240, device=device)
    q = state.body_q.numpy()[car]
    yaw = _heading(q[3:7])
    test.assertGreater(abs(yaw), 0.2, "skid-steer should rotate in place")
    test.assertLess(math.hypot(float(q[0]), float(q[1])), 0.3, "rotation should not translate much")


def test_controller_pipeline_runs(test, device):
    model, car = _build_car(device, nv.DriveMode.GENERIC)
    vehicles = nv.WheeledVehicles(model)
    vehicles.configure_solver_contacts()
    _drive(model, car, vehicles, drive=0.5, steer=0.0, steps=10, device=device)
    # diagnostics are finite
    test.assertTrue(np.isfinite(vehicles.dynamics.omega.numpy()).all())
    test.assertTrue(np.isfinite(vehicles.patch.fz.numpy()).all())


class TestWheeledVehiclesController(unittest.TestCase):
    pass


add_function_test(TestWheeledVehiclesController, "test_drive_forward", test_drive_forward, devices=get_test_devices())
add_function_test(
    TestWheeledVehiclesController,
    "test_skid_steer_rotates_in_place",
    test_skid_steer_rotates_in_place,
    devices=get_test_devices(),
)
add_function_test(
    TestWheeledVehiclesController,
    "test_controller_pipeline_runs",
    test_controller_pipeline_runs,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
