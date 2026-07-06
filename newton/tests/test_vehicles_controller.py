# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.vehicles as nv
from newton.tests.vehicles_test_utils import add_function_test, get_test_devices

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
    # some scrub-induced drift is expected (asymmetric drive/brake slip), but it
    # should rotate rather than drive away
    test.assertLess(math.hypot(float(q[0]), float(q[1])), 0.4, "rotation should not translate much")


def _steer_front_axle(vehicles, steer_rad):
    """Bake a fixed steering lock into the front wheels' tire forward axis.

    The rigid test car has no steering joint, so rotate the front axle's tire
    frame to emulate a held steering angle. The contact geometry is unchanged;
    only the direction the tire pushes rotates.
    """
    fa = vehicles.data.forward_axis.numpy()
    axle_row = vehicles.data.axle_row.numpy()
    c, s = math.cos(steer_rad), math.sin(steer_rad)
    for i in range(len(fa)):
        if int(axle_row[i]) == 0:  # front axle
            x, y, z = float(fa[i][0]), float(fa[i][1]), float(fa[i][2])
            fa[i] = (c * x - s * y, s * x + c * y, z)
    vehicles.data.forward_axis.assign(fa)


def test_steered_launch_does_not_spin_out(test, device):
    """Flooring the throttle from a held steering lock must corner, not pirouette.

    A motor far stronger than the tire's traction limit spins the wheels up well
    past the ground speed, so the whole friction circle is spent on longitudinal
    slip and no lateral grip is left to hold the turn -- the car spins on the spot.
    Sizing the motor near ``mu*Fz*r`` keeps the wheels rolling and the yaw bounded.
    """
    model, car = _build_car(device, nv.DriveMode.GENERIC)
    # traction torque per wheel is ~mu*Fz*r ~ 1.6 N*m here; size the motor near it
    vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=200.0, motor_max_torque=2.0))
    vehicles.configure_solver_contacts()
    _steer_front_axle(vehicles, math.radians(25.0))

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

    vehicles.set_commands(drive=0.0, steer=0.0)
    run(60)  # settle onto the wheels
    prev = _heading(state_0.body_q.numpy()[car][3:7])
    yaw_rates = []
    steps = 300
    vehicles.set_commands(drive=1.0, steer=0.0)  # steering is baked into the tire axis
    for _ in range(steps):
        run(1)
        heading = _heading(state_0.body_q.numpy()[car][3:7])
        yaw_rates.append(abs(((heading - prev + math.pi) % (2.0 * math.pi) - math.pi) / dt))
        prev = heading

    test.assertTrue(np.isfinite(state_0.body_qd.numpy()).all())
    # Judge the sustained (steady-state) yaw rate, not a single-frame peak: a pirouette
    # is a *sustained* high yaw rate, whereas a peak is transient-sensitive and made a
    # peak-based check flaky. A traction-sized motor corners at a steady ~3 rad/s here;
    # an over-strong motor that wheelspins sustains ~9 rad/s.
    steady_yaw_rate = float(np.mean(yaw_rates[steps // 2 :]))
    test.assertLess(steady_yaw_rate, 6.0, f"steered launch spun out (steady yaw rate {steady_yaw_rate:.1f} rad/s)")
    q = state_0.body_q.numpy()[car]
    test.assertGreater(math.hypot(float(q[0]), float(q[1])), 0.3, "car should travel, not pirouette in place")


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
    "test_steered_launch_does_not_spin_out",
    test_steered_launch_does_not_spin_out,
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
