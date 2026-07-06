# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
import newton.vehicles as nv
from newton._src.vehicles.contact import WheelContactPatch
from newton._src.vehicles.wheel import WheelDynamics, apply_wheel_dynamics
from newton.tests.vehicles_test_utils import add_function_test, get_test_devices


def _fill(arr, value):
    host = arr.numpy()
    host[...] = value
    arr.assign(host)


def _setup(device, *, radius=0.2):
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    body = builder.add_body()
    shape = builder.add_shape_cylinder(body, radius=radius, half_height=0.05)
    nv.set_vehicle(builder, 0, drive_mode=int(nv.DriveMode.GENERIC))
    nv.add_wheel(builder, shape=shape, vehicle_id=0, wheel_id=0, radius=radius, width=0.1)
    model = builder.finalize(device=device)
    data = nv.read_vehicle_model_data(model)
    dyn = WheelDynamics(data.wheel_count, device=model.device)
    patch = WheelContactPatch(data.wheel_count, device=model.device)
    state = model.state()
    # identity pose at origin, at rest
    state.body_q.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    state.body_qd.assign(np.zeros((1, 6), dtype=np.float32))
    return model, data, dyn, patch, state


def _activate_patch(patch, *, center, normal, fz):
    patch.active.assign(np.array([True]))
    patch.center.assign(np.array([center], dtype=np.float32))
    patch.normal.assign(np.array([normal], dtype=np.float32))
    patch.fz.assign(np.array([fz], dtype=np.float32))


def test_free_spin_up(test, device):
    model, data, dyn, patch, state = _setup(device)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 2.0)
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    dt = 0.001
    for _ in range(100):
        apply_wheel_dynamics(model, state, data, patch, dyn, dt)
    # omega = tau/I * t = 2/0.01 * 0.1 = 20
    test.assertAlmostEqual(float(dyn.omega.numpy()[0]), 20.0, delta=0.5)


def test_brake_to_zero_no_reverse(test, device):
    model, data, dyn, patch, state = _setup(device)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 0.0)
    _fill(dyn.brake_target, 1.0)
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    dyn.omega.assign(np.array([10.0], dtype=np.float32))
    dt = 0.001
    min_omega = 10.0
    for _ in range(150):
        apply_wheel_dynamics(model, state, data, patch, dyn, dt)
        min_omega = min(min_omega, float(dyn.omega.numpy()[0]))
    test.assertGreaterEqual(min_omega, -1.0e-6)  # never reverses
    test.assertAlmostEqual(float(dyn.omega.numpy()[0]), 0.0, places=5)


def test_tire_reaction_and_injection(test, device):
    model, data, dyn, patch, state = _setup(device)
    body = int(data.wheel_body.numpy()[0])
    _activate_patch(patch, center=(0.0, 0.0, -0.2), normal=(0.0, 0.0, 1.0), fz=100.0)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 5.0)
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    _fill(dyn.c_long, 20.0)
    _fill(dyn.c_lat, 20.0)
    _fill(dyn.mu_override, 1.0)
    _fill(dyn.min_ref, 0.5)
    dyn.omega.assign(np.array([5.0], dtype=np.float32))  # already driving (omega*r > v_long=0)

    state.clear_forces()
    apply_wheel_dynamics(model, state, data, patch, dyn, 0.001)

    test.assertGreater(float(dyn.f_long.numpy()[0]), 0.0)  # forward traction
    bf = state.body_f.numpy()[body]
    test.assertGreater(float(bf[0]), 0.0)  # forward force injected along +X
    # tire reaction torque slows spin-up vs free (free would be 5 + 0.001*5/0.01 = 5.5)
    test.assertLess(float(dyn.omega.numpy()[0]), 5.5)


def test_force_applied_at_ground_contact_not_biased_patch(test, device):
    # The solver's per-wheel contact points are not symmetric about the wheel
    # centerline, so their mean (the reported patch center) is biased sideways.
    # The tire wrench must be applied at the geometric ground contact
    # (center - radius*normal), not that biased center; otherwise the large drive
    # force gets a lateral lever and injects a spurious yaw torque that makes a
    # sprung car veer under hard acceleration. Here the patch center is deliberately
    # offset 3 cm sideways and 2 cm fore, and the injected yaw torque must stay ~0.
    model, data, dyn, patch, state = _setup(device, radius=0.2)
    body = int(data.wheel_body.numpy()[0])
    _activate_patch(patch, center=(0.02, 0.03, -0.2), normal=(0.0, 0.0, 1.0), fz=100.0)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 5.0)
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    _fill(dyn.c_long, 20.0)
    _fill(dyn.c_lat, 20.0)
    _fill(dyn.mu_override, 1.0)
    _fill(dyn.min_ref, 0.5)
    dyn.omega.assign(np.array([5.0], dtype=np.float32))  # driving (omega*r > v_long=0)

    state.clear_forces()
    apply_wheel_dynamics(model, state, data, patch, dyn, 0.001)

    bf = state.body_f.numpy()[body]  # spatial_vector: [force(3), torque(3)]
    fx = float(bf[0])
    yaw_torque = float(bf[5])  # torque about +Z (world up)
    test.assertGreater(fx, 0.0)  # forward traction present
    # offset = -radius*normal has no in-plane component, so no yaw lever; a biased
    # center would give yaw_torque ~ -0.03 * fx (a few N*m here).
    test.assertLess(abs(yaw_torque), 1.0e-3 * max(abs(fx), 1.0))


def test_low_speed_impulse_never_reverses_slip(test, device):
    # At high mu and tiny lateral velocity the tire must stick or reduce slip, never
    # reverse it: the implicit solve either takes the stick solution (impulse that
    # exactly zeroes the slip) or applies a passive, budget-clamped impulse. This
    # replaces the deleted low-speed lateral-cap band-aid regression.
    model, data, dyn, patch, state = _setup(device, radius=0.2)
    fz, mu, v_lat = 40.0, 3.0, 0.01
    _activate_patch(patch, center=(0.0, 0.0, -0.2), normal=(0.0, 0.0, 1.0), fz=fz)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 0.0)
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    _fill(dyn.c_long, 20.0)
    _fill(dyn.c_lat, 200.0)  # very stiff laterally -> explicit force would saturate at mu*Fz
    _fill(dyn.mu_override, mu)  # very high grip: the historic explosion regime
    _fill(dyn.min_ref, 0.5)
    _fill(dyn.static_mu_scale, 1.0)
    # tiny lateral velocity (+Y), no longitudinal motion or spin; body_qd = [lin, ang]
    state.body_qd.assign(np.array([[0.0, v_lat, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    dt = 1.0 / 240.0
    apply_wheel_dynamics(model, state, data, patch, dyn, dt)

    f_lat = float(dyn.f_lat.numpy()[0])
    test.assertLessEqual(f_lat * v_lat, 0.0)  # opposes the lateral slip
    # friction circle: the impulse never exceeds the budget mu*fz*dt
    test.assertLessEqual(abs(f_lat) * dt, mu * fz * dt * (1.0 + 1.0e-5))
    if int(dyn.stick.numpy()[0]) != 1:
        # not stuck: the impulse must not exceed what would reverse the contact's
        # lateral velocity (bounded by the body mass, an upper bound on the
        # contact's lateral effective mass)
        mass = 1.0 / float(model.body_inv_mass.numpy()[0])
        test.assertLessEqual(abs(f_lat) * dt, mass * v_lat * (1.0 + 1.0e-4))


def test_hard_brake_lockup_no_lateral_kick(test, device):
    # A locked wheel sliding straight must produce force opposing motion with a
    # negligible lateral component (no direction chatter at kappa -> -1, where the
    # brush model's theoretical-slip transform amplifies any lateral slip).
    model, data, dyn, patch, state = _setup(device, radius=0.2)
    _activate_patch(patch, center=(0.0, 0.0, -0.2), normal=(0.0, 0.0, 1.0), fz=100.0)
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    _fill(dyn.drive_target, 0.0)
    _fill(dyn.brake_target, 1000.0)  # far above the wheel's spin momentum: hard lock
    _fill(dyn.tau_max, 100.0)
    _fill(dyn.inertia, 0.01)
    _fill(dyn.c_long, 20.0)
    _fill(dyn.c_lat, 20.0)
    _fill(dyn.mu_override, 2.5)
    _fill(dyn.min_ref, 0.5)
    _fill(dyn.static_mu_scale, 1.0)
    # sliding straight ahead at 5 m/s with the wheel not spinning; body_qd = [lin, ang]
    state.body_qd.assign(np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    dt = 1.0 / 240.0
    state.clear_forces()
    apply_wheel_dynamics(model, state, data, patch, dyn, dt)

    f_long = float(dyn.f_long.numpy()[0])
    f_lat = float(dyn.f_lat.numpy()[0])
    test.assertLess(f_long, 0.0)  # opposes the slide
    test.assertLess(abs(f_lat), 0.05 * abs(f_long))  # no lateral kick
    # a hard slide saturates the friction-circle impulse budget
    test.assertGreater(float(dyn.impulse_utilization.numpy()[0]), 0.9)
    # the wheel stays locked (brake capacity exceeds any spin-up)
    test.assertAlmostEqual(float(dyn.omega.numpy()[0]), 0.0, places=6)


class TestWheelDynamics(unittest.TestCase):
    pass


add_function_test(TestWheelDynamics, "test_free_spin_up", test_free_spin_up, devices=get_test_devices())
add_function_test(
    TestWheelDynamics, "test_brake_to_zero_no_reverse", test_brake_to_zero_no_reverse, devices=get_test_devices()
)
add_function_test(
    TestWheelDynamics, "test_tire_reaction_and_injection", test_tire_reaction_and_injection, devices=get_test_devices()
)
add_function_test(
    TestWheelDynamics,
    "test_force_applied_at_ground_contact_not_biased_patch",
    test_force_applied_at_ground_contact_not_biased_patch,
    devices=get_test_devices(),
)
add_function_test(
    TestWheelDynamics,
    "test_low_speed_impulse_never_reverses_slip",
    test_low_speed_impulse_never_reverses_slip,
    devices=get_test_devices(),
)
add_function_test(
    TestWheelDynamics,
    "test_hard_brake_lockup_no_lateral_kick",
    test_hard_brake_lockup_no_lateral_kick,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
