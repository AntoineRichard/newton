# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
import newton.vehicles as nv
from newton._src.vehicles.contact import WheelContactPatch, latch_wheel_loads, update_wheel_contact_patches
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_wheel_on_plane(device, *, radius=0.2, half_width=0.05, start_gap=0.01):
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 0.8
    builder.add_ground_plane(cfg=terrain_cfg)

    # wheel body, cylinder axis rotated from +Z to lie along the world Y axis
    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    wheel_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius + start_gap), wp.quat_identity()))
    wheel_shape = builder.add_shape_cylinder(
        wheel_body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axis_q),
        radius=radius,
        half_height=half_width,
    )
    nv.set_vehicle(builder, 0, drive_mode=int(nv.DriveMode.GENERIC))
    nv.add_wheel(builder, shape=wheel_shape, vehicle_id=0, wheel_id=0, radius=radius, width=2.0 * half_width)

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    return model, wheel_body


def _run(test, device, steps):
    model, wheel_body = _build_wheel_on_plane(device)
    try:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=64, nconmax=32)
    except ImportError as exc:
        test.skipTest(f"MuJoCo not available: {exc}")

    data = nv.read_vehicle_model_data(model)
    patch = WheelContactPatch(data.wheel_count, device=model.device)
    contacts = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        update_wheel_contact_patches(model, state_0, contacts, data, patch)
        solver.step(state_0, state_1, control, contacts, dt)
        solver.update_contacts(contacts, state_0)
        latch_wheel_loads(model, contacts, data, patch)
        state_0, state_1 = state_1, state_0
    return model, wheel_body, data, patch, state_0


def test_patch_geometry(test, device):
    _model, wheel_body, _data, patch, state = _run(test, device, steps=80)
    active = patch.active.numpy()
    normal = patch.normal.numpy()
    center = patch.center.numpy()
    test.assertTrue(bool(active[0]), "wheel patch should be active resting on the plane")
    test.assertGreater(float(normal[0][2]), 0.95)  # support normal ~ +Z
    # Patch center sits in the lower part of the wheel. The un-fixed analytic
    # plane-cylinder helper still emits a spurious equator contact that biases
    # the average upward (~0.066); Phase 1 (preserve_contact_footprint honored)
    # brings this to ~0. Until then, assert it is well below the wheel center.
    test.assertLess(float(center[0][2]), 0.1)
    # wheel did not fall through the plane
    test.assertGreater(float(state.body_q.numpy()[wheel_body][2]), 0.15)


def test_normal_load_matches_weight(test, device):
    model, wheel_body, _data, patch, _state = _run(test, device, steps=200)
    mass = float(model.body_mass.numpy()[wheel_body])
    expected = mass * 9.81
    fz = float(patch.fz.numpy()[0])
    test.assertGreater(fz, 0.0)
    test.assertLess(abs(fz - expected), 0.2 * expected, f"fz={fz} expected~{expected}")


def test_latched_load_zeroes_when_airborne(test, device):
    """fz must not persist across contact loss: a landing wheel starts from the
    fresh measured load, not a stale airborne latch."""
    model, wheel_body = _build_wheel_on_plane(device)
    try:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=64, nconmax=32)
    except ImportError as exc:
        test.skipTest(f"MuJoCo not available: {exc}")

    data = nv.read_vehicle_model_data(model)
    patch = WheelContactPatch(data.wheel_count, device=model.device)
    contacts = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 1.0 / 240.0
    # Settle on the ground so a nonzero load is latched.
    for _ in range(200):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        update_wheel_contact_patches(model, state_0, contacts, data, patch)
        solver.step(state_0, state_1, control, contacts, dt)
        solver.update_contacts(contacts, state_0)
        latch_wheel_loads(model, contacts, data, patch)
        state_0, state_1 = state_1, state_0
    test.assertGreater(float(patch.fz.numpy()[0]), 0.0)

    # Teleport the wheel 1 m up so it loses contact, then run one latch cycle.
    body_q = state_0.body_q.numpy()
    body_q[wheel_body][2] += 1.0
    state_0.body_q.assign(body_q)
    state_0.clear_forces()
    model.collide(state_0, contacts)
    update_wheel_contact_patches(model, state_0, contacts, data, patch)
    solver.step(state_0, state_1, control, contacts, dt)
    solver.update_contacts(contacts, state_0)
    latch_wheel_loads(model, contacts, data, patch)
    test.assertEqual(float(patch.fz.numpy()[0]), 0.0, "airborne wheel must latch fz = 0, not a stale load")


def test_gap_zero_centers_patch(test, device):
    # With configure_solver_contacts (gap=0, condim=1) the spurious analytic
    # plane-cylinder margin contact is gone, so the patch center sits at the
    # ground rather than ~66 mm up the wheel.
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 0.8
    builder.add_ground_plane(cfg=terrain_cfg)
    radius, half_width = 0.2, 0.05
    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    wheel_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius + 0.02), wp.quat_identity()))
    wheel_shape = builder.add_shape_cylinder(
        wheel_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axis_q), radius=radius, half_height=half_width
    )
    nv.set_vehicle(builder, 0, drive_mode=int(nv.DriveMode.GENERIC))
    nv.add_wheel(builder, shape=wheel_shape, vehicle_id=0, wheel_id=0, radius=radius, width=2.0 * half_width)
    model = builder.finalize(device=device)

    vehicles = nv.WheeledVehicles(model)
    vehicles.configure_solver_contacts()  # gap=0 + condim=1
    try:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=64, nconmax=32)
    except ImportError as exc:
        test.skipTest(f"MuJoCo not available: {exc}")

    contacts = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    dt = 1.0 / 240.0
    for _ in range(200):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        vehicles.apply(state_0, contacts, dt)
        solver.step(state_0, state_1, control, contacts, dt)
        solver.update_contacts(contacts, state_0)
        vehicles.latch_loads(contacts)
        state_0, state_1 = state_1, state_0

    test.assertTrue(bool(vehicles.patch.active.numpy()[0]))
    test.assertLess(
        abs(float(vehicles.patch.center.numpy()[0][2])), 0.02, "gap=0 should center the patch at the ground"
    )


class TestVehicleContact(unittest.TestCase):
    pass


add_function_test(TestVehicleContact, "test_patch_geometry", test_patch_geometry, devices=get_test_devices())
add_function_test(
    TestVehicleContact, "test_normal_load_matches_weight", test_normal_load_matches_weight, devices=get_test_devices()
)
add_function_test(
    TestVehicleContact, "test_gap_zero_centers_patch", test_gap_zero_centers_patch, devices=get_test_devices()
)
add_function_test(
    TestVehicleContact,
    "test_latched_load_zeroes_when_airborne",
    test_latched_load_zeroes_when_airborne,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
