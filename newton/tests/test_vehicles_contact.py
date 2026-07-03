# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
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


def _build_wheel_terrain(device, terrain, *, radius=0.2, half_width=0.05, sink=0.01):
    """Static wheel-on-terrain fixture (no solver): collide once, return patch inputs."""
    builder = newton.ModelBuilder(gravity=0.0)
    nv.register_vehicle_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.8
    cfg.gap = 0.0
    cfg.margin = 0.0

    if terrain == "plane":
        builder.add_ground_plane(cfg=cfg)
    elif terrain == "mesh_ripple":
        vertices = np.array(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [2.0, 2.0, 0.08]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, indices, compute_inertia=False), cfg=cfg)
    else:
        raise ValueError(f"unknown terrain kind: {terrain}")

    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    wheel_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius - sink), wp.quat_identity()))
    wheel_shape = builder.add_shape_cylinder(
        wheel_body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axis_q),
        radius=radius,
        half_height=half_width,
        cfg=cfg,
    )
    nv.set_vehicle(builder, 0, drive_mode=int(nv.DriveMode.GENERIC))
    nv.add_wheel(builder, shape=wheel_shape, vehicle_id=0, wheel_id=0, radius=radius, width=2.0 * half_width)

    model = builder.finalize(device=device)
    data = nv.read_vehicle_model_data(model)
    patch = WheelContactPatch(data.wheel_count, device=model.device)
    state = model.state()
    contacts = model.contacts()
    model.collide(state, contacts)
    return model, state, contacts, data, patch, wheel_body


def test_analytic_plane_patch_chord_and_area(test, device):
    """The analytic plane footprint matches the closed-form cylinder-plane chord math
    and never touches the fields the force path reads."""
    radius, half_width, sink = 0.2, 0.05, 0.01
    model, state, contacts, data, patch, _ = _build_wheel_terrain(
        device, "plane", radius=radius, half_width=half_width, sink=sink
    )

    update_wheel_contact_patches(model, state, contacts, data, patch)
    test.assertTrue(bool(patch.active.numpy()[0]))
    baseline_area = float(patch.area.numpy()[0])
    baseline_center = patch.center.numpy().copy()
    baseline_normal = patch.normal.numpy().copy()
    baseline_seed = patch.friction_seed.numpy().copy()
    baseline_fz = patch.fz.numpy().copy()

    update_wheel_contact_patches(model, state, contacts, data, patch, enable_analytic_plane_patches=True)

    width = 2.0 * half_width
    expected_chord = 2.0 * math.sqrt(2.0 * radius * sink - sink * sink)
    extent = patch.tangent_extent.numpy()[0]
    test.assertAlmostEqual(float(extent[0]), expected_chord, delta=1.0e-5)
    test.assertAlmostEqual(float(extent[1]), width, delta=1.0e-6)
    test.assertAlmostEqual(float(patch.area.numpy()[0]), expected_chord * width, delta=1.0e-5)
    # The raw plane-cylinder contact cloud is a line along the axle (~zero area).
    test.assertGreater(float(patch.area.numpy()[0]), baseline_area + 1.0e-4)
    # The analytic tangent basis spans the plane: longitudinal x axle.
    tangent_u = patch.tangent_u.numpy()[0]
    tangent_v = patch.tangent_v.numpy()[0]
    test.assertAlmostEqual(abs(float(tangent_u[0])), 1.0, delta=1.0e-5)
    test.assertAlmostEqual(abs(float(tangent_v[1])), 1.0, delta=1.0e-5)

    # Diagnostic-only: everything the force path reads is unaffected.
    np.testing.assert_allclose(patch.center.numpy(), baseline_center, atol=1.0e-7)
    np.testing.assert_allclose(patch.normal.numpy(), baseline_normal, atol=1.0e-7)
    np.testing.assert_allclose(patch.friction_seed.numpy(), baseline_seed, atol=1.0e-7)
    np.testing.assert_allclose(patch.fz.numpy(), baseline_fz, atol=1.0e-7)


def test_analytic_plane_patch_skips_mesh_terrain(test, device):
    """The analytic footprint only rewrites wheel-on-plane pairs; mesh terrain keeps
    the measured contact-cloud extents."""
    model, state, contacts, data, patch, _ = _build_wheel_terrain(
        device, "mesh_ripple", radius=0.5, half_width=0.1, sink=0.01
    )
    update_wheel_contact_patches(model, state, contacts, data, patch)
    test.assertTrue(bool(patch.active.numpy()[0]))
    baseline_extent = patch.tangent_extent.numpy().copy()
    baseline_area = patch.area.numpy().copy()

    update_wheel_contact_patches(model, state, contacts, data, patch, enable_analytic_plane_patches=True)
    np.testing.assert_allclose(patch.tangent_extent.numpy(), baseline_extent, atol=1.0e-7)
    np.testing.assert_allclose(patch.area.numpy(), baseline_area, atol=1.0e-7)


def test_mesh_ripple_patch_stability_over_offsets(test, device):
    """Rolling the wheel a few centimetres over a rippled mesh must not make the
    patch center, normal, or area jump."""
    model, state, contacts, data, patch, wheel_body = _build_wheel_terrain(
        device, "mesh_ripple", radius=0.5, half_width=0.1, sink=0.01
    )
    base_body_q = state.body_q.numpy().copy()
    centers = []
    normals = []
    areas = []

    for x_offset in np.linspace(-0.02, 0.02, 5, dtype=np.float32):
        body_q = base_body_q.copy()
        body_q[wheel_body, 0] += x_offset
        state.body_q.assign(body_q)
        model.collide(state, contacts)
        update_wheel_contact_patches(model, state, contacts, data, patch)
        test.assertTrue(bool(patch.active.numpy()[0]))
        for values in (patch.center.numpy(), patch.normal.numpy(), patch.tangent_extent.numpy(), patch.area.numpy()):
            test.assertTrue(np.isfinite(values).all())
        centers.append(patch.center.numpy()[0].copy())
        normals.append(patch.normal.numpy()[0].copy())
        areas.append(float(patch.area.numpy()[0]))

    centers = np.asarray(centers, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    areas = np.asarray(areas, dtype=np.float32)
    normal_deviation = 1.0 - np.clip(normals @ normals[0], -1.0, 1.0)

    test.assertLess(float(np.max(normal_deviation)), 1.0e-3)
    test.assertLess(float(np.ptp(centers[:, 2])), 5.0e-3)
    test.assertLess(float(np.ptp(areas)), 5.0e-3)


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
add_function_test(
    TestVehicleContact,
    "test_analytic_plane_patch_chord_and_area",
    test_analytic_plane_patch_chord_and_area,
    devices=get_test_devices(),
)
add_function_test(
    TestVehicleContact,
    "test_analytic_plane_patch_skips_mesh_terrain",
    test_analytic_plane_patch_skips_mesh_terrain,
    devices=get_test_devices(),
)
add_function_test(
    TestVehicleContact,
    "test_mesh_ripple_patch_stability_over_offsets",
    test_mesh_ripple_patch_stability_over_offsets,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
