# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSmoke
from newton.tests.unittest_utils import add_function_test, get_test_devices

H = 0.05
CAPACITY = 128 * 512  # cells; keeps the tests cheap on CPU


def _cell_centers(lower, upper):
    """Cell centers [m] of the cells spanning ``lower`` to ``upper`` (in cell indices)."""
    idx = np.stack(
        np.meshgrid(*(np.arange(lo, hi) for lo, hi in zip(lower, upper, strict=True)), indexing="ij"), axis=-1
    )
    return (idx.reshape(-1, 3) + 0.5) * H


def _divergence(solver, centers, device, world=0):
    """Divergence [1/s] at cell centers, from face velocities sampled exactly at face centers."""
    div = np.zeros(len(centers))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = 0.5 * H
        faces = wp.array(np.concatenate([centers + offset, centers - offset]), dtype=wp.vec3, device=device)
        vel = solver.sample_grid(faces, world=world)[1].numpy()[:, axis]
        div += (vel[: len(centers)] - vel[len(centers) :]) / H
    return div


def _add_tracers(builder, count):
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0)] * count,
        vel=[(0.0, 0.0, 0.0)] * count,
        mass=[0.0] * count,
        radius=[0.02] * count,
    )


def test_projection_removes_divergence(test, device):
    """Keep still air still, then project emitter-driven flow to a divergence-free field."""
    builder = newton.ModelBuilder()
    SolverSmoke.register_custom_attributes(builder)
    builder.add_custom_values(**{"smoke:emitter_radius": 0.15})
    model = builder.finalize(device=device)
    state = model.state()
    control = model.control()
    config = SolverSmoke.Config(
        voxel_size=H, max_active_cell_count=CAPACITY, vorticity_confinement=0.0, pressure_iterations=300
    )
    solver = SolverSmoke(model, config)
    centers = _cell_centers((-4, -4, 1), (4, 4, 7))

    solver.step(state, state, control, None, 0.01)
    still = solver.sample_grid(wp.array(centers, dtype=wp.vec3, device=device))[1].numpy()
    test.assertEqual(np.abs(still).max(), 0.0)

    control.smoke.emitter_position.assign([wp.vec3(0.0, 0.0, 0.2)])
    control.smoke.emitter_velocity.assign([wp.vec3(2.0, 0.0, 1.0)])
    control.smoke.emitter_rate.fill_(5.0)
    for _ in range(3):
        solver.step(state, state, control, None, 0.01)
    speed = np.abs(solver.sample_grid(wp.array(centers, dtype=wp.vec3, device=device))[1].numpy()).max()
    test.assertGreater(speed, 0.1)
    test.assertLess(np.abs(_divergence(solver, centers, device)).max(), 1.0e-2 * speed / H)
    solver.check_sparse_grid_rebuild_status()


def test_moving_collider_pushes_air(test, device):
    """Drag the air with a moving collider while letting none flow through it."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.3, 0.3, 0.3)), is_kinematic=True)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    state = model.state()
    state.body_qd.assign(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    config = SolverSmoke.Config(
        voxel_size=H, max_active_cell_count=CAPACITY, vorticity_confinement=0.0, pressure_iterations=300
    )
    solver = SolverSmoke(model, config)
    solver.step(state, state, None, None, 0.01)

    # Cells 4..7 lie inside the box along each axis; the face at x = 0.3 is interior to it.
    probes = wp.array([(0.3, 0.325, 0.325), (0.45, 0.275, 0.275)], dtype=wp.vec3, device=device)
    u = solver.sample_grid(probes)[1].numpy()[:, 0]
    test.assertAlmostEqual(float(u[0]), 1.0, places=5)
    test.assertGreater(float(u[1]), 0.1)

    centers = _cell_centers((1, 1, 1), (11, 11, 11))
    fluid = np.any((centers < 0.2) | (centers > 0.4), axis=1)
    test.assertLess(np.abs(_divergence(solver, centers[fluid], device)).max(), 0.5)


def test_warm_emitter_lifts_smoke_in_its_world(test, device):
    """Lift smoke and tracers from a warm emitter, only in its own world, with grains inside their puffs."""
    builder = newton.ModelBuilder()
    SolverSmoke.register_custom_attributes(builder)
    for _ in range(2):
        builder.begin_world()
        builder.add_custom_values(**{"smoke:emitter_radius": 0.1})
        _add_tracers(builder, 300)
        builder.end_world()
    model = builder.finalize(device=device)
    state = model.state()
    control = model.control()
    control.smoke.emitter_position.assign([wp.vec3(0.0, 0.0, 0.1)] * 2)
    control.smoke.emitter_rate.assign([20.0, 0.0])
    control.smoke.emitter_heat_rate.assign([20.0, 0.0])

    config = SolverSmoke.Config(voxel_size=H, max_active_cell_count=CAPACITY)
    solver = SolverSmoke(model, config)
    grains = solver.sample_render_grains(state, 4)
    dt = 1.0 / 60.0
    for _ in range(45):
        solver.step(state, state, control, None, dt)
        solver.update_render_grains(state, state, grains, dt)
    solver.check_sparse_grid_rebuild_status()

    column = np.array(
        [(x, y, z) for x in (-0.05, 0.0, 0.05) for y in (-0.05, 0.0, 0.05) for z in np.arange(0.025, 1.0, H)]
    )
    points = wp.array(column, dtype=wp.vec3, device=device)
    density = solver.sample_grid(points, world=0)[0].numpy()
    test.assertTrue(np.all(np.isfinite(density)))
    test.assertGreater(float((density * column[:, 2]).sum() / density.sum()), 0.2)
    test.assertEqual(np.abs(solver.sample_grid(points, world=1)[0].numpy()).max(), 0.0)

    age = state.smoke.particle_age.numpy()
    alive = age < config.tracer_lifetime
    test.assertGreater(np.count_nonzero(alive[:300]), 20)
    test.assertEqual(np.count_nonzero(alive[300:]), 0)
    q = state.particle_q.numpy()
    test.assertGreater(q[alive, 2].mean(), 0.15)

    puff = model.particle_radius.numpy()[alive] + config.tracer_spread_rate * age[alive]
    offsets = np.linalg.norm(grains.numpy()[alive] - q[alive, None, :], axis=-1)
    test.assertTrue(np.all(offsets <= puff[:, None] + 1.0e-5))


class TestSolverSmoke(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
add_function_test(
    TestSolverSmoke, "test_projection_removes_divergence", test_projection_removes_divergence, devices=devices
)
add_function_test(TestSolverSmoke, "test_moving_collider_pushes_air", test_moving_collider_pushes_air, devices=devices)
add_function_test(
    TestSolverSmoke,
    "test_warm_emitter_lifts_smoke_in_its_world",
    test_warm_emitter_lifts_smoke_in_its_world,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
