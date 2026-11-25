# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSmoke
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _divergence(solver: SolverSmoke) -> np.ndarray:
    u, v, w = (a.numpy() for a in solver.face_velocity)
    h = solver.config.voxel_size
    return (u[1:] - u[:-1] + v[:, 1:] - v[:, :-1] + w[:, :, 1:] - w[:, :, :-1]) / h


def test_projection_removes_divergence(test, device):
    """Keep still air still, and make a random velocity field divergence free."""
    config = SolverSmoke.Config(
        grid_lower=(0.0, 0.0, 0.0),
        grid_upper=(0.5, 0.5, 0.5),
        voxel_size=0.05,
        vorticity_confinement=0.0,
        pressure_iterations=300,
    )
    model = newton.ModelBuilder().finalize(device=device)
    solver = SolverSmoke(model, config)
    state = model.state()

    solver.step(state, state, None, None, 0.01)
    for vel in solver.face_velocity:
        test.assertEqual(np.abs(vel.numpy()).max(), 0.0)

    rng = np.random.default_rng(0)
    for vel in solver.face_velocity:
        vel.assign(rng.uniform(-1.0, 1.0, size=vel.shape).astype(np.float32))
    before = np.abs(_divergence(solver)).max()
    solver.step(state, state, None, None, 0.01)
    after = np.abs(_divergence(solver)).max()
    test.assertGreater(before, 10.0)
    test.assertLess(after, 1.0e-2 * before)
    # The ground face carries no flow.
    test.assertEqual(np.abs(solver.face_velocity[2].numpy()[:, :, 0]).max(), 0.0)


def test_moving_collider_pushes_air(test, device):
    """Drag the air with a moving collider while letting none flow through it."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.3, 0.3, 0.3)), is_kinematic=True)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    state = model.state()
    state.body_qd.assign(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    config = SolverSmoke.Config(
        grid_lower=(0.0, 0.0, 0.0),
        grid_upper=(0.6, 0.6, 0.6),
        voxel_size=0.05,
        vorticity_confinement=0.0,
        pressure_iterations=300,
    )
    solver = SolverSmoke(model, config)
    solver.step(state, state, None, None, 0.01)

    u = solver.face_velocity[0].numpy()
    # Cells 4..7 lie inside the box along each axis; face 6 is interior to the solid.
    np.testing.assert_allclose(u[6, 4:8, 4:8], 1.0, atol=1.0e-6)
    # Air just ahead of the box is pushed forward.
    test.assertGreater(u[9, 5, 5], 0.1)
    fluid = np.ones(solver.grid_shape, dtype=bool)
    fluid[4:8, 4:8, 4:8] = False
    test.assertLess(np.abs(_divergence(solver)[fluid]).max(), 0.5)


def test_warm_emitter_lifts_smoke_and_tracers(test, device):
    """Lift smoke and tracers from a warm emitter, with render grains staying in their puffs."""
    builder = newton.ModelBuilder()
    SolverSmoke.register_custom_attributes(builder)
    builder.add_custom_values(**{"smoke:emitter_radius": 0.1})
    tracer_count = 400
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0)] * tracer_count,
        vel=[(0.0, 0.0, 0.0)] * tracer_count,
        mass=[0.0] * tracer_count,
        radius=[0.02] * tracer_count,
    )
    model = builder.finalize(device=device)
    state = model.state()
    control = model.control()
    control.smoke.emitter_position.assign([wp.vec3(0.0, 0.0, 0.1)])
    control.smoke.emitter_rate.fill_(20.0)
    control.smoke.emitter_heat_rate.fill_(20.0)

    config = SolverSmoke.Config(grid_lower=(-0.4, -0.4, 0.0), grid_upper=(0.4, 0.4, 1.0), voxel_size=0.05)
    solver = SolverSmoke(model, config)
    grains = solver.sample_render_grains(state, 4)
    dt = 1.0 / 60.0
    for _ in range(45):
        solver.step(state, state, control, None, dt)
        solver.update_render_grains(state, state, grains, dt)

    density = solver.density.numpy()
    test.assertTrue(np.all(np.isfinite(density)))
    z = config.grid_lower[2] + (np.arange(solver.grid_shape[2]) + 0.5) * config.voxel_size
    smoke_height = float((density.sum(axis=(0, 1)) * z).sum() / density.sum())
    test.assertGreater(smoke_height, 0.2)

    age = state.smoke.particle_age.numpy()
    alive = age < config.tracer_lifetime
    test.assertGreater(np.count_nonzero(alive), 20)
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
    "test_warm_emitter_lifts_smoke_and_tracers",
    test_warm_emitter_lifts_smoke_and_tracers,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
