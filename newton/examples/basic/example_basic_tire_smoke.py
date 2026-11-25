# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Basic Tire Smoke

A small car spins donuts while its rear tires emit smoke, simulated with
:class:`newton.solvers.SolverSmoke`. The car lives in its own model and is
driven kinematically; its shapes are the smoke solver's colliders, so the
chassis and wheels push the air around. Coupling is one way.

Each rear wheel is a smoke emitter. Every frame the example measures how fast
the wheel's contact point slides over the ground and turns that slip into the
emitter's rate, heat, and velocity. Tracer particles carried by the air are
upsampled into many more render grains, drawn as translucent puffs.

Success criteria:
- Rear tires emit smoke as the car spins; the smoke rises, curls, and drifts.
- The smoke fields stay finite and the air velocity stays bounded.

Runs:
- `python -m newton.examples basic_tire_smoke`
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSmoke


@wp.kernel
def donut_car_kernel(
    yaw: float,
    yaw_rate: float,
    pivot: wp.vec3,
    pivot_local: wp.vec3,
    wheel_local: wp.array[wp.vec3],
    wheel_spin: wp.array[float],
    wheel_driven: wp.array[wp.int32],
    driven_spin_rate: float,
    wheel_radius: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Yaw the chassis (body 0) about a point under its front axle; spin the wheels (bodies 1-4)."""
    tid = wp.tid()
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    origin = pivot - wp.quat_rotate(rot, pivot_local)
    ang = wp.vec3(0.0, 0.0, yaw_rate)
    if tid == 0:
        body_q[0] = wp.transform(origin, rot)
        body_qd[0] = wp.spatial_vector(wp.cross(ang, origin - pivot), ang)
        return
    wi = tid - 1
    center = origin + wp.quat_rotate(rot, wheel_local[wi])
    lin = wp.cross(ang, center - pivot)
    axle = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
    spin_rate = driven_spin_rate
    if wheel_driven[wi] == 0:
        # Free wheels roll with the ground.
        spin_rate = wp.dot(lin, wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))) / wheel_radius
    spin = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wheel_spin[wi])
    body_q[tid] = wp.transform(center, rot * spin)
    body_qd[tid] = wp.spatial_vector(lin, ang + spin_rate * axle)


@wp.kernel
def tire_emitters_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    wheel_bodies: wp.array[wp.int32],
    wheel_radius: float,
    slip_threshold: float,
    slip_full: float,
    full_rate: float,
    full_heat_rate: float,
    kick: float,
    emitter_position: wp.array[wp.vec3],
    emitter_velocity: wp.array[wp.vec3],
    emitter_rate: wp.array[float],
    emitter_heat_rate: wp.array[float],
):
    """Turn each wheel's contact-point slip on flat ground (z = 0) into an emitter."""
    e = wp.tid()
    b = wheel_bodies[e]
    X = body_q[b]
    center = wp.transform_get_translation(X)
    contact = wp.vec3(center[0], center[1], 0.0)
    v = wp.spatial_top(body_qd[b]) + wp.cross(
        wp.spatial_bottom(body_qd[b]), contact - wp.transform_point(X, body_com[b])
    )
    slip = wp.length(wp.vec3(v[0], v[1], 0.0))
    touching = center[2] - wheel_radius < 0.25 * wheel_radius
    strength = wp.where(touching, wp.clamp((slip - slip_threshold) / (slip_full - slip_threshold), 0.0, 1.0), 0.0)
    # Sit just outside the tire, on the side its rubber is thrown toward.
    throw = wp.vec3(v[0], v[1], 0.0) / wp.max(slip, 1.0e-6)
    emitter_position[e] = contact + 1.2 * wheel_radius * throw + wp.vec3(0.0, 0.0, 0.4 * wheel_radius)
    # The tire drags fresh smoke along with its surface; hot rubber smoke rises.
    emitter_velocity[e] = kick * wp.vec3(v[0], v[1], 0.2 * slip)
    emitter_rate[e] = strength * full_rate
    emitter_heat_rate[e] = strength * full_heat_rate


@wp.kernel
def grain_style_kernel(
    particle_age: wp.array[float],
    particle_radius: wp.array[float],
    lifetime: float,
    spread_rate: float,
    max_opacity: float,
    grains: wp.array2d[wp.vec3],
    xforms: wp.array2d[wp.transform],
    scales: wp.array2d[wp.vec3],
    colors: wp.array2d[wp.vec3],
    opacities: wp.array2d[float],
):
    pid, g = wp.tid()
    t = particle_age[pid] / lifetime
    if t >= 1.0:
        scales[pid, g] = wp.vec3(0.0)
        opacities[pid, g] = 0.0
        return
    puff = particle_radius[pid] + spread_rate * particle_age[pid]
    xforms[pid, g] = wp.transform(grains[pid, g], wp.quat_identity())
    scales[pid, g] = wp.vec3(0.5 * puff)
    colors[pid, g] = wp.vec3(wp.lerp(0.55, 0.85, t))
    # Fade in quickly, then thin out as the puff grows.
    opacities[pid, g] = max_opacity * wp.min(t * 20.0, 1.0) * (1.0 - t) * (1.0 - t)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer

        # A small, RC-sized car. Chassis is body 0, wheels are bodies 1-4.
        half = (0.2, 0.1, 0.035)
        ride_height = 0.08
        self.wheel_radius = 0.05
        wheel_x, wheel_y = 0.14, 0.125
        wheel_z = self.wheel_radius - ride_height
        wheel_local = [(wheel_x, wheel_y, wheel_z), (wheel_x, -wheel_y, wheel_z)]
        wheel_local += [(-wheel_x, wheel_y, wheel_z), (-wheel_x, -wheel_y, wheel_z)]

        car = newton.ModelBuilder()
        chassis = car.add_body(xform=wp.transform((0.0, 0.0, ride_height)), is_kinematic=True, label="chassis")
        car.add_shape_box(chassis, hx=half[0], hy=half[1], hz=half[2], color=(0.8, 0.15, 0.1))
        wheel_xform = wp.transform(q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi))
        wheels = []
        for i in range(4):
            b = car.add_body(is_kinematic=True, label=f"wheel_{i}")
            car.add_shape_cylinder(b, xform=wheel_xform, radius=self.wheel_radius, half_height=0.02, color=(0.1,) * 3)
            # A light marker so the spin is visible.
            car.add_shape_box(b, xform=wp.transform((0.0, 0.0, 0.03)), hx=0.008, hy=0.021, hz=0.015, color=(0.9,) * 3)
            wheels.append(b)
        car.add_ground_plane()
        self.model = car.finalize()
        self.state = self.model.state()

        # Smoke model: one emitter per rear wheel and a pool of tracer particles.
        config = SolverSmoke.Config(
            voxel_size=0.03,
            max_active_cell_count=1 << 19,
            tracer_rate_scale=15.0,
        )
        tracer_count = 6000
        smoke = newton.ModelBuilder()
        SolverSmoke.register_custom_attributes(smoke)
        for _ in range(2):
            smoke.add_custom_values(**{"smoke:emitter_radius": 2.5 * config.voxel_size})
        smoke.add_particles(
            pos=[(0.0, 0.0, 0.0)] * tracer_count,
            vel=[(0.0, 0.0, 0.0)] * tracer_count,
            mass=[0.0] * tracer_count,
            radius=[config.voxel_size] * tracer_count,
        )
        self.smoke_model = smoke.finalize()
        self.smoke_state = self.smoke_model.state()
        self.smoke_control = self.smoke_model.control()
        # The smoke solver poses the car's shapes from the car's body state.
        self.smoke_state.body_q = self.state.body_q
        self.smoke_state.body_qd = self.state.body_qd

        self.smoke_solver = SolverSmoke(self.smoke_model, config)
        self.smoke_solver.setup_collider(model=self.model)
        self.grains = self.smoke_solver.sample_render_grains(self.smoke_state, 12)
        self.grain_xforms = wp.zeros(self.grains.shape, dtype=wp.transform)
        self.grain_scales = wp.zeros(self.grains.shape, dtype=wp.vec3)
        self.grain_colors = wp.zeros(self.grains.shape, dtype=wp.vec3)
        self.grain_opacities = wp.zeros(self.grains.shape, dtype=float)
        self.grain_materials = wp.full(self.grains.size, wp.vec4(1.0, 0.0, 0.0, 0.0), dtype=wp.vec4)
        self.rear_wheels = wp.array(wheels[2:], dtype=wp.int32)

        # Donut motion.
        self.pivot = wp.vec3(0.0, 0.0, ride_height)
        self.pivot_local = wp.vec3(wheel_x, 0.0, 0.0)
        self.max_yaw_rate = 5.0  # [rad/s]
        self.driven_spin_rate = 6.0 / self.wheel_radius  # rear tire surface at 6 m/s
        self.yaw = 0.0
        self.wheel_local = wp.array(wheel_local, dtype=wp.vec3)
        self.wheel_driven = wp.array([0, 0, 1, 1], dtype=wp.int32)
        self.wheel_spin_host = np.zeros(4, dtype=np.float32)
        self.wheel_spin = wp.array(self.wheel_spin_host, dtype=float)

        self._update_car(0.0, 0.0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-1.5, -1.5, 0.8), pitch=-20.0, yaw=45.0)
        self._log_puff_mesh()

    def _log_puff_mesh(self):
        mesh = newton.Mesh.create_sphere(1.0, num_latitudes=6, num_longitudes=8, compute_inertia=False)
        self.viewer.log_mesh(
            "/smoke_puff",
            wp.array(mesh.vertices, dtype=wp.vec3),
            wp.array(mesh.indices, dtype=wp.int32),
            wp.array(mesh.normals, dtype=wp.vec3),
            hidden=True,
        )

    def _update_car(self, yaw_rate: float, ramp: float):
        wp.launch(
            donut_car_kernel,
            dim=5,
            inputs=[
                self.yaw,
                yaw_rate,
                self.pivot,
                self.pivot_local,
                self.wheel_local,
                self.wheel_spin,
                self.wheel_driven,
                self.driven_spin_rate * ramp,
                self.wheel_radius,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
        )

    def step(self):
        dt = self.frame_dt
        ramp = min(self.sim_time / 1.0, 1.0)
        yaw_rate = self.max_yaw_rate * ramp
        self.yaw += yaw_rate * dt
        # Integrate the wheel spin angles from the angular velocities set last frame.
        q = self.state.body_q.numpy()[1:5]
        qd = self.state.body_qd.numpy()[1:5]
        for i in range(4):
            axle = np.array(wp.quat_rotate(wp.quat(*q[i, 3:]), wp.vec3(0.0, 1.0, 0.0)))
            self.wheel_spin_host[i] += float(np.dot(qd[i, 3:], axle)) * dt
        self.wheel_spin.assign(self.wheel_spin_host)
        self._update_car(yaw_rate, ramp)

        smoke = self.smoke_control.smoke
        wp.launch(
            tire_emitters_kernel,
            dim=len(self.rear_wheels),
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.rear_wheels,
                self.wheel_radius,
                0.3,  # slip [m/s] below which a tire makes no smoke
                2.0,  # slip [m/s] at which emission saturates
                30.0,  # emitter rate [1/s] at full slip
                30.0,  # heat rate [K/s] at full slip
                0.3,  # fraction of the slip velocity given to fresh smoke
            ],
            outputs=[smoke.emitter_position, smoke.emitter_velocity, smoke.emitter_rate, smoke.emitter_heat_rate],
        )
        self.smoke_solver.step(self.smoke_state, self.smoke_state, self.smoke_control, None, dt)
        self.smoke_solver.update_render_grains(self.smoke_state, self.smoke_state, self.grains, dt)
        self.sim_time += dt

    def test_final(self):
        self.smoke_solver.check_sparse_grid_rebuild_status()
        axis = np.arange(-1.0, 1.0, 0.05)
        probes = wp.array([(x, y, z) for x in axis for y in axis for z in (0.05, 0.15, 0.3)], dtype=wp.vec3)
        density, velocity = (a.numpy() for a in self.smoke_solver.sample_grid(probes))
        assert np.all(np.isfinite(density)) and np.all(np.isfinite(velocity)), "smoke fields are not finite"
        assert density.sum() > 1.0, "the tires emitted no smoke"
        assert np.abs(velocity).max() < 50.0, "air velocity blew up"
        age = self.smoke_state.smoke.particle_age.numpy()
        assert np.count_nonzero(age < self.smoke_solver.config.tracer_lifetime) > 0, "no smoke tracers alive"

    def render(self):
        cfg = self.smoke_solver.config
        wp.launch(
            grain_style_kernel,
            dim=self.grains.shape,
            inputs=[
                self.smoke_state.smoke.particle_age,
                self.smoke_model.particle_radius,
                cfg.tracer_lifetime,
                cfg.tracer_spread_rate,
                0.03,
                self.grains,
            ],
            outputs=[self.grain_xforms, self.grain_scales, self.grain_colors, self.grain_opacities],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_instances(
            "/smoke",
            "/smoke_puff",
            self.grain_xforms.flatten(),
            self.grain_scales.flatten(),
            self.grain_colors.flatten(),
            self.grain_materials,
            opacities=self.grain_opacities.flatten(),
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
