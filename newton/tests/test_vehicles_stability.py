# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regime-map acceptance suite for the implicit wheeled-vehicle tire core.

This is the acceptance gate for the tire-core rewrite (design spec §7). The
historic explicit core exploded at high grip (``mu >= 2``) at low speed / under
hard braking: a saturated tire *force* injected onto a LIGHT wheel body
(0.18 kg) flipped sign each substep and pumped the suspension roll/hop mode. A
rigid single-body car cannot reproduce that failure, so every scenario here
runs on a *sprung* fixture -- a heavy chassis carrying four separate light wheel
bodies on prismatic-Z suspension joints -- which is exactly the
light-articulated-wheel structure that drove the instability.

Five scenarios are swept over ``mu in {0.5, 1.0, 2.0, 2.5}`` (the spec's grip
envelope). They probe the regimes where the old core blew up (low-speed steer
reversals, locked-wheel braking, static slope hold) plus two sanity regimes
(bounded steered launch, drift-free straight line). The per-wheel implicit
impulse solve clamps the tangential impulse to ``mu*Fz*dt`` every substep, so no
substep can inject more impulse than the contact can absorb; these tests prove
that fix holds across the envelope.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.vehicles as nv
from newton.tests.test_vehicles_controller import _steer_front_axle
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0
_R = 0.055  # wheel radius [m], matching the rc_car scale
# Corner layout from rc_car.usda suspension anchors: wheelbase 0.324, track 0.296.
_CORNERS = ((0.162, 0.148), (0.162, -0.148), (-0.162, 0.148), (-0.162, -0.148))
_CHASSIS_Z = 0.11  # chassis COM height [m]; wheels rest at z = _R
_CHASSIS_MASS = 2.9  # kg (sprung mass)
_WHEEL_MASS = 0.18  # kg (light wheel body -- the historic instability driver)
# Diagonal inertias from rc_car.usda (chassis PhysicsMassAPI / wheel cylinder).
_CHASSIS_INERTIA = (0.00572267, 0.050838, 0.0547847)
_WHEEL_INERTIA = (0.0001665, 0.00027225, 0.0001665)
_SUSP_KE = 800.0  # suspension drive stiffness [N/m]
_SUSP_KD = 30.0  # suspension drive damping [N·s/m]
_SUSP_LIMIT = 0.025  # suspension travel [m]

MU_SWEEP = (0.5, 1.0, 2.0, 2.5)


def _diag(x, y, z):
    return wp.mat33(x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z)


def _build_sprung_car(device, mu, *, config=None, gravity=None):
    """A sprung car: heavy chassis + four light wheel bodies on prismatic-Z suspension.

    The chassis is a free rigid body (2.9 kg) with collision disabled; each wheel
    is a *separate* 0.18 kg body carrying a rolling cylinder shape and connected
    to the chassis by a prismatic-Z suspension joint (stiffness 800, damping 30,
    +/-0.025 m). There are no axle joints -- wheel spin is integrated
    analytically by the vehicle layer. This light-articulated-wheel structure is
    what pumped the historic roll/hop instability, so it is essential to the gate.

    Args:
        device: Warp device for the finalized model.
        mu: Terrain friction coefficient (seeds the tire via ``friction=-1``).
        config: Optional :class:`newton.vehicles.WheeledConfig`.
        gravity: Optional world gravity vector [m/s²]; tilts to emulate a slope.

    Returns:
        ``(model, chassis, wheel_bodies)``: finalized model, chassis body index,
        and the list of four wheel body indices.
    """
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = mu
    builder.add_ground_plane(cfg=terrain_cfg)

    # Chassis free body (collision off; the wheels carry all ground contact).
    chassis = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, _CHASSIS_Z), wp.quat_identity()),
        mass=_CHASSIS_MASS,
        inertia=_diag(*_CHASSIS_INERTIA),
        lock_inertia=True,
    )
    chassis_cfg = newton.ModelBuilder.ShapeConfig()
    chassis_cfg.has_shape_collision = False
    builder.add_shape_box(chassis, xform=wp.transform(), hx=0.16, hy=0.1, hz=0.03, cfg=chassis_cfg)

    free = builder.add_joint_free(child=chassis)
    axis_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5)
    wheel_bodies = []
    wheel_shapes = []
    susp_joints = []
    for x, y in _CORNERS:
        wheel = builder.add_link(
            xform=wp.transform(wp.vec3(x, y, _R), wp.quat_identity()),
            mass=_WHEEL_MASS,
            inertia=_diag(*_WHEEL_INERTIA),
            lock_inertia=True,
        )
        s = builder.add_shape_cylinder(wheel, xform=wp.transform(wp.vec3(), axis_q), radius=_R, half_height=0.0225)
        # Prismatic-Z suspension: anchor at the wheel attach point in chassis frame.
        j = builder.add_joint_prismatic(
            parent=chassis,
            child=wheel,
            parent_xform=wp.transform(wp.vec3(x, y, _R - _CHASSIS_Z), wp.quat_identity()),
            child_xform=wp.transform(),
            axis=(0.0, 0.0, 1.0),
            target_pos=0.0,
            target_vel=0.0,
            target_ke=_SUSP_KE,
            target_kd=_SUSP_KD,
            limit_lower=-_SUSP_LIMIT,
            limit_upper=_SUSP_LIMIT,
        )
        wheel_bodies.append(wheel)
        wheel_shapes.append(s)
        susp_joints.append(j)
    builder.add_articulation([free, *susp_joints])

    nv.set_vehicle(
        builder, 0, drive_mode=int(nv.DriveMode.GENERIC), wheelbase=0.324, track_width=0.296, steer_limit=0.5
    )
    for i, (s, (x, y)) in enumerate(zip(wheel_shapes, _CORNERS, strict=True)):
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=0,
            wheel_id=i,
            radius=_R,
            width=0.045,
            driven=True,
            side=(-1 if y > 0 else 1),
            axle_row=(0 if x > 0 else 1),
        )

    model = builder.finalize(device=device)
    if gravity is not None:
        n = int(model.gravity.shape[0])
        model.gravity = wp.array([wp.vec3(*gravity)] * n, dtype=wp.vec3, device=device)
    return model, chassis, wheel_bodies


def _roll(quat):
    """Chassis roll angle [rad] about the world/body forward (+X) axis."""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


class _Record:
    """Per-step scalars gathered while a scenario runs."""

    def __init__(self):
        self.wheel_vz = []  # max |vertical velocity| over wheel bodies [m/s]
        self.roll = []  # chassis roll [rad]
        self.util = []  # max impulse utilization over wheels (self-normalized)
        self.circle = []  # max raw friction-circle utilization |F| / (mu*Fz) over wheels
        self.speed = []  # chassis horizontal speed [m/s]
        self.vlat = []  # chassis world-Y (lateral) velocity [m/s]
        self.yaw_rate = []  # chassis yaw rate [rad/s]
        self.stick_all = []  # True if all wheels reported static stick
        self.pos = []  # chassis (x, y, z) [m]
        self.finite = True

    def arr(self, name):
        return np.asarray(getattr(self, name))


def _run_scenario(test, model, chassis, wheel_bodies, vehicles, *, mu, settle, steps, on_step, device):
    """Drive the sprung fixture through ``settle`` warm-up steps then ``steps``
    recorded steps, running ``on_step(i)`` before each recorded substep.

    The substep pipeline mirrors ``test_vehicles_controller._drive``'s inner loop
    exactly (controls -> collide -> tire apply -> solver -> latch). Returns a
    :class:`_Record` of per-step diagnostics used by the scenario assertions.
    """
    try:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=128, nconmax=64)
    except ImportError as exc:
        raise unittest.SkipTest(f"MuJoCo not available: {exc}") from exc
    contacts = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    def step_once():
        nonlocal state_0, state_1
        state_0.clear_forces()
        vehicles.update_controls(control)
        model.collide(state_0, contacts)
        vehicles.apply(state_0, contacts, _DT)
        solver.step(state_0, state_1, control, contacts, _DT)
        solver.update_contacts(contacts, state_0)
        vehicles.latch_loads(contacts)
        state_0, state_1 = state_1, state_0

    vehicles.set_commands(drive=0.0, steer=0.0, brake=0.0)
    for _ in range(settle):
        step_once()

    rec = _Record()
    for i in range(steps):
        on_step(i)
        step_once()

        qd = state_0.body_qd.numpy()
        q = state_0.body_q.numpy()
        util = vehicles.dynamics.impulse_utilization.numpy()
        stick = vehicles.dynamics.stick.numpy()
        f_long = vehicles.dynamics.f_long.numpy()
        f_lat = vehicles.dynamics.f_lat.numpy()
        fz = vehicles.dynamics.normal_load_used.numpy()

        if not (np.isfinite(qd).all() and np.isfinite(q).all()):
            rec.finite = False

        # body_qd is [linear(0:3), angular(3:6)] (see wheel.py spatial_top/bottom).
        rec.wheel_vz.append(max(abs(float(qd[wb][2])) for wb in wheel_bodies))
        v = qd[chassis]
        rec.speed.append(math.hypot(float(v[0]), float(v[1])))
        rec.vlat.append(float(v[1]))
        rec.yaw_rate.append(float(v[5]))
        rec.util.append(float(util.max()))
        # Raw friction-circle utilization: |F| / (mu*Fz), computed from the
        # reported tire force and normal load with the KNOWN terrain mu. Unlike
        # ``impulse_utilization`` (which normalizes by the solver's own budget and
        # so cannot exceed 1 even if the budget is corrupted), this is an
        # independent physical invariant and thus gates a disabled clamp.
        circle = np.hypot(f_long, f_lat) / np.maximum(mu * fz, 1.0e-6)
        rec.circle.append(float(circle.max()))
        rec.stick_all.append(bool((stick == 1).all()))
        cq = q[chassis]
        rec.pos.append((float(cq[0]), float(cq[1]), float(cq[2])))
        rec.roll.append(_roll(cq[3:7]))
    return rec


def test_low_speed_steer_reversals(test, device):
    """Low-speed steering reversals must not pump the roll/hop mode at any grip.

    This is the primary historic failure: at high mu the saturated tire force on
    the light wheel bodies flipped sign each substep and rang the suspension. The
    implicit impulse clamp must keep the wheels planted and the chassis level.
    """
    for mu in MU_SWEEP:
        model, chassis, wheel_bodies = _build_sprung_car(device, mu)
        vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=20.0))
        vehicles.configure_solver_contacts()

        base_fwd = vehicles.data.forward_axis.numpy().copy()

        def on_step(i, v=vehicles, base=base_fwd):
            if i == 0:
                v.set_commands(drive=0.15, steer=0.0)
            if i % 120 == 0:  # flip the baked front steer every 0.5 s
                v.data.forward_axis.assign(base)
                sign = 1.0 if (i // 120) % 2 == 0 else -1.0
                _steer_front_axle(v, math.radians(25.0) * sign)

        rec = _run_scenario(
            test, model, chassis, wheel_bodies, vehicles, mu=mu, settle=120, steps=1200, on_step=on_step, device=device
        )

        test.assertTrue(rec.finite, f"mu={mu}: states went non-finite")
        test.assertLess(
            rec.arr("wheel_vz").max(), 1.0, f"mu={mu}: wheel hop (max |v_z|={rec.arr('wheel_vz').max():.3f})"
        )
        test.assertLess(
            np.abs(rec.arr("roll")).max(), 0.35, f"mu={mu}: chassis roll blew up ({np.abs(rec.arr('roll')).max():.3f})"
        )
        # Raw friction-circle invariant: the applied tire force must never exceed
        # the static budget mu_s*mu*Fz (the stick branch may legitimately use up
        # to static_mu_scale times the kinetic circle). This is what actually
        # gates a disabled impulse clamp (the mutation check in the task brief);
        # the ``impulse_utilization`` diagnostic cannot -- it is clamped to
        # [0, 1] in-kernel by construction, so it is recorded but not asserted.
        circle_bound = vehicles.config.static_mu_scale + 1e-2
        test.assertLessEqual(
            rec.arr("circle").max(),
            circle_bound,
            f"mu={mu}: tire force exceeded the static friction circle ({rec.arr('circle').max():.3f} > {circle_bound:.3f})",
        )


def test_hard_brake_from_top_speed(test, device):
    """Locked-wheel braking from speed must decelerate monotonically, not kick.

    Full throttle for 3 s, then brake=1 (wheels lock). The locked-wheel branch
    must set ``u_long = v_long`` and spend the friction circle on stopping, with
    no lateral kick or yaw spin, coming to rest and staying there.
    """
    for mu in MU_SWEEP:
        model, chassis, wheel_bodies = _build_sprung_car(device, mu)
        vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=30.0))
        vehicles.configure_solver_contacts()

        # One continuous run: 3 s full drive (steps 0..719), then brake to rest
        # (steps 720..). Only the braking window is asserted on.
        _ACCEL = 720

        def on_step(i, v=vehicles, accel=_ACCEL):
            if i == 0:
                v.set_commands(drive=1.0, steer=0.0, brake=0.0)
            elif i == accel:
                v.set_commands(drive=0.0, steer=0.0, brake=1.0)

        rec = _run_scenario(
            test,
            model,
            chassis,
            wheel_bodies,
            vehicles,
            mu=mu,
            settle=120,
            steps=_ACCEL + 720,
            on_step=on_step,
            device=device,
        )

        test.assertTrue(rec.finite, f"mu={mu}: states went non-finite under braking")
        speed = rec.arr("speed")[_ACCEL:]
        vlat = rec.arr("vlat")[_ACCEL:]
        yaw = rec.arr("yaw_rate")[_ACCEL:]
        # Speed must be non-increasing throughout braking, within a tolerance
        # calibrated to the design's structural chatter quantum (spec section 8):
        # near rest the stick solve can overshoot the light wheel body's local
        # response by up to one friction-circle impulse, whose chassis-level
        # velocity quantum is mu*Fz_bar*dt/m_c with Fz_bar the static corner load
        # (M_total*g/4) and m_c = m_wheel + Fz_bar/g the coupled contact mass.
        # For this fixture: Fz_bar = 3.62*9.81/4 = 8.88 N, m_c = 0.18 + 0.905
        # = 1.085 kg, so the quantum is mu*0.0341 m/s; 1.5x margins it. The 0.05
        # floor is the original transient tolerance, binding at low mu.
        fz_bar = (_CHASSIS_MASS + 4.0 * _WHEEL_MASS) * 9.81 / 4.0
        m_c = _WHEEL_MASS + fz_bar / 9.81
        rise_tol = max(0.05, 1.5 * mu * fz_bar * _DT / m_c)
        rises = np.diff(speed)
        test.assertLess(
            rises.max(initial=0.0), rise_tol, f"mu={mu}: speed rose during braking ({rises.max():.3f} > {rise_tol:.3f})"
        )
        test.assertLess(np.abs(vlat).max(), 0.3, f"mu={mu}: lateral kick under braking")
        test.assertLess(np.abs(yaw).max(), 1.0, f"mu={mu}: yaw spin under braking")
        # Locked-wheel braking is where the friction circle actually saturates, so
        # this is the assertion that gates a disabled impulse clamp: without the
        # clamp the tire injects an unbounded stopping impulse (|F| >> mu_s*mu*Fz;
        # the stick branch may legitimately use static_mu_scale times the circle).
        circle_bound = vehicles.config.static_mu_scale + 1e-2
        test.assertLessEqual(
            rec.arr("circle").max(),
            circle_bound,
            f"mu={mu}: braking tire force exceeded the static friction circle ({rec.arr('circle').max():.3f} > {circle_bound:.3f})",
        )
        test.assertLess(speed[-1], 0.05, f"mu={mu}: did not come to rest (final speed {speed[-1]:.3f})")
        # Stays at rest for the last 1 s (240 steps).
        test.assertLess(speed[-240:].max(), 0.05, f"mu={mu}: did not stay at rest")


def test_slope_hold_static_friction(test, device):
    """On a 15 deg incline with brakes locked, static friction must hold the car.

    Gravity is tilted 15 deg (equivalent to the incline) with a flat ground.
    brake=1 from the start. The stick branch must engage: after 5 s the chassis
    must not have crept and all wheels must report static stick.
    """
    g = 9.81
    tilt = math.radians(15.0)
    gravity = (g * math.sin(tilt), 0.0, -g * math.cos(tilt))
    for mu in MU_SWEEP:
        model, chassis, wheel_bodies = _build_sprung_car(device, mu, gravity=gravity)
        vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=20.0))
        vehicles.configure_solver_contacts()

        def on_step(i, v=vehicles):
            if i == 0:
                v.set_commands(drive=0.0, steer=0.0, brake=1.0)

        rec = _run_scenario(
            test, model, chassis, wheel_bodies, vehicles, mu=mu, settle=0, steps=1200, on_step=on_step, device=device
        )

        test.assertTrue(rec.finite, f"mu={mu}: states went non-finite on slope")
        pos = rec.arr("pos")
        disp = math.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1])
        test.assertLess(disp, 0.01, f"mu={mu}: car crept on the slope (disp {disp:.4f} m)")
        test.assertTrue(all(rec.stick_all[-100:]), f"mu={mu}: wheels not sticking at hold")


def test_steered_launch_bounded_yaw(test, device):
    """A traction-sized launch under a held steering lock must corner, not spin.

    Same protocol as ``test_vehicles_controller.test_steered_launch_does_not_spin_out``
    (motor sized near ``mu*Fz*r``, baked 25 deg steer, steady-state yaw judged over
    the second half) but on the sprung fixture and swept over the grip envelope.
    """
    for mu in MU_SWEEP:
        model, chassis, wheel_bodies = _build_sprung_car(device, mu)
        vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=200.0, motor_max_torque=2.0))
        vehicles.configure_solver_contacts()
        _steer_front_axle(vehicles, math.radians(25.0))

        def on_step(i, v=vehicles):
            if i == 0:
                v.set_commands(drive=1.0, steer=0.0)

        rec = _run_scenario(
            test, model, chassis, wheel_bodies, vehicles, mu=mu, settle=60, steps=300, on_step=on_step, device=device
        )

        test.assertTrue(rec.finite, f"mu={mu}: states went non-finite")
        yaw = np.abs(rec.arr("yaw_rate"))
        steady = float(yaw[len(yaw) // 2 :].mean())
        # Grip-limited circular-motion bound: on the Ackermann circle
        # R = wheelbase / tan(steer) the lateral acceleration is yaw^2 * R and
        # cannot exceed mu*g, so yaw <= sqrt(mu*g/R); 1.35x margins transients.
        # A cornering car obeys this at every mu, while a wheelspin pirouette
        # (friction circle spent longitudinally, no lateral grip left) sustains
        # a yaw rate well above it.
        turn_radius = 0.324 / math.tan(math.radians(25.0))  # wheelbase / tan(steer)
        yaw_max = 1.35 * math.sqrt(mu * 9.81 / turn_radius)
        test.assertLess(
            steady, yaw_max, f"mu={mu}: steered launch spun out (steady yaw {steady:.2f} > {yaw_max:.2f} rad/s)"
        )
        pos = rec.arr("pos")
        travel = math.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1])
        test.assertGreater(travel, 0.3, f"mu={mu}: car did not travel (only {travel:.3f} m)")


def test_straight_line_drift_free(test, device):
    """Full-throttle straight-line driving must track straight at every grip."""
    for mu in MU_SWEEP:
        model, chassis, wheel_bodies = _build_sprung_car(device, mu)
        vehicles = nv.WheeledVehicles(model, config=nv.WheeledConfig(max_wheel_speed=30.0))
        vehicles.configure_solver_contacts()

        def on_step(i, v=vehicles):
            if i == 0:
                v.set_commands(drive=1.0, steer=0.0)

        rec = _run_scenario(
            test, model, chassis, wheel_bodies, vehicles, mu=mu, settle=120, steps=720, on_step=on_step, device=device
        )

        test.assertTrue(rec.finite, f"mu={mu}: states went non-finite")
        x, y = rec.pos[-1][0], rec.pos[-1][1]
        test.assertGreater(x, 0.5, f"mu={mu}: car did not travel forward (x={x:.3f})")
        test.assertLess(abs(y), 0.15 * x, f"mu={mu}: car drifted (y={y:.3f}, x={x:.3f})")


class TestWheeledVehiclesStability(unittest.TestCase):
    pass


for _name, _fn in (
    ("test_low_speed_steer_reversals", test_low_speed_steer_reversals),
    ("test_hard_brake_from_top_speed", test_hard_brake_from_top_speed),
    ("test_slope_hold_static_friction", test_slope_hold_static_friction),
    ("test_steered_launch_bounded_yaw", test_steered_launch_bounded_yaw),
    ("test_straight_line_drift_free", test_straight_line_drift_free),
):
    add_function_test(TestWheeledVehiclesStability, _name, _fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
