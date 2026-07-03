# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical wheel-spin dynamics and tire-wrench injection.

One batched kernel per step (replacing separate drive/tire/moment paths):

1. resolve drive torque from the command (torque-limited speed servo or torque),
2. read the contact patch, compute longitudinal/lateral slip from the wheel
   body motion and the analytical spin,
3. solve the implicit tire impulse against the contact effective mass, project
   onto the friction circle, and accumulate the wrench into ``state.body_f``,
4. integrate the analytical spin ``omega`` from the resolved impulse, with a
   resistive (brake/rolling) term that cannot reverse the wheel,
5. optionally apply the drivetrain reaction torque to the wheel body.
"""

from __future__ import annotations

import warp as wp

from .impulse import solve_tire_impulse
from .metadata import VehicleModelData
from .tire import tire_force

DRIVE_SPEED = wp.constant(0)
DRIVE_TORQUE = wp.constant(1)


@wp.func
def _safe_normalize(v: wp.vec3) -> wp.vec3:
    n = wp.length(v)
    if n > 1.0e-9:
        return v / n
    return wp.vec3(0.0, 0.0, 0.0)


class WheelDynamics:
    """Per-wheel tire/spin parameters, state, and diagnostics (flat device arrays).

    Parameter arrays are set once by the controller from
    :class:`newton.vehicles.WheeledConfig`; ``omega``/``drive_target``/
    ``brake_target`` are the mutable command + state; the remaining arrays are
    per-step diagnostics.

    Args:
        wheel_count: Number of wheels.
        device: Warp device for the arrays.
    """

    def __init__(self, wheel_count: int, device: wp.context.Devicelike | None = None):
        self.wheel_count = wheel_count
        n = max(int(wheel_count), 1)
        self.device = wp.get_device(device)

        def z(dt):
            return wp.zeros(n, dtype=dt, device=self.device)

        # parameters
        self.tire_model = z(wp.int32)
        self.drive_input = z(wp.int32)
        self.c_long = z(wp.float32)
        self.c_lat = z(wp.float32)
        self.mu_override = z(wp.float32)
        self.inertia = z(wp.float32)
        self.damping = z(wp.float32)
        self.rolling_resistance = z(wp.float32)
        self.kp = z(wp.float32)
        self.tau_max = z(wp.float32)
        self.max_speed = z(wp.float32)
        self.brake_max = z(wp.float32)
        self.fallback_load = z(wp.float32)
        self.min_ref = z(wp.float32)
        self.pneumatic_trail = z(wp.float32)
        self.apply_reaction = z(wp.int32)
        self.static_mu_scale = z(wp.float32)
        # command + state
        self.omega = z(wp.float32)
        self.drive_target = z(wp.float32)
        self.brake_target = z(wp.float32)
        # diagnostics
        self.kappa = z(wp.float32)
        self.alpha = z(wp.float32)
        self.f_long = z(wp.float32)
        self.f_lat = z(wp.float32)
        self.mz = z(wp.float32)
        self.normal_load_used = z(wp.float32)
        self.stick = z(wp.int32)
        self.impulse_utilization = z(wp.float32)


def apply_wheel_dynamics(model, state, data: VehicleModelData, patch, dyn: WheelDynamics, dt: float) -> None:
    """Compute tire forces, accumulate the wrench into ``state.body_f``, and
    integrate analytical wheel spin.

    Args:
        model: Finalized model (provides ``body_com`` and ``gravity``).
        state: State; reads ``body_q``/``body_qd``, accumulates into ``body_f``.
        data: Vehicle tables.
        patch: Contact patch state from :func:`update_wheel_contact_patches`.
        dyn: Wheel dynamics parameters/state to read and update.
        dt: Substep timestep [s].
    """
    if data.wheel_count == 0:
        return
    # World gravity vector [m/s^2]: used for the supported-mass estimate and to
    # anticipate gravity's in-substep slip contribution (see the kernel).
    gravity = wp.vec3(*(model.gravity.numpy()[0].tolist()))
    dyn.kappa.zero_()
    dyn.alpha.zero_()
    dyn.f_long.zero_()
    dyn.f_lat.zero_()
    dyn.mz.zero_()
    dyn.normal_load_used.zero_()
    dyn.stick.zero_()
    dyn.impulse_utilization.zero_()
    wp.launch(
        _wheel_dynamics_kernel,
        dim=data.wheel_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.body_inv_mass,
            gravity,
            data.wheel_body,
            data.radius,
            data.forward_axis,
            data.axle_axis,
            data.wheel_center,
            patch.active,
            patch.normal,
            patch.fz,
            patch.friction_seed,
            dyn.tire_model,
            dyn.drive_input,
            dyn.c_long,
            dyn.c_lat,
            dyn.mu_override,
            dyn.inertia,
            dyn.damping,
            dyn.rolling_resistance,
            dyn.kp,
            dyn.tau_max,
            dyn.fallback_load,
            dyn.min_ref,
            dyn.pneumatic_trail,
            dyn.apply_reaction,
            dyn.static_mu_scale,
            dyn.drive_target,
            dyn.brake_target,
            dt,
            dyn.omega,
            dyn.kappa,
            dyn.alpha,
            dyn.f_long,
            dyn.f_lat,
            dyn.mz,
            dyn.normal_load_used,
            dyn.stick,
            dyn.impulse_utilization,
            state.body_f,
        ],
        device=dyn.device,
    )


@wp.kernel
def _wheel_dynamics_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    gravity: wp.vec3,
    wheel_body: wp.array[wp.int32],
    radius: wp.array[wp.float32],
    forward_axis: wp.array[wp.vec3],
    axle_axis: wp.array[wp.vec3],
    wheel_center: wp.array[wp.vec3],
    active: wp.array[wp.bool],
    normal: wp.array[wp.vec3],
    fz_latched: wp.array[wp.float32],
    friction_seed: wp.array[wp.float32],
    tire_model: wp.array[wp.int32],
    drive_input: wp.array[wp.int32],
    c_long: wp.array[wp.float32],
    c_lat: wp.array[wp.float32],
    mu_override: wp.array[wp.float32],
    inertia: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    rolling_resistance: wp.array[wp.float32],
    kp: wp.array[wp.float32],
    tau_max: wp.array[wp.float32],
    fallback_load: wp.array[wp.float32],
    min_ref: wp.array[wp.float32],
    pneumatic_trail: wp.array[wp.float32],
    apply_reaction: wp.array[wp.int32],
    static_mu_scale: wp.array[wp.float32],
    drive_target: wp.array[wp.float32],
    brake_target: wp.array[wp.float32],
    dt: float,
    omega: wp.array[wp.float32],
    out_kappa: wp.array[wp.float32],
    out_alpha: wp.array[wp.float32],
    out_f_long: wp.array[wp.float32],
    out_f_lat: wp.array[wp.float32],
    out_mz: wp.array[wp.float32],
    out_normal_load: wp.array[wp.float32],
    out_stick: wp.array[wp.int32],
    out_utilization: wp.array[wp.float32],
    body_f: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    body = wheel_body[w]
    if body < 0:
        return
    inv_i = 1.0 / wp.max(inertia[w], 1.0e-9)
    om = omega[w]
    r = radius[w]

    # drive torque from command
    if drive_input[w] == DRIVE_SPEED:
        tau_drive = kp[w] * (drive_target[w] - om)
    else:
        tau_drive = drive_target[w]
    tau_drive = wp.clamp(tau_drive, -tau_max[w], tau_max[w])

    handled = int(0)

    if active[w]:
        X_wb = body_q[body]
        rot = wp.transform_get_rotation(X_wb)
        n = normal[w]
        fwd_world = wp.quat_rotate(rot, forward_axis[w])
        fwd_t = _safe_normalize(fwd_world - n * wp.dot(fwd_world, n))
        if wp.length(fwd_t) > 1.0e-6:
            lat_t = _safe_normalize(wp.cross(n, fwd_t))
            twist = body_qd[body]
            v_lin = wp.spatial_top(twist)
            w_ang = wp.spatial_bottom(twist)
            # Apply the tire wrench at the wheel's geometric ground contact (wheel
            # center minus radius along the support normal), not the raw mean of the
            # solver's contact points. Those points are not symmetric about the wheel
            # centerline (and ride up the wheel's sides), so the lateral bias turns the
            # large longitudinal drive force into a spurious yaw torque
            # (tau_z ~ -offset_y * F_x) that makes the car veer under hard acceleration.
            # The geometric contact point is also the textbook tire application point.
            # Apply the tire wrench at the wheel's geometric ground contact: the wheel
            # center (shape center, not the body COM -- they differ when several wheels
            # share one chassis body, e.g. skid-steer) projected to the ground by the
            # radius along the support normal. Using the solver's averaged contact-point
            # center instead biases the point sideways (its points are not symmetric
            # about the wheel centerline), turning the drive force into a spurious yaw
            # torque that makes a sprung car veer under hard acceleration.
            wheel_off = wp.quat_rotate(rot, wheel_center[w] - body_com[body])
            offset = wheel_off - r * n
            v_contact = v_lin + wp.cross(w_ang, offset)
            v_long = wp.dot(v_contact, fwd_t)
            v_lat = wp.dot(v_contact, lat_t)
            ref = wp.max(wp.abs(v_long), wp.max(min_ref[w], 1.0e-4))

            fz = fz_latched[w]
            if fz <= 0.0:
                fz = fallback_load[w]
            if fz > 0.0:
                handled = 1
                mu = mu_override[w]
                if mu < 0.0:
                    mu = friction_seed[w]
                mu = wp.max(mu, 0.0)

                # --- free velocities (drive torque advances spin; brake handled below)
                omega_free = om + dt * (tau_drive - damping[w] * om) * inv_i
                # brake locks the wheel this substep if its impulse capacity
                # exceeds the free spin momentum (conservative: ignores the
                # tire's own spin-up torque)
                locked = brake_target[w] * dt * inv_i >= wp.abs(omega_free)

                # --- slip state and operating-point tire force (for the secant)
                # Anticipate gravity's in-substep tangential contribution (spec
                # §4.2): without it a stick impulse zeroes slip at the start of
                # the substep and gravity re-accelerates the vehicle during it,
                # leaving a structural creep of ~g*sin(theta)*dt/2 on slopes.
                g_long = dt * wp.dot(gravity, fwd_t)
                g_lat = dt * wp.dot(gravity, lat_t)
                if locked:
                    u_long = v_long + g_long  # wheel surface is stationary: slip = ground speed
                else:
                    u_long = v_long - omega_free * r + g_long
                u_lat = v_lat + g_lat
                kappa = -u_long / ref
                alpha = wp.atan2(u_lat, ref)
                f0 = tire_force(tire_model[w], kappa, alpha, fz, mu, c_long[w], c_lat[w], 0.0)

                # secant impulse stiffness dt*C, capped by the linear-regime slope
                k_lin_long = dt * c_long[w] * fz / ref
                k_lin_lat = dt * c_lat[w] * fz / ref
                k_long = wp.min(dt * wp.abs(f0[0]) / wp.max(wp.abs(u_long), 1.0e-6), k_lin_long)
                k_lat = wp.min(dt * wp.abs(f0[1]) / wp.max(wp.abs(u_lat), 1.0e-6), k_lin_lat)

                # --- slip-space effective mass: coupled tangential mass + spin mobility
                # The wheel body's axle joint is FIXED (spin is analytical) and its
                # suspension is prismatic, so in the tangential plane the wheel is
                # rigidly coupled to its corner of the chassis. The free-body
                # Delassus overstated slip mobility ~15x (spurious rotational term
                # included), which made the stick impulse under-correct into a
                # steady, mu-independent slope creep. Use the supported-mass
                # estimate m_c = m_wheel + Fz/|g| instead (spec §4.2, amended).
                g_mag = wp.length(gravity)
                inv_m = body_inv_mass[body]
                if inv_m > 0.0 and g_mag > 1.0e-6:
                    m_c = 1.0 / inv_m + fz / g_mag
                elif inv_m > 0.0:
                    m_c = 1.0 / inv_m
                else:
                    m_c = 1.0e9  # kinematic wheel body: effectively immovable
                a11 = 1.0 / m_c
                a12 = 0.0
                a22 = 1.0 / m_c
                if not locked:
                    a11 = a11 + r * r * inv_i  # spinning wheel adds slip mobility

                budget = mu * fz * dt
                sol = solve_tire_impulse(
                    u_long, u_lat, a11, a12, a22, k_long, k_lat, budget, static_mu_scale[w] * budget
                )
                f_long = sol[0] / dt
                f_lat = sol[1] / dt

                # self-aligning moment from the *resolved* lateral force
                util = sol[5]
                mz = -f_lat * pneumatic_trail[w] * wp.max(1.0 - util, 0.0)

                force_world = fwd_t * f_long + lat_t * f_lat
                torque_world = wp.cross(offset, force_world) + mz * n
                wp.atomic_add(body_f, body, wp.spatial_vector(force_world, torque_world))

                # --- spin update from the resolved impulse
                if locked:
                    om_new = 0.0
                else:
                    om_new = omega_free - sol[0] * r * inv_i
                    # residual resistive torques (brake below capacity + rolling)
                    # brake toward zero without reversing
                    resist = (brake_target[w] + rolling_resistance[w]) * dt * inv_i
                    if om_new > 0.0:
                        om_new = wp.max(om_new - resist, 0.0)
                    elif om_new < 0.0:
                        om_new = wp.min(om_new + resist, 0.0)
                omega[w] = om_new

                out_kappa[w] = kappa
                out_alpha[w] = alpha
                out_f_long[w] = f_long
                out_f_lat[w] = f_lat
                out_mz[w] = mz
                out_normal_load[w] = fz
                out_stick[w] = wp.int32(sol[4])
                out_utilization[w] = util

    if handled == 0:
        # no contact (or no load): drive/damping integrate the free spin;
        # resistive torques (brake + rolling) brake toward zero without reversing
        omega_mid = om + dt * (tau_drive - damping[w] * om) * inv_i
        resist = (brake_target[w] + rolling_resistance[w]) * dt * inv_i
        if omega_mid > 0.0:
            om_new = wp.max(omega_mid - resist, 0.0)
        elif omega_mid < 0.0:
            om_new = wp.min(omega_mid + resist, 0.0)
        else:
            om_new = 0.0
        omega[w] = om_new

    # drivetrain reaction torque on the wheel body about the axle (optional)
    if apply_reaction[w] != 0:
        X_wb = body_q[body]
        axle_world = wp.quat_rotate(wp.transform_get_rotation(X_wb), axle_axis[w])
        brake_sign = float(0.0)
        if om > 0.0:
            brake_sign = 1.0
        elif om < 0.0:
            brake_sign = -1.0
        reaction = axle_world * (-tau_drive + brake_target[w] * brake_sign)
        wp.atomic_add(body_f, body, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), reaction))
