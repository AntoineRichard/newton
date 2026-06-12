# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical wheel-spin dynamics and tire-wrench injection.

One batched kernel per step (replacing separate drive/tire/moment paths):

1. resolve drive torque from the command (torque-limited speed servo or torque),
2. read the contact patch, compute longitudinal/lateral slip from the wheel
   body motion and the analytical spin,
3. evaluate the tire force and accumulate the wrench at the patch into
   ``state.body_f``,
4. integrate the analytical spin ``omega`` semi-implicitly in the slip↔spin
   coupling, with a resistive (brake/rolling) term that cannot reverse the wheel,
5. optionally apply the drivetrain reaction torque to the wheel body.
"""

from __future__ import annotations

import warp as wp

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
        self.fallback_load = z(wp.float32)
        self.min_ref = z(wp.float32)
        self.apply_reaction = z(wp.int32)
        # command + state
        self.omega = z(wp.float32)
        self.drive_target = z(wp.float32)
        self.brake_target = z(wp.float32)
        # diagnostics
        self.kappa = z(wp.float32)
        self.alpha = z(wp.float32)
        self.f_long = z(wp.float32)
        self.f_lat = z(wp.float32)
        self.normal_load_used = z(wp.float32)


def apply_wheel_dynamics(model, state, data: VehicleModelData, patch, dyn: WheelDynamics, dt: float) -> None:
    """Compute tire forces, accumulate the wrench into ``state.body_f``, and
    integrate analytical wheel spin.

    Args:
        model: Finalized model (provides ``body_com``).
        state: State; reads ``body_q``/``body_qd``, accumulates into ``body_f``.
        data: Vehicle tables.
        patch: Contact patch state from :func:`update_wheel_contact_patches`.
        dyn: Wheel dynamics parameters/state to read and update.
        dt: Substep timestep [s].
    """
    if data.wheel_count == 0:
        return
    dyn.kappa.zero_()
    dyn.alpha.zero_()
    dyn.f_long.zero_()
    dyn.f_lat.zero_()
    dyn.normal_load_used.zero_()
    wp.launch(
        _wheel_dynamics_kernel,
        dim=data.wheel_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            data.wheel_body,
            data.radius,
            data.forward_axis,
            data.axle_axis,
            patch.active,
            patch.center,
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
            dyn.apply_reaction,
            dyn.drive_target,
            dyn.brake_target,
            dt,
            dyn.omega,
            dyn.kappa,
            dyn.alpha,
            dyn.f_long,
            dyn.f_lat,
            dyn.normal_load_used,
            state.body_f,
        ],
        device=dyn.device,
    )


@wp.kernel
def _wheel_dynamics_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    wheel_body: wp.array[wp.int32],
    radius: wp.array[wp.float32],
    forward_axis: wp.array[wp.vec3],
    axle_axis: wp.array[wp.vec3],
    active: wp.array[wp.bool],
    center: wp.array[wp.vec3],
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
    apply_reaction: wp.array[wp.int32],
    drive_target: wp.array[wp.float32],
    brake_target: wp.array[wp.float32],
    dt: float,
    omega: wp.array[wp.float32],
    out_kappa: wp.array[wp.float32],
    out_alpha: wp.array[wp.float32],
    out_f_long: wp.array[wp.float32],
    out_f_lat: wp.array[wp.float32],
    out_normal_load: wp.array[wp.float32],
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

    f_long = float(0.0)
    denom = inertia[w]

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
            com_world = wp.transform_point(X_wb, body_com[body])
            offset = center[w] - com_world
            v_contact = v_lin + wp.cross(w_ang, offset)
            v_long = wp.dot(v_contact, fwd_t)
            v_lat = wp.dot(v_contact, lat_t)

            ref = wp.max(wp.abs(v_long), wp.max(min_ref[w], 1.0e-4))
            kappa = (om * r - v_long) / ref
            alpha = wp.atan2(v_lat, ref)

            fz = fz_latched[w]
            if fz <= 0.0:
                fz = fallback_load[w]
            if fz > 0.0:
                mu = mu_override[w]
                if mu < 0.0:
                    mu = friction_seed[w]
                if mu < 0.0:
                    mu = 0.0
                f = tire_force(tire_model[w], kappa, alpha, fz, mu, c_long[w], c_lat[w])
                f_long = f[0]
                f_lat = f[1]
                force_world = fwd_t * f_long + lat_t * f_lat
                torque_world = wp.cross(offset, force_world)
                wp.atomic_add(body_f, body, wp.spatial_vector(force_world, torque_world))
                denom = inertia[w] + dt * c_long[w] * r * r / ref
                out_kappa[w] = kappa
                out_alpha[w] = alpha
                out_f_long[w] = f_long
                out_f_lat[w] = f_lat
                out_normal_load[w] = fz

    # analytical spin: active/driving torques integrate; resistive torques brake toward zero
    tau_active = tau_drive - f_long * r - damping[w] * om
    omega_mid = om + dt * tau_active / wp.max(denom, 1.0e-9)
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
