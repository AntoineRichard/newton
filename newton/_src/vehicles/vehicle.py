# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-vehicle command mapping for heterogeneous drive modes.

A single batched kernel maps each vehicle's normalized ``(drive, steer, brake)``
command to per-wheel drive/brake targets and per-wheel Ackermann steering joint
targets, branching on the per-vehicle ``drive_mode`` so Ackermann, skid-steer,
and generic vehicles coexist in one model with no host-side branching.
"""

from __future__ import annotations

import warp as wp

from .metadata import VehicleModelData
from .wheel import DRIVE_SPEED, WheelDynamics

DRIVE_MODE_GENERIC = wp.constant(0)
DRIVE_MODE_ACKERMANN = wp.constant(1)
DRIVE_MODE_SKID_STEER = wp.constant(2)


class VehicleCommands:
    """Normalized per-vehicle command buffers (length ``vehicle_count``).

    Args:
        vehicle_count: Number of vehicles.
        device: Warp device for the arrays.

    Attributes:
        drive: Longitudinal command in [-1, 1] per vehicle.
        steer: Steering/turn command in [-1, 1] per vehicle.
        brake: Brake command in [0, 1] per vehicle.
    """

    def __init__(self, vehicle_count: int, device: wp.context.Devicelike | None = None):
        self.vehicle_count = vehicle_count
        n = max(int(vehicle_count), 1)
        self.device = wp.get_device(device)
        self.drive = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.steer = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.brake = wp.zeros(n, dtype=wp.float32, device=self.device)


@wp.func
def _ackermann_angle(delta: float, wheelbase: float, half_track: float, side: float) -> float:
    if wp.abs(delta) < 1.0e-4 or wheelbase < 1.0e-6:
        return delta
    turn_radius = wheelbase / wp.tan(delta)
    denom = turn_radius + side * half_track
    if denom >= 0.0 and denom < 1.0e-6:
        denom = 1.0e-6
    if denom < 0.0 and denom > -1.0e-6:
        denom = -1.0e-6
    return wp.atan(wheelbase / denom)


def update_vehicle_controls(control, data: VehicleModelData, dyn: WheelDynamics, cmd: VehicleCommands) -> None:
    """Map per-vehicle commands to per-wheel drive/brake targets and steering joints.

    Writes ``dyn.drive_target``/``dyn.brake_target`` for every wheel and
    ``control.joint_target_pos`` for steerable Ackermann wheels.

    Args:
        control: Control object whose ``joint_target_pos`` receives steering targets.
        data: Vehicle tables.
        dyn: Wheel dynamics whose drive/brake targets are written.
        cmd: Per-vehicle normalized commands.
    """
    if data.wheel_count == 0:
        return
    wp.launch(
        _command_kernel,
        dim=data.wheel_count,
        inputs=[
            data.wheel_vehicle,
            data.driven,
            data.steerable,
            data.side,
            data.steer_dof,
            data.radius,
            data.drive_mode,
            data.wheelbase,
            data.track_width,
            data.steer_limit,
            dyn.drive_input,
            dyn.max_speed,
            dyn.tau_max,
            dyn.brake_max,
            cmd.drive,
            cmd.steer,
            cmd.brake,
            dyn.drive_target,
            dyn.brake_target,
            control.joint_target_pos,
        ],
        device=dyn.device,
    )


@wp.kernel
def _command_kernel(
    wheel_vehicle: wp.array[wp.int32],
    driven: wp.array[wp.int32],
    steerable: wp.array[wp.int32],
    side: wp.array[wp.int32],
    steer_dof: wp.array[wp.int32],
    radius: wp.array[wp.float32],
    drive_mode: wp.array[wp.int32],
    wheelbase: wp.array[wp.float32],
    track_width: wp.array[wp.float32],
    steer_limit: wp.array[wp.float32],
    drive_input: wp.array[wp.int32],
    max_speed: wp.array[wp.float32],
    tau_max: wp.array[wp.float32],
    brake_max: wp.array[wp.float32],
    cmd_drive: wp.array[wp.float32],
    cmd_steer: wp.array[wp.float32],
    cmd_brake: wp.array[wp.float32],
    drive_target: wp.array[wp.float32],
    brake_target: wp.array[wp.float32],
    joint_target_pos: wp.array[wp.float32],
):
    w = wp.tid()
    v = wheel_vehicle[w]
    mode = drive_mode[v]
    d = cmd_drive[v]
    s = cmd_steer[v]
    brk = cmd_brake[v]

    # drive target
    if driven[w] != 0:
        base = d
        if mode == DRIVE_MODE_SKID_STEER:
            base = wp.clamp(d + wp.float32(side[w]) * s, -1.0, 1.0)
        if drive_input[w] == DRIVE_SPEED:
            drive_target[w] = base * max_speed[w]  # target wheel angular speed [rad/s]
        else:
            drive_target[w] = base * tau_max[w]  # target torque [N·m]
    else:
        drive_target[w] = 0.0

    brake_target[w] = wp.clamp(brk, 0.0, 1.0) * brake_max[w]

    # steering target (Ackermann)
    if steerable[w] != 0 and mode == DRIVE_MODE_ACKERMANN:
        dof = steer_dof[w]
        if dof >= 0:
            delta = wp.clamp(s, -1.0, 1.0) * steer_limit[v]
            joint_target_pos[dof] = _ackermann_angle(delta, wheelbase[v], 0.5 * track_width[v], wp.float32(side[w]))
