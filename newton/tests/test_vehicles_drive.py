# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.vehicles as nv
from newton._src.vehicles.vehicle import VehicleCommands, update_vehicle_controls
from newton._src.vehicles.wheel import WheelDynamics
from newton.tests.unittest_utils import add_function_test, get_test_devices

_LAYOUT = ((0.5, 0.3, 0), (0.5, -0.3, 0), (-0.5, 0.3, 1), (-0.5, -0.3, 1))


def _add_car(
    builder, *, vehicle_id, wheel_id_start, drive_mode, steer_front, wheelbase=1.0, track=0.6, steer_limit=0.5
):
    nv.set_vehicle(
        builder, vehicle_id, drive_mode=int(drive_mode), wheelbase=wheelbase, track_width=track, steer_limit=steer_limit
    )
    chassis = builder.add_body()
    for i, (x, y, axle_row) in enumerate(_LAYOUT):
        wb = builder.add_body(xform=wp.transform(wp.vec3(x, y, 0.1), wp.quat_identity()))
        s = builder.add_shape_cylinder(wb, radius=0.1, half_height=0.05)
        steerable = steer_front and axle_row == 0
        steer_joint = builder.add_joint_revolute(chassis, wb, axis=(0.0, 0.0, 1.0)) if steerable else -1
        nv.add_wheel(
            builder,
            shape=s,
            vehicle_id=vehicle_id,
            wheel_id=wheel_id_start + i,
            radius=0.1,
            width=0.1,
            driven=True,
            steerable=steerable,
            side=(-1 if y > 0 else 1),  # left = -1, right = +1
            axle_row=axle_row,
            steer_joint=steer_joint,
        )


def _fill(arr, value):
    host = arr.numpy()
    host[...] = value
    arr.assign(host)


def _setup(device, drive_mode, steer_front):
    builder = newton.ModelBuilder()
    nv.register_vehicle_attributes(builder)
    _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=drive_mode, steer_front=steer_front)
    model = builder.finalize(device=device)
    data = nv.read_vehicle_model_data(model)
    dyn = WheelDynamics(data.wheel_count, device=model.device)
    cmd = VehicleCommands(data.vehicle_count, device=model.device)
    control = model.control()
    _fill(dyn.drive_input, int(nv.DriveInput.SPEED))
    _fill(dyn.max_speed, 10.0)
    _fill(dyn.tau_max, 20.0)
    _fill(dyn.brake_max, 20.0)
    return model, data, dyn, cmd, control


def test_ackermann_inner_outer(test, device):
    _model, data, dyn, cmd, control = _setup(device, nv.DriveMode.ACKERMANN, steer_front=True)
    cmd.steer.assign(np.array([1.0], dtype=np.float32))  # full left
    update_vehicle_controls(control, data, dyn, cmd)

    steerable = data.steerable.numpy()
    side = data.side.numpy()
    steer_dof = data.steer_dof.numpy()
    jtp = control.joint_target_pos.numpy()
    left = right = None
    for w in range(data.wheel_count):
        if steerable[w]:
            tgt = float(jtp[int(steer_dof[w])])
            if side[w] == -1:
                left = tgt
            else:
                right = tgt
    test.assertIsNotNone(left)
    test.assertIsNotNone(right)
    # inner wheel (left, toward turn center for a left turn) steers more
    test.assertGreater(abs(left), abs(right))
    # Ackermann condition: cot(outer) - cot(inner) = track / wheelbase
    cot_diff = 1.0 / math.tan(right) - 1.0 / math.tan(left)
    test.assertAlmostEqual(cot_diff, 0.6 / 1.0, delta=0.03)


def test_ackermann_zero_steer(test, device):
    _model, data, dyn, cmd, control = _setup(device, nv.DriveMode.ACKERMANN, steer_front=True)
    cmd.steer.assign(np.array([0.0], dtype=np.float32))
    update_vehicle_controls(control, data, dyn, cmd)
    steerable = data.steerable.numpy()
    steer_dof = data.steer_dof.numpy()
    jtp = control.joint_target_pos.numpy()
    for w in range(data.wheel_count):
        if steerable[w]:
            test.assertAlmostEqual(float(jtp[int(steer_dof[w])]), 0.0, places=5)


def test_skid_steer_differential(test, device):
    _model, data, dyn, cmd, control = _setup(device, nv.DriveMode.SKID_STEER, steer_front=False)
    # pure turn: opposite side speeds
    cmd.drive.assign(np.array([0.0], dtype=np.float32))
    cmd.steer.assign(np.array([1.0], dtype=np.float32))
    update_vehicle_controls(control, data, dyn, cmd)
    side = data.side.numpy()
    targets = dyn.drive_target.numpy()
    for w in range(data.wheel_count):
        if side[w] == -1:
            test.assertLess(float(targets[w]), 0.0)  # left wheels reverse
        else:
            test.assertGreater(float(targets[w]), 0.0)  # right wheels forward

    # straight drive: equal positive speeds
    cmd.drive.assign(np.array([1.0], dtype=np.float32))
    cmd.steer.assign(np.array([0.0], dtype=np.float32))
    update_vehicle_controls(control, data, dyn, cmd)
    targets = dyn.drive_target.numpy()
    test.assertTrue((targets > 0.0).all())
    test.assertAlmostEqual(float(targets.min()), float(targets.max()), places=4)


def test_speed_vs_torque_mode(test, device):
    _model, data, dyn, cmd, control = _setup(device, nv.DriveMode.GENERIC, steer_front=False)
    cmd.drive.assign(np.array([1.0], dtype=np.float32))
    # SPEED: target = max_speed / radius
    update_vehicle_controls(control, data, dyn, cmd)
    radius = data.radius.numpy()
    targets = dyn.drive_target.numpy()
    for w in range(data.wheel_count):
        test.assertAlmostEqual(float(targets[w]), 10.0 / float(radius[w]), delta=1e-2)
    # TORQUE: target = tau_max
    _fill(dyn.drive_input, int(nv.DriveInput.TORQUE))
    update_vehicle_controls(control, data, dyn, cmd)
    targets = dyn.drive_target.numpy()
    test.assertTrue((abs(targets - 20.0) < 1e-3).all())


class TestVehicleDrive(unittest.TestCase):
    pass


add_function_test(
    TestVehicleDrive, "test_ackermann_inner_outer", test_ackermann_inner_outer, devices=get_test_devices()
)
add_function_test(TestVehicleDrive, "test_ackermann_zero_steer", test_ackermann_zero_steer, devices=get_test_devices())
add_function_test(
    TestVehicleDrive, "test_skid_steer_differential", test_skid_steer_differential, devices=get_test_devices()
)
add_function_test(TestVehicleDrive, "test_speed_vs_torque_mode", test_speed_vs_torque_mode, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
