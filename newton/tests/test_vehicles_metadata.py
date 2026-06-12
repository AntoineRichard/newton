# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
import newton.vehicles as nv
from newton._src.geometry.flags import ShapeFlags

# wheel layout: (x, y, axle_row) -> front (row 0) at +x, rear (row 1) at -x
_LAYOUT = ((0.5, 0.3, 0), (0.5, -0.3, 0), (-0.5, 0.3, 1), (-0.5, -0.3, 1))


def _add_car(builder, *, vehicle_id, wheel_id_start, drive_mode, steer_front):
    nv.set_vehicle(builder, vehicle_id, drive_mode=int(drive_mode), wheelbase=1.0, track_width=0.6, steer_limit=0.5)
    chassis = builder.add_body()
    shapes = []
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
            side=(-1 if y > 0 else 1),
            axle_row=axle_row,
            steer_joint=steer_joint,
        )
        shapes.append(s)
    return shapes


class TestVehiclesImport(unittest.TestCase):
    def test_public_import(self):
        self.assertTrue(hasattr(nv, "WheeledVehicles"))
        self.assertTrue(hasattr(nv, "register_vehicle_attributes"))
        self.assertTrue(hasattr(nv, "add_wheel"))
        self.assertEqual(int(nv.DriveMode.ACKERMANN), 1)
        self.assertEqual(int(nv.WheeledVehicles.DriveMode.SKID_STEER), 2)


class TestVehicleMetadata(unittest.TestCase):
    def test_register_and_finalize(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        b = builder.add_body()
        builder.add_shape_cylinder(b, radius=0.1, half_height=0.05)
        model = builder.finalize()
        ns = getattr(model, nv.VEHICLE_NAMESPACE)
        self.assertTrue(hasattr(ns, "is_wheel"))
        self.assertEqual(len(ns.is_wheel.numpy()), model.shape_count)

    def test_add_wheel_sets_attrs_and_flag(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        shapes = _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        ns = getattr(model, nv.VEHICLE_NAMESPACE)
        is_wheel = ns.is_wheel.numpy()
        radius = ns.radius.numpy()
        flags = model.shape_flags.numpy()
        for s in shapes:
            self.assertTrue(bool(is_wheel[s]))
            self.assertAlmostEqual(float(radius[s]), 0.1, places=5)
            self.assertTrue(int(flags[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

    def test_read_flat_tables(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 4)
        self.assertEqual(data.vehicle_count, 1)
        self.assertTrue((data.wheel_vehicle.numpy() == 0).all())
        self.assertTrue((abs(data.radius.numpy() - 0.1) < 1e-5).all())
        self.assertEqual(int(data.drive_mode.numpy()[0]), int(nv.DriveMode.ACKERMANN))
        self.assertAlmostEqual(float(data.wheelbase.numpy()[0]), 1.0, places=5)
        self.assertEqual(int(data.vehicle_wheel_count.numpy()[0]), 4)

    def test_steer_dof_resolution(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        steer_dof = data.steer_dof.numpy()
        steerable = data.steerable.numpy()
        axle_row = data.axle_row.numpy()
        for w in range(data.wheel_count):
            if axle_row[w] == 0:  # front wheels are steerable
                self.assertEqual(int(steerable[w]), 1)
                self.assertGreaterEqual(int(steer_dof[w]), 0)
            else:
                self.assertEqual(int(steer_dof[w]), -1)

    def test_heterogeneous_two_vehicles(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        _add_car(builder, vehicle_id=1, wheel_id_start=4, drive_mode=nv.DriveMode.SKID_STEER, steer_front=False)
        model = builder.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        self.assertEqual(int(data.drive_mode.numpy()[0]), int(nv.DriveMode.ACKERMANN))
        self.assertEqual(int(data.drive_mode.numpy()[1]), int(nv.DriveMode.SKID_STEER))
        self.assertEqual(list(data.wheel_vehicle.numpy()), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_replication_preserves_ids(self):
        sub = newton.ModelBuilder()
        nv.register_vehicle_attributes(sub)
        _add_car(sub, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)

        main = newton.ModelBuilder()
        main.add_builder(sub)
        main.add_builder(sub)
        model = main.finalize()
        data = nv.read_vehicle_model_data(model)
        self.assertEqual(data.wheel_count, 8)
        self.assertEqual(data.vehicle_count, 2)
        self.assertEqual(list(data.wheel_vehicle.numpy()), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_controller_construction(self):
        builder = newton.ModelBuilder()
        nv.register_vehicle_attributes(builder)
        _add_car(builder, vehicle_id=0, wheel_id_start=0, drive_mode=nv.DriveMode.ACKERMANN, steer_front=True)
        model = builder.finalize()
        vehicles = nv.WheeledVehicles(model)
        self.assertEqual(vehicles.wheel_count, 4)
        self.assertEqual(vehicles.vehicle_count, 1)


if __name__ == "__main__":
    unittest.main()
