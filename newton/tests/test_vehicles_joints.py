# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
import newton.vehicles as nv


class TestAxleJoints(unittest.TestCase):
    def _car(self, wheel_label=None):
        builder = newton.ModelBuilder()
        chassis = builder.add_link()
        wheel = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.2, 0.0), wp.quat_identity()), label=wheel_label)
        free = builder.add_joint_free(child=chassis)
        axle = builder.add_joint_revolute(parent=chassis, child=wheel, axis=(0.0, 1.0, 0.0), label="axle")
        builder.add_articulation([free, axle])
        return builder, axle

    def test_revolute_axle_converted_to_fixed(self):
        builder, axle = self._car()
        converted = nv.configure_wheel_axle_joints(builder, axle_joint_labels=["axle"])
        self.assertEqual(converted, (axle,))
        model = builder.finalize()
        self.assertEqual(int(model.joint_type.numpy()[axle]), int(newton.JointType.FIXED))
        # the revolute spin DOF is gone: free(6) + fixed(0)
        self.assertEqual(int(model.joint_dof_count), 6)

    def test_resolve_by_wheel_body_label(self):
        builder, axle = self._car(wheel_label="wheel0")
        converted = nv.configure_wheel_axle_joints(builder, wheel_body_labels=["wheel0"])
        self.assertEqual(converted, (axle,))
        model = builder.finalize()
        self.assertEqual(int(model.joint_type.numpy()[axle]), int(newton.JointType.FIXED))
        self.assertEqual(int(model.joint_dof_count), 6)

    def test_missing_label_raises(self):
        builder, _ = self._car()
        with self.assertRaises(ValueError):
            nv.configure_wheel_axle_joints(builder, axle_joint_labels=["nope"])


if __name__ == "__main__":
    unittest.main()
