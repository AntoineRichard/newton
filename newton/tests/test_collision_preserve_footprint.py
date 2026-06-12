# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton._src.geometry.flags import ShapeFlags


class TestPreserveContactFootprintFlag(unittest.TestCase):
    def test_flag_exists_and_is_unique(self):
        self.assertTrue(hasattr(ShapeFlags, "PRESERVE_CONTACT_FOOTPRINT"))
        bit = int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT)
        self.assertNotEqual(bit, 0)
        # single bit (power of two)
        self.assertEqual(bit & (bit - 1), 0)
        # does not collide with any existing flag
        others = (
            int(ShapeFlags.VISIBLE)
            | int(ShapeFlags.COLLIDE_SHAPES)
            | int(ShapeFlags.COLLIDE_PARTICLES)
            | int(ShapeFlags.SITE)
            | int(ShapeFlags.HYDROELASTIC)
        )
        self.assertEqual(bit & others, 0)

    def test_shape_config_sets_flag(self):
        builder = newton.ModelBuilder()
        b = builder.add_body()
        cfg = newton.ModelBuilder.ShapeConfig(preserve_contact_footprint=True)
        s = builder.add_shape_cylinder(b, radius=0.1, half_height=0.05, cfg=cfg)
        model = builder.finalize()
        flags = model.shape_flags.numpy()
        self.assertTrue(int(flags[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

    def test_shape_config_default_off(self):
        builder = newton.ModelBuilder()
        b = builder.add_body()
        s = builder.add_shape_cylinder(b, radius=0.1, half_height=0.05)
        model = builder.finalize()
        self.assertFalse(int(model.shape_flags.numpy()[s]) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

    def test_flags_property_roundtrip(self):
        cfg = newton.ModelBuilder.ShapeConfig(preserve_contact_footprint=True)
        self.assertTrue(int(cfg.flags) & int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT))

        cfg2 = newton.ModelBuilder.ShapeConfig()
        self.assertFalse(cfg2.preserve_contact_footprint)
        cfg2.flags = (
            int(ShapeFlags.PRESERVE_CONTACT_FOOTPRINT) | int(ShapeFlags.VISIBLE) | int(ShapeFlags.COLLIDE_SHAPES)
        )
        self.assertTrue(cfg2.preserve_contact_footprint)


if __name__ == "__main__":
    unittest.main()
