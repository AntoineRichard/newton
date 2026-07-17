# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import newton

    # The recorder drives the MPPI track example, which imports the external
    # track_gen package; skip (not error) when it is absent or incompatible.
    import newton.examples.vehicles.example_vehicle_mppi_track  # noqa: F401

    _HAVE_SIM = True
except Exception:
    _HAVE_SIM = False


@unittest.skipUnless(_HAVE_SIM, "newton with sim extra + track_gen required")
class TestGenerateSmoke(unittest.TestCase):
    def test_record_short_run(self):
        from tirecal import format as tcformat  # noqa: PLC0415
        from tirecal import generate  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "smoke.h5"
            generate.main(
                [
                    "--out",
                    str(out),
                    "--laps",
                    "1",
                    "--track-seed0",
                    "0",
                    "--frames-max",
                    "20",
                    "--test",
                ]
            )
            session = tcformat.read_session(out)
        self.assertEqual(len(session.laps), 1)
        lap = session.laps[0]
        t = lap.controls.shape[0]
        self.assertGreaterEqual(t, 5)
        self.assertEqual(lap.body_q.shape, (t + 1, 7))
        self.assertEqual(lap.body_qd.shape, (t + 1, 6))
        self.assertEqual(lap.wheel_omega.shape[0], t + 1)
        self.assertEqual(lap.control_names, ("drive", "steer", "brake"))
        for key in ("joint_q", "joint_qd", "body_q_full", "body_qd_full", "trans_long", "trans_lat", "patch_fz"):
            self.assertIn(key, lap.sim_state)
            self.assertEqual(lap.sim_state[key].shape[0], t + 1)
        self.assertTrue(np.isfinite(lap.body_q).all())
        self.assertEqual(session.vehicle["tire_model"], "brush")
        self.assertIn("c_long", session.vehicle["tire_params"])
