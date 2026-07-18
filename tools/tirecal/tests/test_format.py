# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tirecal import format as tcformat


def _random_lap(rng, steps=20, wheels=4, actions=3):
    t1 = steps + 1
    return tcformat.Lap(
        time=np.arange(t1, dtype=np.float64) / 60.0,
        body_q=rng.standard_normal((t1, 7)).astype(np.float32),
        body_qd=rng.standard_normal((t1, 6)).astype(np.float32),
        wheel_omega=rng.standard_normal((t1, wheels)).astype(np.float32),
        controls=rng.standard_normal((steps, actions)).astype(np.float32),
        control_names=("drive", "steer", "brake"),
        meta={"sim_dt": 1.0 / 480.0, "sim_substeps": 8, "control_rate_hz": 60.0, "track_seed": 7},
        sim_state={
            "joint_q": rng.standard_normal((t1, 11)).astype(np.float32),
            "joint_qd": rng.standard_normal((t1, 10)).astype(np.float32),
            "body_q_full": rng.standard_normal((t1, 5, 7)).astype(np.float32),
            "body_qd_full": rng.standard_normal((t1, 5, 6)).astype(np.float32),
            "trans_long": rng.standard_normal((t1, wheels)).astype(np.float32),
            "trans_lat": rng.standard_normal((t1, wheels)).astype(np.float32),
            "patch_fz": rng.standard_normal((t1, wheels)).astype(np.float32),
        },
    )


class TestFormatRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        session = tcformat.Session(
            vehicle={"mass": 4.2, "tire_model": "brush", "tire_params": {"c_long": 20.0}},
            laps=[_random_lap(rng), _random_lap(rng, steps=33)],
            source="newton-sim",
            notes="unit test",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            back = tcformat.read_session(path)
        self.assertEqual(back.source, "newton-sim")
        self.assertEqual(back.vehicle["tire_params"]["c_long"], 20.0)
        self.assertEqual(len(back.laps), 2)
        for a, b in zip(session.laps, back.laps, strict=True):
            np.testing.assert_array_equal(a.body_q, b.body_q)
            np.testing.assert_array_equal(a.controls, b.controls)
            self.assertEqual(a.control_names, b.control_names)
            self.assertEqual(a.meta["track_seed"], b.meta["track_seed"])
            for k, v in a.sim_state.items():
                np.testing.assert_array_equal(v, b.sim_state[k])

    def test_optional_sim_state(self):
        rng = np.random.default_rng(1)
        lap = _random_lap(rng)
        lap.sim_state = None
        session = tcformat.Session(vehicle={}, laps=[lap])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            back = tcformat.read_session(path)
        self.assertIsNone(back.laps[0].sim_state)

    def test_rejects_newer_schema(self):
        import h5py  # noqa: PLC0415

        rng = np.random.default_rng(2)
        session = tcformat.Session(vehicle={}, laps=[_random_lap(rng)])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            tcformat.write_session(path, session)
            with h5py.File(path, "a") as f:
                f.attrs["schema_version"] = tcformat.SCHEMA_VERSION + 1
            with self.assertRaises(ValueError):
                tcformat.read_session(path)

    def test_format_module_is_standalone(self):
        # In-process checks can't work: sibling tests legitimately import newton.
        # A clean interpreter proves importing tirecal.format pulls in no newton module.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tirecal.format, sys; "
                "bad = [m for m in sys.modules if m == 'newton' or m.startswith('newton.')]; "
                "assert not bad, f'tirecal.format imported newton modules: {bad}'",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
