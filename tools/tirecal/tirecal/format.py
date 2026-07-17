# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""HDF5 trajectory session format, schema v1. Standalone: numpy + h5py only.

See ``SCHEMA.md`` for field semantics, SI units, and frame conventions. This
module is the contract shared by simulator recordings and future real-vehicle
logs; it must never import newton.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

SCHEMA_VERSION = 1

_LAP_ARRAYS = ("time", "body_q", "body_qd", "wheel_omega", "controls")


@dataclass
class Lap:
    """One recorded lap at control rate.

    State datasets have T+1 rows (samples 0..T); ``controls`` has T rows, where
    ``controls[t]`` drives the transition from state sample ``t`` to ``t+1``.
    """

    time: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    wheel_omega: np.ndarray
    controls: np.ndarray
    control_names: tuple
    meta: dict = field(default_factory=dict)
    sim_state: dict | None = None


@dataclass
class Session:
    vehicle: dict
    laps: list
    source: str = "newton-sim"
    notes: str = ""


def write_session(path, session: Session) -> None:
    """Write a session to ``path``, overwriting any existing file."""
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["source"] = session.source
        f.attrs["notes"] = session.notes
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.create_group("vehicle").attrs["meta"] = json.dumps(session.vehicle)
        laps = f.create_group("laps")
        for i, lap in enumerate(session.laps):
            g = laps.create_group(f"{i:04d}")
            g.attrs["meta"] = json.dumps(lap.meta)
            g.attrs["control_names"] = json.dumps(list(lap.control_names))
            for name in _LAP_ARRAYS:
                g.create_dataset(name, data=np.asarray(getattr(lap, name)))
            if lap.sim_state is not None:
                s = g.create_group("sim_state")
                for k, v in lap.sim_state.items():
                    s.create_dataset(k, data=np.asarray(v))


def read_session(path) -> Session:
    """Read a session; raises ValueError on a newer schema version."""
    with h5py.File(path, "r") as f:
        version = int(f.attrs["schema_version"])
        if version > SCHEMA_VERSION:
            raise ValueError(f"session schema v{version} is newer than supported v{SCHEMA_VERSION}")
        vehicle = json.loads(f["vehicle"].attrs["meta"])
        laps = []
        for key in sorted(f["laps"]):
            g = f["laps"][key]
            sim_state = None
            if "sim_state" in g:
                sim_state = {k: g["sim_state"][k][...] for k in g["sim_state"]}
            laps.append(
                Lap(
                    **{name: g[name][...] for name in _LAP_ARRAYS},
                    control_names=tuple(json.loads(g.attrs["control_names"])),
                    meta=json.loads(g.attrs["meta"]),
                    sim_state=sim_state,
                )
            )
        return Session(vehicle=vehicle, laps=laps, source=str(f.attrs["source"]), notes=str(f.attrs["notes"]))
