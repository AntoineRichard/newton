# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Task 11 training-session merges (train36 + synth8).

Two merges back the Pacejka calibration findings:

``main`` (default) builds ``train36.h5`` from the four track-generator families
(bezier, hull, checkpoint, repulsive). Laps are interleaved across families by
position fraction ``(lap_index + 1) / family_lap_count``, so every prefix and
suffix of the merged lap list mixes families -- in particular the last 4 laps
(the tune.py holdout tail) are the final completed lap of each of the four
families. Within a family, incomplete (frame-capped, mostly stalled) laps are
ordered first so they land in the training split rather than flooding the
holdout tail with near-standstill segments.

``main_synth`` builds ``pacejka-synth8.h5`` from the pacejka-default synthetic
recordings (a single bezier family, ordered by track seed) that back the
synthetic-recovery test.

Standalone: numpy + h5py via ``tirecal.format`` only; does not import newton.

Run from the worktree root:
    python -m tirecal.merge          # -> tirecal-data/train36.h5
    python -m tirecal.merge synth    # -> tirecal-data/pacejka-synth8.h5
"""

import glob
import sys

from tirecal import format as tcformat

FAMILY_FILES = {
    "bezier": sorted(glob.glob("tirecal-data/probe-bezier.h5")) + sorted(glob.glob("tirecal-data/bezier-b*.h5")),
    "hull": sorted(glob.glob("tirecal-data/probe-hull.h5")) + sorted(glob.glob("tirecal-data/hull-h*.h5")),
    "checkpoint": sorted(glob.glob("tirecal-data/probe-checkpoint.h5"))
    + sorted(glob.glob("tirecal-data/checkpoint-c*.h5")),
    "repulsive": sorted(glob.glob("tirecal-data/probe-repulsive.h5"))
    + sorted(glob.glob("tirecal-data/repulsive-r*.h5")),
}
OUT = "tirecal-data/train36.h5"


def main() -> None:
    vehicle = None
    family_laps = {}
    for family, files in FAMILY_FILES.items():
        laps = []
        for path in files:
            session = tcformat.read_session(path)
            if vehicle is None:
                vehicle = session.vehicle
            else:
                # vehicle meta must be identical across recordings; keep the first
                assert session.vehicle["tire_model"] == vehicle["tire_model"], path
                assert session.vehicle["tire_params"] == vehicle["tire_params"], path
                assert session.vehicle["config"] == vehicle["config"], path
            laps.extend(session.laps)
        # Within-family order: incomplete (frame-capped, mostly stalled) laps
        # first so they land in training, then by track seed. The holdout tail
        # (last 4 merged laps) must be completed laps -- a 3600-frame stalled
        # lap in the holdout would flood it with near-standstill segments and
        # artificially soften the 0.2 m position-error gate.
        laps.sort(key=lambda lap: (bool(lap.meta["lap_complete"]), lap.meta["track_seed"]))
        family_laps[family] = laps

    order = []  # (fraction, family, lap)
    for family, laps in family_laps.items():
        n = len(laps)
        for i, lap in enumerate(laps):
            order.append(((i + 1) / n, family, lap))
    order.sort(key=lambda t: (t[0], t[1]))

    merged = [lap for _, _, lap in order]
    for frac, family, lap in order:
        m = lap.meta
        print(
            f"{frac:5.3f} {family:10s} seed={m['track_seed']:3d} frames={m['frames']:4d} complete={m['lap_complete']}"
        )
    tcformat.write_session(OUT, tcformat.Session(vehicle=vehicle, laps=merged))
    tail = [(f, lap.meta["track_seed"]) for _, f, lap in order[-4:]]
    print(f"wrote {len(merged)} laps to {OUT}; holdout tail (last 4): {tail}")


def main_synth() -> None:
    """Merge the pacejka-default synthetic recordings (seed order, one family)."""
    vehicle, laps = None, []
    for path in sorted(glob.glob("tirecal-data/pacejka-s*.h5")):
        session = tcformat.read_session(path)
        vehicle = vehicle or session.vehicle
        assert session.vehicle["tire_params"] == vehicle["tire_params"], path
        laps.extend(session.laps)
    laps.sort(key=lambda lap: lap.meta["track_seed"])
    tcformat.write_session("tirecal-data/pacejka-synth8.h5", tcformat.Session(vehicle=vehicle, laps=laps))
    print(
        f"wrote {len(laps)} laps to tirecal-data/pacejka-synth8.h5 (seeds {[lap.meta['track_seed'] for lap in laps]})"
    )


if __name__ == "__main__":
    main_synth() if "synth" in sys.argv[1:] else main()
