# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Final validation gate: closed-loop MPPI laps, brush vs tuned Pacejka.

Runs the MPPI track example headless twice on the same track seed -- once on
brush, once on Pacejka with parameters from a tune report -- and compares lap
progress and off-track counts.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import seed_fit


def run_laps(tire_model, mf_params, track_seed, frames, num_samples, track_generator):
    """Run one closed-loop session; returns hero progress and off-track counts.

    Args:
        tire_model: ``"brush"`` (the example's native model) or ``"pacejka"``.
        mf_params: Magic Formula coefficients by name (``seed_fit.PARAM_NAMES``);
            only used when ``tire_model == "pacejka"``.
        track_seed: Track generation seed (use one outside the training set).
        frames: Closed-loop frames to simulate (60 fps).
        num_samples: MPPI samples (planner quality).
        track_generator: track_gen generator family.
    """
    import newton  # noqa: PLC0415 (deferred: keep newton optional at import time)
    from newton.examples.vehicles.example_vehicle_mppi_track import Example  # noqa: PLC0415

    from .generate import _example_args  # noqa: PLC0415

    viewer = newton.viewer.ViewerNull(num_frames=frames)
    args = _example_args(Example.create_parser(), num_samples, track_seed, "none", track_generator, False)
    ex = Example(viewer, args)
    if tire_model == "pacejka":
        import newton.vehicles as nv  # noqa: PLC0415

        dyn = ex.vehicles.dynamics
        host = dyn.tire_model.numpy()
        host[...] = int(nv.TireModel.PACEJKA)
        dyn.tire_model.assign(host)
        theta = np.array([mf_params[k] for k in seed_fit.PARAM_NAMES], dtype=np.float64)
        dyn.mf_params.assign(np.tile(theta, (max(dyn.wheel_count, 1), 1)).astype(np.float32))
    elif tire_model != "brush":
        raise ValueError(f"unsupported tire_model {tire_model!r}; choose 'brush' or 'pacejka'")
    oob_frames = 0
    for _ in range(frames):
        ex.step()
        oob_frames += int(ex.contact.oob.numpy()[0])
    meters = float(ex.total_s.numpy()[0])
    return {"meters": meters, "laps": meters / ex.track_len, "oob_frames": oob_frames}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, help="tune report JSON with best_params")
    p.add_argument("--track-seed", type=int, default=100, help="use a seed outside the training set")
    p.add_argument("--track-generator", default="bezier", help="track_gen generator family")
    p.add_argument("--frames", type=int, default=3600)
    p.add_argument("--num-samples", type=int, default=1024)
    a = p.parse_args(argv)
    with open(a.report) as f:
        best = json.load(f)["best_params"]
    brush = run_laps("brush", None, a.track_seed, a.frames, a.num_samples, a.track_generator)
    pacejka = run_laps("pacejka", best, a.track_seed, a.frames, a.num_samples, a.track_generator)
    ratio = pacejka["meters"] / max(brush["meters"], 1e-9)
    print(f"brush:   {brush}")
    print(f"pacejka: {pacejka}")
    print(f"distance ratio (pacejka/brush): {ratio:.3f}  (gate: within 3% -> >= 0.97)")


if __name__ == "__main__":
    main()
