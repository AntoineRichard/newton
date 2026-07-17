# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record brush-model MPPI laps into a tirecal session file.

Drives the existing MPPI track example headless (hero world only is recorded)
across a range of track seeds and writes laps through :mod:`tirecal.format`.
This module imports newton; the format it writes does not require it.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from . import format as tcformat
from . import seed_fit

CONTROL_NAMES = ("drive", "steer", "brake")

# Default Pacejka Magic-Formula coefficients (per unit normal load), in
# ``seed_fit.PARAM_NAMES`` order (bx, cx, ex, by, cy, ey, rbx1, rbx2, rby1,
# rby2). This is the synthetic-recovery ground truth the ``pacejka`` recording
# mode stamps into every wheel's ``mf_params`` row. Task 3's public
# ``newton.vehicles.PacejkaParams`` defaults MUST match these values.
PACEJKA_DEFAULT_THETA = np.array([12.5, 1.6, 0.4, 14.0, 1.35, 0.3, 12.0, 10.0, 10.0, 8.0], dtype=np.float64)


def _example_args(parser, num_samples, track_seed, brake_mode, track_generator, test_mode):
    args = parser.parse_args([])
    args.num_samples = num_samples
    args.track_seed = track_seed
    args.brake_mode = brake_mode
    args.track_generator = track_generator
    args.test = test_mode
    return args


def _capture_row(ex):
    """Hero-slice state row (device -> host); returns dict of 1-sample arrays."""
    dyn, patch = ex.vehicles.dynamics, ex.vehicles.patch
    body_q_full = ex.state_0.body_q.numpy()[: ex.bodies_per_world].copy()
    body_qd_full = ex.state_0.body_qd.numpy()[: ex.bodies_per_world].copy()
    return {
        "joint_q": ex.state_0.joint_q.numpy()[: ex.dofs_per_world].copy(),
        "joint_qd": ex.state_0.joint_qd.numpy()[: ex.vel_dofs_per_world].copy(),
        "body_q_full": body_q_full,
        "body_qd_full": body_qd_full,
        "trans_long": dyn.trans_long.numpy()[: ex.wheels_per_world].copy(),
        "trans_lat": dyn.trans_lat.numpy()[: ex.wheels_per_world].copy(),
        "patch_fz": patch.fz.numpy()[: ex.wheels_per_world].copy(),
        "wheel_omega": dyn.omega.numpy()[: ex.wheels_per_world].copy(),
        "chassis_q": body_q_full[ex._chassis0].copy(),
        "chassis_qd": body_qd_full[ex._chassis0].copy(),
    }


def _override_tire_model(ex, tire_model: str) -> None:
    """Post-construction tire-model override (the example hardcodes brush).

    The public ``PacejkaParams`` config (Task 3) is not yet wired into
    ``WheeledConfig``, so the ``pacejka`` mode stamps the kernel-level tire
    selector and Magic-Formula coefficients directly onto the flat device
    tables from :data:`PACEJKA_DEFAULT_THETA`.
    """
    if tire_model == "brush":
        return
    import newton.vehicles as nv  # noqa: PLC0415 (deferred: keep newton optional at import time)

    dyn = ex.vehicles.dynamics
    # Task 3's public PacejkaParams will add TireModel.PACEJKA; until then the
    # kernel-level selector is the literal 2 (tire.TIRE_PACEJKA).
    pacejka_code = int(getattr(nv.TireModel, "PACEJKA", 2))
    host = dyn.tire_model.numpy()
    host[...] = pacejka_code
    dyn.tire_model.assign(host)
    dyn.mf_params.assign(np.tile(PACEJKA_DEFAULT_THETA, (max(dyn.wheel_count, 1), 1)).astype(np.float32))


def record_lap(track_seed, num_samples, frames_max, brake_mode, track_generator, tire_model, test_mode):
    import newton  # noqa: PLC0415 (deferred: keep newton optional at import time)
    from newton.examples.vehicles.example_vehicle_mppi_track import Example  # noqa: PLC0415

    viewer = newton.viewer.ViewerNull(num_frames=frames_max)
    args = _example_args(Example.create_parser(), num_samples, track_seed, brake_mode, track_generator, test_mode)
    ex = Example(viewer, args)
    _override_tire_model(ex, tire_model)

    rows = [_capture_row(ex)]
    controls = []
    lap_complete = False
    for _ in range(frames_max):
        ex.step()
        cmd = ex.vehicles.commands
        controls.append([float(cmd.drive.numpy()[0]), float(cmd.steer.numpy()[0]), float(cmd.brake.numpy()[0])])
        rows.append(_capture_row(ex))
        if float(ex.total_s.numpy()[0]) >= ex.track_len:
            lap_complete = True
            break

    steps = len(controls)
    stack = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    cfg = dataclasses.asdict(ex.vehicles.config)
    # Task 3's PacejkaParams is not yet part of WheeledConfig, so the ground-truth
    # coefficients come from PACEJKA_DEFAULT_THETA rather than the config object.
    pacejka = dict(zip(seed_fit.PARAM_NAMES, PACEJKA_DEFAULT_THETA.tolist(), strict=True))
    lap = tcformat.Lap(
        time=np.arange(steps + 1, dtype=np.float64) * ex.frame_dt,
        body_q=stack["chassis_q"],
        body_qd=stack["chassis_qd"],
        wheel_omega=stack["wheel_omega"],
        controls=np.asarray(controls, dtype=np.float32),
        control_names=CONTROL_NAMES,
        meta={
            "sim_dt": ex.sim_dt,
            "sim_substeps": ex.sim_substeps,
            "control_rate_hz": ex.fps,
            "track_seed": int(ex.track_seed),
            "track_generator": track_generator,
            "frames": steps,
            "lap_complete": lap_complete,
            "track_len": float(ex.track_len),
        },
        sim_state={
            k: stack[k]
            for k in ("joint_q", "joint_qd", "body_q_full", "body_qd_full", "trans_long", "trans_lat", "patch_fz")
        },
    )
    tire_params = (
        {"c_long": cfg["longitudinal_stiffness"], "c_lat": cfg["lateral_stiffness"], "mu": cfg["friction"]}
        if tire_model == "brush"
        else pacejka
    )
    vehicle = {
        "config": cfg,
        "pacejka_defaults": pacejka,
        "tire_model": tire_model,
        "tire_params": tire_params,
        "frame_dt": ex.frame_dt,
        "sim_substeps": ex.sim_substeps,
        "bodies_per_world": ex.bodies_per_world,
        "dofs_per_world": ex.dofs_per_world,
        "vel_dofs_per_world": ex.vel_dofs_per_world,
        "wheels_per_world": ex.wheels_per_world,
        "chassis_index": int(ex._chassis0),
    }
    return lap, vehicle


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output session .h5 path")
    p.add_argument("--laps", type=int, default=24, help="number of laps (one track seed each)")
    p.add_argument("--track-seed0", type=int, default=0, help="first track seed; seed s records lap s")
    p.add_argument("--num-samples", type=int, default=1024, help="MPPI samples (planner quality)")
    p.add_argument("--frames-max", type=int, default=3600, help="frame cap per lap (60 fps)")
    p.add_argument("--brake-mode", default="none", choices=("none", "channel", "esc"))
    p.add_argument("--track-generator", default="bezier")
    p.add_argument(
        "--tire-model",
        default="brush",
        choices=("brush", "pacejka"),
        help="pacejka records with PacejkaParams defaults (synthetic-recovery mode)",
    )
    p.add_argument("--test", action="store_true", help="tiny planner (32 worlds) for smoke tests")
    a = p.parse_args(argv)

    laps, vehicle = [], None
    for i in range(a.laps):
        lap, veh = record_lap(
            a.track_seed0 + i, a.num_samples, a.frames_max, a.brake_mode, a.track_generator, a.tire_model, a.test
        )
        vehicle = vehicle or veh
        laps.append(lap)
        print(f"lap {i}: seed={lap.meta['track_seed']} frames={lap.meta['frames']} complete={lap.meta['lap_complete']}")
    tcformat.write_session(a.out, tcformat.Session(vehicle=vehicle, laps=laps))
    print(f"wrote {len(laps)} laps to {a.out}")


if __name__ == "__main__":
    main()
