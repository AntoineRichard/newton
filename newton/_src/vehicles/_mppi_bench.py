# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless benchmark harness for the MPPI racing example.

Internal (not public API). Drives ``example_vehicle_mppi_track`` over one or
more fixed tracks with a null viewer, logging per-frame executed commands and
timing ``step()``, and emits the equal-compute metrics the DIAL-MPC annealing
plan compares against (lap distance, cost, executed-command jitter, steering
reversals, hero out-of-bounds fraction, steps/s).

Usage::

    from newton._src.vehicles._mppi_bench import bench, TUNING_TRACK, VALIDATION_TRACKS

    result = bench([TUNING_TRACK], frames=240)
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import warp as wp

# The five fixed benchmark tracks (generator, seed, TrackGenConfig overrides),
# reproducing the candidate-grid identities chosen for the annealing plan.
TUNING_TRACK = ("hull", 4, {"hull_displacement": 0.35})
VALIDATION_TRACKS = [
    ("bezier", 0, {}),
    ("bezier", 9, {"rad": 0.25, "min_num_points": 12, "max_num_points": 15}),
    ("checkpoint", 5, {"checkpoint_count": 18}),
    ("repulsive", 3, {"repulsive_grow_mult_min": 3.0, "repulsive_grow_mult_max": 3.5}),
]
ALL_TRACKS = [TUNING_TRACK, *VALIDATION_TRACKS]


def _track_id(generator: str, seed: int, params: dict) -> str:
    extra = " ".join(f"{k}={v}" for k, v in params.items())
    return f"{generator} s{seed}" + (f" {extra}" if extra else "")


def _build_example(
    generator,
    seed,
    params,
    num_samples,
    horizon,
    rollout_substeps,
    device,
    brake_mode="none",
    w_underspeed=None,
    w_stall_arc=None,
    anti_stall_drive=None,
    ess_max=None,
    n_knots=None,
):
    # Lazy import: the example lives in newton.examples and pulls in the viewer
    # stack; keep it out of module import time.
    import importlib  # noqa: PLC0415

    ex_mod = importlib.import_module("newton.examples.vehicles.example_vehicle_mppi_track")
    import newton  # noqa: PLC0415

    viewer = newton.viewer.ViewerNull(num_frames=10**9)
    args = SimpleNamespace(
        test=False,
        num_samples=num_samples,
        horizon=horizon,
        rollout_substeps=rollout_substeps,
        track_seed=seed,
        track_generator=generator,
        track_param=dict(params),
        device=device,
        brake_mode=brake_mode,
        w_underspeed=w_underspeed,
        w_stall_arc=w_stall_arc,
        anti_stall_drive=anti_stall_drive,
        ess_max=ess_max,
        n_knots=n_knots,
    )
    example = ex_mod.Example(viewer, args)
    return example, viewer


def bench_track(
    generator,
    seed,
    params=None,
    *,
    frames=240,
    num_samples=1024,
    horizon=48,
    rollout_substeps=4,
    device="cuda:0",
    warmup=1,
    brake_mode="none",
    sigma_horizon_factor=1.0,
    w_underspeed=None,
    w_stall_arc=None,
    anti_stall_drive=None,
    ess_max=None,
    n_knots=None,
    beta=None,
    w_rate=None,
    w_exec=None,
    zero_mean_fraction=0.0,
    tsallis_q=1.0,
) -> dict:
    """Run one track headless and return its metric dict.

    Args:
        generator: track_gen generator family.
        seed: base track seed.
        params: generator-specific ``TrackGenConfig`` overrides.
        frames: executed frames to simulate (excludes warmup for steps/s).
        num_samples: MPPI samples K (= worlds).
        horizon: MPPI horizon H.
        rollout_substeps: solver substeps per rollout step.
        device: Warp device.
        warmup: leading frames excluded from the steps/s timer (the first frame
            captures the CUDA graph and is not representative).
    """
    params = params or {}
    example, viewer = _build_example(
        generator,
        seed,
        params,
        num_samples,
        horizon,
        rollout_substeps,
        device,
        brake_mode,
        w_underspeed,
        w_stall_arc,
        anti_stall_drive,
        ess_max,
        n_knots,
    )
    if sigma_horizon_factor != 1.0:
        example.planner._set_sigma_horizon_factor(sigma_horizon_factor)
    # Task 3b literature-informed knobs (default-off no-ops): RA-MPPI zero-mean
    # sample fraction and Tsallis deformed-exponential weighting exponent.
    if zero_mean_fraction != 0.0:
        example.planner._set_zero_mean_fraction(zero_mean_fraction)
    if tsallis_q != 1.0:
        example.planner._set_tsallis_q(tsallis_q)
    # Decision-5 removal-ablation knobs: override the existing smoothness
    # machinery post-construction (AR(1) beta, the w_rate cost, the t=0
    # exec-coupling weight w_exec) to test what the spline prior subsumes.
    if beta is not None:
        example.planner.set_beta(beta)
    if w_rate is not None or w_exec is not None:
        cp = example.cost_params.numpy()
        if w_rate is not None:
            cp[4] = float(w_rate)
        if w_exec is not None:
            cp[9] = float(w_exec)
        example.cost_params.assign(cp)

    drive = np.empty(frames, dtype=np.float64)
    steer = np.empty(frames, dtype=np.float64)
    brake = np.empty(frames, dtype=np.float64)
    speed = np.empty(frames, dtype=np.float64)
    v_des = np.empty(frames, dtype=np.float64)
    mean_cost = np.empty(frames, dtype=np.float64)
    best_cost = np.empty(frames, dtype=np.float64)
    ess = np.empty(frames, dtype=np.float64)
    hero_s = np.empty(frames, dtype=np.float64)
    oob = np.zeros(frames, dtype=np.int32)

    timed = 0.0
    for f in range(frames):
        t0 = time.perf_counter()
        example.step()
        wp.synchronize_device(example.model.device)
        dt = time.perf_counter() - t0
        if f >= warmup:
            timed += dt
        u = example.u_prev.numpy()  # executed (drive, raw steer) for this frame
        drive[f] = float(u[0])
        steer[f] = float(u[1])
        brake[f] = float(example.vehicles.commands.brake.numpy()[0])  # executed friction brake
        tel = example._telemetry
        speed[f] = tel["speed"]
        v_des[f] = tel["v_des"]
        mean_cost[f] = tel["mean_cost"]
        best_cost[f] = tel["best_cost"]
        ess[f] = tel["ess"]
        hero_s[f] = float(example.car_s.numpy()[0])
        oob[f] = int(tel["hero_oob"])

    lap_dist = float(example.total_s.numpy()[0])
    track_len = float(example.track_len)
    v_profile = example._v_profile.numpy().astype(np.float64)
    cum_s = example._cum_s.numpy().astype(np.float64)
    example_w_under = example.cost_params.numpy()[8]
    example_w_stall_arc = example.cost_params.numpy()[10]
    viewer.close()

    # hairpin-entry braking: locate the two deepest reference-speed minima
    # (well separated) within the arc the hero actually covered, and measure
    # the peak decel and brake the hero produced inside the 5 m of centerline
    # leading into each
    reached = cum_s <= max(lap_dist, 0.0)  # spawn is at s = 0
    profile_masked = np.where(reached, v_profile, np.inf)
    order = np.argsort(profile_masked)
    hairpin_idx = []
    for i in order:
        if not np.isfinite(profile_masked[i]):
            break
        s_h = float(cum_s[i])
        if all(min(abs(s_h - float(cum_s[j])), track_len - abs(s_h - float(cum_s[j]))) > 8.0 for j in hairpin_idx):
            hairpin_idx.append(int(i))
        if len(hairpin_idx) == 2:
            break
    decel = np.concatenate([[0.0], np.maximum(0.0, -np.diff(speed)) * 60.0])
    hairpins = []
    for i in hairpin_idx:
        s_h = float(cum_s[i])
        rel = (s_h - hero_s) % track_len  # forward distance from hero to the apex
        window = rel <= 5.0
        hairpins.append(
            {
                "s": s_h,
                "v_min_ref": float(v_profile[i]),
                "peak_decel": float(decel[window].max()) if window.any() else 0.0,
                "mean_decel": float(decel[window].mean()) if window.any() else 0.0,
                "max_brake": float(brake[window].max()) if window.any() else 0.0,
                "frames_in_window": int(window.sum()),
            }
        )

    dd = np.diff(drive)
    dst = np.diff(steer)
    # steering reversals: sign changes of the executed-steer increment above a
    # small deadband (ignores numerical chatter around zero)
    ds_sign = np.sign(np.where(np.abs(dst) > 1e-3, dst, 0.0))
    nz = ds_sign[ds_sign != 0.0]
    reversals = int(np.sum(nz[1:] * nz[:-1] < 0.0)) if nz.size > 1 else 0

    steps_per_s = (frames - warmup) / timed if timed > 0 else float("nan")

    return {
        "track": _track_id(generator, seed, params),
        "generator": generator,
        "seed": seed,
        "params": params,
        "track_len_m": track_len,
        "frames": frames,
        "num_samples": num_samples,
        "horizon": horizon,
        "rollout_substeps": rollout_substeps,
        "rollout_budget": num_samples * horizon * rollout_substeps,
        "lap_dist_m": lap_dist,
        "lap_frac": lap_dist / track_len if track_len else float("nan"),
        "mean_cost": float(mean_cost.mean()),
        "best_cost": float(best_cost.mean()),
        "mean_ess": float(ess.mean()),
        "drms_drive": float(np.sqrt(np.mean(dd**2))),
        "drms_steer": float(np.sqrt(np.mean(dst**2))),
        "steer_reversals": reversals,
        "hero_oob_frac": float(oob.mean()),
        "finished": bool(lap_dist >= 4.0),
        "steps_per_s": steps_per_s,
        "mean_speed": float(speed.mean()),
        "max_speed": float(speed.max()),
        # peak decel the hero actually achieved [m/s^2]: the braking-authority
        # ceiling the planner could exploit this run
        "peak_decel": float(np.max(np.maximum(0.0, -np.diff(speed))) * 60.0) if frames > 1 else 0.0,
        "mean_brake": float(brake.mean()),
        "max_brake": float(brake.max()),
        "hairpins": hairpins,
        "brake_mode": brake_mode,
        "sigma_horizon_factor": sigma_horizon_factor,
        "w_underspeed": float(example_w_under),
        "w_stall_arc": float(example_w_stall_arc),
        "anti_stall_drive": float(example._anti_stall_drive),
        "ess_max": ess_max,
        "n_knots": n_knots,
        "beta": beta,
        "w_rate": w_rate,
        "w_exec": w_exec,
        "zero_mean_fraction": zero_mean_fraction,
        "tsallis_q": tsallis_q,
    }


def bench(tracks=None, *, frames=240, config_overrides=None) -> dict:
    """Benchmark a list of ``(generator, seed, params)`` tracks.

    Returns a dict with ``per_track`` (list of metric dicts) and ``aggregate``
    means over the finished tracks.
    """
    tracks = tracks if tracks is not None else ALL_TRACKS
    overrides = config_overrides or {}
    per_track = []
    for generator, seed, params in tracks:
        per_track.append(bench_track(generator, seed, params, frames=frames, **overrides))

    keys = ("lap_dist_m", "drms_drive", "drms_steer", "steer_reversals", "hero_oob_frac", "steps_per_s")
    agg = {k: float(np.mean([r[k] for r in per_track])) for k in keys}
    return {"per_track": per_track, "aggregate": agg, "config_overrides": overrides}
