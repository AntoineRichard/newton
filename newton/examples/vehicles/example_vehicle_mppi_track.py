# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Vehicle MPPI Track
#
# Races the rc_car fixture around a procedurally generated closed track
# using an MPPI controller whose rollout model is the simulator itself:
# num-samples replicated worlds are collocated at the origin (cross-world
# collision is filtered), world 0 is the rendered hero executing the
# optimized command and worlds 1..K-1 evaluate noise-perturbed command
# sequences every frame. Track generation, out-of-bounds tests, and
# checkpoint progress come from the track_gen package.
#
# Command: python -m newton.examples vehicle_mppi_track --viewer gl
#
###########################################################################

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.vehicles as nv

try:
    import track_gen
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.collision import CollisionChecker
    from track_gen.props import PropSampler
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("This example requires the track_gen package: pip install -e <path-to-track_gen>") from exc

_ASSET_DIR = Path(newton.examples.get_asset("wheeled"))

TRACK_HALF_WIDTH = 0.5  # [m]
# per-generator generation settings: scale calibrated to a ~22-26 m footprint,
# N_max sized to the generator's perimeter (repulsive tracks are space-filling:
# ~128 m of track in a 26 m footprint vs ~85 m for bezier; the other families
# normalize to a bezier-like footprint at the same scale, ~360 points max at
# half_width 0.5, so bezier's settings fit them)
TRACK_GENERATOR_SETTINGS = {
    "bezier": (17.0, 512),
    "checkpoint": (17.0, 512),
    "hull": (17.0, 512),
    "polar": (17.0, 512),
    "repulsive": (10.0, 1024),
    "voronoi": (17.0, 512),
}
CONE_SPACING = 0.5  # [m]
CAR_HALF_EXTENTS = (0.29, 0.15)  # oriented OOB box [m] (Slash-class rc car)
MAX_TRACK_ATTEMPTS = 32

# reference speed profile (curvature-limited steady state + forward accel pass
# + backward braking pass, the standard racing decomposition): braking
# foresight lives in the profile, not the MPPI horizon
A_LAT_MAX = 16.0  # [m/s^2] usable lateral acceleration for v_ss = sqrt(a_lat/|kappa|), ~80% of mu*g
A_ACC = 6.0  # [m/s^2] forward pass acceleration limit
A_BRK = 14.0  # [m/s^2] backward pass braking limit
V_CAP = 17.0  # [m/s] profile ceiling (full motor speed; the profile does the discipline)
PROX_MARGIN = 0.15  # [m] graded wall-proximity band
# anti-stall speed floor [m/s]: the under-speed shortfall penalty only pulls
# the car up to this speed (or v_des if lower), not to v_des itself, so it
# repairs the braking-trap standstill (measured: parked at 0.1 m/s with
# v_des ~ 5) without adding a speed-tracking gradient during normal running
# (which measurably raises drive jitter ~40%)
V_STALL = 1.5
# esc mode: transmitter-style mapping of the negative drive half-axis. The
# drive bounds stay [-0.3, 1.0] (same as none mode; the tight lower bound
# limits how deep the nominal can sink into braking, and clamping the
# symmetric exploration noise there biases the post-clamp noise mean
# positive, helping the softmax average recover toward throttle when the
# cost landscape is flat). Commands in (-ESC_DEADBAND, 0) coast -- a neutral
# zone like a real transmitter, so near-zero exploration noise does not drag
# the ESC brake -- and the remaining (-0.3, -ESC_DEADBAND) span maps linearly
# to the full [0, 1] brake range.
ESC_DEADBAND = 0.05
ESC_BRAKE_GAIN = 1.0 / (1.0 - ESC_DEADBAND)
# esc-mode anti-stall crawl assist: minimum executed drive while the hero is
# below V_STALL. The cost-side shortfall terms alone cannot break the parked
# equilibrium under heavy softmax averaging (Tsallis q > 1, ESS ~800): from a
# standstill with a deep-braking nominal, escape needs sustained throttle the
# sampler around that nominal essentially never draws, so every sample's
# rollout arc is ~0 and the shortfall is a constant offset with no gradient --
# while any escape-ward sample still pays the ~100-unit t=0 exec-coupling
# cost. Flooring only the EXECUTED hero command (never the sampled rollouts)
# physically restores forward motion, which restores arc diversity across the
# samples, and the progress/shortfall costs then pull the nominal itself back
# to throttle.
ANTI_STALL_DRIVE = 0.5
# consecutive frames the hero must sit below V_STALL before the crawl assist
# engages: a genuine stall lasts 60+ frames, while a legitimate hairpin-apex
# dip below V_STALL lasts only a few -- engaging instantly measurably chatters
# the executed drive in hull's hairpin cluster (drive dRMS +28% at q = 1.0)
ANTI_STALL_DWELL_FRAMES = 18
# multiplier on the t=0 command-rate term that couples each new plan to the
# previously executed command (the within-rollout w_rate alone is far too
# weak to suppress replan-to-replan steering flicker)
EXEC_COUPLING = 25.0
STEER_SPEED_REF = 12.0  # [m/s] dual-rate reference: steer authority halves at this speed^2 ratio

MINIMAP_SIZE = 240.0  # [px] minimap window edge length
MINIMAP_MARGIN = 12.0  # [px] gap to the viewport's bottom-right corner
MINIMAP_PAD = 0.08  # fractional padding around the track bounding box
MINIMAP_TRAIL_MAX = 3600  # trail ring-buffer capacity (one entry per frame)

# F1-broadcast HUD cluster, arranged around the bottom-right minimap
GMETER_SIZE = 150.0  # [px] G-meter widget edge length (square, left of minimap)
HUD_GAP = 10.0  # [px] gap between HUD cluster elements
HUD_PANEL_W = 496.0  # [px] speed + throttle/brake panel width (left of the G-meter)
HUD_BAR_SEGMENTS = 48  # LED-style segments per throttle/brake bar
GMETER_G_EDGE = 2.5  # [g] acceleration magnitude mapped to the widget edge
GMETER_TRAIL_MAX = 20  # G-dot history length (faint trail)
GMETER_EMA = 0.3  # blend for the exponential-moving-average dot smoothing
GRAVITY = 9.81  # [m/s^2] used to express accelerations in g units
# system fonts for the HUD text, in preference order (JetBrains Mono, then
# Noto); the HUD falls back to the default imgui font when none is installed
HUD_FONT_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Regular.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
)
HUD_FONT_BOLD_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Bold.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
)


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.kernel
def _gather_car_pose(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    chassis: wp.array[wp.int32],
    pos: wp.array[wp.vec2f],
    yaw: wp.array[float],
    speed: wp.array[float],
):
    e = wp.tid()
    tf = body_q[chassis[e]]
    p = wp.transform_get_translation(tf)
    pos[e] = wp.vec2f(p[0], p[1])
    yaw[e] = _quat_yaw(wp.transform_get_rotation(tf))
    qd = body_qd[chassis[e]]
    # linear velocity is the first three entries (world frame)
    speed[e] = wp.sqrt(qd[0] * qd[0] + qd[1] * qd[1])


@wp.kernel
def _project_centerline(
    center: wp.array[wp.vec2f],
    cum_s: wp.array[float],
    v_profile: wp.array[float],
    count: int,
    pos: wp.array[wp.vec2f],
    s_out: wp.array[float],
    v_des: wp.array[float],
):
    e = wp.tid()
    p = pos[e]
    best_d2 = float(1.0e12)
    best_s = float(0.0)
    best_i = int(0)
    for i in range(count):
        a = center[i]
        j = i + 1
        if j == count:
            j = 0
        b = center[j]
        ab = b - a
        denom = wp.max(wp.dot(ab, ab), 1.0e-9)
        u = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + u * ab
        d2 = wp.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_s = cum_s[i] + u * wp.sqrt(wp.dot(ab, ab))
            best_i = i
    s_out[e] = best_s
    v_des[e] = v_profile[best_i]


@wp.kernel
def _accumulate_lap_distance(
    s: wp.array[float],
    total_len: float,
    s_prev: wp.array[float],
    total_s: wp.array[float],
):
    ds = s[0] - s_prev[0]
    if ds < -0.5 * total_len:
        ds += total_len
    elif ds > 0.5 * total_len:
        ds -= total_len
    total_s[0] = total_s[0] + ds
    s_prev[0] = s[0]


@wp.kernel
def _broadcast_slice_tf(snap: wp.array[wp.transform], n_per: int, dst: wp.array[wp.transform]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_sv(snap: wp.array[wp.spatial_vector], n_per: int, dst: wp.array[wp.spatial_vector]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.kernel
def _broadcast_slice_f32(snap: wp.array[float], n_per: int, dst: wp.array[float]):
    w, i = wp.tid()
    dst[w * n_per + i] = snap[i]


@wp.func
def _steer_scale(speed: float) -> float:
    # speed-sensitive steering (RC dual-rate analog): at speed, large steering
    # angles only saturate the front tires, so commands beyond a few degrees
    # are physically indistinguishable to the cost and the planner's noise
    # accumulates freely in the saturated region (straight-line wobble).
    # Scaling the command keeps the explored range meaningful at every speed.
    return 1.0 / (1.0 + (speed / STEER_SPEED_REF) ** 2.0)


# --brake-mode codes: how a sampled command reaches the brake actuator.
# The rc_car has no friction brake -- ``brake_target`` models the ESC's
# regen/drag braking (resistive torque, cannot reverse the wheel), so the
# question is only how the planner's action space maps onto it.
BRAKE_MODE_NONE = 0  # brake always 0 (today's behavior; negative drive = weak motor drag)
BRAKE_MODE_CHANNEL = 1  # third sampled channel in [-1, 1], rectified to [0, 1]
BRAKE_MODE_ESC = 2  # RC-transmitter-faithful: drive < 0 -> ESC brake |drive|, drive 0


@wp.func
def _map_brake(drive_cmd: float, brake_cmd: float, brake_mode: int) -> wp.vec2f:
    """Map (drive, optional brake channel) to executed (drive, brake).

    ``channel`` mode rectifies the sampled brake channel: a one-sided [0, 1]
    bound makes the clamped exploration noise strictly positive, so the
    softmax update ratchets the nominal brake up until the car cannot launch
    (measured: permanent ~0.5 brake, lap distance 0). ``esc`` mode folds
    braking into the throttle axis like a real RC transmitter: negative
    drive engages the ESC brake instead of requesting reverse wheel speed.
    """
    if brake_mode == BRAKE_MODE_ESC:
        if drive_cmd < -ESC_DEADBAND:
            return wp.vec2f(0.0, wp.min((-drive_cmd - ESC_DEADBAND) * ESC_BRAKE_GAIN, 1.0))
        if drive_cmd < 0.0:
            return wp.vec2f(0.0, 0.0)  # neutral zone: coast
        return wp.vec2f(drive_cmd, 0.0)
    if brake_mode == BRAKE_MODE_CHANNEL:
        return wp.vec2f(drive_cmd, wp.max(brake_cmd, 0.0))
    return wp.vec2f(drive_cmd, 0.0)


@wp.kernel
def _apply_sample_commands(
    samples: wp.array3d[float],
    t: int,
    brake_mode: int,
    speed: wp.array[float],
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    bc = float(0.0)
    if brake_mode == BRAKE_MODE_CHANNEL:
        bc = samples[v, t, 2]
    db = _map_brake(samples[v, t, 0], bc, brake_mode)
    drive[v] = db[0]
    steer[v] = samples[v, t, 1] * _steer_scale(speed[v])
    brake[v] = db[1]


@wp.kernel
def _apply_nominal_command(
    nominal: wp.array2d[float],
    brake_mode: int,
    anti_stall_drive: float,
    stall_frames: wp.array[wp.int32],
    speed: wp.array[float],
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    v = wp.tid()
    bc = float(0.0)
    if brake_mode == BRAKE_MODE_CHANNEL:
        bc = nominal[0, 2]
    db = _map_brake(nominal[0, 0], bc, brake_mode)
    drive[v] = db[0]
    steer[v] = nominal[0, 1] * _steer_scale(speed[v])
    brake[v] = db[1]
    # anti-stall crawl assist (executed command only; sampled rollouts are
    # never floored): once the hero has dwelt below the stall gate, override
    # braking/coasting with a minimum drive so it physically escapes the
    # parked equilibrium the cost landscape cannot break under heavy softmax
    # averaging. The dwell keeps legitimate hairpin-apex dips untouched.
    if anti_stall_drive > 0.0 and stall_frames[0] >= ANTI_STALL_DWELL_FRAMES:
        drive[v] = wp.max(drive[v], anti_stall_drive)
        brake[v] = 0.0


@wp.kernel
def _zero_plan_buffers(
    costs: wp.array[float], dead: wp.array[wp.int32], back_dist: wp.array[float], arc: wp.array[float]
):
    e = wp.tid()
    costs[e] = 0.0
    dead[e] = 0
    back_dist[e] = 0.0
    arc[e] = 0.0


@wp.kernel
def _record_start_oob(oob: wp.array[wp.int32], start_oob: wp.array[wp.int32]):
    e = wp.tid()
    start_oob[e] = oob[e]


@wp.kernel
def _accumulate_cost(
    s: wp.array[float],
    v_des: wp.array[float],
    speed: wp.array[float],
    oob: wp.array[wp.int32],
    clearance: wp.array[float],
    start_oob: wp.array[wp.int32],
    samples: wp.array3d[float],
    t: int,
    horizon: int,
    total_len: float,
    stall_arc_floor: float,
    params: wp.array[float],
    # [w_progress, w_speed, w_steer, kill, w_rate, w_prox, w_term, w_center, w_under, w_exec, w_stall_arc]
    u_prev: wp.array[float],
    s_prev: wp.array[float],
    back_dist: wp.array[float],
    arc: wp.array[float],
    dead: wp.array[wp.int32],
    costs: wp.array[float],
):
    e = wp.tid()
    if dead[e] == 1:
        return
    # time-decayed kill: crashing later is strictly cheaper, which gives the
    # planner a braking gradient even when every sample ends in a wall
    kill = params[3] * wp.pow(0.9, float(t))
    if oob[e] == 1 and start_oob[e] == 0:
        dead[e] = 1
        costs[e] = costs[e] + kill
        return
    # arc-length progress along the centerline, unwrapped at the lap seam
    ds = s[e] - s_prev[e]
    if ds < -0.5 * total_len:
        ds += total_len
    elif ds > 0.5 * total_len:
        ds -= total_len
    s_prev[e] = s[e]
    if wp.abs(ds) > 5.0:  # teleport/NaN guard
        ds = 0.0
    if start_oob[e] == 0:
        # backward travel along the track beyond a small back-out budget is
        # killed, so reverse stays a brake/recovery tool (this also covers
        # turned-around forward driving, which velocity checks would miss)
        back_dist[e] = back_dist[e] + wp.max(0.0, -ds)
        if back_dist[e] > 1.0:
            dead[e] = 1
            costs[e] = costs[e] + kill
            return
        # net arc covered over the whole rollout so far, for the trajectory
        # progress-shortfall term below
        arc[e] = arc[e] + ds
    c = -params[0] * ds
    # one-sided reference-speed penalty: the profile caps corner-entry speed,
    # progress reward alone pushes speed up everywhere else
    over = wp.max(0.0, speed[e] - v_des[e])
    c += params[1] * over * over
    # under-speed shortfall penalty (default 0): the progress reward alone
    # leaves standstill nearly cost-free, so on the esc mode's symmetric
    # drive axis a deep-negative (braking) nominal is a warm-start trap --
    # the softmax noise average is zero-mean and nothing climbs back out
    # (measured: parked at 0.1 m/s for 60+ frames with v_des ~5). A small
    # quadratic shortfall restores the climb-out gradient without touching
    # corner-entry braking, which targets v_des itself, not below it.
    under = wp.max(0.0, wp.min(v_des[e], V_STALL) - speed[e])
    c += params[8] * under * under
    if t == horizon - 1:
        # terminal over-speed cost: a cheap stand-in for a value function at
        # the horizon end (Vazquez-style terminal speed limit)
        c += params[6] * over * over
        # trajectory-level progress-shortfall (anti-stall): penalize how far the
        # whole rollout fell short of a modest floor arc (V_STALL * horizon_dt).
        # Unlike the per-step under-speed term -- which penalizes every
        # near-stall sample's instantaneous speed EQUALLY and so gives the
        # softmax no gradient to prefer an escaping rollout over a parked one --
        # this single terminal term scales with the whole rollout's distance,
        # so it is immune to the low-speed per-step cost flatness that traps the
        # planner (esc braking overshoot -> standstill, seen under heavy q>1
        # averaging with ESS ~800). It is silent whenever a rollout averages
        # above V_STALL (all racing rollouts do), so it does not disturb
        # cornering or top speed. Applied only to on-track (non-recovery) plans.
        if start_oob[e] == 0:
            short = wp.max(0.0, stall_arc_floor - arc[e])
            c += params[10] * short * short
    steer = samples[e, t, 1]
    c += params[2] * steer * steer
    if t > 0:
        # small command-rate penalty for smooth trajectories
        dd = samples[e, t, 0] - samples[e, t - 1, 0]
        dst = samples[e, t, 1] - samples[e, t - 1, 1]
        c += params[4] * (dd * dd + dst * dst)
    else:
        # couple the plan's first command to the command the hero just
        # executed: without this, replans are free to flick the steering
        # every frame (the dominant straight-line wobble mode — the
        # within-rollout rate penalty cannot see across replan boundaries).
        # Held in its own weight (params[9], default w_rate * EXEC_COUPLING)
        # so the Task-3 removal ablation can drop it independently of w_rate.
        dd = samples[e, 0, 0] - u_prev[0]
        dst = samples[e, 0, 1] - u_prev[1]
        c += params[9] * (dd * dd + dst * dst)
    # graded wall proximity: a gradient before contact instead of a cliff at it
    c += params[5] * wp.max(0.0, PROX_MARGIN - clearance[e])
    # light clearance reward (AutoRally-style sloped track cost): breaks the
    # cost tie between centered-straight and swerving samples on straights,
    # so the softmax average stops being noise-dominated there
    c += -params[7] * wp.max(0.0, clearance[e])
    if start_oob[e] == 1:
        # recovery mode (rollout began outside the band, e.g. after a crash):
        # no kills; penalize distance outside the band so plans steer back in
        c += params[0] * wp.max(0.0, -clearance[e])
    costs[e] = costs[e] + c


@wp.kernel
def _update_stall_frames(speed: wp.array[float], stall_frames: wp.array[wp.int32]):
    # consecutive executed frames the hero has spent below the stall gate
    if speed[0] < V_STALL:
        stall_frames[0] = stall_frames[0] + 1
    else:
        stall_frames[0] = 0


@wp.kernel
def _store_executed_command(nominal: wp.array2d[float], u_prev: wp.array[float]):
    u_prev[0] = nominal[0, 0]
    u_prev[1] = nominal[0, 1]


@wp.kernel
def _record_ribbon(
    body_q: wp.array[wp.transform],
    chassis0: int,
    t: int,
    ribbon: wp.array[wp.vec3],
):
    p = wp.transform_get_translation(body_q[chassis0])
    ribbon[t] = wp.vec3(p[0], p[1], p[2] + 0.05)


def _speed_profile(center, seg_len):
    """Curvature-limited speed profile with forward-accel and backward-braking passes."""
    prev = np.roll(center, 1, axis=0)
    nxt = np.roll(center, -1, axis=0)
    v1 = center - prev
    v2 = nxt - center
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    angles = np.arctan2(cross, (v1 * v2).sum(axis=1))
    ds = 0.5 * (np.linalg.norm(v1, axis=1) + np.linalg.norm(v2, axis=1))
    kappa = np.abs(angles) / np.maximum(ds, 1e-6)
    kappa = np.convolve(np.concatenate([kappa[-2:], kappa, kappa[:2]]), np.ones(5) / 5.0, mode="valid")
    v = np.minimum(np.sqrt(A_LAT_MAX / np.maximum(kappa, 1e-6)), V_CAP)
    n = len(center)
    for _ in range(2):  # two wrap laps so the closed loop converges
        for i in range(n):  # forward pass: acceleration limit
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * A_ACC * seg_len[i]))
        for i in range(n - 1, -1, -1):  # backward pass: braking limit
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * A_BRK * seg_len[i]))
    return v.astype(np.float32)


def _build_model(num_worlds):
    manifest = json.loads((_ASSET_DIR / "manifest.json").read_text())
    asset = next(a for a in manifest["assets"] if a["name"] == "rc_car")
    rd = asset["reference_dimensions"]

    car = newton.ModelBuilder()
    nv.register_vehicle_attributes(car)
    newton.solvers.SolverMuJoCo.register_custom_attributes(car)
    car.add_usd(str(_ASSET_DIR / asset["file"]))
    nv.configure_wheel_axle_joints(car, axle_joint_labels=asset["axle_joint_labels"])

    joint_by_label = {label: i for i, label in enumerate(car.joint_label)}
    shape_by_label = {label: i for i, label in enumerate(car.shape_label)}
    nv.set_vehicle(
        car,
        0,
        drive_mode=int(nv.DriveMode.ACKERMANN),
        wheelbase=rd["wheelbase_m"],
        track_width=rd["track_width_m"],
        steer_limit=math.radians(rd["steering_limit_deg"]),
    )
    steering = asset["steering_joint_labels"]
    for wheel_id, (body_label, shape_label) in enumerate(
        zip(asset["wheel_body_labels"], asset["wheel_shape_labels"], strict=True)
    ):
        name = body_label.split("/")[-1]
        front = "front" in name
        left = "left" in name
        steer_joint = joint_by_label[steering[0 if left else 1]] if front else -1
        nv.add_wheel(
            car,
            shape=shape_by_label[shape_label],
            vehicle_id=0,
            wheel_id=wheel_id,
            radius=rd["wheel_radius_m"],
            width=rd["wheel_width_m"],
            driven=True,
            steerable=front,
            side=(-1 if left else 1),
            axle_row=(0 if front else 1),
            steer_joint=steer_joint,
        )

    scene = newton.ModelBuilder()
    nv.register_vehicle_attributes(scene)
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(car, num_worlds, spacing=(0.0, 0.0, 0.0))
    terrain_cfg = newton.ModelBuilder.ShapeConfig()
    terrain_cfg.mu = 1.0
    scene.add_ground_plane(cfg=terrain_cfg)
    model = scene.finalize()

    joint_type = model.joint_type.numpy()
    joint_child = model.joint_child.numpy()
    free_children = joint_child[joint_type == int(newton.JointType.FREE)]
    if len(free_children) != num_worlds:
        raise RuntimeError(f"expected {num_worlds} free joints, found {len(free_children)}")
    return model, np.sort(free_children).astype(np.int32)


def _parse_track_params(items):
    """Parse ``KEY=VALUE`` overrides into a ``TrackGenConfig`` kwargs dict.

    Values are coerced to int, then float, then left as a string, so
    generator-specific knobs (e.g. ``hull_displacement=0.35``) can be forwarded
    verbatim from the command line or the benchmark harness.
    """
    params = {}
    for item in items or ():
        if isinstance(items, dict):
            params[item] = items[item]
            continue
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        for cast in (int, float):
            try:
                params[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            params[key] = raw
    return params


def _generate_track(num_envs, seed, generator, device, extra_params=None):
    """Generates one track shared by all envs; retries invalid seeds.

    ``extra_params`` forwards generator-specific ``TrackGenConfig`` knobs (e.g.
    ``hull_displacement``, ``checkpoint_count``) so the benchmark protocol can
    reproduce exact track identities beyond the seed.
    """
    if generator not in TRACK_GENERATOR_SETTINGS:
        raise ValueError(f"unsupported --track-generator {generator!r}; choose from {sorted(TRACK_GENERATOR_SETTINGS)}")
    scale, n_max = TRACK_GENERATOR_SETTINGS[generator]
    extra = dict(extra_params) if extra_params else {}
    for attempt in range(MAX_TRACK_ATTEMPTS):
        seeds = wp.array(np.full(num_envs, seed + attempt, dtype=np.int32), dtype=wp.int32, device=device)
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=str(device))
        config = TrackGenConfig(
            num_envs=num_envs,
            generator=generator,
            scale=scale,
            half_width=TRACK_HALF_WIDTH,
            N_max=n_max,
            device=str(device),
            **extra,
        )
        track = TrackGenerator(config, rng).generate()
        if bool(track.valid.numpy()[0]):
            return track, seed + attempt
    raise RuntimeError(f"no valid track after {MAX_TRACK_ATTEMPTS} attempts from seed {seed}")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self._test_mode = getattr(args, "test", False)

        self.num_worlds = 32 if self._test_mode else args.num_samples
        self.model, chassis_ids = _build_model(self.num_worlds)
        self.chassis = wp.array(chassis_ids, dtype=wp.int32, device=self.model.device)
        self._chassis0 = int(chassis_ids[0])
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.dofs_per_world = self.model.joint_coord_count // self.num_worlds
        self.vel_dofs_per_world = self.model.joint_dof_count // self.num_worlds

        self.vehicles = nv.WheeledVehicles(
            self.model,
            config=nv.WheeledConfig(
                max_wheel_speed=315.0,
                motor_max_torque=1.0,
                angular_damping=0.0005,
                friction=2.0,
                longitudinal_stiffness=20.0,
                lateral_stiffness=40.0,
            ),
        )
        self.vehicles.configure_solver_contacts()

        # --- track generation and spawn placement ------------------------
        # spawn edits model.joint_q, so it must precede solver/state creation
        # (mjData qpos and State.joint_q are initialized from model.joint_q)
        self.track_params = _parse_track_params(getattr(args, "track_param", None))
        self.track, self.track_seed = _generate_track(
            self.num_worlds, args.track_seed, args.track_generator, self.model.device, self.track_params
        )
        self._spawn_on_track()

        # njmax/nconmax are per world; size the shared contact buffer for all worlds
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=192, nconmax=48)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=max(1024, 48 * self.num_worlds))
        self.contacts = self.model.contacts(pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- centerline arc-length progress, speed profile, collision ----

        self.car_pos = wp.zeros(self.num_worlds, dtype=wp.vec2f, device=self.model.device)
        self.car_yaw = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_speed = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_s = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_v_des = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.model.device)
        self.car_half_extents = wp.array(
            np.tile(np.array(CAR_HALF_EXTENTS, dtype=np.float32), (self.num_worlds, 1)),
            dtype=wp.vec2f,
            device=self.model.device,
        )
        center = self._env0_polyline(self.track.center)
        seg_len = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
        self.track_len = float(seg_len.sum())
        cum_s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])]).astype(np.float32)
        v_profile = _speed_profile(center, seg_len)
        self._center = wp.array(center, dtype=wp.vec2f, device=self.model.device)
        self._cum_s = wp.array(cum_s, dtype=wp.float32, device=self.model.device)
        self._v_profile = wp.array(v_profile, dtype=wp.float32, device=self.model.device)
        self._center_count = len(center)
        self.checker = CollisionChecker(self.track, max_boxes=1, method="segments")
        self.checker.bind_inputs(position=self.car_pos, yaw=self.car_yaw, half_extents=self.car_half_extents)

        # --- MPPI planner and plan-cycle buffers -------------------------
        horizon = 8 if self._test_mode else args.horizon
        self.rollout_substeps = 2 if self._test_mode else args.rollout_substeps
        if self.rollout_substeps < 1:
            raise ValueError("rollout-substeps must be >= 1")
        total_substeps = horizon * self.rollout_substeps + self.sim_substeps
        if total_substeps % 2 != 0:
            raise ValueError(
                "horizon * rollout_substeps + 8 must be even so state buffers "
                "return to their starting roles each frame (CUDA graph replay)"
            )
        # brake mode (Step 2b diagnosis, default "none"): the per-step motor
        # servo (negative drive) can only shed ~1 N·m of torque, which
        # decelerates no harder than coasting; both non-default modes give the
        # planner the ESC's ~20 N·m regen/drag brake so it can express real
        # threshold braking into hairpins. "channel" samples a third command;
        # "esc" is RC-transmitter-faithful: braking folds into the throttle
        # axis, negative drive = ESC brake (as on a real VXL, where the
        # negative transmitter half engages brake, not reverse).
        mode_name = getattr(args, "brake_mode", "none") or "none"
        try:
            self._brake_mode = {"none": BRAKE_MODE_NONE, "channel": BRAKE_MODE_CHANNEL, "esc": BRAKE_MODE_ESC}[
                mode_name
            ]
        except KeyError:
            raise ValueError(f"unsupported --brake-mode {mode_name!r}") from None
        dim = 3 if self._brake_mode == BRAKE_MODE_CHANNEL else 2
        # drive in [-0.3, 1.0]: enough negative torque to brake hard for
        # corners (measured backward travel is cost-killed, so reverse cannot
        # become a cruise mode); the drive cap is fully open because the
        # reference speed profile, not the command bound, disciplines
        # corner-entry speed (uncapping measured faster AND smoother).
        # In esc mode the negative half never means reverse (it is remapped to
        # the ESC brake), so the axis opens to the full [-1, 1] transmitter
        # range.
        drive_lo = -1.0 if self._brake_mode == BRAKE_MODE_ESC else -0.3
        sigma = (0.35, 0.45, 0.35)[:dim]
        beta = (0.85, 0.6, 0.6)[:dim]  # brake wants steer-like (lighter) smoothing
        # channel mode: brake is sampled in [-1, 1] and rectified to [0, 1] at
        # the apply kernels (negative half = "no brake") so exploration noise
        # around a zero nominal stays zero-mean instead of ratcheting the
        # brake on
        bounds_lo = (drive_lo, -1.0, -1.0)[:dim]
        bounds_hi = (1.0, 1.0, 1.0)[:dim]
        # spline-knot control parameterization (Task 3, default-off per
        # Decision 3): represent the plan by n_knots coarse control points
        # interpolated up to the horizon. None reproduces the per-step sampler.
        n_knots = getattr(args, "n_knots", None)
        self.planner = nv.ControllerMPPI(
            config=nv.ControllerMPPI.Config(
                num_samples=self.num_worlds,
                horizon=horizon,
                dim=dim,
                sigma=sigma,
                # temperature tuned so the effective sample size sits in the
                # healthy 5-20% of K band (ESS ~30-90 here); at 0.05 the
                # softmax was argmin-degenerate (ESS = 1)
                temperature=15.0,
                # more noise smoothing on drive than steering, per the
                # colored-noise MPPI guidance
                beta=beta,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
            ),
            device=self.model.device,
            _n_knots=n_knots,
        )
        device = self.model.device
        E = self.num_worlds
        self.costs = wp.zeros(E, dtype=wp.float32, device=device)
        self.dead = wp.zeros(E, dtype=wp.int32, device=device)
        self.s_prev = wp.zeros(E, dtype=wp.float32, device=device)
        self.u_prev = wp.zeros(2, dtype=wp.float32, device=device)
        self.start_oob = wp.zeros(E, dtype=wp.int32, device=device)
        self.back_dist = wp.zeros(E, dtype=wp.float32, device=device)
        self.arc = wp.zeros(E, dtype=wp.float32, device=device)
        # [w_progress, w_speed, w_steer, kill, w_rate, w_prox, w_term, w_center, w_under, w_exec, w_stall_arc]
        # w_under defaults to 0 outside esc mode (preserves the none/channel
        # baselines); in esc mode it repairs the braking-trap standstill.
        # w_exec (the t=0 exec-coupling weight) defaults to w_rate*EXEC_COUPLING
        # so behavior is unchanged; the Task-3 removal ablation can zero it.
        # w_stall_arc: trajectory-level progress-shortfall (Task 3c anti-stall
        # hardening), default 0. Unlike the per-step w_under it discriminates a
        # parked rollout from an escaping one -- but measured on the q = 1.2
        # bezier-s0 crawl it is NOT sufficient alone (w up to 200 still parks:
        # from a standstill with a deep-braking nominal no sample's rollout
        # escapes, so the shortfall is a constant offset with no gradient) and
        # NOT needed once the executed-command anti-stall floor exists (w = 0
        # vs 80 identical within noise). Kept as an off-by-default knob and a
        # recorded finding.
        w_under = getattr(args, "w_underspeed", None)
        if w_under is None:
            w_under = 0.5 if self._brake_mode == BRAKE_MODE_ESC else 0.0
        w_stall_arc = getattr(args, "w_stall_arc", None)
        if w_stall_arc is None:
            w_stall_arc = 0.0
        # anti-stall crawl assist (executed drive floor below V_STALL),
        # esc-mode default; see ANTI_STALL_DRIVE
        anti_stall = getattr(args, "anti_stall_drive", None)
        if anti_stall is None:
            anti_stall = ANTI_STALL_DRIVE if self._brake_mode == BRAKE_MODE_ESC else 0.0
        self._anti_stall_drive = float(anti_stall)
        self.stall_frames = wp.zeros(1, dtype=wp.int32, device=device)
        # ESS-aware temperature sharpening (default-off; see _adapt_temperature)
        ess_max = getattr(args, "ess_max", None)
        self._ess_max = float(ess_max) if ess_max is not None else None
        self._adaptive_temp = float(self.planner.config.temperature)
        w_rate = 2.0
        w_exec = w_rate * EXEC_COUPLING
        self.cost_params = wp.array(
            [30.0, 2.0, 0.05, 1000.0, w_rate, 10.0, 20.0, 2.0, float(w_under), w_exec, float(w_stall_arc)],
            dtype=wp.float32,
            device=device,
        )
        # floor arc a rollout should cover to not count as stalled: a mean speed
        # of V_STALL over the horizon (each horizon step advances one frame_dt)
        self._stall_arc_floor = float(V_STALL * horizon * self.frame_dt)
        self.ribbon = wp.zeros(horizon, dtype=wp.vec3, device=device)
        # hero lap odometer (display + tests): unwrapped centerline arc length
        self.total_s = wp.zeros(1, dtype=wp.float32, device=device)
        self.frame_s_prev = wp.zeros(1, dtype=wp.float32, device=device)

        # hero-slice snapshots (world 0 leads every per-world array)
        self.snap_joint_q = wp.zeros(self.dofs_per_world, dtype=wp.float32, device=device)
        self.snap_joint_qd = wp.zeros(self.vel_dofs_per_world, dtype=wp.float32, device=device)
        self.snap_body_q = wp.zeros(self.bodies_per_world, dtype=wp.transform, device=device)
        self.snap_body_qd = wp.zeros(self.bodies_per_world, dtype=wp.spatial_vector, device=device)
        wheels_per_world = self.vehicles.dynamics.omega.shape[0] // E
        self.wheels_per_world = wheels_per_world
        self.snap_omega = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_long = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_trans_lat = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)
        self.snap_fz = wp.zeros(wheels_per_world, dtype=wp.float32, device=device)

        self.graph = None
        self._telemetry = {
            "speed": 0.0,
            "laps": 0,
            "meters": 0.0,
            "v_des": 0.0,
            "ess": 0.0,
            "alive": 1.0,
            "best_cost": 0.0,
            "mean_cost": 0.0,
            "drive": 0.0,
            "steer": 0.0,
            "hero_oob": 0,
        }
        self._nominal_plan = np.zeros((horizon, 2), dtype=np.float32)
        # minimap state: hero trail ring buffer plus last hero pose (host side)
        self._trail = deque(maxlen=MINIMAP_TRAIL_MAX)
        self._hero_xy = (0.0, 0.0)
        self._hero_yaw = 0.0
        self._minimap_ok = True
        self._minimap_boundary_px = None
        # HUD state: planar acceleration in the car frame (g units) and its
        # inputs; mirrors the minimap's graceful-degradation guard
        self._prev_v = None  # previous hero planar velocity [m/s], world frame
        self._accel_car_g = (0.0, 0.0)  # (right, forward) EMA-smoothed [g]
        self._gmeter_trail = deque(maxlen=GMETER_TRAIL_MAX)
        self._hud_ok = True
        self._hud_font = None  # system font faces, loaded lazily on first draw
        self._hud_font_bold = None
        self._hud_font_tried = False
        self.ui_temperature = float(self.planner.config.temperature)
        self.ui_sigma_drive, self.ui_sigma_steer = (float(v) for v in self.planner.config.sigma[:2])
        self.ui_cost = list(self.cost_params.numpy())

        # prime pose/projection buffers so the lap odometer starts at the
        # spawn arc length (eager launches, before any CUDA graph capture)
        self._gather_and_track()
        wp.copy(self.frame_s_prev, self.car_s, count=1)

        self._init_track_render()
        self.follow_camera = True
        self.viewer.set_model(self.model)
        # render only the hero world, exactly where it simulates: all worlds
        # are collocated at the origin, so disable the viewer's automatic
        # per-world grid offsets (they would draw the car away from the track)
        if hasattr(self.viewer, "set_visible_worlds"):
            self.viewer.set_visible_worlds([0])
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self._set_follow_camera()
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 65.0

    # --- track helpers ---------------------------------------------------

    def _env0_polyline(self, flat_vec2):
        count = int(self.track.count.numpy()[0])
        return flat_vec2.numpy()[:count]

    def _spawn_on_track(self):
        center = self._env0_polyline(self.track.center)
        p0, p1 = center[0], center[1]
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        joint_q = self.model.joint_q.numpy()
        joint_type = self.model.joint_type.numpy()
        q_start = self.model.joint_q_start.numpy()
        for j in np.flatnonzero(joint_type == int(newton.JointType.FREE)):
            qs = int(q_start[j])
            joint_q[qs + 0] = p0[0]
            joint_q[qs + 1] = p0[1]
            # keep the authored spawn height joint_q[qs + 2]
            joint_q[qs + 3 : qs + 7] = [q[0], q[1], q[2], q[3]]
        self.model.joint_q.assign(joint_q)

    def _init_track_render(self):
        device = self.model.device
        inner = self._env0_polyline(self.track.inner)
        outer = self._env0_polyline(self.track.outer)

        def _loop_lines(poly, z):
            pts = np.column_stack([poly, np.full(len(poly), z, dtype=np.float32)])
            starts = pts
            ends = np.roll(pts, -1, axis=0)
            return (
                wp.array(starts, dtype=wp.vec3, device=device),
                wp.array(ends, dtype=wp.vec3, device=device),
            )

        self._boundary_lines = [_loop_lines(inner, 0.01), _loop_lines(outer, 0.01)]

        # minimap world->map transform: fit the padded track bounding box into
        # the map square preserving aspect ratio (track is static, so once)
        self._map_inner = inner
        self._map_outer = outer
        pts = np.vstack([inner, outer])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        extent = (hi - lo) * (1.0 + 2.0 * MINIMAP_PAD)
        self._map_center = 0.5 * (lo + hi)
        self._map_scale = MINIMAP_SIZE / max(float(extent[0]), float(extent[1]), 1e-6)

        # cone poses along both boundaries (env 0 only)
        xforms = []
        for boundary in ("inner", "outer"):
            props = PropSampler(self.track, spacing=CONE_SPACING, boundary=boundary, mode="points").sample()
            n = int(props.count.numpy()[0])
            pos = props.position.numpy()[:n]
            yaw = props.yaw.numpy()[:n]
            for p, y in zip(pos, yaw, strict=True):
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(y))
                xforms.append(wp.transform(wp.vec3(float(p[0]), float(p[1]), 0.0), q))
        self._cone_xforms = wp.array(xforms, dtype=wp.transform, device=device)
        self._cone_colors = wp.array([wp.vec3(1.0, 0.35, 0.05)], dtype=wp.vec3, device=device)
        self._cone_mesh = self._load_cone_mesh()

    @staticmethod
    def _load_cone_mesh():
        stage = Usd.Stage.Open(newton.examples.get_asset("cone.usda"))
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/cone"))
        vertices = np.array(usd_mesh.GetPointsAttr().Get())
        indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        mesh = newton.Mesh(vertices, indices)
        mesh.finalize()
        return mesh

    # --- MPPI plan/execute cycle ------------------------------------------

    def _snapshot_hero(self):
        wp.copy(self.snap_joint_q, self.state_0.joint_q, count=self.dofs_per_world)
        wp.copy(self.snap_joint_qd, self.state_0.joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.snap_body_q, self.state_0.body_q, count=self.bodies_per_world)
        wp.copy(self.snap_body_qd, self.state_0.body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(self.snap_omega, dyn.omega, count=self.wheels_per_world)
        wp.copy(self.snap_trans_long, dyn.trans_long, count=self.wheels_per_world)
        wp.copy(self.snap_trans_lat, dyn.trans_lat, count=self.wheels_per_world)
        wp.copy(self.snap_fz, patch.fz, count=self.wheels_per_world)

    def _broadcast_hero(self):
        # Broadcasting joint_q/joint_qd is what actually teleports the particles:
        # SolverMuJoCo re-ingests state joint coords into mjData qpos/qvel every
        # step (update_data_interval=1), so no mujoco_warp.reset_data call is
        # needed on rollout refresh. The only mjData state that survives the
        # teleport is qacc_warmstart (constraint warm start); measured impact on
        # racing metrics is nil, so it is deliberately left untouched.
        dev = self.model.device
        E = self.num_worlds
        wp.launch(
            _broadcast_slice_f32,
            dim=(E, self.dofs_per_world),
            inputs=[self.snap_joint_q, self.dofs_per_world, self.state_0.joint_q],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_f32,
            dim=(E, self.vel_dofs_per_world),
            inputs=[self.snap_joint_qd, self.vel_dofs_per_world, self.state_0.joint_qd],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_tf,
            dim=(E, self.bodies_per_world),
            inputs=[self.snap_body_q, self.bodies_per_world, self.state_0.body_q],
            device=dev,
        )
        wp.launch(
            _broadcast_slice_sv,
            dim=(E, self.bodies_per_world),
            inputs=[self.snap_body_qd, self.bodies_per_world, self.state_0.body_qd],
            device=dev,
        )
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        n = self.wheels_per_world
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_omega, n, dyn.omega], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_long, n, dyn.trans_long], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_trans_lat, n, dyn.trans_lat], device=dev)
        wp.launch(_broadcast_slice_f32, dim=(E, n), inputs=[self.snap_fz, n, patch.fz], device=dev)

    def _restore_hero(self):
        wp.copy(self.state_0.joint_q, self.snap_joint_q, count=self.dofs_per_world)
        wp.copy(self.state_0.joint_qd, self.snap_joint_qd, count=self.vel_dofs_per_world)
        wp.copy(self.state_0.body_q, self.snap_body_q, count=self.bodies_per_world)
        wp.copy(self.state_0.body_qd, self.snap_body_qd, count=self.bodies_per_world)
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        wp.copy(dyn.omega, self.snap_omega, count=self.wheels_per_world)
        wp.copy(dyn.trans_long, self.snap_trans_long, count=self.wheels_per_world)
        wp.copy(dyn.trans_lat, self.snap_trans_lat, count=self.wheels_per_world)
        wp.copy(patch.fz, self.snap_fz, count=self.wheels_per_world)

    def _gather_and_track(self):
        dev = self.model.device
        wp.launch(
            _gather_car_pose,
            dim=self.num_worlds,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.chassis,
                self.car_pos,
                self.car_yaw,
                self.car_speed,
            ],
            device=dev,
        )
        wp.launch(
            _project_centerline,
            dim=self.num_worlds,
            inputs=[self._center, self._cum_s, self._v_profile, self._center_count, self.car_pos],
            outputs=[self.car_s, self.car_v_des],
            device=dev,
        )
        # query() refreshes and returns the same preallocated contact buffers
        self.contact = self.checker.query()

    def _plan_and_execute(self):
        cmd = self.vehicles.commands
        dev = self.model.device
        horizon = self.planner.config.horizon
        rollout_dt = self.frame_dt / self.rollout_substeps

        self._snapshot_hero()
        self._broadcast_hero()
        self.planner.sample()
        wp.launch(
            _zero_plan_buffers, dim=self.num_worlds, inputs=[self.costs, self.dead, self.back_dist, self.arc], device=dev
        )
        self._gather_and_track()
        wp.copy(self.s_prev, self.car_s, count=self.num_worlds)
        wp.launch(_record_start_oob, dim=self.num_worlds, inputs=[self.contact.oob, self.start_oob], device=dev)

        for t in range(horizon):
            wp.launch(
                _apply_sample_commands,
                dim=self.num_worlds,
                inputs=[self.planner.samples, t, self._brake_mode, self.car_speed, cmd.drive, cmd.steer, cmd.brake],
                device=dev,
            )
            for _ in range(self.rollout_substeps):
                self._substep(rollout_dt)
            self._gather_and_track()
            wp.launch(
                _accumulate_cost,
                dim=self.num_worlds,
                inputs=[
                    self.car_s,
                    self.car_v_des,
                    self.car_speed,
                    self.contact.oob,
                    self.contact.distance,
                    self.start_oob,
                    self.planner.samples,
                    t,
                    horizon,
                    self.track_len,
                    self._stall_arc_floor,
                    self.cost_params,
                    self.u_prev,
                    self.s_prev,
                    self.back_dist,
                    self.arc,
                    self.dead,
                    self.costs,
                ],
                device=dev,
            )
            wp.launch(
                _record_ribbon,
                dim=1,
                inputs=[self.state_0.body_q, self._chassis0, t, self.ribbon],
                device=dev,
            )

        self.planner.update(self.costs)
        wp.launch(_store_executed_command, dim=1, inputs=[self.planner.nominal, self.u_prev], device=dev)
        self._restore_hero()
        wp.launch(
            _apply_nominal_command,
            dim=self.num_worlds,
            inputs=[
                self.planner.nominal,
                self._brake_mode,
                self._anti_stall_drive,
                self.stall_frames,
                self.car_speed,
                cmd.drive,
                cmd.steer,
                cmd.brake,
            ],
            device=dev,
        )
        for _ in range(self.sim_substeps):
            self._substep(self.sim_dt)
        self.planner.shift()
        self._gather_and_track()
        # anti-stall dwell counter: advanced once per executed frame, from the
        # hero's post-step speed (rollouts never touch it)
        wp.launch(_update_stall_frames, dim=1, inputs=[self.car_speed, self.stall_frames], device=dev)
        # hero lap odometer for telemetry and tests (rollouts never touch it:
        # it only advances here, once per executed frame)
        wp.launch(
            _accumulate_lap_distance,
            dim=1,
            inputs=[self.car_s, self.track_len, self.frame_s_prev, self.total_s],
            device=dev,
        )

    def step(self):
        if self.graph is None and self.model.device.is_cuda:
            track_gen.set_capturing(True)
            with wp.ScopedCapture() as capture:
                self._plan_and_execute()
            self.graph = capture.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._plan_and_execute()
        self.sim_time += self.frame_dt
        self._update_telemetry()
        self._adapt_temperature()

    def _adapt_temperature(self):
        """ESS-aware softmax temperature sharpening (default-off).

        When the effective sample size exceeds ``--ess-max``, the per-step
        costs are too flat for the configured temperature and the weighted
        update degenerates toward the raw sample mean -- measured failure:
        under Tsallis q > 1 averaging the ESS reaches ~400+ on hairpin
        approach, the braking commitment lags ~12 frames, and the car blows
        the corner. Sharpening (lowering) the temperature restores commitment;
        the temperature relaxes back toward the configured baseline once the
        ESS re-enters the healthy band. Runs on the host between frames and
        writes through :meth:`ControllerMPPI.set_temperature` (a device-array
        write), so the CUDA graph stays valid.
        """
        if self._ess_max is None:
            return
        ess = self._telemetry["ess"]
        base = float(self.planner.config.temperature)
        if ess > self._ess_max:
            self._adaptive_temp = max(self._adaptive_temp * 0.75, 0.2 * base)
        elif ess < 0.5 * self._ess_max and self._adaptive_temp < base:
            self._adaptive_temp = min(self._adaptive_temp * 1.1, base)
        self.planner.set_temperature(self._adaptive_temp)

    def _update_telemetry(self):
        t = self._telemetry
        qd = self.state_0.body_qd.numpy()[self._chassis0]
        t["speed"] = float(np.linalg.norm(qd[:2]))  # linear velocity is entries 0:3
        tf = self.state_0.body_q.numpy()[self._chassis0]
        x, y, z, w = tf[3], tf[4], tf[5], tf[6]
        self._hero_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self._hero_xy = (float(tf[0]), float(tf[1]))
        self._trail.append(self._hero_xy)
        # G-meter: differentiate planar velocity on the host, rotate into the
        # car frame (forward = +y up on screen, right = +x right on screen),
        # convert to g, and EMA-smooth so the broadcast dot stays readable
        v = np.array([qd[0], qd[1]], dtype=np.float64)
        if self._prev_v is not None:
            a = (v - self._prev_v) * self.fps  # world-frame planar accel [m/s^2]
            c, s = math.cos(self._hero_yaw), math.sin(self._hero_yaw)
            a_fwd = (a[0] * c + a[1] * s) / GRAVITY
            a_right = (a[0] * s - a[1] * c) / GRAVITY
            pr, pf = self._accel_car_g
            self._accel_car_g = (
                (1.0 - GMETER_EMA) * pr + GMETER_EMA * a_right,
                (1.0 - GMETER_EMA) * pf + GMETER_EMA * a_fwd,
            )
            self._gmeter_trail.append(self._accel_car_g)
        self._prev_v = v
        total = float(self.total_s.numpy()[0])
        t["meters"] = total
        t["laps"] = int(total // self.track_len) if total >= 0.0 else 0
        t["v_des"] = float(self.car_v_des.numpy()[0])
        t["ess"] = float(self.planner.ess.numpy()[0])
        t["hero_oob"] = int(self.contact.oob.numpy()[0])
        dead = self.dead.numpy()
        costs = self.costs.numpy()
        t["alive"] = 1.0 - float(dead.mean())
        t["best_cost"] = float(costs.min())
        t["mean_cost"] = float(costs.mean())
        nominal = self.planner.nominal.numpy()
        t["drive"], t["steer"] = float(nominal[0, 0]), float(nominal[0, 1])
        self._nominal_plan = nominal

    def gui(self, ui):
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        ui.separator()
        ui.text("Controller output")
        t = self._telemetry
        ui.text(f"Drive: {t['drive']:+.2f}   Steer: {t['steer']:+.2f}")
        ui.plot_lines("drive plan", np.ascontiguousarray(self._nominal_plan[:, 0]))
        ui.plot_lines("steer plan", np.ascontiguousarray(self._nominal_plan[:, 1]))
        ui.separator()
        ui.text("Race")
        ui.text(f"Speed: {t['speed']:.2f} m/s (ref {t['v_des']:.2f})")
        ui.text(f"Laps: {t['laps']}   Distance: {t['meters']:.0f} m")
        ui.text(f"OOB: {t['hero_oob']}")
        ui.separator()
        ui.text("Planner")
        ui.text(f"Alive: {100.0 * t['alive']:.0f}%   ESS: {t['ess']:.0f}")
        ui.text(f"Cost best/mean: {t['best_cost']:.1f} / {t['mean_cost']:.1f}")
        changed_t, self.ui_temperature = ui.slider_float("Temperature", self.ui_temperature, 0.5, 50.0)
        if changed_t:
            self.planner.set_temperature(self.ui_temperature)
        changed_d, self.ui_sigma_drive = ui.slider_float("Sigma drive", self.ui_sigma_drive, 0.05, 1.0)
        changed_s, self.ui_sigma_steer = ui.slider_float("Sigma steer", self.ui_sigma_steer, 0.05, 1.0)
        if changed_d or changed_s:
            new_sigma = self.planner.sigma.numpy()
            new_sigma[0], new_sigma[1] = self.ui_sigma_drive, self.ui_sigma_steer
            self.planner.sigma.assign(new_sigma)
        changed = False
        labels = (
            "W progress",
            "W speed",
            "W steer",
            "Kill penalty",
            "W rate",
            "W proximity",
            "W terminal",
            "W center",
            "W underspeed",
        )
        for i, label in enumerate(labels):
            hi = 1.0 if i == 2 else (2000.0 if i == 3 else (20.0 if i in (1, 4, 7) else 100.0))
            c, self.ui_cost[i] = ui.slider_float(label, self.ui_cost[i], 0.0, hi)
            changed = changed or c
        if changed:
            self.cost_params.assign(np.array(self.ui_cost, dtype=np.float32))
        self._draw_minimap(ui)
        self._draw_gmeter(ui)
        self._draw_hud_bars(ui)

    @staticmethod
    def _hud_load_font(imgui, candidates):
        for path in candidates:
            if Path(path).is_file():
                try:
                    font = imgui.get_io().fonts.add_font_from_file_ttf(path, imgui.get_font_size())
                except Exception:
                    font = None
                if font is not None:
                    return font
        return None

    def _hud_get_font(self, imgui, bold=False):
        """Return the HUD font (system JetBrains Mono or Noto, imgui default)."""
        if not self._hud_font_tried:
            self._hud_font_tried = True
            self._hud_font = self._hud_load_font(imgui, HUD_FONT_CANDIDATES)
            self._hud_font_bold = self._hud_load_font(imgui, HUD_FONT_BOLD_CANDIDATES)
        font = (self._hud_font_bold or self._hud_font) if bold else self._hud_font
        return font if font is not None else imgui.get_font()

    def _hud_text_size(self, imgui, text, font_size, bold=False):
        """Measure ``text`` in the HUD font at ``font_size`` pixels."""
        imgui.push_font(self._hud_get_font(imgui, bold), font_size)
        try:
            return imgui.calc_text_size(text).x
        finally:
            imgui.pop_font()

    def _hud_caption(self, imgui, draw, x, y, text, align="left", span=0.0):
        """Bold white uppercase caption in the broadcast style (1.1x font).

        ``align`` is "left", "right" (text ends at ``x``), or "center"
        (centered within ``span`` pixels starting at ``x``).
        """
        font_size = 1.1 * imgui.get_font_size()
        if align == "right":
            x -= self._hud_text_size(imgui, text, font_size, bold=True)
        elif align == "center":
            x += 0.5 * (span - self._hud_text_size(imgui, text, font_size, bold=True))
        col = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        draw.add_text(self._hud_get_font(imgui, bold=True), font_size, imgui.ImVec2(x, y), col, text)

    @staticmethod
    def _hud_window_flags(imgui):
        return (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_inputs
            | imgui.WindowFlags_.no_nav
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_saved_settings
        )

    def _draw_minimap(self, imgui):
        """Overlay window anchored bottom-right showing track, trail, and car."""
        if not self._minimap_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            y0 = viewport.pos.y + viewport.size.y - MINIMAP_SIZE - MINIMAP_MARGIN
            imgui.set_next_window_pos(imgui.ImVec2(x0, y0))
            imgui.set_next_window_size(imgui.ImVec2(MINIMAP_SIZE, MINIMAP_SIZE))
            imgui.set_next_window_bg_alpha(0.45)
            flags = (
                imgui.WindowFlags_.no_title_bar
                | imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_scrollbar
                | imgui.WindowFlags_.no_collapse
                | imgui.WindowFlags_.no_inputs
                | imgui.WindowFlags_.no_nav
                | imgui.WindowFlags_.no_focus_on_appearing
                | imgui.WindowFlags_.no_saved_settings
            )
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
            try:
                visible = imgui.begin("##track_minimap", None, flags)[0]
                try:
                    if visible:
                        self._draw_minimap_contents(imgui, x0, y0)
                finally:
                    imgui.end()
            finally:
                imgui.pop_style_var()
        except Exception:
            self._minimap_ok = False  # degrade silently on missing imgui API

    def _draw_minimap_contents(self, imgui, x0, y0):
        cx = x0 + 0.5 * MINIMAP_SIZE
        cy = y0 + 0.5 * MINIMAP_SIZE
        mx, my = float(self._map_center[0]), float(self._map_center[1])
        s = self._map_scale

        def to_px(p):
            # imgui y grows down, world y grows up
            return imgui.ImVec2(cx + (p[0] - mx) * s, cy - (p[1] - my) * s)

        draw = imgui.get_window_draw_list()
        gray = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.55, 0.55, 0.6, 0.9))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.66, 1.0, 0.9))  # electric blue
        orange = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.55, 0.1, 1.0))

        # static boundary loops: pixel coords only depend on the window corner
        if self._minimap_boundary_px is None or self._minimap_boundary_px[0] != (x0, y0):
            loops = [[to_px(p) for p in poly] for poly in (self._map_inner, self._map_outer)]
            self._minimap_boundary_px = ((x0, y0), loops)
        for loop in self._minimap_boundary_px[1]:
            draw.add_polyline(loop, gray, imgui.ImDrawFlags_.closed, 1.5)

        if len(self._trail) >= 2:
            step = max(1, len(self._trail) // 600)  # cap per-frame point count
            trail = list(self._trail)[::step]
            if trail[-1] != self._trail[-1]:
                trail.append(self._trail[-1])
            draw.add_polyline([to_px(p) for p in trail], blue, imgui.ImDrawFlags_.none, 1.5)

        car = to_px(self._hero_xy)
        tick = 9.0
        tip = imgui.ImVec2(car.x + tick * math.cos(self._hero_yaw), car.y - tick * math.sin(self._hero_yaw))
        draw.add_line(car, tip, orange, 2.0)
        draw.add_circle_filled(car, 4.0, orange)
        self._hud_caption(imgui, draw, x0 + 8.0, y0 + 6.0, "TRACK")

    def _draw_gmeter(self, imgui):
        """F1-style G-meter anchored just left of the bottom-right minimap."""
        if not self._hud_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            mm_x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            x0 = mm_x0 - HUD_GAP - GMETER_SIZE
            y0 = viewport.pos.y + viewport.size.y - GMETER_SIZE - MINIMAP_MARGIN
            imgui.set_next_window_pos(imgui.ImVec2(x0, y0))
            imgui.set_next_window_size(imgui.ImVec2(GMETER_SIZE, GMETER_SIZE))
            imgui.set_next_window_bg_alpha(0.45)
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
            try:
                visible = imgui.begin("##hud_gmeter", None, self._hud_window_flags(imgui))[0]
                try:
                    if visible:
                        self._draw_gmeter_contents(imgui, x0, y0)
                finally:
                    imgui.end()
            finally:
                imgui.pop_style_var()
        except Exception:
            self._hud_ok = False  # degrade silently on missing imgui API

    def _draw_gmeter_contents(self, imgui, x0, y0):
        cx = x0 + 0.5 * GMETER_SIZE
        cy = y0 + 0.5 * GMETER_SIZE
        radius = 0.5 * GMETER_SIZE - 14.0
        ppg = radius / GMETER_G_EDGE  # pixels per g

        draw = imgui.get_window_draw_list()
        cross = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.75, 0.75, 0.8, 0.8))
        ring = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.6, 0.6, 0.65, 0.35))
        red = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.95, 0.15, 0.15, 1.0))
        trail_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.95, 0.35, 0.35, 0.35))

        center = imgui.ImVec2(cx, cy)
        for g in (1.0, 2.0):  # faint 1 g and 2 g reference rings
            draw.add_circle(center, g * ppg, ring, 48, 1.0)
        draw.add_line(imgui.ImVec2(cx - radius, cy), imgui.ImVec2(cx + radius, cy), cross, 1.0)
        draw.add_line(imgui.ImVec2(cx, cy - radius), imgui.ImVec2(cx, cy + radius), cross, 1.0)

        def to_dot(a_right, a_fwd):
            # +x right = accel to the car's right; +y up = forward accel
            dx = a_right * ppg
            dy = -a_fwd * ppg
            mag = math.hypot(dx, dy)
            if mag > radius:  # clamp the vector to the widget edge
                dx *= radius / mag
                dy *= radius / mag
            return imgui.ImVec2(cx + dx, cy + dy)

        for a_right, a_fwd in self._gmeter_trail:
            draw.add_circle_filled(to_dot(a_right, a_fwd), 2.0, trail_col)
        a_right, a_fwd = self._accel_car_g
        draw.add_circle_filled(to_dot(a_right, a_fwd), 5.0, red)

        self._hud_caption(imgui, draw, x0, y0 + 6.0, "ACCELERATION", align="center", span=GMETER_SIZE)
        mag = math.hypot(a_right, a_fwd)
        self._hud_caption(imgui, draw, x0 + GMETER_SIZE - 8.0, y0 + GMETER_SIZE - 26.0, f"{mag:.1f} G", align="right")

    def _draw_hud_bars(self, imgui):
        """Speed readout + throttle/brake panel left of the G-meter, same height."""
        if not self._hud_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            mm_x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            x0 = mm_x0 - HUD_GAP - GMETER_SIZE - HUD_GAP - HUD_PANEL_W
            y0 = viewport.pos.y + viewport.size.y - GMETER_SIZE - MINIMAP_MARGIN
            imgui.set_next_window_pos(imgui.ImVec2(x0, y0))
            imgui.set_next_window_size(imgui.ImVec2(HUD_PANEL_W, GMETER_SIZE))
            imgui.set_next_window_bg_alpha(0.45)
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
            try:
                visible = imgui.begin("##hud_bars", None, self._hud_window_flags(imgui))[0]
                try:
                    if visible:
                        self._draw_hud_bars_contents(imgui, x0, y0, HUD_PANEL_W)
                finally:
                    imgui.end()
            finally:
                imgui.pop_style_var()
        except Exception:
            self._hud_ok = False  # degrade silently on missing imgui API

    def _draw_hud_bars_contents(self, imgui, x0, y0, width):
        t = self._telemetry
        drive = t["drive"]
        throttle = max(0.0, min(1.0, drive))  # forward drive command in [0, 1]
        brake = max(0.0, min(1.0, -drive))  # motor braking = |negative drive|
        speed_kmh = t["speed"] * 3.6

        draw = imgui.get_window_draw_list()
        white = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.7, 0.7, 0.75, 0.5))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.66, 1.0, 0.95))  # electric blue
        red = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.9, 0.15, 0.15, 0.95))

        pad = 12.0
        bx = x0 + pad
        bw = width - 2.0 * pad

        # top row: SPEED caption + km/h readout centered as one group, the
        # readout drawn at 2x scale via the font+size draw-list overload
        # (set_window_font_scale is absent in this imgui binding)
        label = f"{int(round(speed_kmh))} km/h"
        font_size = 2.0 * imgui.get_font_size()
        cap_gap = 14.0
        num_w = self._hud_text_size(imgui, label, font_size, bold=True)
        cap_w = self._hud_text_size(imgui, "SPEED", 1.1 * imgui.get_font_size(), bold=True)
        gx = x0 + 0.5 * (width - cap_w - cap_gap - num_w)
        self._hud_caption(imgui, draw, gx, y0 + 14.0, "SPEED")
        draw.add_text(
            self._hud_get_font(imgui, bold=True),
            font_size,
            imgui.ImVec2(gx + cap_w + cap_gap, y0 + 8.0),
            white,
            label,
        )

        # labels above full-width segmented LED bars: dim outlines mark the
        # full range, lit segments grow left->right (the last one fills
        # proportionally)
        bar_h = 14.0
        seg_gap = 3.0
        seg_w = (bw - (HUD_BAR_SEGMENTS - 1) * seg_gap) / HUD_BAR_SEGMENTS
        rows = ((y0 + 52.0, throttle, blue, "THROTTLE"), (y0 + 100.0, brake, red, "BRAKE"))
        for label_y, frac, fill, name in rows:
            self._hud_caption(imgui, draw, x0 + pad, label_y, name)
            y = label_y + 1.2 * imgui.get_font_size() + 4.0
            for i in range(HUD_BAR_SEGMENTS):
                sx = bx + i * (seg_w + seg_gap)
                draw.add_rect(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w, y + bar_h), outline, 1.0)
                lit = max(0.0, min(1.0, frac * HUD_BAR_SEGMENTS - i))
                if lit > 0.0:
                    draw.add_rect_filled(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w * lit, y + bar_h), fill, 1.0)

    def _substep(self, dt):
        self.state_0.clear_forces()
        self.vehicles.update_controls(self.control)
        self.model.collide(self.state_0, self.contacts)
        self.vehicles.apply(self.state_0, self.contacts, dt)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.vehicles.latch_loads(self.contacts)
        self.state_0, self.state_1 = self.state_1, self.state_0

    # --- rendering -------------------------------------------------------

    def _set_follow_camera(self):
        if not hasattr(self.viewer, "set_camera"):
            return
        tf = self.state_0.body_q.numpy()[self._chassis0]
        x, y, z, w = tf[3], tf[4], tf[5], tf[6]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        cam = tf[:3] - 2.8 * forward + np.array([0.0, 0.0, 1.25], dtype=np.float32)
        self.viewer.set_camera(
            pos=wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])),
            pitch=-18.0,
            yaw=math.degrees(yaw),
        )

    def render(self):
        if self.follow_camera:
            self._set_follow_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes(
            "/track/cones",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self._cone_xforms,
            colors=self._cone_colors,
            geo_src=self._cone_mesh,
        )
        for i, (starts, ends) in enumerate(self._boundary_lines):
            self.viewer.log_lines(f"/track/boundary_{i}", starts, ends, (0.35, 0.35, 0.4))
        horizon = self.planner.config.horizon
        self.viewer.log_lines("/mppi/plan", self.ribbon[: horizon - 1], self.ribbon[1:horizon], (0.1, 0.9, 0.3))
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.body_q.numpy()[: self.bodies_per_world]).all():
            raise ValueError("non-finite hero poses")
        if not np.isfinite(self.costs.numpy()).all():
            raise ValueError("non-finite MPPI costs")

    def test_final(self):
        hero_q = self.state_0.body_q.numpy()[self._chassis0]
        if not np.isfinite(hero_q).all():
            raise ValueError("non-finite hero pose")
        meters = float(self.total_s.numpy()[0])
        if meters < 4.0:
            raise ValueError(f"hero covered only {meters:.2f} m of centerline")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # lap pace saturates at K=512, but steering smoothness keeps
        # improving with survivors-per-update: 1024 buys -19% steering jitter
        # for -6% fps (2048: -30% jitter, -20% fps) — 1024 is the sweet spot
        parser.add_argument(
            "--num-samples",
            type=int,
            default=1024,
            help="MPPI samples K (= simulated worlds; sample 0 is the hero)",
        )
        parser.add_argument("--horizon", type=int, default=48, help="MPPI planning horizon in control steps")
        parser.add_argument(
            "--n-knots",
            type=int,
            default=None,
            help="experimental: represent the plan by N coarse control knots interpolated to the "
            "horizon (default: None = per-step control)",
        )
        parser.add_argument("--rollout-substeps", type=int, default=4, help="solver substeps per rollout control step")
        parser.add_argument("--track-seed", type=int, default=0, help="base seed for track generation")
        parser.add_argument(
            "--w-underspeed",
            type=float,
            default=None,
            help="under-speed shortfall cost weight; default: 0.5 in esc brake mode "
            "(repairs the braking standstill trap), 0.0 otherwise",
        )
        parser.add_argument(
            "--w-stall-arc",
            type=float,
            default=None,
            help="trajectory-level progress-shortfall cost weight (anti-stall); "
            "default 0.0 (measured: neither sufficient nor needed once the "
            "anti-stall drive floor is active)",
        )
        parser.add_argument(
            "--anti-stall-drive",
            type=float,
            default=None,
            help="executed-command drive floor while the hero is below the stall speed "
            "(crawl assist; never applied to sampled rollouts); default: 0.5 in esc "
            "brake mode, 0.0 otherwise",
        )
        parser.add_argument(
            "--ess-max",
            type=float,
            default=None,
            help="ESS-aware temperature sharpening (default: off): when the effective "
            "sample size exceeds this, the softmax temperature is lowered between "
            "frames until the ESS re-enters the band (restores update commitment "
            "when per-step costs are too flat, e.g. under Tsallis q > 1 averaging)",
        )
        parser.add_argument(
            "--brake-mode",
            type=str,
            default="none",
            choices=("none", "channel", "esc"),
            help="how the planner reaches the ESC regen brake: none (today: brake "
            "always 0), channel (third sampled command), or esc (RC-transmitter-"
            "faithful: negative drive engages the brake instead of reverse)",
        )
        parser.add_argument(
            "--track-generator",
            type=str,
            default="bezier",
            choices=sorted(TRACK_GENERATOR_SETTINGS),
            help="track_gen generator family",
        )
        parser.add_argument(
            "--track-param",
            action="append",
            metavar="KEY=VALUE",
            help="generator-specific TrackGenConfig override (repeatable), e.g. --track-param hull_displacement=0.35",
        )
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
