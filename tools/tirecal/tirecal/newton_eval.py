# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched segment-replay evaluator: (candidate x segment) -> replicated worlds.

The only module the optimization loop needs newton for. Loads a recorded
session, slices laps into fixed windows, resets each world to a recorded
window start, replays the recorded controls open-loop under candidate Magic
Formula parameters, and accumulates weighted state error against the
recording.

Reference conventions (verified empirically against the recording and the
MPPI example's ``_gather_car_pose``): ``body_qd`` carries world-frame linear
velocity in entries 0:3 and angular velocity in entries 3:6, so the planar
reference velocity is ``body_qd[:, 0:2]`` and the yaw rate ``body_qd[:, 5]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class EvalConfig:
    """Replay/loss configuration. Weights multiply squared per-step errors."""

    window_s: float = 2.0
    stride_s: float = 1.0
    max_worlds: int = 4096
    w_pos: float = 1.0
    w_yaw: float = 2.0
    w_vel: float = 0.3
    w_yawrate: float = 0.1
    tire_model: str = "pacejka"  # "brush" runs the recorded generator params (self-test)


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.kernel
def _scatter_slice_f32(
    src: wp.array[wp.float32],
    seg_of_world: wp.array[wp.int32],
    n_per: int,
    dst: wp.array[wp.float32],
):
    w, i = wp.tid()
    dst[w * n_per + i] = src[seg_of_world[w] * n_per + i]


@wp.kernel
def _scatter_slice_tf(
    src: wp.array[wp.transform],
    seg_of_world: wp.array[wp.int32],
    n_per: int,
    dst: wp.array[wp.transform],
):
    w, i = wp.tid()
    dst[w * n_per + i] = src[seg_of_world[w] * n_per + i]


@wp.kernel
def _scatter_slice_sv(
    src: wp.array[wp.spatial_vector],
    seg_of_world: wp.array[wp.int32],
    n_per: int,
    dst: wp.array[wp.spatial_vector],
):
    w, i = wp.tid()
    dst[w * n_per + i] = src[seg_of_world[w] * n_per + i]


@wp.kernel
def _zero_error_buffers(cost: wp.array[wp.float32], pos_err: wp.array[wp.float32]):
    w = wp.tid()
    cost[w] = 0.0
    pos_err[w] = 0.0


@wp.kernel
def _apply_recorded_commands(
    ctrl: wp.array3d[wp.float32],
    seg_of_world: wp.array[wp.int32],
    t: int,
    drive: wp.array[wp.float32],
    steer: wp.array[wp.float32],
    brake: wp.array[wp.float32],
):
    w = wp.tid()
    s = seg_of_world[w]
    drive[w] = ctrl[s, t, 0]
    steer[w] = ctrl[s, t, 1]
    brake[w] = ctrl[s, t, 2]


@wp.kernel
def _accumulate_error(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    chassis: wp.array[wp.int32],
    seg_of_world: wp.array[wp.int32],
    ref_pos: wp.array[wp.vec2f],
    ref_yaw: wp.array[wp.float32],
    ref_vel: wp.array[wp.vec2f],
    ref_yawrate: wp.array[wp.float32],
    t: int,
    window_steps: int,
    w_pos: float,
    w_yaw: float,
    w_vel: float,
    w_yawrate: float,
    cost: wp.array[wp.float32],
    pos_err: wp.array[wp.float32],
):
    w = wp.tid()
    idx = seg_of_world[w] * window_steps + t
    tf = body_q[chassis[w]]
    p = wp.transform_get_translation(tf)
    dp = wp.vec2f(p[0], p[1]) - ref_pos[idx]
    dyaw = _quat_yaw(wp.transform_get_rotation(tf)) - ref_yaw[idx]
    dyaw = wp.atan2(wp.sin(dyaw), wp.cos(dyaw))
    qd = body_qd[chassis[w]]
    # linear velocity is entries 0:3 (world frame), angular is 3:6
    dv = wp.vec2f(qd[0], qd[1]) - ref_vel[idx]
    dwz = qd[5] - ref_yawrate[idx]
    d2 = wp.dot(dp, dp)
    cost[w] = cost[w] + w_pos * d2 + w_yaw * dyaw * dyaw + w_vel * wp.dot(dv, dv) + w_yawrate * dwz * dwz
    pos_err[w] = pos_err[w] + wp.sqrt(d2)


def _lap_yaw(body_q: np.ndarray) -> np.ndarray:
    """Chassis yaw [rad] from recorded (x, y, z, qx, qy, qz, qw) rows."""
    qx, qy, qz, qw = (body_q[:, 3 + i].astype(np.float64) for i in range(4))
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class ReplayEvaluator:
    """Scores Magic Formula candidates by open-loop replay against a recording.

    Maps (candidate x segment) onto replicated simulation worlds: each world is
    reset to a recorded window start, replays the recorded controls, and
    accumulates weighted trajectory error against the recorded reference.

    Args:
        session: A :class:`tirecal.format.Session` recorded with ``sim_state``.
        cfg: Replay/loss configuration.
    """

    def __init__(self, session, cfg: EvalConfig):
        import newton  # noqa: PLC0415 (deferred: keep newton imports out of module import)
        import newton.vehicles as nv  # noqa: PLC0415
        from newton.examples.vehicles.example_vehicle_mppi_track import _build_model  # noqa: PLC0415

        if cfg.tire_model not in ("pacejka", "brush"):
            raise ValueError(f"unsupported tire_model {cfg.tire_model!r}; choose 'pacejka' or 'brush'")
        self.cfg = cfg
        meta = session.vehicle
        self.frame_dt = float(meta["frame_dt"])
        self.sim_substeps = int(meta["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._window_steps = max(1, int(round(cfg.window_s / self.frame_dt)))
        stride_steps = max(1, int(round(cfg.stride_s / self.frame_dt)))

        # --- slice laps into fixed replay windows -------------------------
        laps = [lap for lap in session.laps if lap.sim_state]
        if not laps:
            raise ValueError("no lap carries sim_state; v1 replay requires exact recorded resets")
        segments = []  # (lap, t0)
        for lap in laps:
            frames = int(lap.controls.shape[0])
            for t0 in range(0, frames - self._window_steps + 1, stride_steps):
                segments.append((lap, t0))
        if not segments:
            raise ValueError(
                f"no lap is long enough for one {self._window_steps}-step window "
                f"({cfg.window_s} s at {1.0 / self.frame_dt:.0f} Hz)"
            )
        S = len(segments)
        T = self._window_steps
        self._num_segments = S
        self.candidates_per_chunk = max(1, cfg.max_worlds // S)
        self.num_worlds = self.candidates_per_chunk * S

        # --- replicated worlds, vehicle layer, solver (mirrors the example) --
        self.model, chassis_ids = _build_model(self.num_worlds)
        device = self.model.device
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.dofs_per_world = self.model.joint_coord_count // self.num_worlds
        self.vel_dofs_per_world = self.model.joint_dof_count // self.num_worlds
        self.wheels_per_world = int(meta["wheels_per_world"])
        if self.bodies_per_world != int(meta["bodies_per_world"]):
            raise ValueError("recorded bodies_per_world does not match the rebuilt model")
        if self.dofs_per_world != int(meta["dofs_per_world"]):
            raise ValueError("recorded dofs_per_world does not match the rebuilt model")
        if self.vel_dofs_per_world != int(meta["vel_dofs_per_world"]):
            raise ValueError("recorded vel_dofs_per_world does not match the rebuilt model")
        if int(chassis_ids[0]) != int(meta["chassis_index"]):
            raise ValueError("recorded chassis_index does not match the rebuilt model")
        self.chassis = wp.array(chassis_ids, dtype=wp.int32, device=device)

        wcfg_dict = dict(meta["config"])
        wcfg_dict.pop("pacejka", None)
        wcfg = nv.WheeledConfig(**wcfg_dict)
        wcfg.pacejka = nv.PacejkaParams(**meta["pacejka_defaults"])
        wcfg.tire_model = int(nv.TireModel.PACEJKA if cfg.tire_model == "pacejka" else nv.TireModel.BRUSH)
        self.vehicles = nv.WheeledVehicles(self.model, config=wcfg)
        self.vehicles.configure_solver_contacts()

        # njmax/nconmax are per world; size the shared contact buffer for all worlds
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=192, nconmax=48)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=max(1024, 48 * self.num_worlds))
        self.contacts = self.model.contacts(pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- per-segment device data (built once) -------------------------
        def f32(rows, n_per):
            return wp.array(
                np.ascontiguousarray(np.stack(rows).reshape(S * n_per).astype(np.float32)),
                dtype=wp.float32,
                device=device,
            )

        st = [lap.sim_state for lap, _ in segments]
        t0s = [t0 for _, t0 in segments]
        self._seg_joint_q = f32([s["joint_q"][t0] for s, t0 in zip(st, t0s, strict=True)], self.dofs_per_world)
        self._seg_joint_qd = f32([s["joint_qd"][t0] for s, t0 in zip(st, t0s, strict=True)], self.vel_dofs_per_world)
        body_q = np.stack([s["body_q_full"][t0] for s, t0 in zip(st, t0s, strict=True)]).reshape(
            S * self.bodies_per_world, 7
        )
        self._seg_body_q = wp.array(body_q.astype(np.float32), dtype=wp.transform, device=device)
        body_qd = np.stack([s["body_qd_full"][t0] for s, t0 in zip(st, t0s, strict=True)]).reshape(
            S * self.bodies_per_world, 6
        )
        self._seg_body_qd = wp.array(body_qd.astype(np.float32), dtype=wp.spatial_vector, device=device)
        self._seg_omega = f32([lap.wheel_omega[t0] for lap, t0 in segments], self.wheels_per_world)
        self._seg_trans_long = f32([s["trans_long"][t0] for s, t0 in zip(st, t0s, strict=True)], self.wheels_per_world)
        self._seg_trans_lat = f32([s["trans_lat"][t0] for s, t0 in zip(st, t0s, strict=True)], self.wheels_per_world)
        self._seg_fz = f32([s["patch_fz"][t0] for s, t0 in zip(st, t0s, strict=True)], self.wheels_per_world)

        ctrl = np.stack([lap.controls[t0 : t0 + T] for lap, t0 in segments])  # [S, T, 3]
        self._ctrl = wp.array(np.ascontiguousarray(ctrl.astype(np.float32)), dtype=wp.float32, device=device)

        # references from samples t0+1 .. t0+T (controls[t] drives t -> t+1)
        ref_q = np.stack([lap.body_q[t0 + 1 : t0 + T + 1] for lap, t0 in segments]).reshape(S * T, 7)
        ref_qd = np.stack([lap.body_qd[t0 + 1 : t0 + T + 1] for lap, t0 in segments]).reshape(S * T, 6)
        self._ref_pos = wp.array(ref_q[:, :2].astype(np.float32), dtype=wp.vec2f, device=device)
        self._ref_yaw = wp.array(_lap_yaw(ref_q).astype(np.float32), dtype=wp.float32, device=device)
        self._ref_vel = wp.array(ref_qd[:, :2].astype(np.float32), dtype=wp.vec2f, device=device)
        self._ref_yawrate = wp.array(ref_qd[:, 5].astype(np.float32), dtype=wp.float32, device=device)

        self._seg_of_world = wp.array((np.arange(self.num_worlds) % S).astype(np.int32), dtype=wp.int32, device=device)
        self._cost = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self._pos_err = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self._graph = None

    @property
    def num_segments(self) -> int:
        """Number of replay segments sliced from the session's laps."""
        return self._num_segments

    @property
    def window_steps(self) -> int:
        """Replay window length in control steps."""
        return self._window_steps

    # --- device-side replay ------------------------------------------------

    def _substep(self, dt: float) -> None:
        # verbatim mirror of the MPPI example's substep sequence
        self.state_0.clear_forces()
        self.vehicles.update_controls(self.control)
        self.model.collide(self.state_0, self.contacts)
        self.vehicles.apply(self.state_0, self.contacts, dt)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.vehicles.latch_loads(self.contacts)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def _scatter_resets(self) -> None:
        dev = self.model.device
        E = self.num_worlds
        for src, n_per, dst in (
            (self._seg_joint_q, self.dofs_per_world, self.state_0.joint_q),
            (self._seg_joint_qd, self.vel_dofs_per_world, self.state_0.joint_qd),
        ):
            wp.launch(_scatter_slice_f32, dim=(E, n_per), inputs=[src, self._seg_of_world, n_per, dst], device=dev)
        wp.launch(
            _scatter_slice_tf,
            dim=(E, self.bodies_per_world),
            inputs=[self._seg_body_q, self._seg_of_world, self.bodies_per_world, self.state_0.body_q],
            device=dev,
        )
        wp.launch(
            _scatter_slice_sv,
            dim=(E, self.bodies_per_world),
            inputs=[self._seg_body_qd, self._seg_of_world, self.bodies_per_world, self.state_0.body_qd],
            device=dev,
        )
        dyn, patch = self.vehicles.dynamics, self.vehicles.patch
        n = self.wheels_per_world
        for src, dst in (
            (self._seg_omega, dyn.omega),
            (self._seg_trans_long, dyn.trans_long),
            (self._seg_trans_lat, dyn.trans_lat),
            (self._seg_fz, patch.fz),
        ):
            wp.launch(_scatter_slice_f32, dim=(E, n), inputs=[src, self._seg_of_world, n, dst], device=dev)
        wp.launch(_zero_error_buffers, dim=E, inputs=[self._cost, self._pos_err], device=dev)

    def _replay_window(self) -> None:
        dev = self.model.device
        E = self.num_worlds
        cmd = self.vehicles.commands
        self._scatter_resets()
        for t in range(self._window_steps):
            wp.launch(
                _apply_recorded_commands,
                dim=E,
                inputs=[self._ctrl, self._seg_of_world, t, cmd.drive, cmd.steer, cmd.brake],
                device=dev,
            )
            for _ in range(self.sim_substeps):
                self._substep(self.sim_dt)
            wp.launch(
                _accumulate_error,
                dim=E,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.chassis,
                    self._seg_of_world,
                    self._ref_pos,
                    self._ref_yaw,
                    self._ref_vel,
                    self._ref_yawrate,
                    t,
                    self._window_steps,
                    self.cfg.w_pos,
                    self.cfg.w_yaw,
                    self.cfg.w_vel,
                    self.cfg.w_yawrate,
                    self._cost,
                    self._pos_err,
                ],
                device=dev,
            )

    def _run_generation(self) -> None:
        """Run one full replay (scatter resets + window) for the current mf_params."""
        if self.model.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture() as capture:
                    self._replay_window()
                self._graph = capture.graph
            wp.capture_launch(self._graph)
        else:
            self._replay_window()

    def _write_params(self, chunk: np.ndarray) -> None:
        """Stamp candidate rows into per-wheel mf_params (host-side, once per chunk)."""
        if self.cfg.tire_model == "brush":
            return  # self-test path: the recorded brush parameters drive the tires
        dyn = self.vehicles.dynamics
        assert dyn.mf_params.shape == (self.num_worlds * self.wheels_per_world, 10)
        # wheel i belongs to world i // wheels_per_world; world w runs candidate
        # w // num_segments, so each candidate row repeats S * wheels_per_world times
        rows = np.repeat(chunk.astype(np.float32), self._num_segments * self.wheels_per_world, axis=0)
        dyn.mf_params.assign(rows)

    def _run_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replay one padded chunk; returns host (cost, pos_err), each [num_worlds]."""
        self._write_params(chunk)
        self._run_generation()
        return self._cost.numpy(), self._pos_err.numpy()

    # --- public API ---------------------------------------------------------

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        """Mean weighted per-step segment error per candidate (lower is better).

        Args:
            params: Candidate Magic Formula parameters, shape [P, 10], columns
                in :func:`newton.vehicles.PacejkaParams.names` order.

        Returns:
            Fitness per candidate, shape [P]: mean over the candidate's segments
            of the accumulated weighted error, divided by ``window_steps``.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != 10:
            raise ValueError(f"params must have shape (P, 10), got {params.shape}")
        P = params.shape[0]
        C = self.candidates_per_chunk
        S = self._num_segments
        fitness = np.empty(P, dtype=np.float64)
        for start in range(0, P, C):
            chunk = params[start : start + C]
            n = chunk.shape[0]
            if n < C:  # pad the last chunk by repeating row 0; padded fitness discarded
                chunk = np.concatenate([chunk, np.repeat(params[:1], C - n, axis=0)])
            cost, _ = self._run_chunk(chunk)
            fitness[start : start + n] = cost.reshape(C, S).mean(axis=1)[:n] / self._window_steps
        return fitness

    def mean_position_error(self, params_row: np.ndarray) -> float:
        """Mean per-step chassis position error [m] of one candidate over all segments.

        Args:
            params_row: One candidate's Magic Formula parameters, shape [10]
                (ignored when ``cfg.tire_model == "brush"``).
        """
        params_row = np.asarray(params_row, dtype=np.float64).reshape(1, 10)
        chunk = np.repeat(params_row, self.candidates_per_chunk, axis=0)
        _, pos_err = self._run_chunk(chunk)
        # worlds 0 .. S-1 hold candidate 0's segments
        return float(pos_err[: self._num_segments].mean() / self._window_steps)
