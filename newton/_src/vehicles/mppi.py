# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sampling-based MPPI planner over batched simulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@wp.kernel
def _sample_sequences(
    nominal: wp.array2d[float],
    sigma: wp.array[float],
    bounds_lo: wp.array[float],
    bounds_hi: wp.array[float],
    beta: wp.array[float],
    seed: int,
    counter: wp.array[wp.int32],
    noise: wp.array3d[float],
    samples: wp.array3d[float],
):
    k, a = wp.tid()
    horizon = nominal.shape[0]
    if k == 0:
        # sample 0 is always the zero-noise nominal
        for t in range(horizon):
            noise[0, t, a] = 0.0
            samples[0, t, a] = nominal[t, a]
        return
    # counter advances the offset (not the seed) so each planner seed is an
    # independent stream: frame n+1 of seed s never replays frame n of seed s+1
    state = wp.rand_init(seed, (counter[0] * noise.shape[0] + k) * nominal.shape[1] + a)
    b = beta[0]
    scale = wp.sqrt(wp.max(0.0, 1.0 - b * b))
    n = float(0.0)
    for t in range(horizon):
        eps = sigma[a] * wp.randn(state)
        if t == 0:
            n = eps
        else:
            n = b * n + scale * eps
        s = wp.clamp(nominal[t, a] + n, bounds_lo[a], bounds_hi[a])
        samples[k, t, a] = s
        # effective (post-clamp) noise so update() respects the bounds
        noise[k, t, a] = s - nominal[t, a]


@wp.kernel
def _advance_counter(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


# the reductions below are deliberately single-thread O(K) loops: K is small
# (hundreds to a few thousand) and a deterministic loop keeps update() free of
# atomics, so results are bit-stable across runs and CUDA graph replays
@wp.kernel
def _min_cost(costs: wp.array[float], out: wp.array[float]):
    m = costs[0]
    for k in range(1, costs.shape[0]):
        m = wp.min(m, costs[k])
    out[0] = m


@wp.kernel
def _softmax_weights(
    costs: wp.array[float],
    min_cost: wp.array[float],
    temperature: wp.array[float],
    weights: wp.array[float],
):
    k = wp.tid()
    weights[k] = wp.exp(-(costs[k] - min_cost[0]) / wp.max(temperature[0], 1.0e-6))


@wp.kernel
def _sum_weights(weights: wp.array[float], out: wp.array[float]):
    s = float(0.0)
    for k in range(weights.shape[0]):
        s += weights[k]
    out[0] = s


@wp.kernel
def _update_nominal(
    noise: wp.array3d[float],
    weights: wp.array[float],
    weight_sum: wp.array[float],
    bounds_lo: wp.array[float],
    bounds_hi: wp.array[float],
    nominal: wp.array2d[float],
):
    t, a = wp.tid()
    acc = float(0.0)
    for k in range(weights.shape[0]):
        acc += weights[k] * noise[k, t, a]
    nominal[t, a] = wp.clamp(nominal[t, a] + acc / wp.max(weight_sum[0], 1.0e-9), bounds_lo[a], bounds_hi[a])


@wp.kernel
def _shift_nominal(nominal: wp.array2d[float]):
    a = wp.tid()
    for t in range(nominal.shape[0] - 1):
        nominal[t, a] = nominal[t + 1, a]


class ControllerMPPI:
    """Model Predictive Path Integral planner over externally simulated rollouts.

    The planner owns the nominal command sequence and the sampled candidate
    sequences; the caller owns the rollouts and the per-sample cost array.
    One replan cycle is::

        planner.sample()  # fill planner.samples [K, H, A]
        costs = rollout(...)  # caller: simulate sample k, accumulate cost[k]
        planner.update(costs)  # softmax-weighted update of the nominal
        command = planner.nominal  # execute row 0, then
        planner.shift()  # warm-start the next cycle

    All methods launch Warp kernels only (no host synchronization), so the
    full cycle can be recorded into a CUDA graph. Temperature and noise
    smoothing live in device arrays and remain adjustable while captured via
    :meth:`set_temperature` and :meth:`set_beta`.
    """

    @dataclass
    class Config:
        num_samples: int = 1024
        """Number of sampled command sequences K. Sample 0 is always the zero-noise nominal."""
        horizon: int = 32
        """Planning horizon H in control steps."""
        dim: int = 2
        """Number of command channels A per step."""
        sigma: tuple[float, ...] = (0.3, 0.4)
        """Per-channel exploration noise standard deviation, length ``dim``."""
        temperature: float = 0.05
        """Softmax temperature; lower concentrates the update on the best samples."""
        beta: float = 0.7
        """Per-step noise smoothing in [0, 1); 0 is white noise."""
        bounds_lo: tuple[float, ...] = (-1.0, -1.0)
        """Per-channel lower command bounds, length ``dim``."""
        bounds_hi: tuple[float, ...] = (1.0, 1.0)
        """Per-channel upper command bounds, length ``dim``."""
        seed: int = 0
        """Base RNG seed; resampling advances an internal device counter."""

    def __init__(self, config: Config | None = None, device: wp.context.Device | str | None = None):
        cfg = config if config is not None else ControllerMPPI.Config()
        if cfg.num_samples < 2:
            raise ValueError("num_samples must be >= 2 (sample 0 is the nominal)")
        if cfg.horizon < 2:
            raise ValueError("horizon must be >= 2")
        for name in ("sigma", "bounds_lo", "bounds_hi"):
            if len(getattr(cfg, name)) != cfg.dim:
                raise ValueError(f"{name} must have length dim={cfg.dim}")
        if not 0.0 <= cfg.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.config = cfg
        self.device = wp.get_device(device)
        k, h, a = cfg.num_samples, cfg.horizon, cfg.dim
        with wp.ScopedDevice(self.device):
            self.nominal = wp.zeros((h, a), dtype=wp.float32)
            self.samples = wp.zeros((k, h, a), dtype=wp.float32)
            self.noise = wp.zeros((k, h, a), dtype=wp.float32)
            self.weights = wp.zeros(k, dtype=wp.float32)
            self.sigma = wp.array(cfg.sigma, dtype=wp.float32)
            self.bounds_lo = wp.array(cfg.bounds_lo, dtype=wp.float32)
            self.bounds_hi = wp.array(cfg.bounds_hi, dtype=wp.float32)
            self._temperature = wp.array([cfg.temperature], dtype=wp.float32)
            self._beta = wp.array([cfg.beta], dtype=wp.float32)
            self._counter = wp.zeros(1, dtype=wp.int32)
            self._min_cost = wp.zeros(1, dtype=wp.float32)
            self._weight_sum = wp.zeros(1, dtype=wp.float32)

    def set_temperature(self, value: float) -> None:
        """Sets the softmax temperature (safe while a CUDA graph is captured)."""
        self._temperature.fill_(float(value))

    def set_beta(self, value: float) -> None:
        """Sets the noise smoothing factor (safe while a CUDA graph is captured)."""
        self._beta.fill_(float(value))

    def sample(self) -> None:
        """Fills :attr:`samples` with the clamped nominal plus smoothed Gaussian noise."""
        cfg = self.config
        wp.launch(
            _sample_sequences,
            dim=(cfg.num_samples, cfg.dim),
            inputs=[
                self.nominal,
                self.sigma,
                self.bounds_lo,
                self.bounds_hi,
                self._beta,
                cfg.seed,
                self._counter,
            ],
            outputs=[self.noise, self.samples],
            device=self.device,
        )
        wp.launch(_advance_counter, dim=1, inputs=[self._counter], device=self.device)

    def update(self, costs: wp.array[float]) -> None:
        """Applies the MPPI softmax-weighted noise average to the nominal.

        Args:
            costs: Per-sample accumulated rollout costs, shape [num_samples].

        Raises:
            ValueError: If ``costs`` does not have shape [num_samples] or lives
                on a different device than the planner.
        """
        cfg = self.config
        if costs.shape[0] != cfg.num_samples:
            raise ValueError(f"costs must have shape [{cfg.num_samples}], got {costs.shape}")
        if costs.device != self.device:
            raise ValueError(f"costs must live on device {self.device}, got {costs.device}")
        wp.launch(_min_cost, dim=1, inputs=[costs, self._min_cost], device=self.device)
        wp.launch(
            _softmax_weights,
            dim=cfg.num_samples,
            inputs=[costs, self._min_cost, self._temperature, self.weights],
            device=self.device,
        )
        wp.launch(_sum_weights, dim=1, inputs=[self.weights, self._weight_sum], device=self.device)
        wp.launch(
            _update_nominal,
            dim=(cfg.horizon, cfg.dim),
            inputs=[self.noise, self.weights, self._weight_sum, self.bounds_lo, self.bounds_hi, self.nominal],
            device=self.device,
        )

    def shift(self) -> None:
        """Rolls the nominal one step forward, repeating the final row."""
        wp.launch(_shift_nominal, dim=self.config.dim, inputs=[self.nominal], device=self.device)
