# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sampling-based MPPI planner over batched simulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _sample_sequences(
    nominal: wp.array2d[float],
    sigma: wp.array[float],
    sigma_schedule: wp.array[float],
    bounds_lo: wp.array[float],
    bounds_hi: wp.array[float],
    beta: wp.array[float],
    seed: int,
    counter: wp.array[wp.int32],
    zero_mean_count: wp.array[wp.int32],
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
    b = beta[a]
    scale = wp.sqrt(wp.max(0.0, 1.0 - b * b))
    # RA-MPPI zero-mean fraction: the first zero_mean_count non-hero samples
    # draw pure smoothed noise (u = eps) instead of nominal + eps, which pulls
    # the softmax average back toward zero and damps accumulated nominal drift.
    # Default 0 reproduces the plain sampler bit-for-bit. The stored noise is
    # ALWAYS the deviation from nominal (s - nominal), so update() -- which
    # adds the weighted average of noise to nominal -- stays consistent whether
    # or not a sample is zero-mean.
    zero_mean = k <= zero_mean_count[0]
    n = float(0.0)
    for t in range(horizon):
        eps = sigma[a] * sigma_schedule[t] * wp.randn(state)
        if t == 0:
            n = eps
        else:
            n = b * n + scale * eps
        if zero_mean:
            s = wp.clamp(n, bounds_lo[a], bounds_hi[a])
        else:
            s = wp.clamp(nominal[t, a] + n, bounds_lo[a], bounds_hi[a])
        samples[k, t, a] = s
        # effective (post-clamp) deviation from nominal so update() respects
        # the bounds and remains unbiased for zero-mean samples
        noise[k, t, a] = s - nominal[t, a]


@wp.kernel
def _advance_counter(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


@wp.kernel
def _interp_seq(
    knots: wp.array2d[float],
    j0: wp.array[wp.int32],
    j1: wp.array[wp.int32],
    frac: wp.array[float],
    out: wp.array2d[float],
):
    # linear interpolation of the knot decision variable up to the fine
    # horizon; a convex combination of clamped knots stays within bounds
    t, a = wp.tid()
    f = frac[t]
    out[t, a] = knots[j0[t], a] * (1.0 - f) + knots[j1[t], a] * f


@wp.kernel
def _interp_samples(
    knot_samples: wp.array3d[float],
    j0: wp.array[wp.int32],
    j1: wp.array[wp.int32],
    frac: wp.array[float],
    out: wp.array3d[float],
):
    k, t, a = wp.tid()
    f = frac[t]
    out[k, t, a] = knot_samples[k, j0[t], a] * (1.0 - f) + knot_samples[k, j1[t], a] * f


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
    tsallis_q: wp.array[float],
    weights: wp.array[float],
):
    k = wp.tid()
    x = -(costs[k] - min_cost[0]) / wp.max(temperature[0], 1.0e-6)
    q = tsallis_q[0]
    if q == 1.0:
        # exact softmax (Boltzmann) weighting -- bit-identical default path
        weights[k] = wp.exp(x)
    else:
        # Tsallis / deformed-exponential weighting: w ~ exp_q(-cost/lambda),
        # exp_q(x) = [1 + (1 - q) x]_+^{1/(1-q)}. As q grows the weight sharpens
        # from soft averaging toward hard elite (CEM-style) selection, trading
        # averaging smoothness for elite concentration (watch ESS). q -> 1
        # recovers the exponential above.
        base = wp.max(1.0 + (1.0 - q) * x, 0.0)
        weights[k] = wp.pow(base, 1.0 / (1.0 - q))


@wp.kernel
def _sum_weights(weights: wp.array[float], out: wp.array[float], ess: wp.array[float]):
    s = float(0.0)
    s2 = float(0.0)
    for k in range(weights.shape[0]):
        s += weights[k]
        s2 += weights[k] * weights[k]
    out[0] = s
    # effective sample size 1/sum(w_norm^2); diagnostic for temperature tuning
    ess[0] = s * s / wp.max(s2, 1.0e-30)


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


@wp.kernel
def _shift_knots(knots: wp.array2d[float], delta: float):
    # warm-start the knot spline by ONE fine control step (not one knot):
    # new_knots[j] = spline(j + delta), delta = (n-1)/(H-1) knot units.
    # The ascending in-place loop is safe because delta <= 1 (delta == 1 only
    # when n_knots == horizon): writing knot j reads knots j0 = min(j+1, n-2)
    # and j0+1, which are >= j and have not been overwritten yet. The last knot
    # repeats (spline extended by holding the final value).
    a = wp.tid()
    n = knots.shape[0]
    for j in range(n - 1):
        p = float(j) + delta
        j0 = wp.min(int(p), n - 2)
        f = p - float(j0)
        knots[j, a] = knots[j0, a] * (1.0 - f) + knots[j0 + 1, a] * f


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

    After :meth:`update`, :attr:`ess` holds the effective sample size
    ``1 / sum(w_normalized^2)`` (shape [1]); healthy values are roughly 5-20%
    of ``num_samples`` — near 1 means the softmax collapsed onto a single
    rollout (temperature too low for the cost spread).

    The optional private ``_n_knots`` constructor argument switches the
    decision variable to ``_n_knots`` coarse control knots linearly
    interpolated up to the horizon (DIAL-MPC-style spline parameterization),
    a lower-dimensional and inherently smoother control. It is experimental
    and default-off (``None`` reproduces the per-step sampler bit-for-bit);
    :attr:`samples` and :attr:`nominal` keep their horizon-resolution shapes
    regardless, so the rollout caller is unaffected.

    Two further private, default-off smoothing knobs from the MPPI literature
    are available: ``_zero_mean_fraction`` (RA-MPPI) draws a fraction of the
    samples as pure noise rather than noise about the nominal, and
    ``_tsallis_q`` replaces the softmax weighting with a deformed exponential
    that sharpens toward elite selection as ``q`` grows. Both default to the
    bit-identical baseline (``0.0`` and ``1.0``) and are runtime-tunable via
    :meth:`_set_zero_mean_fraction` and :meth:`_set_tsallis_q`.
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
        beta: float | tuple[float, ...] = 0.7
        """Per-step noise smoothing in [0, 1); 0 is white noise. A scalar
        applies to all channels; a tuple of length ``dim`` sets it per channel
        (drive typically wants more smoothing than steering)."""
        bounds_lo: tuple[float, ...] = (-1.0, -1.0)
        """Per-channel lower command bounds, length ``dim``."""
        bounds_hi: tuple[float, ...] = (1.0, 1.0)
        """Per-channel upper command bounds, length ``dim``."""
        seed: int = 0
        """Base RNG seed; resampling advances an internal device counter."""

    def __init__(
        self,
        config: Config | None = None,
        device: wp.context.Device | str | None = None,
        *,
        _n_knots: int | None = None,
        _zero_mean_fraction: float = 0.0,
        _tsallis_q: float = 1.0,
    ):
        cfg = config if config is not None else ControllerMPPI.Config()
        if cfg.num_samples < 2:
            raise ValueError("num_samples must be >= 2 (sample 0 is the nominal)")
        if cfg.horizon < 2:
            raise ValueError("horizon must be >= 2")
        for name in ("sigma", "bounds_lo", "bounds_hi"):
            if len(getattr(cfg, name)) != cfg.dim:
                raise ValueError(f"{name} must have length dim={cfg.dim}")
        betas = (cfg.beta,) * cfg.dim if isinstance(cfg.beta, int | float) else tuple(cfg.beta)
        if len(betas) != cfg.dim:
            raise ValueError(f"beta must be a scalar or have length dim={cfg.dim}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("beta must be in [0, 1)")
        # spline-knot control parameterization (experimental, private per
        # Decision 3): the decision variable becomes n_knots coarse control
        # points linearly interpolated up to the fine horizon. None reproduces
        # the per-step sampler bit-for-bit.
        self._use_knots = _n_knots is not None
        if self._use_knots:
            if _n_knots < 2:
                raise ValueError("_n_knots must be >= 2")
            if _n_knots > cfg.horizon:
                raise ValueError("_n_knots must be <= horizon")
        self._n_dec = _n_knots if self._use_knots else cfg.horizon
        self.config = cfg
        self.device = wp.get_device(device)
        k, h, a = cfg.num_samples, cfg.horizon, cfg.dim
        n = self._n_dec
        with wp.ScopedDevice(self.device):
            self.nominal = wp.zeros((h, a), dtype=wp.float32)
            self.samples = wp.zeros((k, h, a), dtype=wp.float32)
            if self._use_knots:
                # decision variable and noise live at knot resolution; the
                # per-sample knot commands are interpolated into `samples`
                self._knots = wp.zeros((n, a), dtype=wp.float32)
                self._knot_samples = wp.zeros((k, n, a), dtype=wp.float32)
                self.noise = wp.zeros((k, n, a), dtype=wp.float32)
                # fixed knot->horizon interpolation weights (t maps to a
                # fractional knot index); precomputed once, graph-safe
                p = np.arange(h, dtype=np.float64) * (n - 1) / (h - 1)
                j0 = np.clip(np.floor(p).astype(np.int32), 0, n - 2)
                self._interp_j0 = wp.array(j0, dtype=wp.int32)
                self._interp_j1 = wp.array(j0 + 1, dtype=wp.int32)
                self._interp_frac = wp.array((p - j0).astype(np.float32), dtype=wp.float32)
            else:
                # decision variable is the nominal itself; sampling writes
                # straight into `samples` (no interpolation), bit-identical
                self._knots = self.nominal
                self.noise = wp.zeros((k, h, a), dtype=wp.float32)
            self.weights = wp.zeros(k, dtype=wp.float32)
            self.sigma = wp.array(cfg.sigma, dtype=wp.float32)
            # per-decision-step noise scale (experimental, private): defaults
            # to all-ones, which reproduces the flat-sigma sampler bit-for-bit.
            # With knots this schedule acts at knot resolution.
            self._sigma_schedule = wp.ones(n, dtype=wp.float32)
            self.bounds_lo = wp.array(cfg.bounds_lo, dtype=wp.float32)
            self.bounds_hi = wp.array(cfg.bounds_hi, dtype=wp.float32)
            self._temperature = wp.array([cfg.temperature], dtype=wp.float32)
            self._beta = wp.array(betas, dtype=wp.float32)
            # RA-MPPI zero-mean sample count and Tsallis weighting exponent
            # (experimental, private): defaults reproduce the plain sampler and
            # exact softmax bit-for-bit. Held in device arrays so they stay
            # graph-capturable and runtime-tunable like set_temperature/set_beta.
            self._zero_mean_count = wp.zeros(1, dtype=wp.int32)
            self._tsallis_q = wp.array([1.0], dtype=wp.float32)
            self._counter = wp.zeros(1, dtype=wp.int32)
            self._min_cost = wp.zeros(1, dtype=wp.float32)
            self._weight_sum = wp.zeros(1, dtype=wp.float32)
            self.ess = wp.zeros(1, dtype=wp.float32)
        self._set_zero_mean_fraction(_zero_mean_fraction)
        self._set_tsallis_q(_tsallis_q)

    def set_temperature(self, value: float) -> None:
        """Sets the softmax temperature (safe while a CUDA graph is captured)."""
        self._temperature.fill_(float(value))

    def set_beta(self, value) -> None:
        """Sets the noise smoothing factor(s) (safe while a CUDA graph is captured).

        Args:
            value: Scalar applied to all channels, or a sequence of length ``dim``.
        """
        if isinstance(value, int | float):
            self._beta.fill_(float(value))
        else:
            self._beta.assign(np.asarray(value, dtype=np.float32))

    def _set_sigma_horizon_factor(self, value: float) -> None:
        """Sets the horizon-annealed noise schedule (experimental, private).

        Scales the per-step exploration noise by ``value ** (t / (H - 1))``:
        the executed step ``t = 0`` keeps the configured ``sigma`` while the
        far horizon end explores with ``value * sigma`` (DIAL-MPC-style
        horizon annealing). ``1.0`` restores the flat schedule bit-for-bit.
        Safe while a CUDA graph is captured (device-array write only).

        Args:
            value: Far-horizon noise multiplier; must be > 0.
        """
        if value <= 0.0:
            raise ValueError("sigma horizon factor must be > 0")
        n = self._n_dec
        schedule = float(value) ** (np.arange(n, dtype=np.float64) / (n - 1))
        self._sigma_schedule.assign(schedule.astype(np.float32))

    def _set_zero_mean_fraction(self, value: float) -> None:
        """Sets the RA-MPPI zero-mean sample fraction (experimental, private).

        The first ``ceil(value * num_samples)`` non-hero samples (sample 0 stays
        the pure nominal) draw pure smoothed noise ``u = eps`` rather than
        ``nominal + eps``, a Williams-lineage smoothing trick that pulls the
        softmax average toward zero and damps accumulated nominal drift. ``0.0``
        (default) reproduces the plain sampler bit-for-bit. Safe while a CUDA
        graph is captured (device-array write only).

        Args:
            value: Fraction of samples drawn zero-mean, in [0, 1].
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("zero-mean fraction must be in [0, 1]")
        count = int(np.ceil(value * self.config.num_samples))
        self._zero_mean_count.fill_(count)

    def _set_tsallis_q(self, value: float) -> None:
        """Sets the Tsallis weighting exponent q (experimental, private).

        Replaces the softmax weight ``exp(-cost/lambda)`` with the deformed
        exponential ``exp_q(-cost/lambda)``; ``q`` grows the weighting from soft
        averaging toward hard elite selection (trading averaging smoothness for
        elite concentration). ``1.0`` (default) is the exact softmax path,
        bit-for-bit. Safe while a CUDA graph is captured (device-array write
        only).

        Args:
            value: Deformed-exponential exponent; must be > 0.
        """
        if value <= 0.0:
            raise ValueError("tsallis q must be > 0")
        self._tsallis_q.fill_(float(value))

    def _sync_nominal(self) -> None:
        """Interpolates the knot decision variable into the horizon nominal.

        No-op when knots are disabled (the nominal is the decision variable).
        """
        if not self._use_knots:
            return
        wp.launch(
            _interp_seq,
            dim=(self.config.horizon, self.config.dim),
            inputs=[self._knots, self._interp_j0, self._interp_j1, self._interp_frac, self.nominal],
            device=self.device,
        )

    def sample(self) -> None:
        """Fills :attr:`samples` with the clamped nominal plus smoothed Gaussian noise.

        With knots enabled, noise is sampled at knot resolution and the
        per-sample knot commands are linearly interpolated up to the horizon,
        so :attr:`samples` keeps shape ``[num_samples, horizon, dim]`` either way.
        """
        cfg = self.config
        knot_out = self._knot_samples if self._use_knots else self.samples
        wp.launch(
            _sample_sequences,
            dim=(cfg.num_samples, cfg.dim),
            inputs=[
                self._knots,
                self.sigma,
                self._sigma_schedule,
                self.bounds_lo,
                self.bounds_hi,
                self._beta,
                cfg.seed,
                self._counter,
                self._zero_mean_count,
            ],
            outputs=[self.noise, knot_out],
            device=self.device,
        )
        wp.launch(_advance_counter, dim=1, inputs=[self._counter], device=self.device)
        if self._use_knots:
            wp.launch(
                _interp_samples,
                dim=(cfg.num_samples, cfg.horizon, cfg.dim),
                inputs=[self._knot_samples, self._interp_j0, self._interp_j1, self._interp_frac, self.samples],
                device=self.device,
            )

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
            inputs=[costs, self._min_cost, self._temperature, self._tsallis_q, self.weights],
            device=self.device,
        )
        wp.launch(_sum_weights, dim=1, inputs=[self.weights, self._weight_sum, self.ess], device=self.device)
        wp.launch(
            _update_nominal,
            dim=(self._n_dec, cfg.dim),
            inputs=[self.noise, self.weights, self._weight_sum, self.bounds_lo, self.bounds_hi, self._knots],
            device=self.device,
        )
        self._sync_nominal()

    def shift(self) -> None:
        """Rolls the nominal one step forward, repeating the final row.

        With knots enabled, the knot spline is resampled one fine control
        step later (``delta = (n_knots - 1) / (horizon - 1)`` knot units), so
        the warm-start advances by exactly one executed step either way.
        """
        if self._use_knots:
            delta = (self._n_dec - 1) / (self.config.horizon - 1)
            wp.launch(_shift_knots, dim=self.config.dim, inputs=[self._knots, float(delta)], device=self.device)
        else:
            wp.launch(_shift_nominal, dim=self.config.dim, inputs=[self._knots], device=self.device)
        self._sync_nominal()
