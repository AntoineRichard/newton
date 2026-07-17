# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask/tell black-box optimizer strategies. Standalone: numpy only.

Protocol: ``x = strategy.ask()`` returns a ``(population, n)`` candidate
matrix; ``strategy.tell(fitness)`` reports the ``(population,)`` fitness of
exactly that matrix (lower is better). ``best_x``/``best_f`` track the best
candidate ever told.
"""

from __future__ import annotations

import numpy as np


class StrategyGA:
    """Genetic algorithm: tournament selection, SBX crossover, polynomial
    mutation, elitism. Bounds are enforced by clipping."""

    def __init__(
        self,
        bounds_lo,
        bounds_hi,
        population: int = 64,
        x0=None,
        init_spread: float = 0.25,
        seed: int = 0,
        tournament_k: int = 3,
        crossover_prob: float = 0.9,
        eta_c: float = 15.0,
        mutation_prob: float | None = None,
        eta_m: float = 20.0,
        elite: int = 2,
    ):
        self.lo = np.asarray(bounds_lo, dtype=np.float64)
        self.hi = np.asarray(bounds_hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape or np.any(self.hi <= self.lo):
            raise ValueError("invalid bounds")
        self.n = self.lo.size
        self.population = int(population)
        self.rng = np.random.default_rng(seed)
        self.tournament_k = tournament_k
        self.crossover_prob = crossover_prob
        self.eta_c = eta_c
        self.mutation_prob = 1.0 / self.n if mutation_prob is None else mutation_prob
        self.eta_m = eta_m
        self.elite = int(elite)
        span = self.hi - self.lo
        if x0 is None:
            self._pop = self.lo + self.rng.random((self.population, self.n)) * span
        else:
            x0 = np.clip(np.asarray(x0, dtype=np.float64), self.lo, self.hi)
            self._pop = x0 + self.rng.standard_normal((self.population, self.n)) * init_spread * span
            self._pop[0] = x0  # keep the seed itself in the population
        self._pop = np.clip(self._pop, self.lo, self.hi)
        self._fit = None
        self._pending = self._pop.copy()
        self.best_x = None
        self.best_f = np.inf

    def ask(self) -> np.ndarray:
        return self._pending.copy()

    def tell(self, fitness) -> None:
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.shape != (self.population,):
            raise ValueError(f"expected fitness shape ({self.population},), got {fitness.shape}")
        pop = self._pending
        i_best = int(np.argmin(fitness))
        if fitness[i_best] < self.best_f:
            self.best_f = float(fitness[i_best])
            self.best_x = pop[i_best].copy()
        order = np.argsort(fitness)
        nxt = np.empty_like(pop)
        nxt[: self.elite] = pop[order[: self.elite]]
        i = self.elite
        while i < self.population:
            p1 = pop[self._tournament(fitness)]
            p2 = pop[self._tournament(fitness)]
            c1, c2 = self._sbx(p1, p2)
            nxt[i] = self._mutate(c1)
            if i + 1 < self.population:
                nxt[i + 1] = self._mutate(c2)
            i += 2
        self._pending = np.clip(nxt, self.lo, self.hi)

    def _tournament(self, fitness) -> int:
        idx = self.rng.integers(0, self.population, self.tournament_k)
        return int(idx[np.argmin(fitness[idx])])

    def _sbx(self, p1, p2):
        if self.rng.random() >= self.crossover_prob:
            return p1.copy(), p2.copy()
        u = self.rng.random(self.n)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0)),
        )
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        return c1, c2

    def _mutate(self, x):
        span = self.hi - self.lo
        mask = self.rng.random(self.n) < self.mutation_prob
        u = self.rng.random(self.n)
        delta = np.where(
            u < 0.5,
            (2.0 * u) ** (1.0 / (self.eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_m + 1.0)),
        )
        return np.where(mask, x + delta * span, x)
