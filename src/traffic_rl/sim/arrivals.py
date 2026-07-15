"""Poisson arrival streams with independent per-stream RNGs.

Each approach and each ped movement gets its own child stream spawned from one
SeedSequence, so adding or reseeding one stream never perturbs the others —
required for paired-seed (common random numbers) comparisons.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_APPROACHES, N_PED_MOVEMENTS, DemandConfig


class ArrivalStreams:
    def __init__(self, seed: int, demand: DemandConfig, dt: float):
        children = np.random.SeedSequence(seed).spawn(N_APPROACHES + N_PED_MOVEMENTS)
        self._veh_rngs = [np.random.Generator(np.random.PCG64(c)) for c in children[:N_APPROACHES]]
        self._ped_rngs = [np.random.Generator(np.random.PCG64(c)) for c in children[N_APPROACHES:]]
        self._veh_lam = [r * dt / 3600.0 for r in demand.vehicle_rates]
        self._ped_lam = [r * dt / 3600.0 for r in demand.ped_rates]

    def vehicle_counts(self) -> np.ndarray:
        pairs = zip(self._veh_rngs, self._veh_lam, strict=True)
        return np.array([rng.poisson(lam) for rng, lam in pairs], dtype=np.int64)

    def ped_counts(self) -> np.ndarray:
        pairs = zip(self._ped_rngs, self._ped_lam, strict=True)
        return np.array([rng.poisson(lam) for rng, lam in pairs], dtype=np.int64)
