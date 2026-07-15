"""Poisson arrival streams with independent per-stream RNGs.

Each approach and each ped movement gets its own child stream spawned from one
SeedSequence, so adding or reseeding one stream never perturbs the others —
required for paired-seed (common random numbers) comparisons. Rates may be
constant or follow a DemandSchedule (time-varying, e.g. loaded from real count
data); with a schedule, the same seed still yields the same draws for the same
schedule.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import (
    N_APPROACHES,
    N_PED_MOVEMENTS,
    DemandConfig,
    DemandSchedule,
    demand_at,
)


class ArrivalStreams:
    def __init__(
        self,
        seed: int,
        demand: DemandConfig,
        dt: float,
        schedule: DemandSchedule | None = None,
    ):
        children = np.random.SeedSequence(seed).spawn(N_APPROACHES + N_PED_MOVEMENTS)
        self._veh_rngs = [np.random.Generator(np.random.PCG64(c)) for c in children[:N_APPROACHES]]
        self._ped_rngs = [np.random.Generator(np.random.PCG64(c)) for c in children[N_APPROACHES:]]
        self._dt = dt
        self._demand = demand
        self._schedule = schedule

    def _rates(self, t: float) -> DemandConfig:
        if self._schedule is not None:
            return demand_at(self._schedule, t)
        return self._demand

    def vehicle_counts(self, t: float = 0.0) -> np.ndarray:
        rates = self._rates(t).vehicle_rates
        pairs = zip(self._veh_rngs, rates, strict=True)
        return np.array(
            [rng.poisson(r * self._dt / 3600.0) for rng, r in pairs], dtype=np.int64
        )

    def ped_counts(self, t: float = 0.0) -> np.ndarray:
        rates = self._rates(t).ped_rates
        pairs = zip(self._ped_rngs, rates, strict=True)
        return np.array(
            [rng.poisson(r * self._dt / 3600.0) for rng, r in pairs], dtype=np.int64
        )
