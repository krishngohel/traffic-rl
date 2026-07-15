"""Fixed-time (pretimed) control: a cycle clock over per-phase green durations.

Drives both the naive 50/50 baseline and Webster-optimized plans. Uses the
absolute sim clock, so if the state machine delays a switch (ped lock), the plan
re-synchronizes on the next cycle instead of drifting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_rl.config import N_PHASES, SimConfig
from traffic_rl.controllers.base import Controller, Observation


@dataclass(frozen=True)
class FixedTimePlan:
    greens: tuple[float, ...]  # actual signal green per phase, seconds

    def cycle(self, yellow: float, all_red: float) -> float:
        return sum(self.greens) + N_PHASES * (yellow + all_red)


class FixedTimeController(Controller):
    name = "fixed"

    def __init__(self, plan: FixedTimePlan):
        self.plan = plan

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        y, ar = config.timing.yellow, config.timing.all_red
        self._cycle = self.plan.cycle(y, ar)
        # Phase p owns the clock segment [starts[p], starts[p+1]).
        starts = [0.0]
        for g in self.plan.greens:
            starts.append(starts[-1] + g + y + ar)
        self._starts = starts

    def act(self, obs: Observation) -> int:
        c = obs.t % self._cycle
        for p in range(N_PHASES):
            if c < self._starts[p + 1]:
                return p
        return N_PHASES - 1


# 60 s cycle: 25 s green each (23 s effective after 2 s startup), 3 y + 2 ar per
# phase. Demand-blind 50/50 — the naive baseline.
NAIVE_PLAN = FixedTimePlan(greens=(25.0, 25.0))


class NaiveController(FixedTimeController):
    name = "naive"

    def __init__(self) -> None:
        super().__init__(NAIVE_PLAN)
