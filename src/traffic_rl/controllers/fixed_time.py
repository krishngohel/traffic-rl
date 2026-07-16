"""Fixed-time (pretimed) control: a cycle clock over per-phase green durations.

Drives both the naive 50/50 baseline and Webster-optimized plans. Uses the
absolute sim clock, so if the state machine delays a switch (ped lock), the plan
re-synchronizes on the next cycle instead of drifting. Green lengths are per
phase in the config's phase-table order (e.g. NS-left, NS-through, EW-left,
EW-through for a protected-left site).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_rl.config import SimConfig
from traffic_rl.controllers.base import Controller, Observation


@dataclass(frozen=True)
class FixedTimePlan:
    greens: tuple[float, ...]  # actual signal green per phase, seconds

    def cycle(self, yellow: float, all_red: float) -> float:
        """Cycle length under uniform clearance intervals (legacy two-phase
        callers: the corridor stack). Prefer cycle_for(config) for per-phase
        clearances."""
        return sum(self.greens) + len(self.greens) * (yellow + all_red)

    def cycle_for(self, config: SimConfig) -> float:
        timing = config.timing
        phases = config.phases
        assert len(self.greens) == len(phases), (
            f"plan has {len(self.greens)} greens for {len(phases)} phases"
        )
        return sum(
            g + timing.yellow_for(p) + timing.all_red_for(p)
            for g, p in zip(self.greens, phases, strict=True)
        )


class FixedTimeController(Controller):
    name = "fixed"

    def __init__(self, plan: FixedTimePlan | None):
        """plan=None -> equal 25 s greens per phase (the naive baseline)."""
        self.plan = plan

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        timing = config.timing
        phases = config.phases
        plan = self.plan if self.plan is not None else naive_plan(config)
        self._cycle = plan.cycle_for(config)
        self._n_phases = len(phases)
        # Phase p owns the clock segment [starts[p], starts[p+1]).
        starts = [0.0]
        for g, p in zip(plan.greens, phases, strict=True):
            starts.append(starts[-1] + g + timing.yellow_for(p) + timing.all_red_for(p))
        self._starts = starts

    def act(self, obs: Observation) -> int:
        c = obs.t % self._cycle
        for p in range(self._n_phases):
            if c < self._starts[p + 1]:
                return p
        return self._n_phases - 1


class ScheduledFixedTimeController(Controller):
    """Time-of-day plans: a different fixed-time plan per schedule interval —
    what a real signal-retiming study installs."""

    name = "scheduled"

    def __init__(self, plans: list[tuple[float, FixedTimePlan]]):
        """plans: (start_second, plan), sorted ascending; first must start at 0."""
        self.plans = sorted(plans, key=lambda p: p[0])

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self._controllers = []
        for start, plan in self.plans:
            controller = FixedTimeController(plan)
            controller.reset(config, rng)
            self._controllers.append((start, controller))

    def act(self, obs: Observation) -> int:
        active = self._controllers[0][1]
        for start, controller in self._controllers:
            if start <= obs.t:
                active = controller
            else:
                break
        return active.act(obs)


def naive_plan(config: SimConfig) -> FixedTimePlan:
    """Demand-blind equal split: 25 s green per phase, whatever the phases are.
    For the legacy 2-phase layout this is the original 60 s naive cycle."""
    return FixedTimePlan(greens=(25.0,) * config.n_phases)


# Legacy alias: the Phase 1-4 naive plan for the 2-phase layout.
NAIVE_PLAN = FixedTimePlan(greens=(25.0, 25.0))


class NaiveController(FixedTimeController):
    name = "naive"

    def __init__(self) -> None:
        super().__init__(None)  # equal split, sized to the config's phase count
