"""Webster (1958) fixed-time optimization via critical-movement analysis,
run honestly in two stages.

Stage 1 (0-900 s): run the naive plan and count detector arrivals per lane
group — the plan is built from *observed* flows, never the scenario's true rates.
Stage 2 (900 s on): compute the Webster plan once and run it. The shared 1200 s
warm-up gives 300 s of post-switch settling before measurement starts.

Each phase's flow ratio y_p is the worst flow/saturation ratio among the lane
groups it serves at saturation (critical movement). Permissive lefts are not a
phase's critical movement (they use leftover opposing-gap capacity — classic
Webster has no better answer, which is itself honest). Phases that serve a ped
movement get the pedestrian service floor; left phases get the left min green.

When a proportional green falls below its floor, the cycle is EXTENDED to the
minimum cycle that clears the floor (rather than shrinking the major phase,
which would make Webster worse than naive).
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_MOVEMENTS, Phase, SignalTimingConfig, SimConfig
from traffic_rl.controllers.base import Controller, Observation
from traffic_rl.controllers.fixed_time import FixedTimeController, FixedTimePlan

C_MIN, C_MAX = 40.0, 150.0
Y_SATURATED = 0.95


def phase_floors(phases: tuple[Phase, ...], timing: SignalTimingConfig) -> tuple[float, ...]:
    """Actual-green floor per phase: ped service for ped-serving phases,
    min green otherwise."""
    floors = []
    for p in phases:
        if p.ped_movement is not None:
            floors.append(max(timing.ped_service(p.ped_movement), timing.min_green_for(p)))
        else:
            floors.append(timing.min_green_for(p))
    return tuple(floors)


def webster_plan(
    group_flows_veh_h: np.ndarray,
    group_sat_veh_h: np.ndarray,
    timing: SignalTimingConfig,
    phases: tuple[Phase, ...],
    green_floors: tuple[float, ...] | None = None,
) -> FixedTimePlan:
    """Pure function: observed per-lane-group flows -> fixed-time plan.

    green_floors are floors on the ACTUAL green per phase (e.g. the ped
    service lock); pass None to disable (used by the numeric unit tests).
    """
    n = len(phases)
    y = np.array(
        [
            max(
                (group_flows_veh_h[g] / group_sat_veh_h[g] for g in p.movements),
                default=0.0,
            )
            for p in phases
        ]
    )
    y = np.maximum(y, 1e-6)
    Y = float(y.sum())
    lost_per_phase = np.array(
        [timing.startup_lost + timing.yellow_for(p) + timing.all_red_for(p) for p in phases]
    )
    L = float(lost_per_phase.sum())

    if Y >= Y_SATURATED:
        cycle = C_MAX
    else:
        cycle = float(np.clip((1.5 * L + 5.0) / (1.0 - Y), C_MIN, C_MAX))

    if green_floors is not None:
        # Effective-green floors implied by the actual-green floors.
        floors_eff = np.array(
            [max(f, timing.min_green_for(p)) for f, p in zip(green_floors, phases, strict=True)]
        ) - timing.startup_lost
        # Minimum cycle at which every proportional split clears its floor:
        # g_eff_p = (y_p / Y) * (C - L) >= floor_p  =>  C >= L + floor_p * Y / y_p
        needed = float((L + floors_eff * Y / y).max())
        cycle = float(np.clip(max(cycle, needed), C_MIN, C_MAX))

    g_eff = (y / Y) * (cycle - L)
    if green_floors is not None:
        floors_eff = np.array(
            [max(f, timing.min_green_for(p)) for f, p in zip(green_floors, phases, strict=True)]
        ) - timing.startup_lost
        # C_MAX clamp can bind before every floor clears: top up deficits from
        # the largest surplus phases.
        for _ in range(n):
            deficits = floors_eff - g_eff
            if deficits.max() <= 1e-9:
                break
            i = int(np.argmax(deficits))
            j = int(np.argmax(g_eff - floors_eff))
            transfer = min(deficits[i], g_eff[j] - floors_eff[j])
            g_eff[i] += transfer
            g_eff[j] -= transfer
    greens = tuple(float(g + timing.startup_lost) for g in g_eff)
    return FixedTimePlan(greens=greens)


class WebsterController(Controller):
    name = "webster"

    def __init__(self, observation_window: float = 900.0):
        self.observation_window = observation_window

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._counts = np.zeros(N_MOVEMENTS)
        self._naive = FixedTimeController(None)
        self._naive.reset(config, rng)
        self._planned: FixedTimeController | None = None
        self.plan: FixedTimePlan | None = None  # exposed for tests/inspection

    def act(self, obs: Observation) -> int:
        if obs.t < self.observation_window:
            self._counts += obs.arrivals_last_step
            return self._naive.act(obs)
        if self._planned is None:
            flows = self._counts / self.observation_window * 3600.0
            sat = np.array(
                [self.config.group_sat_flow(g) * 3600.0 for g in range(N_MOVEMENTS)]
            )
            phases = self.config.phases
            self.plan = webster_plan(
                flows,
                sat,
                timing=self.config.timing,
                phases=phases,
                green_floors=phase_floors(phases, self.config.timing),
            )
            self._planned = FixedTimeController(self.plan)
            self._planned.reset(self.config, np.random.default_rng(0))
        return self._planned.act(obs)
