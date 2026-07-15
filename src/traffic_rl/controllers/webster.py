"""Webster (1958) fixed-time optimization, run honestly in two stages.

Stage 1 (0-900 s): run the naive plan and count detector arrivals — the plan is
built from *observed* flows, never the scenario's true rates.
Stage 2 (900 s on): compute the Webster plan once and run it. The shared 1200 s
warm-up gives 300 s of post-switch settling before measurement starts.

When the proportional minor-phase green falls below the pedestrian service floor,
the cycle is EXTENDED to the minimum cycle that clears the floor (rather than
shrinking the major phase, which would make Webster worse than naive).
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_PHASES, PHASE_APPROACHES, SimConfig
from traffic_rl.controllers.base import Controller, Observation
from traffic_rl.controllers.fixed_time import NAIVE_PLAN, FixedTimeController, FixedTimePlan

C_MIN, C_MAX = 40.0, 150.0
Y_SATURATED = 0.95


def webster_plan(
    flows_veh_h: np.ndarray,
    sat_flow_veh_h: float,
    timing,
    green_floor: float | None = None,
) -> FixedTimePlan:
    """Pure function: observed per-approach flows -> fixed-time plan.

    green_floor is a floor on the ACTUAL green per phase (e.g. the 20 s ped
    service lock); pass None to disable (used by the numeric unit tests).
    """
    y = np.array(
        [
            max(flows_veh_h[a] for a in PHASE_APPROACHES[p]) / sat_flow_veh_h
            for p in range(N_PHASES)
        ]
    )
    y = np.maximum(y, 1e-6)
    Y = float(y.sum())
    L = N_PHASES * timing.lost_time_per_phase

    if Y >= Y_SATURATED:
        cycle = C_MAX
    else:
        cycle = float(np.clip((1.5 * L + 5.0) / (1.0 - Y), C_MIN, C_MAX))

    if green_floor is not None:
        # Effective-green floor implied by the actual-green floor.
        floor_eff = max(green_floor, timing.min_green) - timing.startup_lost
        # Minimum cycle at which every proportional split clears the floor:
        # g_eff_p = (y_p / Y) * (C - L) >= floor_eff  =>  C >= L + floor_eff * Y / y_p
        needed = L + floor_eff * Y / y.min()
        cycle = float(np.clip(max(cycle, needed), C_MIN, C_MAX))

    g_eff = (y / Y) * (cycle - L)
    if green_floor is not None:
        floor_eff = max(green_floor, timing.min_green) - timing.startup_lost
        if g_eff.min() < floor_eff:  # C_MAX clamp bound before the floor cleared
            deficit = floor_eff - g_eff.min()
            g_eff[np.argmin(g_eff)] += deficit
            g_eff[np.argmax(g_eff)] -= deficit
    greens = tuple(float(g + timing.startup_lost) for g in g_eff)
    return FixedTimePlan(greens=greens)


class WebsterController(Controller):
    name = "webster"

    def __init__(self, observation_window: float = 900.0):
        self.observation_window = observation_window

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._counts = np.zeros(4)
        self._naive = FixedTimeController(NAIVE_PLAN)
        self._naive.reset(config, rng)
        self._planned: FixedTimeController | None = None
        self.plan: FixedTimePlan | None = None  # exposed for tests/inspection

    def act(self, obs: Observation) -> int:
        if obs.t < self.observation_window:
            self._counts += obs.arrivals_last_step
            return self._naive.act(obs)
        if self._planned is None:
            flows = self._counts / self.observation_window * 3600.0
            self.plan = webster_plan(
                flows,
                sat_flow_veh_h=self.config.sat_flow * 3600.0,
                timing=self.config.timing,
                green_floor=self.config.timing.ped_service,
            )
            self._planned = FixedTimeController(self.plan)
            self._planned.reset(self.config, np.random.default_rng(0))
        return self._planned.act(obs)
