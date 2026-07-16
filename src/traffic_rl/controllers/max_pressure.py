"""Max-pressure control (Varaiya). At a single isolated intersection downstream
queues are zero, so this honestly degenerates to weighted longest-queue-first —
still the standard classical baseline reported in the RL traffic literature.
Generalized to the phase table: a phase's pressure is the weighted sum of the
queues it serves at saturation (permissive lefts count at a discount, since
their service rate depends on opposing gaps)."""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_MOVEMENTS, SimConfig
from traffic_rl.controllers.base import Controller, Observation

_PERMISSIVE_WEIGHT = 0.3  # a filtering left clears far below saturation


class MaxPressureController(Controller):
    name = "max_pressure"

    # 15 s control period: max-pressure is blind to switching cost (~7 s lost per
    # transition), so short periods thrash near saturation. Sweep over
    # {5, 10, 15, 20} s: heavy-scenario p95 falls 83->62 s from 5->20 s while
    # asymmetric stays ~36 s; 15 s is the knee and a standard period in the
    # literature. Recorded in README for fairness.
    def __init__(self, decision_interval: float = 15.0):
        self.decision_interval = decision_interval

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.phases = config.phases
        # Weights = relative saturation flow of each lane group (generalizes
        # the classic unit weights to multi-lane groups).
        base = config.sat_flow
        self._weights = np.array(
            [config.group_sat_flow(g) / base for g in range(N_MOVEMENTS)]
        )
        self._last_decision_t = -np.inf
        self._target = 0

    def act(self, obs: Observation) -> int:
        if obs.t - self._last_decision_t >= self.decision_interval and obs.action_mask.all():
            self._last_decision_t = obs.t
            pressures = np.array(
                [
                    sum(self._weights[g] * obs.queue_lengths[g] for g in p.movements)
                    + sum(
                        _PERMISSIVE_WEIGHT * self._weights[g] * obs.queue_lengths[g]
                        for g in p.permissive_lefts
                    )
                    for p in self.phases
                ]
            )
            # Strict improvement required: hold on ties (hysteresis against chatter).
            if pressures.max() > pressures[obs.phase]:
                self._target = int(np.argmax(pressures))
            else:
                self._target = obs.phase
        return self._target
