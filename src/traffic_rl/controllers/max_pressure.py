"""Max-pressure control (Varaiya). At a single isolated intersection downstream
queues are zero, so this honestly degenerates to weighted longest-queue-first —
still the standard classical baseline reported in the RL traffic literature."""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_PHASES, PHASE_APPROACHES, SimConfig
from traffic_rl.controllers.base import Controller, Observation


class MaxPressureController(Controller):
    name = "max_pressure"

    # 15 s control period: max-pressure is blind to switching cost (7 s lost per
    # transition here), so short periods thrash near saturation. Sweep over
    # {5, 10, 15, 20} s: heavy-scenario p95 falls 83->62 s from 5->20 s while
    # asymmetric stays ~36 s; 15 s is the knee and a standard period in the
    # literature. Recorded in README for fairness.
    def __init__(self, decision_interval: float = 15.0):
        self.decision_interval = decision_interval

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        # Weights generalize to unequal saturation flows; all 1.0 here.
        self._weights = np.ones(4)
        self._last_decision_t = -np.inf
        self._target = 0

    def act(self, obs: Observation) -> int:
        if obs.t - self._last_decision_t >= self.decision_interval and obs.action_mask.all():
            self._last_decision_t = obs.t
            pressures = np.array(
                [
                    sum(self._weights[a] * obs.queue_lengths[a] for a in PHASE_APPROACHES[p])
                    for p in range(N_PHASES)
                ]
            )
            # Strict improvement required: hold on ties (hysteresis against chatter).
            if pressures.max() > pressures[obs.phase]:
                self._target = int(np.argmax(pressures))
            else:
                self._target = obs.phase
        return self._target
