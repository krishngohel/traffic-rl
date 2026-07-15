"""Vehicle-actuated control: green extension with gap-out, like induction-loop
signals. Textbook default parameters, stated in the README for fairness."""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_PHASES, PHASE_APPROACHES, SimConfig
from traffic_rl.controllers.base import Controller, Observation


class ActuatedController(Controller):
    name = "actuated"

    def __init__(
        self,
        min_green: float = 8.0,
        max_green: float = 45.0,
        unit_extension: float = 3.0,
    ):
        self.min_green = min_green
        self.max_green = max_green  # operational max-out, below the machine backstop
        self.unit_extension = unit_extension

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        pass

    def act(self, obs: Observation) -> int:
        phase = obs.phase
        if not obs.is_green or obs.phase_elapsed < self.min_green:
            return phase

        other = (phase + 1) % N_PHASES
        conflicting_call = (
            any(obs.queue_lengths[a] > 0 for a in PHASE_APPROACHES[other])
            or obs.ped_call[other] > 0
        )
        if not conflicting_call:
            return phase  # rest in green

        if obs.phase_elapsed >= self.max_green:
            return other  # max-out
        # Presence detection: a standing queue keeps the loop occupied (gap 0);
        # only an empty approach measures time since the last arrival.
        gap = min(
            0.0 if obs.queue_lengths[a] > 0 else obs.time_since_arrival[a]
            for a in PHASE_APPROACHES[phase]
        )
        if gap >= self.unit_extension:
            return other  # gap-out
        return phase
