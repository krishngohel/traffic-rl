"""Vehicle-actuated control: green extension with gap-out, phase skipping on
no-call, like induction-loop signals. Textbook default parameters, stated in
the README for fairness. Left phases run shorter min/max greens and are
skipped entirely when the bay is empty — exactly what detection hardware buys
a real protected-left signal."""

from __future__ import annotations

import numpy as np

from traffic_rl.config import SLOT_EW_LEFT, SLOT_NS_LEFT, Phase, SimConfig
from traffic_rl.controllers.base import Controller, Observation


def _is_left(phase: Phase) -> bool:
    return phase.slot in (SLOT_NS_LEFT, SLOT_EW_LEFT)


class ActuatedController(Controller):
    name = "actuated"

    def __init__(
        self,
        min_green: float = 8.0,
        max_green: float = 45.0,
        min_green_left: float = 5.0,
        max_green_left: float = 20.0,
        unit_extension: float = 3.0,
    ):
        self.min_green = min_green
        self.max_green = max_green  # operational max-out, below the machine backstop
        self.min_green_left = min_green_left
        self.max_green_left = max_green_left
        self.unit_extension = unit_extension

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.phases = config.phases
        self.n = len(self.phases)

    def _call(self, obs: Observation, p: int) -> bool:
        phase = self.phases[p]
        veh = any(
            obs.queue_lengths[g] > 0 for g in phase.movements + phase.permissive_lefts
        )
        ped = phase.ped_movement is not None and obs.ped_call[phase.ped_movement] > 0
        return veh or ped

    def _next_with_call(self, obs: Observation, current: int) -> int:
        for step in range(1, self.n):
            candidate = (current + step) % self.n
            if self._call(obs, candidate):
                return candidate
        return current

    def act(self, obs: Observation) -> int:
        p = obs.phase
        phase = self.phases[p]
        min_g = self.min_green_left if _is_left(phase) else self.min_green
        max_g = self.max_green_left if _is_left(phase) else self.max_green
        if not obs.is_green or obs.phase_elapsed < min_g:
            return p

        conflicting = any(self._call(obs, q) for q in range(self.n) if q != p)
        if not conflicting:
            return p  # rest in green

        if obs.phase_elapsed >= max_g:
            return self._next_with_call(obs, p)  # max-out
        # Presence detection: a standing queue keeps the loop occupied (gap 0);
        # only an empty lane group measures time since the last arrival.
        gap = min(
            0.0 if obs.queue_lengths[g] > 0 else obs.time_since_arrival[g]
            for g in phase.movements
        )
        if gap >= self.unit_extension:
            return self._next_with_call(obs, p)  # gap-out
        return p
