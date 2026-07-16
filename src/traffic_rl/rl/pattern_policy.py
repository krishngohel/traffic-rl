"""The pattern-aware policy wrapped as a regular Controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.config import SimConfig
from traffic_rl.controllers.base import Controller, Observation
from traffic_rl.rl.dqn import MLP
from traffic_rl.rl.features import slot_action_mask, slot_to_phase
from traffic_rl.rl.patterns import PatternTracker, featurize_with_patterns

PATTERN_WEIGHTS = Path(__file__).parent / "pattern_weights.npz"


class PatternRLController(Controller):
    name = "rl_pattern"

    def __init__(self, weights: Path | str = PATTERN_WEIGHTS):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"pattern RL weights not found at {weights}. "
                "Train them with: traffic-rl-train --pattern"
            )
        self.net = MLP.load(weights)
        self.tracker = PatternTracker()

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.tracker = PatternTracker(dt=config.dt)

    def act(self, obs: Observation) -> int:
        q = self.net.forward(featurize_with_patterns(obs, self.tracker))[0]
        q = np.where(slot_action_mask(obs), q, -np.inf)
        return slot_to_phase(obs, int(np.argmax(q)))
