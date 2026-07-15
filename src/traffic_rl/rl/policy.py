"""The trained policy wrapped as a regular Controller.

Evaluated by the exact same harness, seeds, and metrics as the four classical
baselines — no special treatment in either direction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.config import SimConfig
from traffic_rl.controllers.base import Controller, Observation
from traffic_rl.rl.dqn import MLP
from traffic_rl.rl.features import featurize

DEFAULT_WEIGHTS = Path(__file__).parent / "weights.npz"


class RLController(Controller):
    name = "rl"

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"RL weights not found at {weights}. Train them with: traffic-rl-train"
            )
        self.net = MLP.load(weights)

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:
        pass  # inference is stateless and deterministic

    def act(self, obs: Observation) -> int:
        q = self.net.forward(featurize(obs))[0]
        q = np.where(obs.action_mask, q, -np.inf)
        return int(np.argmax(q))
