"""Shared decentralized policy for the corridor: one network (pun intended) of
weights runs every intersection, seeing its local observation plus the two
downstream arterial queues — the minimal communication real adaptive corridor
systems assume."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.controllers.base import Observation
from traffic_rl.controllers.network import NetworkController
from traffic_rl.rl.dqn import MLP
from traffic_rl.rl.features import N_FEATURES, featurize
from traffic_rl.sim.network import EASTBOUND, WESTBOUND, NetworkConfig

N_NETWORK_FEATURES = N_FEATURES + 2
NETWORK_WEIGHTS = Path(__file__).parent / "network_weights.npz"


def downstream_queues(observations: list[Observation], i: int, n_nodes: int) -> tuple[float, float]:
    down_e = observations[i + 1].queue_lengths[EASTBOUND] if i + 1 < n_nodes else 0.0
    down_w = observations[i - 1].queue_lengths[WESTBOUND] if i - 1 >= 0 else 0.0
    return float(down_e), float(down_w)


def featurize_network(obs: Observation, down_e: float, down_w: float) -> np.ndarray:
    return np.concatenate(
        [featurize(obs), np.log1p([down_e, down_w]).astype(np.float32) / 4.0]
    ).astype(np.float32)


class SharedRLNetworkController(NetworkController):
    name = "rl"

    def __init__(self, weights: Path | str = NETWORK_WEIGHTS):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"network RL weights not found at {weights}. "
                "Train them with: python -m traffic_rl.rl.train_network"
            )
        self.net = MLP.load(weights)

    def reset(self, config: NetworkConfig, rng: np.random.Generator) -> None:
        self.n_nodes = config.n_nodes

    def act(self, observations: list[Observation]) -> list[int]:
        actions = []
        for i, obs in enumerate(observations):
            down_e, down_w = downstream_queues(observations, i, self.n_nodes)
            q = self.net.forward(featurize_network(obs, down_e, down_w))[0]
            q = np.where(obs.action_mask, q, -np.inf)
            actions.append(int(np.argmax(q)))
        return actions
