"""Optional Gymnasium wrapper — the proof of the Phase-1 'zero core changes'
API claim. The in-repo NumPy trainer does not need it; this exists so anyone
can point stable-baselines3 (or any Gym-compatible library) at the sim.

Install with: pip install traffic-rl[rl]
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "TrafficEnv needs gymnasium. Install it with: pip install traffic-rl[rl]"
    ) from e

from traffic_rl.config import MAX_PHASES
from traffic_rl.rl.features import N_FEATURES, featurize, slot_action_mask, slot_to_phase
from traffic_rl.scenarios import make_config
from traffic_rl.sim.core import IntersectionSim

REWARD_SCALE = 50.0


class TrafficEnv(gym.Env):
    """Actions are canonical phase slots (0 NS-left, 1 NS-through, 2 EW-left,
    3 EW-through); `info["action_mask"]` marks the slots legal right now at
    this intersection. Slots the layout lacks are never legal."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: str = "asymmetric", episode_seconds: float = 3600.0):
        self.config = make_config(scenario)
        self.sim = IntersectionSim(self.config)
        self.episode_steps = round(episode_seconds / self.config.dt)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_FEATURES,), np.float32)
        self.action_space = gym.spaces.Discrete(MAX_PHASES)
        self._t = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset(seed if seed is not None else int(self.np_random.integers(2**31)))
        self._obs = obs
        self._t = 0
        return featurize(obs), {"action_mask": slot_action_mask(obs)}

    def step(self, action: int):
        result = self.sim.step(slot_to_phase(self._obs, int(action)))
        self._obs = result.obs
        self._t += 1
        reward = -(
            result.info["wait_accrued_this_step"] + result.info["ped_wait_accrued_this_step"]
        ) / REWARD_SCALE
        truncated = self._t >= self.episode_steps
        info = dict(result.info, action_mask=slot_action_mask(result.obs))
        return featurize(result.obs), reward, False, truncated, info
