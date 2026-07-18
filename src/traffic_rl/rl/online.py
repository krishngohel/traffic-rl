"""Online learning: the DQN keeps training while the simulation runs.

Used by the viewer's --learn mode. The learner warm-starts from the shipped
weights (fine-tuning: gentle learning rate, small exploration) or from scratch
with --learn-fresh, then runs the standard double-DQN update loop on the live
stream of transitions. Exploration is safe by construction — the signal state
machine ignores illegal requests and the max_call_wait backstop guarantees
service no matter what the learner tries.

This is a continuing task, not episodic: transitions never mark done, so the
target always bootstraps (the same truncation rule train.py uses).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.config import MAX_PHASES
from traffic_rl.controllers.base import Observation, StepResult
from traffic_rl.rl.dqn import DQN, MLP, Replay
from traffic_rl.rl.features import N_FEATURES, featurize, slot_action_mask, slot_to_phase
from traffic_rl.rl.policy import DEFAULT_WEIGHTS
from traffic_rl.rl.train import TARGET_SYNC_EVERY, TRAIN_EVERY, shaped_reward

ONLINE_LR = 1e-4  # gentler than training: don't wreck the warm-started policy
EPS_START_WARM, EPS_START_FRESH = 0.15, 1.0
EPS_END = 0.02
EPS_DECAY_STEPS = 60_000
REPLAY_CAPACITY = 100_000
WARMUP_WARM, WARMUP_FRESH = 1_000, 5_000  # replay fill before updates begin
DEFAULT_ONLINE_OUT = Path("results") / "online_weights.npz"


class OnlineLearner:
    """Acts like a controller but learns from every transition it sees."""

    def __init__(
        self,
        seed: int,
        weights: Path | str = DEFAULT_WEIGHTS,
        fresh: bool = False,
    ):
        self.agent = DQN(N_FEATURES, n_actions=MAX_PHASES, lr=ONLINE_LR, seed=seed)
        self.warm = (not fresh) and Path(weights).exists()
        if self.warm:
            self.agent.online = MLP.load(weights)
            self.agent.target.copy_from(self.agent.online)
            self.agent.opt = type(self.agent.opt)(self.agent.online.params, lr=ONLINE_LR)
        self._eps_start = EPS_START_WARM if self.warm else EPS_START_FRESH
        self._warmup = WARMUP_WARM if self.warm else WARMUP_FRESH
        self.replay = Replay(REPLAY_CAPACITY, N_FEATURES, n_actions=MAX_PHASES)
        self.steps = 0
        self.updates = 0
        self.loss_ema = 0.0
        self.reward_ema = 0.0
        self._pending: tuple[np.ndarray, int] | None = None

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.steps / EPS_DECAY_STEPS)
        return self._eps_start + (EPS_END - self._eps_start) * frac

    def act(self, obs: Observation) -> int:
        features = featurize(obs)
        slot = self.agent.act(features, slot_action_mask(obs), self.epsilon)
        self._pending = (features, slot)
        return slot_to_phase(obs, slot)

    def observe(self, result: StepResult) -> None:
        """Feed back the step outcome for the action from the last act() call."""
        if self._pending is None:
            return
        features, slot = self._pending
        self._pending = None
        reward = shaped_reward(result.info, result.obs)
        self.replay.add(
            features, slot, reward, featurize(result.obs),
            slot_action_mask(result.obs), False,
        )
        self.steps += 1
        self.reward_ema += 0.001 * (reward - self.reward_ema)
        if len(self.replay) >= self._warmup and self.steps % TRAIN_EVERY == 0:
            loss = self.agent.train_step(self.replay)
            self.updates += 1
            self.loss_ema += 0.01 * (loss - self.loss_ema)
        if self.steps % TARGET_SYNC_EVERY == 0:
            self.agent.sync_target()

    def on_reset(self) -> None:
        """Sim was reset: drop the transition that would span the boundary.
        The network, optimizer, and replay all survive — that is the point."""
        self._pending = None

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.online.save(path)
