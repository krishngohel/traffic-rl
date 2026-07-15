"""Controller interface and the observation contract.

This is the Phase-2 RL contract: `Observation` holds fixed-size numeric arrays
only, so it flattens directly into a Gymnasium Box; `StepResult.info` carries the
per-step reward ingredients so a wrapper never reaches into sim internals. All
four classical baselines and any future learned policy implement `Controller.act`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from traffic_rl.config import SimConfig


@dataclass(frozen=True)
class Observation:
    t: float
    queue_lengths: np.ndarray  # (4,) vehicles waiting per approach
    oldest_wait: np.ndarray  # (4,) wait of head-of-queue vehicle, 0 if empty
    time_since_arrival: np.ndarray  # (4,) gap detector, seconds since last arrival
    arrivals_last_step: np.ndarray  # (4,) detector pulse counts from the last step
    phase_onehot: np.ndarray  # (2,)
    signal_state_onehot: np.ndarray  # (3,) green / yellow / all-red
    phase_elapsed: float  # seconds since the current green began
    ped_call: np.ndarray  # (2,) pending call per ped movement, 0/1
    action_mask: np.ndarray  # (2,) bool, legal desired phases

    @property
    def phase(self) -> int:
        return int(np.argmax(self.phase_onehot))

    @property
    def is_green(self) -> bool:
        return bool(self.signal_state_onehot[0])


@dataclass(frozen=True)
class StepResult:
    obs: Observation
    info: dict  # departures_this_step, total_queue, wait_accrued_this_step, t


class Controller(ABC):
    name: str = "base"

    def reset(self, config: SimConfig, rng: np.random.Generator) -> None:  # noqa: B027
        """Optional hook, called once before a run; stateless controllers skip it."""

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """Return the desired phase index. Requesting the current phase = hold."""
