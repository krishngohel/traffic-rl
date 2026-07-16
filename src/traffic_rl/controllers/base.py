"""Controller interface and the observation contract.

This is the RL contract: `Observation` holds fixed-size numeric arrays only
(per-lane-group arrays are always (8,); phase-indexed arrays are sized to the
config's phase count, with `phase_slots` mapping each phase to its canonical
slot), so it flattens into a Gymnasium Box; `StepResult.info` carries the
per-step reward ingredients so a wrapper never reaches into sim internals.
All classical baselines and any learned policy implement `Controller.act`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from traffic_rl.config import SimConfig


def _default_slots() -> np.ndarray:
    return np.array([1, 3], dtype=np.int64)  # legacy two-phase: NS-T, EW-T


def _default_lanes() -> np.ndarray:
    return np.ones(8)


@dataclass(frozen=True)
class Observation:
    t: float
    queue_lengths: np.ndarray  # (8,) vehicles waiting per lane group
    oldest_wait: np.ndarray  # (8,) wait of head-of-queue vehicle, 0 if empty
    time_since_arrival: np.ndarray  # (8,) gap detector, seconds since last arrival
    arrivals_last_step: np.ndarray  # (8,) detector pulse counts from the last step
    phase_onehot: np.ndarray  # (n_phases,)
    signal_state_onehot: np.ndarray  # (3,) green / yellow / all-red
    phase_elapsed: float  # seconds since the current green began
    ped_call: np.ndarray  # (2,) pending call per ped movement, 0/1
    action_mask: np.ndarray  # (n_phases,) bool, legal desired phases
    phase_slots: np.ndarray = field(default_factory=_default_slots)  # canonical slot per phase
    group_lanes: np.ndarray = field(default_factory=_default_lanes)  # (8,) lanes per group

    @property
    def phase(self) -> int:
        return int(np.argmax(self.phase_onehot))

    @property
    def n_phases(self) -> int:
        return len(self.phase_onehot)

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
