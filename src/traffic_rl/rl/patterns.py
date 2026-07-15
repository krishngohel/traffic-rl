"""Traffic-pattern tracking: what the policy knows that a gap detector cannot.

A vehicle-actuated controller sees the last ~3 seconds. PatternTracker gives
the learned policy exponential moving estimates of the per-approach arrival
RATE at two time constants — a fast one (~1 min, "what is happening right
now") and a slow one (~15 min, "what kind of hour is this") — computed purely
from the same detector pulses (obs.arrivals_last_step) real hardware has.
Deterministic, stateful, and shared verbatim between training and inference.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import N_APPROACHES
from traffic_rl.controllers.base import Observation
from traffic_rl.rl.features import N_FEATURES, featurize

FAST_TAU = 60.0  # seconds
SLOW_TAU = 900.0
N_PATTERN_FEATURES = 2 * N_APPROACHES
N_FEATURES_PATTERN = N_FEATURES + N_PATTERN_FEATURES
_RATE_NORM = 7.0  # log1p(1000 veh/h) ≈ 6.9


class PatternTracker:
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.fast = np.zeros(N_APPROACHES)
        self.slow = np.zeros(N_APPROACHES)

    def update(self, obs: Observation) -> None:
        rate_now = obs.arrivals_last_step / self.dt * 3600.0  # veh/h, instantaneous
        self.fast += (self.dt / FAST_TAU) * (rate_now - self.fast)
        self.slow += (self.dt / SLOW_TAU) * (rate_now - self.slow)

    def features(self) -> np.ndarray:
        return np.concatenate(
            [np.log1p(self.fast) / _RATE_NORM, np.log1p(self.slow) / _RATE_NORM]
        ).astype(np.float32)


def featurize_with_patterns(obs: Observation, tracker: PatternTracker) -> np.ndarray:
    """Update the tracker with this step's detector pulses, then featurize.
    Call exactly once per sim step (it mutates the tracker)."""
    tracker.update(obs)
    return np.concatenate([featurize(obs), tracker.features()])
