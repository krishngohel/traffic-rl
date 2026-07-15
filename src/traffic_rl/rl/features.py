"""Observation -> normalized feature vector, shared by training and inference.

Kept in one place so the trained policy and the deployed RLController can never
disagree about preprocessing.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.controllers.base import Observation

N_FEATURES = 22


def featurize(obs: Observation) -> np.ndarray:
    return np.concatenate(
        [
            np.log1p(obs.queue_lengths) / 4.0,  # (4) ~0..1 for queues up to ~50
            np.minimum(obs.oldest_wait, 300.0) / 300.0,  # (4)
            np.minimum(obs.time_since_arrival, 30.0) / 30.0,  # (4)
            obs.phase_onehot,  # (2)
            obs.signal_state_onehot,  # (3)
            [min(obs.phase_elapsed, 120.0) / 120.0],  # (1)
            obs.ped_call,  # (2)
            obs.action_mask.astype(np.float64),  # (2)
        ]
    ).astype(np.float32)
