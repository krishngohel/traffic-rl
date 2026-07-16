"""Observation -> normalized feature vector, shared by training and inference.

Kept in one place so the trained policy and the deployed RLController can never
disagree about preprocessing.

The policy lives in CANONICAL SLOT space (0 NS-left, 1 NS-through, 2 EW-left,
3 EW-through): phase one-hots and action masks are scattered from the config's
compact phase indices into the 4 slots via obs.phase_slots, and a slot-exists
mask tells the policy which phases this site has. One set of weights therefore
runs a plain 2-phase signal and a protected-left signal alike.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import MAX_PHASES
from traffic_rl.controllers.base import Observation

# queues(8) + oldest(8) + gaps(8) + lanes(8) + slot phase onehot(4) + state(3)
# + elapsed(1) + ped calls(2) + slot action mask(4) + slot exists(4)
N_FEATURES = 50


def slot_action_mask(obs: Observation) -> np.ndarray:
    """Legal-action mask in canonical slot space (inactive slots illegal)."""
    mask = np.zeros(MAX_PHASES, dtype=bool)
    mask[obs.phase_slots] = obs.action_mask
    return mask


def slot_exists(obs: Observation) -> np.ndarray:
    exists = np.zeros(MAX_PHASES)
    exists[obs.phase_slots] = 1.0
    return exists


def slot_to_phase(obs: Observation, slot: int) -> int:
    """Canonical slot -> this site's compact phase index (hold current phase
    if the slot does not exist here — a masked policy never picks one)."""
    matches = np.flatnonzero(obs.phase_slots == slot)
    return int(matches[0]) if len(matches) else obs.phase


def featurize(obs: Observation) -> np.ndarray:
    phase_slot_onehot = np.zeros(MAX_PHASES)
    phase_slot_onehot[obs.phase_slots[obs.phase]] = 1.0
    return np.concatenate(
        [
            np.log1p(obs.queue_lengths) / 4.0,  # (8) ~0..1 for queues up to ~50
            np.minimum(obs.oldest_wait, 300.0) / 300.0,  # (8)
            np.minimum(obs.time_since_arrival, 30.0) / 30.0,  # (8)
            obs.group_lanes / 2.0,  # (8) capacity awareness: lanes per group
            phase_slot_onehot,  # (4)
            obs.signal_state_onehot,  # (3)
            [min(obs.phase_elapsed, 120.0) / 120.0],  # (1)
            obs.ped_call,  # (2)
            slot_action_mask(obs).astype(np.float64),  # (4)
            slot_exists(obs),  # (4)
        ]
    ).astype(np.float32)
