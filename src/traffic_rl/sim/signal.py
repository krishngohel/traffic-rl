"""Signal state machine — owns every safety invariant.

Controllers only express a desired phase; this machine inserts yellow and all-red,
enforces min green, holds green while a pedestrian walk + clearance is in progress,
and force-switches at the anti-starvation backstop. Illegal requests are ignored,
which is what makes any controller (including a future RL policy) safe by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from traffic_rl.config import N_PHASES, SignalTimingConfig

_EPS = 1e-9


class SignalState(IntEnum):
    GREEN = 0
    YELLOW = 1
    ALL_RED = 2


@dataclass
class TickEvents:
    walk_started: int | None = None  # ped movement index whose walk began this step
    green_started: int | None = None  # phase whose green began this step


class SignalStateMachine:
    def __init__(self, timing: SignalTimingConfig):
        self.timing = timing
        self.reset()

    def reset(self) -> None:
        self.phase = 0
        self.state = SignalState.GREEN
        self.state_elapsed = 0.0
        self.desired = 0
        self.walk_active = False
        self.walk_elapsed = 0.0
        self.walk_served_this_green = False

    # ------------------------------------------------------------------ queries

    @property
    def ped_lock_active(self) -> bool:
        return self.walk_active and self.walk_elapsed < self.timing.ped_service - _EPS

    @property
    def in_walk_window(self) -> bool:
        """True while the WALK indication itself is up (peds may start crossing)."""
        return self.walk_active and self.walk_elapsed < self.timing.walk - _EPS

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(N_PHASES, dtype=bool)
        mask[self.phase] = True
        if (
            self.state == SignalState.GREEN
            and self.state_elapsed >= self.timing.min_green - _EPS
            and not self.ped_lock_active
        ):
            mask[:] = True
        return mask

    # ------------------------------------------------------------------ commands

    def request_phase(self, target: int) -> None:
        # Once yellow has started the transition is committed: requests during
        # yellow/all-red are ignored, otherwise a controller could cancel a
        # backstop-forced switch and starve the conflicting approaches.
        if self.state == SignalState.GREEN and self.legal_actions()[target]:
            self.desired = target

    def tick(self, dt: float, ped_calls: np.ndarray, conflicting_call: bool) -> TickEvents:
        events = TickEvents()
        if self.state == SignalState.GREEN:
            self.state_elapsed += dt
            if self.walk_active:
                self.walk_elapsed += dt
                if self.walk_elapsed >= self.timing.ped_service - _EPS:
                    self.walk_active = False
            # Serve a ped call that arrived mid-green on the current phase, at most
            # once per green, and only if service fits under the backstop.
            if (
                ped_calls[self.phase]
                and not self.walk_active
                and not self.walk_served_this_green
                and self.state_elapsed + self.timing.ped_service
                <= self.timing.max_green_backstop
            ):
                self._start_walk(events)
            # Anti-starvation backstop: force a switch once a conflicting call has
            # waited through a full backstop green (applies to every controller).
            if (
                self.state_elapsed >= self.timing.max_green_backstop - _EPS
                and conflicting_call
            ):
                self.desired = (self.phase + 1) % N_PHASES
            if self.desired != self.phase and self.legal_actions()[self.desired]:
                self.state = SignalState.YELLOW
                self.state_elapsed = 0.0
        elif self.state == SignalState.YELLOW:
            self.state_elapsed += dt
            if self.state_elapsed >= self.timing.yellow - _EPS:
                self.state = SignalState.ALL_RED
                self.state_elapsed = 0.0
        else:  # ALL_RED
            self.state_elapsed += dt
            if self.state_elapsed >= self.timing.all_red - _EPS:
                self.phase = self.desired
                self.state = SignalState.GREEN
                self.state_elapsed = 0.0
                self.walk_served_this_green = False
                events.green_started = self.phase
                if ped_calls[self.phase]:
                    self._start_walk(events)
        return events

    def _start_walk(self, events: TickEvents) -> None:
        self.walk_active = True
        self.walk_elapsed = 0.0
        self.walk_served_this_green = True
        events.walk_started = self.phase
