"""Signal state machine — owns every safety invariant.

Controllers only express a desired phase; this machine inserts the yellow and
all-red clearance intervals (per-phase durations, e.g. ITE formula values),
enforces per-phase min green, holds green while a pedestrian walk + clearance
is in progress, and force-switches at the anti-starvation backstop. Illegal
requests are ignored, which is what makes any controller (including an RL
policy) safe by construction.

Phase count and composition come from the config's phase table (2-phase
legacy through 4-phase protected-left plans); the machine itself is
structure-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from traffic_rl.config import Phase, SignalTimingConfig

_EPS = 1e-9


class SignalState(IntEnum):
    GREEN = 0
    YELLOW = 1
    ALL_RED = 2


@dataclass
class TickEvents:
    walk_started: int | None = None  # ped movement index whose walk began this step
    green_started: int | None = None  # phase whose green began this step
    yellow_started: int | None = None  # phase whose green ended this step (sneaker hook)


class SignalStateMachine:
    def __init__(self, timing: SignalTimingConfig, phases: tuple[Phase, ...]):
        self.timing = timing
        self.phases = phases
        self.n_phases = len(phases)
        # Start on the first through phase (a real controller's rest phase).
        self._rest_phase = next(
            (i for i, p in enumerate(phases) if p.ped_movement is not None), 0
        )
        self.reset()

    def reset(self) -> None:
        self.phase = self._rest_phase
        self.state = SignalState.GREEN
        self.state_elapsed = 0.0
        self.desired = self.phase
        self.walk_active = False
        self.walk_movement = 0
        self.walk_elapsed = 0.0
        self.walk_served_this_green = False
        # Seconds each phase's call has been pending while unserved.
        self.call_age = np.zeros(self.n_phases)

    # ------------------------------------------------------------------ queries

    @property
    def current(self) -> Phase:
        return self.phases[self.phase]

    def _walk_service(self) -> float:
        return self.timing.ped_service(self.walk_movement)

    @property
    def ped_lock_active(self) -> bool:
        return self.walk_active and self.walk_elapsed < self._walk_service() - _EPS

    @property
    def in_walk_window(self) -> bool:
        """True while the WALK indication itself is up (peds may start crossing)."""
        return self.walk_active and self.walk_elapsed < self.timing.walk - _EPS

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(self.n_phases, dtype=bool)
        mask[self.phase] = True
        if (
            self.state == SignalState.GREEN
            and self.state_elapsed >= self.timing.min_green_for(self.current) - _EPS
            and not self.ped_lock_active
        ):
            mask[:] = True
        return mask

    def _next_phase_with_call(self, phase_calls: np.ndarray) -> int:
        """First phase after the current one (cyclic order) with a call."""
        for step in range(1, self.n_phases):
            candidate = (self.phase + step) % self.n_phases
            if phase_calls[candidate]:
                return candidate
        return (self.phase + 1) % self.n_phases

    # ------------------------------------------------------------------ commands

    def request_phase(self, target: int) -> None:
        # Once yellow has started the transition is committed: requests during
        # yellow/all-red are ignored, otherwise a controller could cancel a
        # backstop-forced switch and starve the conflicting approaches.
        if self.state == SignalState.GREEN and self.legal_actions()[target]:
            self.desired = target

    def tick(
        self,
        dt: float,
        ped_calls: np.ndarray,
        conflicting_call: bool,
        phase_calls: np.ndarray | None = None,
    ) -> TickEvents:
        """Advance one step. `phase_calls[p]` = demand exists for phase p
        (used by the backstop to pick a phase actually worth serving)."""
        events = TickEvents()
        if phase_calls is None:
            phase_calls = np.ones(self.n_phases, dtype=bool)
        # Age pending calls; the phase currently green (or being switched to)
        # is being served, so its age stays zero.
        serving = self.phase if self.state == SignalState.GREEN else self.desired
        for p in range(self.n_phases):
            if phase_calls[p] and p != serving:
                self.call_age[p] += dt
            else:
                self.call_age[p] = 0.0
        if self.state == SignalState.GREEN:
            self.state_elapsed += dt
            if self.walk_active:
                self.walk_elapsed += dt
                if self.walk_elapsed >= self._walk_service() - _EPS:
                    self.walk_active = False
            # Serve a ped call that arrived mid-green on the current phase, at most
            # once per green, and only if service fits under the backstop.
            ped = self.current.ped_movement
            if (
                ped is not None
                and ped_calls[ped]
                and not self.walk_active
                and not self.walk_served_this_green
                and self.state_elapsed + self.timing.ped_service(ped)
                <= self.timing.max_green_backstop
            ):
                self._start_walk(ped, events)
            # Anti-starvation backstop: force a switch once a conflicting call has
            # waited through a full backstop green (applies to every controller).
            if (
                self.state_elapsed >= self.timing.max_green_backstop - _EPS
                and conflicting_call
            ):
                self.desired = self._next_phase_with_call(phase_calls)
            # Second guarantee (>2 phases): a controller cycling between two
            # phases could starve a third forever without ever tripping the
            # green-duration backstop. Any call waiting past max_call_wait is
            # force-served. Fires every tick until the transition commits, so a
            # controller cannot cancel it by re-requesting the current phase.
            if self.call_age.max() >= self.timing.max_call_wait - _EPS:
                self.desired = int(np.argmax(self.call_age))
            if self.desired != self.phase and self.legal_actions()[self.desired]:
                self.state = SignalState.YELLOW
                self.state_elapsed = 0.0
                events.yellow_started = self.phase
        elif self.state == SignalState.YELLOW:
            self.state_elapsed += dt
            if self.state_elapsed >= self.timing.yellow_for(self.current) - _EPS:
                self.state = SignalState.ALL_RED
                self.state_elapsed = 0.0
        else:  # ALL_RED
            self.state_elapsed += dt
            if self.state_elapsed >= self.timing.all_red_for(self.current) - _EPS:
                self.phase = self.desired
                self.state = SignalState.GREEN
                self.state_elapsed = 0.0
                self.walk_served_this_green = False
                events.green_started = self.phase
                ped = self.current.ped_movement
                if ped is not None and ped_calls[ped]:
                    self._start_walk(ped, events)
        return events

    def _start_walk(self, movement: int, events: TickEvents) -> None:
        self.walk_active = True
        self.walk_movement = movement
        self.walk_elapsed = 0.0
        self.walk_served_this_green = True
        events.walk_started = movement
