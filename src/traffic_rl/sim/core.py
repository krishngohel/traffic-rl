"""IntersectionSim — the single source of truth for dynamics and event logging.

API shape is the Phase-2 RL contract: reset(seed) -> Observation,
step(action) -> StepResult. The harness, the live viewer, and a future Gymnasium
wrapper all consume this same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from traffic_rl.config import (
    N_APPROACHES,
    N_PED_MOVEMENTS,
    N_PHASES,
    PHASE_APPROACHES,
    SimConfig,
)
from traffic_rl.controllers.base import Observation, StepResult
from traffic_rl.sim.arrivals import ArrivalStreams
from traffic_rl.sim.queues import ApproachQueue
from traffic_rl.sim.signal import SignalState, SignalStateMachine

_GAP_CAP = 999.0  # reported cap for the time-since-arrival gap detector


@dataclass
class EventLog:
    veh_arrival: list[float] = field(default_factory=list)
    veh_depart: list[float] = field(default_factory=list)  # nan while queued
    veh_approach: list[int] = field(default_factory=list)
    ped_arrival: list[float] = field(default_factory=list)
    ped_walk_start: list[float] = field(default_factory=list)  # nan while waiting
    ped_movement: list[int] = field(default_factory=list)
    step_t: list[float] = field(default_factory=list)
    step_phase: list[int] = field(default_factory=list)
    step_state: list[int] = field(default_factory=list)
    step_queues: list[tuple[int, ...]] = field(default_factory=list)

    def finalize(self) -> dict[str, np.ndarray]:
        return {
            "veh_arrival": np.asarray(self.veh_arrival),
            "veh_depart": np.asarray(self.veh_depart),
            "veh_approach": np.asarray(self.veh_approach, dtype=np.int64),
            "ped_arrival": np.asarray(self.ped_arrival),
            "ped_walk_start": np.asarray(self.ped_walk_start),
            "ped_movement": np.asarray(self.ped_movement, dtype=np.int64),
            "step_t": np.asarray(self.step_t),
            "step_phase": np.asarray(self.step_phase, dtype=np.int64),
            "step_state": np.asarray(self.step_state, dtype=np.int64),
            "step_queues": np.asarray(self.step_queues, dtype=np.int64),
        }


class IntersectionSim:
    def __init__(self, config: SimConfig):
        self.config = config
        self._has_reset = False

    def reset(self, seed: int) -> Observation:
        cfg = self.config
        self.t = 0.0
        self.streams = ArrivalStreams(seed, cfg.demand, cfg.dt)
        self.signal = SignalStateMachine(cfg.timing)
        self.queues = [
            ApproachQueue(cfg.sat_flow, cfg.timing.startup_lost) for _ in range(N_APPROACHES)
        ]
        self.ped_call_pending = np.zeros(N_PED_MOVEMENTS, dtype=bool)
        self.waiting_peds: list[list[int]] = [[] for _ in range(N_PED_MOVEMENTS)]
        self.log = EventLog()
        self._last_arrival_counts = np.zeros(N_APPROACHES, dtype=np.int64)
        self._has_reset = True
        return self._observe()

    @property
    def event_log(self) -> EventLog:
        return self.log

    def action_mask(self) -> np.ndarray:
        return self.signal.legal_actions()

    def step(self, action: int) -> StepResult:
        assert self._has_reset, "call reset(seed) first"
        cfg = self.config
        dt = cfg.dt
        t_new = self.t + dt

        # 1. Controller intent (illegal requests ignored by the state machine).
        self.signal.request_phase(int(action))

        # 2. Advance the signal. Conflicting-call flag (for the backstop) uses the
        #    state as the controller saw it.
        conflicting_call = self._conflicting_call(self.signal.phase)
        events = self.signal.tick(dt, self.ped_call_pending, conflicting_call)

        if events.green_started is not None:
            for a in PHASE_APPROACHES[events.green_started]:
                self.queues[a].on_green_start()
        if events.walk_started is not None:
            self._serve_walk(events.walk_started, t_new)

        # 3. Arrivals.
        veh_counts = self.streams.vehicle_counts()
        for a in range(N_APPROACHES):
            for _ in range(int(veh_counts[a])):
                veh_id = len(self.log.veh_arrival)
                self.log.veh_arrival.append(t_new)
                self.log.veh_depart.append(np.nan)
                self.log.veh_approach.append(a)
                self.queues[a].add(veh_id, t_new)
        self._last_arrival_counts = veh_counts

        ped_counts = self.streams.ped_counts()
        for m in range(N_PED_MOVEMENTS):
            for _ in range(int(ped_counts[m])):
                ped_id = len(self.log.ped_arrival)
                self.log.ped_arrival.append(t_new)
                self.log.ped_movement.append(m)
                if m == self.signal.phase and self.signal.in_walk_window:
                    self.log.ped_walk_start.append(t_new)  # crosses immediately
                else:
                    self.log.ped_walk_start.append(np.nan)
                    self.waiting_peds[m].append(ped_id)
                    self.ped_call_pending[m] = True

        # 4. Discharge on green.
        departures = 0
        if self.signal.state == SignalState.GREEN:
            for a in PHASE_APPROACHES[self.signal.phase]:
                for veh_id in self.queues[a].discharge(self.signal.state_elapsed, dt):
                    self.log.veh_depart[veh_id] = t_new
                    departures += 1

        # 5. Bookkeeping.
        queue_lengths = tuple(len(q) for q in self.queues)
        total_queue = sum(queue_lengths)
        self.log.step_t.append(t_new)
        self.log.step_phase.append(self.signal.phase)
        self.log.step_state.append(int(self.signal.state))
        self.log.step_queues.append(queue_lengths)
        self.t = t_new

        obs = self._observe()
        peds_waiting = sum(len(w) for w in self.waiting_peds)
        info = {
            "t": t_new,
            "departures_this_step": departures,
            "total_queue": total_queue,
            "wait_accrued_this_step": total_queue * dt,
            "peds_waiting": peds_waiting,
            "ped_wait_accrued_this_step": peds_waiting * dt,
        }
        return StepResult(obs=obs, info=info)

    # ------------------------------------------------------------------ helpers

    def _conflicting_call(self, phase: int) -> bool:
        conflicting_approaches = PHASE_APPROACHES[(phase + 1) % N_PHASES]
        if any(len(self.queues[a]) > 0 for a in conflicting_approaches):
            return True
        return bool(self.ped_call_pending[(phase + 1) % N_PED_MOVEMENTS])

    def _serve_walk(self, movement: int, t: float) -> None:
        for ped_id in self.waiting_peds[movement]:
            self.log.ped_walk_start[ped_id] = t
        self.waiting_peds[movement].clear()
        self.ped_call_pending[movement] = False

    def _observe(self) -> Observation:
        t = self.t
        queue_lengths = np.array([len(q) for q in self.queues], dtype=np.float64)
        oldest_wait = np.zeros(N_APPROACHES)
        gaps = np.zeros(N_APPROACHES)
        for a, q in enumerate(self.queues):
            oldest = q.oldest_arrival()
            oldest_wait[a] = (t - oldest) if oldest is not None else 0.0
            gaps[a] = min(t - q.last_arrival_t, _GAP_CAP)
        phase_onehot = np.zeros(N_PHASES)
        phase_onehot[self.signal.phase] = 1.0
        state_onehot = np.zeros(3)
        state_onehot[int(self.signal.state)] = 1.0
        return Observation(
            t=t,
            queue_lengths=queue_lengths,
            oldest_wait=oldest_wait,
            time_since_arrival=gaps,
            arrivals_last_step=self._last_arrival_counts.astype(np.float64),
            phase_onehot=phase_onehot,
            signal_state_onehot=state_onehot,
            phase_elapsed=(
                self.signal.state_elapsed if self.signal.state == SignalState.GREEN else 0.0
            ),
            ped_call=self.ped_call_pending.astype(np.float64),
            action_mask=self.signal.legal_actions(),
        )
