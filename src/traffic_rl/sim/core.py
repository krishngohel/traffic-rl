"""IntersectionSim — the single source of truth for dynamics and event logging.

API shape is the Phase-2 RL contract: reset(seed) -> Observation,
step(action) -> StepResult. The harness, the live viewer, and the Gymnasium
wrapper all consume this same object.

Phase 5: lane groups (through+right and left-turn per approach), protected and
permissive left phases from the config's phase table, gap-acceptance service
for permissive lefts (with sneakers at phase end), shared-lane left friction,
and per-phase clearance intervals. See docs/DESIGN_PHASE5.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from traffic_rl.config import (
    N_APPROACHES,
    N_MOVEMENTS,
    N_PED_MOVEMENTS,
    OPPOSING,
    SNEAKERS_PER_PHASE,
    SimConfig,
    left_group,
    permissive_capacity,
    through_group,
)
from traffic_rl.controllers.base import Observation, StepResult
from traffic_rl.sim.arrivals import ArrivalStreams
from traffic_rl.sim.queues import MovementQueue
from traffic_rl.sim.signal import SignalState, SignalStateMachine

_GAP_CAP = 999.0  # reported cap for the time-since-arrival gap detector
_OPP_RATE_TAU = 120.0  # EMA time constant for opposing-flow estimate (s)


@dataclass
class EventLog:
    veh_arrival: list[float] = field(default_factory=list)
    veh_depart: list[float] = field(default_factory=list)  # nan while queued
    veh_group: list[int] = field(default_factory=list)  # lane group 0..7
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
            "veh_group": np.asarray(self.veh_group, dtype=np.int64),
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
        self.phases = config.phases
        self.n_phases = len(self.phases)
        self._phase_slots = np.array([p.slot for p in self.phases], dtype=np.int64)
        self._group_lanes = np.array(
            [config.group_lanes(g) for g in range(N_MOVEMENTS)], dtype=np.float64
        )
        self._has_reset = False

    def reset(self, seed: int) -> Observation:
        cfg = self.config
        self.t = 0.0
        self.streams = ArrivalStreams(
            seed, cfg.demand, cfg.dt, layout=cfg.layout, schedule=cfg.demand_schedule
        )
        self.signal = SignalStateMachine(cfg.timing, self.phases)
        self.queues = [
            MovementQueue(cfg.group_sat_flow(g), cfg.timing.startup_lost)
            for g in range(N_MOVEMENTS)
        ]
        self.ped_call_pending = np.zeros(N_PED_MOVEMENTS, dtype=bool)
        self.waiting_peds: list[list[int]] = [[] for _ in range(N_PED_MOVEMENTS)]
        self.log = EventLog()
        self._last_arrival_counts = np.zeros(N_MOVEMENTS, dtype=np.int64)
        self._pending_injections = [0] * N_APPROACHES
        # EMA of each approach's through-group arrival rate (veh/s): the
        # opposing-flow estimate the permissive-left gap model consumes.
        self._through_rate_ema = np.zeros(N_APPROACHES)
        self._has_reset = True
        return self._observe()

    def inject_vehicles(self, approach: int, count: int) -> None:
        """Queue vehicles (e.g. arriving from an upstream intersection) to enter
        the given approach's through group during the next step's arrivals."""
        self._pending_injections[approach] += count

    @property
    def event_log(self) -> EventLog:
        return self.log

    def action_mask(self) -> np.ndarray:
        return self.signal.legal_actions()

    # ------------------------------------------------------------------ dynamics

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
        events = self.signal.tick(
            dt, self.ped_call_pending, conflicting_call, self._phase_calls()
        )

        departures: list[tuple[int, int]] = []  # (veh_id, group)
        if events.yellow_started is not None:
            # Sneakers: lefts that were in the box waiting for a gap complete
            # their turn as the permissive phase ends.
            for g in self.phases[events.yellow_started].permissive_lefts:
                for veh_id in self.queues[g].pop(SNEAKERS_PER_PHASE):
                    departures.append((veh_id, g))
        if events.green_started is not None:
            started = self.phases[events.green_started]
            for g in started.movements + started.permissive_lefts:
                self.queues[g].on_green_start()
        if events.walk_started is not None:
            self._serve_walk(events.walk_started, t_new)

        # 3. Arrivals (rates may be time-varying under a demand schedule).
        # Poisson draws happen for every lane group (keeps streams aligned across
        # configs), but masked approaches discard theirs and are fed by
        # injection from upstream instead. Order per group: external first,
        # then injected — the network mirror relies on this FIFO order.
        veh_counts = self.streams.vehicle_counts(self.t)
        arrivals = np.zeros(N_MOVEMENTS, dtype=np.int64)
        for g in range(N_MOVEMENTS):
            approach = g % N_APPROACHES
            external = int(veh_counts[g]) if cfg.external_vehicle_arrivals[approach] else 0
            injected = 0
            if g < N_APPROACHES:  # injections always enter the through group
                injected = self._pending_injections[g]
                self._pending_injections[g] = 0
            for _ in range(external + injected):
                veh_id = len(self.log.veh_arrival)
                self.log.veh_arrival.append(t_new)
                self.log.veh_depart.append(np.nan)
                self.log.veh_group.append(g)
                self.queues[g].add(veh_id, t_new)
            arrivals[g] = external + injected
        self._last_arrival_counts = arrivals
        self._through_rate_ema += (dt / _OPP_RATE_TAU) * (
            arrivals[:N_APPROACHES] / dt - self._through_rate_ema
        )

        ped_counts = self.streams.ped_counts(self.t)
        walk_movement = (
            self.signal.walk_movement if self.signal.in_walk_window else None
        )
        for m in range(N_PED_MOVEMENTS):
            for _ in range(int(ped_counts[m])):
                ped_id = len(self.log.ped_arrival)
                self.log.ped_arrival.append(t_new)
                self.log.ped_movement.append(m)
                if m == walk_movement:
                    self.log.ped_walk_start.append(t_new)  # crosses immediately
                else:
                    self.log.ped_walk_start.append(np.nan)
                    self.waiting_peds[m].append(ped_id)
                    self.ped_call_pending[m] = True

        # 4. Discharge on green.
        if self.signal.state == SignalState.GREEN:
            phase = self.phases[self.signal.phase]
            elapsed = self.signal.state_elapsed
            for g in phase.movements:
                mult = self._shared_left_multiplier(g)
                service = self.queues[g].saturation_service(elapsed, dt, mult)
                for veh_id in self.queues[g].discharge(service):
                    departures.append((veh_id, g))
            for g in phase.permissive_lefts:
                service = self._permissive_service(g) * dt
                for veh_id in self.queues[g].discharge(service):
                    departures.append((veh_id, g))

        departures_by_group = np.zeros(N_MOVEMENTS, dtype=np.int64)
        for veh_id, g in departures:
            self.log.veh_depart[veh_id] = t_new
            departures_by_group[g] += 1

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
            "departures_this_step": len(departures),
            "departures_by_group": departures_by_group,
            "arrivals_by_group": arrivals,
            # Legacy per-approach views (through + left of each approach).
            "departures_by_approach": (
                departures_by_group[:N_APPROACHES] + departures_by_group[N_APPROACHES:]
            ),
            "arrivals_by_approach": arrivals[:N_APPROACHES] + arrivals[N_APPROACHES:],
            "total_queue": total_queue,
            "wait_accrued_this_step": total_queue * dt,
            "peds_waiting": peds_waiting,
            "ped_wait_accrued_this_step": peds_waiting * dt,
        }
        return StepResult(obs=obs, info=info)

    # ------------------------------------------------------------------ helpers

    def _shared_left_multiplier(self, group: int) -> float:
        """Discharge multiplier for a through group whose lane carries SHARED
        lefts: the left share flows at the permissive rate, the rest at
        saturation. Point-queue approximation, applied to the whole group."""
        if group >= N_APPROACHES:
            return 1.0
        approach = group
        demand = self.streams.demand_now(self.t)
        p_left = demand.shared_left_fraction(approach, self.config.layout)
        if p_left <= 0.0:
            return 1.0
        c_perm = self._permissive_service(left_group(approach), from_approach=approach)
        s_lane = self.config.sat_flow  # single-lane saturation, veh/s
        return (1.0 - p_left) + p_left * min(c_perm / s_lane, 1.0)

    def _permissive_service(self, group: int, from_approach: int | None = None) -> float:
        """Gap-acceptance service rate (veh/s) for a left group filtering
        through opposing traffic. Zero while the opposing through queue is
        discharging (no gaps); otherwise the classic exponential-gap capacity
        against the opposing arrival-rate estimate."""
        approach = from_approach if from_approach is not None else group - N_APPROACHES
        opposing = OPPOSING[approach]
        if len(self.queues[through_group(opposing)]) > 0:
            return 0.0
        return permissive_capacity(self._through_rate_ema[opposing])

    def _served_groups(self, phase_idx: int) -> tuple[int, ...]:
        p = self.phases[phase_idx]
        return p.movements + p.permissive_lefts

    def _conflicting_call(self, phase_idx: int) -> bool:
        served = set(self._served_groups(phase_idx))
        for g in range(N_MOVEMENTS):
            if g not in served and len(self.queues[g]) > 0:
                return True
        current_ped = self.phases[phase_idx].ped_movement
        return any(
            self.ped_call_pending[m] for m in range(N_PED_MOVEMENTS) if m != current_ped
        )

    def _phase_calls(self) -> np.ndarray:
        calls = np.zeros(self.n_phases, dtype=bool)
        for i, p in enumerate(self.phases):
            has_veh = any(len(self.queues[g]) > 0 for g in p.movements + p.permissive_lefts)
            has_ped = p.ped_movement is not None and self.ped_call_pending[p.ped_movement]
            calls[i] = has_veh or has_ped
        return calls

    def _serve_walk(self, movement: int, t: float) -> None:
        for ped_id in self.waiting_peds[movement]:
            self.log.ped_walk_start[ped_id] = t
        self.waiting_peds[movement].clear()
        self.ped_call_pending[movement] = False

    def _observe(self) -> Observation:
        t = self.t
        queue_lengths = np.array([len(q) for q in self.queues], dtype=np.float64)
        oldest_wait = np.zeros(N_MOVEMENTS)
        gaps = np.zeros(N_MOVEMENTS)
        for g, q in enumerate(self.queues):
            oldest = q.oldest_arrival()
            oldest_wait[g] = (t - oldest) if oldest is not None else 0.0
            gaps[g] = min(t - q.last_arrival_t, _GAP_CAP)
        phase_onehot = np.zeros(self.n_phases)
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
            phase_slots=self._phase_slots,
            group_lanes=self._group_lanes,
        )
