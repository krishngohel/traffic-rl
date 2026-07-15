"""Controllers for the corridor network.

- IndependentNetworkController: one single-intersection controller per node,
  no communication (what most real corridors actually run).
- GreenWaveController: coordinated fixed-time — a common cycle with per-node
  Webster splits and offsets equal to the link travel time, so eastbound
  platoons ride a green wave. Two-stage like Webster: observes flows under the
  naive plan for 900 s, never peeks at true rates.
- NetworkMaxPressureController: max-pressure with REAL downstream queues at
  last — pressure(phase) = sum of (upstream - downstream) queue over served
  approaches, the setting Varaiya's theory is actually about.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from traffic_rl.config import N_PHASES, PHASE_APPROACHES
from traffic_rl.controllers.base import Controller, Observation
from traffic_rl.controllers.fixed_time import NAIVE_PLAN, FixedTimeController, FixedTimePlan
from traffic_rl.sim.network import EASTBOUND, WESTBOUND, NetworkConfig

C_MIN, C_MAX = 40.0, 150.0


class NetworkController(ABC):
    name: str = "network-base"

    def reset(self, config: NetworkConfig, rng: np.random.Generator) -> None:  # noqa: B027
        """Optional hook, called once before a run."""

    @abstractmethod
    def act(self, observations: list[Observation]) -> list[int]:
        """One desired phase per node, given every node's observation."""


class IndependentNetworkController(NetworkController):
    def __init__(self, factory: Callable[[], Controller], name: str):
        self.factory = factory
        self.name = name

    def reset(self, config: NetworkConfig, rng: np.random.Generator) -> None:
        self.controllers = [self.factory() for _ in range(config.n_nodes)]
        for i, c in enumerate(self.controllers):
            c.reset(config.node_config(i), rng)

    def act(self, observations: list[Observation]) -> list[int]:
        return [c.act(o) for c, o in zip(self.controllers, observations, strict=True)]


class _OffsetFixedTime(FixedTimeController):
    """Fixed-time plan whose cycle clock is shifted by a per-node offset."""

    def __init__(self, plan: FixedTimePlan, offset: float):
        super().__init__(plan)
        self.offset = offset

    def act(self, obs: Observation) -> int:
        return super().act(dataclasses.replace(obs, t=(obs.t - self.offset) % self._cycle))


def _plan_for_cycle(flows: np.ndarray, cycle: float, timing) -> FixedTimePlan:
    """Webster-style splits for a GIVEN common cycle, with the ped floor."""
    sat = 1800.0
    y = np.array(
        [max(flows[a] for a in PHASE_APPROACHES[p]) / sat for p in range(N_PHASES)]
    )
    y = np.maximum(y, 1e-6)
    L = N_PHASES * timing.lost_time_per_phase
    floor_eff = max(timing.ped_service, timing.min_green) - timing.startup_lost
    g_eff = (y / y.sum()) * (cycle - L)
    for p in range(N_PHASES):
        if g_eff[p] < floor_eff:
            deficit = floor_eff - g_eff[p]
            g_eff[p] += deficit
            g_eff[np.argmax(g_eff)] -= deficit
    return FixedTimePlan(greens=tuple(float(g + timing.startup_lost) for g in g_eff))


class GreenWaveController(NetworkController):
    name = "greenwave"

    def __init__(self, observation_window: float = 900.0):
        self.observation_window = observation_window

    def reset(self, config: NetworkConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._counts = np.zeros((config.n_nodes, 4))
        self._naive = [FixedTimeController(NAIVE_PLAN) for _ in range(config.n_nodes)]
        for i, c in enumerate(self._naive):
            c.reset(config.node_config(i), rng)
        self._coordinated: list[_OffsetFixedTime] | None = None

    def act(self, observations: list[Observation]) -> list[int]:
        t = observations[0].t
        if t < self.observation_window:
            for i, obs in enumerate(observations):
                self._counts[i] += obs.arrivals_last_step
            return [c.act(o) for c, o in zip(self._naive, observations, strict=True)]
        if self._coordinated is None:
            self._build_plans()
        return [c.act(o) for c, o in zip(self._coordinated, observations, strict=True)]

    def _build_plans(self) -> None:
        timing = self.config.timing
        flows = self._counts / self.observation_window * 3600.0
        # Common cycle: the largest per-node Webster cycle (the busiest node
        # binds the corridor), with the ped-floor min-cycle rule applied.
        from traffic_rl.controllers.webster import webster_plan

        cycles = []
        for i in range(self.config.n_nodes):
            plan = webster_plan(flows[i], 1800.0, timing, green_floor=timing.ped_service)
            cycles.append(plan.cycle(timing.yellow, timing.all_red))
        common = float(np.clip(max(cycles), C_MIN, C_MAX))
        self._coordinated = []
        for i in range(self.config.n_nodes):
            plan = _plan_for_cycle(flows[i], common, timing)
            offset = (i * self.config.link_travel) % common
            controller = _OffsetFixedTime(plan, offset)
            controller.reset(self.config.node_config(i), np.random.default_rng(0))
            self._coordinated.append(controller)


class NetworkMaxPressureController(NetworkController):
    name = "max_pressure_net"

    def __init__(self, decision_interval: float = 15.0):
        self.decision_interval = decision_interval

    def reset(self, config: NetworkConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._last_decision_t = -np.inf
        self._targets = [0] * config.n_nodes

    def _downstream(self, observations: list[Observation], i: int, a: int) -> float:
        if a == EASTBOUND and i + 1 < self.config.n_nodes:
            return float(observations[i + 1].queue_lengths[EASTBOUND])
        if a == WESTBOUND and i - 1 >= 0:
            return float(observations[i - 1].queue_lengths[WESTBOUND])
        return 0.0  # cross streets and corridor exits have empty downstream

    def act(self, observations: list[Observation]) -> list[int]:
        t = observations[0].t
        if t - self._last_decision_t >= self.decision_interval:
            self._last_decision_t = t
            for i, obs in enumerate(observations):
                if not obs.action_mask.all():
                    self._targets[i] = obs.phase
                    continue
                pressures = np.array(
                    [
                        sum(
                            obs.queue_lengths[a] - self._downstream(observations, i, a)
                            for a in PHASE_APPROACHES[p]
                        )
                        for p in range(N_PHASES)
                    ]
                )
                if pressures.max() > pressures[obs.phase]:
                    self._targets[i] = int(np.argmax(pressures))
                else:
                    self._targets[i] = obs.phase
        return [
            t_ if obs.action_mask[t_] else obs.phase
            for t_, obs in zip(self._targets, observations, strict=True)
        ]
