"""Corridor network: K signalized intersections along an EW arterial.

Eastbound traffic enters at node 0's W approach and traverses every node;
westbound enters at node K-1's E approach. Cross-street (NS) traffic enters and
exits locally at each node. Departing arterial vehicles arrive at the next
intersection after a fixed link travel time — the propagation that makes
coordination (green waves, downstream-aware pressure) matter.

Each node is a full IntersectionSim (same signal safety machine, ped logic, and
per-node event log). NetworkSim adds routing, journey-wait accounting (a
vehicle's wait is the SUM of its queue waits across every node it passes), and
paired-seed reproducibility at the network level.

Model limits, stated: through-traffic only (no turning between arterial and
cross streets), fixed free-flow link time, no spillback between intersections.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from traffic_rl.config import (
    N_APPROACHES,
    DemandConfig,
    SignalTimingConfig,
    SimConfig,
)
from traffic_rl.controllers.base import Observation
from traffic_rl.sim.core import IntersectionSim

EASTBOUND, WESTBOUND = 3, 2  # approach index a vehicle of that direction queues on


@dataclass(frozen=True)
class NetworkDemandConfig:
    arterial_east: float  # veh/h entering the corridor eastbound (node 0, W approach)
    arterial_west: float  # veh/h entering westbound (node K-1, E approach)
    cross: float  # veh/h per NS approach at every node
    peds: float  # peds/h per movement at every node


@dataclass(frozen=True)
class NetworkConfig:
    demand: NetworkDemandConfig | None = None  # uniform corridor demand...
    # ...or per-node demand (e.g. from real count data). Exactly one required.
    # Only the externally-fed rates matter per node (N/S everywhere, W at node
    # 0, E at the last node); internal arterial approaches are fed by links.
    node_demands: tuple[DemandConfig, ...] | None = None
    node_schedules: tuple | None = None  # per-node DemandSchedule, optional
    n_nodes: int = 4
    link_travel: float = 20.0  # seconds between adjacent intersections
    timing: SignalTimingConfig = field(default_factory=SignalTimingConfig)
    dt: float = 1.0
    warmup: float = 1200.0
    measured: float = 3600.0

    @property
    def horizon(self) -> float:
        return self.warmup + self.measured

    @property
    def n_steps(self) -> int:
        return round(self.horizon / self.dt)

    def node_config(self, i: int) -> SimConfig:
        if self.node_demands is not None:
            node_demand = self.node_demands[i]
        elif self.demand is not None:
            d = self.demand
            node_demand = DemandConfig(
                vehicle_rates=(
                    d.cross,
                    d.cross,
                    d.arterial_west if i == self.n_nodes - 1 else 0.0,
                    d.arterial_east if i == 0 else 0.0,
                ),
                ped_rates=(d.peds, d.peds),
            )
        else:
            raise ValueError("NetworkConfig needs either demand or node_demands")
        return SimConfig(
            demand=node_demand,
            timing=self.timing,
            dt=self.dt,
            warmup=self.warmup,
            measured=self.measured,
            demand_schedule=self.node_schedules[i] if self.node_schedules else None,
            external_vehicle_arrivals=(
                True,
                True,
                i == self.n_nodes - 1,  # E approach fed by link unless last node
                i == 0,  # W approach fed by link unless first node
            ),
        )


@dataclass
class NetworkStepResult:
    observations: list[Observation]
    info: dict


class NetworkSim:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.nodes = [IntersectionSim(config.node_config(i)) for i in range(config.n_nodes)]

    def reset(self, seed: int) -> list[Observation]:
        node_seeds = np.random.SeedSequence(seed).generate_state(self.config.n_nodes)
        obs = [node.reset(int(s)) for node, s in zip(self.nodes, node_seeds, strict=True)]
        self.t = 0.0
        k = self.config.n_nodes
        # FIFO mirror of every node approach queue, holding global vehicle ids.
        self._mirror = [[deque() for _ in range(N_APPROACHES)] for _ in range(k)]
        # Injections queued for each node/approach's NEXT step, in order.
        self._pending_gids = [[[] for _ in range(N_APPROACHES)] for _ in range(k)]
        # Vehicles travelling a link: per (node, approach), FIFO of (due_t, gid).
        self._transit = [[deque() for _ in range(N_APPROACHES)] for _ in range(k)]
        # Journey records, indexed by gid.
        self.entry_t: list[float] = []
        self.journey_wait: list[float] = []
        self.exit_t: list[float] = []  # nan while in the network
        self._last_arrival: list[float] = []
        self._queued: list[bool] = []
        self._step_total_queue: list[int] = []
        return obs

    # ------------------------------------------------------------------ helpers

    def _new_journey(self, t: float) -> int:
        gid = len(self.entry_t)
        self.entry_t.append(t)
        self.journey_wait.append(0.0)
        self.exit_t.append(np.nan)
        self._last_arrival.append(t)
        self._queued.append(True)
        return gid

    def _route_departure(self, node_i: int, approach: int, gid: int, t: float) -> None:
        self.journey_wait[gid] += t - self._last_arrival[gid]
        self._queued[gid] = False
        if approach == EASTBOUND and node_i + 1 < self.config.n_nodes:
            self._transit[node_i + 1][EASTBOUND].append((t + self.config.link_travel, gid))
        elif approach == WESTBOUND and node_i - 1 >= 0:
            self._transit[node_i - 1][WESTBOUND].append((t + self.config.link_travel, gid))
        else:
            self.exit_t[gid] = t  # NS vehicles and arterial vehicles leaving the corridor

    # ------------------------------------------------------------------ stepping

    def step(self, actions: list[int]) -> NetworkStepResult:
        cfg = self.config
        t_new = self.t + cfg.dt

        # 1. Deliver link arrivals that come due this step.
        for i, node in enumerate(self.nodes):
            for a in (EASTBOUND, WESTBOUND):
                transit = self._transit[i][a]
                while transit and transit[0][0] <= t_new:
                    _, gid = transit.popleft()
                    node.inject_vehicles(a, 1)
                    self._pending_gids[i][a].append(gid)

        # 2. Step every node, then reconcile the gid mirror from its info.
        observations: list[Observation] = []
        wait_accrued = 0.0
        ped_wait_accrued = 0.0
        total_queue = 0
        node_queues = []
        for i, node in enumerate(self.nodes):
            result = node.step(int(actions[i]))
            observations.append(result.obs)
            info = result.info
            wait_accrued += info["wait_accrued_this_step"]
            ped_wait_accrued += info["ped_wait_accrued_this_step"]
            total_queue += info["total_queue"]
            node_queues.append(info["total_queue"])
            for a in range(N_APPROACHES):
                injected = self._pending_gids[i][a]
                external = int(info["arrivals_by_approach"][a]) - len(injected)
                # External arrivals enter the queue first (see core.step), then
                # the injected vehicles, in injection order.
                for _ in range(external):
                    self._mirror[i][a].append(self._new_journey(t_new))
                for gid in injected:
                    self._last_arrival[gid] = t_new
                    self._queued[gid] = True
                    self._mirror[i][a].append(gid)
                injected.clear()
                for _ in range(int(info["departures_by_approach"][a])):
                    self._route_departure(i, a, self._mirror[i][a].popleft(), t_new)

        self._step_total_queue.append(total_queue)
        self.t = t_new
        info = {
            "t": t_new,
            "total_queue": total_queue,
            "node_queues": node_queues,
            "wait_accrued_this_step": wait_accrued,
            "ped_wait_accrued_this_step": ped_wait_accrued,
        }
        return NetworkStepResult(observations=observations, info=info)

    # ------------------------------------------------------------------ results

    def journey_log(self) -> dict[str, np.ndarray]:
        """Journey-level arrays shaped like a single-node EventLog, so the same
        metric definitions (warm-up, censoring, per-run p95) apply verbatim.
        Censored journey waits are lower bounds: accrued wait so far, plus the
        current queue wait if the vehicle is standing in one."""
        entry = np.asarray(self.entry_t)
        exit_t = np.asarray(self.exit_t)
        wait = np.asarray(self.journey_wait, dtype=np.float64).copy()
        queued = np.asarray(self._queued)
        last = np.asarray(self._last_arrival)
        censored = np.isnan(exit_t)
        wait[censored & queued] += self.t - last[censored & queued]
        # Encode as arrival/depart pairs that reproduce `wait` under
        # metrics.compute_run_metrics (depart - arrival = journey wait).
        depart = np.where(censored, np.nan, entry + wait)
        ped_arrival = np.concatenate([np.asarray(n.log.ped_arrival) for n in self.nodes])
        ped_walk = np.concatenate([np.asarray(n.log.ped_walk_start) for n in self.nodes])
        n_steps = len(self._step_total_queue)
        return {
            "veh_arrival": entry,
            "veh_depart": depart,
            "veh_approach": np.zeros(len(entry), dtype=np.int64),
            "veh_wait_lower_bound": wait,  # exact for completed journeys
            "ped_arrival": ped_arrival,
            "ped_walk_start": ped_walk,
            "ped_movement": np.zeros(len(ped_arrival), dtype=np.int64),
            "step_t": np.arange(1, n_steps + 1, dtype=np.float64) * self.config.dt,
            "step_phase": np.zeros(n_steps, dtype=np.int64),
            "step_state": np.zeros(n_steps, dtype=np.int64),
            "step_queues": np.asarray(self._step_total_queue, dtype=np.int64).reshape(-1, 1),
        }
