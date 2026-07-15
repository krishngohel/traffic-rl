"""Frozen configuration dataclasses shared by the sim, controllers, and harness."""

from __future__ import annotations

from dataclasses import dataclass, field

# Approach indices: 0=N, 1=S, 2=E, 3=W.
# Phase 0 serves N+S vehicles (peds cross the E-W street in parallel).
# Phase 1 serves E+W vehicles (peds cross the N-S street in parallel).
N_APPROACHES = 4
N_PHASES = 2
N_PED_MOVEMENTS = 2
PHASE_APPROACHES: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))
APPROACH_NAMES = ("N", "S", "E", "W")


@dataclass(frozen=True)
class SignalTimingConfig:
    min_green: float = 8.0
    # Anti-starvation backstop, not an operational max: forces a switch only when a
    # conflicting call exists. Kept above the largest green an admissible Webster
    # plan can emit so fixed-time plans are never truncated.
    max_green_backstop: float = 120.0
    yellow: float = 3.0
    all_red: float = 2.0
    startup_lost: float = 2.0
    walk: float = 7.0
    ped_clearance: float = 13.0  # 16 m crossing / 1.2 m/s

    @property
    def ped_service(self) -> float:
        """Minimum green time consumed once a walk starts (walk + clearance)."""
        return self.walk + self.ped_clearance

    @property
    def lost_time_per_phase(self) -> float:
        return self.startup_lost + self.yellow + self.all_red


@dataclass(frozen=True)
class DemandConfig:
    vehicle_rates: tuple[float, float, float, float]  # veh/h per approach (N, S, E, W)
    ped_rates: tuple[float, float]  # peds/h per movement (with-NS, with-EW)


# Time-varying demand: (start_second, DemandConfig) breakpoints, sorted ascending.
# The demand in force at time t is the last breakpoint with start <= t.
DemandSchedule = tuple[tuple[float, DemandConfig], ...]


def demand_at(schedule: DemandSchedule, t: float) -> DemandConfig:
    current = schedule[0][1]
    for start, demand in schedule:
        if start <= t:
            current = demand
        else:
            break
    return current


@dataclass(frozen=True)
class SimConfig:
    demand: DemandConfig
    timing: SignalTimingConfig = field(default_factory=SignalTimingConfig)
    dt: float = 1.0
    sat_flow: float = 1800.0 / 3600.0  # veh/s per lane group
    warmup: float = 1200.0  # Webster: 900 s observe + 300 s settle; identical for all
    measured: float = 3600.0
    # Optional time-varying demand (e.g. loaded from real count data). When set
    # it overrides `demand` for arrival generation; `demand` remains the
    # fallback/summary value.
    demand_schedule: DemandSchedule | None = None
    # Approaches that generate their own Poisson arrivals. A network sim sets
    # internal approaches to False and feeds them by injection instead.
    external_vehicle_arrivals: tuple[bool, bool, bool, bool] = (True, True, True, True)

    @property
    def horizon(self) -> float:
        return self.warmup + self.measured

    @property
    def n_steps(self) -> int:
        return round(self.horizon / self.dt)
