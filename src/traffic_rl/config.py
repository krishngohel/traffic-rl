"""Frozen configuration dataclasses shared by the sim, controllers, and harness.

Movement/lane-group indexing (see docs/DESIGN_PHASE5.md):
  0..3  through+right group per approach (N, S, E, W)
  4..7  left-turn group per approach (N, S, E, W); empty unless the approach
        has a left-turn bay.

Phases live in canonical slots (0 NS-left, 1 NS-through, 2 EW-left,
3 EW-through); runtime indices are compact but each phase records its slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

# Approach indices: 0=N, 1=S, 2=E, 3=W.
N_APPROACHES = 4
N_MOVEMENTS = 8  # 4 through+right groups, then 4 left groups
N_PED_MOVEMENTS = 2  # 0 walks with NS traffic (crosses EW street), 1 with EW
MAX_PHASES = 4
APPROACH_NAMES = ("N", "S", "E", "W")
OPPOSING = (1, 0, 3, 2)  # opposing approach per approach

# Canonical phase slots.
SLOT_NS_LEFT, SLOT_NS_THROUGH, SLOT_EW_LEFT, SLOT_EW_THROUGH = range(4)
SLOT_NAMES = ("NS-left", "NS-through", "EW-left", "EW-through")


def through_group(approach: int) -> int:
    return approach


def left_group(approach: int) -> int:
    return N_APPROACHES + approach


class LeftTurnTreatment(IntEnum):
    SHARED = 0  # no bay: lefts queue in the through group (friction multiplier)
    PERMISSIVE = 1  # bay, filters through opposing gaps during the through phase
    PROTECTED = 2  # bay + protected-only left phase


@dataclass(frozen=True)
class Phase:
    slot: int  # canonical slot id (index into SLOT_NAMES)
    movements: tuple[int, ...]  # lane groups served at saturation flow
    permissive_lefts: tuple[int, ...] = ()  # left groups filtering during this phase
    ped_movement: int | None = None  # ped movement served concurrently
    # Timing overrides; None -> take the scalar default from SignalTimingConfig.
    min_green: float | None = None
    yellow: float | None = None
    all_red: float | None = None

    @property
    def name(self) -> str:
        return SLOT_NAMES[self.slot]


@dataclass(frozen=True)
class GeometryConfig:
    """Site geometry, the inputs the ITE/MUTCD timing formulas need."""

    ns_speed_mph: float = 30.0  # 85th-percentile approach speed, N/S approaches
    ew_speed_mph: float = 30.0
    ns_street_width_m: float = 15.0  # curb-to-curb width of the N-S street
    ew_street_width_m: float = 15.0
    grade: float = 0.0  # approach grade (rise/run), used in the yellow formula


_MPH_TO_MS = 0.44704
_REACTION_S = 1.0
_DECEL_MS2 = 3.05  # 10 ft/s^2
_G_MS2 = 9.81
_VEH_LENGTH_M = 6.0
_PED_SPEED_MS = 1.2  # MUTCD 3.5 ft/s


def ite_yellow(speed_mph: float, grade: float = 0.0) -> float:
    """ITE kinematic yellow-change interval, clamped to [3, 6] s."""
    v = speed_mph * _MPH_TO_MS
    y = _REACTION_S + v / (2.0 * _DECEL_MS2 + 2.0 * _G_MS2 * grade)
    return round(min(max(y, 3.0), 6.0), 1)


def ite_all_red(speed_mph: float, crossing_width_m: float) -> float:
    """ITE red-clearance interval (W + L) / v, clamped to [1, 4] s."""
    v = speed_mph * _MPH_TO_MS
    ar = (crossing_width_m + _VEH_LENGTH_M) / v
    return round(min(max(ar, 1.0), 4.0), 1)


def mutcd_ped_clearance(crossing_width_m: float) -> float:
    """Flashing-don't-walk time at 3.5 ft/s (MUTCD)."""
    return round(crossing_width_m / _PED_SPEED_MS, 1)


@dataclass(frozen=True)
class SignalTimingConfig:
    min_green: float = 8.0
    min_green_left: float = 5.0
    # Anti-starvation backstop, not an operational max: forces a switch only when a
    # conflicting call exists. Kept above the largest green an admissible Webster
    # plan can emit so fixed-time plans are never truncated.
    max_green_backstop: float = 120.0
    # Second starvation guarantee, needed once there are >2 phases: a controller
    # that keeps cycling BETWEEN two phases (each green under the backstop)
    # could otherwise leave a third phase's call waiting forever. Any phase
    # whose call has waited this long is force-served. Kept above the longest
    # admissible cycle so fixed-time plans never trip it.
    max_call_wait: float = 200.0
    yellow: float = 3.0
    all_red: float = 2.0
    startup_lost: float = 2.0
    walk: float = 7.0
    ped_clearance: float = 13.0  # 16 m crossing / 1.2 m/s
    # Per-movement clearance override (crossing the EW street, crossing the NS
    # street); None -> the scalar ped_clearance for both.
    ped_clearances: tuple[float, float] | None = None

    def ped_clearance_for(self, movement: int) -> float:
        if self.ped_clearances is not None:
            return self.ped_clearances[movement]
        return self.ped_clearance

    def ped_service(self, movement: int = 0) -> float:
        """Minimum green time consumed once a walk starts (walk + clearance)."""
        return self.walk + self.ped_clearance_for(movement)

    def min_green_for(self, phase: Phase) -> float:
        if phase.min_green is not None:
            return phase.min_green
        return self.min_green_left if phase.slot in (SLOT_NS_LEFT, SLOT_EW_LEFT) else self.min_green

    def yellow_for(self, phase: Phase) -> float:
        return phase.yellow if phase.yellow is not None else self.yellow

    def all_red_for(self, phase: Phase) -> float:
        return phase.all_red if phase.all_red is not None else self.all_red

    @property
    def lost_time_per_phase(self) -> float:
        return self.startup_lost + self.yellow + self.all_red


def timing_from_geometry(geometry: GeometryConfig, **overrides) -> SignalTimingConfig:
    """Build a timing config whose clearance intervals come from the ITE/MUTCD
    formulas. Per-phase yellow/all-red land on the phases via build_phases."""
    return SignalTimingConfig(
        yellow=max(ite_yellow(geometry.ns_speed_mph, geometry.grade),
                   ite_yellow(geometry.ew_speed_mph, geometry.grade)),
        all_red=max(ite_all_red(geometry.ns_speed_mph, geometry.ew_street_width_m),
                    ite_all_red(geometry.ew_speed_mph, geometry.ns_street_width_m)),
        ped_clearances=(
            mutcd_ped_clearance(geometry.ew_street_width_m),  # movement 0 crosses EW street
            mutcd_ped_clearance(geometry.ns_street_width_m),
        ),
        **overrides,
    )


@dataclass(frozen=True)
class IntersectionLayout:
    """Lane configuration: left-turn treatment and lane counts per approach."""

    left_turn: tuple[LeftTurnTreatment, LeftTurnTreatment, LeftTurnTreatment, LeftTurnTreatment] = (
        LeftTurnTreatment.SHARED,
    ) * 4
    through_lanes: tuple[int, int, int, int] = (1, 1, 1, 1)
    # Left bays are always modeled as a single lane.

    def has_bay(self, approach: int) -> bool:
        return self.left_turn[approach] != LeftTurnTreatment.SHARED

    def protected_lefts(self, street: str) -> tuple[int, ...]:
        approaches = (0, 1) if street == "NS" else (2, 3)
        return tuple(
            left_group(a)
            for a in approaches
            if self.left_turn[a] == LeftTurnTreatment.PROTECTED
        )

    def permissive_bay_lefts(self, street: str) -> tuple[int, ...]:
        approaches = (0, 1) if street == "NS" else (2, 3)
        return tuple(
            left_group(a)
            for a in approaches
            if self.left_turn[a] == LeftTurnTreatment.PERMISSIVE
        )


def build_phases(
    layout: IntersectionLayout,
    geometry: GeometryConfig | None = None,
) -> tuple[Phase, ...]:
    """Canonical-order phase table for a layout (leading dual lefts)."""
    phases: list[Phase] = []
    for street, slot_left, slot_through, approaches, ped in (
        ("NS", SLOT_NS_LEFT, SLOT_NS_THROUGH, (0, 1), 0),
        ("EW", SLOT_EW_LEFT, SLOT_EW_THROUGH, (2, 3), 1),
    ):
        speed = None
        crossed_width = None
        if geometry is not None:
            speed = geometry.ns_speed_mph if street == "NS" else geometry.ew_speed_mph
            crossed_width = (
                geometry.ew_street_width_m if street == "NS" else geometry.ns_street_width_m
            )
        yellow = ite_yellow(speed, geometry.grade) if speed is not None else None
        all_red = ite_all_red(speed, crossed_width) if speed is not None else None
        protected = layout.protected_lefts(street)
        if protected:
            phases.append(
                Phase(slot=slot_left, movements=protected, yellow=yellow, all_red=all_red)
            )
        phases.append(
            Phase(
                slot=slot_through,
                movements=tuple(through_group(a) for a in approaches),
                permissive_lefts=layout.permissive_bay_lefts(street),
                ped_movement=ped,
                yellow=yellow,
                all_red=all_red,
            )
        )
    return tuple(phases)


# Legacy 2-phase table: through-only, no bays — the Phase 1-4 model exactly.
TWO_PHASE_LAYOUT = IntersectionLayout()
TWO_PHASE = build_phases(TWO_PHASE_LAYOUT)

# Turn fractions: (left, through, right) per approach; rights fold into the
# through+right group; RTOR is not modeled (stated limit).
TurnFractions = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
ALL_THROUGH: TurnFractions = ((0.0, 1.0, 0.0),) * 4


@dataclass(frozen=True)
class DemandConfig:
    vehicle_rates: tuple[float, float, float, float]  # veh/h per approach (N, S, E, W)
    ped_rates: tuple[float, float]  # peds/h per movement (with-NS, with-EW)
    turn_fractions: TurnFractions = ALL_THROUGH

    def movement_rates(self, layout: IntersectionLayout) -> tuple[float, ...]:
        """veh/h per lane group (8,) under a layout: bays pull the left share
        out of the through group; SHARED keeps everything in the through group."""
        rates = [0.0] * N_MOVEMENTS
        for a in range(N_APPROACHES):
            total = self.vehicle_rates[a]
            l_frac = self.turn_fractions[a][0]
            if layout.has_bay(a):
                rates[left_group(a)] = total * l_frac
                rates[through_group(a)] = total * (1.0 - l_frac)
            else:
                rates[through_group(a)] = total
        return tuple(rates)

    def shared_left_fraction(self, approach: int, layout: IntersectionLayout) -> float:
        """Left share riding in the through group (nonzero only for SHARED)."""
        if layout.has_bay(approach):
            return 0.0
        return self.turn_fractions[approach][0]


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


# Permissive-left gap acceptance (docs/DESIGN_PHASE5.md).
CRITICAL_GAP_S = 4.5
FOLLOW_UP_S = 2.5
SNEAKERS_PER_PHASE = 2


def permissive_capacity(opposing_veh_s: float) -> float:
    """Gap-acceptance capacity (veh/s) of a permissive left against an
    opposing through flow, classic exponential-gap formula."""
    q = max(opposing_veh_s, 0.0)
    if q < 1e-9:
        return 1.0 / FOLLOW_UP_S
    return q * math.exp(-q * CRITICAL_GAP_S) / (1.0 - math.exp(-q * FOLLOW_UP_S))


@dataclass(frozen=True)
class SimConfig:
    demand: DemandConfig
    timing: SignalTimingConfig = field(default_factory=SignalTimingConfig)
    layout: IntersectionLayout = TWO_PHASE_LAYOUT
    geometry: GeometryConfig | None = None
    dt: float = 1.0
    sat_flow: float = 1800.0 / 3600.0  # veh/s per LANE
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
    def phases(self) -> tuple[Phase, ...]:
        return build_phases(self.layout, self.geometry)

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def group_lanes(self, group: int) -> int:
        return self.layout.through_lanes[group] if group < N_APPROACHES else 1

    def group_sat_flow(self, group: int) -> float:
        return self.sat_flow * self.group_lanes(group)

    @property
    def horizon(self) -> float:
        return self.warmup + self.measured

    @property
    def n_steps(self) -> int:
        return round(self.horizon / self.dt)
