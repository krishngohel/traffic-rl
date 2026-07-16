"""Named demand scenarios. Rates are veh/h PER APPROACH and peds/h PER MOVEMENT.

The three legacy scenarios (symmetric / asymmetric / heavy) are through-only on
the 2-phase layout — the Phase 1-4 benchmark, kept reproducible. The turning
scenarios exercise the Phase 5 model: left-turn bays, protected/permissive
phasing, and ITE clearance intervals from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from traffic_rl.config import (
    DemandConfig,
    GeometryConfig,
    IntersectionLayout,
    LeftTurnTreatment,
    SimConfig,
    timing_from_geometry,
)

_S = LeftTurnTreatment.SHARED
_P = LeftTurnTreatment.PERMISSIVE
_X = LeftTurnTreatment.PROTECTED


@dataclass(frozen=True)
class Scenario:
    demand: DemandConfig
    layout: IntersectionLayout = IntersectionLayout()
    geometry: GeometryConfig | None = None


SCENARIOS: dict[str, Scenario] = {
    # ------------------------- legacy through-only benchmark (Phases 1-4)
    # Balanced, comfortably stable under any sane controller.
    "symmetric": Scenario(
        DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(60, 60))
    ),
    # NS heavy / EW light: tuned so the naive 50/50 split runs NS near capacity
    # (v/c ≈ 0.9) — heavily loaded and unstable run to run — while demand-aware
    # controllers keep NS far below their capacity.
    "asymmetric": Scenario(
        DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40))
    ),
    # Near-saturation stress test on all approaches.
    "heavy": Scenario(
        DemandConfig(vehicle_rates=(550, 550, 550, 550), ped_rates=(80, 80))
    ),
    # ------------------------- Phase 5 turning scenarios
    # Suburban arterial x collector: fast NS arterial with heavy protected
    # lefts, permissive bays on the side street. The classic quad-left site.
    "arterial_lefts": Scenario(
        DemandConfig(
            vehicle_rates=(560, 540, 260, 250),
            ped_rates=(25, 25),
            turn_fractions=(
                (0.22, 0.68, 0.10),
                (0.20, 0.70, 0.10),
                (0.15, 0.70, 0.15),
                (0.15, 0.70, 0.15),
            ),
        ),
        layout=IntersectionLayout(left_turn=(_X, _X, _P, _P), through_lanes=(2, 2, 1, 1)),
        geometry=GeometryConfig(
            ns_speed_mph=45.0, ew_speed_mph=30.0,
            ns_street_width_m=22.0, ew_street_width_m=14.0,
        ),
    ),
    # Downtown grid corner: slow, narrow, no bays (shared permissive lefts),
    # heavy pedestrians — where left friction and walk service dominate.
    "downtown_shared": Scenario(
        DemandConfig(
            vehicle_rates=(340, 330, 310, 300),
            ped_rates=(140, 140),
            turn_fractions=(
                (0.12, 0.76, 0.12),
                (0.12, 0.76, 0.12),
                (0.10, 0.78, 0.12),
                (0.10, 0.78, 0.12),
            ),
        ),
        layout=IntersectionLayout(left_turn=(_S, _S, _S, _S)),
        geometry=GeometryConfig(
            ns_speed_mph=25.0, ew_speed_mph=25.0,
            ns_street_width_m=12.0, ew_street_width_m=12.0,
        ),
    ),
}


def make_config(scenario: str, **overrides) -> SimConfig:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    s = SCENARIOS[scenario]
    kwargs: dict = {"demand": s.demand, "layout": s.layout, "geometry": s.geometry}
    if s.geometry is not None:
        kwargs["timing"] = timing_from_geometry(s.geometry)
    kwargs.update(overrides)
    return SimConfig(**kwargs)
