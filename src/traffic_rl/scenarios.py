"""Named demand scenarios. Rates are veh/h PER APPROACH and peds/h PER MOVEMENT."""

from __future__ import annotations

from traffic_rl.config import DemandConfig, SimConfig

SCENARIOS: dict[str, DemandConfig] = {
    # Balanced, comfortably stable under any sane controller.
    "symmetric": DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(60, 60)),
    # NS heavy / EW light: tuned so the naive 50/50 split runs NS near capacity
    # (v/c ≈ 0.9) — heavily loaded and unstable run to run — while demand-aware
    # controllers keep NS far below their capacity.
    "asymmetric": DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40)),
    # Near-saturation stress test on all approaches.
    "heavy": DemandConfig(vehicle_rates=(550, 550, 550, 550), ped_rates=(80, 80)),
}


def make_config(scenario: str, **overrides) -> SimConfig:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    return SimConfig(demand=SCENARIOS[scenario], **overrides)
