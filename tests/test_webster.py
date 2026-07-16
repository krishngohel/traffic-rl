import numpy as np
import pytest

from traffic_rl.config import TWO_PHASE, SignalTimingConfig
from traffic_rl.controllers.webster import C_MAX, C_MIN, webster_plan

TIMING = SignalTimingConfig()
SAT = np.full(8, 1800.0)


def plan_for(flows4, floors):
    """Two-phase plan from per-approach through flows (legacy test shape)."""
    flows = np.concatenate([np.asarray(flows4, dtype=float), np.zeros(4)])
    return webster_plan(flows, SAT, TIMING, TWO_PHASE, green_floors=floors)


def cycle_of(plan) -> float:
    return plan.cycle(TIMING.yellow, TIMING.all_red)


def test_textbook_numbers_no_floor():
    # y = (0.35, 0.175), Y = 0.525, L = 14 => C = 26 / 0.475 ≈ 54.74 s, splits 2:1.
    plan = plan_for([630.0, 630.0, 315.0, 315.0], None)
    assert cycle_of(plan) == pytest.approx(54.74, abs=0.1)
    g_eff = np.array(plan.greens) - TIMING.startup_lost
    assert g_eff[0] / g_eff[1] == pytest.approx(2.0, abs=1e-6)


def test_min_cycle_extension_for_ped_floor():
    # Asymmetric flows: proportional minor split (~6 s eff) is far below the 20 s
    # ped service floor => extend the cycle, never shrink the major phase.
    floor = TIMING.ped_service(0)
    plan = plan_for([550.0, 550.0, 150.0, 150.0], (floor, floor))
    # C = L + floor_eff * Y / y_min = 14 + 18 * 0.38889 / 0.08333 = 98.0
    assert cycle_of(plan) == pytest.approx(98.0, abs=0.5)
    assert plan.greens[1] == pytest.approx(20.0, abs=0.1)  # minor exactly at floor
    assert plan.greens[0] > plan.greens[1] * 3  # major phase not sacrificed


def test_floor_noop_when_cleared():
    flows = [630.0, 630.0, 315.0, 315.0]
    with_floor = plan_for(flows, (15.0, 15.0))
    without = plan_for(flows, None)
    assert with_floor.greens == pytest.approx(without.greens)


def test_saturated_fallback_and_clamps():
    near_sat = [900.0, 900.0, 850.0, 850.0]  # Y ≈ 0.97
    assert cycle_of(plan_for(near_sat, None)) == pytest.approx(C_MAX)
    tiny = [50.0, 50.0, 40.0, 40.0]
    assert cycle_of(plan_for(tiny, None)) == pytest.approx(C_MIN)


def test_extreme_asymmetry_respects_floor_at_cmax():
    floor = TIMING.ped_service(0)
    plan = plan_for([1500.0, 1500.0, 20.0, 20.0], (floor, floor))
    assert cycle_of(plan) == pytest.approx(C_MAX)
    assert plan.greens[1] >= 20.0 - 1e-6


def test_four_phase_critical_movement():
    """Protected-left table: left phases sized from left-bay flows."""
    from traffic_rl.config import IntersectionLayout, LeftTurnTreatment, build_phases

    layout = IntersectionLayout(left_turn=(LeftTurnTreatment.PROTECTED,) * 4)
    phases = build_phases(layout)
    assert len(phases) == 4
    # Heavy NS through, meaningful lefts everywhere.
    flows = np.array([500.0, 480.0, 250.0, 240.0, 120.0, 110.0, 80.0, 70.0])
    plan = webster_plan(flows, SAT, TIMING, phases, green_floors=None)
    ns_left, ns_thru, ew_left, ew_thru = plan.greens
    assert ns_thru > ew_thru > ns_left > ew_left  # ordered by critical flow
    assert min(plan.greens) > 0.0
