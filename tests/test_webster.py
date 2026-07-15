import numpy as np
import pytest

from traffic_rl.config import SignalTimingConfig
from traffic_rl.controllers.webster import C_MAX, C_MIN, webster_plan

TIMING = SignalTimingConfig()
SAT = 1800.0


def cycle_of(plan) -> float:
    return plan.cycle(TIMING.yellow, TIMING.all_red)


def test_textbook_numbers_no_floor():
    # y = (0.35, 0.175), Y = 0.525, L = 14 => C = 26 / 0.475 ≈ 54.74 s, splits 2:1.
    flows = np.array([630.0, 630.0, 315.0, 315.0])
    plan = webster_plan(flows, SAT, TIMING, green_floor=None)
    assert cycle_of(plan) == pytest.approx(54.74, abs=0.1)
    g_eff = np.array(plan.greens) - TIMING.startup_lost
    assert g_eff[0] / g_eff[1] == pytest.approx(2.0, abs=1e-6)


def test_min_cycle_extension_for_ped_floor():
    # Asymmetric flows: proportional minor split (~6 s eff) is far below the 20 s
    # ped service floor => extend the cycle, never shrink the major phase.
    flows = np.array([550.0, 550.0, 150.0, 150.0])
    plan = webster_plan(flows, SAT, TIMING, green_floor=TIMING.ped_service)
    # C = L + floor_eff * Y / y_min = 14 + 18 * 0.38889 / 0.08333 = 98.0
    assert cycle_of(plan) == pytest.approx(98.0, abs=0.5)
    assert plan.greens[1] == pytest.approx(20.0, abs=0.1)  # minor exactly at floor
    assert plan.greens[0] > plan.greens[1] * 3  # major phase not sacrificed


def test_floor_noop_when_cleared():
    flows = np.array([630.0, 630.0, 315.0, 315.0])
    with_floor = webster_plan(flows, SAT, TIMING, green_floor=15.0)
    without = webster_plan(flows, SAT, TIMING, green_floor=None)
    assert with_floor.greens == pytest.approx(without.greens)


def test_saturated_fallback_and_clamps():
    near_sat = np.array([900.0, 900.0, 850.0, 850.0])  # Y ≈ 0.97
    assert cycle_of(webster_plan(near_sat, SAT, TIMING, green_floor=None)) == pytest.approx(C_MAX)
    tiny = np.array([50.0, 50.0, 40.0, 40.0])
    assert cycle_of(webster_plan(tiny, SAT, TIMING, green_floor=None)) == pytest.approx(C_MIN)


def test_extreme_asymmetry_respects_floor_at_cmax():
    flows = np.array([1500.0, 1500.0, 20.0, 20.0])
    plan = webster_plan(flows, SAT, TIMING, green_floor=TIMING.ped_service)
    assert cycle_of(plan) == pytest.approx(C_MAX)
    assert plan.greens[1] >= 20.0 - 1e-6
