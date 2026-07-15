import numpy as np
import pytest

from traffic_rl.config import DemandConfig, SimConfig, demand_at
from traffic_rl.controllers.fixed_time import (
    FixedTimePlan,
    ScheduledFixedTimeController,
)
from traffic_rl.data import load_counts_csv
from traffic_rl.optimize import _average_demand, _candidate_plans, optimize_interval
from traffic_rl.sim.arrivals import ArrivalStreams

CSV = """time,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
07:00,520,480,180,160,30,25
07:30,610,590,210,190,40,35
08:30,430,410,170,150,25,20
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


def test_load_counts_infers_intervals_and_rates(csv_path):
    schedule, duration = load_counts_csv(csv_path)
    assert duration == pytest.approx(1800.0 + 3600.0 + 3600.0)  # 30m + 60m + trailing 60m
    starts = [s for s, _ in schedule]
    assert starts == [0.0, 1800.0, 5400.0]
    # First interval is 30 min: counts double to hourly rates.
    assert schedule[0][1].vehicle_rates == pytest.approx((1040, 960, 360, 320))
    # Second interval is 60 min: counts are already rates.
    assert schedule[1][1].vehicle_rates == pytest.approx((610, 590, 210, 190))
    # Lookup semantics: last breakpoint <= t.
    assert demand_at(schedule, 0.0) is schedule[0][1]
    assert demand_at(schedule, 1800.0) is schedule[1][1]
    assert demand_at(schedule, 9999.0) is schedule[2][1]


def test_load_counts_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("time,north_veh\n07:00,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_counts_csv(bad)
    unordered = tmp_path / "unordered.csv"
    unordered.write_text(CSV.replace("08:30", "07:10"), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_counts_csv(unordered)


def test_scheduled_arrivals_follow_the_schedule():
    lo = DemandConfig(vehicle_rates=(0, 0, 0, 0), ped_rates=(0, 0))
    hi = DemandConfig(vehicle_rates=(3600, 3600, 3600, 3600), ped_rates=(0, 0))
    streams = ArrivalStreams(1, lo, dt=1.0, schedule=((0.0, lo), (100.0, hi)))
    early = sum(int(streams.vehicle_counts(t).sum()) for t in range(100))
    late = sum(int(streams.vehicle_counts(t).sum()) for t in range(100, 200))
    assert early == 0
    assert late > 300  # ~4 veh/s expected


def test_scheduled_fixed_time_switches_plans():
    controller = ScheduledFixedTimeController(
        [(0.0, FixedTimePlan(greens=(25.0, 25.0))), (3600.0, FixedTimePlan(greens=(60.0, 20.0)))]
    )
    config = SimConfig(demand=DemandConfig(vehicle_rates=(0, 0, 0, 0), ped_rates=(0, 0)))
    controller.reset(config, np.random.default_rng(0))
    assert controller._controllers[0][1].plan.greens == (25.0, 25.0)
    # After the switch time the second plan's longer NS green governs: at
    # t=3600+40 the 50/50 plan would be in EW, the 60/20 plan still in NS.
    from tests.conftest import drive  # noqa: F401  (import check only)

    from traffic_rl.controllers.base import Observation

    def obs_at(t):
        return Observation(
            t=t, queue_lengths=np.zeros(4), oldest_wait=np.zeros(4),
            time_since_arrival=np.zeros(4), arrivals_last_step=np.zeros(4),
            phase_onehot=np.array([1.0, 0.0]), signal_state_onehot=np.array([1.0, 0, 0]),
            phase_elapsed=10.0, ped_call=np.zeros(2), action_mask=np.array([True, True]),
        )

    assert controller.act(obs_at(40.0)) == 1  # 50/50 plan: past 25+5 => EW
    assert controller.act(obs_at(3640.0)) == 0  # 60/20 plan: still NS at cycle t=40


def test_candidate_plans_respect_floors():
    demand = DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40))
    from traffic_rl.config import SignalTimingConfig

    timing = SignalTimingConfig()
    for plan in _candidate_plans(demand, timing):
        assert min(plan.greens) >= timing.ped_service - 1e-6
        assert 40.0 - 1 <= plan.cycle(timing.yellow, timing.all_red) <= 150.0 + 25.0


def test_optimize_interval_beats_naive_on_asymmetric():
    from traffic_rl.config import SignalTimingConfig
    from traffic_rl.controllers.fixed_time import NAIVE_PLAN, FixedTimeController
    from traffic_rl.eval.harness import run_controller, run_seeds

    demand = DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40))
    timing = SignalTimingConfig()
    seeds = run_seeds(7, 3)
    plan, score = optimize_interval(demand, timing, seeds)
    config = SimConfig(demand=demand, timing=timing, warmup=600.0, measured=1800.0)
    naive = np.mean(
        [run_controller(FixedTimeController(NAIVE_PLAN), config, s)["p95_wait"] for s in seeds]
    )
    assert score < naive


def test_average_demand_time_weighted():
    a = DemandConfig(vehicle_rates=(100, 100, 100, 100), ped_rates=(0, 0))
    b = DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(0, 0))
    avg = _average_demand(((0.0, a), (1800.0, b)), 3600.0)
    assert avg.vehicle_rates == pytest.approx((250, 250, 250, 250))
