import numpy as np
import pytest

from traffic_rl.config import (
    TWO_PHASE_LAYOUT,
    DemandConfig,
    LeftTurnTreatment,
    SimConfig,
    demand_at,
)
from traffic_rl.controllers.fixed_time import (
    FixedTimePlan,
    ScheduledFixedTimeController,
)
from traffic_rl.data import (
    DEFAULT_SITE,
    SiteConfig,
    left_turn_warrant,
    load_counts_csv,
)
from traffic_rl.optimize import (
    _average_demand,
    _candidate_plans,
    build_site_layout,
    optimize_interval,
)
from traffic_rl.sim.arrivals import ArrivalStreams

CSV = """time,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
07:00,520,480,180,160,30,25
07:30,610,590,210,190,40,35
08:30,430,410,170,150,25,20
"""

_TMC_HEADER = "time," + ",".join(
    f"{p}_{m}"
    for p in ("north", "south", "east", "west")
    for m in ("left", "thru", "right")
) + ",ped_ns,ped_ew"
TMC_CSV = _TMC_HEADER + """
07:00,120,380,40,110,370,45,30,140,25,28,135,22,30,25
08:00,150,420,50,140,410,55,35,160,30,32,150,28,40,35
09:00,80,300,35,75,290,40,25,120,20,22,115,18,25,20
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


@pytest.fixture
def tmc_path(tmp_path):
    p = tmp_path / "tmc.csv"
    p.write_text(TMC_CSV, encoding="utf-8")
    return p


def test_load_counts_infers_intervals_and_rates(csv_path):
    schedule, duration, has_tmc = load_counts_csv(csv_path)
    assert not has_tmc
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


def test_load_tmc_counts(tmc_path):
    schedule, duration, has_tmc = load_counts_csv(tmc_path)
    assert has_tmc
    d = schedule[0][1]
    # Totals per approach = L + T + R.
    assert d.vehicle_rates[0] == pytest.approx(120 + 380 + 40)
    # Turn fractions recovered.
    assert d.turn_fractions[0][0] == pytest.approx(120 / 540)
    assert d.turn_fractions[2][2] == pytest.approx(25 / 195)


def test_load_counts_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("time,north_veh\n07:00,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="need either"):
        load_counts_csv(bad)
    unordered = tmp_path / "unordered.csv"
    unordered.write_text(CSV.replace("08:30", "07:10"), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_counts_csv(unordered)


def test_left_turn_warrant_and_layout(tmc_path):
    schedule, _, has_tmc = load_counts_csv(tmc_path)
    warranted, notes = left_turn_warrant(schedule, DEFAULT_SITE)
    # NS lefts: 150/h volume at 08:00 with ~465/h opposing thru+right
    # -> cross product ~70k >= 50k. EW lefts (~35/h x ~180) stay unwarranted.
    assert warranted[0] and warranted[1]
    assert not warranted[2] and not warranted[3]
    assert notes
    layout, _ = build_site_layout(schedule, DEFAULT_SITE, has_tmc)
    assert layout.left_turn[0] == LeftTurnTreatment.PROTECTED
    assert layout.left_turn[2] == LeftTurnTreatment.PERMISSIVE
    # No bays -> shared, whatever the warrant says.
    no_bays = SiteConfig(geometry=DEFAULT_SITE.geometry, left_bays=(False,) * 4)
    layout2, _ = build_site_layout(schedule, no_bays, has_tmc)
    assert all(t == LeftTurnTreatment.SHARED for t in layout2.left_turn)


def test_scheduled_arrivals_follow_the_schedule():
    lo = DemandConfig(vehicle_rates=(0, 0, 0, 0), ped_rates=(0, 0))
    hi = DemandConfig(vehicle_rates=(3600, 3600, 3600, 3600), ped_rates=(0, 0))
    streams = ArrivalStreams(
        1, lo, dt=1.0, layout=TWO_PHASE_LAYOUT, schedule=((0.0, lo), (100.0, hi))
    )
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

    from traffic_rl.controllers.base import Observation

    def obs_at(t):
        return Observation(
            t=t, queue_lengths=np.zeros(8), oldest_wait=np.zeros(8),
            time_since_arrival=np.zeros(8), arrivals_last_step=np.zeros(8),
            phase_onehot=np.array([1.0, 0.0]), signal_state_onehot=np.array([1.0, 0, 0]),
            phase_elapsed=10.0, ped_call=np.zeros(2), action_mask=np.array([True, True]),
        )

    assert controller.act(obs_at(40.0)) == 1  # 50/50 plan: past 25+5 => EW
    assert controller.act(obs_at(3640.0)) == 0  # 60/20 plan: still NS at cycle t=40


def test_candidate_plans_respect_floors():
    demand = DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40))
    config = SimConfig(demand=demand)
    timing = config.timing
    plans = _candidate_plans(demand, config)
    assert plans
    for plan in plans:
        assert min(plan.greens) >= timing.ped_service(0) - 1e-6
        assert 40.0 - 1 <= plan.cycle_for(config) <= 180.0 + 1


def test_candidate_plans_four_phase():
    from traffic_rl.config import IntersectionLayout

    demand = DemandConfig(
        vehicle_rates=(600, 580, 250, 240),
        ped_rates=(30, 30),
        turn_fractions=((0.25, 0.65, 0.10),) * 4,
    )
    layout = IntersectionLayout(left_turn=(LeftTurnTreatment.PROTECTED,) * 4)
    config = SimConfig(demand=demand, layout=layout)
    plans = _candidate_plans(demand, config)
    assert plans
    for plan in plans:
        assert len(plan.greens) == 4
        # Left phases at least their min green; through phases at the ped floor.
        assert plan.greens[0] >= config.timing.min_green_left - 1e-6
        assert plan.greens[1] >= config.timing.ped_service(0) - 1e-6


def test_optimize_interval_beats_naive_on_asymmetric():
    from traffic_rl.controllers.fixed_time import NAIVE_PLAN, FixedTimeController
    from traffic_rl.eval.harness import run_controller, run_seeds

    demand = DemandConfig(vehicle_rates=(620, 620, 150, 150), ped_rates=(40, 40))
    base_config = SimConfig(demand=demand, warmup=600.0, measured=1800.0)
    seeds = run_seeds(7, 3)
    plan, score = optimize_interval(demand, base_config, seeds)
    naive = np.mean(
        [
            run_controller(FixedTimeController(NAIVE_PLAN), base_config, s)["p95_wait"]
            for s in seeds
        ]
    )
    assert score < naive


def test_average_demand_time_weighted():
    a = DemandConfig(vehicle_rates=(100, 100, 100, 100), ped_rates=(0, 0))
    b = DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(0, 0))
    avg = _average_demand(((0.0, a), (1800.0, b)), 3600.0)
    assert avg.vehicle_rates == pytest.approx((250, 250, 250, 250))
