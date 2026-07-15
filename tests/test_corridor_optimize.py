import pytest

from traffic_rl.config import SignalTimingConfig
from traffic_rl.controllers.fixed_time import FixedTimePlan
from traffic_rl.controllers.network import (
    CoordinatedPlan,
    ScheduledCoordinatedController,
)
from traffic_rl.data import is_corridor_csv, load_corridor_counts_csv
from traffic_rl.optimize import _corridor_candidates, _corridor_offsets

CORRIDOR_CSV = """time,node,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
07:00,0,180,170,420,520,25,20
07:00,1,220,210,430,510,30,25
08:00,0,210,200,380,780,35,30
08:00,1,260,250,390,770,40,35
"""


@pytest.fixture
def corridor_path(tmp_path):
    p = tmp_path / "corridor.csv"
    p.write_text(CORRIDOR_CSV, encoding="utf-8")
    return p


def test_corridor_csv_detection(corridor_path, tmp_path):
    assert is_corridor_csv(corridor_path)
    single = tmp_path / "single.csv"
    single.write_text(
        "time,north_veh,south_veh,east_veh,west_veh\n07:00,1,2,3,4\n", encoding="utf-8"
    )
    assert not is_corridor_csv(single)


def test_load_corridor_counts(corridor_path):
    schedules, duration, n_nodes = load_corridor_counts_csv(corridor_path)
    assert n_nodes == 2
    assert duration == pytest.approx(7200.0)
    assert len(schedules) == 2 and len(schedules[0]) == 2
    assert schedules[1][0][1].vehicle_rates == pytest.approx((220, 210, 430, 510))
    assert schedules[0][1][1].vehicle_rates == pytest.approx((210, 200, 380, 780))


def test_load_corridor_rejects_mismatched_times(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(CORRIDOR_CSV.replace("08:00,1", "08:30,1"), encoding="utf-8")
    with pytest.raises(ValueError, match="same time intervals"):
        load_corridor_counts_csv(bad)


def test_load_corridor_rejects_gapped_nodes(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(CORRIDOR_CSV.replace(",1,", ",2,"), encoding="utf-8")
    with pytest.raises(ValueError, match="must cover 0..1"):
        load_corridor_counts_csv(bad)


def test_offset_schemes():
    assert _corridor_offsets("east", 3, 20.0, 90.0) == (0.0, 20.0, 40.0)
    assert _corridor_offsets("west", 3, 20.0, 90.0) == (40.0, 20.0, 0.0)
    assert _corridor_offsets("zero", 3, 20.0, 90.0) == (0.0, 0.0, 0.0)
    # Offsets wrap at the cycle.
    assert _corridor_offsets("east", 3, 50.0, 60.0) == (0.0, 50.0, 40.0)


def test_corridor_candidates_share_common_cycle():
    timing = SignalTimingConfig()
    flows = [(180, 170, 420, 520), (260, 250, 390, 770)]
    candidates = _corridor_candidates(flows, timing, 20.0, 2)
    assert len(candidates) == len({c.scheme for c in candidates}) * 2  # scales x schemes
    y, ar = timing.yellow, timing.all_red
    for cand in candidates:
        cycles = {round(p.cycle(y, ar), 6) for p in cand.node_plans}
        assert len(cycles) == 1, "coordination requires one common cycle"
        for plan in cand.node_plans:
            assert min(plan.greens) >= timing.ped_service - 1e-6


def test_scheduled_coordinated_controller_runs_and_switches(corridor_path):
    from traffic_rl.eval.network_harness import run_network_controller
    from traffic_rl.sim.network import NetworkConfig

    schedules, duration, n_nodes = load_corridor_counts_csv(corridor_path)
    plan_a = CoordinatedPlan(
        node_plans=(FixedTimePlan((25.0, 25.0)),) * 2, offsets=(0.0, 20.0), scheme="east"
    )
    plan_b = CoordinatedPlan(
        node_plans=(FixedTimePlan((40.0, 20.0)),) * 2, offsets=(0.0, 0.0), scheme="zero"
    )
    controller = ScheduledCoordinatedController([(0.0, plan_a), (3600.0, plan_b)])
    config = NetworkConfig(
        node_demands=tuple(s[0][1] for s in schedules),
        node_schedules=tuple(schedules),
        n_nodes=n_nodes,
        warmup=600.0,
        measured=duration - 600.0,
    )
    metrics = run_network_controller(controller, config, seed=5)
    assert metrics["n_vehicles"] > 500
    assert 0 < metrics["mean_wait"] <= metrics["p95_wait"]
