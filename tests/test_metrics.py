import numpy as np
import pytest

from traffic_rl.config import DemandConfig, SimConfig
from traffic_rl.eval.metrics import compute_run_metrics, mean_ci, paired_diff_ci

CONFIG = SimConfig(demand=DemandConfig(vehicle_rates=(0, 0, 0, 0), ped_rates=(0, 0)))
LO, HI = CONFIG.warmup, CONFIG.horizon  # 1200, 4800


def make_log(arrivals, waits, censored_flags=None, approaches=None):
    arrivals = np.asarray(arrivals, dtype=float)
    waits = np.asarray(waits, dtype=float)
    if censored_flags is None:
        censored = np.zeros(len(arrivals), bool)
    else:
        censored = np.asarray(censored_flags)
    departs = np.where(censored, np.nan, arrivals + waits)
    n_steps = int(HI)
    return {
        "veh_arrival": arrivals,
        "veh_depart": departs,
        "veh_approach": np.zeros(len(arrivals), dtype=np.int64)
        if approaches is None
        else np.asarray(approaches),
        "ped_arrival": np.array([1300.0, 2000.0]),
        "ped_walk_start": np.array([1330.0, np.nan]),
        "ped_movement": np.array([0, 1], dtype=np.int64),
        "step_t": np.arange(1, n_steps + 1, dtype=float),
        "step_phase": np.zeros(n_steps, dtype=np.int64),
        "step_state": np.zeros(n_steps, dtype=np.int64),
        "step_queues": np.zeros((n_steps, 4), dtype=np.int64),
    }


def test_exact_percentiles_and_warmup_cut():
    # 20 in-window vehicles with waits 1..20, plus one pre-warm-up vehicle with a
    # huge wait that must be excluded.
    arrivals = [500.0] + [2000.0] * 20
    waits = [9999.0] + list(range(1, 21))
    m = compute_run_metrics(make_log(arrivals, waits), CONFIG)
    assert m["n_vehicles"] == 20
    assert m["p95_wait"] == pytest.approx(np.percentile(np.arange(1, 21), 95))
    assert m["mean_wait"] == pytest.approx(10.5)
    assert m["max_wait"] == 20.0
    assert m["n_censored"] == 0 and not m["p95_is_lower_bound"]


def test_censored_lower_bound_flag():
    # 10 vehicles, 2 censored (20%) => censored wait = horizon - arrival, flagged.
    arrivals = [4000.0] * 10
    waits = [10.0] * 8 + [0.0, 0.0]
    censored = [False] * 8 + [True, True]
    m = compute_run_metrics(make_log(arrivals, waits, censored), CONFIG)
    assert m["n_censored"] == 2
    assert m["censored_fraction"] == pytest.approx(0.2)
    assert m["p95_is_lower_bound"] and m["unstable"]
    assert m["max_wait"] == pytest.approx(HI - 4000.0)  # lower-bound wait included


def test_ped_waits_and_censoring():
    m = compute_run_metrics(make_log([2000.0], [5.0]), CONFIG)
    assert m["n_peds"] == 2
    # One served after 30 s, one censored at horizon (4800 - 2000).
    assert m["ped_mean_wait"] == pytest.approx((30.0 + 2800.0) / 2)


def test_mean_ci_and_paired_ci():
    v = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
    ci = mean_ci(v)
    assert ci["mean"] == pytest.approx(10.0)
    # t(4, 0.975) = 2.776, s = sqrt(2.5), n = 5.
    half = 2.776 * np.sqrt(2.5) / np.sqrt(5)
    assert ci["hi"] - ci["mean"] == pytest.approx(half, rel=1e-3)

    base = np.array([100.0, 110.0, 90.0])
    other = np.array([30.0, 35.0, 25.0])
    d = paired_diff_ci(base, other)
    assert d["mean"] == pytest.approx(70.0)
    assert d["lo"] > 0, "clearly better controller must have a positive paired CI"
