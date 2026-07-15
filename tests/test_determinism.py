import numpy as np

from traffic_rl.eval.harness import run_single
from traffic_rl.scenarios import make_config


def test_same_seed_bit_identical():
    config = make_config("asymmetric")
    a = run_single("naive", config, seed=1234)
    b = run_single("naive", config, seed=1234)
    assert a["p95_wait"] == b["p95_wait"]
    assert np.array_equal(a["veh_waits"], b["veh_waits"])
    assert np.array_equal(a["queue_timeseries"], b["queue_timeseries"])


def test_different_seeds_differ():
    config = make_config("asymmetric")
    a = run_single("naive", config, seed=1)
    b = run_single("naive", config, seed=2)
    assert not np.array_equal(a["veh_waits"], b["veh_waits"])
