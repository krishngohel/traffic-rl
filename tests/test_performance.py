import time

import pytest

from traffic_rl.eval.harness import run_single
from traffic_rl.scenarios import make_config


@pytest.mark.slow
def test_800x_real_time():
    """One 4800 s run must take < 6 s wall (≈ one sim-hour in < 4.5 s)."""
    config = make_config("heavy")  # worst case: biggest queues
    run_single("naive", config, seed=0)  # warm the caches
    t0 = time.perf_counter()
    run_single("naive", config, seed=1)
    wall = time.perf_counter() - t0
    assert wall < 6.0, f"4800 sim-seconds took {wall:.2f} s wall (needs 800x)"
