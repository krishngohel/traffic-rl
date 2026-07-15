import numpy as np

from traffic_rl.config import DemandConfig
from traffic_rl.sim.arrivals import ArrivalStreams

DEMAND = DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(60, 60))


def test_poisson_moments():
    streams = ArrivalStreams(seed=1, demand=DEMAND, dt=1.0)
    n = 20_000
    counts = np.array([streams.vehicle_counts() for _ in range(n)])
    lam = 400 / 3600
    se_mean = np.sqrt(lam / n)
    for a in range(4):
        assert abs(counts[:, a].mean() - lam) < 3 * se_mean
        # Poisson: variance == mean (loose 10% tolerance at this sample size).
        assert abs(counts[:, a].var() - lam) < 0.1 * lam


def test_stream_independence():
    """Ped demand must not perturb vehicle draws (paired-seed requirement)."""
    a = ArrivalStreams(seed=9, demand=DEMAND, dt=1.0)
    hot_peds = DemandConfig(vehicle_rates=DEMAND.vehicle_rates, ped_rates=(999, 999))
    b = ArrivalStreams(seed=9, demand=hot_peds, dt=1.0)
    for _ in range(500):
        av, bv = a.vehicle_counts(), b.vehicle_counts()
        a.ped_counts(), b.ped_counts()
        assert (av == bv).all()
