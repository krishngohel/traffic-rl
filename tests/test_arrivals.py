import numpy as np

from traffic_rl.config import (
    TWO_PHASE_LAYOUT,
    DemandConfig,
    IntersectionLayout,
    LeftTurnTreatment,
)
from traffic_rl.sim.arrivals import ArrivalStreams

DEMAND = DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(60, 60))


def test_poisson_moments():
    streams = ArrivalStreams(seed=1, demand=DEMAND, dt=1.0, layout=TWO_PHASE_LAYOUT)
    n = 20_000
    counts = np.array([streams.vehicle_counts() for _ in range(n)])
    lam = 400 / 3600
    se_mean = np.sqrt(lam / n)
    for a in range(4):
        assert abs(counts[:, a].mean() - lam) < 3 * se_mean
        # Poisson: variance == mean (loose 10% tolerance at this sample size).
        assert abs(counts[:, a].var() - lam) < 0.1 * lam
    # Through-only demand: left-bay groups draw nothing.
    assert counts[:, 4:].sum() == 0


def test_stream_independence():
    """Ped demand must not perturb vehicle draws (paired-seed requirement)."""
    a = ArrivalStreams(seed=9, demand=DEMAND, dt=1.0, layout=TWO_PHASE_LAYOUT)
    hot_peds = DemandConfig(vehicle_rates=DEMAND.vehicle_rates, ped_rates=(999, 999))
    b = ArrivalStreams(seed=9, demand=hot_peds, dt=1.0, layout=TWO_PHASE_LAYOUT)
    for _ in range(500):
        av, bv = a.vehicle_counts(), b.vehicle_counts()
        a.ped_counts(), b.ped_counts()
        assert (av == bv).all()


def test_bay_split_conserves_demand():
    """With a left bay, left + through rates equal the approach total."""
    bays = IntersectionLayout(left_turn=(LeftTurnTreatment.PROTECTED,) * 4)
    demand = DemandConfig(
        vehicle_rates=(600, 600, 600, 600),
        ped_rates=(0, 0),
        turn_fractions=((0.25, 0.65, 0.10),) * 4,
    )
    rates = demand.movement_rates(bays)
    for a in range(4):
        assert abs(rates[a] + rates[4 + a] - 600.0) < 1e-9
        assert abs(rates[4 + a] - 150.0) < 1e-9
    # SHARED keeps everything in the through group.
    shared_rates = demand.movement_rates(IntersectionLayout())
    assert shared_rates[:4] == (600.0,) * 4
    assert shared_rates[4:] == (0.0,) * 4
