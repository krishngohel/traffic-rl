import numpy as np
import pytest

from traffic_rl.rl.dqn import MLP
from traffic_rl.rl.patterns import (
    FAST_TAU,
    N_FEATURES_PATTERN,
    PatternTracker,
    featurize_with_patterns,
)
from traffic_rl.scenarios import make_config
from traffic_rl.sim.core import IntersectionSim


def obs_with_arrivals(counts):
    from traffic_rl.controllers.base import Observation

    arrivals = np.zeros(8)
    arrivals[: len(counts)] = counts
    return Observation(
        t=0.0, queue_lengths=np.zeros(8), oldest_wait=np.zeros(8),
        time_since_arrival=np.zeros(8), arrivals_last_step=arrivals,
        phase_onehot=np.array([1.0, 0.0]), signal_state_onehot=np.array([1.0, 0, 0]),
        phase_elapsed=0.0, ped_call=np.zeros(2), action_mask=np.array([True, True]),
    )


def test_tracker_converges_to_true_rate():
    tracker = PatternTracker(dt=1.0)
    # Steady 0.2 veh/step = 720 veh/h on approach 0, for many time constants.
    for _ in range(3600):
        tracker.update(obs_with_arrivals((0.2, 0, 0, 0)))
    assert tracker.fast[0] == pytest.approx(720.0, rel=0.02)
    assert tracker.slow[0] == pytest.approx(720.0, rel=0.05)
    assert tracker.fast[1] == 0.0


def test_fast_tracker_responds_faster_than_slow():
    tracker = PatternTracker(dt=1.0)
    for _ in range(int(FAST_TAU)):  # one fast time constant after a step change
        tracker.update(obs_with_arrivals((0.5, 0, 0, 0)))
    assert tracker.fast[0] > 3 * tracker.slow[0]


def test_pattern_features_shape_and_range():
    sim = IntersectionSim(make_config("heavy"))
    obs = sim.reset(3)
    tracker = PatternTracker()
    for _ in range(200):
        f = featurize_with_patterns(obs, tracker)
        obs = sim.step(0).obs
    assert f.shape == (N_FEATURES_PATTERN,)
    assert np.isfinite(f).all()
    assert (f[-16:] >= 0).all() and (f[-16:] <= 1.5).all()


def test_pattern_controller_deterministic_and_masked(tmp_path):
    from traffic_rl.rl.pattern_policy import PatternRLController

    net = MLP((N_FEATURES_PATTERN, 96, 96, 4), np.random.default_rng(8))
    path = tmp_path / "w.npz"
    net.save(path)
    a = PatternRLController(weights=path)
    b = PatternRLController(weights=path)
    config = make_config("asymmetric")
    a.reset(config, np.random.default_rng(0))
    b.reset(config, np.random.default_rng(0))
    sim = IntersectionSim(config)
    obs = sim.reset(4)
    for _ in range(300):
        act_a = a.act(obs)
        assert obs.action_mask[act_a]
        assert act_a == b.act(obs)  # same weights + same stream => same tracker state
        obs = sim.step(act_a).obs


def test_pattern_training_smoke(tmp_path):
    from traffic_rl.rl.train import train_pattern_policy

    out = tmp_path / "w.npz"
    train_pattern_policy(steps=6000, seed=1, out=out)
    assert out.exists()
    loaded = MLP.load(out)
    assert loaded.forward(np.zeros(N_FEATURES_PATTERN)).shape == (1, 4)
