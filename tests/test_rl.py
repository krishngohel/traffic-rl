import numpy as np
import pytest

from traffic_rl.rl.dqn import DQN, MLP, Replay
from traffic_rl.rl.features import N_FEATURES, featurize
from traffic_rl.scenarios import make_config
from traffic_rl.sim.core import IntersectionSim


def test_featurize_shape_and_determinism():
    sim = IntersectionSim(make_config("asymmetric"))
    obs = sim.reset(7)
    for _ in range(50):
        obs = sim.step(0).obs
    f1, f2 = featurize(obs), featurize(obs)
    assert f1.shape == (N_FEATURES,) and f1.dtype == np.float32
    assert np.array_equal(f1, f2)
    assert np.isfinite(f1).all()


def test_mlp_save_load_roundtrip(tmp_path):
    net = MLP((N_FEATURES, 64, 64, 2), np.random.default_rng(3))
    x = np.random.default_rng(4).normal(size=(5, N_FEATURES))
    path = tmp_path / "w.npz"
    net.save(path)
    loaded = MLP.load(path)
    assert np.allclose(net.forward(x), loaded.forward(x))


def test_dqn_learns_a_bandit():
    """Action 1 always pays 1, action 0 pays 0 — Q must learn the ordering."""
    agent = DQN(N_FEATURES, seed=11, lr=1e-3)
    replay = Replay(2048, N_FEATURES)
    s = np.zeros(N_FEATURES, dtype=np.float32)
    mask = np.array([True, True])
    for i in range(1024):
        a = i % 2
        replay.add(s, a, float(a), s, mask, True)  # done: target == reward
    for _ in range(2000):
        agent.train_step(replay)
    q = agent.online.forward(s)[0]
    assert q[1] > q[0] + 0.5
    assert agent.greedy(s, mask) == 1
    # Masking must veto the greedy pick.
    assert agent.greedy(s, np.array([True, False])) == 0


def test_rl_controller_respects_mask_and_is_deterministic(tmp_path):
    from traffic_rl.rl.policy import RLController

    net = MLP((N_FEATURES, 64, 64, 2), np.random.default_rng(5))
    path = tmp_path / "w.npz"
    net.save(path)
    controller = RLController(weights=path)
    sim = IntersectionSim(make_config("symmetric"))
    obs = sim.reset(1)
    for _ in range(200):
        a1 = controller.act(obs)
        assert obs.action_mask[a1], "controller picked a masked action"
        assert controller.act(obs) == a1
        obs = sim.step(a1).obs


def test_gymnasium_env_api():
    gym = pytest.importorskip("gymnasium")
    from traffic_rl.rl.env import TrafficEnv

    env = TrafficEnv("symmetric", episode_seconds=60)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    obs, info = env.reset(seed=3)
    assert obs.shape == (N_FEATURES,) and "action_mask" in info
    total = 0.0
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(i % 2)
        total += reward
        assert not terminated
    assert truncated
    assert total <= 0.0  # waiting can only cost
