"""Train the shared corridor policy on randomized network demand.

One DQN, every intersection: each node contributes its own transition each step
(local observation + downstream queues -> action -> LOCAL reward), all into one
shared replay buffer. Local rewards (own queue + own waiting peds) are the
standard decentralized credit-assignment choice; the downstream-queue features
are what lets a node learn not to flood a saturated neighbor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from traffic_rl.rl.dqn import DQN, Replay
from traffic_rl.rl.network_policy import (
    N_NETWORK_FEATURES,
    NETWORK_WEIGHTS,
    downstream_queues,
    featurize_network,
)
from traffic_rl.sim.network import NetworkConfig, NetworkDemandConfig, NetworkSim

REWARD_SCALE = 20.0
EPISODE_SECONDS = 1200.0
TRAIN_EVERY = 4  # network steps (= 4 * n_nodes transitions)
TARGET_SYNC_EVERY = 2000
REPLAY_CAPACITY = 300_000
WARMUP_STEPS = 5_000
EPS_START, EPS_END = 1.0, 0.05


def sample_network_demand(rng: np.random.Generator) -> NetworkDemandConfig:
    return NetworkDemandConfig(
        arterial_east=float(rng.uniform(300, 900)),
        arterial_west=float(rng.uniform(300, 900)),
        cross=float(rng.uniform(80, 350)),
        peds=float(rng.uniform(10, 60)),
    )


def train(
    steps: int,
    seed: int,
    out: Path,
    n_nodes: int = 4,
    eps_decay_steps: int = 300_000,
) -> DQN:
    rng = np.random.default_rng(seed)
    agent = DQN(N_NETWORK_FEATURES, seed=seed)
    replay = Replay(REPLAY_CAPACITY, N_NETWORK_FEATURES)
    episode_steps = round(EPISODE_SECONDS)

    global_step = 0
    episode = 0
    t0 = time.perf_counter()
    recent_rewards: list[float] = []
    while global_step < steps:
        config = NetworkConfig(demand=sample_network_demand(rng), n_nodes=n_nodes)
        sim = NetworkSim(config)
        observations = sim.reset(int(rng.integers(2**31)))
        features = [
            featurize_network(o, *downstream_queues(observations, i, n_nodes))
            for i, o in enumerate(observations)
        ]
        ep_reward = 0.0
        for _ in range(episode_steps):
            eps = EPS_START + (EPS_END - EPS_START) * min(1.0, global_step / eps_decay_steps)
            actions = [
                agent.act(features[i], observations[i].action_mask, eps)
                for i in range(n_nodes)
            ]
            result = sim.step(actions)
            next_obs = result.observations
            next_features = [
                featurize_network(o, *downstream_queues(next_obs, i, n_nodes))
                for i, o in enumerate(next_obs)
            ]
            for i in range(n_nodes):
                local = -(
                    float(result.info["node_queues"][i])
                    + sum(len(w) for w in sim.nodes[i].waiting_peds)
                ) / REWARD_SCALE
                replay.add(
                    features[i], actions[i], local, next_features[i],
                    next_obs[i].action_mask, False,
                )
                ep_reward += local
            observations, features = next_obs, next_features
            global_step += 1
            if len(replay) >= WARMUP_STEPS and global_step % TRAIN_EVERY == 0:
                agent.train_step(replay)
            if global_step % TARGET_SYNC_EVERY == 0:
                agent.sync_target()
            if global_step >= steps:
                break
        episode += 1
        recent_rewards.append(ep_reward / episode_steps)
        if episode % 25 == 0:
            rate = global_step / (time.perf_counter() - t0)
            print(
                f"episode {episode:5d}  step {global_step:8d}  "
                f"mean step-reward (last 25 ep) {np.mean(recent_rewards[-25:]):8.3f}  "
                f"eps {eps:.2f}  [{rate:,.0f} steps/s]"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    agent.online.save(out)
    print(f"saved network weights -> {out}")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared corridor RL policy.")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--out", type=Path, default=NETWORK_WEIGHTS)
    args = parser.parse_args()
    train(args.steps, args.seed, args.out, n_nodes=args.nodes)


if __name__ == "__main__":
    main()
