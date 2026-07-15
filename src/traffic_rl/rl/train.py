"""Train the DQN on randomized demand — one policy, no per-scenario tuning.

Each episode samples fresh Poisson rates (per-approach vehicles U(100, 650)
veh/h, per-movement pedestrians U(20, 90) peds/h) so the policy must generalize
rather than memorize a scenario. Reward is the negative of total person-waiting
(vehicles + pedestrians) accrued per step. Fully seeded and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from traffic_rl.config import DemandConfig, SimConfig
from traffic_rl.rl.dqn import DQN, Replay
from traffic_rl.rl.features import N_FEATURES, featurize
from traffic_rl.rl.policy import DEFAULT_WEIGHTS
from traffic_rl.sim.core import IntersectionSim

REWARD_SCALE = 50.0
EPISODE_SECONDS = 1200.0
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 2000
REPLAY_CAPACITY = 200_000
WARMUP_STEPS = 5_000  # pure exploration before training starts
EPS_START, EPS_END = 1.0, 0.05


def sample_demand(rng: np.random.Generator) -> DemandConfig:
    veh = tuple(float(rng.uniform(100, 650)) for _ in range(4))
    ped = tuple(float(rng.uniform(20, 90)) for _ in range(2))
    return DemandConfig(vehicle_rates=veh, ped_rates=ped)


def epsilon(step: int, decay_steps: int) -> float:
    frac = min(1.0, step / decay_steps)
    return EPS_START + (EPS_END - EPS_START) * frac


def train(
    steps: int,
    seed: int,
    out: Path,
    log_path: Path | None = None,
    hidden: int = 64,
    gamma: float = 0.99,
    lr: float = 3e-4,
    eps_decay_steps: int = 300_000,
) -> DQN:
    rng = np.random.default_rng(seed)
    agent = DQN(N_FEATURES, hidden=hidden, gamma=gamma, lr=lr, seed=seed)
    replay = Replay(REPLAY_CAPACITY, N_FEATURES)
    episode_steps = round(EPISODE_SECONDS)
    log_rows: list[dict] = []

    global_step = 0
    episode = 0
    t0 = time.perf_counter()
    while global_step < steps:
        config = SimConfig(demand=sample_demand(rng))
        sim = IntersectionSim(config)
        obs = sim.reset(int(rng.integers(2**31)))
        features = featurize(obs)
        ep_reward = 0.0
        for _ in range(episode_steps):
            action = agent.act(features, obs.action_mask, epsilon(global_step, eps_decay_steps))
            result = sim.step(action)
            reward = -(
                result.info["wait_accrued_this_step"]
                + result.info["ped_wait_accrued_this_step"]
            ) / REWARD_SCALE
            next_features = featurize(result.obs)
            # Episode end is a truncation, not a terminal state: done stays
            # False so the target keeps bootstrapping.
            replay.add(features, action, reward, next_features, result.obs.action_mask, False)
            obs, features = result.obs, next_features
            ep_reward += reward
            global_step += 1
            if len(replay) >= WARMUP_STEPS and global_step % TRAIN_EVERY == 0:
                agent.train_step(replay)
            if global_step % TARGET_SYNC_EVERY == 0:
                agent.sync_target()
            if global_step >= steps:
                break
        episode += 1
        eps_now = epsilon(global_step, eps_decay_steps)
        log_rows.append(
            {"episode": episode, "step": global_step, "mean_reward": ep_reward / episode_steps,
             "epsilon": round(eps_now, 3)}
        )
        if episode % 25 == 0:
            recent = np.mean([r["mean_reward"] for r in log_rows[-25:]])
            rate = global_step / (time.perf_counter() - t0)
            print(
                f"episode {episode:5d}  step {global_step:8d}  "
                f"mean step-reward (last 25 ep) {recent:8.3f}  "
                f"eps {eps_now:.2f}  [{rate:,.0f} steps/s]"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    agent.online.save(out)
    print(f"saved weights -> {out}")
    if log_path:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"saved training log -> {log_path}")
    return agent


# ------------------------------------------------------------- pattern recipe
#
# The pattern-aware policy: PatternTracker demand features, n-step returns,
# a wider net, and training demand that VARIES WITHIN episodes (piecewise
# schedules) so anticipating the pattern actually pays during training.

N_STEP = 5
PATTERN_HIDDEN = 96
PATTERN_EPISODE_SECONDS = 1800.0


def sample_schedule_config(rng: np.random.Generator) -> SimConfig:
    """Piecewise demand: three 600 s regimes inside one episode."""
    regimes = [sample_demand(rng) for _ in range(3)]
    schedule = tuple((600.0 * k, d) for k, d in enumerate(regimes))
    return SimConfig(demand=regimes[0], demand_schedule=schedule)


def mixed_episode_factory(rng: np.random.Generator) -> SimConfig:
    if rng.random() < 0.5:
        return SimConfig(demand=sample_demand(rng))
    return sample_schedule_config(rng)


def train_pattern_policy(
    steps: int,
    seed: int,
    out: Path,
    episode_factory=mixed_episode_factory,
    eps_decay_steps: int = 400_000,
    log_prefix: str = "",
    init_weights: Path | None = None,
) -> DQN:
    """Shared training loop for the pattern recipe (also used by the
    site-specific trainer in optimize.py, with a data-driven factory).

    init_weights warm-starts from an existing policy (fine-tuning): the site
    trainer starts from the general pattern policy so short runs adapt to the
    site instead of relearning traffic control from scratch."""
    from traffic_rl.rl.dqn import MLP
    from traffic_rl.rl.patterns import N_FEATURES_PATTERN, PatternTracker, featurize_with_patterns

    rng = np.random.default_rng(seed)
    # n-step returns: the stored transition spans N_STEP steps, so the bootstrap
    # discount is gamma**N_STEP while rewards inside the window use gamma.
    gamma = 0.99
    agent = DQN(
        N_FEATURES_PATTERN, hidden=PATTERN_HIDDEN, gamma=gamma**N_STEP, lr=3e-4, seed=seed
    )
    eps_start = EPS_START
    if init_weights is not None and Path(init_weights).exists():
        agent.online = MLP.load(init_weights)
        agent.target.copy_from(agent.online)
        agent.opt = type(agent.opt)(agent.online.params, lr=1e-4)  # gentler fine-tune
        eps_decay_steps = min(eps_decay_steps, max(steps // 4, 1))
        eps_start = 0.3  # mostly exploit the inherited policy from the start
        print(f"{log_prefix}fine-tuning from {init_weights}")
    replay = Replay(REPLAY_CAPACITY, N_FEATURES_PATTERN)
    episode_steps = round(PATTERN_EPISODE_SECONDS)
    gamma_pows = gamma ** np.arange(N_STEP)

    global_step = 0
    episode = 0
    t0 = time.perf_counter()
    recent: list[float] = []
    while global_step < steps:
        config = episode_factory(rng)
        sim = IntersectionSim(config)
        tracker = PatternTracker(dt=config.dt)
        obs = sim.reset(int(rng.integers(2**31)))
        features = featurize_with_patterns(obs, tracker)
        pending: list[tuple[np.ndarray, int]] = []
        rewards: list[float] = []
        ep_reward = 0.0
        for _ in range(episode_steps):
            eps = eps_start + (EPS_END - eps_start) * min(1.0, global_step / eps_decay_steps)
            action = agent.act(features, obs.action_mask, eps)
            result = sim.step(action)
            reward = -(
                result.info["wait_accrued_this_step"]
                + result.info["ped_wait_accrued_this_step"]
            ) / REWARD_SCALE
            next_features = featurize_with_patterns(result.obs, tracker)
            pending.append((features, action))
            rewards.append(reward)
            if len(pending) == N_STEP:
                s0, a0 = pending.pop(0)
                n_step_return = float(np.dot(gamma_pows, rewards[:N_STEP]))
                rewards.pop(0)
                replay.add(s0, a0, n_step_return, next_features, result.obs.action_mask, False)
            obs, features = result.obs, next_features
            ep_reward += reward
            global_step += 1
            if len(replay) >= WARMUP_STEPS and global_step % TRAIN_EVERY == 0:
                agent.train_step(replay)
            if global_step % TARGET_SYNC_EVERY == 0:
                agent.sync_target()
            if global_step >= steps:
                break
        episode += 1
        recent.append(ep_reward / episode_steps)
        if episode % 25 == 0:
            rate = global_step / (time.perf_counter() - t0)
            print(
                f"{log_prefix}episode {episode:5d}  step {global_step:8d}  "
                f"mean step-reward (last 25 ep) {np.mean(recent[-25:]):8.3f}  "
                f"eps {eps:.2f}  [{rate:,.0f} steps/s]"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    agent.online.save(out)
    print(f"saved pattern weights -> {out}")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the traffic-rl DQN policy.")
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=Path("results") / "training_log.csv")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps-decay", type=int, default=300_000)
    parser.add_argument(
        "--pattern", action="store_true",
        help="train the pattern-aware recipe (demand-rate features, n-step returns)",
    )
    args = parser.parse_args()
    if args.pattern:
        from traffic_rl.rl.pattern_policy import PATTERN_WEIGHTS

        train_pattern_policy(
            args.steps, args.seed, args.out or PATTERN_WEIGHTS,
            eps_decay_steps=args.eps_decay,
        )
    else:
        train(args.steps, args.seed, args.out or DEFAULT_WEIGHTS, args.log,
              hidden=args.hidden, gamma=args.gamma, lr=args.lr,
              eps_decay_steps=args.eps_decay)


if __name__ == "__main__":
    main()
