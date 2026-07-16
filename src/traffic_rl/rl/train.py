"""Train the DQN on randomized demand AND randomized intersection layouts —
one policy, no per-scenario tuning.

Each episode samples fresh Poisson rates (per-approach vehicles U(100, 650)
veh/h, per-movement pedestrians U(20, 90) peds/h), turning fractions, and a
left-turn treatment (through-only, permissive bays, protected lefts), so the
policy must generalize across intersection types. Actions live in canonical
slot space (see rl/features.py), which is what makes one set of weights valid
at any layout. Reward is the negative of total person-waiting (vehicles +
pedestrians) accrued per step. Fully seeded and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from traffic_rl.config import (
    MAX_PHASES,
    DemandConfig,
    IntersectionLayout,
    LeftTurnTreatment,
    SimConfig,
)
from traffic_rl.rl.dqn import DQN, Replay
from traffic_rl.rl.features import N_FEATURES, featurize, slot_action_mask, slot_to_phase
from traffic_rl.rl.policy import DEFAULT_WEIGHTS
from traffic_rl.sim.core import IntersectionSim

REWARD_SCALE = 50.0
EPISODE_SECONDS = 1200.0


def shaped_reward(info: dict, obs) -> float:
    """Person-waiting accrued this step. (A squared-queue fairness term was
    tried and destabilized training — exploration-phase queues make the
    penalty explode; documented negative result.)"""
    return -(
        info["wait_accrued_this_step"] + info["ped_wait_accrued_this_step"]
    ) / REWARD_SCALE
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 2000
REPLAY_CAPACITY = 200_000
WARMUP_STEPS = 5_000  # pure exploration before training starts
EPS_START, EPS_END = 1.0, 0.05


def sample_demand(
    rng: np.random.Generator,
    with_turns: bool = True,
    lanes: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> DemandConfig:
    veh = tuple(float(rng.uniform(100, 650)) * lanes[a] for a in range(4))
    ped = tuple(float(rng.uniform(20, 90)) for _ in range(2))
    if not with_turns:
        return DemandConfig(vehicle_rates=veh, ped_rates=ped)
    turns = []
    for a in range(4):
        left = float(rng.uniform(0.03, 0.30))
        # A left bay is one lane no matter how wide the approach: cap the left
        # volume near a protected bay's practical capacity (~260 veh/h).
        if veh[a] > 0:
            left = min(left, 260.0 / veh[a])
        right = float(rng.uniform(0.03, 0.20))
        turns.append((left, 1.0 - left - right, right))
    return DemandConfig(vehicle_rates=veh, ped_rates=ped, turn_fractions=tuple(turns))


def rescale_to_flow_ratio(
    demand: DemandConfig, layout: IntersectionLayout, rng: np.random.Generator
) -> DemandConfig:
    """Rescale vehicle rates so the site's total Webster flow ratio Y lands in
    a controlled U(0.45, 0.92) band. Raw per-lane sampling leaves many
    multi-phase episodes hopelessly oversaturated — replay fills with
    unwinnable states and learning collapses (measured, not conjectured:
    a specialist trained on such episodes scored p95 2400 s)."""
    from traffic_rl.config import build_phases

    phases = build_phases(layout)
    rates = demand.movement_rates(layout)
    y_total = sum(
        max(
            (rates[g] / (1800.0 * (layout.through_lanes[g] if g < 4 else 1)) for g in p.movements),
            default=0.0,
        )
        for p in phases
    )
    if y_total <= 1e-9:
        return demand
    target = float(rng.uniform(0.45, 0.92))
    scale = target / y_total
    return DemandConfig(
        vehicle_rates=tuple(v * scale for v in demand.vehicle_rates),
        ped_rates=demand.ped_rates,
        turn_fractions=demand.turn_fractions,
    )


def sample_layout(rng: np.random.Generator) -> IntersectionLayout:
    """Mix of intersection types so the slot-space policy sees them all:
    through-only (legacy), permissive bays, protected on one street, full quad —
    each with single- or multi-lane through groups (capacity varies 2x)."""
    r = rng.random()
    P, X = LeftTurnTreatment.PERMISSIVE, LeftTurnTreatment.PROTECTED
    # Two lane configs (uniform, and the arterial's major-street pattern):
    # a wider menu diluted the small net's capacity without adding skill.
    lanes_choices = ((1, 1, 1, 1), (2, 2, 1, 1))
    lanes = lanes_choices[rng.integers(0, len(lanes_choices))]
    if r < 0.20:
        return IntersectionLayout(through_lanes=lanes)  # 2-phase, shared lefts
    if r < 0.38:
        return IntersectionLayout(left_turn=(P, P, P, P), through_lanes=lanes)
    if r < 0.68:
        # The suburban-arterial pattern (protected on the major street) gets
        # extra weight: it is where phase allocation is hardest to learn.
        return IntersectionLayout(left_turn=(X, X, P, P), through_lanes=lanes)
    return IntersectionLayout(left_turn=(X, X, X, X), through_lanes=lanes)


def _sample_timing(rng: np.random.Generator):
    """A quarter of episodes run geometry-derived clearance intervals (one
    representative arterial geometry), the rest the defaults. Fully random
    geometry per episode was tried and diluted the small net — negative
    result, documented."""
    from traffic_rl.config import GeometryConfig, SignalTimingConfig, timing_from_geometry

    if rng.random() < 0.75:
        return SignalTimingConfig()
    return timing_from_geometry(
        GeometryConfig(
            ns_speed_mph=45.0, ew_speed_mph=30.0,
            ns_street_width_m=22.0, ew_street_width_m=14.0,
        )
    )


def sample_config(rng: np.random.Generator) -> SimConfig:
    layout = sample_layout(rng)
    with_turns = any(t != LeftTurnTreatment.SHARED for t in layout.left_turn) or rng.random() < 0.5
    demand = rescale_to_flow_ratio(
        sample_demand(rng, with_turns, lanes=layout.through_lanes), layout, rng
    )
    return SimConfig(demand=demand, layout=layout, timing=_sample_timing(rng))


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
    agent = DQN(N_FEATURES, n_actions=MAX_PHASES, hidden=hidden, gamma=gamma, lr=lr, seed=seed)
    replay = Replay(REPLAY_CAPACITY, N_FEATURES, n_actions=MAX_PHASES)
    episode_steps = round(EPISODE_SECONDS)
    log_rows: list[dict] = []

    global_step = 0
    episode = 0
    t0 = time.perf_counter()
    while global_step < steps:
        config = sample_config(rng)
        sim = IntersectionSim(config)
        obs = sim.reset(int(rng.integers(2**31)))
        features = featurize(obs)
        ep_reward = 0.0
        for _ in range(episode_steps):
            slot = agent.act(features, slot_action_mask(obs), epsilon(global_step, eps_decay_steps))
            result = sim.step(slot_to_phase(obs, slot))
            reward = shaped_reward(result.info, result.obs)
            next_features = featurize(result.obs)
            # Episode end is a truncation, not a terminal state: done stays
            # False so the target keeps bootstrapping.
            replay.add(features, slot, reward, next_features, slot_action_mask(result.obs), False)
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
    layout = sample_layout(rng)
    regimes = [
        rescale_to_flow_ratio(
            sample_demand(rng, lanes=layout.through_lanes), layout, rng
        )
        for _ in range(3)
    ]
    schedule = tuple((600.0 * k, d) for k, d in enumerate(regimes))
    return SimConfig(
        demand=regimes[0], demand_schedule=schedule, layout=layout,
        timing=_sample_timing(rng),
    )


def mixed_episode_factory(rng: np.random.Generator) -> SimConfig:
    if rng.random() < 0.5:
        return sample_config(rng)
    return sample_schedule_config(rng)


def train_pattern_policy(
    steps: int,
    seed: int,
    out: Path,
    episode_factory=mixed_episode_factory,
    eps_decay_steps: int = 400_000,
    log_prefix: str = "",
    init_weights: Path | None = None,
    hidden: int = PATTERN_HIDDEN,
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
        N_FEATURES_PATTERN, n_actions=MAX_PHASES, hidden=hidden,
        gamma=gamma**N_STEP, lr=3e-4, seed=seed,
    )
    eps_start = EPS_START
    if init_weights is not None and Path(init_weights).exists():
        agent.online = MLP.load(init_weights)
        agent.target.copy_from(agent.online)
        agent.opt = type(agent.opt)(agent.online.params, lr=1e-4)  # gentler fine-tune
        eps_decay_steps = min(eps_decay_steps, max(steps // 4, 1))
        eps_start = 0.3  # mostly exploit the inherited policy from the start
        print(f"{log_prefix}fine-tuning from {init_weights}")
    replay = Replay(REPLAY_CAPACITY, N_FEATURES_PATTERN, n_actions=MAX_PHASES)
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
            slot = agent.act(features, slot_action_mask(obs), eps)
            result = sim.step(slot_to_phase(obs, slot))
            reward = shaped_reward(result.info, result.obs)
            next_features = featurize_with_patterns(result.obs, tracker)
            pending.append((features, slot))
            rewards.append(reward)
            if len(pending) == N_STEP:
                s0, a0 = pending.pop(0)
                n_step_return = float(np.dot(gamma_pows, rewards[:N_STEP]))
                rewards.pop(0)
                replay.add(
                    s0, a0, n_step_return, next_features,
                    slot_action_mask(result.obs), False,
                )
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
            hidden=args.hidden if args.hidden != 64 else PATTERN_HIDDEN,
        )
    else:
        train(args.steps, args.seed, args.out or DEFAULT_WEIGHTS, args.log,
              hidden=args.hidden, gamma=args.gamma, lr=args.lr,
              eps_decay_steps=args.eps_decay)


if __name__ == "__main__":
    main()
