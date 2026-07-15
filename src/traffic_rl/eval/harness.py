"""Experiment driver: paired-seed runs of every controller on every scenario.

Statistical honesty rules live here and in metrics.py:
- n_runs independent seeds; run k uses the SAME seed for every controller
  (common random numbers), so summary.json can report paired difference CIs.
- Per-run metrics go to runs.csv; pooled wait distributions to waits.npz
  (illustrative only); queue time series to queues.npz; aggregates + paired
  Δ-vs-naive CIs to summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from traffic_rl.config import SimConfig
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.eval.metrics import compute_run_metrics, mean_ci, paired_diff_ci
from traffic_rl.scenarios import SCENARIOS, make_config
from traffic_rl.sim.core import IntersectionSim

_ARRAY_KEYS = ("veh_waits", "ped_waits", "queue_timeseries")
_CI_METRICS = (
    "p95_wait",
    "mean_wait",
    "max_wait",
    "throughput_veh_per_h",
    "mean_queue",
    "ped_p95_wait",
    "ped_mean_wait",
)


def run_seeds(base_seed: int, n_runs: int) -> list[int]:
    return [int(s) for s in np.random.SeedSequence(base_seed).generate_state(n_runs)]


def run_single(controller_name: str, config: SimConfig, seed: int) -> dict:
    sim = IntersectionSim(config)
    controller = CONTROLLER_REGISTRY[controller_name]()
    obs = sim.reset(seed)
    controller.reset(config, np.random.default_rng(seed))
    for _ in range(config.n_steps):
        obs = sim.step(controller.act(obs)).obs
    metrics = compute_run_metrics(sim.event_log.finalize(), config)
    metrics["seed"] = seed
    return metrics


def run_experiment(
    controllers: list[str],
    scenarios: list[str],
    n_runs: int,
    base_seed: int,
    out_dir: Path,
) -> None:
    seeds = run_seeds(base_seed, n_runs)
    for scenario in scenarios:
        config = make_config(scenario)
        per_controller: dict[str, list[dict]] = {}
        for name in controllers:
            t0 = time.perf_counter()
            runs = [run_single(name, config, seed) for seed in seeds]
            per_controller[name] = runs
            _write_controller_outputs(out_dir / scenario / name, runs)
            wall = time.perf_counter() - t0
            agg = mean_ci(np.array([r["p95_wait"] for r in runs]))
            bound = "≥" if any(r["p95_is_lower_bound"] for r in runs) else " "
            print(
                f"[{scenario:>10}] {name:<13} p95 {bound}{agg['mean']:7.1f} s "
                f"(95% CI {agg['lo']:6.1f}–{agg['hi']:6.1f}, n={n_runs}) "
                f"unstable {sum(r['unstable'] for r in runs)}/{n_runs}  "
                f"[{wall:.1f}s wall]"
            )
        _write_summary(out_dir / scenario, scenario, config, per_controller, seeds)


def _write_controller_outputs(dest: Path, runs: list[dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    scalar_keys = [k for k in runs[0] if k not in _ARRAY_KEYS]
    with open(dest / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r[k] for k in scalar_keys})
    np.savez_compressed(
        dest / "waits.npz",
        veh_waits=np.concatenate([r["veh_waits"] for r in runs]),
        ped_waits=np.concatenate([r["ped_waits"] for r in runs]),
    )
    np.savez_compressed(
        dest / "queues.npz",
        queue_timeseries=np.stack([r["queue_timeseries"] for r in runs]),
    )


def _write_summary(
    dest: Path,
    scenario: str,
    config: SimConfig,
    per_controller: dict[str, list[dict]],
    seeds: list[int],
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "scenario": scenario,
        "n_runs": len(seeds),
        "seeds": seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "vehicle_rates": config.demand.vehicle_rates,
            "ped_rates": config.demand.ped_rates,
            "warmup": config.warmup,
            "measured": config.measured,
            "dt": config.dt,
        },
        "controllers": {},
        "paired_vs_naive": {},
    }
    for name, runs in per_controller.items():
        summary["controllers"][name] = {
            m: mean_ci(np.array([r[m] for r in runs])) for m in _CI_METRICS
        } | {
            "n_unstable": int(sum(r["unstable"] for r in runs)),
            "p95_any_lower_bound": bool(any(r["p95_is_lower_bound"] for r in runs)),
        }
    if "naive" in per_controller:
        naive_p95 = np.array([r["p95_wait"] for r in per_controller["naive"]])
        for name, runs in per_controller.items():
            if name == "naive":
                continue
            other_p95 = np.array([r["p95_wait"] for r in runs])
            summary["paired_vs_naive"][name] = paired_diff_ci(naive_p95, other_p95)
    with open(dest / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the traffic-rl evaluation harness.")
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLER_REGISTRY))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    for c in args.controllers:
        if c not in CONTROLLER_REGISTRY:
            parser.error(f"unknown controller {c!r}; choose from {sorted(CONTROLLER_REGISTRY)}")
    for s in args.scenarios:
        if s not in SCENARIOS:
            parser.error(f"unknown scenario {s!r}; choose from {sorted(SCENARIOS)}")
    run_experiment(args.controllers, args.scenarios, args.runs, args.seed, args.out)


if __name__ == "__main__":
    main()
