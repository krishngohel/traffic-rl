"""Network experiment driver: same statistical rules, journey-level waits.

A vehicle's wait is the sum of its queue waits across every intersection it
passes. Measured window filters on NETWORK ENTRY time; vehicles still in the
corridor at the horizon are censored with an exact lower bound (accrued wait,
plus the current queue wait if standing in one) — the elapsed-time rule from
the single-node metrics would overstate journey waits, so this module owns its
own censoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from traffic_rl.eval.metrics import (
    CENSORED_FRACTION_LIMIT,
    UNSTABLE_FINAL_QUEUE_RATIO,
    mean_ci,
    paired_diff_ci,
)
from traffic_rl.sim.network import NetworkConfig, NetworkDemandConfig, NetworkSim

NETWORK_SCENARIOS: dict[str, NetworkDemandConfig] = {
    # Balanced arterial, moderate cross streets.
    "corridor": NetworkDemandConfig(arterial_east=600, arterial_west=600, cross=150, peds=30),
    # Directional rush: heavy eastbound platoons — the green-wave showcase.
    "corridor_rush": NetworkDemandConfig(arterial_east=850, arterial_west=400, cross=180, peds=30),
    # Strong cross-street demand: coordination must not starve the side streets.
    "corridor_cross": NetworkDemandConfig(arterial_east=500, arterial_west=500, cross=350, peds=50),
}


def network_controller_registry() -> dict:
    from traffic_rl.controllers import CONTROLLER_REGISTRY
    from traffic_rl.controllers.network import (
        GreenWaveController,
        IndependentNetworkController,
        NetworkMaxPressureController,
    )

    def rl_net():
        from traffic_rl.rl.network_policy import SharedRLNetworkController

        return SharedRLNetworkController()

    return {
        "naive": lambda: IndependentNetworkController(CONTROLLER_REGISTRY["naive"], "naive"),
        "actuated": lambda: IndependentNetworkController(
            CONTROLLER_REGISTRY["actuated"], "actuated"
        ),
        "greenwave": GreenWaveController,
        "max_pressure": NetworkMaxPressureController,
        "rl": rl_net,
    }


def compute_network_metrics(sim: NetworkSim, config: NetworkConfig) -> dict:
    log = sim.journey_log()
    lo, hi = config.warmup, config.horizon

    entry = log["veh_arrival"]
    wait = log["veh_wait_lower_bound"]
    censored_all = np.isnan(log["veh_depart"])
    scored = (entry >= lo) & (entry < hi)
    w, c = wait[scored], censored_all[scored]

    p_arr, p_ws = log["ped_arrival"], log["ped_walk_start"]
    p_scored = (p_arr >= lo) & (p_arr < hi)
    pa, pw = p_arr[p_scored], p_ws[p_scored]
    ped_waits = np.where(np.isnan(pw), hi - pa, pw - pa)

    in_window = (log["step_t"] >= lo) & (log["step_t"] < hi)
    total_queue = log["step_queues"][in_window].sum(axis=1)

    n = len(w)
    censored_fraction = float(c.mean()) if n else 0.0
    mean_queue = float(total_queue.mean()) if len(total_queue) else 0.0
    final_queue = float(total_queue[-1]) if len(total_queue) else 0.0
    return {
        "n_vehicles": n,
        "p95_wait": float(np.percentile(w, 95)) if n else 0.0,
        "mean_wait": float(w.mean()) if n else 0.0,
        "max_wait": float(w.max()) if n else 0.0,
        "throughput_veh_per_h": float((~c).sum() / config.measured * 3600.0),
        "mean_queue": mean_queue,
        "max_queue": float(total_queue.max()) if len(total_queue) else 0.0,
        "final_queue": final_queue,
        "n_censored": int(c.sum()),
        "censored_fraction": censored_fraction,
        "p95_is_lower_bound": censored_fraction > CENSORED_FRACTION_LIMIT,
        "unstable": bool(
            censored_fraction > CENSORED_FRACTION_LIMIT
            or (mean_queue > 0 and final_queue > UNSTABLE_FINAL_QUEUE_RATIO * mean_queue)
        ),
        "n_peds": int(len(ped_waits)),
        "ped_p95_wait": float(np.percentile(ped_waits, 95)) if len(ped_waits) else 0.0,
        "ped_mean_wait": float(ped_waits.mean()) if len(ped_waits) else 0.0,
        "veh_waits": w,
        "ped_waits": ped_waits,
        "queue_timeseries": total_queue,
    }


def run_network_controller(controller, config: NetworkConfig, seed: int) -> dict:
    sim = NetworkSim(config)
    observations = sim.reset(seed)
    controller.reset(config, np.random.default_rng(seed))
    for _ in range(config.n_steps):
        observations = sim.step(controller.act(observations)).observations
    metrics = compute_network_metrics(sim, config)
    metrics["seed"] = seed
    return metrics


_ARRAY_KEYS = ("veh_waits", "ped_waits", "queue_timeseries")
_CI_METRICS = ("p95_wait", "mean_wait", "throughput_veh_per_h", "ped_p95_wait", "ped_mean_wait")


def run_network_experiment(
    controllers: list[str],
    scenarios: list[str],
    n_runs: int,
    base_seed: int,
    out_dir: Path,
    n_nodes: int = 4,
) -> None:
    registry = network_controller_registry()
    seeds = [int(s) for s in np.random.SeedSequence(base_seed).generate_state(n_runs)]
    for scenario in scenarios:
        config = NetworkConfig(demand=NETWORK_SCENARIOS[scenario], n_nodes=n_nodes)
        per_controller: dict[str, list[dict]] = {}
        for name in controllers:
            t0 = time.perf_counter()
            runs = [run_network_controller(registry[name](), config, s) for s in seeds]
            per_controller[name] = runs
            dest = out_dir / scenario / name
            dest.mkdir(parents=True, exist_ok=True)
            scalar_keys = [k for k in runs[0] if k not in _ARRAY_KEYS]
            with open(dest / "runs.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=scalar_keys)
                writer.writeheader()
                writer.writerows([{k: r[k] for k in scalar_keys} for r in runs])
            np.savez_compressed(
                dest / "waits.npz",
                veh_waits=np.concatenate([r["veh_waits"] for r in runs]),
                ped_waits=np.concatenate([r["ped_waits"] for r in runs]),
            )
            np.savez_compressed(
                dest / "queues.npz",
                queue_timeseries=np.stack([r["queue_timeseries"] for r in runs]),
            )
            agg = mean_ci(np.array([r["p95_wait"] for r in runs]))
            bound = "≥" if any(r["p95_is_lower_bound"] for r in runs) else " "
            print(
                f"[{scenario:>14}] {name:<13} journey p95 {bound}{agg['mean']:7.1f} s "
                f"(95% CI {agg['lo']:6.1f}–{agg['hi']:6.1f}, n={n_runs}) "
                f"unstable {sum(r['unstable'] for r in runs)}/{n_runs}  "
                f"[{time.perf_counter() - t0:.1f}s wall]"
            )
        _write_summary(out_dir / scenario, scenario, config, per_controller, seeds)


def _write_summary(dest, scenario, config, per_controller, seeds) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "scenario": scenario,
        "n_runs": len(seeds),
        "seeds": seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_nodes": config.n_nodes,
            "link_travel": config.link_travel,
            "arterial_east": config.demand.arterial_east,
            "arterial_west": config.demand.arterial_west,
            "cross": config.demand.cross,
            "peds": config.demand.peds,
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
        base = np.array([r["p95_wait"] for r in per_controller["naive"]])
        for name, runs in per_controller.items():
            if name != "naive":
                other = np.array([r["p95_wait"] for r in runs])
                summary["paired_vs_naive"][name] = paired_diff_ci(base, other)
    with open(dest / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    registry = network_controller_registry()
    parser = argparse.ArgumentParser(description="Evaluate controllers on the corridor network.")
    parser.add_argument("--controllers", nargs="+", default=list(registry))
    parser.add_argument("--scenarios", nargs="+", default=list(NETWORK_SCENARIOS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results") / "network")
    args = parser.parse_args()
    run_network_experiment(
        args.controllers, args.scenarios, args.runs, args.seed, args.out, n_nodes=args.nodes
    )


if __name__ == "__main__":
    main()
