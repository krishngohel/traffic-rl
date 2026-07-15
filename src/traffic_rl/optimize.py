"""Signal-retiming study in a box: feed in real traffic counts, get back
optimized time-of-day signal plans and an honest projected improvement.

Pipeline:
1. Load per-approach counts (data.load_counts_csv) into a demand schedule.
2. For each interval: compute the Webster plan from the observed flows, then
   refine with a local search over (cycle scale x split shift), each candidate
   scored on paired-seed simulations of that interval's demand.
3. Assemble the winners into a time-of-day ScheduledFixedTimeController.
4. Evaluate the recommended plan against baselines (naive 50/50, single Webster
   plan on the average flows, actuated, max-pressure, RL, and optionally the
   intersection's current plan) on the FULL profile with paired seeds.
5. Write report.md, report.json, and comparison.png.

All comparisons use the same statistical rules as the main harness: per-run
p95 vehicle wait, mean +/- t-based 95% CI over paired seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_rl.config import DemandConfig, DemandSchedule, SignalTimingConfig, SimConfig
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.controllers.fixed_time import (
    FixedTimeController,
    FixedTimePlan,
    ScheduledFixedTimeController,
)
from traffic_rl.controllers.webster import webster_plan
from traffic_rl.data import load_counts_csv
from traffic_rl.eval.harness import run_controller, run_seeds
from traffic_rl.eval.metrics import mean_ci, paired_diff_ci

CYCLE_SCALES = (0.75, 1.0, 1.25)
SPLIT_SHIFTS = (-0.08, 0.0, 0.08)
REFINE_RUNS = 5
MIN_EVAL_SECONDS = 1800.0


def _candidate_plans(demand: DemandConfig, timing: SignalTimingConfig) -> list[FixedTimePlan]:
    base = webster_plan(
        np.array(demand.vehicle_rates), 1800.0, timing, green_floor=timing.ped_service
    )
    y, ar, lost = timing.yellow, timing.all_red, timing.startup_lost
    base_cycle = base.cycle(y, ar)
    eff = np.array(base.greens) - lost
    base_split = eff[0] / eff.sum()
    floor_eff = max(timing.ped_service, timing.min_green) - lost

    plans, seen = [], set()
    for scale in CYCLE_SCALES:
        cycle = float(np.clip(base_cycle * scale, 40.0, 150.0))
        # Effective green available per cycle: C - L, L = 2 x (startup + y + ar).
        eff_total = cycle - 2 * (lost + y + ar)
        for shift in SPLIT_SHIFTS:
            split = float(np.clip(base_split + shift, 0.15, 0.85))
            g0 = max(eff_total * split, floor_eff)
            g1 = max(eff_total - g0, floor_eff)
            plan = FixedTimePlan(greens=(round(g0 + lost, 1), round(g1 + lost, 1)))
            if plan.greens not in seen:
                seen.add(plan.greens)
                plans.append(plan)
    return plans


def optimize_interval(
    demand: DemandConfig, timing: SignalTimingConfig, seeds: list[int]
) -> tuple[FixedTimePlan, float]:
    """Best fixed-time plan for one interval's demand: (plan, mean per-run p95)."""
    config = SimConfig(demand=demand, timing=timing, warmup=600.0, measured=1800.0)
    best_plan, best_score = None, np.inf
    for plan in _candidate_plans(demand, timing):
        p95s = [
            run_controller(FixedTimeController(plan), config, seed)["p95_wait"]
            for seed in seeds
        ]
        score = float(np.mean(p95s))
        if score < best_score:
            best_plan, best_score = plan, score
    return best_plan, best_score


def _average_demand(schedule: DemandSchedule, duration: float) -> DemandConfig:
    starts = [s for s, _ in schedule] + [duration]
    veh = np.zeros(4)
    ped = np.zeros(2)
    for (start, demand), end in zip(schedule, starts[1:], strict=True):
        w = (end - start) / duration
        veh += w * np.array(demand.vehicle_rates)
        ped += w * np.array(demand.ped_rates)
    return DemandConfig(vehicle_rates=tuple(veh), ped_rates=tuple(ped))


def optimize_from_counts(
    csv_path: Path,
    runs: int = 10,
    seed: int = 42,
    out_dir: Path = Path("results") / "optimize",
    current_plan: FixedTimePlan | None = None,
    include_rl: bool = True,
) -> dict:
    schedule, duration = load_counts_csv(csv_path)
    if duration < MIN_EVAL_SECONDS + 900.0:
        raise ValueError(
            f"need at least {(MIN_EVAL_SECONDS + 900) / 60:.0f} min of data, got "
            f"{duration / 60:.0f} min"
        )
    timing = SignalTimingConfig()
    refine_seeds = run_seeds(seed + 1, REFINE_RUNS)

    print(f"loaded {csv_path}: {len(schedule)} intervals, {duration / 3600:.1f} h")
    tod_plans: list[tuple[float, FixedTimePlan]] = []
    for start, demand in schedule:
        plan, score = optimize_interval(demand, timing, refine_seeds)
        tod_plans.append((start, plan))
        flows = tuple(round(v) for v in demand.vehicle_rates)
        print(
            f"  interval @ {start / 3600:5.2f} h  veh/h {flows}"
            f"  -> cycle {plan.cycle(timing.yellow, timing.all_red):5.1f} s"
            f"  greens {plan.greens}  (refine p95 {score:.1f} s)"
        )

    # Full-profile comparison, identical rules for every contender.
    config = SimConfig(
        demand=schedule[0][1],
        timing=timing,
        warmup=900.0,
        measured=duration - 900.0,
        demand_schedule=schedule,
    )
    seeds = run_seeds(seed, runs)
    avg_flows = _average_demand(schedule, duration)
    contenders: dict[str, object] = {
        "optimized_tod": ScheduledFixedTimeController(tod_plans),
        "naive_50_50": CONTROLLER_REGISTRY["naive"](),
        "webster_avg_flows": FixedTimeController(
            webster_plan(
                np.array(avg_flows.vehicle_rates), 1800.0, timing,
                green_floor=timing.ped_service,
            )
        ),
        "actuated": CONTROLLER_REGISTRY["actuated"](),
        "max_pressure": CONTROLLER_REGISTRY["max_pressure"](),
    }
    if current_plan is not None:
        contenders["current_plan"] = FixedTimeController(current_plan)
    if include_rl:
        try:
            contenders["rl"] = CONTROLLER_REGISTRY["rl"]()
        except FileNotFoundError:
            print("(no trained RL weights found — skipping rl contender)")

    results: dict[str, dict] = {}
    for name, controller in contenders.items():
        runs_out = [run_controller(controller, config, s) for s in seeds]
        results[name] = {
            "p95": mean_ci(np.array([r["p95_wait"] for r in runs_out])),
            "mean": mean_ci(np.array([r["mean_wait"] for r in runs_out])),
            "ped_p95": mean_ci(np.array([r["ped_p95_wait"] for r in runs_out])),
            "p95_runs": [r["p95_wait"] for r in runs_out],
            "n_unstable": int(sum(r["unstable"] for r in runs_out)),
        }
        print(
            f"  {name:<18} p95 {results[name]['p95']['mean']:6.1f} s "
            f"(CI {results[name]['p95']['lo']:.1f}-{results[name]['p95']['hi']:.1f})"
        )

    report = _build_report(csv_path, schedule, timing, tod_plans, results, runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (out_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    try:
        _render_chart(results, out_dir / "comparison.png")
    except ImportError:
        print("(matplotlib not installed — skipping comparison.png)")
    print(f"report -> {out_dir}\\report.md")
    return report


def _build_report(csv_path, schedule, timing, tod_plans, results, runs) -> dict:
    y, ar = timing.yellow, timing.all_red
    baseline_key = "current_plan" if "current_plan" in results else "naive_50_50"
    base_runs = np.array(results[baseline_key]["p95_runs"])
    opt_runs = np.array(results["optimized_tod"]["p95_runs"])
    diff = paired_diff_ci(base_runs, opt_runs)
    reduction_pct = 100.0 * diff["mean"] / float(np.mean(base_runs)) if base_runs.any() else 0.0
    return {
        "source": str(csv_path),
        "n_runs": runs,
        "baseline": baseline_key,
        "recommended_plans": [
            {
                "start_hour": start / 3600.0,
                "cycle_s": round(plan.cycle(y, ar), 1),
                "green_ns_s": plan.greens[0],
                "green_ew_s": plan.greens[1],
            }
            for start, plan in tod_plans
        ],
        "comparison": {
            name: {
                "p95_wait_s": {k: round(v, 2) for k, v in r["p95"].items()},
                "mean_wait_s": {k: round(v, 2) for k, v in r["mean"].items()},
                "ped_p95_wait_s": {k: round(v, 2) for k, v in r["ped_p95"].items()},
                "n_unstable": r["n_unstable"],
            }
            for name, r in results.items()
        },
        "headline": {
            "p95_reduction_vs_baseline_s": {k: round(v, 2) for k, v in diff.items()},
            "p95_reduction_pct": round(reduction_pct, 1),
        },
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Signal timing optimization report",
        "",
        f"Source: `{report['source']}` — {report['n_runs']} paired simulation runs per contender.",
        "",
        "## Recommended time-of-day plans",
        "",
        "| start | cycle (s) | NS green (s) | EW green (s) |",
        "|---|---|---|---|",
    ]
    for p in report["recommended_plans"]:
        h = int(p["start_hour"])
        m = int(round((p["start_hour"] - h) * 60))
        lines.append(
            f"| {h:02d}:{m:02d} | {p['cycle_s']} | {p['green_ns_s']} | {p['green_ew_s']} |"
        )
    d = report["headline"]
    lines += [
        "",
        "## Headline",
        "",
        f"Optimized time-of-day plans cut p95 wait by "
        f"**{d['p95_reduction_vs_baseline_s']['mean']} s "
        f"({d['p95_reduction_pct']}%)** vs `{report['baseline']}` "
        f"(paired 95% CI {d['p95_reduction_vs_baseline_s']['lo']}"
        f"–{d['p95_reduction_vs_baseline_s']['hi']} s).",
        "",
        "## Full comparison (p95 wait, mean of paired runs, 95% CI)",
        "",
        "| controller | p95 wait (s) | mean wait (s) | ped p95 (s) | unstable runs |",
        "|---|---|---|---|---|",
    ]
    ordered = sorted(report["comparison"].items(), key=lambda kv: kv[1]["p95_wait_s"]["mean"])
    for name, r in ordered:
        lines.append(
            f"| {name} | {r['p95_wait_s']['mean']} "
            f"[{r['p95_wait_s']['lo']}, {r['p95_wait_s']['hi']}] "
            f"| {r['mean_wait_s']['mean']} | {r['ped_p95_wait_s']['mean']} "
            f"| {r['n_unstable']}/{report['n_runs']} |"
        )
    lines += [
        "",
        "Adaptive controllers (actuated / max-pressure / rl) are included for context: "
        "they need detection hardware, while the recommended plans are drop-in "
        "fixed-time settings. All waits are simulated from the supplied counts "
        "under the model's stated limits (point-queue, no turns).",
        "",
    ]
    return "\n".join(lines)


def _render_chart(results: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from traffic_rl.eval.charts import GRID, INK, MUTED, SURFACE

    palette = ["#4a3aa7", "#2a78d6", "#1baf7a", "#eda100", "#008300", "#e34948", "#e87ba4"]
    ordered = sorted(results.items(), key=lambda kv: kv[1]["p95"]["mean"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_name, r) in enumerate(ordered):
        stats = r["p95"]
        err = [[stats["mean"] - stats["lo"]], [stats["hi"] - stats["mean"]]]
        ax.bar(i, stats["mean"], width=0.55, color=palette[i % len(palette)], zorder=3)
        ax.errorbar(i, stats["mean"], yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1, zorder=4)
        ax.annotate(
            f"{stats['mean']:.0f}", (i, stats["hi"]), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=9, color=INK,
        )
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([n for n, _ in ordered], fontsize=8.5, color=MUTED)
    ax.set_ylabel("p95 wait (s)", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Projected p95 wait from supplied counts", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize signal timing from a traffic-count CSV."
    )
    parser.add_argument("data", type=Path, help="counts CSV (see traffic_rl/data.py for format)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results") / "optimize")
    parser.add_argument("--no-rl", action="store_true", help="skip the RL contender")
    parser.add_argument(
        "--current-greens", type=float, nargs=2, metavar=("NS", "EW"), default=None,
        help="benchmark against the intersection's current plan (green seconds per phase)",
    )
    args = parser.parse_args()
    current = FixedTimePlan(greens=tuple(args.current_greens)) if args.current_greens else None
    optimize_from_counts(
        args.data, runs=args.runs, seed=args.seed, out_dir=args.out,
        current_plan=current, include_rl=not args.no_rl,
    )


if __name__ == "__main__":
    main()
