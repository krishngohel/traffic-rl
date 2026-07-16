"""Signal-retiming study in a box: feed in real traffic counts, get back
optimized time-of-day signal plans and an honest projected improvement.

Pipeline:
1. Load counts (data.load_counts_csv) into a demand schedule — either legacy
   per-approach totals or full turning-movement counts (TMC).
2. With TMC data: run the protected-left warrant per approach (cross-product
   screening), build the phase table (protected / permissive lefts), and take
   clearance intervals from the ITE/MUTCD formulas over the site geometry
   (--site site.json; conservative defaults otherwise, stated in the report).
3. For each interval: compute the Webster plan from the observed flows, then
   refine with a local search over (cycle scale x through-split shift), each
   candidate scored on paired-seed simulations of that interval's demand.
4. Assemble the winners into a time-of-day ScheduledFixedTimeController.
5. Evaluate the recommended plan against baselines (naive, single Webster
   plan on the average flows, actuated, max-pressure, RL, and optionally the
   intersection's current plan) on the FULL profile with paired seeds.
6. Write report.md (including an implementable timing sheet), report.json,
   and comparison.png.

All comparisons use the same statistical rules as the main harness: per-run
p95 vehicle wait, mean +/- t-based 95% CI over paired seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_rl.config import (
    N_MOVEMENTS,
    DemandConfig,
    DemandSchedule,
    IntersectionLayout,
    LeftTurnTreatment,
    Phase,
    SignalTimingConfig,
    SimConfig,
    demand_at,
    timing_from_geometry,
)
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.controllers.fixed_time import (
    FixedTimeController,
    FixedTimePlan,
    ScheduledFixedTimeController,
)
from traffic_rl.controllers.webster import phase_floors, webster_plan
from traffic_rl.data import DEFAULT_SITE, SiteConfig, left_turn_warrant, load_counts_csv
from traffic_rl.eval.harness import run_controller, run_seeds
from traffic_rl.eval.metrics import mean_ci, paired_diff_ci

CYCLE_SCALES = (0.75, 1.0, 1.25)
SPLIT_SHIFTS = (-0.08, 0.0, 0.08)
# Webster splits minimize MEAN delay; the p95 objective often wants different
# left-phase allocations (shorter: less lost time; or longer: the left queue
# is the unluckiest movement). The search tries both directions, reallocating
# against the same street's through phase.
LEFT_SCALES = (0.7, 1.0, 1.4)
REFINE_RUNS = 5
MIN_EVAL_SECONDS = 1800.0


def build_site_layout(
    schedule: DemandSchedule, site: SiteConfig, has_tmc: bool
) -> tuple[IntersectionLayout, list[str]]:
    """Left-turn treatment per approach from the warrant + site lane config."""
    if not has_tmc:
        return IntersectionLayout(through_lanes=site.through_lanes), []
    warranted, notes = left_turn_warrant(schedule, site)
    treatments = []
    for a in range(4):
        if not site.left_bays[a]:
            treatments.append(LeftTurnTreatment.SHARED)
        elif warranted[a]:
            treatments.append(LeftTurnTreatment.PROTECTED)
        else:
            treatments.append(LeftTurnTreatment.PERMISSIVE)
    layout = IntersectionLayout(
        left_turn=tuple(treatments), through_lanes=site.through_lanes
    )
    return layout, notes


def _movement_flows(demand: DemandConfig, layout: IntersectionLayout) -> np.ndarray:
    return np.array(demand.movement_rates(layout))


def _group_sat(config: SimConfig) -> np.ndarray:
    return np.array([config.group_sat_flow(g) * 3600.0 for g in range(N_MOVEMENTS)])


def _through_phase_indices(phases: tuple[Phase, ...]) -> tuple[int, int]:
    """(NS-through, EW-through) compact indices."""
    ns = next(i for i, p in enumerate(phases) if p.slot == 1)
    ew = next(i for i, p in enumerate(phases) if p.slot == 3)
    return ns, ew


def _candidate_plans(demand: DemandConfig, config: SimConfig) -> list[FixedTimePlan]:
    """Webster base plan, then a local search: cycle scales x shifting
    effective green between the two THROUGH phases (left splits stay
    Webster-proportional, floored)."""
    timing = config.timing
    phases = config.phases
    floors = phase_floors(phases, timing)
    base = webster_plan(
        _movement_flows(demand, config.layout), _group_sat(config), timing, phases,
        green_floors=floors,
    )
    lost = timing.startup_lost
    clearance = [timing.yellow_for(p) + timing.all_red_for(p) for p in phases]
    base_greens = np.array(base.greens)
    eff = base_greens - lost
    ns_t, ew_t = _through_phase_indices(phases)
    floors_eff = np.array(floors) - lost

    left_indices = [i for i, p in enumerate(phases) if p.slot in (0, 2)]
    same_street_through = {i: (ns_t if phases[i].slot == 0 else ew_t) for i in left_indices}

    plans, seen = [], set()
    left_scales = LEFT_SCALES if left_indices else (1.0,)
    for scale in CYCLE_SCALES:
        eff_scaled = np.maximum(eff * scale, floors_eff)
        for left_scale in left_scales:
            base_g = eff_scaled.copy()
            for i in left_indices:
                shrunk = max(base_g[i] * left_scale, floors_eff[i])
                base_g[same_street_through[i]] += base_g[i] - shrunk
                base_g[i] = shrunk
            for shift in SPLIT_SHIFTS:
                g = base_g.copy()
                through_total = g[ns_t] + g[ew_t]
                delta = shift * through_total
                g[ns_t] = max(g[ns_t] + delta, floors_eff[ns_t])
                g[ew_t] = max(through_total - g[ns_t], floors_eff[ew_t])
                g[ns_t] = through_total - g[ew_t]
                cycle = float((g + lost).sum() + sum(clearance))
                if not 40.0 <= cycle <= 180.0:
                    continue
                greens = tuple(round(float(x + lost), 1) for x in g)
                if greens not in seen:
                    seen.add(greens)
                    plans.append(FixedTimePlan(greens=greens))
    # Equal splits (the naive plan) enter the tournament too: the recommended
    # plan can then never lose to naive, and short-cycle equal splits are
    # sometimes genuinely p95-optimal at protected-left sites.
    for equal in (20.0, 25.0, 30.0):
        greens = tuple(max(equal, f + lost) for f in floors_eff)
        greens = tuple(round(float(x), 1) for x in greens)
        if greens not in seen:
            seen.add(greens)
            plans.append(FixedTimePlan(greens=greens))
    return plans


def optimize_interval(
    demand: DemandConfig, base_config: SimConfig, seeds: list[int]
) -> tuple[FixedTimePlan, float]:
    """Best fixed-time plan for one interval's demand: (plan, mean per-run p95)."""
    from dataclasses import replace

    config = replace(
        base_config, demand=demand, demand_schedule=None, warmup=600.0, measured=1800.0
    )
    best_plan, best_score = None, np.inf
    for plan in _candidate_plans(demand, config):
        p95s = [
            run_controller(FixedTimeController(plan), config, seed)["p95_wait"]
            for seed in seeds
        ]
        score = float(np.mean(p95s))
        if score < best_score:
            best_plan, best_score = plan, score
    return best_plan, best_score


def _scale_plan_greens(
    plan: FixedTimePlan, config: SimConfig, which: str, factor: float
) -> FixedTimePlan:
    """Scale left-phase or all greens, respecting each phase's floor."""
    timing = config.timing
    phases = config.phases
    floors = phase_floors(phases, timing)
    greens = []
    for p, g, floor in zip(phases, plan.greens, floors, strict=True):
        is_left = p.slot in (0, 2)
        scale = factor if (which == "all" or (which == "lefts") == is_left) else 1.0
        greens.append(round(max(g * scale, floor), 1))
    return FixedTimePlan(greens=tuple(greens))


def _full_profile_tournament(
    tod_plans: list[tuple[float, FixedTimePlan]],
    config: SimConfig,
    seeds: list[int],
) -> list[tuple[float, FixedTimePlan]]:
    """Pick the best whole-day schedule by paired sims on the full profile.
    Contestants: the per-interval refine winners, variants with scaled left
    greens (carryover protection for the peak), and flat equal splits."""
    floors = phase_floors(config.phases, config.timing)
    equal = FixedTimePlan(
        greens=tuple(round(max(25.0, f), 1) for f in floors)
    )
    contestants: dict[str, list[tuple[float, FixedTimePlan]]] = {
        "refined": tod_plans,
        "refined lefts x1.25": [
            (s, _scale_plan_greens(p, config, "lefts", 1.25)) for s, p in tod_plans
        ],
        "refined all x1.15": [
            (s, _scale_plan_greens(p, config, "all", 1.15)) for s, p in tod_plans
        ],
        "equal splits": [(0.0, equal)],
    }
    best_name, best_schedule, best_score = None, None, np.inf
    for name, schedule_plans in contestants.items():
        controller = ScheduledFixedTimeController(schedule_plans)
        p95s = [run_controller(controller, config, s)["p95_wait"] for s in seeds]
        score = float(np.mean(p95s))
        if score < best_score:
            best_name, best_schedule, best_score = name, schedule_plans, score
    print(f"  full-profile tournament -> {best_name} (p95 {best_score:.1f} s)")
    return best_schedule


def _average_demand(schedule: DemandSchedule, duration: float) -> DemandConfig:
    starts = [s for s, _ in schedule] + [duration]
    veh = np.zeros(4)
    ped = np.zeros(2)
    lefts = np.zeros(4)
    rights = np.zeros(4)
    for (start, demand), end in zip(schedule, starts[1:], strict=True):
        w = (end - start) / duration
        rates = np.array(demand.vehicle_rates)
        veh += w * rates
        ped += w * np.array(demand.ped_rates)
        lefts += w * rates * np.array([t[0] for t in demand.turn_fractions])
        rights += w * rates * np.array([t[2] for t in demand.turn_fractions])
    fractions = []
    for a in range(4):
        if veh[a] > 0:
            fl, fr = lefts[a] / veh[a], rights[a] / veh[a]
            fractions.append((float(fl), float(1.0 - fl - fr), float(fr)))
        else:
            fractions.append((0.0, 1.0, 0.0))
    return DemandConfig(
        vehicle_rates=tuple(veh), ped_rates=tuple(ped), turn_fractions=tuple(fractions)
    )


def site_episode_factory(schedule: DemandSchedule, duration: float, base_config: SimConfig):
    """Training episodes sampled AROUND the site's measured pattern: a random
    30-minute window of the day, with the rates scaled and jittered so the
    policy learns the site's shape without memorizing one exact profile.
    Episodes run on the SITE's layout and timing."""
    from dataclasses import replace

    def factory(rng: np.random.Generator) -> SimConfig:
        t0 = rng.uniform(0.0, max(duration - 1800.0, 1.0))
        scale = rng.uniform(0.8, 1.25)
        jitter = rng.uniform(0.9, 1.1, size=4)
        regimes = []
        for k in range(3):
            d = demand_at(schedule, min(t0 + 600.0 * k, duration - 1.0))
            veh = tuple(float(v) * scale * j for v, j in zip(d.vehicle_rates, jitter, strict=True))
            ped = tuple(float(p) * scale for p in d.ped_rates)
            regimes.append(
                DemandConfig(vehicle_rates=veh, ped_rates=ped, turn_fractions=d.turn_fractions)
            )
        episode_schedule = tuple((600.0 * k, d) for k, d in enumerate(regimes))
        return replace(
            base_config,
            demand=regimes[0],
            demand_schedule=episode_schedule,
            warmup=1200.0,
            measured=3600.0,
        )

    return factory


def optimize_from_counts(
    csv_path: Path,
    runs: int = 10,
    seed: int = 42,
    out_dir: Path = Path("results") / "optimize",
    current_plan: FixedTimePlan | None = None,
    include_rl: bool = True,
    train_site_steps: int = 0,
    site: SiteConfig | None = None,
    site_from_file: bool = False,
) -> dict:
    schedule, duration, has_tmc = load_counts_csv(csv_path)
    if duration < MIN_EVAL_SECONDS + 900.0:
        raise ValueError(
            f"need at least {(MIN_EVAL_SECONDS + 900) / 60:.0f} min of data, got "
            f"{duration / 60:.0f} min"
        )
    site = site or DEFAULT_SITE
    layout, warrant_notes = build_site_layout(schedule, site, has_tmc)
    if has_tmc:
        timing = timing_from_geometry(site.geometry)
        geometry = site.geometry
    else:
        # Legacy through-only data: keep the Phase 1-4 default timings so old
        # studies stay reproducible.
        timing, geometry = SignalTimingConfig(), None
    base_config = SimConfig(
        demand=schedule[0][1], timing=timing, layout=layout, geometry=geometry,
        warmup=900.0, measured=duration - 900.0, demand_schedule=schedule,
    )
    phases = base_config.phases
    refine_seeds = run_seeds(seed + 1, REFINE_RUNS)

    treatments = ", ".join(
        f"{name}={layout.left_turn[a].name.lower()}"
        for a, name in enumerate(("N", "S", "E", "W"))
    )
    print(
        f"loaded {csv_path}: {len(schedule)} intervals, {duration / 3600:.1f} h"
        + (f", TMC data -> phasing: {treatments}" if has_tmc else " (through-only data)")
    )
    for note in warrant_notes:
        print(f"  protected-left warrant met — {note}")

    tod_plans: list[tuple[float, FixedTimePlan]] = []
    for start, demand in schedule:
        plan, score = optimize_interval(demand, base_config, refine_seeds)
        tod_plans.append((start, plan))
        flows = tuple(round(v) for v in demand.vehicle_rates)
        print(
            f"  interval @ {start / 3600:5.2f} h  veh/h {flows}"
            f"  -> cycle {plan.cycle_for(base_config):5.1f} s"
            f"  greens {plan.greens}  (refine p95 {score:.1f} s)"
        )

    # Interval refinement cannot see queue carryover between intervals or that
    # the day's p95 is dominated by the peak, so the assembled schedule and a
    # few systematic variants fight a final tournament on the FULL profile
    # (fresh seeds — the report's evaluation seeds stay untouched).
    tod_plans = _full_profile_tournament(
        tod_plans, base_config, run_seeds(seed + 2, REFINE_RUNS)
    )

    # Full-profile comparison, identical rules for every contender.
    config = base_config
    seeds = run_seeds(seed, runs)
    avg_flows = _average_demand(schedule, duration)
    contenders: dict[str, object] = {
        "optimized_tod": ScheduledFixedTimeController(tod_plans),
        "naive_equal_split": CONTROLLER_REGISTRY["naive"](),
        "webster_avg_flows": FixedTimeController(
            webster_plan(
                _movement_flows(avg_flows, layout), _group_sat(config), timing, phases,
                green_floors=phase_floors(phases, timing),
            )
        ),
        "actuated": CONTROLLER_REGISTRY["actuated"](),
        "max_pressure": CONTROLLER_REGISTRY["max_pressure"](),
    }
    if current_plan is not None:
        contenders["current_plan"] = FixedTimeController(current_plan)
    if include_rl:
        for name in ("rl", "rl_pattern"):
            try:
                contenders[name] = CONTROLLER_REGISTRY[name]()
            except FileNotFoundError:
                print(f"(no trained {name} weights found — skipping)")
    if train_site_steps > 0:
        from traffic_rl.rl.pattern_policy import PATTERN_WEIGHTS, PatternRLController
        from traffic_rl.rl.train import train_pattern_policy

        out_dir.mkdir(parents=True, exist_ok=True)
        site_weights = out_dir / "site_weights.npz"
        print(f"training site-specific policy ({train_site_steps:,} steps) ...")
        train_pattern_policy(
            train_site_steps, seed, site_weights,
            episode_factory=site_episode_factory(schedule, duration, base_config),
            log_prefix="  [site] ",
            init_weights=PATTERN_WEIGHTS if PATTERN_WEIGHTS.exists() else None,
        )
        contenders["rl_site_trained"] = PatternRLController(weights=site_weights)

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

    report = _build_report(
        csv_path, config, tod_plans, results, runs,
        has_tmc=has_tmc, warrant_notes=warrant_notes, site_from_file=site_from_file,
    )
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


def _build_report(
    csv_path, config: SimConfig, tod_plans, results, runs,
    has_tmc: bool = False, warrant_notes: list[str] | None = None,
    site_from_file: bool = False,
) -> dict:
    timing = config.timing
    phases = config.phases
    baseline_key = "current_plan" if "current_plan" in results else "naive_equal_split"
    base_runs = np.array(results[baseline_key]["p95_runs"])
    opt_runs = np.array(results["optimized_tod"]["p95_runs"])
    diff = paired_diff_ci(base_runs, opt_runs)
    reduction_pct = 100.0 * diff["mean"] / float(np.mean(base_runs)) if base_runs.any() else 0.0
    return {
        "source": str(csv_path),
        "n_runs": runs,
        "baseline": baseline_key,
        "has_turning_movements": has_tmc,
        "geometry_assumed_defaults": has_tmc and not site_from_file,
        "phasing": {
            "phase_names": [p.name for p in phases],
            "left_treatments": [t.name.lower() for t in config.layout.left_turn],
            "warrant_notes": warrant_notes or [],
        },
        "clearance_intervals": [
            {
                "phase": p.name,
                "yellow_s": round(timing.yellow_for(p), 1),
                "all_red_s": round(timing.all_red_for(p), 1),
                "min_green_s": round(timing.min_green_for(p), 1),
            }
            for p in phases
        ],
        "pedestrian_timing": [
            {
                "crossing": ("across EW street (with NS traffic)",
                             "across NS street (with EW traffic)")[m],
                "walk_s": timing.walk,
                "flashing_dont_walk_s": round(timing.ped_clearance_for(m), 1),
            }
            for m in range(2)
        ],
        "recommended_plans": [
            {
                "start_hour": start / 3600.0,
                "cycle_s": round(plan.cycle_for(config), 1),
                "greens_s": {p.name: g for p, g in zip(phases, plan.greens, strict=True)},
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
    phase_names = report["phasing"]["phase_names"]
    lines = [
        "# Signal timing optimization report",
        "",
        f"Source: `{report['source']}` — {report['n_runs']} paired simulation runs per contender.",
        "",
    ]
    if report["has_turning_movements"]:
        treatments = report["phasing"]["left_treatments"]
        lines += [
            "## Phasing (from turning-movement counts)",
            "",
            f"Left-turn treatment per approach (N, S, E, W): "
            f"{', '.join(treatments)}.",
            "",
        ]
        for note in report["phasing"]["warrant_notes"]:
            lines.append(f"- Protected-left warrant met — {note}")
        if report["phasing"]["warrant_notes"]:
            lines.append("")
        lines += [
            "## Clearance intervals (ITE formulas from site geometry)",
            "",
            "| phase | min green (s) | yellow (s) | all-red (s) |",
            "|---|---|---|---|",
        ]
        for c in report["clearance_intervals"]:
            lines.append(
                f"| {c['phase']} | {c['min_green_s']} | {c['yellow_s']} | {c['all_red_s']} |"
            )
        lines += [
            "",
            "| pedestrian crossing | walk (s) | flashing don't walk (s) |",
            "|---|---|---|",
        ]
        for p in report["pedestrian_timing"]:
            lines.append(
                f"| {p['crossing']} | {p['walk_s']} | {p['flashing_dont_walk_s']} |"
            )
        lines.append("")
        if report.get("geometry_assumed_defaults"):
            lines += [
                "*Geometry not supplied (`--site site.json`): clearance intervals "
                "assume 30 mph approaches and 15 m street widths — replace with "
                "surveyed values before implementation.*",
                "",
            ]
    lines += [
        "## Recommended time-of-day plans",
        "",
        "| start | cycle (s) | " + " | ".join(f"{n} green (s)" for n in phase_names) + " |",
        "|---|---|" + "---|" * len(phase_names),
    ]
    for p in report["recommended_plans"]:
        h = int(p["start_hour"])
        m = int(round((p["start_hour"] - h) * 60))
        greens = " | ".join(str(p["greens_s"][n]) for n in phase_names)
        lines.append(f"| {h:02d}:{m:02d} | {p['cycle_s']} | {greens} |")
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
    model_limits = (
        "point-queue, no RTOR, no protected+permissive (FYA) phasing, bay storage "
        "not capacity-limited"
        if report["has_turning_movements"]
        else "point-queue, no turning movements"
    )
    lines += [
        "",
        "Adaptive controllers (actuated / max-pressure / rl) are included for context: "
        "they need detection hardware, while the recommended plans are drop-in "
        "fixed-time settings. All waits are simulated from the supplied counts "
        f"under the model's stated limits ({model_limits}). Treat projections as a "
        "screening study, not a signed-off timing sheet.",
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


# --------------------------------------------------------------------- corridor

CORRIDOR_CYCLE_SCALES = (1.0, 1.2)
OFFSET_SCHEMES = ("east", "west", "zero")
CORRIDOR_REFINE_RUNS = 3
CORRIDOR_REFINE_WARMUP = 300.0
CORRIDOR_REFINE_MEASURED = 1200.0


def _corridor_offsets(scheme: str, n_nodes: int, link_travel: float, cycle: float):
    if scheme == "east":
        return tuple((i * link_travel) % cycle for i in range(n_nodes))
    if scheme == "west":
        return tuple(((n_nodes - 1 - i) * link_travel) % cycle for i in range(n_nodes))
    return tuple(0.0 for _ in range(n_nodes))


def _corridor_candidates(node_flows, timing, link_travel: float, n_nodes: int):
    """Coordinated-plan candidates: common cycle scales x progression schemes.
    The corridor model is through-only: flows are per-approach totals padded
    into the 8-group layout."""
    from traffic_rl.config import TWO_PHASE
    from traffic_rl.controllers.network import CoordinatedPlan, _plan_for_cycle

    sat = np.full(N_MOVEMENTS, 1800.0)
    padded = [
        np.concatenate([np.asarray(node_flows[i], dtype=float), np.zeros(4)])
        for i in range(n_nodes)
    ]
    base_cycles = [
        webster_plan(
            padded[i], sat, timing, TWO_PHASE,
            green_floors=phase_floors(TWO_PHASE, timing),
        ).cycle(timing.yellow, timing.all_red)
        for i in range(n_nodes)
    ]
    common_base = max(base_cycles)
    candidates = []
    for scale in CORRIDOR_CYCLE_SCALES:
        cycle = float(np.clip(common_base * scale, 40.0, 150.0))
        node_plans = tuple(
            _plan_for_cycle(padded[i], cycle, timing) for i in range(n_nodes)
        )
        for scheme in OFFSET_SCHEMES:
            offsets = _corridor_offsets(scheme, n_nodes, link_travel, cycle)
            candidates.append(CoordinatedPlan(node_plans=node_plans, offsets=offsets,
                                              scheme=scheme))
    return candidates


def optimize_corridor_interval(node_demands, timing, link_travel, seeds):
    """Best coordinated plan for one interval: (plan, mean per-run journey p95)."""
    from traffic_rl.controllers.network import ScheduledCoordinatedController
    from traffic_rl.eval.network_harness import run_network_controller
    from traffic_rl.sim.network import NetworkConfig

    n_nodes = len(node_demands)
    config = NetworkConfig(
        node_demands=tuple(node_demands), n_nodes=n_nodes, link_travel=link_travel,
        timing=timing, warmup=CORRIDOR_REFINE_WARMUP, measured=CORRIDOR_REFINE_MEASURED,
    )
    node_flows = [d.vehicle_rates for d in node_demands]
    best_plan, best_score = None, np.inf
    for plan in _corridor_candidates(node_flows, timing, link_travel, n_nodes):
        controller = ScheduledCoordinatedController([(0.0, plan)])
        p95s = [
            run_network_controller(controller, config, seed)["p95_wait"] for seed in seeds
        ]
        score = float(np.mean(p95s))
        if score < best_score:
            best_plan, best_score = plan, score
    return best_plan, best_score


def optimize_corridor_from_counts(
    csv_path: Path,
    runs: int = 10,
    seed: int = 42,
    out_dir: Path = Path("results") / "optimize",
    link_travel: float = 20.0,
    include_rl: bool = True,
) -> dict:
    from traffic_rl.controllers.network import ScheduledCoordinatedController
    from traffic_rl.data import load_corridor_counts_csv
    from traffic_rl.eval.network_harness import (
        network_controller_registry,
        run_network_controller,
    )
    from traffic_rl.sim.network import NetworkConfig

    schedules, duration, n_nodes = load_corridor_counts_csv(csv_path)
    if duration < MIN_EVAL_SECONDS + 900.0:
        raise ValueError(
            f"need at least {(MIN_EVAL_SECONDS + 900) / 60:.0f} min of data, got "
            f"{duration / 60:.0f} min"
        )
    timing = SignalTimingConfig()
    refine_seeds = run_seeds(seed + 1, CORRIDOR_REFINE_RUNS)
    print(
        f"loaded {csv_path}: corridor of {n_nodes} intersections, "
        f"{len(schedules[0])} intervals, {duration / 3600:.1f} h, "
        f"link travel {link_travel:.0f} s"
    )

    tod_plans = []
    for k, (start, _) in enumerate(schedules[0]):
        node_demands = [schedules[i][k][1] for i in range(n_nodes)]
        plan, score = optimize_corridor_interval(node_demands, timing, link_travel,
                                                 refine_seeds)
        tod_plans.append((start, plan))
        greens = "; ".join(f"{p.greens[0]:.0f}/{p.greens[1]:.0f}" for p in plan.node_plans)
        print(
            f"  interval @ {start / 3600:5.2f} h  -> common cycle "
            f"{plan.cycle(timing.yellow, timing.all_red):5.1f} s, {plan.scheme}-wave, "
            f"greens NS/EW {greens}  (refine journey p95 {score:.1f} s)"
        )

    config = NetworkConfig(
        node_demands=tuple(s[0][1] for s in schedules),
        node_schedules=tuple(schedules),
        n_nodes=n_nodes,
        link_travel=link_travel,
        timing=timing,
        warmup=900.0,
        measured=duration - 900.0,
    )
    seeds = run_seeds(seed, runs)
    registry = network_controller_registry()
    contenders: dict[str, object] = {
        "optimized_tod_coordinated": ScheduledCoordinatedController(tod_plans),
        "naive_uncoordinated": registry["naive"](),
        "greenwave_observed": registry["greenwave"](),
        "actuated": registry["actuated"](),
        "max_pressure": registry["max_pressure"](),
    }
    if include_rl:
        try:
            contenders["rl_shared"] = registry["rl"]()
        except FileNotFoundError:
            print("(no trained network RL weights found — skipping rl contender)")

    results: dict[str, dict] = {}
    for name, controller in contenders.items():
        runs_out = [run_network_controller(controller, config, s) for s in seeds]
        results[name] = {
            "p95": mean_ci(np.array([r["p95_wait"] for r in runs_out])),
            "mean": mean_ci(np.array([r["mean_wait"] for r in runs_out])),
            "ped_p95": mean_ci(np.array([r["ped_p95_wait"] for r in runs_out])),
            "p95_runs": [r["p95_wait"] for r in runs_out],
            "n_unstable": int(sum(r["unstable"] for r in runs_out)),
        }
        print(
            f"  {name:<26} journey p95 {results[name]['p95']['mean']:6.1f} s "
            f"(CI {results[name]['p95']['lo']:.1f}-{results[name]['p95']['hi']:.1f})"
        )

    y, ar = timing.yellow, timing.all_red
    baseline_key = "naive_uncoordinated"
    diff = paired_diff_ci(
        np.array(results[baseline_key]["p95_runs"]),
        np.array(results["optimized_tod_coordinated"]["p95_runs"]),
    )
    base_mean = float(np.mean(results[baseline_key]["p95_runs"]))
    report = {
        "source": str(csv_path),
        "mode": "corridor",
        "n_nodes": n_nodes,
        "link_travel_s": link_travel,
        "n_runs": runs,
        "baseline": baseline_key,
        "recommended_plans": [
            {
                "start_hour": start / 3600.0,
                "cycle_s": round(plan.cycle(y, ar), 1),
                "scheme": plan.scheme,
                "offsets_s": [round(o, 1) for o in plan.offsets],
                "node_greens_s": [list(p.greens) for p in plan.node_plans],
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
            "p95_reduction_pct": round(100.0 * diff["mean"] / base_mean, 1) if base_mean else 0,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (out_dir / "report.md").write_text(_render_corridor_markdown(report), encoding="utf-8")
    try:
        _render_chart(results, out_dir / "comparison.png")
    except ImportError:
        print("(matplotlib not installed — skipping comparison.png)")
    print(f"report -> {out_dir}\\report.md")
    return report


def _render_corridor_markdown(report: dict) -> str:
    lines = [
        "# Corridor signal timing optimization report",
        "",
        f"Source: `{report['source']}` — {report['n_nodes']} intersections, "
        f"link travel {report['link_travel_s']} s, {report['n_runs']} paired runs "
        "per contender. Waits are JOURNEY waits: the sum of a vehicle's queue "
        "waits across every signal it passes.",
        "",
        "## Recommended coordinated time-of-day plans",
        "",
        "| start | cycle (s) | wave | offsets (s) | per-node greens NS/EW (s) |",
        "|---|---|---|---|---|",
    ]
    for p in report["recommended_plans"]:
        h = int(p["start_hour"])
        m = int(round((p["start_hour"] - h) * 60))
        greens = "; ".join(f"{g[0]:.0f}/{g[1]:.0f}" for g in p["node_greens_s"])
        offsets = ", ".join(f"{o:.0f}" for o in p["offsets_s"])
        lines.append(f"| {h:02d}:{m:02d} | {p['cycle_s']} | {p['scheme']} | {offsets} | {greens} |")
    d = report["headline"]
    lines += [
        "",
        "## Headline",
        "",
        f"Coordinated time-of-day plans cut journey p95 wait by "
        f"**{d['p95_reduction_vs_baseline_s']['mean']} s "
        f"({d['p95_reduction_pct']}%)** vs `{report['baseline']}` "
        f"(paired 95% CI {d['p95_reduction_vs_baseline_s']['lo']}"
        f"–{d['p95_reduction_vs_baseline_s']['hi']} s).",
        "",
        "## Full comparison (journey p95, mean of paired runs, 95% CI)",
        "",
        "| controller | journey p95 (s) | mean wait (s) | ped p95 (s) | unstable runs |",
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
        "Model limits: point-queue, through-traffic only (no turns between the "
        "arterial and cross streets), fixed link travel time, no spillback. "
        "Treat projections as a screening study.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize signal timing from a traffic-count CSV "
        "(single intersection, or a corridor when the CSV has a 'node' column)."
    )
    parser.add_argument("data", type=Path, help="counts CSV (see traffic_rl/data.py for format)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results") / "optimize")
    parser.add_argument("--no-rl", action="store_true", help="skip the RL contender")
    parser.add_argument(
        "--link-travel", type=float, default=20.0,
        help="corridor mode: free-flow seconds between adjacent intersections",
    )
    parser.add_argument(
        "--current-greens", type=float, nargs="+", metavar="G", default=None,
        help="single-intersection mode: benchmark the current plan — one green "
        "per phase in phase-table order (2 values for a 2-phase site, up to 4 "
        "with protected lefts)",
    )
    parser.add_argument(
        "--site", type=Path, default=None,
        help="site JSON: geometry (speeds, street widths) and lane configuration "
        "for the ITE/MUTCD clearance formulas and the left-turn warrant",
    )
    parser.add_argument(
        "--train-site", type=int, nargs="?", const=600_000, default=0, metavar="STEPS",
        help="single-intersection mode: also TRAIN a site-specific ML policy on demand "
        "patterns sampled around this data (default 600k steps, ~2 min) and include it "
        "in the comparison",
    )
    args = parser.parse_args()

    from traffic_rl.data import is_corridor_csv

    if is_corridor_csv(args.data):
        if args.train_site:
            parser.error("--train-site currently supports single-intersection data only")
        optimize_corridor_from_counts(
            args.data, runs=args.runs, seed=args.seed, out_dir=args.out,
            link_travel=args.link_travel, include_rl=not args.no_rl,
        )
    else:
        from traffic_rl.data import load_site_json

        current = (
            FixedTimePlan(greens=tuple(args.current_greens)) if args.current_greens else None
        )
        site = load_site_json(args.site) if args.site else None
        optimize_from_counts(
            args.data, runs=args.runs, seed=args.seed, out_dir=args.out,
            current_plan=current, include_rl=not args.no_rl,
            train_site_steps=args.train_site,
            site=site, site_from_file=args.site is not None,
        )


if __name__ == "__main__":
    main()
