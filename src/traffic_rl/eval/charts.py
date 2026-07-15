"""Chart set rendered from a results directory produced by the harness.

Four PNGs: the headline p95 bars (with 95% CIs and "≥" censoring annotations),
pooled wait ECDFs, queue-length time series (mean ± 1σ band), and pedestrian
p95 bars (no winning by starving pedestrians).

Colors are a CVD-validated categorical palette assigned to controllers in fixed
order (color follows the entity, never its rank); sub-3:1 slots are relieved by
direct value labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTROLLER_ORDER = ["naive", "webster", "actuated", "max_pressure", "rl"]
LABELS = {
    "naive": "Naive 50/50",
    "webster": "Webster (1958)",
    "actuated": "Actuated",
    "max_pressure": "Max-pressure",
    "rl": "RL (DQN)",
}
SERIES = {  # validated categorical palette, fixed slot order
    "naive": "#2a78d6",
    "webster": "#1baf7a",
    "actuated": "#eda100",
    "max_pressure": "#008300",
    "rl": "#4a3aa7",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _load(results: Path):
    scenarios = sorted(
        (p.name for p in results.iterdir() if (p / "summary.json").exists()),
        key=lambda s: ["symmetric", "asymmetric", "heavy"].index(s)
        if s in ("symmetric", "asymmetric", "heavy")
        else 99,
    )
    data = {}
    for s in scenarios:
        with open(results / s / "summary.json", encoding="utf-8") as f:
            data[s] = json.load(f)
    return scenarios, data


def _controllers_in(summary: dict) -> list[str]:
    present = list(summary["controllers"])
    return [c for c in CONTROLLER_ORDER if c in present] + [
        c for c in present if c not in CONTROLLER_ORDER
    ]


def _bar_panel(ax, summary: dict, metric: str, flag_key: str | None):
    controllers = _controllers_in(summary)
    for i, c in enumerate(controllers):
        stats = summary["controllers"][c][metric]
        color = SERIES.get(c, MUTED)
        err = [[stats["mean"] - stats["lo"]], [stats["hi"] - stats["mean"]]]
        ax.bar(i, stats["mean"], width=0.55, color=color, zorder=3)
        ax.errorbar(i, stats["mean"], yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1, zorder=4)
        bound = "≥ " if flag_key and summary["controllers"][c].get(flag_key) else ""
        ax.annotate(
            f"{bound}{stats['mean']:.0f}",
            (i, stats["hi"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels([LABELS.get(c, c) for c in controllers], fontsize=8.5)
    ax.set_title(summary["scenario"], fontsize=11, color=INK, pad=10)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)


def chart_p95_bar(scenarios, data, out: Path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.6), sharey=False)
    for ax, s in zip(np.atleast_1d(axes), scenarios, strict=True):
        _bar_panel(ax, data[s], "p95_wait", "p95_any_lower_bound")
        ax.set_ylabel("p95 wait (s)" if s == scenarios[0] else "")
    fig.suptitle(
        "The unluckiest 1-in-20 driver: p95 wait, mean of 20 paired runs (95% CI)",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "p95_bar.png", dpi=160)
    plt.close(fig)


def chart_ped_wait_bar(scenarios, data, out: Path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.6))
    for ax, s in zip(np.atleast_1d(axes), scenarios, strict=True):
        _bar_panel(ax, data[s], "ped_p95_wait", None)
        ax.set_ylabel("pedestrian p95 wait (s)" if s == scenarios[0] else "")
    fig.suptitle(
        "Pedestrians are not sacrificed: ped p95 wait, mean of 20 runs (95% CI)",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "ped_wait_bar.png", dpi=160)
    plt.close(fig)


def chart_wait_ecdf(scenarios, data, results: Path, out: Path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.6), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), scenarios, strict=True):
        for c in _controllers_in(data[s]):
            waits = np.load(results / s / c / "waits.npz")["veh_waits"]
            waits = np.sort(np.maximum(waits, 0.5))
            y = np.arange(1, len(waits) + 1) / len(waits)
            ax.plot(waits, y, color=SERIES.get(c, MUTED), lw=2, label=LABELS.get(c, c))
        ax.set_xscale("log")
        ax.set_xlabel("wait (s, log)")
        ax.set_title(s, fontsize=11, color=INK)
        ax.axhline(0.95, color=BASELINE, lw=1, ls=":")
    np.atleast_1d(axes)[0].set_ylabel("fraction of drivers")
    np.atleast_1d(axes)[-1].annotate(
        "p95", (0.98, 0.95), xycoords=("axes fraction", "data"),
        fontsize=8.5, color=MUTED, va="bottom", ha="right",
    )
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Wait distribution, all 20 runs pooled (illustrative — CIs live on per-run p95)",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "wait_ecdf.png", dpi=160)
    plt.close(fig)


def chart_queue_timeseries(scenarios, data, results: Path, out: Path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.6), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), scenarios, strict=True):
        for c in _controllers_in(data[s]):
            q = np.load(results / s / c / "queues.npz")["queue_timeseries"]
            # 60 s rolling mean: removes within-cycle sawtooth, keeps the trend.
            kernel = np.ones(60) / 60.0
            q = np.apply_along_axis(
                lambda r, k=kernel: np.convolve(r, k, mode="valid"), 1, q
            )
            t = (np.arange(q.shape[1]) + 60 + data[s]["config"]["warmup"]) / 60.0
            mean, std = q.mean(axis=0), q.std(axis=0)
            color = SERIES.get(c, MUTED)
            ax.plot(t, mean, color=color, lw=2, label=LABELS.get(c, c))
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.set_xlabel("sim time (min)")
        ax.set_title(s, fontsize=11, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("vehicles queued (60 s rolling mean ± 1σ)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Queue growth over the measured hour, 20 runs", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "queue_timeseries.png", dpi=160)
    plt.close(fig)


def make_charts(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    scenarios, data = _load(results)
    chart_p95_bar(scenarios, data, out)
    chart_wait_ecdf(scenarios, data, results, out)
    chart_queue_timeseries(scenarios, data, results, out)
    chart_ped_wait_bar(scenarios, data, out)
    return [out / n for n in
            ("p95_bar.png", "wait_ecdf.png", "queue_timeseries.png", "ped_wait_bar.png")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render charts from harness results.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / "charts"
    for p in make_charts(args.results, out):
        print(p)


if __name__ == "__main__":
    main()
