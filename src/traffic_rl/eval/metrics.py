"""Single source of truth for wait definitions, warm-up, censoring, and statistics.

Definitions (applied identically to every controller):
- Vehicle wait = depart_time - arrival_time (point-queue: no free-flow travel to
  subtract). Pedestrian wait = walk_start - arrival.
- Only arrivals inside the measured window [warmup, horizon) are scored.
- Vehicles/peds still waiting at the horizon are censored: included with the
  lower-bound wait (horizon - arrival) and counted. If more than
  CENSORED_FRACTION_LIMIT of scored vehicles are censored, the run's p95 is only
  a lower bound ("≥") and the run is flagged unstable — censoring would otherwise
  silently *deflate* the p95 of exactly the worst controllers.
- Headline statistic: per-run p95, aggregated across runs with a t-based 95% CI
  (runs are the i.i.d. units; a pooled percentile has no valid interval).
- Superiority claims use paired per-seed differences (common random numbers).
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import SimConfig

CENSORED_FRACTION_LIMIT = 0.05
UNSTABLE_FINAL_QUEUE_RATIO = 3.0

# Two-sided 95% Student-t critical values by degrees of freedom.
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000,
}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    candidates = [k for k in _T_TABLE if k <= df]
    return _T_TABLE[max(candidates)] if candidates else _T_TABLE[1]


def compute_run_metrics(log_arrays: dict[str, np.ndarray], config: SimConfig) -> dict:
    """Metrics for one finished run, from a finalized EventLog dict."""
    lo, hi = config.warmup, config.horizon

    arr, dep = log_arrays["veh_arrival"], log_arrays["veh_depart"]
    scored = (arr >= lo) & (arr < hi)
    a, d = arr[scored], dep[scored]
    censored = np.isnan(d)
    waits = np.where(censored, hi - a, d - a)

    p_arr, p_ws = log_arrays["ped_arrival"], log_arrays["ped_walk_start"]
    p_scored = (p_arr >= lo) & (p_arr < hi)
    pa, pw = p_arr[p_scored], p_ws[p_scored]
    p_censored = np.isnan(pw)
    ped_waits = np.where(p_censored, hi - pa, pw - pa)

    in_window = (log_arrays["step_t"] >= lo) & (log_arrays["step_t"] < hi)
    total_queue = log_arrays["step_queues"][in_window].sum(axis=1)

    n = len(waits)
    censored_fraction = float(censored.mean()) if n else 0.0
    mean_queue = float(total_queue.mean()) if len(total_queue) else 0.0
    final_queue = float(total_queue[-1]) if len(total_queue) else 0.0
    unstable = censored_fraction > CENSORED_FRACTION_LIMIT or (
        mean_queue > 0 and final_queue > UNSTABLE_FINAL_QUEUE_RATIO * mean_queue
    )

    return {
        "n_vehicles": n,
        "p95_wait": float(np.percentile(waits, 95)) if n else 0.0,
        "mean_wait": float(waits.mean()) if n else 0.0,
        "max_wait": float(waits.max()) if n else 0.0,
        "throughput_veh_per_h": float((~censored).sum() / config.measured * 3600.0),
        "mean_queue": mean_queue,
        "max_queue": float(total_queue.max()) if len(total_queue) else 0.0,
        "final_queue": final_queue,
        "n_censored": int(censored.sum()),
        "censored_fraction": censored_fraction,
        "p95_is_lower_bound": censored_fraction > CENSORED_FRACTION_LIMIT,
        "unstable": bool(unstable),
        "n_peds": int(len(ped_waits)),
        "ped_p95_wait": float(np.percentile(ped_waits, 95)) if len(ped_waits) else 0.0,
        "ped_mean_wait": float(ped_waits.mean()) if len(ped_waits) else 0.0,
        "veh_waits": waits,  # arrays for pooled outputs; stripped before CSV
        "ped_waits": ped_waits,
        "queue_timeseries": total_queue,
    }


def mean_ci(values: np.ndarray) -> dict:
    """Mean with a two-sided t-based 95% CI, treating each value as one i.i.d. run."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    m = float(v.mean()) if n else 0.0
    if n < 2:
        return {"mean": m, "lo": m, "hi": m, "std": 0.0, "n": n}
    half = t_crit(n - 1) * v.std(ddof=1) / np.sqrt(n)
    return {"mean": m, "lo": m - half, "hi": m + half, "std": float(v.std(ddof=1)), "n": n}


def paired_diff_ci(baseline: np.ndarray, other: np.ndarray) -> dict:
    """CI on mean(baseline - other) over paired seeds. Positive = other is better."""
    return mean_ci(np.asarray(baseline) - np.asarray(other))
