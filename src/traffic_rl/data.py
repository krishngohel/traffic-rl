"""Load real traffic-count data into a demand schedule.

Two accepted CSV shapes:

1. Legacy per-approach totals (through-only, the Phase 3 format):

    time,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
    07:00,520,480,180,160,30,25

2. Turning-movement counts (TMC — the standard intersection count study):

    time,north_left,north_thru,north_right,south_left,...,west_right,ped_ns,ped_ew
    07:00,95,380,45,90,360,40,...

- `time` is the interval start, as HH:MM (or plain seconds).
- Counts are what was observed DURING the interval, per approach movement
  (approach = the arm the vehicle enters from: `north_*` is traffic on the
  north leg heading south).
- Ped columns (optional, default 0) are crossing counts per movement:
  `ped_ns` walks with NS traffic (crossing the E-W street), `ped_ew` with EW.
- Interval length is inferred from consecutive rows (a single row means 1 h),
  and counts are converted to hourly Poisson rates for the simulator.

An optional site JSON (--site site.json) provides geometry for the ITE/MUTCD
timing formulas and lane configuration:

    {
      "ns_speed_mph": 45, "ew_speed_mph": 30,
      "ns_street_width_m": 22, "ew_street_width_m": 14,
      "through_lanes": [2, 2, 1, 1],
      "left_bays": [true, true, true, true]
    }
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from traffic_rl.config import (
    APPROACH_NAMES,
    DemandConfig,
    DemandSchedule,
    GeometryConfig,
)

VEH_COLUMNS = ("north_veh", "south_veh", "east_veh", "west_veh")
PED_COLUMNS = ("ped_ns", "ped_ew")
_APPROACH_PREFIXES = ("north", "south", "east", "west")
TMC_COLUMNS = tuple(
    f"{prefix}_{move}" for prefix in _APPROACH_PREFIXES for move in ("left", "thru", "right")
)


@dataclass(frozen=True)
class SiteConfig:
    """Optional geometry / lane configuration accompanying a count study."""

    geometry: GeometryConfig
    through_lanes: tuple[int, int, int, int] = (1, 1, 1, 1)
    left_bays: tuple[bool, bool, bool, bool] = (True, True, True, True)


DEFAULT_SITE = SiteConfig(geometry=GeometryConfig())


def load_site_json(path: Path | str) -> SiteConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    geo_keys = (
        "ns_speed_mph", "ew_speed_mph", "ns_street_width_m", "ew_street_width_m", "grade",
    )
    geometry = GeometryConfig(**{k: raw[k] for k in geo_keys if k in raw})
    return SiteConfig(
        geometry=geometry,
        through_lanes=tuple(raw.get("through_lanes", (1, 1, 1, 1))),
        left_bays=tuple(raw.get("left_bays", (True, True, True, True))),
    )


def _parse_time(value: str) -> float:
    value = value.strip()
    if ":" in value:
        hours, minutes = value.split(":")
        return int(hours) * 3600.0 + int(minutes) * 60.0
    return float(value)


def _read_rows(path: Path | str) -> tuple[list[dict], bool]:
    """Returns (rows, has_turning_movements)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    has_tmc = all(c in rows[0] for c in TMC_COLUMNS)
    has_legacy = all(c in rows[0] for c in VEH_COLUMNS)
    if not (has_tmc or has_legacy):
        raise ValueError(
            f"{path}: need either per-approach totals ({', '.join(VEH_COLUMNS)}) or "
            f"turning-movement columns ({_APPROACH_PREFIXES[0]}_left/thru/right, ...); "
            f"got {sorted(rows[0])}"
        )
    return rows, has_tmc


def _demand_from_row(row: dict, hours: float, has_tmc: bool) -> DemandConfig:
    ped = tuple(float(row.get(c) or 0.0) / hours for c in PED_COLUMNS)
    if not has_tmc:
        veh = tuple(float(row[c]) / hours for c in VEH_COLUMNS)
        return DemandConfig(vehicle_rates=veh, ped_rates=ped)
    totals, fractions = [], []
    for prefix in _APPROACH_PREFIXES:
        left = float(row[f"{prefix}_left"]) / hours
        thru = float(row[f"{prefix}_thru"]) / hours
        right = float(row[f"{prefix}_right"]) / hours
        total = left + thru + right
        totals.append(total)
        if total > 0:
            fractions.append((left / total, thru / total, right / total))
        else:
            fractions.append((0.0, 1.0, 0.0))
    return DemandConfig(
        vehicle_rates=tuple(totals), ped_rates=ped, turn_fractions=tuple(fractions)
    )


def _schedule_from_rows(
    rows: list[dict], label: str, has_tmc: bool
) -> tuple[DemandSchedule, float]:
    starts = [_parse_time(r["time"]) for r in rows]
    if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
        raise ValueError(f"{label}: 'time' must be strictly increasing")
    # Interval lengths: gap to the next row; the final interval reuses the
    # previous length (or 1 h for a single-row file).
    lengths = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    lengths.append(lengths[-1] if lengths else 3600.0)

    schedule = []
    for row, start, length in zip(rows, starts, lengths, strict=True):
        demand = _demand_from_row(row, length / 3600.0, has_tmc)
        schedule.append((start - starts[0], demand))
    return tuple(schedule), float(sum(lengths))


def is_corridor_csv(path: Path | str) -> bool:
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    return "node" in [h.strip() for h in header]


def load_counts_csv(path: Path | str) -> tuple[DemandSchedule, float, bool]:
    """Single intersection. Returns (schedule re-based to t=0, duration s,
    has_turning_movements)."""
    rows, has_tmc = _read_rows(path)
    schedule, duration = _schedule_from_rows(rows, str(path), has_tmc)
    return schedule, duration, has_tmc


def load_corridor_counts_csv(path: Path | str) -> tuple[list[DemandSchedule], float, int]:
    """Corridor CSV: same columns plus `node` (0 = west-most intersection,
    increasing eastward). Every node must report the same time intervals.
    Turning columns are accepted but folded into approach totals — the
    corridor model is through-only (stated limit).
    Returns (per-node schedules, duration s, n_nodes)."""
    rows, has_tmc = _read_rows(path)
    if "node" not in rows[0]:
        raise ValueError(f"{path}: corridor data needs a 'node' column")
    if has_tmc:
        print(
            "note: corridor mode folds turning counts into approach totals "
            "(the network model is through-only)"
        )
    by_node: dict[int, list[dict]] = {}
    for row in rows:
        by_node.setdefault(int(row["node"]), []).append(row)
    n_nodes = len(by_node)
    if sorted(by_node) != list(range(n_nodes)):
        raise ValueError(f"{path}: 'node' must cover 0..{n_nodes - 1}, got {sorted(by_node)}")
    times = [tuple(r["time"] for r in by_node[i]) for i in range(n_nodes)]
    if any(t != times[0] for t in times[1:]):
        raise ValueError(f"{path}: every node must report the same time intervals")
    schedules, durations = [], []
    for i in range(n_nodes):
        schedule, duration = _schedule_from_rows(by_node[i], f"{path} node {i}", has_tmc)
        if has_tmc:  # fold turns into totals for the through-only network model
            schedule = tuple(
                (
                    start,
                    DemandConfig(vehicle_rates=d.vehicle_rates, ped_rates=d.ped_rates),
                )
                for start, d in schedule
            )
        schedules.append(schedule)
        durations.append(duration)
    return schedules, durations[0], n_nodes


# ------------------------------------------------------------------ warrants

LEFT_VOLUME_WARRANT_VEH_H = 240.0
CROSS_PRODUCT_WARRANT_1LANE = 50_000.0
CROSS_PRODUCT_WARRANT_2LANE = 100_000.0


def left_turn_warrant(
    schedule: DemandSchedule,
    site: SiteConfig,
) -> tuple[tuple[bool, bool, bool, bool], list[str]]:
    """Protected-left screening per approach: warranted if ANY interval has
    left volume >= 240 veh/h, or left x opposing-through cross product >=
    50 000 (one opposing through lane; 100 000 for 2+). Returns (per-approach
    warranted flags, human-readable trigger notes)."""
    from traffic_rl.config import OPPOSING

    warranted = [False] * 4
    notes: list[str] = []
    for start, demand in schedule:
        for a in range(4):
            if warranted[a]:
                continue
            left = demand.vehicle_rates[a] * demand.turn_fractions[a][0]
            opp = OPPOSING[a]
            opp_thru = demand.vehicle_rates[opp] * (
                demand.turn_fractions[opp][1] + demand.turn_fractions[opp][2]
            )
            threshold = (
                CROSS_PRODUCT_WARRANT_2LANE
                if site.through_lanes[opp] >= 2
                else CROSS_PRODUCT_WARRANT_1LANE
            )
            if left >= LEFT_VOLUME_WARRANT_VEH_H:
                warranted[a] = True
                notes.append(
                    f"{APPROACH_NAMES[a]}: left volume {left:.0f} veh/h >= "
                    f"{LEFT_VOLUME_WARRANT_VEH_H:.0f} at {start / 3600:.2f} h"
                )
            elif left * opp_thru >= threshold:
                warranted[a] = True
                notes.append(
                    f"{APPROACH_NAMES[a]}: cross product {left:.0f} x {opp_thru:.0f} = "
                    f"{left * opp_thru:,.0f} >= {threshold:,.0f} at {start / 3600:.2f} h"
                )
    return tuple(warranted), notes
