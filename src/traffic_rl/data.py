"""Load real traffic-count data into a demand schedule.

Expected CSV format (the shape of a standard intersection count study):

    time,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
    07:00,520,480,180,160,30,25
    08:00,610,590,210,190,40,35
    09:00,430,410,170,150,25,20

- `time` is the interval start, as HH:MM (or plain seconds).
- Vehicle columns are the counts observed DURING the interval, per approach.
- Ped columns (optional, default 0) are crossing counts per movement:
  `ped_ns` walks with NS traffic (crossing the E-W street), `ped_ew` with EW.
- Interval length is inferred from consecutive rows (a single row means 1 h),
  and counts are converted to hourly Poisson rates for the simulator.
"""

from __future__ import annotations

import csv
from pathlib import Path

from traffic_rl.config import DemandConfig, DemandSchedule

VEH_COLUMNS = ("north_veh", "south_veh", "east_veh", "west_veh")
PED_COLUMNS = ("ped_ns", "ped_ew")


def _parse_time(value: str) -> float:
    value = value.strip()
    if ":" in value:
        hours, minutes = value.split(":")
        return int(hours) * 3600.0 + int(minutes) * 60.0
    return float(value)


def _read_rows(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = [c for c in VEH_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}; expected "
            f"time[,node],{','.join(VEH_COLUMNS)}[,{','.join(PED_COLUMNS)}]"
        )
    return rows


def _schedule_from_rows(rows: list[dict], label: str) -> tuple[DemandSchedule, float]:
    starts = [_parse_time(r["time"]) for r in rows]
    if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
        raise ValueError(f"{label}: 'time' must be strictly increasing")
    # Interval lengths: gap to the next row; the final interval reuses the
    # previous length (or 1 h for a single-row file).
    lengths = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    lengths.append(lengths[-1] if lengths else 3600.0)

    schedule = []
    for row, start, length in zip(rows, starts, lengths, strict=True):
        hours = length / 3600.0
        veh = tuple(float(row[c]) / hours for c in VEH_COLUMNS)
        ped = tuple(float(row.get(c) or 0.0) / hours for c in PED_COLUMNS)
        schedule.append((start - starts[0], DemandConfig(vehicle_rates=veh, ped_rates=ped)))
    return tuple(schedule), float(sum(lengths))


def is_corridor_csv(path: Path | str) -> bool:
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    return "node" in [h.strip() for h in header]


def load_counts_csv(path: Path | str) -> tuple[DemandSchedule, float]:
    """Single intersection. Returns (schedule re-based to t=0, duration s)."""
    return _schedule_from_rows(_read_rows(path), str(path))


def load_corridor_counts_csv(path: Path | str) -> tuple[list[DemandSchedule], float, int]:
    """Corridor CSV: same columns plus `node` (0 = west-most intersection,
    increasing eastward). Every node must report the same time intervals.
    Returns (per-node schedules, duration s, n_nodes)."""
    rows = _read_rows(path)
    if "node" not in rows[0]:
        raise ValueError(f"{path}: corridor data needs a 'node' column")
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
        schedule, duration = _schedule_from_rows(by_node[i], f"{path} node {i}")
        schedules.append(schedule)
        durations.append(duration)
    return schedules, durations[0], n_nodes
