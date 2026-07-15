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


def load_counts_csv(path: Path | str) -> tuple[DemandSchedule, float]:
    """Returns (demand schedule re-based to t=0, total duration in seconds)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = [c for c in VEH_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}; expected "
            f"time,{','.join(VEH_COLUMNS)}[,{','.join(PED_COLUMNS)}]"
        )

    starts = [_parse_time(r["time"]) for r in rows]
    if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
        raise ValueError(f"{path}: 'time' must be strictly increasing")
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
