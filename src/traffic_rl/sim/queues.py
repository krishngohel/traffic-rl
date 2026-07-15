"""Point-queue model of one approach: FIFO arrivals, saturation-flow discharge."""

from __future__ import annotations

from collections import deque


class ApproachQueue:
    def __init__(self, sat_flow: float, startup_lost: float):
        self.sat_flow = sat_flow
        self.startup_lost = startup_lost
        self.vehicles: deque[tuple[int, float]] = deque()  # (veh_id, arrival_t)
        self.credit = 0.0
        self.last_arrival_t = -1e9

    def add(self, veh_id: int, t: float) -> None:
        self.vehicles.append((veh_id, t))
        self.last_arrival_t = t

    def on_green_start(self) -> None:
        self.credit = 0.0

    def discharge(self, green_elapsed: float, dt: float) -> list[int]:
        """Pop vehicles that cross the stop line during this green step.

        `green_elapsed` is the green time elapsed at the END of the step. No flow
        during the first `startup_lost` seconds of green, then `sat_flow` veh/s.
        """
        effective = max(0.0, min(dt, green_elapsed - self.startup_lost))
        self.credit += self.sat_flow * effective
        departed: list[int] = []
        while self.credit >= 1.0 and self.vehicles:
            veh_id, _ = self.vehicles.popleft()
            departed.append(veh_id)
            self.credit -= 1.0
        if not self.vehicles:
            # Credit cannot bank while the queue is empty (no vehicle to discharge).
            self.credit = min(self.credit, 1.0)
        return departed

    def __len__(self) -> int:
        return len(self.vehicles)

    def oldest_arrival(self) -> float | None:
        return self.vehicles[0][1] if self.vehicles else None
