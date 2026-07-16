"""Point-queue model of one lane group: FIFO arrivals, credit-based discharge.

The queue is service-rate agnostic: each step the sim computes the group's
service (saturation flow for a protected green, gap-acceptance capacity for a
permissive left, friction-scaled flow for a shared lane) and deposits it as
credit; whole vehicles pop when credit reaches 1.
"""

from __future__ import annotations

from collections import deque


class MovementQueue:
    def __init__(self, sat_flow: float, startup_lost: float):
        self.sat_flow = sat_flow  # veh/s at saturation (all lanes of the group)
        self.startup_lost = startup_lost
        self.vehicles: deque[tuple[int, float]] = deque()  # (veh_id, arrival_t)
        self.credit = 0.0
        self.last_arrival_t = -1e9

    def add(self, veh_id: int, t: float) -> None:
        self.vehicles.append((veh_id, t))
        self.last_arrival_t = t

    def on_green_start(self) -> None:
        self.credit = 0.0

    def saturation_service(self, green_elapsed: float, dt: float, multiplier: float = 1.0) -> float:
        """Vehicles-worth of service for a protected green step: no flow during
        the first `startup_lost` seconds of green, then sat_flow (optionally
        scaled, e.g. shared-lane left-turn friction)."""
        effective = max(0.0, min(dt, green_elapsed - self.startup_lost))
        return self.sat_flow * multiplier * effective

    def discharge(self, service: float) -> list[int]:
        """Deposit `service` vehicles-worth of credit and pop whole vehicles."""
        self.credit += service
        departed: list[int] = []
        while self.credit >= 1.0 and self.vehicles:
            veh_id, _ = self.vehicles.popleft()
            departed.append(veh_id)
            self.credit -= 1.0
        if not self.vehicles:
            # Credit cannot bank while the queue is empty (no vehicle to discharge).
            self.credit = min(self.credit, 1.0)
        return departed

    def pop(self, n: int) -> list[int]:
        """Pop up to n vehicles immediately (sneakers at end of permissive green)."""
        departed = []
        for _ in range(min(n, len(self.vehicles))):
            veh_id, _ = self.vehicles.popleft()
            departed.append(veh_id)
        return departed

    def __len__(self) -> int:
        return len(self.vehicles)

    def oldest_arrival(self) -> float | None:
        return self.vehicles[0][1] if self.vehicles else None


# Backward-compatible alias (Phase 1-4 name).
ApproachQueue = MovementQueue
