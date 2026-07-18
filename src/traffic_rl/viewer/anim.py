"""Presentation-layer car animation for the viewer.

The sim is a point-queue model — vehicles have no positions, only queue
membership and departure events. This module invents smooth, continuous motion
on top: cars spawn upstream and drive in, ease forward as their queue
advances, then follow a curved path through the intersection (left / straight
/ right) and drive off. Everything is parameterized by SIM time, so the
animation speeds up with the viewer's speed multiplier and stays deterministic.

Visual lane discipline is three lanes per approach — left-only, through,
right-only — mapped onto the sim's two lane groups (left bay, through+right).
Through-group cars are assigned straight/right cosmetically from a per-id hash
weighted by the scenario's turn fractions; the sim's FIFO discharge order is
respected exactly, whichever lane a car stands in.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import (
    N_APPROACHES,
    LeftTurnTreatment,
    SimConfig,
    left_group,
    through_group,
)

# ------------------------------------------------------------- lane geometry
LANE_W = 3.4
LANE_X = {"L": -1.7, "S": -5.1, "R": -8.5}  # local x of lane centers, approach 0
ROAD_HALF = 10.2  # 3 lanes each way
STOP = 14.0  # stop line distance from center
QUEUE_START = 16.8  # first queued car's center
CAR_SPACING = 6.6
SPAWN_DIST = 82.0
DESPAWN_DIST = 92.0

APPROACH_SPEED = 13.0  # m per sim-second
CREEP_SPEED = 4.5
CROSS_SPEED = {"L": 7.0, "S": 11.5, "R": 5.5}
EXIT_SPEED = 14.0

# Local-frame rotation per approach: approach 0 (from north, heading south)
# is the identity; the rest are rotations of it.
_ROT = {
    0: np.array([[1.0, 0.0], [0.0, 1.0]]),
    1: np.array([[-1.0, 0.0], [0.0, -1.0]]),
    2: np.array([[0.0, 1.0], [-1.0, 0.0]]),
    3: np.array([[0.0, -1.0], [1.0, 0.0]]),
}


def to_world(approach: int, x: float, y: float) -> np.ndarray:
    return _ROT[approach] @ np.array([x, y])


def _bezier_polyline(p0, p1, p2, n=24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _path_polyline(approach: int, turn: str) -> np.ndarray:
    """Local-frame polyline from the stop line through the intersection and
    out to the despawn distance. Right-hand traffic."""
    x = LANE_X[turn]
    if turn == "S":
        pts = np.array([[x, STOP], [x, -DESPAWN_DIST]])
    elif turn == "L":
        # Heading south, turning left -> exits east in the inner eastbound lane.
        arc = _bezier_polyline(
            np.array([x, STOP]), np.array([x, -1.7]), np.array([STOP, -1.7])
        )
        pts = np.vstack([arc, [DESPAWN_DIST, -1.7]])
    else:
        # Turning right -> exits west in the curb westbound lane.
        arc = _bezier_polyline(
            np.array([x, STOP]), np.array([x, 8.5]), np.array([-STOP, 8.5]), n=16
        )
        pts = np.vstack([arc, [-DESPAWN_DIST, 8.5]])
    return np.array([to_world(approach, px, py) for px, py in pts])


class _Path:
    def __init__(self, pts: np.ndarray):
        self.pts = pts
        seg = np.diff(pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])

    def at(self, dist: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, unit heading) at arc distance."""
        d = min(max(dist, 0.0), self.length - 1e-6)
        i = int(np.searchsorted(self.cum, d, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        frac = (d - self.cum[i]) / max(self.seg_len[i], 1e-9)
        pos = self.pts[i] + frac * (self.pts[i + 1] - self.pts[i])
        heading = (self.pts[i + 1] - self.pts[i]) / max(self.seg_len[i], 1e-9)
        return pos, heading


# Precomputed crossing paths for every (approach, turn).
_PATHS = {(a, t): _Path(_path_polyline(a, t)) for a in range(4) for t in "LSR"}

SPAWNING, QUEUED, CROSSING = 0, 1, 2


class AnimCar:
    __slots__ = ("veh_id", "approach", "turn", "state", "dist", "pos", "heading",
                 "target_dist", "cross_t0", "speed", "accel", "pitch", "braking")

    def __init__(self, veh_id: int, approach: int, turn: str):
        self.veh_id = veh_id
        self.approach = approach
        self.turn = turn
        self.state = SPAWNING
        # While inbound, `dist` is distance from intersection center along the
        # approach (decreasing toward QUEUE_START); while crossing it is arc
        # distance along the crossing path.
        self.dist = SPAWN_DIST
        self.target_dist = QUEUE_START
        self.pos = to_world(approach, LANE_X[turn], SPAWN_DIST)
        self.heading = to_world(approach, 0.0, -1.0)
        self.cross_t0 = 0.0
        self.speed = APPROACH_SPEED
        self.accel = 0.0
        self.pitch = 0.0  # body dive (+nose down) under braking, squat under power
        self.braking = False

    def start_crossing(self) -> None:
        self.state = CROSSING
        self.dist = 0.0
        self.pos, self.heading = _PATHS[(self.approach, self.turn)].at(0.0)

    def _dynamics(self, new_speed: float, dt: float, stopped: bool) -> None:
        if dt > 1e-9:
            self.accel = (new_speed - self.speed) / dt
        self.speed = new_speed
        # Drivers hold the brake at a red; the body dives with decel, squats
        # with accel, and settles quickly (smoothed toward the target).
        self.braking = stopped or self.accel < -0.6
        target = float(np.clip(-self.accel * 0.010, -0.045, 0.065))
        if stopped:
            target = 0.0
        self.pitch += min(1.0, 6.0 * dt) * (target - self.pitch)

    def update(self, dt: float) -> bool:
        """Advance by dt sim-seconds. Returns False when the car despawns."""
        if self.state == CROSSING:
            path = _PATHS[(self.approach, self.turn)]
            frac = self.dist / max(path.length, 1e-9)
            v = CROSS_SPEED[self.turn] + (EXIT_SPEED - CROSS_SPEED[self.turn]) * min(
                1.0, frac * 1.6
            )
            self.dist += v * dt
            if self.dist >= path.length:
                return False
            self.pos, self.heading = path.at(self.dist)
            self._dynamics(v, dt, stopped=False)
            return True
        # Inbound: proportional ease toward the target slot — approach fast,
        # brake into the queue, creep as the queue advances.
        gap = self.dist - self.target_dist
        v = 0.0
        if gap > 0.05:
            v = min(APPROACH_SPEED, max(CREEP_SPEED, gap * 0.9))
            self.dist = max(self.target_dist, self.dist - v * dt)
        self.pos = to_world(self.approach, LANE_X[self.turn], self.dist)
        if self.state == SPAWNING and abs(self.dist - self.target_dist) < 0.1:
            self.state = QUEUED
        self._dynamics(v, dt, stopped=v < 0.05)
        return True

    def blinker_on(self, anim_t: float) -> bool:
        """Turn-signal flash: left/right turners blink while queued and
        through the first two-thirds of the crossing, at ~1.5 Hz."""
        if self.turn == "S":
            return False
        if self.state == CROSSING:
            path = _PATHS[(self.approach, self.turn)]
            if self.dist / max(path.length, 1e-9) > 0.66:
                return False
        return (anim_t * 1.5) % 1.0 < 0.55


class CarAnimator:
    """Owns every animated car; syncs to the sim after each step."""

    MAX_TRACKED_PER_LANE = 28

    def __init__(self, config: SimConfig):
        self.config = config
        self.cars: dict[int, AnimCar] = {}
        self._n_seen_arrivals = 0
        self._crossing: list[AnimCar] = []
        # Per-approach probability that a through-group car turns right, and
        # (shared-lane sites) turns left, from the scenario's turn fractions.
        demand, layout = config.demand, config.layout
        self._p_right = np.zeros(N_APPROACHES)
        self._p_left_shared = np.zeros(N_APPROACHES)
        for a in range(N_APPROACHES):
            fl, ft, fr = demand.turn_fractions[a]
            shared = layout.left_turn[a] == LeftTurnTreatment.SHARED
            in_group = ft + fr + (fl if shared else 0.0)
            if in_group > 1e-9:
                self._p_right[a] = fr / in_group
                self._p_left_shared[a] = (fl / in_group) if shared else 0.0

    def reset(self) -> None:
        self.cars.clear()
        self._crossing.clear()
        self._n_seen_arrivals = 0

    def _assign_turn(self, veh_id: int, group: int) -> str:
        if group >= N_APPROACHES:
            return "L"  # dedicated left bay
        a = group
        u = (((veh_id * 2654435761) & 0xFFFFFFFF) >> 8) / float(1 << 24)
        if u < self._p_left_shared[a]:
            return "L"
        if u < self._p_left_shared[a] + self._p_right[a]:
            return "R"
        return "S"

    def on_sim_step(self, sim) -> None:
        log = sim.log
        # New arrivals -> spawn cars driving in (skip deep-overflow spawns).
        n = len(log.veh_arrival)
        for veh_id in range(self._n_seen_arrivals, n):
            g = log.veh_group[veh_id]
            approach = g % N_APPROACHES
            turn = self._assign_turn(veh_id, g)
            lane_count = sum(
                1 for c in self.cars.values()
                if c.state != CROSSING and c.approach == approach and c.turn == turn
            )
            if lane_count < self.MAX_TRACKED_PER_LANE:
                self.cars[veh_id] = AnimCar(veh_id, approach, turn)
        self._n_seen_arrivals = n
        # Departures -> start the crossing animation.
        for veh_id, car in list(self.cars.items()):
            if car.state != CROSSING and log.veh_depart[veh_id] == log.veh_depart[veh_id]:
                car.start_crossing()
        # Re-slot every inbound car: FIFO order within its visual lane.
        lanes: dict[tuple[int, str], list[AnimCar]] = {}
        for veh_id in sorted(self.cars):
            car = self.cars[veh_id]
            if car.state != CROSSING:
                lanes.setdefault((car.approach, car.turn), []).append(car)
        for cars in lanes.values():
            for i, car in enumerate(cars):
                car.target_dist = QUEUE_START + i * CAR_SPACING

    def update(self, dt: float) -> None:
        gone = [veh_id for veh_id, car in self.cars.items() if not car.update(dt)]
        for veh_id in gone:
            del self.cars[veh_id]

    def overflow(self, sim) -> dict[int, int]:
        """Per-approach count of queued vehicles beyond what is animated."""
        out = {}
        for a in range(N_APPROACHES):
            queued = len(sim.queues[through_group(a)]) + len(sim.queues[left_group(a)])
            tracked = sum(
                1 for c in self.cars.values()
                if c.approach == a and c.state != CROSSING
            )
            if queued > tracked:
                out[a] = queued - tracked
        return out
