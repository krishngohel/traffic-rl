"""Live 3D viewer: watch a controller run the intersection, at 1x to 1024x.

Software-projected 3D rendered with pygame-ce polygons (painter's algorithm,
sun-shaded box geometry) — no extra dependencies beyond the [viewer] extra.
Vehicles get a deterministic body style (sedan / SUV / pickup / van) and color
from their id. Consumes the identical IntersectionSim + Controller objects the
harness uses; rendering is decoupled from the 1 s sim timestep by an
accumulator.

Keys: Space pause · +/- speed · R reset with next seed · Esc quit.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The viewer needs pygame-ce. Install it with: pip install traffic-rl[viewer]"
    ) from e

from traffic_rl.config import PHASE_APPROACHES
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.scenarios import SCENARIOS, make_config
from traffic_rl.sim.core import IntersectionSim
from traffic_rl.sim.signal import SignalState

W, H = 1024, 800

# ----------------------------------------------------------------- world layout
# Meters. x east, y north, z up. Intersection center at origin.
ROAD_HALF = 8.0  # two 4 m lanes per road
STOP = 12.0  # stop-line distance from center
LANE = 4.0  # lane-center offset from road centerline
CAR_SPACING = 6.5
QUEUE_START = 15.5  # first queued car's center distance from intersection center
CROSSWALK_MID = 10.4  # crosswalk band center distance
MAX_DRAWN_FAR, MAX_DRAWN_NEAR = 18, 8  # N/S extend away from camera; E toward it

# Approach: (queue direction away from center, lane offset) — right-hand traffic.
# N southbound (lane west), S northbound (east), E westbound (north), W eastbound.
QUEUE_DIR = {0: (0.0, 1.0), 1: (0.0, -1.0), 2: (1.0, 0.0), 3: (-1.0, 0.0)}
LANE_OFF = {0: (-LANE, 0.0), 1: (LANE, 0.0), 2: (0.0, LANE), 3: (0.0, -LANE)}
MAX_DRAWN = {0: MAX_DRAWN_FAR, 1: MAX_DRAWN_FAR, 2: MAX_DRAWN_NEAR, 3: MAX_DRAWN_FAR}

# ----------------------------------------------------------------------- colors
GRASS = (74, 106, 66)
ASPHALT = (52, 52, 56)
PAINT = (222, 222, 214)
CENTERLINE = (208, 170, 60)
WALK_TINT = (140, 225, 150)
POLE_COLOR = (70, 72, 76)
HEAD_COLOR = (38, 38, 40)
PED_COLOR = (235, 220, 190)
HUD_INK = (235, 235, 225)
HUD_DIM = (160, 160, 150)
RED, AMBER, GREEN_ON = (225, 60, 50), (240, 175, 40), (80, 210, 90)
LIGHT_OFF = (66, 66, 66)

CAR_COLORS = [
    (178, 40, 52),  # crimson
    (58, 104, 188),  # blue
    (188, 192, 198),  # silver
    (236, 234, 226),  # white
    (56, 58, 62),  # charcoal
    (52, 122, 74),  # forest
    (222, 122, 44),  # orange
    (46, 148, 150),  # teal
]

# Car body styles: list of (offset along heading, size (length, width, height),
# base z, is_cabin). Cabins render as tinted glass (darkened body color).
CAR_MODELS = {
    "sedan": [
        ((0.0, 0.0), (4.4, 1.9, 1.0), 0.25, False),
        ((-0.35, 0.0), (2.3, 1.7, 0.75), 1.25, True),
    ],
    "suv": [
        ((0.0, 0.0), (4.6, 2.0, 1.45), 0.3, False),
        ((-0.2, 0.0), (2.7, 1.85, 0.8), 1.75, True),
    ],
    "pickup": [
        ((1.05, 0.0), (2.0, 2.0, 1.7), 0.3, False),
        ((-1.35, 0.0), (2.7, 2.0, 0.95), 0.3, False),
    ],
    "van": [((0.0, 0.0), (5.0, 2.0, 2.1), 0.3, False)],
}
MODEL_NAMES = list(CAR_MODELS)

SUN = np.array([-0.40, 0.30, -0.87])
SUN = SUN / np.linalg.norm(SUN)

# Six faces of a unit box as vertex-index quads + outward normals.
_BOX_CORNERS = np.array(
    [[sx, sy, sz] for sz in (0, 1) for sy in (-0.5, 0.5) for sx in (-0.5, 0.5)]
)
_BOX_FACES = [
    ((0, 1, 3, 2), (0, 0, -1)),
    ((4, 5, 7, 6), (0, 0, 1)),
    ((0, 1, 5, 4), (0, -1, 0)),
    ((2, 3, 7, 6), (0, 1, 0)),
    ((1, 3, 7, 5), (1, 0, 0)),
    ((0, 2, 6, 4), (-1, 0, 0)),
]


class Camera:
    """Elevated three-quarter perspective view from the east."""

    def __init__(self, eye=(95.0, -15.0, 65.0), target=(0.0, 5.0, 0.0), focal=1000.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.focal = focal
        fwd = np.asarray(target) - self.eye
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, np.array([0.0, 0.0, 1.0]))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n,3) world points -> (n,2) screen coords + (n,) camera depths."""
        v = np.atleast_2d(pts) - self.eye
        d = np.maximum(v @ self.fwd, 0.5)
        sx = W / 2 + self.focal * (v @ self.right) / d
        sy = H / 2 - self.focal * (v @ self.up) / d
        return np.stack([sx, sy], axis=1), d


def _shade(color, normal) -> tuple[int, int, int]:
    b = 0.42 + 0.58 * max(0.0, float(np.dot(normal, -SUN)))
    return tuple(min(255, int(c * b)) for c in color)


def box_faces(center, size, color, cam: Camera, out: list) -> None:
    """Append (depth, screen_quad, shaded_color) faces of an axis-aligned box."""
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    corners = _BOX_CORNERS * size + center  # z of `center` is the box BASE
    scr, _ = cam.project(corners)
    for idx, n in _BOX_FACES:
        normal = np.asarray(n, dtype=np.float64)
        face_center = corners[list(idx)].mean(axis=0)
        if np.dot(face_center - cam.eye, normal) >= 0:
            continue  # backface
        depth = float(np.linalg.norm(face_center - cam.eye))
        out.append((depth, scr[list(idx)], _shade(color, normal)))


def _flat_quad(cam: Camera, screen, corners_xy, color, z=0.02) -> None:
    pts = np.array([[x, y, z] for x, y in corners_xy])
    scr, _ = cam.project(pts)
    pygame.draw.polygon(screen, color, [tuple(p) for p in scr])


def _car_style(veh_id: int) -> tuple[str, tuple[int, int, int]]:
    h = (veh_id * 2654435761) & 0xFFFFFFFF
    return MODEL_NAMES[(h >> 4) % len(MODEL_NAMES)], CAR_COLORS[(h >> 9) % len(CAR_COLORS)]


class ViewerApp:
    def __init__(self, controller_name: str, scenario: str, speed: float, seed: int):
        self.controller_name = controller_name
        self.scenario = scenario
        self.speed = speed
        self.seed = seed
        self.config = make_config(scenario)
        self.sim = IntersectionSim(self.config)
        self.cam = Camera()
        self.paused = False
        self._reset()

    def _reset(self) -> None:
        self.controller = CONTROLLER_REGISTRY[self.controller_name]()
        self.obs = self.sim.reset(self.seed)
        self.controller.reset(self.config, np.random.default_rng(self.seed))
        self._acc = 0.0
        self._mean_wait = 0.0
        self._n_departed = 0

    def _advance(self, frame_dt: float) -> None:
        self._acc += self.speed * frame_dt
        dt = self.config.dt
        while self._acc >= dt:
            self.obs = self.sim.step(self.controller.act(self.obs)).obs
            self._acc -= dt
        dep = np.asarray(self.sim.log.veh_depart)
        if len(dep) and len(dep) != self._n_departed:
            arr = np.asarray(self.sim.log.veh_arrival)
            done = ~np.isnan(dep)
            if done.any():
                self._mean_wait = float((dep[done] - arr[done]).mean())
            self._n_departed = len(dep)

    # ------------------------------------------------------------------ drawing

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.fill(GRASS)
        self._draw_ground(screen)
        faces: list = []
        overflow_labels: list[tuple[np.ndarray, str]] = []
        for a in range(4):
            self._queue_faces(a, faces, overflow_labels)
            self._signal_pole_faces(a, faces)
        self._ped_faces(faces)
        faces.sort(key=lambda f: -f[0])  # painter: far to near
        for _, quad, color in faces:
            pygame.draw.polygon(screen, color, [tuple(p) for p in quad])
        self._draw_signal_lamps(screen)
        for pos, text in overflow_labels:
            label = font.render(text, True, HUD_INK)
            screen.blit(label, label.get_rect(center=(int(pos[0]), int(pos[1]))))
        self._draw_hud(screen, font)

    def _draw_ground(self, screen) -> None:
        cam = self.cam
        # Roads (extents chosen to stay in front of the camera).
        _flat_quad(cam, screen, [(-ROAD_HALF, -90), (ROAD_HALF, -90), (ROAD_HALF, 140),
                                 (-ROAD_HALF, 140)], ASPHALT, z=0.0)
        _flat_quad(cam, screen, [(-100, -ROAD_HALF), (70, -ROAD_HALF), (70, ROAD_HALF),
                                 (-100, ROAD_HALF)], ASPHALT, z=0.0)
        # Center lines, interrupted at the intersection box.
        for y0, y1 in ((-90, -ROAD_HALF), (ROAD_HALF, 140)):
            _flat_quad(cam, screen, [(-0.15, y0), (0.15, y0), (0.15, y1), (-0.15, y1)],
                       CENTERLINE)
        for x0, x1 in ((-100, -ROAD_HALF), (ROAD_HALF, 70)):
            _flat_quad(cam, screen, [(x0, -0.15), (x1, -0.15), (x1, 0.15), (x0, 0.15)],
                       CENTERLINE)
        # Stop lines across the approach lane only.
        _flat_quad(cam, screen, [(-ROAD_HALF, STOP), (-0.3, STOP), (-0.3, STOP + 0.6),
                                 (-ROAD_HALF, STOP + 0.6)], PAINT)  # N
        _flat_quad(cam, screen, [(0.3, -STOP - 0.6), (ROAD_HALF, -STOP - 0.6),
                                 (ROAD_HALF, -STOP), (0.3, -STOP)], PAINT)  # S
        _flat_quad(cam, screen, [(STOP, 0.3), (STOP + 0.6, 0.3), (STOP + 0.6, ROAD_HALF),
                                 (STOP, ROAD_HALF)], PAINT)  # E
        _flat_quad(cam, screen, [(-STOP - 0.6, -ROAD_HALF), (-STOP, -ROAD_HALF),
                                 (-STOP, -0.3), (-STOP - 0.6, -0.3)], PAINT)  # W
        self._draw_crosswalks(screen)

    def _crosswalk_active(self, movement: int) -> bool:
        sig = self.sim.signal
        return sig.walk_active and sig.phase == movement

    def _draw_crosswalks(self, screen) -> None:
        cam = self.cam
        half_band = 1.2
        # Movement 0: walks parallel to NS traffic, crossing the EW street at
        # x = ±CROSSWALK_MID; zebra bars are long in x, repeating along y.
        color0 = WALK_TINT if self._crosswalk_active(0) else PAINT
        for side in (-1, 1):
            x0 = side * CROSSWALK_MID - half_band
            for yb in np.arange(-ROAD_HALF + 0.4, ROAD_HALF - 0.4, 1.4):
                _flat_quad(cam, screen, [(x0, yb), (x0 + 2 * half_band, yb),
                                         (x0 + 2 * half_band, yb + 0.7), (x0, yb + 0.7)],
                           color0)
        # Movement 1: crossing the NS street at y = ±CROSSWALK_MID.
        color1 = WALK_TINT if self._crosswalk_active(1) else PAINT
        for side in (-1, 1):
            y0 = side * CROSSWALK_MID - half_band
            for xb in np.arange(-ROAD_HALF + 0.4, ROAD_HALF - 0.4, 1.4):
                _flat_quad(cam, screen, [(xb, y0), (xb + 0.7, y0),
                                         (xb + 0.7, y0 + 2 * half_band), (xb, y0 + 2 * half_band)],
                           color1)

    def _queue_faces(self, a: int, faces: list, overflow_labels: list) -> None:
        n = len(self.sim.queues[a])
        if n == 0:
            return
        dx, dy = QUEUE_DIR[a]
        ox, oy = LANE_OFF[a]
        vertical = dy != 0.0
        drawn = min(n, MAX_DRAWN[a])
        for i, (veh_id, _) in enumerate(list(self.sim.queues[a].vehicles)[:drawn]):
            dist = QUEUE_START + i * CAR_SPACING
            px, py = dx * dist + ox, dy * dist + oy
            model, color = _car_style(veh_id)
            glass = tuple(int(c * 0.45) for c in color)
            for (along, across), (length, width, height), z0, is_cabin in CAR_MODELS[model]:
                cx = px + (dx * along if vertical else along if dx > 0 else -along)
                cy = py + (dy * along if vertical else across)
                if vertical:
                    cx += across
                    size = (width, length, height)
                else:
                    size = (length, width, height)
                box_faces((cx, cy, z0), size, glass if is_cabin else color, self.cam, faces)
        if n > drawn:
            dist = QUEUE_START + drawn * CAR_SPACING + 4.0
            pos, _ = self.cam.project(np.array([[dx * dist + ox, dy * dist + oy, 1.5]]))
            overflow_labels.append((pos[0], f"+{n - drawn}"))

    # Pole beside each stop line on the approaching driver's side.
    POLE_POS = {
        0: (-ROAD_HALF - 1.5, STOP + 0.8),
        1: (ROAD_HALF + 1.5, -STOP - 0.8),
        2: (STOP + 0.8, ROAD_HALF + 1.5),
        3: (-STOP - 0.8, -ROAD_HALF - 1.5),
    }

    def _signal_pole_faces(self, a: int, faces: list) -> None:
        px, py = self.POLE_POS[a]
        box_faces((px, py, 0.0), (0.3, 0.3, 4.4), POLE_COLOR, self.cam, faces)
        vertical = a in (0, 1)
        head_size = (1.0, 0.5, 1.9) if vertical else (0.5, 1.0, 1.9)
        box_faces((px, py, 4.4), head_size, HEAD_COLOR, self.cam, faces)

    def _signal_color(self, a: int) -> tuple[int, int, int]:
        sig = self.sim.signal
        if a not in PHASE_APPROACHES[sig.phase]:
            return RED
        if sig.state == SignalState.GREEN:
            return GREEN_ON
        if sig.state == SignalState.YELLOW:
            return AMBER
        return RED

    def _draw_signal_lamps(self, screen) -> None:
        # Lamps drawn after the sorted faces: heads sit at 4.4-6.3 m, above
        # every car model, so they are never meaningfully occluded.
        for a in range(4):
            px, py = self.POLE_POS[a]
            # Lamps sit on the head face toward the approaching driver, which is
            # the side pointing away from the intersection (the queue direction).
            fx, fy = QUEUE_DIR[a]
            color = self._signal_color(a)
            lit = {RED: 0, AMBER: 1, GREEN_ON: 2}[color]
            centers = np.array(
                [[px + fx * 0.27, py + fy * 0.27, 5.85 - slot * 0.62] for slot in range(3)]
            )
            scr, depth = self.cam.project(centers)
            for slot in range(3):
                r = max(2, int(0.30 * self.cam.focal / depth[slot]))
                lamp = (RED, AMBER, GREEN_ON)[slot] if slot == lit else LIGHT_OFF
                pygame.draw.circle(screen, lamp, (int(scr[slot][0]), int(scr[slot][1])), r)

    def _ped_faces(self, faces: list) -> None:
        sig = self.sim.signal
        # Waiting pedestrians cluster at the corners of their crossing.
        for m in range(2):
            count = min(len(self.sim.waiting_peds[m]), 4)
            for i in range(count):
                if m == 0:
                    px, py = CROSSWALK_MID + 0.4 + 0.9 * i, -ROAD_HALF - 1.6
                else:
                    px, py = ROAD_HALF + 1.6 + 0.9 * i, CROSSWALK_MID + 0.4
                self._figure(px, py, faces)
        # Walkers cross during the walk window.
        if sig.walk_active and sig.in_walk_window:
            m = sig.phase
            frac = min(1.0, sig.walk_elapsed / max(sig.timing.walk, 1e-6))
            span = 2 * (ROAD_HALF + 1.2)
            for side in (-1, 1):
                progress = -ROAD_HALF - 1.2 + frac * span
                progress *= side  # opposite directions on the two crossings
                if m == 0:
                    self._figure(side * CROSSWALK_MID, progress, faces)
                else:
                    self._figure(progress, side * CROSSWALK_MID, faces)

    def _figure(self, px: float, py: float, faces: list) -> None:
        box_faces((px, py, 0.0), (0.5, 0.5, 1.25), PED_COLOR, self.cam, faces)
        box_faces((px, py, 1.3), (0.34, 0.34, 0.34), (90, 70, 58), self.cam, faces)

    def _draw_hud(self, screen, font) -> None:
        sig = self.sim.signal
        q = [len(qu) for qu in self.sim.queues]
        lines = [
            f"{self.controller_name}  ·  {self.scenario}  ·  seed {self.seed}",
            f"t = {self.sim.t:7.0f} s   speed {self.speed:.0f}x"
            + ("   PAUSED" if self.paused else ""),
            f"phase {'NS' if sig.phase == 0 else 'EW'} {sig.state.name}"
            f"  ({sig.state_elapsed:.0f} s in state)",
            f"queues  N {q[0]:>3}  S {q[1]:>3}  E {q[2]:>3}  W {q[3]:>3}",
            f"mean wait so far: {self._mean_wait:5.1f} s   departed: {self._n_departed_done()}",
            "Space pause · +/- speed · R reset · Esc quit",
        ]
        pad, lh = 12, 22
        panel = pygame.Surface((440, pad * 2 + lh * len(lines)), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 175))
        screen.blit(panel, (10, 10))
        for i, line in enumerate(lines):
            color = HUD_DIM if i == len(lines) - 1 else HUD_INK
            screen.blit(font.render(line, True, color), (10 + pad, 10 + pad + i * lh))

    def _n_departed_done(self) -> int:
        return sum(1 for d in self.sim.log.veh_depart if d == d)  # non-nan

    # ------------------------------------------------------------------ loop

    def run(self, smoke_frames: int | None = None) -> None:
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("traffic-rl")
        font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        clock = pygame.time.Clock()
        frames = 0
        running = True
        while running:
            frame_dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.speed = min(self.speed * 2, 1024.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.speed = max(self.speed / 2, 1.0)
                    elif event.key == pygame.K_r:
                        self.seed += 1
                        self._reset()
            if not self.paused:
                self._advance(min(frame_dt, 0.1))
            self.draw(screen, font)
            pygame.display.flip()
            frames += 1
            if smoke_frames is not None and frames >= smoke_frames:
                running = False
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a controller run the intersection.")
    parser.add_argument("--controller", default="actuated", choices=sorted(CONTROLLER_REGISTRY))
    parser.add_argument("--scenario", default="asymmetric", choices=sorted(SCENARIOS))
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", type=int, default=None, metavar="FRAMES",
        help="render N frames on a dummy display and exit (CI/verification)",
    )
    args = parser.parse_args()
    if args.smoke is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    app = ViewerApp(args.controller, args.scenario, args.speed, args.seed)
    app.run(smoke_frames=args.smoke)
    if args.smoke is not None:
        print(f"smoke ok: {args.smoke} frames, sim t = {app.sim.t:.0f} s")


if __name__ == "__main__":
    main()
