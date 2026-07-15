"""Live viewer: watch a controller run the intersection, at 1x to 1024x.

Consumes the identical IntersectionSim + Controller objects the harness uses —
same scenario registry, same dynamics, no forked logic. Rendering is decoupled
from the 1 s sim timestep by a time accumulator.

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

W, H = 860, 860
CX, CY = W // 2, H // 2 + 40
ROAD_HALF = 58  # half road width
STOP = 62  # stop-line distance from center
CAR_W, CAR_L, CAR_GAP = 12, 20, 5
MAX_DRAWN = 22

FIELD = (46, 64, 46)
ASPHALT = (58, 58, 62)
LANE_PAINT = (200, 200, 190)
CAR_COLOR = (120, 170, 235)
HUD_INK = (235, 235, 225)
HUD_DIM = (160, 160, 150)
RED, AMBER, GREEN_ON = (220, 60, 50), (240, 175, 40), (70, 200, 80)
LIGHT_OFF = (70, 70, 70)
WALK_COLOR = (245, 245, 245)

# Approach index -> (unit vector pointing AWAY from the intersection, i.e. the
# direction the queue extends), and whether the approach is vertical.
QUEUE_DIR = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}  # N, S, E, W
APPROACH_LABEL = ["N", "S", "E", "W"]


class ViewerApp:
    def __init__(self, controller_name: str, scenario: str, speed: float, seed: int):
        self.controller_name = controller_name
        self.scenario = scenario
        self.speed = speed
        self.seed = seed
        self.config = make_config(scenario)
        self.sim = IntersectionSim(self.config)
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
        # Live mean wait over departed vehicles (recomputed cheaply from the log).
        dep = np.asarray(self.sim.log.veh_depart)
        if len(dep) and len(dep) != self._n_departed:
            arr = np.asarray(self.sim.log.veh_arrival)
            done = ~np.isnan(dep)
            if done.any():
                self._mean_wait = float((dep[done] - arr[done]).mean())
            self._n_departed = len(dep)

    # ------------------------------------------------------------------ drawing

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.fill(FIELD)
        pygame.draw.rect(screen, ASPHALT, (CX - ROAD_HALF, 0, ROAD_HALF * 2, H))
        pygame.draw.rect(screen, ASPHALT, (0, CY - ROAD_HALF, W, ROAD_HALF * 2))
        for sign in (-1, 1):  # stop lines
            pygame.draw.line(
                screen, LANE_PAINT,
                (CX - ROAD_HALF, CY + sign * STOP), (CX + ROAD_HALF, CY + sign * STOP), 3,
            )
            pygame.draw.line(
                screen, LANE_PAINT,
                (CX + sign * STOP, CY - ROAD_HALF), (CX + sign * STOP, CY + ROAD_HALF), 3,
            )
        self._draw_walk(screen, font)
        for a in range(4):
            self._draw_queue(screen, font, a)
            self._draw_signal_head(screen, a)
        self._draw_hud(screen, font)

    def _lane_offset(self, a: int) -> tuple[int, int]:
        """Right-hand-traffic lane center for an approach, perpendicular to travel.

        N approaches southbound (lane west of center), S northbound (east),
        E westbound (north), W eastbound (south).
        """
        return {0: (-18, 0), 1: (18, 0), 2: (0, -18), 3: (0, 18)}[a]

    def _draw_queue(self, screen, font, a: int) -> None:
        n = len(self.sim.queues[a])
        dx, dy = QUEUE_DIR[a]
        ox, oy = self._lane_offset(a)
        vertical = dy != 0
        car_w, car_l = (CAR_W, CAR_L) if vertical else (CAR_L, CAR_W)
        for i in range(min(n, MAX_DRAWN)):
            dist = STOP + 8 + i * (CAR_L + CAR_GAP)
            px = CX + dx * dist + ox
            py = CY + dy * dist + oy
            rect = pygame.Rect(0, 0, car_w, car_l)
            rect.center = (px, py)
            pygame.draw.rect(screen, CAR_COLOR, rect, border_radius=3)
        if n > MAX_DRAWN:
            dist = STOP + 8 + MAX_DRAWN * (CAR_L + CAR_GAP) + 14
            label = font.render(f"+{n - MAX_DRAWN}", True, HUD_INK)
            screen.blit(label, label.get_rect(center=(CX + dx * dist + ox, CY + dy * dist + oy)))

    def _signal_color(self, a: int) -> tuple[int, int, int]:
        sig = self.sim.signal
        serves_me = a in PHASE_APPROACHES[sig.phase]
        if not serves_me:
            return RED
        if sig.state == SignalState.GREEN:
            return GREEN_ON
        if sig.state == SignalState.YELLOW:
            return AMBER
        return RED  # all-red

    # One corner per approach so heads never overlap: beside the stop line, on
    # the approaching driver's side of the roadway.
    HEAD_POS = {
        0: (-(ROAD_HALF + 16), -(STOP + 12)),  # N
        1: (ROAD_HALF + 16, STOP + 12),  # S
        2: (STOP + 12, -(ROAD_HALF + 16)),  # E
        3: (-(STOP + 12), ROAD_HALF + 16),  # W
    }

    def _draw_signal_head(self, screen, a: int) -> None:
        ox, oy = self.HEAD_POS[a]
        px, py = CX + ox, CY + oy
        color = self._signal_color(a)
        lit = {RED: 0, AMBER: 1, GREEN_ON: 2}[color]
        pygame.draw.rect(screen, (30, 30, 30), (px - 10, py - 26, 20, 52), border_radius=5)
        for slot, slot_color in enumerate((RED, AMBER, GREEN_ON)):
            on = slot == lit
            pygame.draw.circle(
                screen, slot_color if on else LIGHT_OFF, (px, py - 16 + slot * 16), 6
            )

    def _draw_walk(self, screen, font) -> None:
        sig = self.sim.signal
        if not sig.walk_active:
            # Waiting-ped markers at the corners of movements with pending calls.
            for m in range(2):
                if self.sim.ped_call_pending[m]:
                    px = CX - ROAD_HALF - 18 if m == 0 else CX + ROAD_HALF + 18
                    pygame.draw.circle(screen, WALK_COLOR, (px, CY - ROAD_HALF - 18), 6, 2)
            return
        m = sig.phase
        text = "WALK" if sig.in_walk_window else "CLR"
        # Movement 0 walks parallel to NS traffic (crossing the E-W street).
        if m == 0:
            rect = pygame.Rect(CX - ROAD_HALF - 14, CY - ROAD_HALF, 10, ROAD_HALF * 2)
        else:
            rect = pygame.Rect(CX - ROAD_HALF, CY - ROAD_HALF - 14, ROAD_HALF * 2, 10)
        pygame.draw.rect(screen, WALK_COLOR, rect, border_radius=3)
        label = font.render(text, True, WALK_COLOR)
        screen.blit(label, (rect.right + 8, rect.top - 4) if m == 0 else (rect.left, rect.top - 22))

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
            f"mean wait so far: {self._mean_wait:5.1f} s   "
            f"departed: {self._n_departed_done()}",
            "Space pause · +/- speed · R reset · Esc quit",
        ]
        pad, lh = 12, 22
        panel = pygame.Surface((430, pad * 2 + lh * len(lines)), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 170))
        screen.blit(panel, (10, 10))
        for i, line in enumerate(lines):
            color = HUD_DIM if i == len(lines) - 1 else HUD_INK
            screen.blit(font.render(line, True, color), (10 + pad, 10 + pad + i * lh))

    def _n_departed_done(self) -> int:
        dep = self.sim.log.veh_depart
        return sum(1 for d in dep if d == d)  # non-nan

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
