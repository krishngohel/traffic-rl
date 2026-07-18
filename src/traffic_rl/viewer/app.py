"""Live 3D viewer: watch a controller run the intersection, at 1x to 1024x.

Software-projected 3D rendered with pygame-ce polygons (painter's algorithm,
sun-shaded box geometry) — no extra dependencies beyond the [viewer] extra.
Cars are animated continuously by viewer/anim.py: they drive in from upstream,
brake into the queue, creep forward as it advances, then follow curved paths
through the intersection (left / straight / right) and drive off. Approaches
render three lanes — left-only, through, right-only — with painted arrows,
and protected-left approaches get a second signal head with arrow lamps.

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

from traffic_rl.config import LeftTurnTreatment, left_group, through_group
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.scenarios import SCENARIOS, make_config
from traffic_rl.sim.core import IntersectionSim
from traffic_rl.sim.signal import SignalState
from traffic_rl.viewer.anim import LANE_X, ROAD_HALF, STOP, CarAnimator, to_world

W, H = 1120, 840

CROSSWALK_MID = 11.9  # crosswalk band center distance
SIDEWALK_IN, SIDEWALK_OUT = ROAD_HALF + 0.15, ROAD_HALF + 2.6

# ----------------------------------------------------------------------- colors
GRASS = (86, 118, 74)
GRASS_DARK = (76, 106, 66)
ASPHALT = (54, 54, 58)
ASPHALT_SHADOW = (38, 38, 42)
PAINT = (225, 225, 217)
LANE_DASH = (200, 200, 192)
CENTERLINE = (212, 172, 60)
WALK_TINT = (140, 225, 150)
SIDEWALK = (152, 148, 140)
POLE_COLOR = (72, 74, 78)
HEAD_COLOR = (34, 34, 36)
PED_COLOR = (235, 220, 190)
HUD_INK = (235, 235, 225)
HUD_DIM = (160, 160, 150)
RED, AMBER, GREEN_ON = (228, 60, 50), (242, 178, 40), (84, 214, 92)
LIGHT_OFF = (60, 60, 62)
TRUNK = (96, 72, 52)
FOLIAGE = [(58, 96, 52), (66, 108, 56), (52, 88, 60)]
HEADLIGHT, TAILLIGHT = (240, 236, 200), (188, 42, 36)
TIRE = (28, 28, 30)

CAR_COLORS = [
    (172, 38, 50),   # crimson
    (56, 100, 182),  # blue
    (186, 190, 196), # silver
    (238, 236, 228), # white
    (52, 54, 58),    # charcoal
    (48, 118, 70),   # forest
    (218, 118, 42),  # orange
    (44, 144, 146),  # teal
    (94, 70, 132),   # plum
    (208, 176, 60),  # gold
]

# Car body styles: boxes are ((offset along heading, offset across),
# (L, W, H), base z, kind) with kind "body" | "glass" | "trim" (dark plastic:
# bumpers, roof rails). Wheels, hubs, mirrors, and light strips are added
# procedurally from the footprint; "mirror_along" places the side mirrors.
CAR_MODELS = {
    "sedan": {
        "boxes": [
            ((0.0, 0.0), (4.5, 1.88, 0.60), 0.36, "body"),
            ((1.35, 0.0), (1.75, 1.80, 0.26), 0.96, "body"),   # hood
            ((-1.55, 0.0), (1.35, 1.80, 0.30), 0.96, "body"),  # trunk
            ((-0.15, 0.0), (2.15, 1.66, 0.62), 0.96, "glass"),  # greenhouse
            ((-0.15, 0.0), (1.55, 1.44, 0.16), 1.58, "body"),  # roof cap
            ((2.28, 0.0), (0.24, 1.86, 0.34), 0.34, "trim"),   # bumpers
            ((-2.28, 0.0), (0.24, 1.86, 0.34), 0.34, "trim"),
        ],
        "foot": (4.5, 1.88),
        "mirror_along": 0.85,
    },
    "suv": {
        "boxes": [
            ((0.0, 0.0), (4.7, 1.98, 0.80), 0.42, "body"),
            ((1.65, 0.0), (1.35, 1.90, 0.30), 1.22, "body"),   # hood
            ((-0.45, 0.0), (2.85, 1.80, 0.72), 1.22, "glass"),
            ((-0.45, 0.0), (2.35, 1.62, 0.16), 1.94, "body"),  # roof
            ((-0.45, 0.62), (2.05, 0.10, 0.10), 2.10, "trim"),  # roof rails
            ((-0.45, -0.62), (2.05, 0.10, 0.10), 2.10, "trim"),
            ((2.40, 0.0), (0.24, 1.94, 0.40), 0.36, "trim"),
            ((-2.40, 0.0), (0.24, 1.94, 0.40), 0.36, "trim"),
        ],
        "foot": (4.7, 1.98),
        "mirror_along": 1.05,
    },
    "pickup": {
        "boxes": [
            ((-1.30, 0.0), (2.35, 2.0, 0.85), 0.40, "body"),    # bed walls
            ((-1.30, 0.0), (2.05, 1.70, 0.18), 0.62, "trim"),   # bed floor
            ((0.75, 0.0), (2.55, 2.0, 0.90), 0.40, "body"),     # cab base
            ((1.75, 0.0), (0.85, 1.92, 0.28), 1.30, "body"),    # hood nose
            ((0.55, 0.0), (1.55, 1.82, 0.68), 1.30, "glass"),
            ((0.55, 0.0), (1.15, 1.60, 0.14), 1.98, "body"),
            ((2.45, 0.0), (0.26, 1.96, 0.42), 0.34, "trim"),
            ((-2.55, 0.0), (0.26, 1.96, 0.42), 0.34, "trim"),
        ],
        "foot": (5.1, 2.0),
        "mirror_along": 1.25,
    },
    "van": {
        "boxes": [
            ((0.0, 0.0), (5.1, 2.02, 1.10), 0.40, "body"),
            ((2.05, 0.0), (0.95, 1.94, 0.55), 1.50, "glass"),   # windshield
            ((-0.35, 0.0), (3.35, 1.94, 0.85), 1.50, "body"),   # cargo top
            ((-0.35, 0.0), (3.35, 1.94, 0.14), 2.35, "body"),
            ((2.60, 0.0), (0.24, 1.98, 0.42), 0.34, "trim"),
            ((-2.60, 0.0), (0.24, 1.98, 0.42), 0.34, "trim"),
        ],
        "foot": (5.1, 2.02),
        "mirror_along": 1.95,
    },
    "coupe": {
        "boxes": [
            ((0.0, 0.0), (4.2, 1.84, 0.52), 0.34, "body"),
            ((1.30, 0.0), (1.60, 1.76, 0.22), 0.86, "body"),    # long hood
            ((-0.55, 0.0), (1.75, 1.58, 0.52), 0.86, "glass"),  # fastback
            ((-0.55, 0.0), (1.05, 1.36, 0.13), 1.38, "body"),
            ((2.12, 0.0), (0.22, 1.82, 0.30), 0.32, "trim"),
            ((-2.12, 0.0), (0.22, 1.82, 0.30), 0.32, "trim"),
        ],
        "foot": (4.2, 1.84),
        "mirror_along": 0.65,
    },
    "hatch": {
        "boxes": [
            ((0.1, 0.0), (3.7, 1.78, 0.58), 0.36, "body"),
            ((1.25, 0.0), (1.15, 1.70, 0.24), 0.94, "body"),    # stub hood
            ((-0.35, 0.0), (2.15, 1.62, 0.62), 0.94, "glass"),  # tall glass
            ((-0.35, 0.0), (1.75, 1.42, 0.15), 1.56, "body"),
            ((1.92, 0.0), (0.22, 1.76, 0.32), 0.34, "trim"),
            ((-1.72, 0.0), (0.22, 1.76, 0.32), 0.34, "trim"),
        ],
        "foot": (3.7, 1.78),
        "mirror_along": 0.65,
    },
}
MODEL_NAMES = list(CAR_MODELS)
TRIM_COLOR = (40, 40, 44)
HUB_COLOR = (168, 170, 176)
GLASS_TINT = (72, 96, 118)  # cool blue-gray, blended with body color

SUN = np.array([-0.40, 0.30, -0.87])
SUN = SUN / np.linalg.norm(SUN)

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

    def __init__(self, eye=(108.0, -22.0, 74.0), target=(0.0, 4.0, 0.0), focal=1080.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.focal = focal
        fwd = np.asarray(target) - self.eye
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, np.array([0.0, 0.0, 1.0]))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.atleast_2d(pts) - self.eye
        d = np.maximum(v @ self.fwd, 0.5)
        sx = W / 2 + self.focal * (v @ self.right) / d
        sy = H / 2 - self.focal * (v @ self.up) / d
        return np.stack([sx, sy], axis=1), d


_FACE_IDX = np.array([list(idx) for idx, _ in _BOX_FACES])  # (6, 4)
_FACE_NORMALS = np.array([n for _, n in _BOX_FACES], dtype=np.float64)  # (6, 3)


FOG_COLOR = np.array([118.0, 138.0, 112.0])  # distant haze blends toward grass


class BoxBatch:
    """Collects yawed boxes and projects/shades them all in one numpy pass —
    a software 'shader': lambert diffuse + Blinn-style sun specular per
    material, ground-proximity ambient occlusion, and distance haze."""

    def __init__(self):
        self.centers: list = []
        self.sizes: list = []
        self.yaws: list = []
        self.colors: list = []
        self.specs: list = []

    def add(self, center, size, color, yaw: float = 0.0, spec: float = 0.0) -> None:
        """spec is the material's specular strength: 0 matte (foliage,
        asphalt-side boxes), ~0.3 car paint, ~0.9 glass and chrome."""
        self.centers.append(center)
        self.sizes.append(size)
        self.yaws.append(yaw)
        self.colors.append(color)
        self.specs.append(spec)

    def flush(self, cam: Camera, out: list) -> None:
        if not self.centers:
            return
        centers = np.asarray(self.centers, dtype=np.float64)  # (n, 3)
        sizes = np.asarray(self.sizes, dtype=np.float64)
        yaws = np.asarray(self.yaws, dtype=np.float64)
        colors = np.asarray(self.colors, dtype=np.float64)
        specs = np.asarray(self.specs, dtype=np.float64)
        n = len(centers)
        corners = sizes[:, None, :] * _BOX_CORNERS[None, :, :]  # (n, 8, 3)
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        x, y = corners[:, :, 0].copy(), corners[:, :, 1].copy()
        corners[:, :, 0] = x * c - y * s
        corners[:, :, 1] = x * s + y * c
        corners += centers[:, None, :]
        scr, _ = cam.project(corners.reshape(-1, 3))
        scr = scr.reshape(n, 8, 2)
        # Rotate the xy-plane face normals per box (z normals are unchanged).
        normals = np.broadcast_to(_FACE_NORMALS, (n, 6, 3)).copy()
        nx, ny = normals[:, :, 0].copy(), normals[:, :, 1].copy()
        normals[:, :, 0] = nx * c - ny * s
        normals[:, :, 1] = nx * s + ny * c
        face_centers = corners[:, _FACE_IDX, :].mean(axis=2)  # (n, 6, 3)
        view = face_centers - cam.eye
        visible = np.einsum("nfk,nfk->nf", view, normals) < 0.0
        depths = np.linalg.norm(view, axis=2)

        # Diffuse + fake ambient occlusion (faces hugging the ground darken).
        ndotl = np.maximum(0.0, normals @ -SUN)  # (n, 6)
        bright = 0.38 + 0.62 * ndotl
        bright *= 0.80 + 0.20 * np.clip(face_centers[:, :, 2] / 1.6, 0.0, 1.0)
        shaded = colors[:, None, :] * bright[:, :, None]
        # Sun specular: mirror of the light about the normal, dotted with the
        # view direction — glass and chrome catch bright glints.
        if specs.any():
            refl = 2.0 * ndotl[:, :, None] * normals - (-SUN)
            vdir = -view / depths[:, :, None]
            glint = np.clip(np.einsum("nfk,nfk->nf", refl, vdir), 0.0, 1.0) ** 22
            shaded += (glint * specs[:, None] * 235.0)[:, :, None]
        # Distance haze pulls far geometry toward the horizon tone.
        f = np.clip((depths - 110.0) / 260.0, 0.0, 0.45)[:, :, None]
        shaded = shaded * (1.0 - f) + FOG_COLOR * f
        shaded = np.clip(shaded, 0.0, 255.0).astype(np.uint8)
        for i in range(n):
            for face in range(6):
                if visible[i, face]:
                    out.append(
                        (
                            float(depths[i, face]),
                            scr[i, _FACE_IDX[face]],
                            tuple(shaded[i, face]),
                        )
                    )
        self.centers.clear()
        self.sizes.clear()
        self.yaws.clear()
        self.colors.clear()
        self.specs.clear()


def _flat_quad(cam: Camera, screen, corners_xy, color, z=0.02) -> None:
    pts = np.array([[x, y, z] for x, y in corners_xy])
    scr, _ = cam.project(pts)
    pygame.draw.polygon(screen, color, [tuple(p) for p in scr])


def _flat_poly(cam: Camera, screen, pts_xy, color, z=0.03) -> None:
    pts = np.array([[x, y, z] for x, y in pts_xy])
    scr, _ = cam.project(pts)
    pygame.draw.polygon(screen, color, [tuple(p) for p in scr])


def _car_style(veh_id: int) -> tuple[str, tuple[int, int, int]]:
    h = (veh_id * 2654435761) & 0xFFFFFFFF
    return MODEL_NAMES[(h >> 4) % len(MODEL_NAMES)], CAR_COLORS[(h >> 9) % len(CAR_COLORS)]


ROLLING_WINDOW = 3600.0  # seconds of recent departures for the rolling mean wait
AUTOSAVE_EVERY = 100_000  # learner steps between weight autosaves

# Fixed scenery positions (world meters), clear of roads and sidewalks.
TREES = [
    (-20, 20, 1.00), (-34, 30, 0.85), (-48, 19, 1.1), (-22, 46, 0.9),
    (20, 24, 0.95), (33, 38, 1.05), (-26, -22, 1.0), (-44, -30, 0.9),
    (22, -20, 0.85), (38, -27, 1.0), (-58, -18, 0.95), (52, 20, 0.9),
]
BUILDINGS = [
    ((-40, 44), (17, 13, 8.5), (168, 152, 132)),
    ((-62, 26), (12, 11, 6.0), (150, 140, 128)),
    ((34, 52), (15, 12, 7.0), (160, 146, 138)),
    ((-38, -44), (14, 12, 6.5), (156, 148, 130)),
]


class ViewerApp:
    def __init__(
        self,
        controller_name: str,
        scenario: str,
        speed: float,
        seed: int,
        learn: bool = False,
        learn_fresh: bool = False,
        learn_out=None,
    ):
        self.controller_name = controller_name
        self.scenario = scenario
        self.speed = speed
        self.seed = seed
        self.config = make_config(scenario)
        self.sim = IntersectionSim(self.config)
        self.cam = Camera()
        self.paused = False
        self._ground_cache: dict = {}
        self.animator = CarAnimator(self.config)
        self.learner = None
        self.learn_out = None
        if learn:
            from traffic_rl.rl.online import DEFAULT_ONLINE_OUT, OnlineLearner

            self.learner = OnlineLearner(seed, fresh=learn_fresh)
            self.learn_out = learn_out or DEFAULT_ONLINE_OUT
            self._last_autosave = 0
        self._reset()

    def _reset(self) -> None:
        self.controller = (
            None if self.learner else CONTROLLER_REGISTRY[self.controller_name]()
        )
        self.obs = self.sim.reset(self.seed)
        if self.learner:
            self.learner.on_reset()
        else:
            self.controller.reset(self.config, np.random.default_rng(self.seed))
        self.animator.reset()
        self._acc = 0.0
        self._anim_t = self.sim.t
        self._mean_wait = 0.0
        self._rolling_wait = 0.0
        self._n_departed = 0
        # Per-phase green timings shown at each pole: planned greens for
        # fixed-time controllers, measured last-completed green otherwise.
        self._phase_green: dict[int, float] = {}
        self._green_planned = False
        self._prev_sig: tuple | None = None
        self._seed_plan_greens()

    def _seed_plan_greens(self) -> None:
        from traffic_rl.controllers.fixed_time import FixedTimeController, naive_plan

        # Only a static fixed-time plan is known ahead of time; every other
        # controller (actuated, RL, scheduled TOD) shows measured greens.
        if isinstance(self.controller, FixedTimeController):
            plan = self.controller.plan or naive_plan(self.config)
            self._phase_green = dict(enumerate(plan.greens))
            self._green_planned = True

    def _track_green_times(self) -> None:
        """Record each phase's completed green duration from the state machine
        (the ground truth for every controller, adaptive ones included)."""
        sig = self.sim.signal
        if self._prev_sig is not None:
            p_phase, p_state, p_elapsed = self._prev_sig
            ended = p_state == SignalState.GREEN and (
                sig.state != SignalState.GREEN or sig.phase != p_phase
            )
            if ended and not self._green_planned:
                self._phase_green[p_phase] = p_elapsed + self.config.dt
        self._prev_sig = (sig.phase, sig.state, sig.state_elapsed)

    def _advance(self, frame_dt: float) -> None:
        self._acc += self.speed * frame_dt
        dt = self.config.dt
        while self._acc >= dt:
            if self.learner:
                result = self.sim.step(self.learner.act(self.obs))
                self.learner.observe(result)
                self.obs = result.obs
            else:
                self.obs = self.sim.step(self.controller.act(self.obs)).obs
            self.animator.on_sim_step(self.sim)
            self._track_green_times()
            self._acc -= dt
        # Continuous animation clock: sim time plus the un-stepped remainder.
        new_anim_t = self.sim.t + self._acc
        self.animator.update(max(0.0, new_anim_t - self._anim_t))
        self._anim_t = new_anim_t
        if self.learner and self.learner.steps - self._last_autosave >= AUTOSAVE_EVERY:
            self.learner.save(self.learn_out)
            self._last_autosave = self.learner.steps
        dep = np.asarray(self.sim.log.veh_depart)
        if len(dep) and len(dep) != self._n_departed:
            arr = np.asarray(self.sim.log.veh_arrival)
            done = ~np.isnan(dep)
            if done.any():
                waits = dep[done] - arr[done]
                self._mean_wait = float(waits.mean())
                recent = dep[done] >= self.sim.t - ROLLING_WINDOW
                if recent.any():
                    self._rolling_wait = float(waits[recent].mean())
            self._n_departed = len(dep)

    # ------------------------------------------------------------------ drawing

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        self._blit_ground(screen)
        self._draw_car_shadows(screen)
        batch = BoxBatch()
        for a in range(4):
            self._signal_pole_faces(a, batch)
        self._car_faces(batch)
        self._ped_faces(batch)
        self._scenery_faces(batch)
        faces: list = []
        batch.flush(self.cam, faces)
        faces.sort(key=lambda f: -f[0])
        for _, quad, color in faces:
            pygame.draw.polygon(screen, color, [tuple(p) for p in quad])
        self._draw_signal_lamps(screen)
        self._draw_timing_labels(screen, font)
        self._draw_overflow_labels(screen, font)
        self._draw_hud(screen, font)

    # ------------------------------------------------------------ ground layer

    def _blit_ground(self, screen) -> None:
        """The ground never moves; only the crosswalk walk-tint changes. Cache
        one fully drawn surface per walk state and blit it."""
        key = (self._crosswalk_active(0), self._crosswalk_active(1))
        surf = self._ground_cache.get(key)
        if surf is None:
            surf = pygame.Surface((W, H))
            surf.fill(GRASS)
            self._draw_ground(surf)
            self._draw_static_shadows(surf)
            self._ground_cache[key] = surf
        screen.blit(surf, (0, 0))

    def _draw_static_shadows(self, surf) -> None:
        """Sun shadows for trees, buildings, and poles, baked into the cached
        ground (they never move)."""
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        ox, oy = self._SHADOW_SHIFT

        def blob(cx, cy, half_l, half_w, height, alpha=80):
            # Quad stretched from the object's base to where its top's shadow
            # lands — a cheap projected silhouette.
            ex, ey = ox * height, oy * height
            quad = [(cx - half_l, cy - half_w), (cx + half_l, cy - half_w),
                    (cx + half_l + ex, cy - half_w + ey),
                    (cx + half_l + ex, cy + half_w + ey),
                    (cx - half_l + ex, cy + half_w + ey),
                    (cx - half_l, cy + half_w)]
            pts = np.array([[qx, qy, 0.012] for qx, qy in quad])
            scr, _ = self.cam.project(pts)
            pygame.draw.polygon(overlay, (10, 14, 10, alpha), [tuple(p) for p in scr])

        for tx, ty, s in TREES:
            blob(tx, ty, 1.8 * s, 1.8 * s, 5.2 * s)
        for (bx, by), (bl, bw, bh), _color in BUILDINGS:
            blob(bx, by, bl / 2, bw / 2, bh, alpha=70)
        for a in range(4):
            px, py = self.POLE_POS[a]
            blob(px, py, 0.30, 0.30, 4.6, alpha=60)
        surf.blit(overlay, (0, 0))

    def _draw_ground(self, screen) -> None:
        cam = self.cam
        # Grass patches for slight tonal variation.
        for x, y, s in ((-55, 35, 26), (40, -30, 22), (-30, -50, 24), (48, 42, 20)):
            _flat_quad(cam, screen, [(x - s, y - s), (x + s, y - s), (x + s, y + s),
                                     (x - s, y + s)], GRASS_DARK, z=0.0)
        # Sidewalk aprons around each road, then asphalt on top.
        for lo, hi in ((-SIDEWALK_OUT, SIDEWALK_OUT),):
            _flat_quad(cam, screen, [(lo, -95), (hi, -95), (hi, 145), (lo, 145)],
                       SIDEWALK, z=0.005)
            _flat_quad(cam, screen, [(-105, lo), (75, lo), (75, hi), (-105, hi)],
                       SIDEWALK, z=0.005)
        _flat_quad(cam, screen, [(-ROAD_HALF, -95), (ROAD_HALF, -95), (ROAD_HALF, 145),
                                 (-ROAD_HALF, 145)], ASPHALT, z=0.01)
        _flat_quad(cam, screen, [(-105, -ROAD_HALF), (75, -ROAD_HALF), (75, ROAD_HALF),
                                 (-105, ROAD_HALF)], ASPHALT, z=0.01)
        self._draw_road_markings(screen)
        self._draw_crosswalks(screen)
        self._draw_lane_arrows(screen)

    def _draw_road_markings(self, screen) -> None:
        cam = self.cam
        far = 95
        for a in range(4):
            # Double yellow centerline from the crosswalk outward.
            for xo in (-0.55, -0.15):
                quad = [(xo - 0.13, STOP), (xo + 0.13, STOP), (xo + 0.13, far),
                        (xo - 0.13, far)]
                _flat_quad(cam, screen, [to_world(a, x, y) for x, y in quad],
                           CENTERLINE)
            # Dashed white lane dividers between L|S and S|R lanes.
            for xd in (-3.4, -6.8):
                y = STOP + 1.5
                while y < far:
                    quad = [(xd - 0.1, y), (xd + 0.1, y), (xd + 0.1, y + 2.2),
                            (xd - 0.1, y + 2.2)]
                    _flat_quad(cam, screen, [to_world(a, x, yy) for x, yy in quad],
                               LANE_DASH)
                    y += 5.5
            # Stop line across the three approach lanes.
            quad = [(-ROAD_HALF + 0.2, STOP), (-0.9, STOP), (-0.9, STOP + 0.55),
                    (-ROAD_HALF + 0.2, STOP + 0.55)]
            _flat_quad(cam, screen, [to_world(a, x, y) for x, y in quad], PAINT)

    def _crosswalk_active(self, movement: int) -> bool:
        sig = self.sim.signal
        return sig.walk_active and sig.phase == movement

    def _draw_crosswalks(self, screen) -> None:
        cam = self.cam
        half_band = 1.15
        color0 = WALK_TINT if self._crosswalk_active(0) else PAINT
        for side in (-1, 1):
            x0 = side * CROSSWALK_MID - half_band
            for yb in np.arange(-ROAD_HALF + 0.5, ROAD_HALF - 0.5, 1.5):
                _flat_quad(cam, screen, [(x0, yb), (x0 + 2 * half_band, yb),
                                         (x0 + 2 * half_band, yb + 0.75), (x0, yb + 0.75)],
                           color0)
        color1 = WALK_TINT if self._crosswalk_active(1) else PAINT
        for side in (-1, 1):
            y0 = side * CROSSWALK_MID - half_band
            for xb in np.arange(-ROAD_HALF + 0.5, ROAD_HALF - 0.5, 1.5):
                _flat_quad(cam, screen, [(xb, y0), (xb + 0.75, y0),
                                         (xb + 0.75, y0 + 2 * half_band),
                                         (xb, y0 + 2 * half_band)], color1)

    def _draw_lane_arrows(self, screen) -> None:
        """Painted turn arrows on each lane, just behind the stop line."""
        cam = self.cam
        y0 = STOP + 3.2
        for a in range(4):
            for turn in "LSR":
                x = LANE_X[turn]
                pts = self._arrow_pts(x, y0, turn)
                _flat_poly(cam, screen, [to_world(a, px, py) for px, py in pts], PAINT)

    @staticmethod
    def _arrow_pts(x: float, y0: float, turn: str) -> list[tuple[float, float]]:
        # Stem pointing toward the intersection (decreasing y), 2.6 m long.
        s, hw = 2.2, 0.16
        head_l, head_w = 1.0, 0.55
        if turn == "S":
            return [(x - hw, y0 + s), (x + hw, y0 + s), (x + hw, y0 + head_l),
                    (x + head_w, y0 + head_l), (x, y0), (x - head_w, y0 + head_l),
                    (x - hw, y0 + head_l)]
        d = 1.0 if turn == "L" else -1.0
        # Stem, then a bend with the head pointing left/right of travel.
        # Travel is -y; driver's left is +x in this local frame.
        bend_y = y0 + 0.9
        tip_x = x + d * 1.25
        return [
            (x - hw, y0 + s), (x + hw, y0 + s), (x + hw, bend_y + hw),
            (x + d * 0.55, bend_y + hw),
            (x + d * 0.55, bend_y + head_w * 0.9),
            (tip_x, bend_y - 0.05),
            (x + d * 0.55, bend_y - head_w),
            (x + d * 0.55, bend_y - hw) if d > 0 else (x - hw, bend_y - hw),
            (x - hw, bend_y - hw),
        ]

    # -------------------------------------------------------------------- cars

    def _visible_cars(self):
        for car in self.animator.cars.values():
            if abs(car.pos[0]) < 100 and abs(car.pos[1]) < 100:
                yield car

    # Where a unit-height object's shadow lands: xy + h * SUNxy / |SUNz|.
    _SHADOW_SHIFT = (float(SUN[0] / abs(SUN[2])), float(SUN[1] / abs(SUN[2])))

    def _draw_car_shadows(self, screen) -> None:
        """Sun-projected soft shadows on a per-frame alpha layer."""
        if not hasattr(self, "_shadow_surf"):
            self._shadow_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf = self._shadow_surf
        surf.fill((0, 0, 0, 0))
        cam = self.cam
        ox, oy = self._SHADOW_SHIFT
        drew = False
        for car in self._visible_cars():
            model = CAR_MODELS[_car_style(car.veh_id)[0]]
            length, width = model["foot"]
            hx, hy = car.heading
            px, py = car.pos
            px += ox * 0.55  # body mass sits ~0.55 m up; shadow shifts sunward
            py += oy * 0.55
            ax, ay = hx * (length / 2 + 0.35), hy * (length / 2 + 0.35)
            bx, by = -hy * (width / 2 + 0.30), hx * (width / 2 + 0.30)
            quad = [(px - ax - bx, py - ay - by), (px + ax - bx, py + ay - by),
                    (px + ax + bx, py + ay + by), (px - ax + bx, py - ay + by)]
            pts = np.array([[qx, qy, 0.015] for qx, qy in quad])
            scr, _ = cam.project(pts)
            pygame.draw.polygon(surf, (10, 12, 10, 88), [tuple(p) for p in scr])
            drew = True
        if drew:
            screen.blit(surf, (0, 0))

    def _car_faces(self, batch: BoxBatch) -> None:
        for car in self._visible_cars():
            name, color = _car_style(car.veh_id)
            model = CAR_MODELS[name]
            # Tinted glass: mostly the cool tint, a hint of the body color.
            glass = tuple(
                int(0.72 * t + 0.28 * c * 0.5)
                for t, c in zip(GLASS_TINT, color, strict=True)
            )
            hx, hy = car.heading
            yaw = float(np.arctan2(hy, hx))
            px, py = car.pos
            length, width = model["foot"]
            for (along, across), (bl, bw, bh), z0, kind in model["boxes"]:
                cx = px + hx * along - hy * across
                cy = py + hy * along + hx * across
                if kind == "glass":
                    batch.add((cx, cy, z0), (bl, bw, bh), glass, yaw=yaw, spec=0.90)
                elif kind == "trim":
                    batch.add((cx, cy, z0), (bl, bw, bh), TRIM_COLOR, yaw=yaw, spec=0.06)
                else:
                    batch.add((cx, cy, z0), (bl, bw, bh), color, yaw=yaw, spec=0.34)
            # Wheels: tire + a brighter hub poking through the outer face.
            wx, wy = length / 2 - 0.78, width / 2 - 0.02
            for sa in (1, -1):
                for sb in (1, -1):
                    cx = px + hx * (sa * wx) - hy * (sb * wy)
                    cy = py + hy * (sa * wx) + hx * (sb * wy)
                    batch.add((cx, cy, 0.0), (0.74, 0.30, 0.68), TIRE, yaw=yaw)
                    hx2 = px + hx * (sa * wx) - hy * (sb * (wy + 0.06))
                    hy2 = py + hy * (sa * wx) + hx * (sb * (wy + 0.06))
                    batch.add((hx2, hy2, 0.16), (0.34, 0.22, 0.34), HUB_COLOR,
                              yaw=yaw, spec=0.75)
            # Side mirrors.
            ma = model["mirror_along"]
            for sb in (1, -1):
                cx = px + hx * ma - hy * (sb * (width / 2 + 0.14))
                cy = py + hy * ma + hx * (sb * (width / 2 + 0.14))
                batch.add((cx, cy, 0.98), (0.24, 0.20, 0.20), TRIM_COLOR,
                          yaw=yaw, spec=0.3)
            # Head / tail light strips.
            for sa, lcolor in ((1, HEADLIGHT), (-1, TAILLIGHT)):
                cx = px + hx * (sa * (length / 2 + 0.02))
                cy = py + hy * (sa * (length / 2 + 0.02))
                batch.add((cx, cy, 0.52), (0.10, width * 0.70, 0.17), lcolor,
                          yaw=yaw, spec=0.85)

    # ----------------------------------------------------------------- signals

    # Pole beside each stop line on the approaching driver's side (right side).
    POLE_POS = {
        0: (-ROAD_HALF - 1.7, STOP + 1.0),
        1: (ROAD_HALF + 1.7, -STOP - 1.0),
        2: (STOP + 1.0, ROAD_HALF + 1.7),
        3: (-STOP - 1.0, -ROAD_HALF - 1.7),
    }

    def _has_left_head(self, a: int) -> bool:
        return self.config.layout.left_turn[a] == LeftTurnTreatment.PROTECTED

    def _signal_pole_faces(self, a: int, batch: BoxBatch) -> None:
        px, py = self.POLE_POS[a]
        batch.add((px, py, 0.0), (0.32, 0.32, 4.6), POLE_COLOR)
        vertical = a in (0, 1)
        head_size = (1.05, 0.55, 2.0) if vertical else (0.55, 1.05, 2.0)
        batch.add((px, py, 4.6), head_size, HEAD_COLOR)
        if self._has_left_head(a):
            # Arrow head hangs beside the main head, toward the road center.
            ox, oy = self._left_head_offset(a)
            batch.add((px + ox, py + oy, 4.6), head_size, HEAD_COLOR)

    @staticmethod
    def _left_head_offset(a: int) -> tuple[float, float]:
        return {0: (1.25, 0.0), 1: (-1.25, 0.0), 2: (0.0, -1.25), 3: (0.0, 1.25)}[a]

    def _through_color(self, a: int) -> tuple[int, int, int]:
        sig = self.sim.signal
        if through_group(a) not in sig.current.movements:
            return RED
        if sig.state == SignalState.GREEN:
            return GREEN_ON
        if sig.state == SignalState.YELLOW:
            return AMBER
        return RED

    def _left_color(self, a: int) -> tuple[int, int, int]:
        sig = self.sim.signal
        if left_group(a) not in sig.current.movements:
            return RED
        if sig.state == SignalState.GREEN:
            return GREEN_ON
        if sig.state == SignalState.YELLOW:
            return AMBER
        return RED

    def _draw_signal_lamps(self, screen) -> None:
        for a in range(4):
            px, py = self.POLE_POS[a]
            fx, fy = to_world(a, 0.0, 1.0)  # toward the approaching driver
            self._lamp_column(screen, px, py, fx, fy, self._through_color(a))
            if self._has_left_head(a):
                ox, oy = self._left_head_offset(a)
                self._lamp_column(
                    screen, px + ox, py + oy, fx, fy, self._left_color(a), arrow=True, a=a
                )

    def _lamp_column(self, screen, px, py, fx, fy, color, arrow=False, a=0) -> None:
        lit = {RED: 0, AMBER: 1, GREEN_ON: 2}[color]
        centers = np.array(
            [[px + fx * 0.30, py + fy * 0.30, 6.15 - slot * 0.64] for slot in range(3)]
        )
        scr, depth = self.cam.project(centers)
        for slot in range(3):
            r = max(2, int(0.30 * self.cam.focal / depth[slot]))
            lamp = (RED, AMBER, GREEN_ON)[slot] if slot == lit else LIGHT_OFF
            cx, cy = int(scr[slot][0]), int(scr[slot][1])
            pygame.draw.circle(screen, lamp, (cx, cy), r)
            if arrow and slot == lit and r >= 3:
                # Arrow glyph: chevron pointing toward the driver's left.
                lx, ly = to_world(a, 1.0, 0.0)
                sx = lx * 0.9
                pts = [(cx - int(r * 0.7) * (1 if sx >= 0 else -1), cy),
                       (cx + int(r * 0.5) * (1 if sx >= 0 else -1), cy - int(r * 0.6)),
                       (cx + int(r * 0.5) * (1 if sx >= 0 else -1), cy + int(r * 0.6))]
                pygame.draw.polygon(screen, HEAD_COLOR, pts)

    # -------------------------------------------------------- peds and scenery

    def _ped_faces(self, batch: BoxBatch) -> None:
        sig = self.sim.signal
        for m in range(2):
            count = min(len(self.sim.waiting_peds[m]), 4)
            for i in range(count):
                if m == 0:
                    px, py = CROSSWALK_MID + 0.4 + 0.9 * i, -ROAD_HALF - 1.6
                else:
                    px, py = ROAD_HALF + 1.6 + 0.9 * i, CROSSWALK_MID + 0.4
                self._figure(px, py, batch)
        if sig.walk_active and sig.in_walk_window:
            m = sig.phase
            frac = min(1.0, sig.walk_elapsed / max(sig.timing.walk, 1e-6))
            span = 2 * (ROAD_HALF + 1.2)
            for side in (-1, 1):
                progress = -ROAD_HALF - 1.2 + frac * span
                progress *= side
                if m == 0:
                    self._figure(side * CROSSWALK_MID, progress, batch)
                else:
                    self._figure(progress, side * CROSSWALK_MID, batch)

    def _figure(self, px: float, py: float, batch: BoxBatch) -> None:
        batch.add((px, py, 0.0), (0.5, 0.5, 1.25), PED_COLOR)
        batch.add((px, py, 1.3), (0.34, 0.34, 0.34), (90, 70, 58))

    def _scenery_faces(self, batch: BoxBatch) -> None:
        for i, (tx, ty, s) in enumerate(TREES):
            fol = FOLIAGE[i % len(FOLIAGE)]
            batch.add((tx, ty, 0.0), (0.55 * s, 0.55 * s, 2.6 * s), TRUNK)
            batch.add((tx, ty, 2.4 * s), (3.4 * s, 3.4 * s, 2.4 * s), fol)
            batch.add((tx, ty, 4.6 * s), (2.2 * s, 2.2 * s, 1.7 * s), fol)
        for (bx, by), (bl, bw, bh), color in BUILDINGS:
            batch.add((bx, by, 0.0), (bl, bw, bh), color)
            batch.add((bx, by, bh), (bl * 0.94, bw * 0.94, 0.5),
                      tuple(int(c * 0.8) for c in color))

    def _draw_timing_labels(self, screen, font) -> None:
        """Floating card above each pole: green time per phase this light
        serves (planned for fixed plans, measured last green otherwise), with
        the active phase counting up live."""
        sig = self.sim.signal
        phases = self.config.phases
        for a in range(4):
            thru_slot, left_slot = (1, 0) if a in (0, 1) else (3, 2)
            entries = []
            for tag, slot in (("L", left_slot), ("T", thru_slot)):
                idx = next((i for i, p in enumerate(phases) if p.slot == slot), None)
                if idx is None:
                    continue
                green = self._phase_green.get(idx)
                base = f"{green:.0f}s" if green is not None else "--"
                if sig.phase == idx and sig.state == SignalState.GREEN:
                    entries.append((f"{tag} {sig.state_elapsed:.0f}/{base}", GREEN_ON))
                elif sig.phase == idx and sig.state == SignalState.YELLOW:
                    entries.append((f"{tag} {base}", AMBER))
                else:
                    entries.append((f"{tag} {base}", HUD_DIM))
            if not entries:
                continue
            px, py = self.POLE_POS[a]
            pos, _ = self.cam.project(np.array([[px, py, 7.4]]))
            surfs = [font.render(text, True, color) for text, color in entries]
            w = max(s.get_width() for s in surfs) + 10
            h = sum(s.get_height() for s in surfs) + 8
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            panel.fill((10, 10, 10, 165))
            y = 4
            for s in surfs:
                panel.blit(s, (5, y))
                y += s.get_height()
            screen.blit(panel, panel.get_rect(midbottom=(int(pos[0][0]), int(pos[0][1]))))

    def _draw_overflow_labels(self, screen, font) -> None:
        for a, extra in self.animator.overflow(self.sim).items():
            pos, _ = self.cam.project(
                np.array([[*to_world(a, LANE_X["S"], 88.0), 2.0]])
            )
            label = font.render(f"+{extra}", True, HUD_INK)
            screen.blit(label, label.get_rect(center=(int(pos[0][0]), int(pos[0][1]))))

    # --------------------------------------------------------------------- HUD

    def _draw_hud(self, screen, font) -> None:
        sig = self.sim.signal
        q = [
            len(self.sim.queues[through_group(a)]) + len(self.sim.queues[left_group(a)])
            for a in range(4)
        ]
        title = (
            f"rl-online (learning)  ·  {self.scenario}  ·  seed {self.seed}"
            if self.learner
            else f"{self.controller_name}  ·  {self.scenario}  ·  seed {self.seed}"
        )
        lines = [
            title,
            f"t = {self.sim.t:7.0f} s   speed {self.speed:.0f}x"
            + ("   PAUSED" if self.paused else ""),
            f"phase {sig.current.name} {sig.state.name}"
            f"  ({sig.state_elapsed:.0f} s in state)",
            f"queues  N {q[0]:>3}  S {q[1]:>3}  E {q[2]:>3}  W {q[3]:>3}",
            f"mean wait: last hour {self._rolling_wait:5.1f} s"
            f" · overall {self._mean_wait:5.1f} s · departed {self._n_departed_done()}",
        ]
        if self.learner:
            lr = self.learner
            lines.append(
                f"learning  eps {lr.epsilon:.2f}  steps {lr.steps:,}"
                f"  updates {lr.updates:,}  reward ema {lr.reward_ema:6.3f}"
            )
        lines.append("Space pause · +/- speed · R reset · Esc quit")
        pad, lh = 12, 22
        panel = pygame.Surface((520, pad * 2 + lh * len(lines)), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 175))
        screen.blit(panel, (10, 10))
        for i, line in enumerate(lines):
            color = HUD_DIM if i == len(lines) - 1 else HUD_INK
            screen.blit(font.render(line, True, color), (10 + pad, 10 + pad + i * lh))

    def _n_departed_done(self) -> int:
        return sum(1 for d in self.sim.log.veh_depart if d == d)  # non-nan

    # ------------------------------------------------------------------ loop

    def run(self, smoke_frames: int | None = None, screenshot: str | None = None) -> None:
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
        if screenshot:
            pygame.image.save(screen, screenshot)
        if self.learner and self.learner.steps > 0:
            self.learner.save(self.learn_out)
            print(f"saved online-learned weights -> {self.learn_out}")
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
    parser.add_argument(
        "--screenshot", type=str, default=None, metavar="PATH",
        help="save the final frame as a PNG (useful with --smoke)",
    )
    parser.add_argument(
        "--learn", action="store_true",
        help="online learning: the RL policy keeps training while you watch "
        "(warm-started from the shipped weights; ignores --controller)",
    )
    parser.add_argument(
        "--learn-fresh", action="store_true",
        help="with --learn: start from random weights instead of the shipped policy",
    )
    parser.add_argument(
        "--learn-out", type=str, default=None, metavar="PATH",
        help="with --learn: where to save the learned weights "
        "(default results/online_weights.npz; autosaves every 100k steps and on exit)",
    )
    args = parser.parse_args()
    if args.smoke is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    app = ViewerApp(
        args.controller, args.scenario, args.speed, args.seed,
        learn=args.learn or args.learn_fresh, learn_fresh=args.learn_fresh,
        learn_out=args.learn_out,
    )
    app.run(smoke_frames=args.smoke, screenshot=args.screenshot)
    if args.smoke is not None:
        print(f"smoke ok: {args.smoke} frames, sim t = {app.sim.t:.0f} s")


if __name__ == "__main__":
    main()
