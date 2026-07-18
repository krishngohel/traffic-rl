"""Procedural low-poly car meshes for the viewer.

Each car is built from two lofts — a lower body hull and a greenhouse — plus
eight-sided wheel cylinders. A loft is a sequence of cross-section stations
(x along the car, z bottom/top, half-width); quads span consecutive stations,
so a rising roofline becomes a sloped windshield and a falling one becomes the
rear glass. Everything is triangles with precomputed local normals, ready for
the viewer's vectorized shading pass.

Materials: 0 body paint, 1 glass, 2 dark trim, 3 tire, 4 hub/chrome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BODY, GLASS, TRIM, TIRE, HUB = 0, 1, 2, 3, 4


@dataclass
class Mesh:
    verts: np.ndarray  # (V, 3) local coords: x forward, y left, z up
    tris: np.ndarray  # (T, 3) vertex indices
    normals: np.ndarray  # (T, 3) local per-tri normals (unit)
    mats: np.ndarray  # (T,) material ids
    foot: tuple[float, float]  # (length, width) for shadows and spacing


class _Builder:
    def __init__(self):
        self.verts: list = []
        self.tris: list = []
        self.mats: list = []

    def _v(self, x, y, z) -> int:
        self.verts.append((x, y, z))
        return len(self.verts) - 1

    def quad(self, a, b, c, d, mat) -> None:
        """Two triangles for corner points given counter-clockwise viewed from
        outside; normal comes from winding."""
        ia, ib, ic, id_ = (self._v(*p) for p in (a, b, c, d))
        self.tris += [(ia, ib, ic), (ia, ic, id_)]
        self.mats += [mat, mat]

    def build(self, foot) -> Mesh:
        verts = np.asarray(self.verts, dtype=np.float64)
        tris = np.asarray(self.tris, dtype=np.int64)
        e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
        e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(norm, 1e-12)
        return Mesh(
            verts=verts,
            tris=tris,
            normals=n,
            mats=np.asarray(self.mats, dtype=np.int64),
            foot=foot,
        )


def _loft(b: _Builder, stations, side_mat, top_mat, glass_slope=None,
          cap_front=True, cap_rear=True):
    """stations: list of (x, z_lo, z_hi, half_width), nose first (max x).
    Emits left/right side quads, top quads between stations, and end caps.
    If glass_slope is set, top quads steeper than it use GLASS (windshields)."""
    for (x1, lo1, hi1, w1), (x2, lo2, hi2, w2) in zip(stations, stations[1:], strict=False):
        # Right side (y = -w): outward normal -y needs winding accordingly.
        b.quad((x1, -w1, lo1), (x1, -w1, hi1), (x2, -w2, hi2), (x2, -w2, lo2), side_mat)
        # Left side (y = +w).
        b.quad((x1, w1, lo1), (x2, w2, lo2), (x2, w2, hi2), (x1, w1, hi1), side_mat)
        # Top surface.
        mat = top_mat
        if glass_slope is not None and abs(x2 - x1) > 1e-9:
            if abs(hi2 - hi1) / abs(x2 - x1) > glass_slope:
                mat = GLASS
        b.quad((x1, -w1, hi1), (x1, w1, hi1), (x2, w2, hi2), (x2, -w2, hi2), mat)
        # Underside skirt (visible at the wheel cutline from low cameras).
        b.quad((x1, -w1, lo1), (x2, -w2, lo2), (x2, w2, lo2), (x1, w1, lo1), TRIM)
    if cap_front:
        x, lo, hi, w = stations[0]
        b.quad((x, -w, lo), (x, w, lo), (x, w, hi), (x, -w, hi), side_mat)
    if cap_rear:
        x, lo, hi, w = stations[-1]
        b.quad((x, -w, lo), (x, -w, hi), (x, w, hi), (x, w, lo), side_mat)


def _wheel(b: _Builder, cx, cy_side, radius=0.33, width=0.24, sides=8):
    """Eight-sided cylinder, axle across y; outer cap gets a chrome hub."""
    ang = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ring = [(cx + radius * np.cos(a), radius + radius * np.sin(a)) for a in ang]
    # ring z uses radius offset so the wheel sits on the ground (z from 0).
    y_in = cy_side - np.sign(cy_side) * width / 2
    y_out = cy_side + np.sign(cy_side) * width / 2
    n = len(ring)
    for i in range(n):
        (x1, z1), (x2, z2) = ring[i], ring[(i + 1) % n]
        if cy_side > 0:
            b.quad((x1, y_in, z1), (x1, y_out, z1), (x2, y_out, z2), (x2, y_in, z2), TIRE)
        else:
            b.quad((x1, y_in, z1), (x2, y_in, z2), (x2, y_out, z2), (x1, y_out, z1), TIRE)
    # Outer cap: hub fan (as quads collapsed to tris via repeated point).
    for i in range(n):
        (x1, z1), (x2, z2) = ring[i], ring[(i + 1) % n]
        if cy_side > 0:
            b.quad((cx, y_out, radius), (x1, y_out, z1), (x2, y_out, z2),
                   (cx, y_out, radius), HUB)
        else:
            b.quad((cx, y_out, radius), (x2, y_out, z2), (x1, y_out, z1),
                   (cx, y_out, radius), HUB)


def _car(body_st, house_st, foot, wheel_x=(1.45, -1.45), wheel_r=0.33) -> Mesh:
    b = _Builder()
    _loft(b, body_st, BODY, BODY)
    if house_st:
        _loft(b, house_st, GLASS, BODY, glass_slope=0.28, cap_front=False,
              cap_rear=False)
        # Windshield / rear glass caps for the greenhouse ends.
        x, lo, hi, w = house_st[0]
        b.quad((x, -w, lo), (x, w, lo), (x, w, hi), (x, -w, hi), GLASS)
        x, lo, hi, w = house_st[-1]
        b.quad((x, -w, lo), (x, -w, hi), (x, w, hi), (x, w, lo), GLASS)
    half_w = foot[1] / 2
    for wx in wheel_x:
        _wheel(b, wx, half_w - 0.06, radius=wheel_r)
        _wheel(b, wx, -(half_w - 0.06), radius=wheel_r)
    return b.build(foot)


def build_car_meshes() -> dict[str, Mesh]:
    meshes = {}
    # Sedan: long hood, fast windshield, notchback trunk.
    meshes["sedan"] = _car(
        body_st=[
            (2.25, 0.30, 0.58, 0.82),
            (2.05, 0.26, 0.74, 0.90),
            (1.10, 0.24, 0.90, 0.94),
            (-1.30, 0.24, 0.94, 0.94),
            (-2.05, 0.26, 0.86, 0.90),
            (-2.25, 0.32, 0.66, 0.84),
        ],
        house_st=[
            (1.05, 0.88, 0.94, 0.86),
            (0.35, 0.88, 1.42, 0.76),
            (-0.75, 0.88, 1.44, 0.76),
            (-1.35, 0.88, 0.96, 0.84),
        ],
        foot=(4.5, 1.88),
    )
    # SUV: tall, boxy greenhouse, near-vertical tailgate.
    meshes["suv"] = _car(
        body_st=[
            (2.35, 0.34, 0.72, 0.86),
            (2.10, 0.30, 0.94, 0.94),
            (1.15, 0.28, 1.10, 0.97),
            (-2.10, 0.28, 1.10, 0.97),
            (-2.35, 0.34, 0.94, 0.90),
        ],
        house_st=[
            (1.10, 1.04, 1.12, 0.90),
            (0.55, 1.04, 1.78, 0.82),
            (-1.85, 1.04, 1.80, 0.82),
            (-2.10, 1.04, 1.16, 0.86),
        ],
        foot=(4.7, 1.98),
        wheel_x=(1.55, -1.5),
        wheel_r=0.37,
    )
    # Pickup: cab forward, open bed behind.
    b = _Builder()
    _loft(b, [
        (2.55, 0.34, 0.74, 0.86),
        (2.25, 0.30, 1.02, 0.94),
        (1.35, 0.28, 1.10, 0.97),
        (0.15, 0.28, 1.06, 0.97),
        (0.10, 0.28, 0.96, 0.97),   # bed rail drop
        (-2.45, 0.28, 0.96, 0.97),
        (-2.55, 0.34, 0.90, 0.94),
    ], BODY, BODY)
    _loft(b, [
        (1.30, 1.04, 1.12, 0.88),
        (0.80, 1.04, 1.80, 0.82),
        (-0.05, 1.04, 1.82, 0.82),
        (-0.25, 1.04, 1.10, 0.86),
    ], GLASS, BODY, glass_slope=0.28, cap_front=False, cap_rear=False)
    b.quad((1.30, -0.88, 1.04), (1.30, 0.88, 1.04), (1.30, 0.82, 1.80),
           (1.30, -0.82, 1.80), GLASS)
    b.quad((-0.25, -0.86, 1.04), (-0.25, -0.82, 1.82), (-0.25, 0.82, 1.82),
           (-0.25, 0.86, 1.04), GLASS)
    b.quad((-0.35, -0.85, 0.98), (-0.35, 0.85, 0.98), (-2.35, 0.85, 0.98),
           (-2.35, -0.85, 0.98), TRIM)  # bed floor (drawn high: open box look)
    for wx in (1.7, -1.6):
        _wheel(b, wx, 0.94, radius=0.37)
        _wheel(b, wx, -0.94, radius=0.37)
    meshes["pickup"] = b.build((5.1, 2.0))
    # Van: one long volume, steep windshield.
    meshes["van"] = _car(
        body_st=[
            (2.55, 0.34, 0.80, 0.90),
            (2.30, 0.30, 1.10, 0.98),
            (1.55, 0.28, 1.30, 1.0),
            (-2.35, 0.28, 1.30, 1.0),
            (-2.55, 0.34, 1.10, 0.94),
        ],
        house_st=[
            (1.50, 1.24, 1.32, 0.94),
            (1.05, 1.24, 2.05, 0.90),
            (-2.25, 1.24, 2.08, 0.90),
            (-2.45, 1.24, 1.34, 0.92),
        ],
        foot=(5.1, 2.02),
        wheel_x=(1.75, -1.65),
        wheel_r=0.35,
    )
    # Coupe: low, long hood, fastback tail.
    meshes["coupe"] = _car(
        body_st=[
            (2.10, 0.28, 0.52, 0.80),
            (1.90, 0.24, 0.66, 0.88),
            (0.90, 0.22, 0.80, 0.92),
            (-1.40, 0.22, 0.82, 0.92),
            (-2.10, 0.26, 0.62, 0.84),
        ],
        house_st=[
            (0.85, 0.76, 0.84, 0.84),
            (0.15, 0.76, 1.26, 0.72),
            (-0.65, 0.76, 1.26, 0.72),
            (-1.85, 0.76, 0.70, 0.80),  # long fastback slope to the tail
        ],
        foot=(4.2, 1.84),
        wheel_x=(1.35, -1.35),
    )
    # Hatchback: short, tall glass, chopped tail.
    meshes["hatch"] = _car(
        body_st=[
            (1.85, 0.30, 0.58, 0.80),
            (1.60, 0.26, 0.80, 0.86),
            (0.85, 0.24, 0.92, 0.89),
            (-1.55, 0.24, 0.94, 0.89),
            (-1.85, 0.30, 0.80, 0.84),
        ],
        house_st=[
            (0.80, 0.88, 0.96, 0.82),
            (0.20, 0.88, 1.46, 0.74),
            (-1.30, 0.88, 1.48, 0.74),
            (-1.70, 0.88, 0.94, 0.80),  # steep hatch glass
        ],
        foot=(3.7, 1.78),
        wheel_x=(1.15, -1.15),
    )
    return meshes
