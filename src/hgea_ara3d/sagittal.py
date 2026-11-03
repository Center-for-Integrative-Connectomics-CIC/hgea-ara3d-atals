"""Place the 20 sagittal HGEA plates in the coronal 3D ARA (CCFv2 2011).

The sagittal plates are drawn on the sagittal ARA, a different brain, so a plate is not one slice here.
Each plate is a plane through the 3D ARA:

    plate (x = AP mm from bregma, + anterior; y = DV mm)
      -> in-plane similarity + aspect:  u = 1000 s a x,  v = 1000 s y / a,  rotated by th
      -> AP index  = a0 - u' / 25,   DV index = d0 + v' / 25
      -> LR index  = l0 + side * L * 1000 sL / 25 + al * (AP - a0) + be * (DV - d0)     (L = printed lateral mm)

The series is one cut, so s, a, th, a0, d0, sL, al, be are shared (stack fit); each plate then gets small
corrections (shift in AP, DV, LR, scale, rotation). Fit cost: mean distance (capped at CAP_UM) from the
plate's printed ARA lines to the 3D annotation's label edges, plus a penalty for plate brain outside the
annotation brain. Distances come from one 3D distance map, so evaluating any plane is a lookup.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize
from shapely import contains_xy

from hgea_ara3d.space import VOXEL_UM

CAP_UM = 300.0
STEP_MM = 0.02


def edge_distance_3d(annotation: np.ndarray) -> np.ndarray:
    """Distance (µm, float32) from every voxel to the nearest label edge in 3D."""
    edge = np.zeros(annotation.shape, bool)
    for ax in range(3):
        d = np.diff(annotation, axis=ax) != 0
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        edge[tuple(sl_a)] |= d; edge[tuple(sl_b)] |= d
    return (ndimage.distance_transform_edt(~edge) * VOXEL_UM).astype(np.float32)


def line_points(plate, step_mm: float = STEP_MM) -> np.ndarray:
    out = []
    for g in plate.reference["region_lines"] + plate.reference["outlines"]:
        for part in getattr(g, "geoms", [g]):
            ln = part.exterior if part.geom_type == "Polygon" else part
            if ln.length <= 0:
                continue
            n = max(2, int(ln.length / step_mm))
            out.append(np.array([ln.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)]))
    return np.vstack(out)


def brain_points(plate, step_mm: float = 0.1) -> np.ndarray:
    b = plate.reference["brain_mask"][0]
    x0, y0, x1, y1 = b.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, step_mm), np.arange(y0, y1, step_mm))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    return xy[contains_xy(b, xy[:, 0], xy[:, 1])]


GLOBAL = ["s", "asp", "th", "a0", "d0", "l0", "sL", "al", "be"]
LOCAL = ["da", "dd", "dl", "ds", "dth"]


def to_ara(xy_mm: np.ndarray, lateral_mm: float, side: int, g: dict, p: dict | None = None) -> np.ndarray:
    """Plate mm -> ARA voxel coordinates (AP, DV, LR), float."""
    p = p or {}
    s = g["s"] * (1 + p.get("ds", 0.0))
    th = g["th"] + p.get("dth", 0.0)
    u = xy_mm[:, 0] * 1000.0 * s * g["asp"]
    v = xy_mm[:, 1] * 1000.0 * s / g["asp"]
    c, sn = np.cos(th), np.sin(th)
    u, v = c * u - sn * v, sn * u + c * v
    a0 = g["a0"] + p.get("da", 0.0)
    d0 = g["d0"] + p.get("dd", 0.0)
    ap = a0 - u / VOXEL_UM
    dv = d0 + v / VOXEL_UM
    lr = (g["l0"] + side * lateral_mm * 1000.0 * g["sL"] / VOXEL_UM + p.get("dl", 0.0)
          + g["al"] * (ap - a0) + g["be"] * (dv - d0))
    return np.column_stack([ap, dv, lr])


class Fitter:
    def __init__(self, plates, dist3d: np.ndarray, brain3d: np.ndarray, side: int):
        self.plates, self.dist, self.brain, self.side = plates, dist3d, brain3d.astype(np.float32), side
        self.lines = [line_points(p) for p in plates]
        self.bpts = [brain_points(p) for p in plates]

    def plate_cost(self, k: int, g: dict, p: dict | None = None) -> float:
        pl = self.plates[k]
        d = ndimage.map_coordinates(self.dist, to_ara(self.lines[k], pl.lateral_mm, self.side, g, p).T, order=1, cval=CAP_UM)
        inside = ndimage.map_coordinates(self.brain, to_ara(self.bpts[k], pl.lateral_mm, self.side, g, p).T, order=1, cval=0.0)
        return float(np.mean(np.minimum(d, CAP_UM)) + 200.0 * (1.0 - inside.mean()))

    def distances(self, k: int, g: dict, p: dict | None = None) -> np.ndarray:
        pl = self.plates[k]
        return ndimage.map_coordinates(self.dist, to_ara(self.lines[k], pl.lateral_mm, self.side, g, p).T, order=1, cval=CAP_UM)

    def fit_global(self, g0: dict, bounds: dict) -> dict:
        names = GLOBAL
        def cost(vec):
            g = dict(zip(names, vec))
            return float(np.mean([self.plate_cost(k, g) for k in range(len(self.plates))]))
        b = [bounds[n] for n in names]
        x0 = np.clip([g0[n] for n in names], [lo for lo, hi in b], [hi for lo, hi in b])
        r = optimize.minimize(cost, x0, method="Powell", bounds=b, options=dict(xtol=1e-4, ftol=1e-4, maxiter=6000))
        return dict(zip(names, r.x))

    def fit_plate(self, k: int, g: dict) -> dict:
        b = [(-12, 12), (-12, 12), (-8, 8), (-0.06, 0.06), (-0.05, 0.05)]
        def cost(vec):
            p = dict(zip(LOCAL, vec))
            # weak pull towards the stack fit keeps the corrections small and honest
            return self.plate_cost(k, g, p) + 0.2 * (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) / 25 + 200 * (vec[3] ** 2 + vec[4] ** 2)
        r = optimize.minimize(cost, np.zeros(5), method="Powell", bounds=b, options=dict(xtol=1e-4, ftol=1e-4, maxiter=4000))
        return dict(zip(LOCAL, r.x))


def domain_samples(plate, step_mm: float = 0.025, domain_id: dict | None = None):
    """Grid points (plate mm) inside the plate brain with their HGEA domain id (0 = none)."""
    pts = brain_points(plate, step_mm)
    lab = np.zeros(len(pts), np.int16)
    for name, geom in plate.domains.items():
        if domain_id and name in domain_id and not geom.is_empty:
            lab[contains_xy(geom, pts[:, 0], pts[:, 1])] = domain_id[name]
    return pts, lab


FIELD_OF = None


def agreement(plates, g, local, side, volume, domains):
    """Compare a 3D HGEA label volume with the sagittal drawings on their planes.

    Returns per plate and pooled: extent Dice (any HGEA vs any HGEA; mostly brain-to-brain shape),
    field agreement and exact-domain agreement where both have a label (the definitional test)."""
    field = {i + 1: ("CA1" if d.startswith("CA1") else "CA2" if d == "CA2" else "SUB" if d.startswith("SUB") else "DGCA3")
             for i, d in enumerate(domains)}
    dom_id = {d: i + 1 for i, d in enumerate(domains)}
    rows, conf = [], np.zeros((len(domains) + 1, len(domains) + 1), np.int64)
    for p in plates:
        pts, lab = domain_samples(p, 0.025, dom_id)
        v = np.rint(to_ara(pts, p.lateral_mm, side, g, local[p.level])).astype(int)
        ok = np.all((v >= 0) & (v < np.array(volume.shape)), axis=1)
        v, lab = v[ok], lab[ok]
        pr = volume[v[:, 0], v[:, 1], v[:, 2]].astype(int) % 100
        both = (lab > 0) & (pr > 0)
        np.add.at(conf, (lab[both], pr[both]), 1)
        nb = int(both.sum())
        rows.append(dict(level=p.level, lateral_mm=p.lateral_mm, sagittal_pts=int((lab > 0).sum()), volume_pts=int((pr > 0).sum()),
                         both_pts=nb, extent_dice=2 * nb / max(1, (lab > 0).sum() + (pr > 0).sum()),
                         field_agreement=float(np.mean([field[a] == field[c] for a, c in zip(lab[both], pr[both])])) if nb else np.nan,
                         domain_agreement=float((lab[both] == pr[both]).mean()) if nb else np.nan))
    return rows, conf
