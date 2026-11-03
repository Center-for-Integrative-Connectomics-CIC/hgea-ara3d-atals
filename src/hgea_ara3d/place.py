"""Place each HGEA coronal plate on its own slice of the 3D ARA (CCFv2 2011), in-plane exact.

The HGEA plates are ARA 2008 coronal plates, and Allen built the 2011 3D annotation from those same
ARA drawings. So a plate's printed ARA region lines must fall on the label edges of one annotation
slice. For each plate:

    1. search the AP slices around its Allen section number; on each, fit an in-plane similarity
       (scale, aspect, rotation, shift) that minimises the mean distance from the plate's ARA lines
       (`region_lines` + `outlines`, sampled every 20 µm) to the annotation's label edges
       (distances capped at 300 µm so missing or merged lines cannot dominate);
    2. keep the slice with the smallest mean distance (a sharp minimum; see `slice_profile`);
    3. map the plate's HGEA domain polygons onto that slice.

Plate coordinates are plate mm (x = ML from the midline, y = DV); slice coordinates are 25 µm voxels
(DV, LR) of the (AP, DV, LR) ARA3D grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize
from shapely import contains_xy

from hgea_ara3d.space import VOXEL_UM

CAP_UM = 300.0
LINE_STEP_MM = 0.02


def line_points(plate, step_mm: float = LINE_STEP_MM) -> np.ndarray:
    """Sample points (plate mm) along the plate's printed ARA region lines and outlines."""
    out = []
    for g in plate.reference["region_lines"] + plate.reference["outlines"]:
        for part in getattr(g, "geoms", [g]):
            ln = part.exterior if part.geom_type == "Polygon" else part
            n = max(2, int(ln.length / step_mm))
            out.append(np.array([ln.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)]))
    return np.vstack(out)


def to_slice(q, xy_mm: np.ndarray) -> np.ndarray:
    """Plate mm -> slice voxels (DV, LR) under q = (scale, aspect, rotation rad, LR shift, DV shift)."""
    s, asp, th, tx, ty = q
    c, sn = np.cos(th), np.sin(th)
    u = xy_mm[:, 0] * 1000.0 * s * asp
    v = xy_mm[:, 1] * 1000.0 * s / asp
    return np.column_stack([ty + (sn * u + c * v) / VOXEL_UM, tx + (c * u - sn * v) / VOXEL_UM])


def from_slice(q, dv_lr: np.ndarray) -> np.ndarray:
    """Slice voxels (DV, LR) -> plate mm: exact inverse of `to_slice`."""
    s, asp, th, tx, ty = q
    c, sn = np.cos(th), np.sin(th)
    up = (dv_lr[:, 1] - tx) * VOXEL_UM
    vp = (dv_lr[:, 0] - ty) * VOXEL_UM
    u, v = c * up + sn * vp, -sn * up + c * vp
    return np.column_stack([u / (1000.0 * s * asp), v / (1000.0 * s / asp)])


def edge_distance_um(labels2d: np.ndarray) -> np.ndarray:
    """Distance (µm) from every pixel to the nearest label edge of an annotation slice."""
    edges = ndimage.maximum_filter(labels2d, 3) != ndimage.minimum_filter(labels2d, 3)
    return ndimage.distance_transform_edt(~edges) * VOXEL_UM


def line_distances(q, pts_mm: np.ndarray, dist_um: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(dist_um, to_slice(q, pts_mm).T, order=1, cval=CAP_UM)


def brain_dice(q, brain_pts_mm: np.ndarray, brain_slice: np.ndarray) -> float:
    """Overlap of the plate brain outline (sample points inside it) with the annotation brain mask."""
    hit = ndimage.map_coordinates(brain_slice.astype(np.float32), to_slice(q, brain_pts_mm).T, order=1, cval=0.0)
    return float(hit.mean())


def fit_on_slice(pts_mm: np.ndarray, dist_um: np.ndarray, q0, brain_pts_mm: np.ndarray,
                 brain_slice: np.ndarray) -> tuple[np.ndarray, float]:
    """Bounded fit: ARA line distance, plus a penalty if the plate brain leaves the annotation brain
    (stops the trivial solution of shrinking the plate onto one edge)."""
    q0 = np.asarray(q0, float)
    bounds = [(0.85, 1.15), (0.92, 1.08), (-0.09, 0.09), (q0[3] - 25, q0[3] + 25), (q0[4] - 25, q0[4] + 25)]

    def cost(q):
        d = float(np.mean(np.minimum(line_distances(q, pts_mm, dist_um), CAP_UM)))
        return d + 200.0 * (1.0 - brain_dice(q, brain_pts_mm, brain_slice))

    r = optimize.minimize(cost, np.clip(q0, [b[0] for b in bounds], [b[1] for b in bounds]), method="Powell",
                          bounds=bounds, options=dict(xtol=1e-4, ftol=1e-4, maxiter=4000))
    return r.x, float(np.mean(np.minimum(line_distances(r.x, pts_mm, dist_um), CAP_UM)))


@dataclass
class Placement:
    level: int
    section: int
    z: int
    q: np.ndarray
    mean_um: float
    median_um: float
    p90_um: float
    hpf_median_um: float
    profile: list            # [(z, mean_um)] over the searched slices
    margin_um: float         # second-best minus best mean distance (sharpness of the minimum)
    fits: dict | None = None  # z -> in-plane parameters fitted on that slice


def brain_points(plate, step_mm: float = 0.1) -> np.ndarray:
    brain = plate.reference["brain_mask"][0]
    x0, y0, x1, y1 = brain.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, step_mm), np.arange(y0, y1, step_mm))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    return xy[contains_xy(brain, xy[:, 0], xy[:, 1])]


def place_plate(plate, section: int, annotation: np.ndarray, q0, search=range(-10, 5),
                hpf_box_mm=None) -> Placement:
    pts = line_points(plate)
    bpts = brain_points(plate)
    profile, fits = [], {}
    q = np.asarray(q0, float)   # every slice starts from the same q0, so one bad slice cannot drift the next
    for dz in search:
        z = section + dz
        qz, f = fit_on_slice(pts, edge_distance_um(annotation[z]), q, bpts, annotation[z] > 0)
        profile.append((z, f))
        fits[z] = qz
    best = min(profile, key=lambda t: t[1])
    others = sorted(f for z, f in profile if abs(z - best[0]) > 1)
    z = best[0]
    d = line_distances(fits[z], pts, edge_distance_um(annotation[z]))
    hp = d
    if hpf_box_mm is not None:
        x0, y0, x1, y1 = hpf_box_mm
        inbox = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        hp = d[inbox] if inbox.any() else d
    return Placement(plate.level, section, z, fits[z], float(best[1]), float(np.median(d)),
                     float(np.percentile(d, 90)), float(np.median(hp)), profile,
                     float(others[0] - best[1]) if others else float("nan"), fits)


def stack_line(placements: list, a_grid=np.arange(-20, 15, 0.25), b_grid=np.arange(0.90, 1.02, 0.0025)):
    """One line z = a + b * section for the whole plate series, minimising the summed excess line
    distance over every plate's slice profile. The plates are one evenly cut series, so this replaces
    per-plate slice picks, which are poorly determined (neighbouring slices score within a few µm)."""
    prof = [dict(pl.profile) for pl in placements]
    lo = [min(p.values()) for p in prof]
    best = None
    for b in b_grid:
        for a in a_grid:
            zs = [int(round(a + b * pl.section)) for pl in placements]
            if any(z not in p for z, p in zip(zs, prof)):
                continue
            tot = sum(p[z] - m for z, p, m in zip(zs, prof, lo))
            if best is None or tot < best[0]:
                best = (tot, a, b)
    return best[1], best[2], best[0]


def finalise(plate, pl: Placement, z: int, annotation: np.ndarray, hpf_box_mm=None) -> Placement:
    """Use slice z (from the stack line) with the in-plane fit made on it; recompute distances."""
    pts = line_points(plate)
    q = pl.fits[z]
    d = line_distances(q, pts, edge_distance_um(annotation[z]))
    hp = d
    if hpf_box_mm is not None:
        x0, y0, x1, y1 = hpf_box_mm
        inbox = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        hp = d[inbox] if inbox.any() else d
    return Placement(pl.level, pl.section, z, q, float(np.mean(np.minimum(d, CAP_UM))), float(np.median(d)),
                     float(np.percentile(d, 90)), float(np.median(hp)), pl.profile, pl.margin_um, pl.fits)


def domain_labels(plate, q, shape_dv_lr: tuple[int, int], domain_id: dict, upsample: int = 1) -> np.ndarray:
    """HGEA domain labels on the slice grid (optionally finer by `upsample`); left side +100."""
    H, W = shape_dv_lr[0] * upsample, shape_dv_lr[1] * upsample
    rr, cc = np.mgrid[0:H, 0:W]
    vox = np.column_stack([(rr.ravel() + 0.5) / upsample - 0.5, (cc.ravel() + 0.5) / upsample - 0.5])
    mm = from_slice(q, vox)
    lab = np.zeros(H * W, np.int16)
    for side, doms in plate.domains.items():
        off = 100 if side == "left" else 0
        for name, geom in doms.items():
            if name not in domain_id or geom.is_empty:
                continue
            x0, y0, x1, y1 = geom.bounds
            cand = (mm[:, 0] >= x0) & (mm[:, 0] <= x1) & (mm[:, 1] >= y0) & (mm[:, 1] <= y1)
            idx = np.nonzero(cand)[0]
            inside = contains_xy(geom, mm[idx, 0], mm[idx, 1])
            lab[idx[inside]] = domain_id[name] + off
    return lab.reshape(H, W)
