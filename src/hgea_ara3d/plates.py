"""Parse the original HGEA coronal plates (Bienkowski et al. 2018) into labelled geometry.

(From the HGEA-3D project, hgea.hgea_plates.svg, unchanged apart from this note.)

Each `HGEA-Coronal-0NNBW.svg` is an Allen Reference Atlas (ARA) coronal plate exported from
Illustrator with the HGEA domains drawn on top:

- `HGAA_right_borders` / `HGAA_left_borders`: one child per domain, identified by its id
  (`CA1d`, `DGd_x2F_CA3d` = "DGd/CA3d", `SUB_x5F_sp1` = SUB layer 1, `_1_` = duplicate).
  Colours are not unique (CA1d and SUB layer 1 share #EC1C24), so ids are authoritative.
- `Brain_outline...`: ARA reference geometry (brain outline, ventricles, fibre tracts, dashed
  region boundaries) used later to register each plate into CCF.
- A dashed mm grid, a cyan midline, `Bregma -N.NNN mm` and `Coronal Level NN` labels.

Output coordinates are plate millimetres: x = ML from the midline (+ towards the image right),
y = the plate's DV axis (mm, increasing ventrally), exactly as printed on the plate axes.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import set_precision
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from svgpathtools import parse_path

NS = "{http://www.w3.org/2000/svg}"
SHAPES = (NS + "path", NS + "polygon", NS + "rect")
SIDES = {"right": "HGAA_right_borders", "left": "HGAA_left_borders"}
SAMPLE_PX = 1.0  # curve sampling step in plate pixels (~9 µm)

# Plate domain ids -> canonical names. Combined DG/CA3 entries are split later by the CCF
# DG-sg / DG-po / CA3 fields (legend: "DGd/DGpod/CA3d", "DGi/CA3i", "DGv/DGpov/CA3vv").
CANONICAL = {
    "ca1d": "CA1d", "ca1i": "CA1i", "ca1v": "CA1v", "ca1vv": "CA1vv", "ca2": "CA2",
    "ca3dd": "CA3dd", "ca3v": "CA3v",
    "dgd/ca3d": "DGd/CA3d", "dgi/ca3i": "DGi/CA3i", "dgv/ca3vv": "DGv/CA3vv",
    "sub_sp1": "SUB_1", "sub_sp2": "SUB_2", "sub_sp3": "SUB_3", "sub_sp4": "SUB_4",
}


def unescape_id(raw: str | None) -> str:
    s = re.sub(r"_\d+_$", "", raw or "")
    return re.sub(r"_x([0-9A-Fa-f]{2})_", lambda m: chr(int(m.group(1), 16)), s)


@dataclass
class Plate:
    level: int
    bregma_mm: float
    file: str
    midline_px: float
    px_per_mm: tuple[float, float]          # (x, y)
    dv_ref_px: tuple[float, float]          # (pixel y, plate DV mm) of one grid line
    domains: dict[str, dict[str, Polygon]] = field(default_factory=dict)   # side -> name -> geometry (mm)
    extra_lines: dict[str, list[LineString]] = field(default_factory=dict)  # side -> unfilled lines (mm)
    reference: dict[str, list] = field(default_factory=dict)                 # ARA geometry (mm)
    warnings: list[str] = field(default_factory=list)

    def to_mm(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        x = (xy[:, 0] - self.midline_px) / self.px_per_mm[0]
        y = self.dv_ref_px[1] + (xy[:, 1] - self.dv_ref_px[0]) / self.px_per_mm[1]
        return np.column_stack([x, y])


def _subpaths(d: str) -> list[np.ndarray]:
    """Sample a path's subpaths into pixel polylines."""
    out = []
    try:
        path = parse_path(d)
    except Exception:
        return out
    for sub in path.continuous_subpaths():
        length = sub.length(error=1e-3)
        if not np.isfinite(length) or length <= 0:
            continue
        n = int(np.clip(length / SAMPLE_PX, 8, 20000))
        pts = np.array([sub.point(t) for t in np.linspace(0, 1, n)])
        out.append(np.column_stack([pts.real, pts.imag]))
    return out


def _element_rings(e: ET.Element) -> list[np.ndarray]:
    if e.tag == NS + "path":
        return _subpaths(e.get("d", ""))
    if e.tag == NS + "polygon":
        v = np.array([float(t) for t in re.split(r"[\s,]+", e.get("points", "").strip()) if t])
        return [v.reshape(-1, 2)]
    if e.tag == NS + "rect":
        x, y, w, h = (float(e.get(k, 0)) for k in ("x", "y", "width", "height"))
        return [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
    return []


GRID_MM = 1e-4  # snap to 0.1 µm so near-coincident edges don't break GEOS set operations


def polygonal(g):
    """Valid, polygon-only, precision-snapped geometry (make_valid can leave line slivers)."""
    g = make_valid(g)
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        polys = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
        g = make_valid(unary_union(polys)) if polys else Polygon()
    return make_valid(set_precision(g, GRID_MM))


def _filled_geometry(rings: list[np.ndarray]):
    """Even-odd fill of a set of rings (holes come out as holes)."""
    geom = None
    for r in rings:
        if len(r) < 3:
            continue
        p = polygonal(Polygon(r))
        geom = p if geom is None else polygonal(geom.symmetric_difference(p))
    return geom


def _calibration(root: ET.Element) -> tuple[float, tuple[float, float], tuple[float, float], float | None]:
    mid = next((e for e in root.iter(NS + "line") if e.get("id") == "midline"), None)
    lines = [e for e in root.iter(NS + "line") if e.get("stroke-dasharray")]
    vx = np.array(sorted({round(float(e.get("x1")), 2) for e in lines if e.get("x1") == e.get("x2")}))
    hy = np.array(sorted({round(float(e.get("y1")), 2) for e in lines if e.get("y1") == e.get("y2")}))

    def spacing(v):
        d = np.diff(v)
        d = d[(d > 80) & (d < 140)]
        return float(np.median(d))

    ppm = (spacing(vx), spacing(hy))
    # ML = 0 from the vertical grid line under the top-axis "0" label. The cyan `midline`
    # element is not reliable (plate 81 has it displaced by 9 px from both grid and drawing).
    zero = []
    for t in root.iter(NS + "text"):
        m = re.search(r"matrix\(([^)]+)\)", t.get("transform", ""))
        if m and "".join(t.itertext()).strip() == "0":
            a = [float(v) for v in m.group(1).split()]
            if a[5] < 100:
                zero.append(a[4])
    if zero:
        midx = float(vx[np.argmin(np.abs(vx - (zero[0] + 4)))])
    else:
        midx = float(mid.get("x1")) if mid is not None else float(vx[np.argmin(np.abs(vx - 792))])
    cyan = float(mid.get("x1")) if mid is not None else None

    # DV: match horizontal grid lines to the numeric labels on the left axis
    labels = []
    for t in root.iter(NS + "text"):
        txt = "".join(t.itertext()).strip()
        m = re.search(r"matrix\(([^)]+)\)", t.get("transform", ""))
        if m and re.fullmatch(r"\d", txt):
            a = [float(v) for v in m.group(1).split()]
            if a[4] < 200:  # left axis column
                labels.append((a[5], int(txt)))
    pairs = []
    for ly, val in labels:
        j = np.argmin(np.abs(hy - ly))
        if abs(hy[j] - ly) < 15:
            pairs.append((hy[j], val))
    if not pairs:
        raise ValueError("could not tie grid lines to DV labels")
    # consistency: every pair must agree with a single offset
    y0 = np.median([y - v * ppm[1] for y, v in pairs])
    return midx, ppm, (float(y0), 0.0), cyan


def parse_plate(path: Path) -> Plate:
    text = path.read_text(errors="replace")
    root = ET.fromstring(text.encode())
    level = int(re.search(r"Coronal Level\s*(\d+)", text).group(1))
    bregma = float(re.search(r"Bregma\s*(-?[\d.]+)\s*mm", text).group(1))
    midx, ppm, dvref, cyan = _calibration(root)
    plate = Plate(level, bregma, path.name, midx, ppm, dvref)
    if cyan is not None and abs(cyan - midx) > 2:
        plate.warnings.append(f"cyan midline at x={cyan:.1f} px disagrees with grid zero x={midx:.1f} px; using grid")

    for side, gid in SIDES.items():
        groups = [e for e in root.iter() if e.get("id") == gid]
        if len(groups) != 1:
            plate.warnings.append(f"{side}: {len(groups)} '{gid}' groups")
            continue
        parts: dict[str, object] = {}  # insertion order = SVG paint order
        plate.extra_lines[side] = []
        for child in groups[0]:
            raw = unescape_id(child.get("id"))
            filled = [e for e in child.iter() if e.tag in SHAPES and e.get("fill") not in (None, "none")]
            stroked = [e for e in child.iter() if e.tag in SHAPES + (NS + "line",) and e.get("fill") in (None, "none")]
            for e in stroked:
                rings = _element_rings(e) if e.tag != NS + "line" else [
                    np.array([[float(e.get("x1")), float(e.get("y1"))], [float(e.get("x2")), float(e.get("y2"))]])]
                plate.extra_lines[side] += [LineString(plate.to_mm(r)) for r in rings if len(r) > 1]
            if not filled:
                if raw:
                    plate.warnings.append(f"{side}: '{raw}' has no filled shape")
                continue
            name = CANONICAL.get(raw.lower())
            if name is None:
                plate.warnings.append(f"{side}: unknown domain id '{raw}'")
                continue
            for e in filled:
                g = _filled_geometry([plate.to_mm(r) for r in _element_rings(e)])
                if g is None or g.is_empty:
                    continue
                # later shapes paint over earlier ones (e.g. CA3dd drawn on top of DGd/CA3d)
                for other in parts:
                    if other != name:
                        parts[other] = polygonal(parts[other].difference(g))
                parts[name] = polygonal(unary_union([parts[name], g])) if name in parts else g
        plate.domains[side] = {n: g for n, g in parts.items() if not g.is_empty}

    plate.reference = _reference_geometry(root, plate)
    return plate


NON_REFERENCE = ("box_and_coordinate", "stereotaxic_coordinate", "hgaa_", "text", "hgea_figure_legend")
GREY_FILLS = {"#BFBFBF", "#CCCCCC", "#A6A6A6"}
BLACK_FILLS = {None, "#000000", "#231F20", "#050000"}  # no fill attribute = SVG default black


def _reference_geometry(root: ET.Element, plate: Plate) -> dict[str, list]:
    """ARA reference geometry, classified by drawing style (group names vary between plates).

    - region_lines: dashed black strokes (ARA region boundaries)
    - outlines:     solid black strokes (brain outline, major boundaries)
    - fiber_tracts: grey fills;  ventricles: black fills
    - brain_mask:   area enclosed by outlines + filled shapes
    """
    out: dict[str, list] = {"region_lines": [], "outlines": [], "fiber_tracts": [], "ventricles": []}
    groups = [e for e in root if e.tag == NS + "g" and not (e.get("id") or "").lower().startswith(NON_REFERENCE)]
    for g in groups:
        for e in g.iter():
            if e.tag == NS + "line":
                rings = [np.array([[float(e.get("x1")), float(e.get("y1"))], [float(e.get("x2")), float(e.get("y2"))]])]
            elif e.tag in SHAPES:
                rings = _element_rings(e)
            else:
                continue
            fill = e.get("fill")
            fill = fill.upper() if fill and fill != "none" else fill
            for r in rings:
                if len(r) < 2:
                    continue
                mm = plate.to_mm(r)
                if fill in GREY_FILLS and len(r) > 2:
                    out["fiber_tracts"].append(polygonal(Polygon(mm)))
                elif fill in BLACK_FILLS and e.tag != NS + "line" and len(r) > 2:
                    out["ventricles"].append(polygonal(Polygon(mm)))
                elif e.get("stroke-dasharray"):
                    out["region_lines"].append(LineString(mm))
                else:
                    out["outlines"].append(LineString(mm))
    out["brain_mask"] = [_brain_mask(out)]
    return out


def _brain_mask(ref: dict[str, list], gap_mm: float = 0.04):
    """Area inside the outer brain outline.

    Outlines are drawn as separate left/right halves that do not quite meet, so polygonizing
    them fails on some plates. Thicken by `gap_mm`, keep the largest component's exterior,
    fill it and shrink back.
    """
    shapes = [g.buffer(gap_mm) for g in ref["outlines"]] + ref["fiber_tracts"] + ref["ventricles"]
    if not shapes:
        return Polygon()
    blob = polygonal(unary_union(shapes))
    largest = max(getattr(blob, "geoms", [blob]), key=lambda q: Polygon(q.exterior).area)
    return polygonal(Polygon(largest.exterior).buffer(-gap_mm))
