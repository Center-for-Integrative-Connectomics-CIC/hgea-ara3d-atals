"""The original sagittal HGEA plates (Bienkowski et al. 2018) from the authors' Illustrator files.

Sources: HGEA-Sagittal-0NNBW.ai (PDF-compatible Illustrator CS6) and HGEA_full_sagittal.pdf (all 20
plates, same drawings). Each page has optional-content layers: "HGEA" (domains), "Stereotaxic
coordinate" (mm grid, axis labels), "Brain outline, ventricles, and fibers tract" (ARA reference),
"Text", "HGEA Figure Legend".

The files carry no domain ids, so domains are identified exactly as the legend defines them: by fill
colour and effective opacity (fill alpha x enclosing transparency-group alpha), e.g. #EC1C24 at 0.5 =
DGd/CA3d, at 0.8 = CA1d, at 1.0 = SUB_1. Later fills paint over earlier ones (paint order), as in the
coronal parser; a blue DGv/CA3vv shape stacked inside a green DGi/CA3i shape is CA3v (the legend's
green + blue swatch).

A small PDF content-stream interpreter handles q/Q, cm, colours (rg/g/k/sc/scn), ExtGState alpha,
paths (m l c v y h re), fills, clips (W/W* + n, intersected with each fill), Form XObjects (Do, with
/Matrix and /BBox) and marked content (optional-content layers), and records text with positions.

Plate coordinates: x = AP mm from bregma (+ anterior), y = DV mm (as
printed), plus the printed lateral position. Calibrated from the dashed 1-mm grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from hgea_ara3d.plates import polygonal

# layer names vary between files ("Stereotaxic coordinate(s)", "box and coordinate", ...): match by role
def layer_role(name: str) -> str:
    n = (name or "").strip().lower()
    if n in ("hgea", "layer 1"):          # levels 3-4 keep the domains on Illustrator's default "Layer 1"
        return "hgea"
    if "coordinate" in n:
        return "grid"
    if n.startswith("brain outline"):
        return "ref"
    return n

# legend: (fill hex, effective opacity) -> class (the plates' domain vocabulary)
LEGEND_FILLS = {
    ("#EC1C24", 0.5): "DGd/CA3d", ("#00A551", 0.5): "DGi/CA3i", ("#00ADEE", 0.5): "DGv/CA3vv",
    ("#652D90", 0.5): "CA3dd", ("#F6921E", 0.8): "CA2", ("#EC1C24", 0.8): "CA1d", ("#00A551", 0.8): "CA1i",
    ("#00ADEE", 0.8): "CA1v", ("#224099", 0.8): "CA1vv", ("#00BFCA", 0.75): "CA3v",
    ("#EC1C24", 1.0): "SUB_1", ("#00ADEE", 1.0): "SUB_2", ("#F6921E", 1.0): "SUB_3", ("#FFF100", 1.0): "SUB_4",
}
# level 21 (lateral -0.20 mm, the midline plate) is drawn with pure hues at the legend opacities
ALT_FILLS = {("#FF0000", 0.8): "CA1d", ("#FF0000", 0.5): "DGd/CA3d", ("#FF8000", 0.8): "CA2", ("#4000FF", 0.5): "CA3dd"}
UNIQUE_HUES = {"#652D90": "CA3dd", "#224099": "CA1vv", "#00BFCA": "CA3v"}   # no other class shares the hue
GREY = {"#BFBFBF", "#CCCCCC", "#A6A6A6", "#BCBEC0", "#D1D3D4", "#C7C8CA"}
BLACK = {"#000000", "#231F20", "#050000"}


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, c)) * 255)):02X}" for c in rgb)


# ---- content-stream tokenizer --------------------------------------------------------------------
_DELIM = b"()<>[]{}/%"
_WS = b" \t\r\n\x0c\x00"


def _tokens(data: bytes):
    i, n = 0, len(data)
    while i < n:
        ch = data[i:i + 1]
        if ch in (b" ", b"\t", b"\r", b"\n", b"\x0c", b"\x00"):
            i += 1
        elif ch == b"%":
            while i < n and data[i:i + 1] not in (b"\r", b"\n"):
                i += 1
        elif ch == b"(":
            depth, j, buf = 1, i + 1, bytearray()
            while j < n and depth:
                c = data[j:j + 1]
                if c == b"\\":
                    buf += data[j + 1:j + 2]
                    j += 2
                    continue
                if c == b"(":
                    depth += 1
                elif c == b")":
                    depth -= 1
                    if depth == 0:
                        break
                buf += c
                j += 1
            yield ("str", bytes(buf))
            i = j + 1
        elif data[i:i + 2] == b"<<":
            yield ("dict<", None); i += 2
        elif data[i:i + 2] == b">>":
            yield ("dict>", None); i += 2
        elif ch == b"<":
            j = data.index(b">", i)
            h = re.sub(rb"\s", b"", data[i + 1:j])
            yield ("str", bytes.fromhex((h + b"0" * (len(h) % 2)).decode()))
            i = j + 1
        elif ch == b"[":
            yield ("[", None); i += 1
        elif ch == b"]":
            yield ("]", None); i += 1
        elif ch == b"/":
            j = i + 1
            while j < n and data[j:j + 1] not in _WS and data[j:j + 1] not in _DELIM:
                j += 1
            yield ("name", data[i + 1:j].decode("latin1"))
            i = j
        else:
            j = i
            while j < n and data[j:j + 1] not in _WS and data[j:j + 1] not in _DELIM:
                j += 1
            tok = data[i:j].decode("latin1")
            try:
                yield ("num", float(tok))
            except ValueError:
                yield ("op", tok)
            i = j if j > i else i + 1


@dataclass
class Fill:
    layer: str
    hex: str
    alpha: float
    geom: object          # shapely, page points (y up)
    order: int


@dataclass
class Stroke:
    layer: str
    hex: str
    dashed: bool
    line: LineString


@dataclass
class Text:
    layer: str
    text: str
    x: float
    y: float
    size: float


def _mat(a, b, c, d, e, f):
    return np.array([[a, b, 0], [c, d, 0], [e, f, 1]], float)


def _apply(M, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ M
    return p[:, :2]


def _bezier(p0, p1, p2, p3, n=12):
    t = np.linspace(0, 1, n + 1)[1:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _rings_to_geom(rings, evenodd):
    polys = []
    for r in rings:
        if len(r) >= 3:
            p = Polygon(r)
            if p.area > 0 or not p.is_valid:
                polys.append((p, Polygon(r).exterior.is_ccw))
    if not polys:
        return None
    if evenodd or len(polys) == 1:
        g = None
        for p, _ in polys:
            p = polygonal(p)
            g = p if g is None else polygonal(g.symmetric_difference(p))
        return g
    # nonzero: majority orientation = filled, opposite = holes (Illustrator compound paths)
    ccw = sum(p.area for p, o in polys if o)
    cw = sum(p.area for p, o in polys if not o)
    fill_o = ccw >= cw
    pos = unary_union([polygonal(p) for p, o in polys if o == fill_o])
    neg = [polygonal(p) for p, o in polys if o != fill_o]
    return polygonal(pos.difference(unary_union(neg))) if neg else polygonal(pos)


class _Interp:
    def __init__(self, page_resources, ocg_names):
        self.ocg = ocg_names
        self.fills, self.strokes, self.texts = [], [], []
        self.order = 0

    def run(self, data, res, M, alpha_mult, clip, layer_stack):
        gs = dict(M=M, fill=(0, 0, 0), stroke=(0, 0, 0), ca=1.0, clip=clip, dash=False, fcs="DeviceGray")
        stack, ops, path, cur, start, pending_clip = [], [], [], None, None, None
        mc = list(layer_stack)
        tm = None
        text_size = 1.0
        rings = []

        def flush_sub():
            nonlocal path
            if len(path) > 1:
                rings.append(np.array(path))
            path = []

        def layer():
            names = [x for x in mc if x]
            return names[-1] if names else ""

        def paint(fill, stroke, evenodd=False):
            nonlocal rings, pending_clip
            flush_sub()
            if pending_clip is not None:
                cg = _rings_to_geom(pending_clip[0], pending_clip[1])
                if cg is not None:
                    gs["clip"] = cg if gs["clip"] is None else gs["clip"].intersection(cg)
                pending_clip = None
            if fill and rings:
                g = _rings_to_geom([_apply(gs["M"], r) for r in rings], evenodd)
                if g is not None and not g.is_empty:
                    if gs["clip"] is not None:
                        g = polygonal(g.intersection(gs["clip"]))
                    if not g.is_empty:
                        self.fills.append(Fill(layer(), _hex(gs["fill"]), round(gs["ca"] * alpha_mult, 3), g, self.order))
                        self.order += 1
            if stroke and rings:
                for r in rings:
                    self.strokes.append(Stroke(layer(), _hex(gs["stroke"]), gs["dash"], LineString(_apply(gs["M"], r))))
            rings = []

        arr = None
        dict_depth = 0
        for kind, val in _tokens(data):
            if kind == "dict<":
                dict_depth += 1
                stack.append(("dict", None))
                continue
            if kind == "dict>":
                dict_depth -= 1
                continue
            if dict_depth:
                stack.append((kind, val))
                continue
            if kind == "[":
                arr = []
                continue
            if kind == "]":
                stack.append(("arr", arr)); arr = None
                continue
            if arr is not None:
                arr.append(val)
                continue
            if kind != "op":
                stack.append((kind, val))
                continue
            op = val
            nums = [v for k, v in stack if k == "num"]
            if op == "q":
                ops.append(dict(gs))
            elif op == "Q":
                if ops:
                    gs = ops.pop()
            elif op == "cm":
                gs["M"] = _mat(*nums[-6:]) @ gs["M"]
            elif op in ("rg", "RG", "g", "G", "k", "K", "sc", "scn", "SC", "SCN"):
                if op in ("g", "G") or len(nums) == 1:
                    c = (nums[-1],) * 3
                elif op in ("k", "K") or len(nums) == 4:
                    C, Mg, Y, K = nums[-4:]
                    c = ((1 - C) * (1 - K), (1 - Mg) * (1 - K), (1 - Y) * (1 - K))
                else:
                    c = tuple(nums[-3:])
                gs["fill" if op in ("rg", "g", "k", "sc", "scn") else "stroke"] = c
            elif op == "gs":
                name = [v for k, v in stack if k == "name"][-1]
                e = res.get("/ExtGState", {}).get("/" + name)
                if e is not None:
                    e = e.get_object()
                    if "/ca" in e:
                        gs["ca"] = float(e["/ca"])
            elif op == "d":
                a = [v for k, v in stack if k == "arr"]
                gs["dash"] = bool(a and a[-1])
            elif op == "m":
                flush_sub()
                cur = np.array(nums[-2:]); start = cur; path = [cur]
            elif op == "l":
                cur = np.array(nums[-2:]); path.append(cur)
            elif op in ("c", "v", "y"):
                if op == "c":
                    p1, p2, p3 = np.array(nums[-6:-4]), np.array(nums[-4:-2]), np.array(nums[-2:])
                elif op == "v":
                    p1, p2, p3 = cur, np.array(nums[-4:-2]), np.array(nums[-2:])
                else:
                    p1, p2, p3 = np.array(nums[-4:-2]), np.array(nums[-2:]), np.array(nums[-2:])
                path.extend(_bezier(cur, p1, p2, p3))
                cur = p3
            elif op == "h":
                if start is not None and path:
                    path.append(start)
                    cur = start
            elif op == "re":
                flush_sub()
                x, y, w, h = nums[-4:]
                rings.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]))
            elif op in ("W", "W*"):
                flush_sub()
                pending_clip = ([_apply(gs["M"], r) for r in rings], op == "W*")
            elif op in ("f", "F", "f*"):
                paint(True, False, op == "f*")
            elif op in ("B", "B*", "b", "b*"):
                if op in ("b", "b*") and path:
                    path.append(start)
                paint(True, True, "*" in op)
            elif op in ("S", "s"):
                if op == "s" and path:
                    path.append(start)
                paint(False, True)
            elif op == "n":
                paint(False, False)
            elif op == "Do":
                name = [v for k, v in stack if k == "name"][-1]
                xo = res.get("/XObject", {}).get("/" + name)
                if xo is not None:
                    xo = xo.get_object()
                    if xo.get("/Subtype") == "/Form":
                        FM = _mat(*[float(x) for x in xo.get("/Matrix", [1, 0, 0, 1, 0, 0])]) @ gs["M"]
                        bb = [float(x) for x in xo["/BBox"]]
                        bbox = Polygon(_apply(FM, np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])))
                        clip2 = bbox if gs["clip"] is None else gs["clip"].intersection(bbox)
                        self.run(xo.get_data(), xo.get("/Resources", res), FM, alpha_mult * gs["ca"], clip2, mc)
            elif op in ("BDC", "BMC"):
                names = [v for k, v in stack if k == "name"]
                nm = None
                if op == "BDC" and len(names) >= 2:
                    prop = res.get("/Properties", {}).get("/" + names[-1])
                    if prop is not None:
                        o = prop.get_object()
                        nm = str(o.get("/Name") or o.get("/Title") or "") or None
                mc.append(nm)
            elif op == "EMC":
                if mc:
                    mc.pop()
            elif op == "BT":
                tm = np.eye(3)
            elif op == "Tf":
                text_size = nums[-1] if nums else 1.0
            elif op == "Tm":
                tm = _mat(*nums[-6:])
            elif op in ("Td", "TD"):
                if tm is not None:
                    tm = _mat(1, 0, 0, 1, *nums[-2:]) @ tm
            elif op in ("Tj", "TJ", "'", '"'):
                strs = [v for k, v in stack if k == "str"] + [x for k, v in stack if k == "arr" for x in v if isinstance(x, bytes)]
                s = b"".join(strs).decode("latin1")
                if tm is not None and s.strip():
                    T = tm @ gs["M"]
                    x, y = _apply(T, np.array([[0.0, 0.0]]))[0]
                    size = abs(text_size * np.hypot(T[0, 0], T[0, 1]))
                    self.texts.append(Text(layer(), s, x, y, size))
            stack = []
        return self


def read_page(pdf_path: Path, page_index: int = 0) -> _Interp:
    from pypdf import PdfReader
    r = PdfReader(str(pdf_path))
    page = r.pages[page_index]
    res = page["/Resources"]
    ocg = {}
    props = res.get("/Properties", {})
    for k in props:
        o = props[k].get_object()
        if "/Name" in o or "/Title" in o:
            ocg[id(o)] = str(o.get("/Name") or o.get("/Title"))
    it = _Interp(res, ocg)
    return it.run(page.get_contents().get_data(), res, np.eye(3), 1.0, None, [])


# ---- calibration and plate assembly ---------------------------------------------------------------
@dataclass
class AIPlate:
    level: int
    lateral_mm: float
    source: str
    pt_per_mm: tuple[float, float]
    x_ap0: float              # page x of AP = 0 (bregma)
    y_dv0: float              # page y of DV = 0
    domains: dict = field(default_factory=dict)      # class -> shapely (plate mm)
    reference: dict = field(default_factory=dict)    # brain_mask, ventricles, fiber_tracts, outlines (plate mm)
    warnings: list = field(default_factory=list)

    def to_mm(self, xy):
        xy = np.asarray(xy, float)
        return np.column_stack([(self.x_ap0 - xy[:, 0]) / self.pt_per_mm[0], (self.y_dv0 - xy[:, 1]) / self.pt_per_mm[1]])


def _grid(it: _Interp):
    v, h = [], []
    for s in it.strokes:
        if layer_role(s.layer) != "grid" or not s.dashed:
            continue
        xy = np.asarray(s.line.coords)
        if len(xy) != 2:
            continue
        dx, dy = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
        if dy > 500 and dx < 3:
            v.append(xy[:, 0].mean())
        elif dx > 500 and dy < 3:
            h.append(xy[:, 1].mean())
    return np.sort(v), np.sort(h)


def _lattice(pos):
    """Spacing and phase of a 1-mm lattice from (possibly incomplete) grid-line positions."""
    d = np.diff(pos)
    step = np.median(d[(d > 0.6 * np.median(d)) & (d < 1.4 * np.median(d))])
    k = np.round((pos - pos[0]) / step)
    A = np.column_stack([np.ones_like(k), k])
    (p0, step), *_ = np.linalg.lstsq(A, pos, rcond=None)
    resid = pos - (p0 + step * k)
    return p0, step, float(np.abs(resid).max())


def parse_plate(pdf_path: Path, page_index: int = 0) -> AIPlate:
    it = read_page(pdf_path, page_index)
    txt = " ".join(t.text for t in it.texts)
    level = int(re.search(r"Sagittal Level\s*(\d+)", txt).group(1))
    lat = re.search(r"Lateral\s*(-?[\d.]+)\s*mm", txt)
    lateral = float(lat.group(1)) if lat else (-0.20 if level == 21 else np.nan)
    v, h = _grid(it)
    x0, sx, rx = _lattice(v)
    y0, sy, ry = _lattice(h)
    # AP 0 = the vertical grid line under the "Bregma" label; DV origin from the DV axis labels
    breg = next(t for t in it.texts if t.text.strip() == "Bregma")
    bx = breg.x + 0.5 * 3.0 * breg.size        # label centre (~6 glyphs x 0.5 em)
    x_ap0 = x0 + sx * np.round((bx - x0) / sx)
    dv_labels = [(int(t.text.strip()), t.y) for t in it.texts
                 if layer_role(t.layer) == "grid" and re.fullmatch(r"\d", t.text.strip()) and (t.x < 290 or t.x > 2040)]
    if dv_labels:
        # each label sits next to its grid line (slightly below its baseline); DV = (y_dv0 - y) / step
        est = [y + d * sy for d, y in dv_labels]
        y_dv0 = y0 + sy * np.round((np.median(est) - y0) / sy)
    else:
        y_dv0 = np.nan
    p = AIPlate(level, lateral, f"{pdf_path.name}#p{page_index + 1}", (sx, sy), x_ap0, y_dv0)
    if max(rx, ry) > 2.0:
        p.warnings.append(f"grid residual {max(rx, ry):.2f} pt")
    if not dv_labels:
        p.warnings.append("no DV axis labels found")

    # domains: classify HGEA fills by (colour, effective opacity); paint order
    parts: dict[str, object] = {}
    greens = []   # DGi/CA3i fills so far (page coordinates)
    for f in sorted([f for f in it.fills if layer_role(f.layer) == "hgea"], key=lambda f: f.order):
        name = classify_fill(f.hex, f.alpha)
        if name == "DGi/CA3i":
            greens.append(f.geom)
        # the legend's CA3v swatch is green with blue stacked on top: a DGv/CA3vv shape lying inside a
        # DGi/CA3i shape is drawn that way (levels 4, 5, 7) and is CA3v; free-standing blue stays DGv/CA3vv
        if name == "DGv/CA3vv" and greens and f.geom.intersection(unary_union(greens)).area >= 0.9 * f.geom.area:
            name = "CA3v"
        if name is None:
            p.warnings.append(f"unclassified HGEA fill {f.hex} @{f.alpha}")
            continue
        if (f.hex, round(f.alpha, 1)) in ALT_FILLS:
            p.warnings.append(f"non-legend palette {f.hex} @{f.alpha} read as {name}")
        g = polygonal(Polygon(p.to_mm(np.asarray(f.geom.exterior.coords)))) if f.geom.geom_type == "Polygon" and not f.geom.interiors else \
            polygonal(unary_union([Polygon(p.to_mm(np.asarray(q.exterior.coords)), [p.to_mm(np.asarray(i.coords)) for i in q.interiors])
                                   for q in getattr(f.geom, "geoms", [f.geom]) if q.geom_type == "Polygon"]))
        if g.is_empty:
            continue
        for other in parts:
            if other != name:
                parts[other] = polygonal(parts[other].difference(g))
        parts[name] = polygonal(unary_union([parts[name], g])) if name in parts else g
    p.domains = {k: g for k, g in parts.items() if not g.is_empty}

    ref = {"outlines": [], "fiber_tracts": [], "ventricles": [], "region_lines": []}
    to_mm_geom = lambda g: polygonal(unary_union([Polygon(p.to_mm(np.asarray(q.exterior.coords)),
                                                          [p.to_mm(np.asarray(i.coords)) for i in q.interiors])
                                                  for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon"]))
    for f in it.fills:
        if layer_role(f.layer) != "ref":
            continue
        if f.hex in GREY:
            ref["fiber_tracts"].append(to_mm_geom(f.geom))
        elif f.hex in BLACK:
            ref["ventricles"].append(to_mm_geom(f.geom))
    for s in it.strokes:
        if layer_role(s.layer) == "ref":
            ln = LineString(p.to_mm(np.asarray(s.line.coords)))
            (ref["region_lines"] if s.dashed else ref["outlines"]).append(ln)
    ref["brain_mask"] = [_sagittal_brain_mask(ref)]
    bm = ref["brain_mask"][0]
    ref["fiber_tracts"] = [g for g in ref["fiber_tracts"] if g.intersects(bm)]   # drop grey legend arrowheads
    p.reference = ref
    return p


def _near(h1: str, h2: str, tol: int = 12) -> bool:
    a = [int(h1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(h2[i:i + 2], 16) for i in (1, 3, 5)]
    return max(abs(x - y) for x, y in zip(a, b)) <= tol


def classify_fill(hexc: str, alpha: float):
    """Legend class of a fill; colours within 12/255 per channel count as the same swatch
    (e.g. #00A14B for #00A551 on level 3)."""
    for h, n in UNIQUE_HUES.items():
        if _near(hexc, h):
            return n
    for table in (LEGEND_FILLS, ALT_FILLS):
        cands = [(abs(a - alpha), n) for (h, a), n in table.items() if _near(h, hexc, 0 if table is ALT_FILLS else 12)]
        if cands and min(cands)[0] <= 0.06:
            return min(cands)[1]
    return None


def _sagittal_brain_mask(ref: dict, gap_mm: float = 0.04, min_area_mm2: float = 1.0):
    """Union of every closed outline component (cerebrum, cerebellum and brainstem are separate outlines
    on sagittal plates), filled and shrunk back."""
    shapes = [g.buffer(gap_mm) for g in ref["outlines"]] + ref["fiber_tracts"] + ref["ventricles"]
    if not shapes:
        return Polygon()
    blob = polygonal(unary_union(shapes))
    parts = [Polygon(q.exterior) for q in getattr(blob, "geoms", [blob]) if Polygon(q.exterior).area >= min_area_mm2]
    return polygonal(unary_union(parts).buffer(-gap_mm))


def plate_sources(ai_dir: Path, full_pdf: Path | None) -> list[tuple[Path, int]]:
    """Prefer the individual .ai files; fill missing levels from the full PDF."""
    out, have = [], set()
    for f in sorted(ai_dir.glob("HGEA-Sagittal-*BW.ai")):
        out.append((f, 0))
        have.add(int(re.search(r"(\d{3})BW", f.name).group(1)))
    if full_pdf is not None and full_pdf.exists():
        from pypdf import PdfReader
        r = PdfReader(str(full_pdf))
        for i, pg in enumerate(r.pages):
            m = re.search(r"Sagittal Level\s*(\d+)", pg.extract_text())
            if m and int(m.group(1)) not in have:
                out.append((full_pdf, i))
    return out
