"""Step 4: bake the web page data (docs/data) from the 3D ARA and the HGEA build (steps 2-3).

Output: docs/data/
    nissl.u8.gz          3D ARA Nissl, (AP, DV, LR) = 528 x 320 x 456, 25 µm, uint8
    ara.u16.gz           2011 ARA annotation as indices into ontology.json (0 = outside)
    hgea_full.u8.gz      HGEA filled between plates (step 3); left hemisphere +100
    hgea_plates.u8.gz    HGEA on the 33 plate slices only (step 2)
    ontology.json        [{acronym, name, rgb}] for the annotation indices (index 0 = outside)
    meta.json            grid, domains and colours, plates (level, slice, fit, SVG -> slice affine), accuracy, sources
    svg/                 the original HGEA coronal SVG plates (Bienkowski et al. 2018), unchanged
All arrays are gzip-compressed little-endian, C order with LR fastest.
"""

from __future__ import annotations

import gzip
import json
import pickle
import shutil

import numpy as np
import pandas as pd

import _path  # noqa: F401
from hgea_ara3d import place as PL
from hgea_ara3d.config import COLOURS, DOMAINS, SITE_DATA, SVG_DIR, WORK
from hgea_ara3d.space import ARA3D, SHAPE, VOXEL_UM


def write(name: str, a: np.ndarray) -> dict:
    path = SITE_DATA / name
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(a).tobytes())
    return dict(file=name, dtype=str(a.dtype), shape=list(a.shape), bytes_gz=path.stat().st_size)


def svg_affine(plate, q) -> list[float]:
    """SVG user units (px, py) -> slice voxel coordinates (lr, dv), as [A, B, C, D, E, F] with
    lr = A px + B py + C, dv = D px + E py + F; exactly plate.to_mm followed by place.to_slice."""
    px = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    dvlr = PL.to_slice(q, plate.to_mm(px))
    lr, dv = dvlr[:, 1], dvlr[:, 0]
    A, B = (lr[1] - lr[0]) / 1000, (lr[2] - lr[0]) / 1000
    D, E = (dv[1] - dv[0]) / 1000, (dv[2] - dv[0]) / 1000
    return [float(v) for v in (A, B, lr[0], D, E, dv[0])]


def brain_outline(plate, q) -> list[float]:
    """The plate's brain outline in slice voxel coordinates, flat [lr0, dv0, lr1, dv1, ...] (simplified to 0.1 voxel)."""
    from shapely.geometry import Polygon
    ring = Polygon(plate.reference["brain_mask"][0].exterior).simplify(0.0025).exterior
    dvlr = PL.to_slice(q, np.asarray(ring.coords))
    return [round(float(v), 2) for pt in dvlr[:, ::-1] for v in pt]


def main() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    ara = ARA3D()
    onto = ara.ontology
    ids = np.unique(ara.annotation)
    index = np.zeros(int(ids.max()) + 1, np.uint16)
    index[ids] = np.arange(len(ids), dtype=np.uint16)
    ann = index[ara.annotation]
    entries = [dict(acronym="—", name="outside the brain", rgb=[0, 0, 0])]
    for sid in ids[1:]:
        r = onto.loc[sid]
        hexc = str(r.color_hex_triplet)
        entries.append(dict(acronym=r.acronym, name=r["name"], rgb=[int(hexc[k:k + 2], 16) for k in (0, 2, 4)]))
    (SITE_DATA / "ontology.json").write_text(json.dumps(entries, separators=(",", ":")))

    full = np.load(WORK / "hgea_ara3d_full.npz")
    plates = np.load(WORK / "hgea_ara3d.npz")
    place = pd.read_csv(WORK / "placement.csv")
    plates_geo = {p.level: p for p in pickle.load(open(WORK / "plates.pkl", "rb"))}
    (SITE_DATA / "svg").mkdir(exist_ok=True)
    svg_of = {}
    for lv, p in plates_geo.items():
        shutil.copy(SVG_DIR / p.file if (SVG_DIR / str(p.file)).exists() else next(SVG_DIR.glob(f"HGEA-Coronal-{lv:03d}*.svg")), SITE_DATA / "svg" / f"level_{lv}.svg")
        svg_of[lv] = f"svg/level_{lv}.svg"
    lopo = pd.read_csv(WORK / "lopo_summary.csv", index_col=0)
    line = json.loads((WORK / "stack_line.json").read_text())
    files = dict(nissl=write("nissl.u8.gz", ara.nissl), ara=write("ara.u16.gz", ann),
                 hgea_full=write("hgea_full.u8.gz", full["labels"].astype(np.uint8)),
                 hgea_plates=write("hgea_plates.u8.gz", plates["labels"].astype(np.uint8)))
    meta = dict(
        grid=dict(shape=list(SHAPE), axes=["AP", "DV", "LR"], voxel_um=VOXEL_UM, space="3D ARA (Allen CCFv2 2011)"),
        files=files,
        domains=[dict(id=i + 1, name=d, colour=COLOURS[d]) for i, d in enumerate(DOMAINS)],
        plates=[dict(level=int(r.level), z=int(r.z), section=int(r.section_number), bregma_mm=float(r.bregma_mm),
                     line_median_um=round(float(r.median_line_um), 1), hpf_median_um=round(float(r.hpf_median_line_um), 1),
                     svg=svg_of[int(r.level)],
                     svg_affine=svg_affine(plates_geo[int(r.level)], [r.scale, r.aspect, np.radians(r.rotation_deg), r.lr_shift_vox, r.dv_shift_vox]),
                     brain_outline=brain_outline(plates_geo[int(r.level)], [r.scale, r.aspect, np.radians(r.rotation_deg), r.lr_shift_vox, r.dv_shift_vox]))
                for r in place.sort_values("z").itertuples()],
        accuracy=dict(
            placement_line_median_um=round(float(place.median_line_um.median()), 1),
            placement_hpf_median_um=round(float(place.hpf_median_line_um.median()), 1),
            stack_line=dict(a=round(line["a"], 3), b=round(line["b"], 4)),
            fill_guide=str(full["guide"]),
            lopo_label_accuracy={g: round(float(lopo.loc[g, "label_accuracy"]), 3) for g in lopo.index},
            lopo_hgea_dice={g: round(float(lopo.loc[g, "hgea_dice"]), 3) for g in lopo.index}),
        sources=[
            "Bienkowski M.S. et al. Integration of gene expression and brain-wide connectivity reveals the multiscale "
            "organization of mouse hippocampal networks. Nat. Neurosci. 21, 1628-1643 (2018): HGEA coronal plates.",
            "Allen Institute for Brain Science, Allen Reference Atlas (2008) and its 3D reconstruction (CCFv2, 2011).",
        ],
    )
    (SITE_DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    for k, f in files.items():
        print(f"{k:12s} {f['bytes_gz'] / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
