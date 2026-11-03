"""Step 8: robustness report and freeze of the reference 3D ARA HGEA ("reference v1").

Checks on the build used as the reference (hgea_ara3d_full.npz: plates exact + Nissl-guided fill):
    rasterisation   each plate's 25 µm labels vs its 6.25 µm rasterisation of the SVG domains (border
                    distance, µm; agreement): the error added by the 25 µm grid itself
    topology        connected pieces per domain and side (26-connected); pieces under 50 voxels listed
    symmetry        left vs mirrored right: Dice per domain (the plates are drawn per side, so this is a
                    description, not a test)
    provenance      1 = plate slice, 2 = filled between plates
    summary         the accuracy numbers of steps 2, 3, 5 and 7 in one table
Freeze: SHA-256 of every input and of the reference volume, in reference_v1.json.

Output: data_work/reference_v1.json, reference_v1_report.csv, reference_v1_topology.csv,
        data_work/hgea_reference_v1.npz (labels, provenance)
"""

from __future__ import annotations

import hashlib
import json
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd
from scipy import ndimage

from hgea_ara3d.config import ARA3D_DIR, DOMAINS, RAW, SAG_DIR, SVG_DIR, WORK


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def border_um(a: np.ndarray, b: np.ndarray, px_um: float) -> float:
    if not a.any() or not b.any():
        return np.nan
    ea, eb = a & ~ndimage.binary_erosion(a), b & ~ndimage.binary_erosion(b)
    da, db = ndimage.distance_transform_edt(~eb)[ea], ndimage.distance_transform_edt(~ea)[eb]
    return float((da.mean() + db.mean()) / 2 * px_um)


def main() -> None:
    d = np.load(WORK / "hgea_ara3d.npz")
    full = np.load(WORK / "hgea_ara3d_full.npz")
    lab = full["labels"]
    place = pd.read_csv(WORK / "placement.csv").sort_values("z")
    fine_px = float(d["fine_px_um"])
    up = int(round(25.0 / fine_px))

    # rasterisation error: 25 µm plate labels vs the 6.25 µm rasterisation
    ras = []
    for r in place.itertuples():
        fine = d[f"fine_{int(r.level)}"]
        coarse = np.kron(lab[int(r.z)], np.ones((up, up), lab.dtype))
        u = (fine > 0) | (coarse > 0)
        row = dict(level=int(r.level), agreement=float((fine[u] == coarse[u]).mean()))
        bd = [border_um((fine % 100) == i, (coarse % 100) == i, fine_px) for i in range(1, len(DOMAINS) + 1)]
        row["border_um_median"] = float(np.nanmedian(bd))
        ras.append(row)
    ras = pd.DataFrame(ras)

    # topology
    st = np.ones((3, 3, 3), bool)
    topo = []
    for i, name in enumerate(DOMAINS, start=1):
        for side, off in (("right", 0), ("left", 100)):
            m = lab == i + off
            if not m.any():
                continue
            cc, n = ndimage.label(m, structure=st)
            sizes = np.sort(ndimage.sum(m, cc, range(1, n + 1)))[::-1]
            topo.append(dict(domain=name, side=side, voxels=int(m.sum()), pieces=int(n), largest_frac=float(sizes[0] / m.sum()),
                             pieces_under_50=int((sizes < 50).sum())))
    topo = pd.DataFrame(topo)

    # symmetry: left (+100) vs mirrored right
    mir = lab[:, :, ::-1]
    sym = {}
    for i, name in enumerate(DOMAINS, start=1):
        a, b = lab == i, mir == i + 100
        s = a.sum() + b.sum()
        sym[name] = float(2 * (a & b).sum() / s) if s else np.nan

    prov = np.zeros(lab.shape, np.uint8)
    prov[lab > 0] = 2
    prov[place.z.values] = np.where(lab[place.z.values] > 0, 1, 0)
    np.savez_compressed(WORK / "hgea_reference_v1.npz", labels=lab, provenance=prov, domains=np.array(DOMAINS),
                        note="plates exact (Bienkowski coronal SVGs on their 3D ARA slices) + Nissl-guided fill; level 81 CA2 as drawn")

    lopo = pd.read_csv(WORK / "lopo_summary.csv", index_col=0)
    sag = pd.read_csv(WORK / "sagittal_test.csv").set_index("level").loc["all"]
    fc = pd.read_csv(WORK / "fill_comparison.csv").set_index("build")
    rep = [
        ("coronal placement: ARA line distance, median over plates (µm)", place.median_line_um.median()),
        ("coronal placement: hippocampal lines (µm)", place.hpf_median_line_um.median()),
        ("rasterisation: 25 µm vs 6.25 µm plate labels, agreement (median)", ras.agreement.median()),
        ("rasterisation: border distance (µm, median)", ras.border_um_median.median()),
        ("fill LOPO at 2x gap: label accuracy, Nissl-guided (build)", lopo.loc["nissl", "label_accuracy"]),
        ("fill LOPO at 2x gap: label accuracy, plain interpolation", lopo.loc["none", "label_accuracy"]),
        ("fill LOPO at 2x gap: label accuracy, long-axis (best variant)", fc.filter(like="longaxis", axis=0).lopo_label_accuracy.max()),
        ("sagittal placement: ARA line distance (µm, median)", pd.read_csv(WORK / "sagittal_placement.csv").line_median_um.median()),
        ("sagittal test: extent Dice", sag.extent_dice),
        ("sagittal test: field agreement where both label", sag.field_agreement),
        ("sagittal test: domain agreement where both label", sag.domain_agreement),
        ("ceiling: coronal vs sagittal drawings at crossings, domain agreement", sag.crossing_domain_agreement),
        ("topology: domains x sides in one piece (largest piece >= 99%)", float((topo.largest_frac >= 0.99).mean())),
        ("symmetry: median left/right Dice", float(np.nanmedian(list(sym.values())))),
    ]
    rdf = pd.DataFrame(rep, columns=["measure", "value"])
    rdf.to_csv(WORK / "reference_v1_report.csv", index=False)
    topo.to_csv(WORK / "reference_v1_topology.csv", index=False)
    ras.to_csv(WORK / "reference_v1_rasterisation.csv", index=False)

    inputs = sorted([*ARA3D_DIR.rglob("*.raw"), ARA3D_DIR / "structures_2011.csv", *SVG_DIR.glob("*.svg"), *SAG_DIR.glob("*")])
    freeze = dict(name="HGEA 3D ARA reference v1", date=time.strftime("%Y-%m-%d"),
                  decisions=["level 81 lateral CA1 strip kept as CA2 (Bienkowski's definition)",
                             "fill: Nissl-guided (long-axis fill tested, did not beat it)",
                             "between-plate stair-steps left unsmoothed", "sagittal plates: test only"],
                  reference=dict(file="hgea_reference_v1.npz", sha256=sha(WORK / "hgea_reference_v1.npz")),
                  inputs={str(p.relative_to(RAW)): sha(p) for p in inputs},
                  report={m: (round(float(v), 4) if v == v else None) for m, v in rep}, symmetry_dice=sym)
    (WORK / "reference_v1.json").write_text(json.dumps(freeze, indent=1))
    print(rdf.to_string(index=False))
    print("\ntopology: pieces per domain x side (median)", int(topo.pieces.median()), "| largest piece share (min)", round(topo.largest_frac.min(), 3))
    print(topo.sort_values("largest_frac").head(6).to_string(index=False))
    print("\nsymmetry Dice:", {k: round(v, 2) for k, v in sym.items()})


if __name__ == "__main__":
    main()
