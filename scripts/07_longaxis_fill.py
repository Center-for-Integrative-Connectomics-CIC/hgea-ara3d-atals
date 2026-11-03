"""Step 7: fill between plates in hippocampal coordinates (s, t, r), and test it.

For a voxel between plates:
    field   (CA1 / CA2 / DG-CA3 band / SUB) and HGEA extent: from the Nissl-guided fill (step 3), since field
            outlines follow the tissue;
    domain  within that field: distance-weighted vote of the K nearest plate samples in scaled (s, t, r),
            same hemisphere, from every plate (not only the two neighbours).
Plate slices keep their own labels.

Tests (the same ones as the Nissl fill, plus the sagittal drawings):
    leave-one-plate-out: plate k hidden from the samples, and its field map taken from the Nissl
        prediction made from its two neighbours (the 2x gap of step 3), so it is directly comparable;
    sagittal: agreement with the 20 sagittal drawings on their planes (step 5 placement).
A few coordinate scalings are compared; all are reported, not only the best.

Input:  data_work/hgea_ara3d.npz, hgea_ara3d_full.npz, lopo.csv, hippocampal_coords.npz, sagittal_stack.json,
        sagittal_plates.pkl, data_raw/ara3d
Output: data_work/hgea_ara3d_longaxis.npz   filled volume (same format as hgea_ara3d_full.npz)
        data_work/lopo_longaxis.csv          per held-out plate x variant
        data_work/fill_comparison.csv         LOPO and sagittal summary: plain, Nissl, long-axis variants
"""

from __future__ import annotations

import json
import pickle
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hgea_ara3d import interp as I
from hgea_ara3d import sagittal as SG
from hgea_ara3d.config import DOMAINS, WORK
from hgea_ara3d.space import ARA3D

K = 15
FIELD = {i + 1: ("CA1" if d.startswith("CA1") else "CA2" if d == "CA2" else "SUB" if d.startswith("SUB") else "DGCA3")
         for i, d in enumerate(DOMAINS)}
# (s, t, r) weights; one unit of scaled distance ~ one unit of s
VARIANTS = {"s": (1.0, 0.0, 0.0), "s+t": (1.0, 0.3, 0.0), "s+t+r": (1.0, 0.3, 0.15)}


class Voter:
    """K-nearest plate samples in scaled (s, t, r), per hemisphere and field."""

    def __init__(self, coords: np.ndarray, labels: np.ndarray, w):
        self.trees = {}
        w = np.asarray(w)
        keep = w > 0
        self.w, self.keep = w, keep
        for side in (0, 1):
            for f in set(FIELD.values()):
                m = ((labels >= 100) == bool(side)) & np.isin(labels % 100, [i for i, ff in FIELD.items() if ff == f])
                if m.sum() >= 3:
                    X = coords[m][:, keep] * w[keep]
                    self.trees[(side, f)] = (cKDTree(X), labels[m] % 100)

    def vote(self, coords: np.ndarray, field_label: np.ndarray, side: np.ndarray) -> np.ndarray:
        out = field_label.copy()
        for (sd, f), (tree, lab) in self.trees.items():
            m = (side == sd) & np.isin(field_label % 100, [i for i, ff in FIELD.items() if ff == f]) & ~np.isnan(coords).any(1)
            if not m.any():
                continue
            k = min(K, len(lab))
            d, j = tree.query(coords[m][:, self.keep] * self.w[self.keep], k=k)
            d, j = d.reshape(-1, k), j.reshape(-1, k)
            wts = 1.0 / (d + 1e-3)
            cand = lab[j]
            score = np.zeros((m.sum(), len(DOMAINS) + 1))
            for c in range(k):
                np.add.at(score, (np.arange(m.sum()), cand[:, c]), wts[:, c])
            out[m] = score.argmax(1) + (100 if sd else 0)
        return out


def main() -> None:
    ara = ARA3D()
    d = np.load(WORK / "hgea_ara3d.npz")
    plates_vol = d["labels"].astype(np.int16)
    full_nissl = np.load(WORK / "hgea_ara3d_full.npz")["labels"].astype(np.int16)
    C = np.load(WORK / "hippocampal_coords.npz")
    stv = np.stack([C["s"], C["t"], C["r"]], -1).astype(np.float32)
    place = pd.read_csv(WORK / "placement.csv").sort_values("z")
    zs, levels = place.z.tolist(), place.level.tolist()

    pz = np.nonzero(plates_vol)
    samp_lab = plates_vol[pz]
    samp_xyz = np.stack(pz, 1)
    samp_c = stv[pz]
    ok = ~np.isnan(samp_c).any(1)
    samp_lab, samp_xyz, samp_c = samp_lab[ok], samp_xyz[ok], samp_c[ok]
    print(f"{len(samp_lab):,} plate samples with coordinates", flush=True)

    # leave-one-plate-out, field maps from the Nissl prediction at the 2x gap (as in step 3)
    rr, cc = np.nonzero(plates_vol.any(0))
    box = ((max(0, rr.min() - 16), rr.max() + 16), (max(0, cc.min() - 16), cc.max() + 16))
    (r0, r1), (c0, c1) = box
    guide = I.Guide("nissl", ara, box)
    rows = []
    t = time.time()
    for k in range(1, len(zs) - 1):
        za, z, zb = zs[k - 1], zs[k], zs[k + 1]
        truth = plates_vol[z, r0:r1, c0:c1]
        nissl_pred = I.predict(guide, plates_vol[za, r0:r1, c0:c1], za, plates_vol[zb, r0:r1, c0:c1], zb, z)
        rows.append(dict(level=levels[k], variant="nissl", **I.scores(truth, nissl_pred, DOMAINS)))
        keep = samp_xyz[:, 0] != z
        yy, xx = np.nonzero(nissl_pred)
        side = (nissl_pred[yy, xx] >= 100).astype(int)
        cz = stv[z, yy + r0, xx + c0]
        for name, w in VARIANTS.items():
            v = Voter(samp_c[keep], samp_lab[keep], w)
            pred = nissl_pred.copy()
            pred[yy, xx] = v.vote(cz, nissl_pred[yy, xx], side)
            rows.append(dict(level=levels[k], variant=f"longaxis[{name}]", **I.scores(truth, pred, DOMAINS)))
        print(f"held-out level {levels[k]}: " + ", ".join(f"{x['variant']} {x['label_accuracy']:.3f}" for x in rows if x["level"] == levels[k])
              + f"  [{time.time() - t:.0f}s]", flush=True)
    lopo = pd.DataFrame(rows)
    lopo.to_csv(WORK / "lopo_longaxis.csv", index=False)
    plain = pd.read_csv(WORK / "lopo.csv")
    plain = plain[(plain.guide == "none") & plain.level.isin(lopo.level)]
    summ = lopo.groupby("variant")[["label_accuracy", "hgea_dice"]].median()
    summ.loc["plain (no guide)"] = plain[["label_accuracy", "hgea_dice"]].median()
    best = max(VARIANTS, key=lambda n: summ.loc[f"longaxis[{n}]", "label_accuracy"])
    print(summ.round(3), f"\nbuild variant: longaxis[{best}]", flush=True)

    # build: re-assign domains on the non-plate slices inside the Nissl fill's extent
    v = Voter(samp_c, samp_lab, VARIANTS[best])
    out = full_nissl.copy()
    plate_z = set(zs)
    m = (full_nissl > 0)
    m[list(plate_z)] = False
    idx = np.nonzero(m)
    out[idx] = v.vote(stv[idx], full_nissl[idx], (full_nissl[idx] >= 100).astype(int))
    np.savez_compressed(WORK / "hgea_ara3d_longaxis.npz", labels=out.astype(np.uint8), guide=f"nissl field + longaxis[{best}]",
                        domains=np.array(DOMAINS), levels=np.array(levels), z=np.array(zs))
    changed = int((out != full_nissl).sum())
    print(f"built: {int((out > 0).sum()):,} HGEA voxels; {changed:,} re-assigned vs the Nissl fill ({changed / (full_nissl > 0).sum():.1%})", flush=True)

    # sagittal test for both builds
    st = json.loads((WORK / "sagittal_stack.json").read_text())
    g, loc = st["global_"], {int(k): v for k, v in st["local"].items()}
    splates = pickle.load(open(WORK / "sagittal_plates.pkl", "rb"))
    comp = []
    for name, vol in (("nissl", full_nissl), (f"longaxis[{best}]", out)):
        rws, _ = SG.agreement(splates, g, loc, st["side"], vol, DOMAINS)
        df = pd.DataFrame(rws)
        nb = df.both_pts.sum()
        comp.append(dict(build=name, lopo_label_accuracy=summ.loc[name, "label_accuracy"], lopo_hgea_dice=summ.loc[name, "hgea_dice"],
                         sagittal_extent_dice=2 * nb / (df.sagittal_pts.sum() + df.volume_pts.sum()),
                         sagittal_field_agreement=float((df.field_agreement * df.both_pts).sum() / nb),
                         sagittal_domain_agreement=float((df.domain_agreement * df.both_pts).sum() / nb)))
    for n in VARIANTS:
        if f"longaxis[{n}]" != f"longaxis[{best}]":
            comp.append(dict(build=f"longaxis[{n}]", lopo_label_accuracy=summ.loc[f"longaxis[{n}]", "label_accuracy"],
                             lopo_hgea_dice=summ.loc[f"longaxis[{n}]", "hgea_dice"]))
    comp.append(dict(build="plain (no guide)", lopo_label_accuracy=summ.loc["plain (no guide)", "label_accuracy"],
                     lopo_hgea_dice=summ.loc["plain (no guide)", "hgea_dice"]))
    cdf = pd.DataFrame(comp)
    cdf.to_csv(WORK / "fill_comparison.csv", index=False)
    print(cdf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
