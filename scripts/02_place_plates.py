"""Step 2: put the 33 HGEA coronal plates on their own slices of the 3D ARA (CCFv2 2011 Nissl).

No CCFv3 or CCFv3BBP is involved: the target is the Allen 2011 reconstruction of the ARA brain itself,
in which every ARA coronal plate is one AP slice. See hgea_ara3d.place for the method.

Input:  data_work/plates.pkl (step 1), src/hgea_ara3d/plate_sections.csv (Allen section numbers),
        data_raw/ara3d (3D ARA Nissl, 2011 annotation and ontology)
Output: data_work/
            placement.csv        per plate: slice z (from one section -> slice line for the whole series),
                                 distance of its ARA lines to the annotation edges (all and hippocampal),
                                 in-plane parameters
            stack_line.json      z = a + b * Allen section number
            slice_profiles.csv   per plate x searched slice: mean line distance (the z search)
            hgea_ara3d.npz       `labels` (AP, DV, LR) uint8 HGEA domain ids on the plate slices only
                                 (left +100), `levels`, `z`, `domains`; and per plate a 10 µm 2D label
                                 image `fine_LL` on that slice
            qc/level_LL.png      Nissl slice with annotation edges and the HGEA domains
            qc/montage.png       all plates
"""

from __future__ import annotations

import json
import pickle

import _path  # noqa: F401
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.ops import unary_union

from hgea_ara3d import place as PL
from hgea_ara3d.config import COLOURS, DOMAIN_ID, PLATE_SECTIONS, WORK
from hgea_ara3d.space import ARA3D, SHAPE

OUT = WORK
FINE = 4   # 2D labels and QC images at 25 / FINE = 6.25 µm per pixel


def hpf_box(plate, pad=0.3):
    g = unary_union([geom for d in plate.domains.values() for geom in d.values()])
    x0, y0, x1, y1 = g.bounds
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def qc_image(ara: ARA3D, pl, fine: np.ndarray, path: Path, small: bool = False):
    z = pl.z
    nissl = ara.nissl[z].astype(float)
    # upsample to the fine grid for display; annotation edges as 1-pixel lines on that grid
    nis = ndimage.zoom(nissl, FINE, order=1)
    lab = np.kron(ara.annotation[z], np.ones((FINE, FINE), np.uint32))
    edg = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
    img = np.dstack([1 - nis / 255.0 * 0.85] * 3)            # cells dark, as in a Nissl stain
    img[edg] = img[edg] * 0.4 + np.array([0.15, 0.45, 0.85]) * 0.6
    for name, i in DOMAIN_ID.items():
        rgb = np.array([int(COLOURS[name][k:k + 2], 16) / 255 for k in (1, 3, 5)])
        m = (fine % 100) == i
        img[m] = img[m] * 0.45 + rgb * 0.55
        b = m & ~ndimage.binary_erosion(m)
        img[b] = rgb * 0.7
    rr, cc = np.nonzero(fine)
    if not small and len(rr):
        pad = 40 * FINE // 4
        img = img[max(0, rr.min() - pad):rr.max() + pad, max(0, cc.min() - pad):cc.max() + pad]
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qc").mkdir(exist_ok=True)
    ara = ARA3D()
    ann = ara.annotation
    plates = {p.level: p for p in pickle.load(open(WORK / "plates.pkl", "rb"))}
    sec = dict(pd.read_csv(PLATE_SECTIONS)[["level", "section_number"]].values)

    rows, prof_rows, fines = [], [], {}
    vol = np.zeros(SHAPE, np.uint8)
    q0 = [0.98, 1.01, 0.0, ara.ml_mid_vox, 15.4]   # from a hand-checked fit of level 80; same for every plate
    free = {}
    for lv in sorted(plates):
        free[lv] = PL.place_plate(plates[lv], sec[lv], ann, q0, hpf_box_mm=hpf_box(plates[lv]))
        print(f"level {lv}: section {sec[lv]}, best single slice {free[lv].z}", flush=True)
    a, b, excess = PL.stack_line(list(free.values()))
    print(f"stack line: z = {a:.2f} + {b:.4f} * section (summed excess {excess:.1f} µm over {len(free)} plates)")

    for lv in sorted(plates):
        p, f = plates[lv], free[lv]
        pl = PL.finalise(p, f, int(round(a + b * f.section)), ann, hpf_box_mm=hpf_box(p))
        fine = PL.domain_labels(p, pl.q, SHAPE[1:], DOMAIN_ID, upsample=FINE)
        coarse = PL.domain_labels(p, pl.q, SHAPE[1:], DOMAIN_ID, upsample=1)
        vol[pl.z] = np.where(coarse > 0, coarse, vol[pl.z]).astype(np.uint8)
        fines[f"fine_{lv}"] = fine
        s_, asp, th, tx, ty = pl.q
        prof = dict(f.profile)
        rows.append(dict(level=lv, bregma_mm=p.bregma_mm, section_number=pl.section, z=pl.z, ap_um=pl.z * 25.0,
                         z_single_best=f.z, excess_vs_single_best_um=prof[pl.z] - prof[f.z],
                         brain_inside_frac=PL.brain_dice(pl.q, PL.brain_points(p), ann[pl.z] > 0),
                         mean_line_um=pl.mean_um, median_line_um=pl.median_um, p90_line_um=pl.p90_um,
                         hpf_median_line_um=pl.hpf_median_um, scale=s_, aspect=asp, rotation_deg=np.degrees(th),
                         lr_shift_vox=tx, dv_shift_vox=ty, hgea_pixels_25um=int((coarse > 0).sum())))
        prof_rows += [dict(level=lv, z=z, mean_line_um=v) for z, v in f.profile]
        print(f"level {lv}: z {pl.z} (single best {f.z}), ARA line distance median {pl.median_um:.1f} µm "
              f"(hippocampus {pl.hpf_median_um:.1f}), p90 {pl.p90_um:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "placement.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(OUT / "slice_profiles.csv", index=False)
    (OUT / "stack_line.json").write_text(json.dumps(dict(a=a, b=b, summed_excess_um=excess, um_per_section=b * 25.0)))
    np.savez_compressed(OUT / "hgea_ara3d.npz", labels=vol, levels=df.level.values, z=df.z.values,
                        domains=np.array(list(DOMAIN_ID)), fine_px_um=25.0 / FINE, **fines)

    for lv, r in df.set_index("level").iterrows():
        pl = PL.Placement(lv, int(r.section_number), int(r.z), None, r.mean_line_um, r.median_line_um, r.p90_line_um,
                          r.hpf_median_line_um, [], 0.0)
        img = qc_image(ara, pl, fines[f"fine_{lv}"], None)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"HGEA level {lv} on 3D ARA slice z={pl.z} (section {pl.section}); ARA line fit median {r.median_line_um:.1f} µm, "
                     f"hippocampus {r.hpf_median_line_um:.1f} µm", fontsize=9)
        ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()]
        ax.legend(handles, COLOURS.keys(), fontsize=7, loc="upper right", ncol=2, framealpha=0.8)
        fig.tight_layout(); fig.savefig(OUT / "qc" / f"level_{lv}.png", dpi=110); plt.close(fig)

    n = len(df); cols = 6; rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(cols * 3.2, rws * 2.3))
    for ax, (lv, r) in zip(axs.ravel(), df.set_index("level").iterrows()):
        pl = PL.Placement(lv, 0, int(r.z), None, 0, 0, 0, 0, [], 0)
        ax.imshow(qc_image(ara, pl, fines[f"fine_{lv}"], None), interpolation="nearest")
        ax.set_title(f"L{lv} z{int(r.z)} · {r.hpf_median_line_um:.0f} µm", fontsize=8); ax.axis("off")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "qc" / "montage.png", dpi=90); plt.close(fig)

    print(f"line distance median over plates: {df.median_line_um.median():.1f} um (hippocampus {df.hpf_median_line_um.median():.1f} um), "
          f"worst plate {df.loc[df.median_line_um.idxmax(), 'level']} at {df.median_line_um.max():.1f} um")
    print("consecutive plate spacing (slices):", np.diff(df.z.values).tolist())


if __name__ == "__main__":
    main()
