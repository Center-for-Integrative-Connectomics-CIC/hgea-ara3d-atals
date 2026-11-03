"""Step 3: fill the 3D ARA between the HGEA plate slices (step 2), and measure how well it works.

1. Leave-one-plate-out: every inner plate is hidden and predicted from its two neighbours (a gap twice
   the real one, so a conservative test), with each guide: none (plain shape interpolation), Nissl, and
   the 2011 annotation (leaks plate information; shown for comparison only). The guide used for the
   build is chosen by this test, among the guides that do not leak.
2. Build: every slice from the first to the last plate; plate slices keep their own labels.

Input:  data_work/hgea_ara3d.npz and placement.csv (step 2), data_raw/ara3d
Output: data_work/
            hgea_ara3d_full.npz   `labels` (AP, DV, LR) uint8, filled between plates (left +100), `guide`
            lopo.csv              per held-out plate x guide: label accuracy, HGEA Dice, per-domain Dice
                                  and border distance (µm)
            lopo_summary.csv      medians per guide
            qc/full_cuts.png      horizontal and sagittal cuts: plate slices only vs filled
(hgea_ara3d.npz, the plate-slice-only version, is left as it is.)
"""

from __future__ import annotations

import time

import _path  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hgea_ara3d import interp as I
from hgea_ara3d.config import COLOURS, WORK
from hgea_ara3d.space import ARA3D

OUT = WORK
GUIDES = ["none", "nissl", "annotation"]
LEAKS = {"annotation"}


def main() -> None:
    ara = ARA3D()
    d = np.load(OUT / "hgea_ara3d.npz")
    plates_vol = d["labels"]
    domains = list(d["domains"])
    place = pd.read_csv(OUT / "placement.csv").sort_values("z")
    zs, levels = place.z.tolist(), place.level.tolist()

    # one crop box around every plate's HGEA labels, with a margin, for all registrations
    any_lab = plates_vol.any(0)
    rr, cc = np.nonzero(any_lab)
    m = 16
    box = ((max(0, rr.min() - m), min(any_lab.shape[0], rr.max() + m)), (max(0, cc.min() - m), min(any_lab.shape[1], cc.max() + m)))
    (r0, r1), (c0, c1) = box
    plate_lab = {z: plates_vol[z, r0:r1, c0:c1].astype(np.int16) for z in zs}
    print(f"crop box DV {r0}-{r1}, LR {c0}-{c1}; {len(zs)} plates", flush=True)

    guides = {g: I.Guide(g, ara, box) for g in GUIDES}

    rows = []
    t = time.time()
    for k in range(1, len(zs) - 1):
        za, z, zb = zs[k - 1], zs[k], zs[k + 1]
        for g, gd in guides.items():
            pred = I.predict(gd, plate_lab[za], za, plate_lab[zb], zb, z)
            rows.append(dict(level=levels[k], z=z, gap_slices=zb - za, guide=g, **I.scores(plate_lab[z], pred, domains)))
        r = [x for x in rows if x["level"] == levels[k]]
        print(f"held-out level {levels[k]} (gap {zb - za}): " + ", ".join(f"{x['guide']} {x['label_accuracy']:.3f}" for x in r)
              + f"  [{time.time() - t:.0f}s]", flush=True)
    lopo = pd.DataFrame(rows)
    lopo.to_csv(OUT / "lopo.csv", index=False)
    summ = lopo.drop(columns=["level", "z", "gap_slices"]).groupby("guide").median()
    summ.to_csv(OUT / "lopo_summary.csv")
    print(summ[["label_accuracy", "hgea_dice"]].round(3))
    usable = [g for g in GUIDES if g not in LEAKS]
    best = max(usable, key=lambda g: summ.loc[g, "label_accuracy"])
    print(f"build guide: {best}", flush=True)

    full = np.zeros_like(plates_vol)
    gd = guides[best]
    for k in range(len(zs) - 1):
        za, zb = zs[k], zs[k + 1]
        full[za, r0:r1, c0:c1] = plate_lab[za]
        for z in range(za + 1, zb):
            full[z, r0:r1, c0:c1] = I.predict(gd, plate_lab[za], za, plate_lab[zb], zb, z)
    full[zs[-1], r0:r1, c0:c1] = plate_lab[zs[-1]]
    np.savez_compressed(OUT / "hgea_ara3d_full.npz", labels=full.astype(np.uint8), guide=best, domains=np.array(domains),
                        levels=np.array(levels), z=np.array(zs))
    print(f"filled slices {zs[0]}-{zs[-1]}; HGEA voxels {int((full > 0).sum()):,} ({int((plates_vol > 0).sum()):,} on plates)")

    # QC: horizontal and sagittal cuts through the hippocampus, plate slices only vs filled
    rgb = np.zeros((256, 3))
    for i, name in enumerate(domains, start=1):
        rgb[i] = rgb[i + 100] = [int(COLOURS[name][k:k + 2], 16) / 255 for k in (1, 3, 5)]
    lab_any = full.any(axis=(0, 2))
    dv = int(np.median(np.nonzero(lab_any)[0]))                      # horizontal cut through the hippocampus
    lr = int(np.percentile(np.nonzero(full.any(axis=(0, 1)))[0], 25))  # sagittal cut through one hemisphere
    a0, a1 = zs[0] - 6, zs[-1] + 6
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for row, (name, vol) in enumerate([("plate slices only", plates_vol), (f"filled ({best}-guided)", full)]):
        for col, (cut, nis, lab, ttl) in enumerate([
                ("horizontal", ara.nissl[a0:a1, dv, :], vol[a0:a1, dv, :], f"horizontal DV {dv}"),
                ("sagittal", ara.nissl[a0:a1, :, lr].T, vol[a0:a1, :, lr].T, f"sagittal LR {lr}")]):
            img = np.dstack([1 - nis / 255.0 * 0.85] * 3)
            msk = lab > 0
            img[msk] = img[msk] * 0.35 + rgb[lab[msk]] * 0.65
            ax[row, col].imshow(img, interpolation="nearest", aspect="equal")
            ax[row, col].set_title(f"{name} · {ttl} (AP {a0}-{a1})", fontsize=9)
            ax[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "qc" / "full_cuts.png", dpi=100)
    plt.close(fig)



if __name__ == "__main__":
    main()
