"""Step 6 (plan step 1a-b): hippocampal coordinates s (long axis), t (transverse), r (depth) in the 3D ARA.

See hgea_ara3d.coords. Both hemispheres separately.

Input:  data_raw/ara3d, data_work/hgea_ara3d_full.npz (envelope includes the filled HGEA), placement.csv
Output: data_work/hippocampal_coords.npz   env (uint8: 1 right, 2 left), s, t, r (float16, NaN outside),
                                           pole and surface masks (bd_*: uint8 per side bits)
        data_work/coords_check.csv         per coronal plate: s range on the plate, and the s spread of each
                                           HGEA d/i/v boundary on that plate (are boundaries near iso-s?)
        data_work/qc/coords_*.png          s, t, r on plate slices, a horizontal and a sagittal cut; poles
"""

from __future__ import annotations

import time

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from hgea_ara3d import coords as C
from hgea_ara3d.config import DOMAIN_ID, WORK
from hgea_ara3d.space import ARA3D

# d/i/v boundaries checked for being near iso-s on each plate (pairs of neighbouring domains)
BOUNDARIES = [("CA1d", "CA1i"), ("CA1i", "CA1v"), ("CA1v", "CA1vv"), ("DGd/CA3d", "DGi/CA3i"), ("DGi/CA3i", "DGv/CA3vv")]


def main() -> None:
    ara = ARA3D()
    full = np.load(WORK / "hgea_ara3d_full.npz")["labels"]
    plates = np.load(WORK / "hgea_ara3d.npz")["labels"]
    place = pd.read_csv(WORK / "placement.csv").sort_values("z")
    out = {k: np.full(full.shape, np.nan, np.float32) for k in "str"}
    env = np.zeros(full.shape, np.uint8)
    bds = {}
    for side, code in ((1, 1), (-1, 2)):
        t0 = time.time()
        c = C.hippocampal_coords(ara, full, side)
        for k in "str":
            out[k][c["env"]] = c[k][c["env"]]
        env[c["env"]] = code
        for k, v in c.items():
            if k.startswith("bd_"):
                bds[k] = bds.get(k, np.zeros(full.shape, np.uint8)) | (v.astype(np.uint8) * code)
        print(f"side {side:+d}: envelope {int(c['env'].sum()):,} voxels; HGEA voxels inside "
              f"{int(((full > 0) & c['env']).sum()):,} of {int(((full > 0) & ((full >= 100) == (side < 0))).sum()):,}; "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(WORK / "hippocampal_coords.npz", env=env, **{k: v.astype(np.float16) for k, v in out.items()}, **bds)

    # check: do the plates' d/i/v boundaries sit near iso-s?
    rows = []
    for r in place.itertuples():
        z = int(r.z)
        lab = plates[z]
        s = out["s"][z]
        row = dict(level=int(r.level), z=z, s_min=float(np.nanmin(np.where(lab > 0, s, np.nan))) if (lab > 0).any() else np.nan,
                   s_max=float(np.nanmax(np.where(lab > 0, s, np.nan))) if (lab > 0).any() else np.nan)
        for a, b in BOUNDARIES:
            ia, ib = DOMAIN_ID[a], DOMAIN_ID[b]
            m = ((lab % 100) == ia) & ndimage.binary_dilation((lab % 100) == ib)
            vals = s[m & ~np.isnan(s)]
            row[f"{a}|{b}_s_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{a}|{b}_s_iqr"] = float(np.subtract(*np.percentile(vals, [75, 25]))) if len(vals) > 3 else np.nan
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_csv(WORK / "coords_check.csv", index=False)
    for a, b in BOUNDARIES:
        col = f"{a}|{b}_s_median"
        v = cdf[col].dropna()
        print(f"boundary {a}|{b}: on {len(v)} plates, s median {v.median():.3f}, spread across plates (IQR) "
              f"{np.subtract(*np.percentile(v, [75, 25])) if len(v) > 3 else float('nan'):.3f}; within-plate IQR median "
              f"{cdf[f'{a}|{b}_s_iqr'].median():.3f}")

    # QC images
    def show(ax, img, title, cmap="viridis"):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest"); ax.set_title(title, fontsize=8); ax.axis("off")
    zs = place.z.values[[3, 10, 16, 23, 29]]
    fig, axs = plt.subplots(3, len(zs), figsize=(3.2 * len(zs), 7.5))
    for j, z in enumerate(zs):
        for i, k in enumerate("str"):
            show(axs[i, j], out[k][z][40:260, 40:420], f"{k} · slice {z}", "turbo" if k == "s" else "viridis")
    fig.tight_layout(); fig.savefig(WORK / "qc" / "coords_plates.png", dpi=90); plt.close(fig)
    hp = np.nonzero(env.any(axis=(0, 2)))[0]
    dv = int(np.median(hp))
    lr = int(np.percentile(np.nonzero(env.any(axis=(0, 1)))[0], 75))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, k in enumerate("str"):
        show(axs[0, i], out[k][200:420, dv, :], f"{k} · horizontal DV {dv}", "turbo" if k == "s" else "viridis")
        show(axs[1, i], out[k][200:420, :, lr].T, f"{k} · sagittal LR {lr}", "turbo" if k == "s" else "viridis")
    fig.tight_layout(); fig.savefig(WORK / "qc" / "coords_cuts.png", dpi=90); plt.close(fig)
    # poles: max projections of the septal and temporal sets over the Nissl
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    nis = ara.nissl
    for ax, (axis, name) in zip(axs, [(1, "horizontal projection"), (2, "sagittal projection"), (0, "coronal projection")]):
        base = nis.max(axis=axis).astype(float)
        img = np.dstack([base / 255.0] * 3)
        for key, colr in (("bd_septal", [1, 0.2, 0.2]), ("bd_temporal", [0.2, 0.6, 1]), ("bd_dg", [0.3, 1, 0.3]), ("bd_sub", [1, 0.8, 0.2])):
            m = bds[key].max(axis=axis) > 0
            img[m] = img[m] * 0.3 + np.array(colr) * 0.7
        ax.imshow(img if axis != 2 else img.transpose(1, 0, 2)); ax.set_title(f"{name}: red septal pole, blue temporal pole, green DG (t=0), yellow SUB (t=1)", fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(WORK / "qc" / "coords_poles.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
