"""Step 5 (plan step 2): place the 20 sagittal HGEA plates in the coronal 3D ARA, and use them as a test.

The sagittal plates are drawn on a different ARA brain, so here they are a test of the coronal build,
not a build input. See hgea_ara3d.sagittal for the placement model.

1. Placement: stack fit (shared cut) then small per-plate corrections; accuracy = distance from each
   plate's printed ARA lines to the 3D annotation edges, all lines and hippocampal lines.
2. Test of the coronal build on every sagittal plane: label agreement of the filled HGEA (step 3) with the
   sagittal drawing, per plate and per domain, inside the union of the two (sides: the sagittal plates are
   compared with the right hemisphere; the 2011 annotation is mirror-symmetric, so the side does not
   change the placement).
3. Series agreement where they cross: sagittal label vs coronal plate label at points within half a voxel
   of a coronal plate slice.

Input:  data_work/sagittal_plates.pkl (parsed from data_raw/hgea_sagittal_ai), placement.csv,
        hgea_ara3d.npz, hgea_ara3d_full.npz, data_raw/ara3d
Output: data_work/
            sagittal_placement.csv    per plate: fit parameters, line distance (median, hippocampus, p90)
            sagittal_test.csv         per plate (and 'all'): filled-HGEA agreement, coronal-crossing agreement
            sagittal_test_domains.csv per domain: recall / precision / Dice of the filled HGEA on sagittal planes
            sagittal_planes.npz       `labels` (AP, DV, LR) uint8: sagittal drawings on their planes (nearest voxel)
            qc/sagittal_LL.png, qc/sagittal_montage.png
"""

from __future__ import annotations

import json
import pickle

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely import contains_xy
from shapely.ops import unary_union

from hgea_ara3d import sagittal as SG
from hgea_ara3d.config import COLOURS, DOMAIN_ID, DOMAINS, WORK
from hgea_ara3d.space import ARA3D, SHAPE

SIDE = 1


def hpf_box(plate, pad=0.3):
    g = unary_union(list(plate.domains.values()))
    x0, y0, x1, y1 = g.bounds
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def main() -> None:
    (WORK / "qc").mkdir(parents=True, exist_ok=True)
    ara = ARA3D()
    ann = ara.annotation
    dist = np.load(WORK / "edge_dist3d.npy") if (WORK / "edge_dist3d.npy").exists() else SG.edge_distance_3d(ann)
    plates = pickle.load(open(WORK / "sagittal_plates.pkl", "rb"))
    cor = pd.read_csv(WORK / "placement.csv")
    c1, c0 = np.polyfit(-cor.bregma_mm, cor.z, 1)
    s = float(cor.scale.median())

    # 1. placement
    F = SG.Fitter(plates, dist, ann > 0, SIDE)
    g0 = dict(s=s, asp=c1 / 40 / s, th=0.0, a0=c0, d0=float(cor.dv_shift_vox.median()), l0=ara.ml_mid_vox, sL=s, al=0.0, be=0.0)
    bounds = dict(s=(0.85, 1.15), asp=(0.85, 1.15), th=(-0.09, 0.09), a0=(c0 - 25, c0 + 25), d0=(g0["d0"] - 25, g0["d0"] + 25),
                  l0=(ara.ml_mid_vox - 20, ara.ml_mid_vox + 20), sL=(0.8, 1.2), al=(-0.15, 0.15), be=(-0.15, 0.15))
    start = np.median([F.plate_cost(k, g0) for k in range(len(plates))])
    g = F.fit_global(g0, bounds)
    print("stack fit:", {k: round(v, 4) for k, v in g.items()}, f"| median cost {start:.1f} -> "
          f"{np.median([F.plate_cost(k, g) for k in range(len(plates))]):.1f}", flush=True)
    rows, local = [], {}
    for k, p in enumerate(plates):
        loc = F.fit_plate(k, g)
        local[p.level] = loc
        d = F.distances(k, g, loc)
        pts = F.lines[k]
        x0, y0, x1, y1 = hpf_box(p)
        inb = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        d_stack = F.distances(k, g)
        rows.append(dict(level=p.level, lateral_mm=p.lateral_mm, line_median_um=float(np.median(d)),
                         hpf_median_um=float(np.median(d[inb])) if inb.any() else np.nan, p90_um=float(np.percentile(d, 90)),
                         stack_only_median_um=float(np.median(d_stack)), **{f"d_{n}": v for n, v in loc.items()}))
        print(f"sagittal level {p.level} (lateral {p.lateral_mm:+.2f} mm): ARA line distance median {np.median(d):.1f} µm "
              f"(hippocampus {rows[-1]['hpf_median_um']:.1f}; stack only {np.median(d_stack):.1f})", flush=True)
    pdf = pd.DataFrame(rows)
    pdf.to_csv(WORK / "sagittal_placement.csv", index=False)
    (WORK / "sagittal_stack.json").write_text(json.dumps(dict(global_=g, side=SIDE, local={str(k): v for k, v in local.items()})))

    # 2-3. tests: the filled coronal build, and the coronal plates themselves, against the sagittal drawings
    full = np.load(WORK / "hgea_ara3d_full.npz")["labels"]
    plate_only = np.load(WORK / "hgea_ara3d.npz")["labels"]
    zs = set(cor.z.astype(int))
    rows_f, conf = SG.agreement(plates, g, local, SIDE, full, DOMAINS)
    # crossings: only the sagittal points lying on a coronal plate slice
    vol_cross = np.zeros_like(plate_only)
    for z in zs:
        vol_cross[z] = plate_only[z]
    rows_c, _ = SG.agreement(plates, g, local, SIDE, vol_cross, DOMAINS)
    tdf = pd.DataFrame(rows_f)
    cdf = pd.DataFrame(rows_c)[["level", "both_pts", "domain_agreement", "field_agreement"]].rename(
        columns={"both_pts": "crossing_pts", "domain_agreement": "crossing_domain_agreement", "field_agreement": "crossing_field_agreement"})
    tdf = tdf.merge(cdf, on="level")
    for r in tdf.itertuples():
        print(f"sagittal level {r.level}: extent Dice {r.extent_dice:.2f}; where both label: field {r.field_agreement:.2f}, "
              f"domain {r.domain_agreement:.2f} (filled build) · coronal plates at crossings: domain {r.crossing_domain_agreement:.2f}", flush=True)
    w = lambda col, n: float(np.nansum(tdf[col] * tdf[n]) / tdf[n].sum())
    tot = dict(level="all", lateral_mm=np.nan, sagittal_pts=int(tdf.sagittal_pts.sum()), volume_pts=int(tdf.volume_pts.sum()),
               both_pts=int(tdf.both_pts.sum()), extent_dice=2 * tdf.both_pts.sum() / (tdf.sagittal_pts.sum() + tdf.volume_pts.sum()),
               field_agreement=w("field_agreement", "both_pts"), domain_agreement=w("domain_agreement", "both_pts"),
               crossing_pts=int(tdf.crossing_pts.sum()), crossing_domain_agreement=w("crossing_domain_agreement", "crossing_pts"),
               crossing_field_agreement=w("crossing_field_agreement", "crossing_pts"))
    tdf = pd.concat([tdf, pd.DataFrame([tot])], ignore_index=True)
    tdf.to_csv(WORK / "sagittal_test.csv", index=False)
    drows = []
    for i, name in enumerate(DOMAINS, start=1):
        tp, t, pp = conf[i, i], conf[i, :].sum(), conf[:, i].sum()
        others = conf[i].copy(); others[[0, i]] = 0
        drows.append(dict(domain=name, sagittal_pts_where_both=int(t), recall=tp / t if t else np.nan, precision=tp / pp if pp else np.nan,
                          dice=2 * tp / (t + pp) if t + pp else np.nan, most_confused_with=DOMAINS[int(np.argmax(others)) - 1] if others.any() else ""))
    pd.DataFrame(drows).to_csv(WORK / "sagittal_test_domains.csv", index=False)
    vol = np.zeros(SHAPE, np.uint8)
    for p in plates:
        pts, lab = SG.domain_samples(p, 0.025, DOMAIN_ID)
        ijk = np.rint(SG.to_ara(pts, p.lateral_mm, SIDE, g, local[p.level])).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(SHAPE)), axis=1) & (lab > 0)
        vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] = lab[ok]
    np.savez_compressed(WORK / "sagittal_planes.npz", labels=vol, levels=pdf.level.values, domains=np.array(DOMAINS))
    print(f"\nplacement: line distance median over plates {pdf.line_median_um.median():.1f} µm "
          f"(hippocampus {pdf.hpf_median_um.median():.1f}); stack only {pdf.stack_only_median_um.median():.1f}")
    print(f"filled build vs sagittal drawings: extent Dice {tot['extent_dice']:.3f}; where both label: field {tot['field_agreement']:.3f}, "
          f"domain {tot['domain_agreement']:.3f} · coronal plates vs sagittal at crossings: domain {tot['crossing_domain_agreement']:.3f}")

    # QC: each sagittal plane cut from the 3D ARA (Nissl), the filled HGEA as colour, the sagittal drawing as outlines
    rgb = np.zeros((256, 3))
    for i, n in enumerate(DOMAINS, start=1):
        rgb[i] = rgb[i + 100] = [int(COLOURS[n][j:j + 2], 16) / 255 for j in (1, 3, 5)]
    imgs = []
    for k, p in enumerate(plates):
        b = p.reference["brain_mask"][0].bounds
        xs, ys = np.arange(b[0], b[2], 0.025), np.arange(b[1], b[3], 0.025)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        vox = SG.to_ara(grid, p.lateral_mm, SIDE, g, local[p.level])
        nis = ndimage.map_coordinates(ara.nissl.astype(np.float32), vox.T, order=1, cval=0).reshape(gx.shape)
        fl = ndimage.map_coordinates(full, vox.T, order=0, cval=0).reshape(gx.shape) % 100
        img = np.dstack([1 - nis / 255.0 * 0.85] * 3)
        m = fl > 0
        img[m] = img[m] * 0.4 + rgb[fl[m]] * 0.6
        # sagittal drawing outlines in black
        dl = np.zeros(len(grid), np.int16)
        for name, geom in p.domains.items():
            if name in DOMAIN_ID:
                dl[contains_xy(geom, grid[:, 0], grid[:, 1])] = DOMAIN_ID[name]
        dl = dl.reshape(gx.shape)
        e = (ndimage.maximum_filter(dl, 3) != ndimage.minimum_filter(dl, 3))
        img[e] = [0, 0, 0]
        img = img[:, ::-1]          # anterior on the left, as on the plates
        imgs.append(img)
        r = tdf.iloc[k]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, extent=(xs[-1], xs[0], ys[-1], ys[0]))
        ax.set_title(f"sagittal level {p.level} (lateral {p.lateral_mm:.2f} mm) in the 3D ARA · colour = filled HGEA (coronal build), "
                     f"black = sagittal drawing · domain agreement {r.domain_agreement:.2f}", fontsize=8)
        ax.set_xlabel("AP mm from bregma"); ax.set_ylabel("DV mm")
        fig.tight_layout(); fig.savefig(WORK / "qc" / f"sagittal_{p.level}.png", dpi=100); plt.close(fig)
    cols = 5; rws = int(np.ceil(len(imgs) / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(cols * 3.6, rws * 2.6))
    for ax, img, p, (_, r) in zip(axs.ravel(), imgs, plates, tdf.iterrows()):
        ax.imshow(img); ax.set_title(f"S{p.level} · {p.lateral_mm:.2f} mm · {r.domain_agreement:.2f}", fontsize=8); ax.axis("off")
    for ax in axs.ravel()[len(imgs):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(WORK / "qc" / "sagittal_montage.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
