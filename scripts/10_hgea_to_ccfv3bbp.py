"""Step 10 (plan B): carry the reference HGEA into CCFv3BBP and compare it with v4.

Warping: each label (14 domains x 2 sides) is carried as a signed-distance map with linear interpolation,
and each CCFv3BBP voxel takes the label with the most negative warped distance (HGEA only where it is
negative). That avoids the jagged edges of nearest-neighbour label warping.

Reports:
    envelope   HGEA voxels inside / outside the CCFv3BBP hippocampal formation (CA, DG, SUB, ProS, FC, IG),
               per domain: kept and reported, not clipped
    v4         per domain (sides pooled), Dice and mean border distance against v4 (the HGEA-3D project's build),
               with v4's 19 leaves merged into the plates' 14 domains (DGd+CA3d+DGpod, DGi+CA3i, DGv+DGpov+CA3vv)

Input:  data_work/hgea_reference_v1.npz, data_work/to_ccfv3bbp/transforms.json (step 9), the CCFv3BBP package;
        optionally an earlier CCFv3BBP build (config.V4_LEAVES) for the comparison
Output: data_work/hgea_ccfv3bbp.npz (labels on the 566 x 320 x 456 CCFv3BBP grid, left +100), envelope.csv,
        vs_v4.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import _path  # noqa: F401
import ants
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from hgea_ara3d.config import CCFV3BBP_DIR, DOMAINS, V4_LEAVES, WORK

OUT = WORK / "to_ccfv3bbp"
V = 25.0
CLIP = 20.0
HPF_BBP = [375, 726, 502, 484682470, 982, 19]     # CA, DG, SUB, ProS, FC, IG
V4_TO_PLATE = {"CA1d": "CA1d", "CA1i": "CA1i", "CA1v": "CA1v", "CA1vv": "CA1vv", "CA2": "CA2", "CA3dd": "CA3dd", "CA3v": "CA3v",
               "DGd": "DGd/CA3d", "CA3d": "DGd/CA3d", "DGpod": "DGd/CA3d", "DGi": "DGi/CA3i", "CA3i": "DGi/CA3i",
               "DGv": "DGv/CA3vv", "DGpov": "DGv/CA3vv", "CA3vv": "DGv/CA3vv",
               "SUB_1": "SUB_1", "SUB_2": "SUB_2", "SUB_3": "SUB_3", "SUB_4": "SUB_4"}


def sdf(m: np.ndarray) -> np.ndarray:
    if not m.any():
        return np.full(m.shape, CLIP, np.float32)
    return np.clip(ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m), -CLIP, CLIP).astype(np.float32)


def border_um(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return np.nan
    ea, eb = a & ~ndimage.binary_erosion(a), b & ~ndimage.binary_erosion(b)
    return float((ndimage.distance_transform_edt(~eb)[ea].mean() + ndimage.distance_transform_edt(~ea)[eb].mean()) / 2 * V)


def main() -> None:
    T = json.loads((OUT / "transforms.json").read_text())
    tx = [str(OUT / n) for n in T["forward"]]
    origin = tuple(float(v) for v in T["moving_origin_um"])
    ann = tifffile.imread(CCFV3BBP_DIR / "annotation.tiff")
    st = pd.read_csv(CCFV3BBP_DIR / "structures.csv").set_index("id")
    F = ants.from_numpy(np.zeros(ann.shape, np.float32), spacing=(V, V, V))
    lab = np.load(WORK / "hgea_reference_v1.npz")["labels"]

    # bounding box of the HGEA on the ARA grid, to keep the distance maps small
    a, d, l = np.nonzero(lab)
    pad = int(CLIP) + 4
    lo = np.maximum([a.min() - pad, d.min() - pad, l.min() - pad], 0)
    hi = np.minimum([a.max() + pad, d.max() + pad, l.max() + pad], np.array(lab.shape))
    sub = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    o = (origin[0] + lo[0] * V, origin[1] + lo[1] * V, origin[2] + lo[2] * V)

    best = np.full(ann.shape, np.inf, np.float32)
    out = np.zeros(ann.shape, np.uint8)
    for i in range(1, len(DOMAINS) + 1):
        for off in (0, 100):
            m = sub == i + off
            if not m.any():
                continue
            w = ants.apply_transforms(F, ants.from_numpy(sdf(m), origin=o, spacing=(V, V, V)), tx, interpolator="linear",
                                      defaultvalue=CLIP).numpy()
            take = w < best
            best[take], out[take] = w[take], i + off
        print(f"warped {DOMAINS[i - 1]}", flush=True)
    out[best >= 0] = 0
    np.savez_compressed(WORK / "hgea_ccfv3bbp.npz", labels=out, domains=np.array(DOMAINS),
                        note="reference v1 carried from the 3D ARA to CCFv3BBP (step 9 transform, signed-distance warping)")

    paths = st["structure_id_path"].astype(str)
    hpf_ids = np.array(st.index[np.any([paths.str.contains(f"/{i}/", regex=False) for i in HPF_BBP], axis=0)])
    hpf = np.isin(ann, hpf_ids)
    env = []
    for i, name in enumerate(DOMAINS, start=1):
        m = (out % 100) == i
        env.append(dict(domain=name, voxels=int(m.sum()), outside_hpf_frac=float((m & ~hpf).sum() / max(1, m.sum()))))
    m = out > 0
    env.append(dict(domain="all", voxels=int(m.sum()), outside_hpf_frac=float((m & ~hpf).sum() / m.sum())))
    edf = pd.DataFrame(env)
    edf.to_csv(OUT / "envelope.csv", index=False)

    if not V4_LEAVES.exists():
        print(edf.round(3).to_string(index=False)); print('no v4 build at', V4_LEAVES, '- comparison skipped'); return
    z = np.load(V4_LEAVES, allow_pickle=True)
    v4, leaves = z["leaf"], list(z["leaves"])
    v4p = np.zeros(v4.shape, np.int16)
    for k, leaf in enumerate(leaves, start=1):
        v4p[v4 == k] = DOMAINS.index(V4_TO_PLATE[leaf]) + 1
    rows = []
    for i, name in enumerate(DOMAINS, start=1):
        x, y = (out % 100) == i, v4p == i
        rows.append(dict(domain=name, ara_route_voxels=int(x.sum()), v4_voxels=int(y.sum()),
                         dice=float(2 * (x & y).sum() / max(1, x.sum() + y.sum())), border_um=border_um(x, y)))
    x, y = out > 0, v4p > 0
    rows.append(dict(domain="all HGEA", ara_route_voxels=int(x.sum()), v4_voxels=int(y.sum()),
                     dice=float(2 * (x & y).sum() / (x.sum() + y.sum())), border_um=border_um(x, y)))
    vdf = pd.DataFrame(rows)
    vdf.to_csv(OUT / "vs_v4.csv", index=False)
    u = (out > 0) | (v4p > 0)
    print(f"\nlabel agreement with v4 over the union of both: {float(((out % 100) == v4p)[u].mean()):.3f}")
    print(edf.round(3).to_string(index=False))
    print(vdf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
