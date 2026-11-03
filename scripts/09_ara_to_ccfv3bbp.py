"""Step 9: the 3D ARA (CCFv2 2011) -> CCFv3BBP transform, and its accuracy in the hippocampus.

Target: Blue Brain's realignment of the same ARA Nissl brain into CCFv3BBP, which is
`single_animal_nissl.tiff` of the BrainGlobe package ccfv3augmented_mouse_25um (identical to Zenodo
15176439 arav3a_bbp_nisslCOR_25; checked: correlation 1.0000). So this is same-brain, same-stain registration.

    1. start: the CCFv2/CCFv3 grids differ from CCFv3BBP by 14 voxels (350 µm) along AP;
    2. rigid then affine (Mattes MI) on the whole brain at 50 µm;
    3. SyN (CC, radius 2) at full 25 µm on a crop around the hippocampus (+ 1.5 mm margin);
    4. accuracy: the 2011 hippocampal fields carried by the transform vs CCFv3BBP's own fields (Dice, mean
       border distance), and Nissl correlation in the crop, for start / affine / SyN. The fields are defined
       a little differently in 2011 and in CCFv3, so the scores do not reach 1; the gain from each stage is
       the measure.

Input:  data_raw/ara3d (3D ARA Nissl + 2011 annotation), the CCFv3BBP BrainGlobe package (config.CCFV3BBP_DIR)
Output: data_work/to_ccfv3bbp/  transforms (ANTs .mat / .nii.gz), accuracy.csv, qc_*.png
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import _path  # noqa: F401
import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from hgea_ara3d.config import CCFV3BBP_DIR, WORK
from hgea_ara3d.space import ARA3D

OUT = WORK / "to_ccfv3bbp"
AP_OFFSET_VOX = 14
V = 25.0
FIELDS_2011 = {"CA1": [382], "CA2": [423], "CA3": [463], "DG": [726], "SUB": [502]}
FIELDS_BBP = {"CA1": [382], "CA2": [423], "CA3": [463], "DG": [726], "SUB": [502, 484682470]}   # CCFv3 SUB + ProS


def img(a: np.ndarray, origin=(0.0, 0.0, 0.0)) -> ants.ANTsImage:
    return ants.from_numpy(a.astype(np.float32), origin=tuple(float(o) for o in origin), spacing=(V, V, V))


def norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    hi = np.percentile(a[a > 0], 99.5)
    return np.clip(a / hi, 0, 1)


def descendants(structures: pd.DataFrame, ids) -> np.ndarray:
    paths = structures["structure_id_path"].astype(str)
    return np.array(structures.index[np.any([paths.str.contains(f"/{i}/", regex=False) for i in ids], axis=0)])


def border_um(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return np.nan
    ea, eb = a & ~ndimage.binary_erosion(a), b & ~ndimage.binary_erosion(b)
    return float((ndimage.distance_transform_edt(~eb)[ea].mean() + ndimage.distance_transform_edt(~ea)[eb].mean()) / 2 * V)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ara = ARA3D()
    fixed_np = norm(tifffile.imread(CCFV3BBP_DIR / "single_animal_nissl.tiff"))          # ASR, 566 x 320 x 456
    ann_bbp = tifffile.imread(CCFV3BBP_DIR / "annotation.tiff")
    st_bbp = pd.read_csv(CCFV3BBP_DIR / "structures.csv").set_index("id")
    moving_np = norm(ara.nissl)                                                           # AP, DV, LR, 528 x 320 x 456
    F = img(fixed_np)
    M = img(moving_np, origin=(AP_OFFSET_VOX * V, 0, 0))     # moving voxel a sits at fixed voxel a + 14 at the start
    hgea = np.load(WORK / "hgea_reference_v1.npz")["labels"]
    ann11 = ara.annotation

    # 2. rigid + affine at 50 µm
    Fd, Md = ants.resample_image(F, (2 * V,) * 3, use_voxels=False, interp_type=0), ants.resample_image(M, (2 * V,) * 3, use_voxels=False, interp_type=0)
    aff = ants.registration(fixed=Fd, moving=Md, type_of_transform="Affine", aff_metric="mattes",
                            aff_iterations=(400, 200, 100), aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0), verbose=False)
    aff_tx = aff["fwdtransforms"]
    print(f"affine done ({time.time() - t0:.0f}s)", flush=True)

    # 3. SyN on a crop around the hippocampus (fixed space, after affine)
    hg_aff = ants.apply_transforms(F, img((hgea > 0).astype(np.float32), origin=(AP_OFFSET_VOX * V, 0, 0)), aff_tx, interpolator="linear").numpy() > 0.5
    a, d, l = np.nonzero(hg_aff)
    m = 60   # 1.5 mm margin
    lo = np.maximum([a.min() - m, d.min() - m, l.min() - m], 0)
    hi = np.minimum([a.max() + m, d.max() + m, l.max() + m], np.array(fixed_np.shape) - 1)
    Fc = img(fixed_np[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], origin=lo * V)
    syn = ants.registration(fixed=Fc, moving=M, type_of_transform="SyNOnly", initial_transform=aff_tx[0],
                            syn_metric="CC", syn_sampling=2, reg_iterations=(80, 50, 30, 15), grad_step=0.2,
                            flow_sigma=3, total_sigma=0, verbose=False)
    syn_tx = syn["fwdtransforms"]          # [warp, affine]
    print(f"SyN done on crop {tuple(hi - lo)} ({time.time() - t0:.0f}s)", flush=True)
    names = []
    for k, p in enumerate(syn_tx):
        dst = OUT / f"ara_to_ccfv3bbp_{k}{''.join(Path(p).suffixes)}"
        shutil.copy(p, dst)
        names.append(dst.name)
    inv = syn["invtransforms"]
    inv_names = []
    for k, p in enumerate(inv):
        dst = OUT / f"ccfv3bbp_to_ara_{k}{''.join(Path(p).suffixes)}"
        shutil.copy(p, dst)
        inv_names.append(dst.name)
    (OUT / "transforms.json").write_text(json.dumps(dict(forward=names, inverse=inv_names, crop_lo_vox=lo.tolist(), crop_hi_vox=hi.tolist(),
                                                          moving_origin_um=[AP_OFFSET_VOX * V, 0, 0],
                                                          note="apply with ants.apply_transforms(fixed=CCFv3BBP grid, moving=ARA image with origin (350,0,0) µm)")))

    # 4. accuracy of each stage in the hippocampal crop
    def carry(vol: np.ndarray, tx, interp: str) -> np.ndarray:
        """ARA-grid volume -> CCFv3BBP grid; tx = [] means the AP offset only."""
        mv = img(vol, origin=(AP_OFFSET_VOX * V, 0, 0))
        if not tx:
            return ants.resample_image_to_target(mv, F, interp_type=interp).numpy()
        return ants.apply_transforms(F, mv, tx, interpolator=interp).numpy()

    start_tx = []
    rows = []
    crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    for stage, tx in (("start (AP offset only)", start_tx), ("affine", aff_tx), ("affine + SyN", syn_tx)):
        Mw = carry(moving_np, tx, "linear")
        f, w = fixed_np[crop], Mw[crop]
        mk = (f > 0.02) | (w > 0.02)
        row = dict(stage=stage, nissl_corr_crop=float(np.corrcoef(f[mk], w[mk])[0, 1]))
        for fname in FIELDS_2011:
            src = np.isin(ann11, ara.descendants(FIELDS_2011[fname][0])).astype(np.float32)
            src_w = carry(src, tx, "linear") > 0.5
            tgt = np.isin(ann_bbp, descendants(st_bbp, FIELDS_BBP[fname]))
            a_, b_ = src_w[crop], tgt[crop]
            row[f"dice_{fname}"] = float(2 * (a_ & b_).sum() / max(1, a_.sum() + b_.sum()))
            row[f"border_um_{fname}"] = border_um(a_, b_)
        rows.append(row)
        print(stage, {k: round(v, 3) for k, v in row.items() if k != "stage"}, f"({time.time() - t0:.0f}s)", flush=True)
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "accuracy.csv", index=False)

    # QC: a coronal and a sagittal cut through the hippocampus, target vs warped, with checkerboard
    Mw = ants.apply_transforms(F, M, syn_tx, interpolator="linear").numpy()
    za = int((lo[0] + hi[0]) / 2) - 30
    lr = int(np.percentile(l, 75))
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for r_, (tgt, mov, ttl) in enumerate([(fixed_np[za], Mw[za], f"coronal AP {za}"), (fixed_np[:, :, lr].T, Mw[:, :, lr].T, f"sagittal LR {lr}")]):
        cb = (np.indices(tgt.shape).sum(0) // 20) % 2 == 0
        for c_, (im, t2) in enumerate([(tgt, "CCFv3BBP single-animal Nissl (target)"), (mov, "3D ARA Nissl after transform"), (np.where(cb, tgt, mov), "checkerboard")]):
            ax[r_, c_].imshow(im, cmap="gray", vmin=0, vmax=1); ax[r_, c_].set_title(f"{t2} · {ttl}", fontsize=8); ax[r_, c_].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "qc_checkerboard.png", dpi=90); plt.close(fig)
    print(acc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
