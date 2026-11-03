"""Fill the 3D ARA between HGEA plate slices.

In the 3D ARA the plates are flat, parallel slices 2-5 voxels (50-125 µm) apart. For a slice z between
plates k and k+1:
    1. register the guide slice of plate k (and of plate k+1) onto the guide slice at z (2D SyN), so the
       plate's labels follow the real tissue from its slice to z;
    2. warp both plates' labels to z;
    3. blend them by signed distance, weighted by distance in z: each voxel takes the label with the most
       negative blended signed distance, and is HGEA only if that value is negative.
Guides: "nissl" (the 3D ARA Nissl: real tissue at every 25 µm slice, independent of any drawing),
"annotation" (signed distances of the 2011 hippocampal fields; Allen drew these from the ARA plates, so
they leak plate information into a held-out test), or "none" (plain shape interpolation, no warp).
"""

from __future__ import annotations

import ants
import numpy as np
from scipy import ndimage

SDF_CLIP_VOX = 20.0


def sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance in voxels, negative inside; clipped."""
    if not mask.any():
        return np.full(mask.shape, SDF_CLIP_VOX, np.float32)
    d = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    return np.clip(d, -SDF_CLIP_VOX, SDF_CLIP_VOX).astype(np.float32)


def blend(labels: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Multi-label signed-distance blend of label images (same grid)."""
    ids = sorted(set(np.unique(np.concatenate([l.ravel() for l in labels]))) - {0})
    if not ids:
        return np.zeros_like(labels[0])
    best = np.full(labels[0].shape, np.inf, np.float32)
    out = np.zeros_like(labels[0])
    for i in ids:
        s = sum(w * sdf(l == i) for l, w in zip(labels, weights))
        take = s < best
        best[take], out[take] = s[take], i
    out[best >= 0] = 0
    return out


def _img(a: np.ndarray):
    return ants.from_numpy(a.astype(np.float32))


class Guide:
    """2D registration of one ARA slice onto another, on the chosen guide channel(s)."""

    FIELDS = {"CA1": [382], "CA2": [423], "CA3": [463], "DG": [726], "SUB": [502]}

    def __init__(self, kind: str, ara, box):
        self.kind, self.box = kind, box
        (r0, r1), (c0, c1) = box
        if kind == "nissl":
            v = ara.nissl[:, r0:r1, c0:c1].astype(np.float32)
            self.chan = {"nissl": ndimage.gaussian_filter(v, (0, 0.7, 0.7)) / 255.0}
        elif kind == "annotation":
            ann = ara.annotation[:, r0:r1, c0:c1]
            self.chan = {k: np.isin(ann, np.concatenate([ara.descendants(i) for i in ids])) for k, ids in self.FIELDS.items()}
            self.chan["brain"] = ann > 0
        elif kind != "none":
            raise ValueError(kind)

    def transforms(self, z_moving: int, z_fixed: int):
        if self.kind == "none" or z_moving == z_fixed:
            return None
        if self.kind == "nissl":
            f, m = _img(self.chan["nissl"][z_fixed]), _img(self.chan["nissl"][z_moving])
            reg = ants.registration(fixed=f, moving=m, type_of_transform="SyNOnly", syn_metric="CC", syn_sampling=2,
                                    grad_step=0.2, flow_sigma=3, total_sigma=0, reg_iterations=(80, 50, 30), verbose=False)
            return reg["fwdtransforms"]
        used = [k for k, c in self.chan.items() if c[z_fixed].any() and c[z_moving].any()]
        f = {k: _img(sdf(self.chan[k][z_fixed])) for k in used}
        m = {k: _img(sdf(self.chan[k][z_moving])) for k in used}
        extras = [["meansquares", f[k], m[k], 1.0, 1] for k in used[1:]]
        reg = ants.registration(fixed=f[used[0]], moving=m[used[0]], type_of_transform="SyNOnly", syn_metric="meansquares",
                                syn_sampling=1, grad_step=0.2, flow_sigma=3, total_sigma=0, reg_iterations=(80, 50, 30),
                                multivariate_extras=extras, verbose=False)
        return reg["fwdtransforms"]

    def warp_labels(self, labels2d: np.ndarray, z_moving: int, z_fixed: int) -> np.ndarray:
        t = self.transforms(z_moving, z_fixed)
        if t is None:
            return labels2d
        out = ants.apply_transforms(fixed=_img(np.zeros_like(labels2d, np.float32)), moving=_img(labels2d),
                                    transformlist=t, interpolator="genericLabel")
        return np.rint(out.numpy()).astype(labels2d.dtype)


def predict(guide: Guide, lab_a: np.ndarray, za: int, lab_b: np.ndarray, zb: int, z: int) -> np.ndarray:
    """Labels at slice z from the two bracketing plates (cropped grids)."""
    w = (z - za) / (zb - za)
    return blend([guide.warp_labels(lab_a, za, z), guide.warp_labels(lab_b, zb, z)], [1.0 - w, w])


def scores(truth: np.ndarray, pred: np.ndarray, domains: list[str]) -> dict:
    """Agreement of a predicted plate with the real plate (25 µm voxels; sides pooled per domain)."""
    u = (truth > 0) | (pred > 0)
    out = {"label_accuracy": float((truth[u] == pred[u]).mean()) if u.any() else np.nan,
           "hgea_dice": float(2 * ((truth > 0) & (pred > 0)).sum() / ((truth > 0).sum() + (pred > 0).sum()))}
    for i, name in enumerate(domains, start=1):
        t = (truth % 100) == i
        p = (pred % 100) == i
        if not t.any() and not p.any():
            continue
        out[f"dice_{name}"] = float(2 * (t & p).sum() / (t.sum() + p.sum()))
        if t.any() and p.any():
            et, ep = t & ~ndimage.binary_erosion(t), p & ~ndimage.binary_erosion(p)
            dt, dp = ndimage.distance_transform_edt(~ep)[et], ndimage.distance_transform_edt(~et)[ep]
            out[f"border_um_{name}"] = float((dt.mean() + dp.mean()) / 2 * 25.0)
    return out
