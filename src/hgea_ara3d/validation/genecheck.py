"""Check the reconstructed 3D HGEA against Carey et al.'s published 25 µm gene volumes.

Ground truth for "which genes mark which leaf" is HGEA Supplementary Table 2 (Bienkowski et al.
2018): a binary gene x subregion annotation of 258 genes. Carey et al.'s volumes come from an
independent registration of the same Allen ISH data, so agreement tests the placement of our
reconstruction (registration + interpolation), not the HGEA biology itself.

Two tests:
- leaf test: per leaf, genes annotated present should be enriched over the other leaves of the
  same field more than genes annotated absent (AUC; 0.5 = chance).
- boundary test: for adjacent leaves A|B, the annotation's discriminating genes are profiled
  against signed distance to our A|B boundary; a fitted step gives the offset (µm) of the
  gene-defined boundary from ours, with a bootstrap over genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, optimize

# Supplementary Table 2 column -> our leaf name
TABLE_COLUMNS = {"DGd": "DGd", "DGi": "DGi", "DGv": "DGv", "DG_pod": "DGpod", "DG_pov": "DGpov",
                 "CA3dd": "CA3dd", "CA3d": "CA3d", "CA3i": "CA3i", "CA3v": "CA3v", "CA3vv": "CA3vv", "CA2": "CA2",
                 "CA1d": "CA1d", "CA1i": "CA1i", "CA1v": "CA1v", "CA1vv": "CA1vv",
                 "SUB_1": "SUB_1", "SUB_2": "SUB_2", "SUB_3": "SUB_3", "SUB_4": "SUB_4"}
PSEUDO = 0.5


def load_codebook(xlsx, mapping_csv) -> pd.DataFrame:
    """genes (published volume names) x leaves, boolean; only genes with a published volume."""
    t = pd.read_excel(xlsx, header=0).dropna(subset=["Gene"])
    m = pd.read_csv(mapping_csv).set_index("hgea_name")["published_volume"]
    t["volume"] = t["Gene"].astype(str).map(m)
    t = t.dropna(subset=["volume"]).set_index("volume")
    cb = pd.DataFrame({leaf: t[col].astype(str).str.strip().str.lower().eq("x") for col, leaf in TABLE_COLUMNS.items()})
    return cb[cb.any(axis=1)]


def leaf_means(expr: np.ndarray, leaf_of_voxel: np.ndarray, n_leaves: int) -> np.ndarray:
    """(genes, leaves+1) mean expression per leaf index (column 0 unused)."""
    counts = np.bincount(leaf_of_voxel, minlength=n_leaves + 1).astype(np.float64)
    out = np.zeros((expr.shape[1], n_leaves + 1))
    for gi in range(expr.shape[1]):
        out[gi] = np.bincount(leaf_of_voxel, weights=expr[:, gi], minlength=n_leaves + 1) / np.maximum(counts, 1)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def signed_distance_pair(leaf_vol: np.ndarray, a: int, b: int, field_mask: np.ndarray, pad: int = 30):
    """Signed distance (µm) to the A|B interface for voxels of A (negative) and B (positive)."""
    A, B = (leaf_vol == a) & field_mask, (leaf_vol == b) & field_mask
    idx = np.argwhere(A | B)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, leaf_vol.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a_c, b_c = A[sl], B[sl]
    d_to_b = ndimage.distance_transform_edt(~b_c) * 25.0
    d_to_a = ndimage.distance_transform_edt(~a_c) * 25.0
    sd = np.full(a_c.shape, np.nan, np.float32)
    sd[a_c] = -d_to_b[a_c]
    sd[b_c] = d_to_a[b_c]
    return sd, sl


def step(d, lo, hi, d0, w):
    return lo + (hi - lo) / (1 + np.exp(np.clip((d - d0) / w, -50, 50)))


def fit_step(bins: np.ndarray, prof: np.ndarray):
    ok = np.isfinite(prof)
    b, p = bins[ok], prof[ok]
    if len(b) < 6:
        return np.nan, np.nan, np.nan
    p0 = [p[b > 0].mean(), p[b < 0].mean(), 0.0, 50.0]
    try:
        (lo, hi, d0, w), _ = optimize.curve_fit(step, b, p, p0=p0, bounds=([-np.inf, -np.inf, -600, 5], [np.inf, np.inf, 600, 600]), maxfev=5000)
    except Exception:
        return np.nan, np.nan, np.nan
    return float(d0), float(abs(w)), float(hi - lo)


def steepest_transition(bins: np.ndarray, prof: np.ndarray, sigma_bins: float = 1.5, window_um: float = 300.0) -> float:
    """Position of the steepest A->B drop in the (smoothed) profile within +-window of our boundary.

    A logistic step fitted to a profile with a slow ramp puts its midpoint mid-ramp; the steepest
    descent marks where the transition is sharpest, which is what an expert draws as the border.
    """
    ok = np.isfinite(prof)
    if ok.sum() < 6:
        return np.nan
    p = np.interp(bins, bins[ok], prof[ok])
    p = ndimage.gaussian_filter1d(p, sigma_bins)
    g = np.gradient(p, bins)
    sel = np.abs(bins) <= window_um
    return float(bins[sel][np.argmin(g[sel])])
