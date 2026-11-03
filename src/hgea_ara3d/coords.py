"""Hippocampal coordinates in the 3D ARA, per hemisphere.

Envelope: the 2011 ARA hippocampal fields (CA1, CA2, CA3, DG, SUB; the 2011 ontology has no ProS, so it
lies inside SUB), joined with the filled HGEA labels so that domains drawn just outside the ARA fields are
kept; small holes closed; the largest connected piece per side.

Coordinates (each in [0, 1], from a Laplace solution with fixed values on two opposite boundaries and no
flux elsewhere, as in HippUnfold):
    s  long axis     geodesic ratio d_sep / (d_sep + d_tmp) inside the envelope, from the septal pole
                     (envelope voxels within POLE_VOX of its most anterior extent) and the temporal pole
                     (within POLE_VOX of its most ventral extent). A Laplace solution between two small pole
                     caps stays flat away from them, so the geodesic ratio is used: it spaces evenly along
                     the length.
    t  transverse    Laplace, 0 on DG, 1 on SUB (CA3, CA2, CA1 lie in between)
    r  depth         distance ratio d_fib / (d_fib + d_mol): 0 at the fibre tracts (alveus side), 1 in the
                     molecular layers (CA stratum lacunosum-moleculare, SUB and DG molecular layers)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import cg

POLE_VOX = 4
FIELDS = {"CA1": 382, "CA2": 423, "CA3": 463, "DG": 726, "SUB": 502}
MOLECULAR = ["slm", "-mo", "-m"]   # acronym endings of molecular layers in the 2011 ontology


def envelope(ara, hgea_full: np.ndarray, side: int) -> np.ndarray:
    m = ara.mask(*FIELDS.values()) | (hgea_full > 0)
    mid = ara.ml_mid_vox
    lr = np.arange(m.shape[2])
    m &= (lr > mid)[None, None, :] if side > 0 else (lr < mid)[None, None, :]
    m = ndimage.binary_closing(m, iterations=1) & (m | ndimage.binary_dilation(m))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
    return m


def laplace(mask: np.ndarray, zero: np.ndarray, one: np.ndarray, tol: float = 1e-6, maxiter: int = 4000) -> np.ndarray:
    """Solve the Laplace equation on `mask` (6-connected) with u = 0 on `zero`, u = 1 on `one`,
    zero flux on the rest of the boundary. Returns float32 volume (NaN outside the mask)."""
    idx = -np.ones(mask.shape, np.int64)
    pts = np.argwhere(mask)
    idx[tuple(pts.T)] = np.arange(len(pts))
    fixed = np.full(len(pts), np.nan)
    fixed[idx[zero & mask]] = 0.0
    fixed[idx[one & mask]] = 1.0
    free = np.isnan(fixed)
    rows, cols = [], []
    for ax in range(3):
        for sgn in (-1, 1):
            nb = pts.copy(); nb[:, ax] += sgn
            ok = (nb[:, ax] >= 0) & (nb[:, ax] < mask.shape[ax])
            j = np.full(len(pts), -1); j[ok] = idx[tuple(nb[ok].T)]
            ok = j >= 0
            rows.append(np.nonzero(ok)[0]); cols.append(j[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    n = len(pts)
    A = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    L = sparse.diags(deg) - A                      # graph Laplacian (Neumann on the mask boundary)
    Lff = L[free][:, free]
    b = -L[free][:, ~free] @ fixed[~free]
    pre = sparse.diags(1.0 / np.maximum(Lff.diagonal(), 1e-9))
    x, info = cg(Lff, b, rtol=tol, maxiter=maxiter, M=pre)
    u = fixed.copy(); u[free] = x
    out = np.full(mask.shape, np.nan, np.float32)
    out[tuple(pts.T)] = u
    return out


def geodesic(mask: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Shortest-path distance (µm) inside `mask` (26-connected) from the `source` voxels; NaN outside."""
    pts = np.argwhere(mask)
    idx = -np.ones(mask.shape, np.int64)
    idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols, wts = [], [], []
    for off in [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) > (0, 0, 0)]:
        nb = pts + np.array(off)
        ok = np.all((nb >= 0) & (nb < np.array(mask.shape)), axis=1)
        j = np.full(len(pts), -1); j[ok] = idx[tuple(nb[ok].T)]
        ok = j >= 0
        rows.append(np.nonzero(ok)[0]); cols.append(j[ok]); wts.append(np.full(ok.sum(), 25.0 * np.linalg.norm(off)))
    G = sparse.coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pts),) * 2).tocsr()
    src = idx[source & mask]
    d = dijkstra(G, directed=False, indices=src, min_only=True)
    out = np.full(mask.shape, np.nan, np.float32)
    out[tuple(pts.T)] = d
    return out


def boundaries(ara, env: np.ndarray) -> dict:
    """The fixed-value sets for s, t and r inside one hemisphere's envelope."""
    a, d, _ = np.nonzero(env)
    amin, dmax = a.min(), d.max()
    A_idx = np.arange(env.shape[0])[:, None, None]
    D_idx = np.arange(env.shape[1])[None, :, None]
    septal = env & (A_idx <= amin + POLE_VOX)
    temporal = env & (D_idx >= dmax - POLE_VOX)
    dg = env & ara.mask(FIELDS["DG"])
    sub = env & ara.mask(FIELDS["SUB"])
    fib = ara.mask(1009)
    onto = ara.ontology
    mol_ids = [sid for sid, acr in onto.acronym.astype(str).items() if any(acr.endswith(e) for e in MOLECULAR)
               and any(f"/{fid}/" in str(onto.at[sid, "structure_id_path"]) for fid in FIELDS.values())]
    mol = np.isin(ara.annotation, mol_ids)
    return dict(septal=septal, temporal=temporal, dg=dg, sub=sub, fib=fib, mol=mol)


def hippocampal_coords(ara, hgea_full: np.ndarray, side: int) -> dict:
    env = envelope(ara, hgea_full, side)
    bd = boundaries(ara, env)
    ds, dt = geodesic(env, bd["septal"]), geodesic(env, bd["temporal"])
    s = ds / (ds + dt)
    t = laplace(env, bd["dg"], bd["sub"])
    dfib = ndimage.distance_transform_edt(~bd["fib"])
    dmol = ndimage.distance_transform_edt(~bd["mol"])
    r = np.where(env, dfib / np.maximum(dfib + dmol, 1e-6), np.nan).astype(np.float32)
    keep = ("septal", "temporal", "dg", "sub")
    return dict(env=env, s=s.astype(np.float32), t=t, r=r, **{f"bd_{k}": bd[k] for k in keep})
