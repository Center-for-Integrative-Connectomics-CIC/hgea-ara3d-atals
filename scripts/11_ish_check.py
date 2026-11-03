"""Step 11: independent-data check in CCFv3BBP — Blue Brain ISH (Carey et al.) vs the HGEA annotation,
for the ARA-route reference (step 10) and v4, with exactly the same test for both.

Test only: nothing here feeds either build. The gene-check functions and EBRAINS reader come from the HGEA-3D
project (hgea_ara3d.validation, unchanged), so the test matches the one v4 went through.
Caveat (as for v4): the Blue Brain volumes are reconstructed from the Allen ISH that Bienkowski et al.
used, so this tests placement, not the HGEA biology.

Common ground for both atlases:
    vocabulary  the plates' 14 domains (v4's 19 leaves merged; Table 2 columns merged the same way:
                a gene counts as a marker of a merged domain if marked in any of its columns), sides pooled
    voxels      the union of both atlases' HGEA voxels; every gene sampled there
    leaf test   per domain: AUC of its annotated markers vs annotated-absent genes, by enrichment over its
                peer domains (CA1 group, SUB group, DG/CA3 band group, CA2 vs CA1 + CA3dd + CA3v)
    boundary    per adjacent domain pair with >= 3 discriminating genes: the position of the steepest change
                of the gene contrast, relative to the atlas's own border (µm), bootstrap over genes
Output: data_work/ish_check/ leaf.csv, boundary.csv, summary.csv
"""

from __future__ import annotations

import multiprocessing as mp

import _path  # noqa: F401
import numpy as np
import pandas as pd

from hgea_ara3d.config import DOMAINS, HGEA_GENE_VOLUMES, HGEA_TABLE2, V4_LEAVES, WORK
from hgea_ara3d.validation import ebrains
from hgea_ara3d.validation.genecheck import PSEUDO, auc, load_codebook, signed_distance_pair, steepest_transition

OUT = WORK / "ish_check"
CACHE = WORK / "ish_check" / "gene_matrix_cache.npz"     # optional: earlier samples (expr, genes, voxels)
BINS = np.arange(-500, 501, 25.0)
N_BOOT = 100
MERGE = {"DGd/CA3d": ["DGd", "CA3d", "DGpod"], "DGi/CA3i": ["DGi", "CA3i"], "DGv/CA3vv": ["DGv", "DGpov", "CA3vv"]}
V4_TO_PLATE = {leaf: d for d, leaves in MERGE.items() for leaf in leaves}
GROUPS = {"CA1": ["CA1d", "CA1i", "CA1v", "CA1vv"], "SUB": ["SUB_1", "SUB_2", "SUB_3", "SUB_4"],
          "band": ["DGd/CA3d", "DGi/CA3i", "DGv/CA3vv", "CA3dd", "CA3v"]}
_vox = None


def _init(p):
    global _vox
    _vox = np.load(p)


def _fetch(gene):
    try:
        return gene, ebrains.decode(ebrains.download(gene))[tuple(_vox.T)], None
    except Exception as e:  # noqa: BLE001
        return gene, None, str(e)


def peers_of(d: str) -> list[str]:
    for g in GROUPS.values():
        if d in g:
            return [x for x in g if x != d]
    return GROUPS["CA1"] + ["CA3dd", "CA3v"]     # CA2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ours = np.load(WORK / "hgea_ccfv3bbp.npz")["labels"].astype(np.int16) % 100
    atlases = {"ARA route (reference v1)": ours}
    if V4_LEAVES.exists():
        z = np.load(V4_LEAVES, allow_pickle=True)
        v4 = np.zeros(ours.shape, np.int16)
        for k, leaf in enumerate(z["leaves"], start=1):
            v4[z["leaf"] == k] = DOMAINS.index(V4_TO_PLATE.get(leaf, leaf)) + 1
        atlases["v4"] = v4

    cb19 = load_codebook(HGEA_TABLE2, HGEA_GENE_VOLUMES)
    cb = pd.DataFrame({d: cb19[MERGE[d]].any(axis=1) if d in MERGE else cb19[d] for d in DOMAINS})
    genes = list(cb.index)

    # gene matrix over the union of both atlases' voxels: cached v4 rows + fresh rows for the rest
    U = np.any([v > 0 for v in atlases.values()], axis=0)
    vox = np.argwhere(U)
    lin = np.ravel_multi_index(tuple(vox.T), U.shape)
    expr = np.zeros((len(vox), len(genes)), np.uint8)
    hit = np.zeros(len(vox), bool)
    if CACHE.exists():       # reuse earlier samples where the voxels and genes match
        c = np.load(CACHE, allow_pickle=True)
        cg, clin = list(c["genes"]), np.ravel_multi_index(tuple(c["voxels"].T), U.shape)
        if all(g in cg for g in genes):
            order = np.argsort(clin)
            pos = np.clip(np.searchsorted(clin[order], lin), 0, len(clin) - 1)
            hit = clin[order][pos] == lin
            expr[hit] = c["expr"][order[pos[hit]]][:, [cg.index(g) for g in genes]]
    miss = vox[~hit]
    print(f"{len(genes)} genes; {len(vox):,} union voxels, {int(hit.sum()):,} from the cache, {len(miss):,} to sample", flush=True)
    mp_path = OUT / "missing_voxels.npy"
    np.save(mp_path, miss)
    got = {}
    with mp.get_context("spawn").Pool(8, _init, (str(mp_path),)) as pool:
        for k, (g, v, err) in enumerate(pool.imap_unordered(_fetch, genes)):
            if v is not None:
                got[g] = v
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(genes)} genes streamed", flush=True)
    for g, v in got.items():
        expr[~hit, genes.index(g)] = v
    print(f"sampled {len(got)}/{len(genes)} genes", flush=True)
    np.savez_compressed(CACHE, expr=expr, genes=np.array(genes, dtype=object), voxels=vox)

    leaf_rows, b_rows, rng = [], [], np.random.default_rng(0)
    for name, vol in atlases.items():
        lab = vol[tuple(vox.T)]
        counts = np.bincount(lab, minlength=len(DOMAINS) + 1).astype(float)
        means = np.stack([np.bincount(lab, weights=expr[:, j], minlength=len(DOMAINS) + 1) / np.maximum(counts, 1) for j in range(len(genes))])
        for i, d in enumerate(DOMAINS, start=1):
            pr = peers_of(d)
            pi = [DOMAINS.index(p) + 1 for p in pr]
            w = counts[pi]
            if counts[i] == 0 or w.sum() == 0:
                continue
            other = (means[:, pi] * w).sum(1) / w.sum()
            stat = np.log2((means[:, i] + PSEUDO) / (other + PSEUDO))
            scope = cb[[d] + pr].any(axis=1).to_numpy()
            p_, n_ = cb[d].to_numpy() & scope, ~cb[d].to_numpy() & scope
            leaf_rows.append(dict(atlas=name, domain=d, voxels=int(counts[i]), markers=int(p_.sum()), absent=int(n_.sum()),
                                  auc=auc(stat[p_], stat[n_]), markers_enriched_frac=float((stat[p_] > 0).mean()) if p_.any() else np.nan))
        gi = {g: j for j, g in enumerate(genes)}
        order_v = np.argsort(lin)
        for ia, a in enumerate(DOMAINS):
            for b in DOMAINS[ia + 1:]:
                ga = [g for g in genes if cb.at[g, a] and not cb.at[g, b]]
                gb = [g for g in genes if cb.at[g, b] and not cb.at[g, a]]
                if len(ga) + len(gb) < 3:
                    continue
                A, B = DOMAINS.index(a) + 1, DOMAINS.index(b) + 1
                if not (vol == A).any() or not (vol == B).any():
                    continue
                sd, sl = signed_distance_pair(vol, A, B, np.ones(vol.shape, bool))
                if (np.isfinite(sd) & (np.abs(sd) <= 25)).sum() < 100:
                    continue
                near = np.isfinite(sd) & (np.abs(sd) <= BINS[-1])
                cl = np.ravel_multi_index(tuple((np.argwhere(near) + np.array([s.start for s in sl])).T), vol.shape)
                rows = order_v[np.searchsorted(lin[order_v], cl)]
                dist = sd[near]
                bidx = np.digitize(dist, BINS) - 1
                cent = (BINS[:-1] + BINS[1:]) / 2

                def prof(xa, xb):
                    def nm(gs):
                        if not gs:
                            return np.zeros(len(dist))
                        x = expr[rows][:, [gi[g] for g in gs]].astype(np.float32)
                        return (x / (x.mean(0, keepdims=True) + 1e-3)).mean(1)
                    s = nm(xa) - nm(xb)
                    return np.array([s[bidx == k].mean() if (bidx == k).any() else np.nan for k in range(len(BINS) - 1)])

                g0 = steepest_transition(cent, prof(ga, gb))
                boots = np.array([steepest_transition(cent, prof(list(rng.choice(ga, len(ga))) if ga else [], list(rng.choice(gb, len(gb))) if gb else []))
                                  for _ in range(N_BOOT)])
                lo_, hi_ = np.nanpercentile(boots, [2.5, 97.5])
                b_rows.append(dict(atlas=name, A=a, B=b, genes_A=len(ga), genes_B=len(gb), steepest_um=g0, ci95_lo=lo_, ci95_hi=hi_))
        print(f"{name}: leaf and boundary tests done", flush=True)

    ldf, bdf = pd.DataFrame(leaf_rows), pd.DataFrame(b_rows)
    ldf.to_csv(OUT / "leaf.csv", index=False)
    bdf.to_csv(OUT / "boundary.csv", index=False)
    common = set.intersection(*[set(zip(g.A, g.B)) for _, g in bdf.groupby("atlas")])
    in_common = lambda df: np.array([(a, b) in common for a, b in zip(df.A, df.B)], bool)
    summ = []
    for name in atlases:
        L = ldf[ldf.atlas == name]
        Bd = bdf[(bdf.atlas == name).to_numpy() & in_common(bdf)]
        summ.append(dict(atlas=name, median_auc=L.auc.median(), domains_auc_over_0_7=int((L.auc > 0.7).sum()), domains=len(L),
                         common_borders=len(Bd), median_abs_offset_um=float(np.nanmedian(np.abs(Bd.steepest_um))),
                         borders_within_50um=int((np.abs(Bd.steepest_um) <= 50).sum())))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "summary.csv", index=False)
    print(sdf.round(3).to_string(index=False))
    print(ldf.pivot(index="domain", columns="atlas", values="auc").round(3).to_string())
    print(bdf[in_common(bdf)].pivot_table(index=["A", "B"], columns="atlas", values="steepest_um").round(0).to_string())


if __name__ == "__main__":
    main()
