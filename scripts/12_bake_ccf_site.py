"""Step 12: bake the CCFv3BBP edition of the web page (docs/data/ccfv3bbp), same layout as step 4.

    nissl.u8.gz        the ARA Nissl brain as realigned into CCFv3BBP by Blue Brain (single_animal_nissl),
                       566 x 320 x 456, 25 µm, uint8
    ara.u16.gz         the CCFv3BBP annotation as indices into ontology.json
    hgea_full.u8.gz    reference v1 in CCFv3BBP (step 10)
    hgea_plates.u8.gz  the plate slices only, carried with the same transform (nearest label): curved sheets in CCF
    meta.json          as step 4, plus edition texts; plates carry their median CCFv3BBP AP (approximate in CCF)
"""

from __future__ import annotations

import gzip
import json

import _path  # noqa: F401
import ants
import numpy as np
import pandas as pd
import tifffile

from hgea_ara3d.config import CCFV3BBP_DIR, COLOURS, DOMAINS, SITE_DATA, WORK

OUT = SITE_DATA / "ccfv3bbp"
V = 25.0


def write(name: str, a: np.ndarray) -> dict:
    path = OUT / name
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(a).tobytes())
    return dict(file=name, dtype=str(a.dtype), shape=list(a.shape), bytes_gz=path.stat().st_size)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nis = tifffile.imread(CCFV3BBP_DIR / "single_animal_nissl.tiff").astype(np.float32)
    nis8 = np.clip(nis / np.percentile(nis[nis > 0], 99.5) * 255, 0, 255).astype(np.uint8)
    ann = tifffile.imread(CCFV3BBP_DIR / "annotation.tiff")
    st = {d["id"]: d for d in json.loads((CCFV3BBP_DIR / "structures.json").read_text())}
    ids = np.unique(ann)
    index = np.zeros(int(ids.max()) + 1, np.uint16)
    index[ids] = np.arange(len(ids), dtype=np.uint16)
    entries = [dict(acronym="—", name="outside the brain", rgb=[0, 0, 0])]
    for sid in ids[1:]:
        r = st[int(sid)]
        entries.append(dict(acronym=r["acronym"], name=r["name"], rgb=list(r["rgb_triplet"])))
    (OUT / "ontology.json").write_text(json.dumps(entries, separators=(",", ":")))

    full = np.load(WORK / "hgea_ccfv3bbp.npz")["labels"].astype(np.uint8)
    T = json.loads((WORK / "to_ccfv3bbp" / "transforms.json").read_text())
    tx = [str(WORK / "to_ccfv3bbp" / n) for n in T["forward"]]
    plates = np.load(WORK / "hgea_ara3d.npz")["labels"].astype(np.float32)
    F = ants.from_numpy(np.zeros(ann.shape, np.float32), spacing=(V, V, V))
    pw = ants.apply_transforms(F, ants.from_numpy(plates, origin=tuple(float(o) for o in T["moving_origin_um"]), spacing=(V, V, V)),
                               tx, interpolator="genericLabel").numpy().round().astype(np.uint8)

    place = pd.read_csv(WORK / "placement.csv").sort_values("z")
    ara_meta = json.loads((SITE_DATA / "meta.json").read_text())
    plates_meta = []
    for p in ara_meta["plates"]:
        sheet = np.zeros(plates.shape, bool)
        sheet[p["z"]] = plates[p["z"]] > 0
        w = ants.apply_transforms(F, ants.from_numpy(sheet.astype(np.float32), origin=tuple(float(o) for o in T["moving_origin_um"]),
                                                       spacing=(V, V, V)), tx, interpolator="linear").numpy() > 0.5
        ap = np.nonzero(w)[0]
        plates_meta.append(dict(level=p["level"], z=int(np.median(ap)) if len(ap) else None, bregma_mm=p["bregma_mm"],
                                line_median_um=p["line_median_um"], hpf_median_um=p["hpf_median_um"],
                                ap_spread_vox=int(np.percentile(ap, 95) - np.percentile(ap, 5)) if len(ap) else None))
    acc = pd.read_csv(WORK / "to_ccfv3bbp" / "accuracy.csv").set_index("stage")
    env = pd.read_csv(WORK / "to_ccfv3bbp" / "envelope.csv").set_index("domain")
    ish = pd.read_csv(WORK / "ish_check" / "summary.csv").set_index("atlas")
    vs = pd.read_csv(WORK / "to_ccfv3bbp" / "vs_v4.csv").set_index("domain")
    ref = "ARA route (reference v1)"
    rows = [
        ["placement in the 3D ARA, ARA lines (median)", f"{ara_meta['accuracy']['placement_line_median_um']} µm"],
        ["3D ARA → CCFv3BBP, Nissl correlation (hippocampus)", f"{acc.loc['affine + SyN', 'nissl_corr_crop']:.3f}"],
        ["HGEA outside the CCFv3BBP hippocampal formation", f"{env.loc['all', 'outside_hpf_frac'] * 100:.1f}%"],
        ["Blue Brain ISH: marker AUC, median (this · v4)", f"{ish.loc[ref, 'median_auc']:.3f} · {ish.loc['v4', 'median_auc']:.3f}" if "v4" in ish.index else f"{ish.loc[ref, 'median_auc']:.3f}"],
        ["Blue Brain ISH: gene border offset, median (this · v4)", f"{ish.loc[ref, 'median_abs_offset_um']:.0f} · {ish.loc['v4', 'median_abs_offset_um']:.0f} µm" if "v4" in ish.index else f"{ish.loc[ref, 'median_abs_offset_um']:.0f} µm"],
        ["overlap with the earlier CCF build (v4), whole HGEA", f"Dice {vs.loc['all HGEA', 'dice']:.2f}"],
    ]
    files = dict(nissl=write("nissl.u8.gz", nis8), ara=write("ara.u16.gz", index[ann]), hgea_full=write("hgea_full.u8.gz", full),
                 hgea_plates=write("hgea_plates.u8.gz", pw))
    meta = dict(
        edition="ccfv3bbp",
        title="HGEA · CCFv3BBP", subtitle="Reference v1 carried from the 3D ARA into CCFv3BBP (Blue Brain), 25 µm",
        method=("The 3D ARA reference (33 plates on their own Nissl slices, Nissl-guided fill) is carried into CCFv3BBP by "
                "registering the 3D ARA Nissl onto the same brain's Nissl as realigned into CCFv3BBP by Blue Brain (affine + SyN). "
                "Each domain is warped as a signed-distance map. Plate positions here are each plate's median AP in CCFv3BBP: "
                "after the transform a plate is a gently curved sheet, not one slice."),
        grid=dict(shape=list(ann.shape), axes=["AP", "DV", "LR"], voxel_um=V, space="CCFv3BBP (BrainGlobe ccfv3augmented_mouse_25um)"),
        files=files, svg=False,
        domains=[dict(id=i + 1, name=d, colour=COLOURS[d]) for i, d in enumerate(DOMAINS)],
        plates=[p for p in plates_meta if p["z"] is not None],
        accuracy_rows=rows,
        accuracy_note=("ISH: Blue Brain gene volumes (Carey et al.) against Bienkowski et al.'s Table 2, the same test for both builds; "
                       "differences are not significant (gene borders are located to about ±100 µm)."),
        sources=ara_meta["sources"] + ["Piluso S. et al. An extended and improved CCFv3 annotation and Nissl atlas of the entire mouse brain. "
                                       "Imaging Neurosci. (2025): CCFv3BBP and the realigned ARA Nissl."],
    )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    for k, f in files.items():
        print(f"{k:12s} {f['bytes_gz'] / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
