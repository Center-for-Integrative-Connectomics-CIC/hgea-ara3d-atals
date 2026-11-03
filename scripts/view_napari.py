"""View the HGEA plates in the 3D ARA (CCFv2 2011 Nissl brain), in napari. No CCFv3/BBP involved.

Usage: python scripts/view_napari.py [--3d]

Layers: ARA Nissl (cells dark), 2011 ARA annotation (hidden; toggle on for region outlines), HGEA on the
33 plate slices only (scripts/02_place_plates.py) and HGEA filled between plates
(scripts/03_interpolate.py; shown if it exists). Toggle the two HGEA layers to compare. Hidden extras: the sagittal drawings placed on their planes
(scripts/05_place_sagittal.py), the long-axis fill (scripts/07_longaxis_fill.py, comparison only) and the
hippocampal coordinates s, t, r (scripts/06_hippocampal_coords.py).
Keys ] / [ jump to the next / previous plate slice; the corner shows the level, or the two plates a
filled slice lies between. Hover shows the ARA region and both HGEA labels.
"""

from __future__ import annotations

import argparse

import _path  # noqa: F401

import napari
import numpy as np
import pandas as pd

from hgea_ara3d.config import COLOURS, WORK
from hgea_ara3d.space import ARA3D, VOXEL_UM

OUT = WORK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--3d", dest="three_d", action="store_true")
    args = ap.parse_args()
    ara = ARA3D()
    d = np.load(OUT / "hgea_ara3d.npz")
    place = pd.read_csv(OUT / "placement.csv")
    domains = list(d["domains"])
    names = {0: ""} | {i + 1: n for i, n in enumerate(domains)} | {i + 101: f"{n} (left)" for i, n in enumerate(domains)}
    cmap = {0: (0, 0, 0, 0)} | {k: COLOURS[n.replace(" (left)", "")] for k, n in names.items() if k}
    onto = ara.ontology

    v = napari.Viewer(title="HGEA on the 3D ARA (CCFv2 2011)", ndisplay=3 if args.three_d else 2)
    scale = (VOXEL_UM,) * 3
    v.add_image(ara.nissl, name="ARA Nissl", colormap="gray_r", scale=scale, contrast_limits=(0, 200))
    ann = v.add_labels(ara.annotation.astype(np.int32), name="ARA annotation 2011", scale=scale, visible=False, opacity=0.35)
    full_path = OUT / "hgea_ara3d_full.npz"
    full = np.load(full_path)["labels"] if full_path.exists() else None
    if full is not None:
        v.add_labels(full.astype(np.int32), name="HGEA filled 3D", scale=scale, opacity=0.6,
                     colormap=napari.utils.DirectLabelColormap(color_dict=cmap))
    v.add_labels(d["labels"].astype(np.int32), name="HGEA plate slices only", scale=scale, opacity=0.85,
                 colormap=napari.utils.DirectLabelColormap(color_dict=cmap), visible=full is None)
    for fname, lname in (("sagittal_planes.npz", "sagittal drawings (test)"), ("hgea_ara3d_longaxis.npz", "long-axis fill (comparison)")):
        if (OUT / fname).exists():
            v.add_labels(np.load(OUT / fname)["labels"].astype(np.int32), name=lname, scale=scale, opacity=0.8,
                         colormap=napari.utils.DirectLabelColormap(color_dict=cmap), visible=False)
    if (OUT / "hippocampal_coords.npz").exists():
        hc = np.load(OUT / "hippocampal_coords.npz")
        for k, nm in (("s", "long axis s"), ("t", "transverse t"), ("r", "depth r")):
            v.add_image(np.nan_to_num(hc[k].astype(np.float32), nan=0.0), name=f"coordinate: {nm}", scale=scale,
                        colormap="turbo", contrast_limits=(0, 1), visible=False, opacity=0.7)
    zs = place.z.tolist()
    v.dims.set_point(0, zs[len(zs) // 2] * VOXEL_UM)

    def status(event=None):
        z = int(round(v.dims.point[0] / VOXEL_UM))
        hit = place[place.z == z]
        if len(hit):
            v.text_overlay.text = (f"HGEA level {int(hit.level.iloc[0])} (section {int(hit.section_number.iloc[0])}), "
                                   f"line fit {hit.median_line_um.iloc[0]:.1f} µm")
            return
        before, after = place[place.z < z], place[place.z > z]
        v.text_overlay.text = (f"3D ARA slice {z}: filled between levels {int(before.level.iloc[-1])} and {int(after.level.iloc[0])}"
                               if len(before) and len(after) else f"3D ARA slice {z} (outside the plate series)")
    v.text_overlay.visible = True
    v.text_overlay.position = "top_right"
    v.dims.events.point.connect(status)
    status()

    def jump(step):
        z = int(round(v.dims.point[0] / VOXEL_UM))
        nxt = [p for p in zs if (p > z if step > 0 else p < z)]
        if nxt:
            v.dims.set_point(0, (min(nxt) if step > 0 else max(nxt)) * VOXEL_UM)

    v.bind_key("]", lambda _: jump(1), overwrite=True)
    v.bind_key("[", lambda _: jump(-1), overwrite=True)

    @v.mouse_move_callbacks.append
    def hover(viewer, event):
        pos = np.round(np.array(viewer.cursor.position) / VOXEL_UM).astype(int)
        if (pos < 0).any() or (pos >= ara.annotation.shape).any():
            return
        rid = int(ara.annotation[tuple(pos)])
        acr = onto.acronym.get(rid, "") if rid else "outside"
        h = names.get(int(d["labels"][tuple(pos)]), "")
        f = names.get(int(full[tuple(pos)]), "") if full is not None else ""
        viewer.status = f"ARA: {acr}" + (f" · plate: {h}" if h else "") + (f" · filled: {f}" if f else "")

    napari.run()


if __name__ == "__main__":
    main()
