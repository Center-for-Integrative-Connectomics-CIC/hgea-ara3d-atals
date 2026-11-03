"""Paths, sources, HGEA domains and their display colours."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_raw"          # downloads and the HGEA plates (not committed)
WORK = ROOT / "data_work"        # build outputs (not committed)
SITE = ROOT / "docs"             # web page (GitHub Pages root)
SITE_DATA = SITE / "data"

ARA3D_DIR = RAW / "ara3d"
SVG_DIR = RAW / "hgea_svg"
SAG_DIR = RAW / "hgea_sagittal_ai"     # HGEA-Sagittal-0NNBW.ai + HGEA_full_sagittal.pdf (20 plates)
PLATE_SECTIONS = Path(__file__).with_name("plate_sections.csv")   # ARA plate level -> Allen section number

# CCFv3BBP (BrainGlobe package ccfv3augmented_mouse_25um, Piluso et al.): data_raw/ccfv3bbp, or BrainGlobe's own cache
CCFV3BBP_DIR = next((p for p in (RAW / "ccfv3bbp", Path.home() / ".brainglobe" / "ccfv3augmented_mouse_25um_v1.0")
                     if (p / "annotation.tiff").exists()), RAW / "ccfv3bbp")
# Bienkowski et al. Supplementary Table 2 (gene x domain annotation) and the gene -> Blue Brain volume name table,
# for the ISH check (step 11); and optionally an earlier CCFv3BBP HGEA build (v4 leaves) to compare with
HGEA_TABLE2 = RAW / "hgea_tables" / "supp_table2_gene_annotation.xlsx"
HGEA_GENE_VOLUMES = RAW / "hgea_tables" / "hgea_markers_vs_published.csv"
V4_LEAVES = RAW / "v4" / "hgea_leaves.npz"

ALLEN = "https://download.alleninstitute.org/informatics-archive"
ARA3D_FILES = {
    "zips/P56_atlasVolume.zip": f"{ALLEN}/current-release/mouse_annotation/P56_atlasVolume.zip",
    "zips/P56_Mouse_annotation.zip": f"{ALLEN}/current-release/mouse_annotation/P56_Mouse_annotation.zip",
    "structures_2011.csv": f"{ALLEN}/october-2014/annotation/structures.csv",
}

# HGEA plate domains: the 14 leaves drawn on the coronal plates (DG and CA3 subdivisions share a band)
DOMAINS = ["CA1d", "CA1i", "CA1v", "CA1vv", "CA2", "CA3dd", "CA3v", "DGd/CA3d", "DGi/CA3i", "DGv/CA3vv",
           "SUB_1", "SUB_2", "SUB_3", "SUB_4"]
DOMAIN_ID = {d: i + 1 for i, d in enumerate(DOMAINS)}   # left hemisphere: +100
COLOURS = {"CA1d": "#e6194b", "CA1i": "#3cb44b", "CA1v": "#4363d8", "CA1vv": "#000075", "CA2": "#f58231",
           "CA3dd": "#911eb4", "CA3v": "#42d4f4", "DGd/CA3d": "#fabed4", "DGi/CA3i": "#aaffc3",
           "DGv/CA3vv": "#a9def9", "SUB_1": "#ff0000", "SUB_2": "#0082c8", "SUB_3": "#ffa500", "SUB_4": "#ffe119"}
