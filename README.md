# HGEA · 3D ARA

The hippocampal domains of the Hippocampus Gene Expression Atlas (HGEA; Bienkowski et al., 2018) reconstructed in 3D on the brain they were drawn on, the Allen Reference Atlas Nissl brain, and carried into CCFv3BBP.

The web page has two editions:
- **3D ARA:** the 33 HGEA coronal plates on their own slices of the 3D ARA (Allen CCFv2, 2011, 25 µm), with no warping of the drawings, filled in 3D between plates along the Nissl.
- **CCFv3BBP:** the same reference carried into CCFv3BBP (Blue Brain), the space of the Blue Brain ISH gene atlas.

## Views
- **Slice views:** coronal, sagittal and horizontal.
- **3D views:**
  - **3D:** domains as voxel surfaces, with the Nissl as a volume or as three slices.
  - **coronal 3D:** the atlas cut at the coronal position, to step through the brain plate by plate.
- **Layers:**
  - filled 3D or plate slices only
  - domain toggles
  - Bienkowski's original SVG plates, aligned on their slices (3D ARA edition)

## Accuracy

| Measure | Value |
|---|---|
| Plate placement in the 3D ARA: median distance of each plate's printed ARA lines to the 3D annotation | 8.3 µm (hippocampus 8.9 µm) |
| Fill between plates, leave-one-plate-out at twice the plate gap: label accuracy | 0.643 (plain interpolation 0.613) |
| Sagittal HGEA plates (a different brain) placed in the 3D ARA: ARA line distance | 15.6 µm |
| Domain agreement with the sagittal drawings | 0.56 (the coronal and sagittal drawings agree 0.54 with each other) |
| 3D ARA → CCFv3BBP: Nissl correlation in the hippocampus | 0.956 |
| Blue Brain ISH check in CCFv3BBP: median marker AUC | 0.809 (earlier CCF build: 0.786) |

## Build

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/00_fetch_ara3d.py       # 3D ARA from the Allen Institute (~400 MB)
# inputs not downloaded by the scripts (see DATA_LICENSES.md):
#   data_raw/hgea_svg/            HGEA coronal SVG plates
#   data_raw/hgea_sagittal_ai/    HGEA sagittal Illustrator plates
#   data_raw/ccfv3bbp/            BrainGlobe ccfv3augmented_mouse_25um (or ~/.brainglobe)
#   data_raw/hgea_tables/         Bienkowski et al. Supplementary Table 2 + gene -> volume table (ISH check)
.venv/bin/python scripts/01_parse_plates.py
.venv/bin/python scripts/02_place_plates.py      # coronal plates on their 3D ARA slices
.venv/bin/python scripts/03_interpolate.py       # Nissl-guided fill + leave-one-plate-out
.venv/bin/python scripts/04_bake_site.py         # web data, 3D ARA edition
.venv/bin/python scripts/05_place_sagittal.py    # sagittal plates as a test
.venv/bin/python scripts/06_hippocampal_coords.py
.venv/bin/python scripts/07_longaxis_fill.py     # long-axis fill (tested; did not beat the Nissl fill)
.venv/bin/python scripts/08_reference_report.py  # robustness report, reference v1 freeze
.venv/bin/python scripts/09_ara_to_ccfv3bbp.py   # 3D ARA -> CCFv3BBP transform
.venv/bin/python scripts/10_hgea_to_ccfv3bbp.py
.venv/bin/python scripts/11_ish_check.py         # Blue Brain ISH check (test only)
.venv/bin/python scripts/12_bake_ccf_site.py     # web data, CCFv3BBP edition
python3 scripts/serve.py                          # http://127.0.0.1:8777/docs/
```

napari: `python scripts/view_napari.py [--3d]`.

## Licence and sources
The code is under the MIT licence (see `LICENSE`). The data belong to their creators; see `DATA_LICENSES.md`.
