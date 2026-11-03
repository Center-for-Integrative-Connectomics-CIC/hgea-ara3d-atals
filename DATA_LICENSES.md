# Data sources and terms

This repository contains code that builds a 3D atlas from third-party data, and the derived data the web page uses. The data stay the property of their creators. **Check each source's current terms before redistributing**; these notes are pointers, not legal advice.

| Data | Source | Used for | Cite |
|---|---|---|---|
| HGEA coronal and sagittal plates (domain drawings on ARA plates) | Bienkowski et al., original HGEA files | the atlas itself; copies of the coronal SVGs in `docs/data/svg/` | Bienkowski M.S. et al., Nat. Neurosci. 21, 1628–1643 (2018), doi:10.1038/s41593-018-0241-y |
| HGEA Supplementary Table 2 (gene × domain annotation) | Supplementary Information of the article | the ISH check (not shipped) | as above |
| 3D Allen Reference Atlas: Nissl volume and 2011 annotation (CCFv2) | Allen Institute download server, `informatics-archive/current-release/mouse_annotation` | the 3D ARA edition (Nissl and annotation in `docs/data/`) | Allen Institute for Brain Science, Allen Reference Atlas (2008) and its 3D reconstruction (2011); Allen terms of use |
| CCFv3BBP annotation and the ARA Nissl realigned into CCFv3BBP | BrainGlobe `ccfv3augmented_mouse_25um`; Zenodo 15176439 | the CCFv3BBP edition (in `docs/data/ccfv3bbp/`) | Piluso S. et al., Imaging Neurosci. (2025), CC BY 4.0 |
| Blue Brain ISH gene volumes | EBRAINS bucket `deepslice` | the ISH check only (streamed, not shipped) | Carey H., Piluso S. et al., bioRxiv (2026) |

What `docs/data` contains:
- the HGEA labels built here
- the 3D ARA Nissl and 2011 annotation, as indices
- the CCFv3BBP annotation and realigned Nissl
- copies of the coronal SVG plates

**Before hosting publicly:** confirm that the HGEA SVG plates may be redistributed. If not, remove `docs/data/svg/`; the overlay toggle then has nothing to load, and everything else works.
