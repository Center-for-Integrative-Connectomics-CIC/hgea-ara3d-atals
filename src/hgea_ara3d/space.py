"""The 3D ARA: Allen's 2011 reconstruction of the coronal ARA Nissl brain (CCFv2), 25 µm.

Files (data_raw/ara3d, downloaded by scripts/00_fetch_ara3d.py from download.alleninstitute.org/informatics-archive):
    atlasVolume/atlasVolume.raw   Nissl, uint8      current-release/mouse_annotation/P56_atlasVolume.zip
    annotation/annotation.raw     labels, uint32    current-release/mouse_annotation/P56_Mouse_annotation.zip
    structures_2011.csv           ontology for those labels (october-2014/annotation/structures.csv)

The raw files store LR x DV x AP with AP fastest (MHD DimSize 528 320 456). They are returned here as
(AP, DV, LR) arrays: AP index 0 anterior, DV 0 dorsal, LR symmetric about the middle (checked: the
olfactory bulb sits at AP ~42, cerebellum at ~459; LR is the mirror-symmetric axis).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from hgea_ara3d.config import ARA3D_DIR

VOXEL_UM = 25.0
SHAPE = (528, 320, 456)   # AP, DV, LR


class ARA3D:
    def __init__(self, ara3d_dir: Path = ARA3D_DIR):
        self.dir = Path(ara3d_dir)

    def _raw(self, rel: str, dtype) -> np.ndarray:
        a = np.fromfile(self.dir / rel, dtype)
        return a.reshape(SHAPE[2], SHAPE[1], SHAPE[0]).transpose(2, 1, 0)

    @cached_property
    def nissl(self) -> np.ndarray:
        return np.ascontiguousarray(self._raw("atlasVolume/atlasVolume.raw", np.uint8))

    @cached_property
    def annotation(self) -> np.ndarray:
        return np.ascontiguousarray(self._raw("annotation/annotation.raw", np.uint32))

    @cached_property
    def ontology(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "structures_2011.csv").set_index("id")

    def descendants(self, sid: int) -> np.ndarray:
        """sid and every structure below it in the 2011 ontology."""
        paths = self.ontology["structure_id_path"].astype(str)
        return np.array(self.ontology.index[paths.str.contains(f"/{sid}/", regex=False)])

    def mask(self, *sids: int) -> np.ndarray:
        return np.isin(self.annotation, np.concatenate([self.descendants(s) for s in sids]))

    @property
    def ml_mid_vox(self) -> float:
        return (SHAPE[2] - 1) / 2
