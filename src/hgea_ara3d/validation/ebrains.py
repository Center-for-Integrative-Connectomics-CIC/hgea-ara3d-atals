"""Gene expression volumes from the authors' public EBRAINS data-proxy bucket (CC-BY 4.0)."""

from __future__ import annotations

import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

BUCKET = "https://data-proxy.ebrains.eu/api/v1/buckets/deepslice"
PREFIX = "expression_volumes/"
SUFFIX = "_interp_25um.nii.gz"
ATLAS_SHAPE = (566, 320, 456)  # BrainGlobe ASR grid shared by CArea and CCFv3BBP


def list_genes() -> list[str]:
    """All gene symbols with a published 25 µm volume."""
    genes, marker = [], None
    while True:
        q = {"limit": 1000, "prefix": PREFIX}
        if marker:
            q["marker"] = marker
        objs = json.load(urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}"))["objects"]
        if not objs:
            return genes
        genes += [o["name"][len(PREFIX):-len(SUFFIX)] for o in objs if o["name"].endswith(SUFFIX)]
        marker = objs[-1]["name"]


def download(gene: str) -> bytes:
    url = f"{BUCKET}/{PREFIX}{urllib.parse.quote(gene)}{SUFFIX}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def decode(blob: bytes) -> np.ndarray:
    """Published NIfTI (LR, AP, DV) -> uint8 array on the atlas ASR grid.

    Orientation was checked against the atlas mask and Prox1 in dentate gyrus; LR is
    unverifiable against a symmetric atlas and left as stored.
    """
    img = nib.Nifti1Image.from_bytes(gzip.decompress(blob))
    vol = np.asarray(img.dataobj).transpose(1, 2, 0)[::-1, ::-1, :]
    assert vol.shape == ATLAS_SHAPE, vol.shape
    return np.ascontiguousarray(vol)


def load_gene(gene: str, cache_dir: Path) -> np.ndarray:
    """Fetch one gene volume, caching the compressed file locally (~1–20 MB each)."""
    path = cache_dir / f"{gene}.nii.gz"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download(gene))
    return decode(path.read_bytes())
