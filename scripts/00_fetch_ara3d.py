"""Step 0: download the 3D ARA (Allen 2011 coronal Nissl reconstruction, CCFv2) and its 2011 ontology.

Output: data_raw/ara3d/{atlasVolume,annotation}/*.mhd|.raw and structures_2011.csv (about 400 MB unpacked).
The HGEA plates (data_raw/hgea_svg/HGEA-Coronal-0NNBW.svg) come from the original HGEA files and are
not downloaded here.
"""

from __future__ import annotations

import urllib.request
import zipfile

import _path  # noqa: F401
from hgea_ara3d.config import ARA3D_DIR, ARA3D_FILES


def main() -> None:
    for rel, url in ARA3D_FILES.items():
        out = ARA3D_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, out)
    zipfile.ZipFile(ARA3D_DIR / "zips/P56_atlasVolume.zip").extractall(ARA3D_DIR)
    zipfile.ZipFile(ARA3D_DIR / "zips/P56_Mouse_annotation.zip").extractall(ARA3D_DIR / "annotation")
    print("ok:", sorted(p.name for p in ARA3D_DIR.rglob("*.raw")))


if __name__ == "__main__":
    main()
