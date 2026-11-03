"""Step 1: parse the 33 HGEA coronal SVG plates into labelled domain polygons (plate mm).

Input:  data_raw/hgea_svg/HGEA-Coronal-0NNBW.svg
Output: data_work/plates.pkl, data_work/plates_summary.csv (per plate x side x domain: area, parts)
"""

from __future__ import annotations

import pickle

import pandas as pd

import _path  # noqa: F401
from hgea_ara3d.config import SVG_DIR, WORK
from hgea_ara3d.plates import parse_plate


def main() -> None:
    WORK.mkdir(exist_ok=True)
    plates, rows = [], []
    for f in sorted(SVG_DIR.glob("HGEA-Coronal-*.svg")):
        p = parse_plate(f)
        plates.append(p)
        for side, doms in p.domains.items():
            for name, g in doms.items():
                rows.append(dict(level=p.level, bregma_mm=p.bregma_mm, side=side, domain=name,
                                 area_mm2=round(g.area, 4), parts=len(getattr(g, "geoms", [g]))))
        print(f"level {p.level}: {sum(len(d) for d in p.domains.values())} domain shapes {'; '.join(p.warnings) or ''}")
    with open(WORK / "plates.pkl", "wb") as fh:
        pickle.dump(plates, fh)
    pd.DataFrame(rows).to_csv(WORK / "plates_summary.csv", index=False)
    print(f"{len(plates)} plates -> {WORK / 'plates.pkl'}")


if __name__ == "__main__":
    main()
