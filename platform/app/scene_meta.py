"""Metadatos reales de escena del dataset (satélite y fecha) a partir de su CSV."""
from __future__ import annotations

import csv
import re
from functools import lru_cache

from .config import DATASET_ROOT

_FNAME = re.compile(r"^(\d+)\s+(S1[AB])_\w+?_(\d{8})T(\d{6})")


@lru_cache(maxsize=1)
def _scenes() -> dict[int, dict]:
    path = DATASET_ROOT / "CSV" / "Oil_Stats_manual_windows_8bits_800x600.csv"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter=";"):
            m = _FNAME.match(row.get("IMG_FNAME", ""))
            if not m:
                continue
            n = int(m.group(1))
            d = m.group(3)
            t = m.group(4)
            out.setdefault(n, {
                "satellite": f"Sentinel-{m.group(2)[1]}{m.group(2)[2]}",
                "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                "time_utc": f"{t[:2]}:{t[2:4]}",
                "mode": row.get("BEAM_MODE"),
            })
    return out


def for_sample(sample_id: str) -> dict | None:
    m = re.match(r"IMG_(\d+)_TILE", sample_id)
    if not m:
        return None
    return _scenes().get(int(m.group(1)))
