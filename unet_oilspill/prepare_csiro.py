"""Indice CSV del dataset CSIRO Sentinel-1 (recortes 400x400) con particion POR REGION.

Uso: python prepare_csiro.py [--root ../datasets_extra/csiro] [--out ../datasets_extra/csiro/index.csv]

Salida: ruta, etiqueta (0 = sin petroleo, 1 = con petroleo), region, particion.
- Etiqueta 0: prueba (region SIN), validacion (regiones medianas, ~15 %), entrenamiento (resto, GBR incluida).
- Etiqueta 1: sin mascara, no entra al entrenamiento. Particion = 'eval_only' (se usa SIN para la medicion final).
- Verifica que los 3 canales del JPEG sean identicos (gris repetido) y que el tamano sea 400x400.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

TEST_REGIONS = {"SIN"}
VAL_REGIONS = {"GIB", "JAV", "SFr", "PHI", "BAH", "EGY", "NOR", "LUC"}  # 551 de 3725 = 14,8 %
NAME_RE = re.compile(r"_([A-Za-z]+)_cls_([01])\.jpg$")


def load_gray(path: Path) -> np.ndarray:
    """JPEG 400x400 RGB (gris repetido) -> uint8 HxW de un solo canal."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
        raise ValueError(f"canales distintos en {path}")
    return arr[..., 0]


def partition(label: int, region: str) -> str:
    if label == 1:
        return "eval_only"
    if region in TEST_REGIONS:
        return "test"
    return "val" if region in VAL_REGIONS else "train"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here.parent / "datasets_extra" / "csiro")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "index.csv"

    rows, bad = [], 0
    for cls in (0, 1):
        for p in sorted((args.root / str(cls)).glob("*.jpg")):
            m = NAME_RE.search(p.name)
            if not m or int(m.group(2)) != cls:
                raise ValueError(f"nombre inesperado: {p.name}")
            g = load_gray(p)
            if g.shape != (400, 400):
                bad += 1
            rows.append({"path": p.relative_to(args.root).as_posix(), "label": cls,
                         "region": m.group(1), "partition": partition(cls, m.group(1))})
    n0 = sum(r["label"] == 0 for r in rows)
    n1 = sum(r["label"] == 1 for r in rows)
    if (n0, n1) != (3725, 1905):
        raise SystemExit(f"conteos inesperados: {n0} / {n1}")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "region", "partition"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} filas -> {out}; imagenes con tamano distinto de 400x400: {bad}")
    print("canales identicos en todas las imagenes: OK")
    c = Counter((r["label"], r["partition"]) for r in rows)
    for k in sorted(c):
        print(k, c[k])
    for part in ("train", "val", "test"):
        print(part, sorted({r["region"] for r in rows if r["label"] == 0 and r["partition"] == part}))


if __name__ == "__main__":
    main()
