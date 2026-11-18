"""Medicion de falsas alarmas / deteccion con el MISMO pipeline de la plataforma.

Regla (identica a platform/app/main.py): probabilidad > umbral (0,95) y descarte de componentes
de menos de 100 px. Un recorte 'marca derrame' si tras ese filtro queda al menos un pixel
(es decir, al menos una zona de >= 100 px). La plataforma no define otra regla de 'sugiere derrame'.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "platform"
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from app import analysis as A  # noqa: E402
from app import config  # noqa: E402
from app import inference  # noqa: E402

from metrics import Confusion  # noqa: E402

CSIRO_ROOT = ROOT / "datasets_extra" / "csiro"
THRESHOLD = config.DEFAULT_THRESHOLD
MIN_PX = config.MIN_COMPONENT_PX


def load_csiro_rgb(path: Path) -> np.ndarray:
    """JPEG gris (3 canales iguales) -> un canal -> se repite a 3 canales, como las imagenes de radar del modelo."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    g = arr[..., 0]
    return np.stack([g, g, g], axis=-1)


def read_index(partition: str | None = None, label: int | None = None) -> list[dict]:
    rows = list(csv.DictReader((CSIRO_ROOT / "index.csv").open(encoding="utf-8")))
    return [r for r in rows if (partition is None or r["partition"] == partition)
            and (label is None or int(r["label"]) == label)]


def final_mask(seg, rgb: np.ndarray) -> np.ndarray:
    prob = seg.predict_prob(rgb)
    return A.remove_small_components(prob > THRESHOLD, MIN_PX)


def eval_csiro(seg, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        m = final_mask(seg, load_csiro_rgb(CSIRO_ROOT / r["path"]))
        n = int(m.sum())
        out.append({"path": r["path"], "region": r["region"], "label": int(r["label"]),
                    "marked_px": n, "marked": n > 0})
    return out


def rate(results: list[dict], regions: set[str] | None = None) -> dict:
    sel = [x for x in results if regions is None or x["region"] in regions]
    k = sum(x["marked"] for x in sel)
    return {"n": len(sel), "marked": k, "rate": (k / len(sel)) if sel else None}


def eval_val_original(seg) -> dict:
    """Dice/precision/cobertura micro sobre las 43 imagenes de validacion originales (pipeline entregado)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data as D
    conf = Confusion()
    for s in D.read_split(D.DEFAULT_DATASET_ROOT, "val"):
        rgb = np.asarray(Image.open(s.image_path).convert("RGB"))
        conf.update(final_mask(seg, rgb), D.load_mask_binary(s.mask_path) > 0)
    sc = conf.scores()
    return {k: float(sc[k]) for k in ("dice", "precision", "recall", "iou")}
