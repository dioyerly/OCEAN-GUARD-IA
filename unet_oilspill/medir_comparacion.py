"""Mide el modelo NUEVO (best.pt entrenado en Colab) contra el ACTUAL en el mismo conjunto de prueba (split.json).

Uso (desde unet_oilspill/):
  python medir_comparacion.py --nuevo RUTA\\best.pt [--bn RUTA\\bn_decoder_recal.pt] [--out comparacion.json]

Metricas, por separado y en total: Dice, precision y recall; falsas alarmas = % de imagenes SIN derrame donde marca >= 100 px
con prob > 0,95; deteccion = % de imagenes CON derrame donde marca >= 100 px. Si no se da --bn, se recalibra el BatchNorm del
decoder del modelo nuevo con las originales de ENTRENAMIENTO (como la plataforma). No cambia el modelo por defecto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_final as T  # noqa: E402


def tabla(res: dict) -> str:
    f = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    filas = ["{:<22}{:>10}{:>10}".format("", "actual", "nuevo")]
    m = res["modelos"]
    for etiqueta, ruta in (("Dice (total)", ("conjunto_completo", "dice")), ("Precision (total)", ("conjunto_completo", "precision")),
                           ("Recall (total)", ("conjunto_completo", "recall")), ("Dice (con derrame)", ("con_derrame", "dice")),
                           ("Precision (con derr.)", ("con_derrame", "precision")), ("Recall (con derrame)", ("con_derrame", "recall")),
                           ("Deteccion (con derr.)", ("con_derrame", "deteccion")), ("Falsa alarma (sin)", ("sin_derrame", "falsa_alarma"))):
        filas.append("{:<22}{:>10}{:>10}".format(etiqueta, f(m["actual"][ruta[0]][ruta[1]]), f(m["nuevo_final"][ruta[0]][ruta[1]])))
    return "\n".join(filas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuevo", type=Path, default=T.OUT / "best.pt")
    ap.add_argument("--bn", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=T.OUT / "comparacion.json")
    a = ap.parse_args()
    bn = a.bn
    if bn is None:  # recalibra el BatchNorm del decoder con las originales de entrenamiento y lo guarda junto al best.pt
        bn = a.nuevo.with_name("bn_decoder_recal.pt")
        if not bn.exists():
            from app import inference
            seg = inference.Segmenter(checkpoint=a.nuevo, bn_recal=None)
            sp = json.loads((T.OUT / "split.json").read_text(encoding="utf-8"))
            by_id = {s.sample_id: s for n in ("train", "val", "test") for s in T.D.read_split(T.D.DEFAULT_DATASET_ROOT, n)}
            imgs = [by_id[i] for i in sp["items"]["train"]["originales"]]
            torch.save({"state": T.recalibrate_bn(seg._model, imgs), "n_images": len(imgs), "split": "train(final)"}, bn)
    T.stage_compare(nuevo=a.nuevo, bn=bn, out=a.out)
    print(tabla(json.loads(a.out.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
