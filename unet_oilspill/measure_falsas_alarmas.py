"""Mide falsas alarmas (CSIRO etiqueta 0), deteccion (etiqueta 1) y Dice/precision en val original.

Uso: python measure_falsas_alarmas.py --checkpoint RUTA.pt --bn RUTA_bn.pt --out resultado.json
NO evalua el test original ni la region SIN para elegir modelo (SIN solo para el reporte final).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fa_eval as F
from app import inference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seg = inference.Segmenter(checkpoint=args.checkpoint, bn_recal=args.bn)
    assert seg.info["bn_decoder_recalibrated"], "la recalibracion BN no se cargo"
    t0 = time.time()
    res = {"checkpoint": str(args.checkpoint), "bn": str(args.bn),
           "threshold": F.THRESHOLD, "min_component_px": F.MIN_PX}
    val_orig = F.eval_val_original(seg)
    res["val_original"] = val_orig
    print("val original:", val_orig, round(time.time() - t0), "s", flush=True)
    neg_test = F.eval_csiro(seg, F.read_index("test", 0))
    neg_val = F.eval_csiro(seg, F.read_index("val", 0))
    res["fa_test_SIN"] = F.rate(neg_test)
    res["fa_val"] = F.rate(neg_val)
    print("FA test SIN:", res["fa_test_SIN"], "FA val:", res["fa_val"], round(time.time() - t0), "s", flush=True)
    pos = F.eval_csiro(seg, F.read_index("eval_only", 1))
    res["det_SIN"] = F.rate(pos, {"SIN"})
    res["det_all"] = F.rate(pos)
    print("deteccion SIN:", res["det_SIN"], "todos:", res["det_all"], round(time.time() - t0), "s", flush=True)
    res["per_image"] = {"neg_test": neg_test, "neg_val": neg_val, "pos": pos}
    res["seconds"] = round(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
