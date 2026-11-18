"""Recalibra las estadísticas de BatchNorm del DECODER con imágenes de entrenamiento a tamaño completo.

Motivo: el modelo se entrenó con recortes de 256 px (batch 2) y se usa con imágenes completas; las
estadísticas de BatchNorm del decoder quedaron mal ajustadas a ese tamaño. Esta recalibración no usa
etiquetas ni imágenes de validación/test, no modifica best.pt y guarda solo los buffers de BatchNorm.

Uso (desde platform/): ../unet_oilspill/.venv/Scripts/python.exe tools/recalibrate_bn.py
Salida: assets/bn_decoder_recal.pt
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app import config, inference  # noqa: E402
import data as D  # noqa: E402


def main() -> None:
    seg = inference.Segmenter(bn_recal=None)  # modelo base, sin recalibración previa
    model = seg._model
    bns = [m for name, m in model.named_modules()
           if isinstance(m, torch.nn.BatchNorm2d) and not name.startswith("encoder.")]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # promedio acumulado exacto
    model.eval()
    for b in bns:
        b.train()
    train = D.read_split(D.DEFAULT_DATASET_ROOT, "train")  # SOLO entrenamiento
    with torch.no_grad():
        for s in train:
            rgb = np.asarray(Image.open(s.image_path).convert("RGB")).astype(np.float32) / 255.0
            arr = (rgb - D.IMAGENET_MEAN) / D.IMAGENET_STD
            pads = D.pad_amounts(*arr.shape[:2], D.MULTIPLE)
            x = torch.from_numpy(np.ascontiguousarray(D.pad_hw(arr, pads))).permute(2, 0, 1).unsqueeze(0)
            model(x)
    state = {k: v.clone() for k, v in model.state_dict().items()
             if not k.startswith("encoder.") and ("running_mean" in k or "running_var" in k or "num_batches_tracked" in k)}
    out = config.BN_RECAL
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "n_images": len(train), "split": "train"}, out)
    print(f"Guardado {out} con {len(state)} buffers, calibrado con {len(train)} imágenes de entrenamiento.")


if __name__ == "__main__":
    main()
