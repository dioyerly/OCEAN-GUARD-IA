"""Inferencia con la U-Net ResNet34 de referencia (best.pt, época 4).

Reutiliza el código del entrenamiento en unet_oilspill (model.py, data.py), sin duplicarlo.
"""
from __future__ import annotations

import sys
import threading

import numpy as np
import torch

# En Render (512 MB) un solo hilo basta y evita reservar memoria extra por hilo.
torch.set_num_threads(int(__import__("os").environ.get("OCEAN_TORCH_THREADS", "1")))

from .config import BN_RECAL, CHECKPOINT, UNET_DIR

if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

import data as unet_data  # noqa: E402  (requiere sys.path ajustado arriba)
import model as unet_model  # noqa: E402

TILE = 512
OVERLAP = 64
MULTIPLE = unet_data.MULTIPLE


class Segmenter:
    """Carga el modelo una sola vez y expone predicción de probabilidad por píxel."""

    def __init__(self, checkpoint=CHECKPOINT, device: str = "cpu", bn_recal=BN_RECAL):
        self.device = torch.device(device)
        model, weights_report = unet_model.build_model(load_pretrained=False)
        # mmap: los pesos se leen del archivo sin duplicarlos en RAM al cargarlos en el modelo.
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(ck["model"], strict=True)
        recal = False
        if bn_recal is not None and bn_recal.exists():
            # Estadísticas de BatchNorm del decoder recalibradas con imágenes de entrenamiento (tools/recalibrate_bn.py).
            saved = torch.load(bn_recal, map_location="cpu", weights_only=False)["state"]
            result = model.load_state_dict(saved, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(f"claves inesperadas en la recalibración de BN: {result.unexpected_keys}")
            recal = True
        model.to(self.device).eval()
        meta = {"epoch": ck.get("global_epoch"), "phase": ck.get("phase"), "val_dice": float(ck.get("val_dice", 0.0))}
        del ck  # libera el checkpoint completo; el modelo ya tiene sus pesos
        self._model = model
        self._lock = threading.Lock()
        self.info = {
            "checkpoint": str(checkpoint),
            "epoch": meta["epoch"],
            "phase": meta["phase"],
            "val_dice": round(meta["val_dice"], 4),
            "device": str(self.device),
            "weights_sha256_ok": weights_report["sha256"] == unet_model.WEIGHTS_SHA256,
            "bn_decoder_recalibrated": recal,
            "architecture": "U-Net + ResNet34 (smp), salida binaria SLICK",
        }

    @torch.inference_mode()
    def predict_prob(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: HxWx3 uint8. Devuelve HxW float32 con la probabilidad de SLICK."""
        h, w = rgb.shape[:2]
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - unet_data.IMAGENET_MEAN) / unet_data.IMAGENET_STD
        pads = unet_data.pad_amounts(h, w, MULTIPLE)
        padded = unet_data.pad_hw(arr, pads)
        with self._lock:
            if max(h, w) <= TILE:
                prob = self._forward(padded)
            else:
                prob = self._tiled(padded)
        return unet_data.crop_hw(prob, pads).astype(np.float32)

    def _forward(self, arr: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        logits = self._model(x)
        return torch.sigmoid(logits)[0, 0].cpu().numpy()

    def _tiled(self, arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape[:2]
        acc = np.zeros((H, W), dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.float32)
        step = TILE - OVERLAP
        ys = sorted(set(list(range(0, H - TILE + 1, step)) + [H - TILE]))
        xs = sorted(set(list(range(0, W - TILE + 1, step)) + [W - TILE]))
        for y in ys:
            for x in xs:
                tile = arr[y:y + TILE, x:x + TILE]
                acc[y:y + TILE, x:x + TILE] += self._forward(tile)
                cnt[y:y + TILE, x:x + TILE] += 1.0
        return acc / np.maximum(cnt, 1.0)


_segmenter: Segmenter | None = None
_load_error: str | None = None


def get_segmenter() -> Segmenter | None:
    """Devuelve el modelo cargado o None; el error queda en load_error()."""
    global _segmenter, _load_error
    if _segmenter is None and _load_error is None:
        try:
            _segmenter = Segmenter()
        except Exception as exc:  # se muestra en la interfaz, no se oculta
            _load_error = f"{type(exc).__name__}: {exc}"
    return _segmenter


def load_error() -> str | None:
    return _load_error
