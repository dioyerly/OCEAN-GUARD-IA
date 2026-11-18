"""U-Net con encoder ResNet34 (segmentation_models_pytorch), pesos ImageNet1K_V1 locales.

Sin descargas: encoder_weights=None y los pesos se cargan desde pretrained/ con
verificacion de SHA256 y de claves (faltantes y sobrantes).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from data import MULTIPLE

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "pretrained" / "resnet34-b627a593.pth"
WEIGHTS_SHA256 = "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(weights_path: Path = WEIGHTS_PATH) -> tuple[torch.nn.Module, dict]:
    """Construye U-Net ResNet34 con 3 canales de entrada y 1 salida (logits SLICK).

    Lanza RuntimeError si el archivo de pesos no coincide con el SHA256 esperado o si
    hay claves faltantes o sobrantes en el encoder.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"no existen los pesos locales: {weights_path}")
    digest = sha256_file(weights_path)
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"SHA256 no coincide: {digest} != {WEIGHTS_SHA256}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # smp elimina avgpool y fc del encoder; esas claves fc.* son las unicas que se descartan.
    dropped = sorted(k for k in state if k.startswith("fc."))
    provided = {k: v for k, v in state.items() if not k.startswith("fc.")}
    expected = set(model.encoder.state_dict().keys())
    missing_all = sorted(expected - set(provided))
    unexpected = sorted(set(provided) - expected)
    # Excepcion documentada: el checkpoint de torchvision no trae los contadores
    # num_batches_tracked de BatchNorm (no son pesos aprendidos; PyTorch los inicia a 0).
    # torchvision los rellena igual al cargar con strict=True. Cualquier otra clave detiene la carga.
    missing = [k for k in missing_all if not k.endswith("num_batches_tracked")]
    tolerated = [k for k in missing_all if k.endswith("num_batches_tracked")]

    report = {
        "weights_file": str(weights_path),
        "sha256": digest,
        "dropped_fc_keys": dropped,
        "expected_encoder_keys": len(expected),
        "loaded_keys": len(provided),
        "tolerated_missing_num_batches_tracked": len(tolerated),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    if missing or unexpected:
        raise RuntimeError(f"incompatibilidad de claves al cargar pesos: {report}")

    # smp sobrescribe load_state_dict del encoder y no devuelve resultado.
    # strict=False solo es seguro aqui porque las claves ya fueron verificadas arriba.
    model.encoder.load_state_dict(provided, strict=False)
    return model, report


def encoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return list(model.encoder.parameters())


def decoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    enc_ids = {id(p) for p in model.encoder.parameters()}
    return [p for p in model.parameters() if id(p) not in enc_ids]


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(trainable)


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "WEIGHTS_PATH",
    "WEIGHTS_SHA256",
    "MULTIPLE",
    "build_model",
    "sha256_file",
    "encoder_parameters",
    "decoder_parameters",
    "set_encoder_trainable",
    "count_trainable",
]
