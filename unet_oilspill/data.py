"""Carga de imagenes y mascaras del dataset OCEAN GUARD (solo NumPy y Pillow).

Solo lectura: nunca escribe en el directorio del dataset. La conversion
binaria (mask > 0) se hace en memoria.

Convenciones:
- Imagen: RGB normalizado con estadisticas ImageNet (media/desviacion de torchvision).
- Padding: se aplica a imagen normalizada y mascara con 0. Para la imagen, 0 en espacio
  normalizado equivale al color medio de ImageNet; para la mascara, 0 es SEA.
- Entrenamiento: recortes aleatorios de 256x256 (mismas coordenadas en imagen y mascara).
- Validacion y test: imagen completa, con padding a multiplo de 32 (600x800 -> 608x800).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DATASET_ROOT = Path(r"C:\Users\ANDAMIOS\Desktop\OCEAN-GUARD-AI\oilspill-dataset-original")
EXPECTED_COUNTS = {"train": 108, "val": 43, "test": 22}

# ResNet34 con 5 niveles de submuestreo (stride 32): dimensiones multiplo de 32.
MULTIPLE = 32
CROP_SIZE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path


def read_split(root: Path, split: str) -> list[Sample]:
    """Lee splits/<split>.csv oficial y verifica el numero de muestras y que existan los archivos."""
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"split desconocido: {split}")
    samples: list[Sample] = []
    with (root / "splits" / f"{split}.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(
                Sample(row["sample_id"], root / row["image_path"], root / row["mask_path"])
            )
    if len(samples) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split}: {len(samples)} muestras, se esperaban {EXPECTED_COUNTS[split]}"
        )
    for s in samples:
        if not s.image_path.is_file() or not s.mask_path.is_file():
            raise FileNotFoundError(f"{s.sample_id}: falta imagen o mascara")
    return samples


def load_image(path: Path) -> np.ndarray:
    """PNG RGB -> float32 HxWx3 normalizado con estadisticas ImageNet."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def load_mask_binary(path: Path) -> np.ndarray:
    """LABELS-1D -> float32 HxW binaria.

    0 = SEA -> 0.0
    1 = OIL SPILL y 2 = SEEPAGE -> 1.0 (clase positiva SLICK, equivalente a mask > 0)
    """
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 0).astype(np.float32)


def pad_amounts(height: int, width: int, multiple: int = MULTIPLE) -> tuple[int, int, int, int]:
    """Devuelve (top, bottom, left, right) para llevar HxW al multiplo superior.

    Con 600x800 -> (4, 4, 0, 0): 600 pasa a 608.
    """
    ph = (-height) % multiple
    pw = (-width) % multiple
    top = ph // 2
    left = pw // 2
    return top, ph - top, left, pw - left


def pad_hw(arr: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """Aplica padding con ceros a las dos primeras dimensiones (HxW o HxWxC)."""
    top, bottom, left, right = pads
    if arr.ndim == 2:
        widths = ((top, bottom), (left, right))
    else:
        widths = ((top, bottom), (left, right), (0, 0))
    return np.pad(arr, widths, mode="constant", constant_values=0)


def crop_hw(arr: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """Quita el padding para recuperar la resolucion original (HxW o HxWxC)."""
    top, bottom, left, right = pads
    h, w = arr.shape[0], arr.shape[1]
    return arr[top : h - bottom, left : w - right]


def random_crop_coords(
    height: int, width: int, size: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Esquina superior izquierda de un recorte size x size aleatorio dentro de HxW."""
    if height < size or width < size:
        raise ValueError(f"imagen {height}x{width} menor que el recorte {size}")
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return top, left


def crop_pair(
    image: np.ndarray, mask: np.ndarray, top: int, left: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta imagen (HxWxC) y mascara (HxW) con exactamente las mismas coordenadas."""
    img_c = image[top : top + size, left : left + size]
    msk_c = mask[top : top + size, left : left + size]
    return img_c, msk_c
