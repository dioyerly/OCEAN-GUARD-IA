"""Rutas y parámetros de la plataforma OCEAN GUARD AI.

Todas las rutas pueden sobrescribirse con variables de entorno, pero los valores
por defecto apuntan a los artefactos ya existentes del proyecto.
"""
from __future__ import annotations

import os
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PLATFORM_DIR.parent
UNET_DIR = ROOT_DIR / "unet_oilspill"

# Modelo de referencia: época 4 (fase 1), Dice de validación 0,5841. No se reemplaza.
CHECKPOINT = Path(os.environ.get(
    "OCEAN_CHECKPOINT",
    UNET_DIR / "runs" / "unet_resnet34_demo_12ep" / "best.pt",
))

# Dataset original: solo lectura. Se usa para los ejemplos de demostración (split val).
DATASET_ROOT = Path(os.environ.get("OCEAN_DATASET", ROOT_DIR / "oilspill-dataset-original"))

# Datos generados por la plataforma (historial, máscaras, simulaciones).
DATA_DIR = Path(os.environ.get("OCEAN_PLATFORM_DATA", PLATFORM_DIR / "data"))

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SIDE_PX = 4096
# Punto de operación calibrado sobre el split de validación (43 imágenes), con BatchNorm del decoder
# recalibrado con imágenes de entrenamiento: Dice micro 0,60 (umbral 0,5, sin recalibrar) -> ~0,79
# (umbral 0,95 + filtro de 100 px; medido 0,789 con el pipeline entregado). El umbral se ajustó en val: la cifra es optimista y el óptimo es
# agudo. El conjunto test no se ha evaluado.
DEFAULT_THRESHOLD = 0.95
MIN_COMPONENT_PX = 100
BN_RECAL = PLATFORM_DIR / "assets" / "bn_decoder_recal.pt"  # ver tools/recalibrate_bn.py
DEFAULT_PIXEL_M = 10.0  # resolución supuesta si la imagen no trae georreferencia

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DISCLAIMER_DETECTION = (
    "Resultado generado por un modelo de aprendizaje automático entrenado con un dataset "
    "académico. Es una detección asistida que requiere validación humana; no sustituye "
    "vigilancia oficial ni debe usarse para decisiones operativas."
)
DISCLAIMER_SIMULATION = (
    "SIMULACIÓN ILUSTRATIVA. Modelo de deriva simplificado (advección por viento, difusión "
    "y degradación) con supuestos declarados. No es una predicción validada ni un producto "
    "oceanográfico."
)

# Fiabilidad medida en el split de validación con el punto de operación actual
# (umbral 0,95, mínimo 100 px, BN del decoder recalibrado). Es una medición en datos usados para ajustar el umbral.
VAL_RELIABILITY = {
    "precision": 0.873,   # de cada 100 píxeles marcados, ~87 eran derrame de referencia
    "recall": 0.720,      # de cada 100 píxeles de derrame de referencia, el modelo marcó ~72
    "dice": 0.789,
    "note": ("Las 43 imágenes de validación contienen derrame: no se ha podido medir cuántas "
             "falsas alarmas produce el modelo en imágenes sin derrame."),
}
DATASET_REGION = "Golfo de México (región de Cantarell, según la documentación del dataset)"
