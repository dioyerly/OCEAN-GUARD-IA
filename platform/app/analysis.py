"""Análisis de la máscara predicha: componentes, áreas, métricas de referencia y superposición."""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def label_components(mask: np.ndarray, min_pixels: int = 20) -> list[dict]:
    """Componentes conectadas (8 vecinos). Devuelve las de mayor área primero."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q = deque([(y0, x0)])
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if len(pts) >= min_pixels:
            arr = np.array(pts)
            comps.append({
                "area_px": int(len(pts)),
                "centroid_px": [round(float(arr[:, 0].mean()), 1), round(float(arr[:, 1].mean()), 1)],
                "bbox_px": [int(arr[:, 0].min()), int(arr[:, 1].min()),
                            int(arr[:, 0].max()), int(arr[:, 1].max())],
            })
    comps.sort(key=lambda c: c["area_px"], reverse=True)
    return comps


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Elimina componentes (8 vecinos) con menos de min_pixels píxeles, conservando el resto exacto."""
    if min_pixels <= 1 or not mask.any():
        return mask
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    sizes = [0]
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        q = deque([(y0, x0)])
        cnt = 0
        while q:
            y, x = q.popleft()
            cnt += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
        sizes.append(cnt)
    keep = np.array(sizes) >= min_pixels
    keep[0] = False
    return keep[lab]


def binary_scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Dice e IoU de SLICK contra una máscara de referencia (solo para ejemplos del dataset)."""
    p = pred.astype(bool)
    r = ref.astype(bool)
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    eps = 1e-9
    return {
        "dice": round(2 * tp / (2 * tp + fp + fn + eps), 4),
        "iou": round(tp / (tp + fp + fn + eps), 4),
        "reference_slick_pct": round(100.0 * r.mean(), 4),
    }


def overlay(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Superpone el derrame en rojo semitransparente y sus bordes en amarillo."""
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    red = np.array([255.0, 0.0, 0.0])
    out[m] = out[m] * 0.45 + red * 0.55
    interior = (
        np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    )
    edge = m & ~interior
    out[edge] = np.array([255.0, 230.0, 0.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def slick_layer(mask: np.ndarray) -> Image.Image:
    """Capa RGBA transparente: derrame en naranja-rojo con borde claro, para superponer en el mapa."""
    m = mask.astype(bool)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    rgba[m] = (255, 90, 40, 170)
    interior = np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    rgba[m & ~interior] = (255, 220, 120, 235)
    return Image.fromarray(rgba, "RGBA")


COLOR_WARN_CHROMA = 0.03


def input_warnings(rgb: np.ndarray) -> list[dict]:
    """Avisos si la imagen no se parece a lo que el modelo conoce (radar SAR en escala de grises)."""
    out = []
    a = rgb.astype(np.float32)
    chroma = float((a.max(-1) - a.min(-1)).mean() / 255.0)
    if chroma > COLOR_WARN_CHROMA:
        out.append({"code": "color", "text": (
            "Esta imagen tiene color. El modelo solo se entrenó con imágenes de radar (SAR) en escala de grises, "
            "así que este resultado probablemente no es significativo (una captura de pantalla o una foto no sirven).")})
    if min(rgb.shape[:2]) < 256:
        out.append({"code": "small", "text": "La imagen es muy pequeña (menos de 256 px de lado); el resultado puede ser poco fiable."})
    return out


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(bool) * 255).astype(np.uint8), "L")


def to_rgb_uint8(im: Image.Image) -> np.ndarray:
    """Convierte cualquier imagen (incluidos TIFF de 16 bits o float) a RGB uint8."""
    if im.mode in ("F", "I", "I;16", "I;16B", "I;16L"):
        arr = np.asarray(im).astype(np.float64)
        lo, hi = np.nanpercentile(arr, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
        gray = arr.astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    if im.mode == "L":
        gray = np.asarray(im)
        return np.stack([gray, gray, gray], axis=-1)
    return np.asarray(im.convert("RGB"))


def summarize(mask: np.ndarray, prob: np.ndarray, pixel_km2: float | None) -> dict:
    total = int(mask.size)
    slick = int(mask.sum())
    comps = label_components(mask)
    out = {
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "total_px": total,
        "slick_px": slick,
        "slick_pct": round(100.0 * slick / total, 4),
        "mean_prob_slick": round(float(prob[mask].mean()), 4) if slick else None,
        "n_components": len(comps),
        "components": comps[:20],
        "slick_area_km2": round(slick * pixel_km2, 4) if pixel_km2 else None,
    }
    return out
