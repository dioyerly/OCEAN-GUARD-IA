"""Distancia a la costa real (Natural Earth 1:50M, dominio público).

Escala adecuada para distancias de decenas de km. No sirve para navegación ni para decidir
operaciones: es una referencia para entender qué tan cerca está la costa.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from .config import PLATFORM_DIR
from .geo import compass_es

ASSET = PLATFORM_DIR / "assets" / "ne_50m_coastline.geojson"
KM_LAT = 110.574
KM_LON_EQ = 111.320


@lru_cache(maxsize=1)
def _segments() -> tuple[np.ndarray, np.ndarray] | None:
    """Segmentos (A, B) como arrays (n, 2) con columnas (lon, lat)."""
    if not ASSET.exists():
        return None
    gj = json.loads(ASSET.read_text(encoding="utf-8"))
    a_list, b_list = [], []
    for f in gj["features"]:
        geom = f["geometry"]
        lines = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for line in lines:
            arr = np.asarray(line, dtype=np.float64)
            if len(arr) < 2:
                continue
            a_list.append(arr[:-1, :2])
            b_list.append(arr[1:, :2])
    return np.vstack(a_list), np.vstack(b_list)


def available() -> bool:
    return _segments() is not None


def _bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def nearest_coast(lat: float, lon: float, search_deg: float = 12.0) -> dict | None:
    """Costa más cercana: distancia (km), punto y rumbo desde (lat, lon). None si no hay datos/costa."""
    seg = _segments()
    if seg is None:
        return None
    a, b = seg
    near = (
        (np.minimum(a[:, 1], b[:, 1]) <= lat + search_deg) & (np.maximum(a[:, 1], b[:, 1]) >= lat - search_deg)
        & (np.minimum(a[:, 0], b[:, 0]) <= lon + search_deg) & (np.maximum(a[:, 0], b[:, 0]) >= lon - search_deg)
    )
    if not near.any():
        return None
    a, b = a[near], b[near]
    kx = KM_LON_EQ * math.cos(math.radians(lat))
    ax, ay = (a[:, 0] - lon) * kx, (a[:, 1] - lat) * KM_LAT
    bx, by = (b[:, 0] - lon) * kx, (b[:, 1] - lat) * KM_LAT
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = np.where(denom > 0, np.clip(-(ax * dx + ay * dy) / np.where(denom > 0, denom, 1), 0, 1), 0)
    px, py = ax + t * dx, ay + t * dy
    d = np.hypot(px, py)
    i = int(np.argmin(d))
    plat = lat + py[i] / KM_LAT
    plon = lon + px[i] / kx
    brg = _bearing(lat, lon, plat, plon)
    return {
        "distance_km": round(float(d[i]), 1),
        "point": [round(plat, 4), round(plon, 4)],
        "bearing_deg": round(brg, 0),
        "direction_es": compass_es(brg),
    }


def approach(track: list[list[float]]) -> dict | None:
    """Aproximación de una trayectoria [[lat, lon, hora], ...] a la costa.

    Devuelve la distancia inicial, la mínima, la hora en que ocurre y si el desplazamiento
    se acerca a la costa (distancia mínima claramente menor que la inicial).
    """
    if not track:
        return None
    first = nearest_coast(track[0][0], track[0][1])
    if first is None:
        return None
    best = first
    best_h = track[0][2]
    for lat, lon, h in track[1:]:
        nc = nearest_coast(lat, lon)
        if nc and nc["distance_km"] < best["distance_km"]:
            best, best_h = nc, h
    approaching = best["distance_km"] < first["distance_km"] - 0.5
    return {
        "start": first,
        "closest": best,
        "closest_hour": best_h,
        "approaching": approaching,
        "reaches_coast": best["distance_km"] <= 1.0,
        "end_distance_km": nearest_coast(track[-1][0], track[-1][1])["distance_km"] if nearest_coast(track[-1][0], track[-1][1]) else None,
    }
