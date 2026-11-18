"""Georreferencia: lectura de GeoTIFF (EPSG:4326) y cálculo de huellas en lat/lon."""
from __future__ import annotations

import math
from typing import BinaryIO

from PIL import Image

TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_GEOKEYS = 34735
GEOKEY_MODEL_TYPE = 1024
GEOKEY_GEOGRAPHIC_TYPE = 2048
MODEL_TYPE_GEOGRAPHIC = 2
EPSG_WGS84 = 4326

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQUATOR = 111320.0


def _geokey(keys, key_id: int):
    """Busca un GeoKey en GeoKeyDirectoryTag (cabecera de 4 valores, luego entradas de 4)."""
    if not keys or len(keys) < 4:
        return None
    n = int(keys[3])
    for i in range(n):
        base = 4 + 4 * i
        if base + 3 >= len(keys):
            break
        kid, loc, _count, value = keys[base:base + 4]
        if kid == key_id and loc == 0:
            return value
    return None


def read_georef(source: BinaryIO | str) -> dict | None:
    """Devuelve {lon0, lat0, sx, sy, epsg} si el TIFF es geográfico; si no, None."""
    try:
        im = Image.open(source)
    except Exception:
        return None
    tags = getattr(im, "tag_v2", {})
    scale = tags.get(TAG_PIXEL_SCALE)
    tie = tags.get(TAG_TIEPOINT)
    keys = tags.get(TAG_GEOKEYS)
    if not scale or not tie or len(tie) < 6:
        return None
    model_type = _geokey(keys, GEOKEY_MODEL_TYPE)
    if model_type is not None and model_type != MODEL_TYPE_GEOGRAPHIC:
        return None  # proyectado: no soportado en esta versión
    return {
        "lon0": float(tie[3]),
        "lat0": float(tie[4]),
        "sx": float(scale[0]),
        "sy": float(scale[1]),
        "epsg": EPSG_WGS84,
    }


def bounds_from_georef(georef: dict, width: int, height: int) -> list[list[float]]:
    """[[sur, oeste], [norte, este]] de la imagen georreferenciada."""
    north = georef["lat0"]
    south = north - height * georef["sy"]
    west = georef["lon0"]
    east = west + width * georef["sx"]
    return [[south, west], [north, east]]


def manual_bounds(lat: float, lon: float, width: int, height: int, pixel_m: float) -> list[list[float]]:
    """Huella aproximada centrada en (lat, lon) con resolución supuesta pixel_m."""
    if not (math.isfinite(lat) and math.isfinite(lon)) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("coordenadas fuera de rango")
    if abs(lat) > 85:
        raise ValueError("la latitud está demasiado cerca de un polo para calcular una huella")
    if not math.isfinite(pixel_m) or pixel_m <= 0 or width <= 0 or height <= 0:
        raise ValueError("escala o tamaño de imagen no válidos")
    dlat = height * pixel_m / M_PER_DEG_LAT
    dlon = width * pixel_m / (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)))
    return [[lat - dlat / 2, lon - dlon / 2], [lat + dlat / 2, lon + dlon / 2]]


def pixel_size_m(bounds: list[list[float]], width: int, height: int) -> tuple[float, float]:
    """Tamaño aproximado de un píxel en metros (ancho, alto) a partir de la huella."""
    (south, west), (north, east) = bounds
    lat_c = math.radians((north + south) / 2)
    width_m = (east - west) * M_PER_DEG_LON_EQUATOR * math.cos(lat_c)
    height_m = (north - south) * M_PER_DEG_LAT
    return width_m / width, height_m / height


def pixel_area_km2(bounds: list[list[float]], width: int, height: int) -> float:
    px_w, px_h = pixel_size_m(bounds, width, height)
    return (px_w * px_h) / 1e6


COMPASS_ES = ["norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste"]


def compass_es(deg: float) -> str:
    """Punto cardinal en español (8 vientos) para un rumbo en grados."""
    return COMPASS_ES[int(((deg % 360) + 22.5) // 45) % 8]


def destination_point(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Punto a dist_m metros desde (lat, lon) siguiendo un rumbo (esfera de 6371 km)."""
    r = 6371000.0
    d = dist_m / r
    br = math.radians(bearing_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def exif_gps(im) -> tuple[float, float] | None:
    """Latitud y longitud del GPS guardado en los metadatos EXIF de una foto (si existen)."""
    try:
        gps = im.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps:
        return None

    def conv(v, ref):
        d, m, sec = (float(x) for x in v)
        val = d + m / 60.0 + sec / 3600.0
        return -val if str(ref).upper() in ("S", "W") else val

    try:
        lat = conv(gps[2], gps[1])
        lon = conv(gps[4], gps[3])
    except Exception:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    return round(lat, 6), round(lon, 6)


def valid_bounds(b) -> bool:
    """True si la huella [[sur, oeste], [norte, este]] es finita, ordenada y de tamaño plausible (< 5°)."""
    try:
        (s, w), (n, e) = b
        vals = (s, w, n, e)
    except (TypeError, ValueError):
        return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return False
    return -90 <= s < n <= 90 and -180 <= w < e <= 180 and (n - s) < 5 and (e - w) < 5
