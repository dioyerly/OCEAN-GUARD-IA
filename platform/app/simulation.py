"""Simulación ilustrativa de deriva de un derrame (advección + difusión + degradación).

El forzamiento puede ser manual (viento constante) o real (viento y corriente horarios de
Open-Meteo). En ambos casos el modelo es simplificado: no incluye oleaje, mareas, evaporación
detallada ni la física real del petróleo, y no está validado con observaciones.
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image

from .geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR, compass_es, valid_bounds
from .narrative import drift_story

GRID = 96                   # celdas a lo ancho/alto de la huella del análisis
MARGIN = 64                 # celdas extra a cada lado para ver hacia dónde se desplaza
WIND_FACTOR = 0.03          # deriva superficial ≈ 3 % de la velocidad del viento (supuesto)
DIFFUSIVITY_M2S = 50.0      # difusividad horizontal (supuesto)
DECAY_PER_HOUR = 0.02       # degradación/evaporación simplificada (supuesto)
MAX_STEPS = 15000
CELL_MIN_M = 100.0        # celdas más pequeñas exigen pasos de tiempo diminutos (inestabilidad)
MAX_CELLS = 300           # tope de celdas por lado del dominio completo
MILESTONES_H = (3, 6, 12)
N_EVEN_FRAMES = 12
VISIBLE_REL = 0.10
AFFECTED_REL = 0.05


def constant_forcing(speed_ms: float, wind_from_deg: float, hours: float) -> list[dict]:
    n = int(math.ceil(hours)) + 1
    return [{"wind_ms": speed_ms, "wind_from": wind_from_deg, "current_ms": 0.0, "current_to": 0.0} for _ in range(n)]


def velocity(f: dict) -> tuple[float, float]:
    """(u este, v norte) en m/s: deriva por viento (3 %) + corriente superficial si existe."""
    to = math.radians((f["wind_from"] + 180.0) % 360.0)
    w = WIND_FACTOR * f["wind_ms"]
    u, v = w * math.sin(to), w * math.cos(to)
    cur = f.get("current_ms")
    if cur:
        ct = math.radians(f.get("current_to") or 0.0)
        u += cur * math.sin(ct)
        v += cur * math.cos(ct)
    return u, v


def lagrangian_track(lat: float, lon: float, forcing: list[dict], hours: float) -> list[list[float]]:
    """Posición del centro de la mancha cada hora: [[lat, lon, hora], ...] (solo advección)."""
    track = [[round(lat, 5), round(lon, 5), 0.0]]
    x = y = 0.0  # metros hacia el este / norte
    steps_per_h = 6
    dt = 3600.0 / steps_per_h
    total = int(round(hours * steps_per_h))
    for s in range(total):
        f = forcing[min(int(s / steps_per_h), len(forcing) - 1)]
        u, v = velocity(f)
        x += u * dt
        y += v * dt
        if (s + 1) % steps_per_h == 0 or s == total - 1:
            la = lat + y / M_PER_DEG_LAT
            lo = lon + x / (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)))
            track.append([round(la, 5), round(lo, 5), round((s + 1) / steps_per_h, 2)])
    return track


def haversine_km(a: list[float], b: list[float]) -> float:
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dphi = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a: list[float], b: list[float]) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _shift(c: np.ndarray, axis: int, direction: int) -> np.ndarray:
    out = np.zeros_like(c)
    if axis == 1:
        if direction == 1:
            out[:, 1:] = c[:, :-1]
        else:
            out[:, :-1] = c[:, 1:]
    else:
        if direction == 1:
            out[1:, :] = c[:-1, :]
        else:
            out[:-1, :] = c[1:, :]
    return out


def _png_b64(rgba: np.ndarray, scale: int | None = None) -> str:
    if scale is None:
        scale = 3 if max(rgba.shape[:2]) <= 150 else 2 if max(rgba.shape[:2]) <= 250 else 1
    rgba = rgba[::-1]
    im = Image.fromarray(rgba, "RGBA").resize((rgba.shape[1] * scale, rgba.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render(c: np.ndarray, peak: float) -> str:
    norm = np.zeros_like(c) if peak <= 0 else np.clip(c / peak, 0, 1)
    rgba = np.zeros((c.shape[0], c.shape[1], 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 120, 20
    rgba[..., 3] = np.clip(np.sqrt(norm) * 225, 0, 225).astype(np.uint8)
    return _png_b64(rgba)


def _render_zone(zone: np.ndarray) -> str:
    rgba = np.zeros((zone.shape[0], zone.shape[1], 4), dtype=np.uint8)
    rgba[zone] = (255, 196, 0, 90)
    return _png_b64(rgba)


def _mask_origin(mask: np.ndarray, bounds: list[list[float]]) -> tuple[float, float]:
    """Centroide de la mancha en lat/lon (la imagen tiene el norte arriba)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    (south, west), (north, east) = bounds
    return (north - (float(ys.mean()) + 0.5) / h * (north - south),
            west + (float(xs.mean()) + 0.5) / w * (east - west))


def circular_mean_deg(values: list[float]) -> float:
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    return (math.degrees(math.atan2(s, c)) + 360) % 360


def _assert_finite(obj, path: str = "resultado") -> None:
    """Falla con un mensaje claro si algún número del resultado no es finito (nunca se convierte en cero)."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"La simulación produjo un valor no válido en «{path}».")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k != "png_b64":
                _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")


def simulate(
    mask: np.ndarray,
    bounds: list[list[float]],
    wind_speed_ms: float | None,
    wind_from_deg: float | None,
    hours: float,
    forcing: list[dict] | None = None,
    forcing_source: str = "viento constante indicado por el usuario",
) -> dict:
    """mask: HxW booleano (norte arriba). bounds: [[sur, oeste], [norte, este]] de la imagen."""
    if not mask.any():
        raise ValueError("No hay píxeles de derrame en el análisis: nada que simular.")
    if not (1 <= hours <= 72):
        raise ValueError("La duración debe estar entre 1 y 72 horas.")
    if forcing is None:
        if wind_speed_ms is None or wind_from_deg is None:
            raise ValueError("Indique la velocidad y la dirección del viento.")
        if not (0 <= wind_speed_ms <= 30):
            raise ValueError("La velocidad del viento debe estar entre 0 y 30 m/s.")
        if not (0 <= wind_from_deg <= 360):
            raise ValueError("La dirección del viento debe estar entre 0 y 360 grados.")
        forcing = constant_forcing(wind_speed_ms, wind_from_deg, hours)
        mode = "manual"
    else:
        mode = "weather"
    for i, f in enumerate(forcing):
        vals = (f.get("wind_ms"), f.get("wind_from"), f.get("current_ms") or 0.0, f.get("current_to") or 0.0)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"Las condiciones de viento o corriente de la hora {i} no son válidas; no se simula.")
    if not valid_bounds(bounds):
        raise ValueError("La huella geográfica del análisis no es válida; no se simula sobre un mapa.")
    vel = [velocity(f) for f in forcing]

    (south, west), (north, east) = bounds
    lat_c = math.radians((north + south) / 2)
    m_per_deg_lon = M_PER_DEG_LON_EQUATOR * math.cos(lat_c)
    width_m = (east - west) * m_per_deg_lon
    height_m = (north - south) * M_PER_DEG_LAT
    if not (math.isfinite(width_m) and math.isfinite(height_m)) or width_m <= 0 or height_m <= 0:
        raise ValueError("El tamaño de la escena no es válido; no se puede simular.")

    # Dominio dimensionado según el desplazamiento: la huella, más lo que se mueve la mancha hacia cada lado,
    # más un margen para ver cómo se esparce. Así la trayectoria siempre cabe y la masa no sale del dominio.
    o_lat, o_lon = _mask_origin(mask, bounds)
    track0 = lagrangian_track(o_lat, o_lon, forcing, hours)
    de = [(p[1] - o_lon) * m_per_deg_lon for p in track0]   # metros hacia el este respecto al origen
    dn = [(p[0] - o_lat) * M_PER_DEG_LAT for p in track0]   # metros hacia el norte
    spread_x, spread_y = 0.5 * width_m, 0.5 * height_m
    m_east, m_west = spread_x + max(0.0, max(de)), spread_x + max(0.0, -min(de))
    m_north, m_south = spread_y + max(0.0, max(dn)), spread_y + max(0.0, -min(dn))
    total_w, total_h = width_m + m_east + m_west, height_m + m_north + m_south

    # Celdas de al menos CELL_MIN_M (paso de tiempo estable razonable) y como mucho MAX_CELLS por lado.
    cell = max(CELL_MIN_M, max(total_w, total_h) / MAX_CELLS)
    gx = max(8, int(round(width_m / cell)))
    gy = max(8, int(round(height_m / cell)))
    dx = width_m / gx
    dy = height_m / gy
    mx_e, mx_w = int(math.ceil(m_east / dx)), int(math.ceil(m_west / dx))
    my_n, my_s = int(math.ceil(m_north / dy)), int(math.ceil(m_south / dy))
    Gx, Gy = gx + mx_w + mx_e, gy + my_s + my_n
    south_e = south - my_s * dy / M_PER_DEG_LAT
    north_e = north + my_n * dy / M_PER_DEG_LAT
    west_e = west - mx_w * dx / m_per_deg_lon
    east_e = east + mx_e * dx / m_per_deg_lon
    bounds_e = [[south_e, west_e], [north_e, east_e]]

    frac = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize((gx, gy), Image.BOX)
    core = np.asarray(frac, dtype=np.float64)[::-1] / 255.0
    c = np.zeros((Gy, Gx), dtype=np.float64)
    c[my_s:my_s + gy, mx_w:mx_w + gx] = core
    c0_mass = float(c.sum())
    peak0 = float(c.max())

    def centroid_of(arr: np.ndarray):
        total = arr.sum()
        if total <= 0:
            return None
        yy, xx = np.indices(arr.shape)
        cy = float((yy * arr).sum() / total) + 0.5
        cx = float((xx * arr).sum() / total) + 0.5
        return [round(south_e + cy * dy / M_PER_DEG_LAT, 5), round(west_e + cx * dx / m_per_deg_lon, 5)]

    origin = centroid_of(c)

    n_used = int(math.ceil(hours)) + 1
    vmax = max(math.hypot(u, v) for u, v in vel[:n_used]) or 0.0
    limits = [600.0, 0.2 * min(dx, dy) ** 2 / DIFFUSIVITY_M2S]
    if vmax > 0:
        limits.append(0.5 * min(dx, dy) / vmax)
    dt = min(limits)
    total_s = hours * 3600.0
    n_steps = int(math.ceil(total_s / dt))
    if n_steps > MAX_STEPS:  # nunca se estira el paso por encima del estable: la simulación se volvería inestable
        raise ValueError("La duración pedida exige demasiados pasos de cálculo para el tamaño de esta escena; "
                         "pruebe con menos horas.")
    dt = total_s / n_steps

    times = {0.0, float(hours)} | {hours * k / N_EVEN_FRAMES for k in range(1, N_EVEN_FRAMES)}
    times |= {float(m) for m in MILESTONES_H if m <= hours}
    frame_steps = sorted({int(round(t * 3600.0 / dt)) for t in times})

    frames: list[dict] = []
    zone = np.zeros((Gy, Gx), dtype=bool)

    def snapshot(step: int) -> None:
        if not np.isfinite(c).all():
            raise ValueError("La simulación se volvió numéricamente inestable; pruebe con otra duración o viento.")
        peak = float(c.max())
        visible = c > VISIBLE_REL * peak if peak > 0 else np.zeros_like(c, dtype=bool)
        zone[:] = zone | (c > AFFECTED_REL * peak0)
        frames.append({
            "hour": round(step * dt / 3600.0, 2),
            "png_b64": _render(c, peak0),
            "area_km2": round(float(visible.sum() * dx * dy / 1e6), 3),
            "remaining_mass_pct": round(100.0 * float(c.sum()) / c0_mass, 1) if c0_mass > 0 else 0.0,
            "centroid": centroid_of(c),
        })

    snapshot(0)
    for s in range(1, n_steps + 1):
        u, v = vel[min(int(((s - 0.5) * dt) / 3600.0), len(vel) - 1)]
        if u > 0:
            c = c - u * dt / dx * (c - _shift(c, 1, 1))
        elif u < 0:
            c = c - u * dt / dx * (_shift(c, 1, -1) - c)
        if v > 0:
            c = c - v * dt / dy * (c - _shift(c, 0, 1))
        elif v < 0:
            c = c - v * dt / dy * (_shift(c, 0, -1) - c)
        pad = np.pad(c, 1, mode="edge")
        lap = ((pad[:-2, 1:-1] + pad[2:, 1:-1] - 2 * c) / dy ** 2
               + (pad[1:-1, :-2] + pad[1:-1, 2:] - 2 * c) / dx ** 2)
        c = np.maximum((c + DIFFUSIVITY_M2S * dt * lap) * math.exp(-DECAY_PER_HOUR * dt / 3600.0), 0.0)
        if s in frame_steps[1:]:
            snapshot(s)

    track = lagrangian_track(origin[0], origin[1], forcing, hours) if origin else []
    end = track[-1][:2] if track else None
    dist_km = haversine_km(origin, end) if origin and end else 0.0
    brg = bearing_deg(origin, end) if dist_km > 0.05 else None

    stages = []
    for h in (0,) + MILESTONES_H:
        if h > hours:
            continue
        fr = min(frames, key=lambda f: abs(f["hour"] - h))
        if abs(fr["hour"] - h) <= 0.06:
            stages.append({"hour": h, "area_km2": fr["area_km2"], "remaining_mass_pct": fr["remaining_mass_pct"]})
    if frames[-1]["hour"] not in [s_["hour"] for s_ in stages]:
        stages.append({"hour": round(frames[-1]["hour"], 2), "area_km2": frames[-1]["area_km2"],
                       "remaining_mass_pct": frames[-1]["remaining_mass_pct"]})

    used = forcing[:n_used]
    winds = [f["wind_ms"] for f in used]
    wind_mean = sum(winds) / len(winds)
    wind_from_mean = circular_mean_deg([f["wind_from"] for f in used])
    speed_kmh = dist_km / hours
    if not math.isfinite(speed_kmh):
        raise ValueError("No se pudo calcular la velocidad de deriva.")
    has_currents = any((f.get("current_ms") or 0) > 0 for f in used)
    result = {
        "disclaimer": "SIMULACIÓN ILUSTRATIVA. Deriva simplificada con supuestos declarados; no es una predicción validada.",
        "mode": mode,
        "params": {"wind_speed_ms": round(wind_mean, 2), "wind_from_deg": round(wind_from_mean, 0), "hours": hours},
        "forcing": {
            "mode": mode,
            "source": forcing_source,
            "wind_mean_ms": round(wind_mean, 2),
            "wind_max_ms": round(max(winds), 2),
            "wind_from_mean_deg": round(wind_from_mean, 0),
            "uses_currents": has_currents,
        },
        "assumptions": {
            "drift_factor_of_wind": WIND_FACTOR,
            "diffusivity_m2_s": DIFFUSIVITY_M2S,
            "decay_per_hour": DECAY_PER_HOUR,
            "grid": f"{Gx}x{Gy}",
            "time_step_s": round(dt, 1),
            "domain": "la huella del análisis más un margen para ver el desplazamiento; el aceite que sale de ese dominio se pierde",
            "physics": ("sin oleaje, mareas ni química del petróleo" if has_currents
                        else "sin corrientes marinas, oleaje ni mareas"),
        },
        "footprint": bounds,
        "bounds": bounds_e,
        "origin": origin,
        "track": track,
        "drift": {
            "bearing_to_deg": round(brg, 1) if brg is not None else None,
            "direction_es": compass_es(brg) if brg is not None else "sin desplazamiento apreciable",
            "speed_kmh": round(speed_kmh, 3),
            "distance_km": round(dist_km, 2),
            "end": [round(end[0], 5), round(end[1], 5)] if end and brg is not None else None,
        },
        "stages": stages,
        "affected_zone": {
            "png_b64": _render_zone(zone),
            "area_km2": round(float(zone.sum() * dx * dy / 1e6), 2),
            "note": "Zona por la que, en este escenario, pasaría una concentración apreciable de la mancha.",
        },
        "narrative": drift_story(brg, dist_km, hours, stages, wind_mean, wind_from_mean, mode, has_currents),
        "frames": frames,
    }
    _assert_finite(result)
    return result
