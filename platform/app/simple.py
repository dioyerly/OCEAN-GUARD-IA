"""Explicaciones simples de la simulación: dirección en español, tabla por horas, costa y nivel de gravedad.

Todo se deriva de la simulación y del contexto que ya calculan simulation.py y context.py.
"""
from __future__ import annotations

from .geo import compass_es
from .simulation import haversine_km

STAGE_HOURS = (6, 12, 24, 48)

# Reglas de gravedad (baja / media / alta). Cada condición suma puntos; el total decide el nivel.
SEVERITY_RULES = {
    "coast_close_hours": 24,     # llega a la costa en <= 24 h: suma 2 puntos; llegando después: suma 1
    "area_growth_much_pct": 50,  # el área crece >= 50 % entre el inicio y el final de lo simulado: "mucho" (suma 1)
    "wind_strong_ms": 10.0,      # viento medio >= 10 m/s (~36 km/h): "fuerte" (suma 1)
    "high_from_points": 3,       # 3 o más puntos = alta
    "medium_from_points": 1,     # 1 o 2 puntos = media; 0 = baja
}
SEVERITY_LABEL = {"low": "baja", "medium": "media", "high": "alta"}


def direction_text(bearing_deg: float | None) -> str:
    """'hacia el noreste' (8 direcciones) o 'sin desplazamiento apreciable'."""
    if bearing_deg is None:
        return "sin desplazamiento apreciable"
    return f"hacia el {compass_es(bearing_deg)}"


def _num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def severity(reaches_coast: bool, coast_hours: float | None, growth_pct: float | None,
             wind_mean_ms: float, coast_known: bool = True) -> dict:
    """Nivel de gravedad con una línea de «por qué»."""
    r = SEVERITY_RULES
    points, why = 0, []
    if reaches_coast and coast_hours is not None:
        points += 2 if coast_hours <= r["coast_close_hours"] else 1
        why.append(f"llega a la costa en {_num(coast_hours, 0)} horas")
    elif not coast_known:
        why.append("no se pudo evaluar la costa (sin ubicación)")
    else:
        why.append("no llega a la costa en el tiempo simulado")
    if growth_pct is not None and growth_pct >= r["area_growth_much_pct"]:
        points += 1
        why.append("el área crece mucho")
    if wind_mean_ms >= r["wind_strong_ms"]:
        points += 1
        why.append("el viento es fuerte")
    else:
        why.append("el viento no es fuerte")
    level = "high" if points >= r["high_from_points"] else "medium" if points >= r["medium_from_points"] else "low"
    sentence = f"Gravedad {SEVERITY_LABEL[level]} porque " + ", ".join(why[:-1]) + " y " + why[-1] + "."
    return {"level": level, "label": SEVERITY_LABEL[level], "points": points, "why": sentence}


def explain_simulation(sim: dict, context: dict, located: bool) -> dict:
    """Bloque `simple` de la respuesta de /simulate."""
    track = sim.get("track") or []
    origin = sim.get("origin")
    drift = sim["drift"]
    hours = sim["params"]["hours"]
    stage_area = {s["hour"]: s["area_km2"] for s in sim.get("stages", [])}
    base_area = stage_area.get(0)
    rows = []
    growth_by_hour: dict[int, float | None] = {}
    for h in STAGE_HOURS:
        if h > hours + 1e-6 or not track or origin is None:
            continue
        pt = min(track, key=lambda t: abs(t[2] - h))
        area = stage_area.get(h)
        growth = round(100.0 * (area - base_area) / base_area, 0) if area is not None and base_area else None
        growth_by_hour[h] = growth  # solo para la regla de gravedad; no se muestra al usuario
        rows.append({"hour": h, "displacement_km": round(haversine_km(origin, pt[:2]), 1)})
    coast = None
    reaches, coast_hours = False, None
    ap = (context or {}).get("coast")
    if ap and located:
        reaches = bool(ap["reaches_coast"])
        coast_hours = ap["closest_hour"] if reaches else None
        coast = {"distance_km": ap["start"]["distance_km"], "direction_es": ap["start"]["direction_es"],
                 "point": ap["start"]["point"], "reaches": reaches, "hours": coast_hours}
    last = rows[-1] if rows else None
    sev = severity(reaches, coast_hours, growth_by_hour.get(last["hour"]) if last else None,
                   sim["forcing"]["wind_mean_ms"], coast_known=coast is not None)
    # Zona que podría verse afectada (aproximada): desde la mancha visible al final hasta la zona barrida.
    later = [v for h, v in stage_area.items() if h > 0 and v is not None]
    swept = (sim.get("affected_zone") or {}).get("area_km2")
    cands = [v for v in later + ([swept] if swept is not None else []) if v is not None]
    area_range = [round(min(cands), 1), round(max(cands), 1)] if cands else None
    return {
        "direction_text": direction_text(drift.get("bearing_to_deg")),
        "speed_kmh": drift["speed_kmh"],
        "speed_text": f"unos {_num(drift['speed_kmh'], 1)} km/h",
        "stages": rows,
        "coast": coast,
        "severity": sev,
        "affected_range_km2": area_range,
        "horizon_hours": hours,
    }
