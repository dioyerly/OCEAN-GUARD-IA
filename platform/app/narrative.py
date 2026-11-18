"""Explicaciones en español sencillo para resultados y simulaciones.

Nada aquí inventa datos: todo se deriva de la máscara predicha, de la georreferencia real
(si existe) y de las mediciones de validación declaradas en config.
"""
from __future__ import annotations

from . import config
from .geo import compass_es


# Regla del semáforo (a partir de lo que el modelo ya calcula: probabilidad media y tamaño de la zona mayor).
#   verde    = el modelo no marcó ninguna zona
#   amarillo = hay zona marcada, pero poco segura o pequeña  -> conviene revisarla
#   rojo     = probabilidad media >= red_mean_prob y zona mayor >= red_min_component_px
TRAFFIC_RULES = {"red_mean_prob": 0.97, "red_min_component_px": 1000}
TRAFFIC_LABEL = {"green": "No se detecta derrame",
                 "yellow": "Posible derrame: conviene revisarlo",
                 "red": "Derrame probable"}


def traffic_light(result: dict) -> str:
    """'green' | 'yellow' | 'red' según TRAFFIC_RULES."""
    comps = result.get("components") or []
    largest = comps[0]["area_px"] if comps else 0
    if result.get("slick_px", 0) == 0 or not comps:
        return "green"
    if ((result.get("mean_prob_slick") or 0.0) >= TRAFFIC_RULES["red_mean_prob"]
            and largest >= TRAFFIC_RULES["red_min_component_px"]):
        return "red"
    return "yellow"


def _num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def location_state(location_kind: str | None) -> str:
    kind = (location_kind or "").lower()
    if kind.startswith("georreferencia"):
        return "georef"
    if kind.startswith("manual"):
        return "manual"
    return "none"


def coords_text(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return ""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "O"
    return f"{_num(abs(lat), 4)}° {ns}, {_num(abs(lon), 4)}° {ew}"


def surface_phrase(pct: float) -> str:
    if pct < 1:
        return "menos del 1 % de la superficie analizada"
    return f"aproximadamente el {_num(pct, 1)} % de la superficie analizada"


def confidence(result: dict) -> dict:
    """Indicador orientativo: seguridad del modelo en lo marcado y tamaño de la zona principal."""
    comps = result.get("components") or []
    largest = comps[0]["area_px"] if comps else 0
    mean_prob = result.get("mean_prob_slick") or 0.0
    if result.get("slick_px", 0) == 0:
        return {"label": "—", "level": "none",
                "text": "No hay zonas marcadas, por lo que no se calcula un nivel de confianza."}
    if mean_prob >= 0.97 and largest >= 1000:
        level, label = "high", "Alta"
    elif mean_prob >= 0.95 and largest >= 300:
        level, label = "medium", "Media"
    else:
        level, label = "low", "Baja"
    rel = config.VAL_RELIABILITY
    text = (
        f"Indicador orientativo basado en lo seguro que está el modelo y en el tamaño de la zona principal. "
        f"En pruebas con imágenes del dataset, de cada 100 puntos que el modelo marcó como derrame, "
        f"unos {round(rel['precision'] * 100)} lo eran; y encontró unos {round(rel['recall'] * 100)} de cada 100 "
        f"puntos de derrame reales. {rel['note']}"
    )
    return {"label": label, "level": level, "text": text}


def loc_text_short(state: str, lat, lon, a: dict) -> str:
    """Dónde está, en pocas palabras."""
    if state == "none" or lat is None or lon is None:
        return ""
    region = config.DATASET_REGION if (a.get("source") or "").startswith("ejemplo") else None
    return (f"en el {region.split(' (')[0]} " if region else "en ") + f"({coords_text(lat, lon)})"


def describe(a: dict) -> dict:
    """Resumen comprensible de un análisis: titular, explicación, confianza y ubicación."""
    result = a.get("result") or {}
    n = a.get("n_components") or 0
    pct = a.get("slick_pct") or 0.0
    area = a.get("slick_area_km2")
    state = location_state(a.get("location_kind"))
    scene = result.get("scene")
    conf = confidence(result)
    warnings = result.get("warnings") or []
    if any(w["code"] == "color" for w in warnings):
        conf = {"label": "No evaluable", "level": "none",
                "text": "La imagen no se parece a los datos de entrenamiento (radar en escala de grises); no se puede estimar la confianza."}

    if pct <= 0 or n == 0:
        level = "none"
        headline = "No se identificaron zonas compatibles con un derrame"
        text = ("El modelo no marcó ninguna zona de esta imagen como posible derrame. Esto no garantiza que "
                "no exista uno: el modelo puede no reconocer algunos casos. Se recomienda revisar la imagen.")
    else:
        level = {"high": "high", "medium": "medium", "low": "low"}[conf["level"]]
        zonas = "una zona" if n == 1 else f"{n} zonas"
        headline = "Posible mancha compatible con un derrame"
        area_txt = (f" (unos {_num(area, 1)} km²{', escala supuesta' if result.get('scale_assumed') else ''})"
                    if area is not None else "")
        text = (
            f"El modelo identificó {zonas} compatible{'s' if n != 1 else ''} con un posible derrame. "
            f"Ocupa{'n' if n != 1 else ''} {surface_phrase(pct)}{area_txt}. "
            "Esto no confirma que se trate de petróleo: pueden existir manchas parecidas de otro origen "
            "(por ejemplo, zonas de calma del mar). Se recomienda revisar visualmente el resultado."
        )

    lat, lon = a.get("center_lat"), a.get("center_lon")
    if state == "georef" and "GPS de la foto" in (a.get("location_kind") or ""):
        loc_text = (f"Ubicación leída del GPS guardado en la foto: {coords_text(lat, lon)}. "
                    "La escala de la imagen es aproximada, así que el tamaño de la zona también lo es.")
        region = None
    elif state == "georef":
        region = config.DATASET_REGION if (a.get("source") or "").startswith("ejemplo") else None
        loc_text = ("Ubicación tomada de las coordenadas reales guardadas en el archivo"
                    + (f": {coords_text(lat, lon)}" if lat is not None else "") + ".")
        if region:
            loc_text += f" Zona: {region}."
    elif state == "manual":
        loc_text = (f"Ubicación aproximada indicada por usted ({coords_text(lat, lon)}). "
                    "No está verificada y el tamaño real depende de la resolución supuesta.")
    else:
        loc_text = ("Escena satelital sin ubicación geográfica confirmada: no se muestra sobre un mapa "
                    "para no sugerir una posición que no se conoce. Puede indicar dónde está para consultar el clima real; "
                    "mientras tanto puede ver un escenario de deriva sobre la propia imagen.")

    acquired = None
    if scene:
        acquired = f"{scene.get('satellite')} · {scene.get('date')} {scene.get('time_utc')} UTC"

    color = traffic_light(result) if (pct > 0 and n > 0) else "green"
    where = loc_text_short(state, lat, lon, a)
    if color == "green":
        phrase = "El modelo no marcó ninguna zona sospechosa en esta imagen. Esto no garantiza que no exista un derrame."
    else:
        zonas_t = "una zona" if n == 1 else f"{n} zonas"
        phrase = (f"El modelo marcó {zonas_t} que {'podría' if color == 'yellow' else 'probablemente'} "
                  f"{'ser un derrame' if n == 1 else 'ser derrames'}"
                  + (f", de unos {_num(area, 1)} km²" if area is not None else "")
                  + (f", {where}" if where else "")
                  + ". Es una sugerencia del sistema, no una confirmación: una persona debe revisarla.")
    traffic = {"color": color, "label": TRAFFIC_LABEL[color], "phrase": phrase,
               "area_km2": area if color != "green" else None, "where": where if color != "green" else None}
    technical = {"confidence": conf, "threshold": config.DEFAULT_THRESHOLD, "min_component_px": config.MIN_COMPONENT_PX,
                 "reliability": config.VAL_RELIABILITY, "traffic_rules": TRAFFIC_RULES,
                 "disclaimer": config.DISCLAIMER_DETECTION, "details": text,
                 "model_probability": result.get("mean_prob_slick")}
    return {
        "traffic": traffic,
        "technical": technical,
        "level": level,
        "headline": headline,
        "text": text,
        "confidence": conf,
        "location": {"state": state, "text": loc_text, "coords": coords_text(lat, lon),
                     "region": config.DATASET_REGION if state == "georef" and (a.get("source") or "").startswith("ejemplo") else None},
        "acquired": acquired,
        "warnings": [w["text"] for w in warnings],
    }


def drift_story(bearing_to: float | None, distance_km: float, hours: float, stages: list[dict],
                wind_mean: float, wind_from: float, mode: str = "manual", has_currents: bool = False) -> str:
    """Narración de la simulación en español sencillo."""
    if mode == "weather":
        cond = (f"Con las condiciones reales de viento (en promedio {_num(wind_mean, 1)} m/s, desde el "
                f"{compass_es(wind_from)}){' y corrientes superficiales' if has_currents else ''}")
    else:
        cond = (f"Con un viento de {_num(wind_mean, 0)} m/s que sopla desde el {compass_es(wind_from)}")
    if bearing_to is None:
        move = "la simulación no muestra un desplazamiento apreciable del centro de la mancha"
    else:
        move = (f"la simulación muestra un desplazamiento aproximado hacia el {compass_es(bearing_to)} "
                f"(unos {_num(distance_km, 1)} km en {_num(hours, 0)} horas)")
    lines = [f"El posible derrame comienza en la ubicación marcada. {cond}, {move}."]
    if len(stages) >= 2:
        first, last = stages[0], stages[-1]
        change = "se extiende" if last["area_km2"] > first["area_km2"] * 1.05 else "se mantiene en un tamaño similar"
        lines.append(
            f"El área marcada pasa de unos {_num(first['area_km2'], 1)} km² al inicio a unos "
            f"{_num(last['area_km2'], 1)} km² a las {_num(last['hour'], 0)} horas: la mancha {change}."
        )
    lines.append("Es un escenario ilustrativo con supuestos sencillos; no incluye oleaje, mareas ni la química real del petróleo.")
    return " ".join(lines)
