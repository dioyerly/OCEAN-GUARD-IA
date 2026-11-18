"""Contexto de un evento: clima real, costa, trayectoria estimada, nivel de atención y qué hacer.

Todo se calcula con datos reales (clima de Open-Meteo, costa de Natural Earth) y reglas
declaradas. Las recomendaciones son orientación general, no un plan de contingencia oficial.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from . import coast, narrative, weather
from . import simulation as S
from .geo import compass_es

# Umbrales orientativos de trabajo en el agua (reglas generales citadas en guías de respuesta).
WIND_OP_MAX = 8.0     # m/s (~15 nudos): por encima, barreras y recuperadores pierden eficacia
WAVE_OP_MAX = 1.0     # m
WIND_STRONG = 12.0    # m/s: dispersa y desplaza la mancha con rapidez
WAVE_HIGH = 2.0       # m
CURRENT_NOTABLE = 0.35  # m/s (~0,7 nudos)
GENERAL_NOTE = ("Orientación general basada en prácticas comunes de respuesta a derrames. No sustituye el plan de "
                "contingencia ni las instrucciones de la autoridad competente.")


def origin_from_mask(mask: np.ndarray, bounds: list[list[float]]) -> list[float]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    (south, west), (north, east) = bounds
    lat = north - (float(ys.mean()) + 0.5) / h * (north - south)
    lon = west + (float(xs.mean()) + 0.5) / w * (east - west)
    return [round(lat, 5), round(lon, 5)]


def reference_time(a: dict) -> datetime | None:
    scene = (a.get("result") or {}).get("scene")
    if scene and scene.get("date"):
        try:
            return datetime.fromisoformat(f"{scene['date']}T{scene.get('time_utc') or '00:00'}:00+00:00")
        except ValueError:
            return None
    return None


def _cdiff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def weather_stats(rows: list[dict]) -> dict:
    winds = [r["wind_ms"] for r in rows]
    n = len(rows)
    third = max(1, n // 3)
    early = S.circular_mean_deg([r["wind_from"] for r in rows[:third]])
    late = S.circular_mean_deg([r["wind_from"] for r in rows[-third:]])
    peak = max(rows, key=lambda r: r["wind_ms"])
    gusts = [r["gust_ms"] for r in rows if r["gust_ms"] is not None]
    waves = [r["wave_m"] for r in rows if r["wave_m"] is not None]
    curs = [r["current_ms"] for r in rows if r["current_ms"] is not None]
    max_drift_ms = max(math.hypot(*S.velocity(r)) for r in rows)
    return {
        "wind_mean_ms": round(sum(winds) / n, 1),
        "wind_max_ms": round(max(winds), 1),
        "wind_max_at": peak["t"],
        "gust_max_ms": round(max(gusts), 1) if gusts else None,
        "wind_from_mean_deg": round(S.circular_mean_deg([r["wind_from"] for r in rows]), 0),
        "wind_from_early_deg": round(early, 0),
        "wind_from_late_deg": round(late, 0),
        "wind_turn_deg": round(_cdiff(early, late), 0),
        "wave_max_m": round(max(waves), 1) if waves else None,
        "current_max_ms": round(max(curs), 2) if curs else None,
        "max_drift_kmh": round(max_drift_ms * 3.6, 2),
    }


def operational_windows(rows: list[dict], min_hours: int = 3) -> list[dict]:
    """Tramos continuos con viento y olas dentro de los umbrales de trabajo."""
    out, run = [], []

    def ok(r):
        return r["wind_ms"] <= WIND_OP_MAX and (r["wave_m"] is None or r["wave_m"] <= WAVE_OP_MAX)

    for r in rows + [None]:
        if r is not None and ok(r):
            run.append(r)
            continue
        if len(run) >= min_hours:
            out.append({"start": run[0]["t"], "end": run[-1]["t"], "hours": len(run)})
        run = []
    out.sort(key=lambda w: (-w["hours"], w["start"]))
    return out[:3]


def _hhmm(iso: str) -> str:
    return iso[11:16]


def _day_hhmm(iso: str) -> str:
    return f"{iso[8:10]}/{iso[5:7]} {iso[11:16]}"


def risk_level(a: dict, approach: dict | None, stats: dict | None) -> dict:
    pts, reasons = 0, []
    area = a.get("slick_area_km2")
    if area is not None:
        if area >= 5:
            pts += 2
            reasons.append(f"La zona marcada es extensa (≈ {area:.1f} km²).")
        elif area >= 0.5:
            pts += 1
            reasons.append(f"La zona marcada es de tamaño medio (≈ {area:.1f} km²).")
        else:
            reasons.append(f"La zona marcada es pequeña (≈ {area:.2f} km²).")
    else:
        reasons.append("No se puede estimar el área en km² sin ubicación y escala.")
    if approach:
        d0 = approach["start"]["distance_km"]
        if approach["reaches_coast"]:
            pts += 3
            reasons.append(f"En el escenario estimado la mancha podría alcanzar la costa en unas {approach['closest_hour']:.0f} h.")
        elif approach["approaching"] and approach["closest"]["distance_km"] < 20:
            pts += 2
            reasons.append(f"La deriva la acerca a menos de {approach['closest']['distance_km']:.0f} km de la costa.")
        elif d0 < 30:
            pts += 2
            reasons.append(f"Está muy cerca de la costa (≈ {d0:.0f} km).")
        elif d0 < 100:
            pts += 1
            reasons.append(f"Está relativamente cerca de la costa (≈ {d0:.0f} km).")
        else:
            reasons.append(f"Está lejos de la costa (≈ {d0:.0f} km).")
    if stats:
        if stats["wind_max_ms"] >= WIND_STRONG or (stats["wave_max_m"] or 0) >= WAVE_HIGH:
            pts += 1
            reasons.append("El viento o el oleaje previstos son fuertes: pueden esparcir la mancha y dificultar el trabajo.")
        else:
            reasons.append("El viento y el oleaje son moderados.")
    label = "Baja" if pts <= 1 else "Moderada" if pts <= 3 else "Alta"
    return {"level": {"Baja": "low", "Moderada": "medium", "Alta": "high"}[label], "label": label, "points": pts, "reasons": reasons}


def build_recommendations(a: dict, wx: dict | None, stats: dict | None, windows: list[dict],
                          approach: dict | None, located: bool = True) -> list[dict]:
    recs = [{
        "title": "Confirmar antes de actuar",
        "text": "La detección es una sugerencia del modelo. Confírmela con otra fuente: una imagen óptica o de otro radar, un sobrevuelo o un reporte en el lugar.",
        "why": "Pueden confundirse con petróleo otras manchas oscuras del mar (zonas de calma, algas).",
    }, {
        "title": "Avisar a la autoridad competente",
        "text": "Si se confirma, informe a la autoridad marítima o ambiental de la zona, con las coordenadas y la imagen.",
        "why": "La contención y la limpieza deben coordinarse con quien tiene la responsabilidad y los medios.",
    }]
    if not located:
        recs.append({"title": "Indicar dónde está para afinar el análisis",
                     "text": "Sin ubicación no se puede consultar el clima real ni la distancia a la costa. Indique el lugar (por nombre o coordenadas) y el análisis se recalculará con datos reales.",
                     "why": "El rumbo y la velocidad de la deriva dependen del viento y las corrientes del lugar."})
    if stats and wx:
        turn = stats["wind_turn_deg"]
        why = (f"El viento cambiará de dirección unos {turn:.0f}° (de {compass_es(stats['wind_from_early_deg'])} a {compass_es(stats['wind_from_late_deg'])}), "
               "lo que cambiaría el rumbo de la deriva." if turn >= 60 else "La mancha cambia con el viento y las corrientes.")
        recs.append({"title": "Seguir la evolución cada 6–12 horas",
                     "text": "Vuelva a observar la zona (nueva imagen satelital o sobrevuelo) para ver si la mancha se mueve, se extiende o se rompe.",
                     "why": why})
    else:
        recs.append({"title": "Seguir la evolución",
                     "text": "Vuelva a observar la zona en las próximas horas con una nueva imagen o un sobrevuelo.",
                     "why": "Sin datos de clima no se puede anticipar el rumbo de la deriva."})

    if stats and wx:
        adverse = stats["wind_max_ms"] >= WIND_STRONG or (stats["wave_max_m"] or 0) >= WAVE_HIGH
        if windows:
            w = windows[0]
            lim = f"viento ≤ {WIND_OP_MAX:.0f} m/s" + (f" y olas ≤ {WAVE_OP_MAX:.0f} m" if wx["has_waves"] else "")
            recs.append({"title": "Aprovechar la mejor ventana para contener",
                         "text": f"La ventana más favorable para trabajar en el agua con barreras flotantes y recuperadores es de {_day_hhmm(w['start'])} a {_day_hhmm(w['end'])} UTC ({w['hours']} h seguidas con {lim}).",
                         "why": "Con viento fuerte u olas altas las barreras pierden eficacia y la mancha se esparce más rápido."})
        else:
            recs.append({"title": "Priorizar monitoreo y protección de costa",
                         "text": "En las próximas horas no hay tramos largos con condiciones favorables: contener en el agua sería poco eficaz.",
                         "why": f"El viento llega a {stats['wind_max_ms']:.0f} m/s" + (f" y las olas a {stats['wave_max_m']:.1f} m" if stats["wave_max_m"] else "") + "."})
        if adverse:
            recs.append({"title": "Esperar un empeoramiento",
                         "text": f"El viento puede llegar a {stats['wind_max_ms']:.0f} m/s (ráfagas de hasta {stats['gust_max_ms'] or stats['wind_max_ms']:.0f} m/s): la deriva podría acelerarse hasta unos {stats['max_drift_kmh']:.1f} km/h.",
                         "why": "El viento fuerte empuja y esparce la mancha, y hace más difícil llegar a ella."})
    else:
        recs.append({"title": "Contención en el agua",
                     "text": "Las barreras flotantes y los recuperadores funcionan mejor con mar en calma. Evalúe el clima del lugar antes de enviar equipos.",
                     "why": "No se pudo consultar el clima desde la plataforma."})

    if approach:
        c = approach["closest"]
        if approach["reaches_coast"]:
            recs.append({"title": "Prealertar y proteger la costa",
                         "text": f"En este escenario la mancha podría alcanzar la costa hacia el {c['direction_es']} en unas {approach['closest_hour']:.0f} h. Avise a las zonas costeras cercanas y prepare barreras de protección en los sitios sensibles.",
                         "why": "Proteger la costa antes de la llegada es más eficaz que limpiar después."})
        elif approach["approaching"] and c["distance_km"] < 40:
            recs.append({"title": "Vigilar la costa más cercana",
                         "text": f"La deriva acerca la mancha a unos {c['distance_km']:.0f} km de la costa hacia el {c['direction_es']}. Manténgase atento a cambios del viento.",
                         "why": "Un cambio de viento podría llevarla a la orilla."})
        else:
            recs.append({"title": "Costa fuera de alcance en este escenario",
                         "text": f"La costa más cercana está a unos {approach['start']['distance_km']:.0f} km hacia el {approach['start']['direction_es']} y, con las condiciones usadas, la mancha no llegaría a ella. Siga vigilando.",
                         "why": "El escenario depende del clima; si cambia, conviene recalcular."})
    recs.append({
        "title": "Seguridad y decisiones técnicas",
        "text": "No se acerque sin equipo y personal capacitados: los vapores del petróleo pueden ser tóxicos o inflamables. El uso de dispersantes u otras técnicas lo decide únicamente la autoridad competente.",
        "why": "Una mala intervención puede aumentar el daño.",
    })
    for i, r in enumerate(recs, 1):
        r["n"] = i
    return recs


def manual_conditions(speed_ms: float, wind_from: float, hours: float) -> dict:
    """Condiciones sintéticas (viento constante indicado por el usuario) con el mismo formato que el clima real."""
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    from datetime import timedelta
    rows = [{
        "t": (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00Z"), "wind_ms": speed_ms, "wind_from": wind_from,
        "gust_ms": None, "wave_m": None, "wave_from": None, "current_ms": None, "current_to": None,
    } for i in range(int(math.ceil(hours)) + 1)]
    return {"available": True, "mode": "manual", "source": "Viento constante indicado por el usuario (no es clima real)",
            "start_utc": rows[0]["t"], "hours": len(rows), "hourly": rows, "has_waves": False, "has_currents": False,
            "notes": ["Condiciones indicadas manualmente: no se consultó el clima real."]}


def build(a: dict, mask: np.ndarray, hours: int = 48, wx: dict | None = None, located: bool = True) -> dict:
    """Contexto de un análisis. Con ubicación consulta clima real y costa; sin ubicación solo usa `wx` manual."""
    bounds = a["bounds"]
    origin = origin_from_mask(mask, bounds)
    if wx is None and not located:
        raise ValueError("Sin ubicación se necesitan condiciones manuales.")
    if wx is None:
        wx = weather.fetch_conditions(origin[0], origin[1], reference_time(a), hours)
    rows = wx.get("hourly") if wx.get("available") else None
    stats = weather_stats(rows) if rows else None
    windows = operational_windows(rows) if rows else []

    track = S.lagrangian_track(origin[0], origin[1], rows, len(rows)) if rows else []
    approach = None
    if located:
        approach = coast.approach(track) if track else None
        if approach is None:
            start = coast.nearest_coast(origin[0], origin[1])
            approach = {"start": start, "closest": start, "closest_hour": 0, "approaching": False, "reaches_coast": False,
                        "end_distance_km": None} if start else None

    risk = risk_level(a, approach, stats)
    if not located:
        risk["reasons"].append("Sin ubicación: no se puede evaluar la distancia a la costa ni el clima real.")
    recs = build_recommendations(a, wx if rows else None, stats, windows, approach, located)

    impact = []
    if not located:
        impact.append("No se conoce la ubicación: no se evaluó la costa ni se consultó el clima real. Lo que sigue es un escenario con el viento indicado, mostrado sobre la propia imagen.")
    if approach:
        s0 = approach["start"]
        impact.append(f"La costa más cercana está a unos {s0['distance_km']:.0f} km hacia el {s0['direction_es']}.")
        if track and approach["reaches_coast"]:
            impact.append(f"En este escenario, el centro de la mancha podría llegar a la costa en unas {approach['closest_hour']:.0f} horas.")
        elif track and approach["approaching"]:
            impact.append(f"El desplazamiento la acerca a unos {approach['closest']['distance_km']:.0f} km de la costa (a las {approach['closest_hour']:.0f} h), sin llegar a tocarla.")
        elif track:
            impact.append("En este escenario no se acerca a la costa.")
    if stats:
        txt = f"Viento medio de {stats['wind_mean_ms']:.0f} m/s (máximo {stats['wind_max_ms']:.0f} m/s)"
        if stats["wave_max_m"] is not None:
            txt += f", olas de hasta {stats['wave_max_m']:.1f} m"
        if stats["current_max_ms"]:
            txt += f", corrientes de hasta {stats['current_max_ms']:.2f} m/s"
        impact.append(txt + ".")
        impact.append(f"Con ese viento y esas corrientes, el centro de la mancha se movería como máximo unos {stats['max_drift_kmh']:.1f} km/h.")
        if stats["wind_turn_deg"] >= 60:
            impact.append(f"El viento gira unos {stats['wind_turn_deg']:.0f}° durante el periodo: la mancha cambiaría de rumbo.")

    return {
        "origin": origin,
        "weather": wx,
        "stats": stats,
        "windows": windows,
        "track": track,
        "coast": approach,
        "risk": risk,
        "impact": impact,
        "recommendations": recs,
        "general_note": GENERAL_NOTE,
    }
