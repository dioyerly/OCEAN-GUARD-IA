"""Clima y mar reales desde Open-Meteo (API pública, sin clave).

- Imágenes recientes: pronóstico de viento, olas y corrientes superficiales.
- Imágenes antiguas (p. ej. los ejemplos de 2020): viento histórico real (reanálisis) de esa fecha.
  Las olas y corrientes históricas no están disponibles para esas fechas y se informa así.

Solo se envían latitud, longitud y fechas. Cada fallo se registra con su código HTTP y el mensaje del
servicio (log `oceanguard.weather` y `recent_errors()`), se reintenta solo lo que merece reintento
(límite de consultas, errores del servidor, cortes de red) y, si no hay respuesta, la plataforma sigue
funcionando con viento manual. Nunca se inventan datos meteorológicos.
"""
from __future__ import annotations

import json
import logging
import math
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone

log = logging.getLogger("oceanguard.weather")

TIMEOUT_S = 25
ATTEMPTS = 3
MAX_RETRY_WAIT_S = 10
CACHE_TTL_S = 1800
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

HISTORICAL_AFTER_DAYS = 6       # más antigua que esto -> archivo histórico (tiene unos días de retraso)
ARCHIVE_MIN_DATE = datetime(1940, 1, 1, tzinfo=timezone.utc)
FORECAST_MAX_DAYS = 16          # el pronóstico llega hasta ~16 días
FORECAST_MAX_PAST_DAYS = 92

_cache: dict[tuple, tuple[float, dict]] = {}
_recent_errors: deque = deque(maxlen=20)


class WeatherError(Exception):
    """Fallo de la consulta meteorológica, con tipo y código HTTP (si lo hubo)."""

    def __init__(self, kind: str, message: str, code: int | None = None, service: str | None = None):
        super().__init__(message)
        self.kind, self.message, self.code, self.service = kind, message, code, service


def recent_errors() -> list[dict]:
    """Últimos fallos de consulta (para diagnóstico): servicio, código HTTP, mensaje y parámetros."""
    return list(_recent_errors)


def _record(service: str, code: int | None, reason: str, params: dict) -> None:
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "service": service, "http": code,
             "reason": reason[:300],
             "params": {k: v for k, v in params.items() if k in ("latitude", "longitude", "start_date", "end_date",
                                                                  "past_days", "forecast_days", "hourly", "name")}}
    _recent_errors.append(entry)
    log.warning("Open-Meteo %s falló: HTTP %s · %s · %s", service, code, reason[:200], entry["params"])


def _server_reason(body: str) -> str:
    """Open-Meteo responde {"error": true, "reason": "..."}; si no, devuelve el texto tal cual."""
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("reason"):
            return str(data["reason"])
    except ValueError:
        pass
    return body.strip()[:200] or "sin detalle"


def _retry_wait(err: urllib.error.HTTPError, attempt: int) -> float:
    try:
        wait = float(err.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        wait = float(2 ** attempt)
    return min(max(wait, 1.0), MAX_RETRY_WAIT_S)


def _get(url: str, params: dict, service: str) -> dict:
    key = (url, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            with urllib.request.urlopen(full, timeout=TIMEOUT_S) as r:
                data = json.load(r)
            _cache[key] = (time.time(), data)
            return data
        except urllib.error.HTTPError as e:
            try:
                reason = _server_reason(e.read().decode("utf-8", "replace"))
            except Exception:
                reason = str(e.reason)
            _record(service, e.code, reason, params)
            if e.code == 429 or e.code >= 500:
                if not last:
                    time.sleep(_retry_wait(e, attempt))
                    continue
                if e.code == 429:
                    raise WeatherError("rate_limit", "El servicio meteorológico limitó las consultas (HTTP 429). "
                                       "Espere un minuto y vuelva a intentarlo, o use el modo manual.", e.code, service)
                raise WeatherError("server", f"El servicio meteorológico tuvo un error (HTTP {e.code}). "
                                   "Vuelva a intentarlo en unos minutos o use el modo manual.", e.code, service)
            raise WeatherError("bad_request", f"El servicio meteorológico rechazó la consulta (HTTP {e.code}): {reason}. "
                               "Use el modo manual.", e.code, service)
        except (socket.timeout, TimeoutError):
            _record(service, None, "tiempo de espera agotado", params)
            if not last:
                time.sleep(1.0 + attempt)
                continue
            raise WeatherError("timeout", "El servicio meteorológico no respondió a tiempo. Vuelva a intentarlo "
                               "o use el modo manual.", None, service)
        except urllib.error.URLError as e:
            _record(service, None, f"sin conexión: {e.reason}", params)
            if not last:
                time.sleep(1.0 + attempt)
                continue
            raise WeatherError("network", "No hay conexión con el servicio meteorológico. Revise su conexión a "
                               "internet o use el modo manual.", None, service)
        except ValueError as e:  # JSON inválido
            _record(service, None, f"respuesta no válida: {e}", params)
            raise WeatherError("bad_response", "El servicio meteorológico devolvió una respuesta no válida. "
                               "Use el modo manual.", None, service)
    raise WeatherError("network", "No se pudo consultar el servicio meteorológico.", None, service)  # pragma: no cover


def _hourly_map(data: dict, fields: list[str]) -> dict[str, dict]:
    h = data.get("hourly") or {}
    times = h.get("time") or []
    out: dict[str, dict] = {}
    for i, t in enumerate(times):
        out[t] = {f: (h.get(f) or [None] * len(times))[i] for f in fields}
    return out


def _num(x) -> float | None:
    """float finito o None (los valores ausentes o indefinidos no se convierten en cero)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _unavailable(kind: str, reason: str, code: int | None = None) -> dict:
    return {"available": False, "reason": reason, "error": {"kind": kind, "http": code}}


def fetch_conditions(lat: float, lon: float, ref: datetime | None, hours: int = 48) -> dict:
    """Condiciones horarias desde `ref` (UTC; None = ahora) durante `hours` horas.

    Distingue tres casos: pronóstico (fecha actual o muy reciente), histórico (más de 6 días atrás) y
    fecha no disponible (futuro lejano o anterior a 1940). Sin coordenadas válidas no consulta nada.
    """
    lat_f, lon_f = _num(lat), _num(lon)
    if lat_f is None or lon_f is None or abs(lat_f) > 90 or abs(lon_f) > 180:
        return _unavailable("invalid_input", "Las coordenadas no son válidas: no se consulta el clima real.")
    if not (1 <= hours <= 24 * 14):
        return _unavailable("invalid_input", "La duración pedida no es válida para consultar el clima.")

    now = datetime.now(timezone.utc)
    ref = (ref or now).astimezone(timezone.utc)
    start = ref.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=hours)
    age_days = (now - start).total_seconds() / 86400.0
    lat_r, lon_r = round(lat_f, 2), round(lon_f, 2)
    wind_fields = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
    marine_fields = ["wave_height", "wave_direction", "ocean_current_velocity", "ocean_current_direction"]
    notes: list[str] = []
    marine_map: dict = {}

    if start < ARCHIVE_MIN_DATE:
        return _unavailable("unavailable_date", "No hay datos meteorológicos anteriores a 1940.")
    if age_days < -(FORECAST_MAX_DAYS - 1):
        return _unavailable("unavailable_date", f"La fecha está a más de {FORECAST_MAX_DAYS - 1} días en el futuro: "
                            "el pronóstico no llega tan lejos.")
    historical = age_days > HISTORICAL_AFTER_DAYS
    try:
        if historical:
            last_available = (now - timedelta(days=HISTORICAL_AFTER_DAYS - 1)).replace(minute=0, second=0, microsecond=0)
            end_clamped = min(end, last_available)
            wind = _get(ARCHIVE_URL, {
                "latitude": lat_r, "longitude": lon_r, "start_date": start.date().isoformat(),
                "end_date": end_clamped.date().isoformat(), "hourly": ",".join(wind_fields[:2]),
                "wind_speed_unit": "ms", "timezone": "UTC",
            }, "archivo histórico")
            wind_map = _hourly_map(wind, wind_fields[:2])
            source = "Open-Meteo · datos históricos reales (reanálisis) de esa fecha"
            mode = "historical"
            notes.append("Es el viento real que hubo en esa fecha, no un pronóstico. Olas y corrientes históricas no están disponibles.")
        else:
            past_days = int(min(FORECAST_MAX_PAST_DAYS, max(1, math.ceil(max(age_days, 0)) + 1)))
            forecast_days = int(min(FORECAST_MAX_DAYS, max(2, math.ceil((end - now).total_seconds() / 86400.0) + 1)))
            fparams = {"latitude": lat_r, "longitude": lon_r, "hourly": ",".join(wind_fields),
                       "forecast_days": forecast_days, "past_days": past_days,
                       "wind_speed_unit": "ms", "timezone": "UTC"}
            wind = _get(FORECAST_URL, fparams, "pronóstico")
            wind_map = _hourly_map(wind, wind_fields)
            source = "Open-Meteo · pronóstico meteorológico y marino"
            mode = "forecast"
            try:
                marine = _get(MARINE_URL, {"latitude": lat_r, "longitude": lon_r, "hourly": ",".join(marine_fields),
                                           "forecast_days": forecast_days, "past_days": past_days, "timezone": "UTC"},
                              "marino")
                marine_map = _hourly_map(marine, marine_fields)
            except WeatherError:
                notes.append("No se pudieron obtener olas y corrientes; se usa solo el viento.")
    except WeatherError as exc:
        return _unavailable(exc.kind, exc.message, exc.code)

    rows = []
    t = start
    while t < end:
        key = t.strftime("%Y-%m-%dT%H:00")
        w = wind_map.get(key)
        ws, wd = _num((w or {}).get("wind_speed_10m")), _num((w or {}).get("wind_direction_10m"))
        if ws is not None and wd is not None and 0 <= ws <= 100:
            m = marine_map.get(key, {})
            cur_kmh = _num(m.get("ocean_current_velocity"))
            rows.append({
                "t": key + "Z",
                "wind_ms": ws,
                "wind_from": wd % 360.0,
                "gust_ms": _num((w or {}).get("wind_gusts_10m")),
                "wave_m": _num(m.get("wave_height")),
                "wave_from": _num(m.get("wave_direction")),
                "current_ms": round(cur_kmh / 3.6, 3) if cur_kmh is not None else None,
                "current_to": _num(m.get("ocean_current_direction")),
            })
        t += timedelta(hours=1)
    if len(rows) < 3:
        return _unavailable("no_data", "El servicio meteorológico no devolvió datos suficientes para esa fecha y lugar.")
    if len(rows) < hours:
        notes.append(f"El servicio solo devolvió {len(rows)} de las {hours} horas pedidas.")
    return {
        "available": True,
        "mode": mode,
        "source": source,
        "start_utc": rows[0]["t"],
        "hours": len(rows),
        "hourly": rows,
        "has_waves": any(r["wave_m"] is not None for r in rows),
        "has_currents": any(r["current_ms"] is not None for r in rows),
        "notes": notes,
    }


def geocode(query: str, count: int = 6) -> list[dict]:
    """Busca lugares por nombre (Open-Meteo Geocoding). Solo se envía el texto buscado."""
    data = _get(GEOCODE_URL, {"name": query, "count": count, "language": "es", "format": "json"}, "geocodificación")
    out = []
    for r in data.get("results") or []:
        la, lo = _num(r.get("latitude")), _num(r.get("longitude"))
        if la is None or lo is None:
            continue
        out.append({"name": r.get("name"), "admin1": r.get("admin1"), "country": r.get("country"), "lat": la, "lon": lo})
    return out
