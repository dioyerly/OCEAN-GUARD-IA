"""Pruebas de robustez: NaN en JSON, errores de Open-Meteo, encuadre y los 10 casos obligatorios.

El clima se simula con respuestas fijas (sin internet) para que cada caso sea determinista.
"""
import email.message
import io
import json
import math
import os
import sys
import tempfile
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))
os.environ.setdefault("OCEAN_PLATFORM_DATA", tempfile.mkdtemp(prefix="ogtest_"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import config, geo, storage, weather  # noqa: E402
from app.main import app  # noqa: E402
import data as unet_data  # noqa: E402

client = TestClient(app)


# ---------------- utilidades ----------------
def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _val_png(index=0) -> bytes:
    s = unet_data.read_split(config.DATASET_ROOT, "val")[index]
    return _png(np.asarray(Image.open(s.image_path).convert("RGB")))


def _strict_json(text: str):
    """Parsea rechazando NaN/Infinity (no son JSON válido)."""
    def bad(token):
        raise ValueError(f"JSON no válido: contiene {token}")
    return json.loads(text, parse_constant=bad)


def _all_finite(obj, path="$"):
    if isinstance(obj, float):
        assert math.isfinite(obj), f"no finito en {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k != "png_b64":
                _all_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _all_finite(v, f"{path}[{i}]")


def _small_scene_upload(scene_km="3.4", **data):
    """Escena pequeña (457x224 px a ~7 m/px): la que desbordaba la simulación original."""
    img = np.zeros((224, 457), dtype=np.uint8) + 90
    img[80:140, 100:330] = 15  # mancha oscura grande, como en las capturas del usuario
    rgb = np.stack([img] * 3, axis=-1)
    mark = np.zeros((224, 457), dtype=np.uint8)
    mark[80:140, 100:330] = 255
    files = {"file": ("pequena.png", _png(rgb), "image/png"), "mask": ("m.png", _png(mark), "image/png")}
    d = {"scene_km": scene_km}
    d.update(data)
    r = client.post("/api/analyze", files=files, data=d)
    assert r.status_code == 200, r.text
    return r.json()


def _fake_hourly(start: datetime, end: datetime, speed=6.0, wind_from=90.0, marine=False):
    times, n = [], int((end - start).total_seconds() // 3600) + 1
    for i in range(n):
        times.append((start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00"))
    if marine:
        return {"hourly": {"time": times, "wave_height": [1.1] * n, "wave_direction": [80.0] * n,
                           "ocean_current_velocity": [0.7] * n, "ocean_current_direction": [270.0] * n}}
    return {"hourly": {"time": times, "wind_speed_10m": [speed] * n, "wind_direction_10m": [wind_from] * n,
                       "wind_gusts_10m": [speed + 2] * n}}


@pytest.fixture
def fake_weather(monkeypatch):
    """Sustituye la consulta a Open-Meteo por datos fijos y registra los parámetros enviados."""
    calls = []

    def _get(url, params, service):
        calls.append((service, dict(params)))
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if "archive" in url:
            s = datetime.fromisoformat(params["start_date"]).replace(tzinfo=timezone.utc)
            e = datetime.fromisoformat(params["end_date"]).replace(tzinfo=timezone.utc) + timedelta(hours=23)
            return _fake_hourly(s, e)
        start, end = now - timedelta(days=int(params["past_days"])), now + timedelta(days=int(params["forecast_days"]))
        return _fake_hourly(start, end, marine="marine" in url)

    monkeypatch.setattr(weather, "_get", _get)
    weather._cache.clear()
    return calls


# ---------------- ERROR 1: NaN ----------------
def test_small_scene_simulation_is_stable_and_json_strict():
    a = _small_scene_upload()
    assert a["bounds"] is None  # imagen sin ubicación
    for hours in (6, 12, 48, 72):
        r = client.post(f"/api/analyses/{a['id']}/simulate",
                        json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": hours})
        assert r.status_code == 200, r.text
        sim = _strict_json(r.text)  # falla si hay NaN/Infinity
        _all_finite(sim)
        assert sim["assumptions"]["time_step_s"] > 0
        assert sim["frames"][-1]["remaining_mass_pct"] > 0


def test_simulation_rejects_invalid_inputs_with_clear_messages():
    from app import simulation as S
    mask = np.zeros((50, 50), bool)
    mask[10:30, 10:30] = True
    bounds = [[19.4, -92.1], [19.5, -92.0]]
    with pytest.raises(ValueError, match="no son válidas"):
        S.simulate(mask, bounds, None, None, 6, forcing=[{"wind_ms": float("nan"), "wind_from": 90.0}] * 8)
    with pytest.raises(ValueError, match="huella geográfica"):
        S.simulate(mask, [[float("nan"), -92.1], [19.5, -92.0]], 5.0, 90.0, 6)
    with pytest.raises(ValueError, match="nada que simular"):
        S.simulate(np.zeros((50, 50), bool), bounds, 5.0, 90.0, 6)


def test_api_json_never_emits_nan_even_if_stored_data_has_it():
    a = _small_scene_upload()
    result = dict(a["result"])
    result["mean_prob_slick"] = float("nan")  # dato dañado en la base: json.dumps lo guarda como NaN
    with storage.connect() as con:
        con.execute("UPDATE analyses SET result_json=? WHERE id=?", (json.dumps(result), a["id"]))
    r = client.get(f"/api/analyses/{a['id']}")
    assert r.status_code == 200
    got = _strict_json(r.text)
    assert got["result"]["mean_prob_slick"] is None  # null, no cero


def test_invalid_coordinates_are_rejected_not_computed():
    img = {"file": ("p.png", _val_png(0), "image/png")}
    assert client.post("/api/analyze", files=img, data={"lat": "89.9", "lon": "10"}).status_code == 400
    assert "polo" in client.post("/api/analyze", files={"file": ("p.png", _val_png(0), "image/png")},
                                 data={"lat": "89.9", "lon": "10"}).json()["detail"]
    assert client.post("/api/analyze", files={"file": ("p.png", _val_png(0), "image/png")},
                       data={"lat": "95", "lon": "10"}).status_code == 400


# ---------------- ERROR 2: Open-Meteo ----------------
def _http_error(code, body=b"", headers=None):
    hdrs = email.message.Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError("https://x", code, "msg", hdrs, io.BytesIO(body))


class _Resp(io.BytesIO):
    pass


def _scripted(monkeypatch, steps):
    """urlopen que devuelve/lanza lo indicado en `steps`, en orden."""
    seq, calls = list(steps), []

    def fake(url, timeout=None):
        calls.append(url)
        step = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(step, Exception):
            raise step
        return _Resp(json.dumps(step).encode())

    monkeypatch.setattr(weather.urllib.request, "urlopen", fake)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    weather._cache.clear()
    weather._recent_errors.clear()
    return calls


def test_http_429_is_retried_and_then_succeeds(monkeypatch):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    payload = _fake_hourly(now - timedelta(days=2), now + timedelta(days=4))
    calls = _scripted(monkeypatch, [_http_error(429, b'{"error":true,"reason":"Too many requests"}', {"Retry-After": "1"}),
                                    payload])
    wx = weather.fetch_conditions(19.4, -92.0, None, 24)
    assert wx["available"] and wx["mode"] == "forecast" and len(calls) >= 2
    err = weather.recent_errors()[0]
    assert err["http"] == 429 and "Too many" in err["reason"]  # el código y el mensaje real quedan registrados


def test_http_429_persistent_reports_rate_limit(monkeypatch):
    _scripted(monkeypatch, [_http_error(429, b'{"error":true,"reason":"Minutely API request limit exceeded"}')])
    wx = weather.fetch_conditions(19.4, -92.0, None, 24)
    assert wx["available"] is False and wx["error"] == {"kind": "rate_limit", "http": 429}
    assert "429" in wx["reason"] and "modo manual" in wx["reason"]


def test_http_400_shows_the_servers_reason_and_is_not_retried(monkeypatch):
    calls = _scripted(monkeypatch, [_http_error(400, b'{"error":true,"reason":"start_date is out of range"}')])
    ref = datetime.now(timezone.utc) - timedelta(days=40)
    wx = weather.fetch_conditions(19.4, -92.0, ref, 24)
    assert wx["available"] is False and wx["error"]["kind"] == "bad_request" and wx["error"]["http"] == 400
    assert "HTTP 400" in wx["reason"] and "start_date is out of range" in wx["reason"]
    assert len(calls) == 1  # un 400 no se reintenta
    assert weather.recent_errors()[0]["params"]["latitude"] == 19.4


def test_timeout_and_no_connection_are_distinguished(monkeypatch):
    import socket
    _scripted(monkeypatch, [socket.timeout("timed out")])
    assert weather.fetch_conditions(19.4, -92.0, None, 24)["error"]["kind"] == "timeout"
    _scripted(monkeypatch, [urllib.error.URLError("getaddrinfo failed")])
    wx = weather.fetch_conditions(19.4, -92.0, None, 24)
    assert wx["error"]["kind"] == "network" and "conexión" in wx["reason"]


def test_no_request_without_valid_coordinates_or_for_unavailable_dates(monkeypatch):
    calls = _scripted(monkeypatch, [{}])
    for lat, lon in ((float("nan"), -92.0), (95.0, 10.0), (19.4, float("inf")), (None, None)):
        wx = weather.fetch_conditions(lat, lon, None, 24)
        assert wx["available"] is False and wx["error"]["kind"] == "invalid_input"
    far_future = datetime.now(timezone.utc) + timedelta(days=60)
    assert weather.fetch_conditions(19.4, -92.0, far_future, 24)["error"]["kind"] == "unavailable_date"
    assert weather.fetch_conditions(19.4, -92.0, datetime(1900, 1, 1, tzinfo=timezone.utc), 24)["error"]["kind"] == "unavailable_date"
    assert calls == []  # no se llamó al servicio


def test_forecast_historical_and_recent_dates_use_the_right_endpoint(fake_weather):
    now = datetime.now(timezone.utc)
    wx = weather.fetch_conditions(19.4, -92.0, None, 24)  # ahora -> pronóstico
    assert wx["mode"] == "forecast" and wx["has_waves"] and wx["has_currents"]
    fake_weather.clear()
    wx = weather.fetch_conditions(19.4, -92.0, now - timedelta(days=3), 24)  # reciente -> pronóstico con días pasados
    assert wx["available"] and wx["mode"] == "forecast"
    assert fake_weather[0][1]["past_days"] >= 4
    fake_weather.clear()
    wx = weather.fetch_conditions(19.4, -92.0, datetime(2020, 4, 25, 12, tzinfo=timezone.utc), 48)  # histórico
    assert wx["mode"] == "historical" and not wx["has_waves"] and not wx["has_currents"]
    service, p = fake_weather[0]
    assert service == "archivo histórico" and p["start_date"] == "2020-04-25" and p["end_date"] == "2020-04-27"
    fake_weather.clear()
    wx = weather.fetch_conditions(19.4, -92.0, now - timedelta(days=7), 72)  # el fin se recorta a lo disponible
    assert wx["available"] and wx["mode"] == "historical"
    assert datetime.fromisoformat(fake_weather[0][1]["end_date"]).date() <= (now - timedelta(days=5)).date()
    assert any("solo devolvió" in n for n in wx["notes"])


# ---------------- Los 10 casos obligatorios ----------------
def test_case1_image_without_location_has_scenarios_and_image_simulation():
    a = _small_scene_upload()
    assert a["bounds"] is None and a["summary"]["location"]["state"] == "none"
    sc = client.get(f"/api/analyses/{a['id']}/scenarios")
    assert sc.status_code == 200 and _strict_json(sc.text)["located"] is False
    wx = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "weather", "hours": 12})
    assert wx.status_code == 400 and "Sin ubicación" in wx.json()["detail"]  # no se pide clima sin ubicación


def test_case2_and_3_valid_coordinates_and_manual_location(fake_weather):
    a = _small_scene_upload(lat="19.45", lon="-92.0")
    assert geo.valid_bounds(a["bounds"]) and a["summary"]["location"]["state"] == "manual"
    assert "aproximada" in a["summary"]["location"]["text"]
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 6, "wind_from_deg": 45, "hours": 12})
    assert r.status_code == 200
    b = _small_scene_upload()
    upd = client.post(f"/api/analyses/{b['id']}/location", json={"lat": 19.5, "lon": -92.1, "scene_km": 3.4})
    assert upd.status_code == 200 and upd.json()["summary"]["location"]["state"] == "manual"


def test_case4_weather_available_drives_the_simulation(fake_weather):
    a = _small_scene_upload(lat="19.45", lon="-92.0")
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "weather", "hours": 12})
    assert r.status_code == 200, r.text
    sim = _strict_json(r.text)
    _all_finite(sim)
    assert sim["mode"] == "weather" and sim["weather_mode"] == "forecast"
    assert sim["forcing"]["uses_currents"] is True and sim["drift"]["direction_es"] in ("oeste", "suroeste", "noroeste", "sur")
    ctx = client.get(f"/api/analyses/{a['id']}/context")
    assert ctx.status_code == 200 and _strict_json(ctx.text)["weather"]["available"] is True


def test_case5_weather_unavailable_does_not_block_the_app(monkeypatch):
    a = _small_scene_upload(lat="19.45", lon="-92.0")
    _scripted(monkeypatch, [_http_error(503, b'{"error":true,"reason":"Service Unavailable"}')])
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "weather", "hours": 12})
    assert r.status_code == 503 and "HTTP 503" in r.json()["detail"] and "modo manual" in r.json()["detail"]
    ctx = client.get(f"/api/analyses/{a['id']}/context")  # el contexto sigue funcionando sin clima
    assert ctx.status_code == 200
    c = _strict_json(ctx.text)
    assert c["weather"]["available"] is False and len(c["recommendations"]) >= 4 and c["coast"] is not None
    manual = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12})
    assert manual.status_code == 200  # con el viento manual se puede seguir
    assert any(e["http"] == 503 for e in weather.recent_errors())
    assert client.get("/api/status").json()["weather_recent_errors"]  # visible para diagnóstico


def test_case6_manual_wind_and_case7_no_nan(fake_weather):
    for loc in ({}, {"lat": "19.45", "lon": "-92.0"}):
        a = _small_scene_upload(**loc)
        for wind, frm in ((0, 0), (3, 270), (22, 45)):
            r = client.post(f"/api/analyses/{a['id']}/simulate",
                            json={"mode": "manual", "wind_speed_ms": wind, "wind_from_deg": frm, "hours": 24})
            assert r.status_code == 200, r.text
            _all_finite(_strict_json(r.text))


def test_case8_historical_example_uses_the_real_date(fake_weather):
    ex = client.get("/api/examples").json()["items"][0]
    a = client.post(f"/api/examples/{ex['sample_id']}/analyze").json()
    assert a["bounds"] is not None
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "weather", "hours": 12})
    assert r.status_code == 200, r.text
    sim = _strict_json(r.text)
    assert sim["weather_mode"] == "historical"
    archive = [p for s, p in fake_weather if s == "archivo histórico"][0]
    assert archive["start_date"].startswith("2020-") and archive["end_date"] >= archive["start_date"]
    assert any("no un pronóstico" in n for n in sim["weather_notes"])


def test_case9_map_framing_contains_scene_and_track_with_confirmed_location(fake_weather):
    ex = client.get("/api/examples").json()["items"][0]
    a = client.post(f"/api/examples/{ex['sample_id']}/analyze").json()
    sim = _strict_json(client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 10, "wind_from_deg": 90, "hours": 24}).text)
    (fs, fw), (fn, fe) = sim["footprint"]
    (bs, bw), (bn, be) = sim["bounds"]
    assert geo.valid_bounds(sim["footprint"]) and bs <= fs < fn <= bn and bw <= fw < fe <= be  # huella dentro del dominio
    for lat, lon, _h in sim["track"]:
        assert bs <= lat <= bn and bw <= lon <= be  # la trayectoria cabe en el dominio simulado
    assert sim["origin"][0] == pytest.approx((fs + fn) / 2, abs=(fn - fs))  # el origen está sobre la escena
    # los límites coinciden con la georreferencia real del GeoTIFF del dataset (EPSG:4326, Golfo de México)
    assert -93.2 < fw < fe < -91.8 and 18.9 < fs < fn < 20.1


def test_invalid_stored_bounds_are_treated_as_unlocated():
    a = _small_scene_upload(lat="19.45", lon="-92.0")
    storage.update_location(a["id"], [[float("nan"), -92.1], [19.5, -92.0]], "manual: x", 19.45, -92.0, 1.0, a["result"])
    got = client.get(f"/api/analyses/{a['id']}").json()
    assert got["bounds"] is None and got["summary"]["location"]["state"] == "none"
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12})
    assert r.status_code == 200 and r.json()["unlocated"] is True  # se simula sobre la imagen, sin inventar posición
    listed = [x for x in client.get("/api/analyses").json()["items"] if x["id"] == a["id"]][0]
    assert listed["bounds"] is None


def test_case10_simulation_on_image_without_georeference():
    a = _small_scene_upload()
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12})
    sim = _strict_json(r.text)
    _all_finite(sim)
    assert sim["unlocated"] is True and sim["image_size"] == {"width": 457, "height": 224}
    # marco local (no geográfico) centrado en (0,0): nunca se coloca sobre una costa real
    (fs, fw), (fn, fe) = sim["footprint"]
    assert abs((fs + fn) / 2) < 0.01 and abs((fw + fe) / 2) < 0.01
    assert sim["context"]["coast"] is None and "No se conoce la ubicación" in sim["context"]["impact"][0]
