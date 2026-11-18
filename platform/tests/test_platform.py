"""Pruebas de la API de OCEAN GUARD AI.

Ejecutar desde platform/:
  <venv>\\Scripts\\python.exe -m pytest -q tests

Usa un directorio temporal para los datos generados y solo lee el split de validación del
dataset (nunca el de test).
"""
import base64
import io
import os
import sys
import tempfile
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))
os.environ["OCEAN_PLATFORM_DATA"] = tempfile.mkdtemp(prefix="ogtest_")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import config  # noqa: E402
from app.main import app  # noqa: E402
import data as unet_data  # noqa: E402

client = TestClient(app)


def _val_png_bytes(index: int = 0) -> bytes:
    samples = unet_data.read_split(config.DATASET_ROOT, "val")
    buf = io.BytesIO()
    Image.open(samples[index].image_path).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_status_reports_reference_model():
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True, body["model_error"]
    assert body["model"]["epoch"] == 4
    assert body["model"]["val_dice"] == pytest.approx(0.5841, abs=1e-4)
    assert body["model"]["weights_sha256_ok"] is True


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "OCEAN GUARD AI" in r.text


def test_upload_analysis_with_manual_location_produces_files():
    r = client.post(
        "/api/analyze",
        files={"file": ("escena.png", _val_png_bytes(0), "image/png")},
        data={"lat": "19.4", "lon": "-92.1", "pixel_m": "10"},
    )
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["width"] == 800 and a["height"] == 600
    assert 0 <= a["slick_pct"] <= 100
    assert a["bounds"] and a["bounds"][0][0] < a["bounds"][1][0]
    assert a["slick_area_km2"] is not None
    assert "manual" in a["location_kind"]
    for kind in ("image", "mask", "overlay"):
        f = client.get(f"/api/analyses/{a['id']}/file/{kind}")
        assert f.status_code == 200 and f.headers["content-type"] == "image/png"


def test_upload_rejects_bad_format_and_corrupt_image():
    r = client.post("/api/analyze", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert r.status_code == 415
    assert "Formato no admitido" in r.json()["detail"]
    r = client.post("/api/analyze", files={"file": ("roto.png", b"no es png", "image/png")})
    assert r.status_code == 422
    assert "No se pudo leer" in r.json()["detail"]


def test_examples_reference_comparison_uses_only_val():
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert r.json()["split"] == "val"
    assert r.json()["count"] == 43
    sid = r.json()["items"][0]["sample_id"]
    a = client.post(f"/api/examples/{sid}/analyze").json()
    assert a["reference"]["dice"] is not None
    assert 0 <= a["reference"]["iou"] <= 1
    assert a["bounds"] is not None  # GeoTIFF del dataset georreferenciado


def test_example_outside_val_is_rejected():
    r = client.post("/api/examples/IMG_99_TILE_999/analyze")
    assert r.status_code == 404


def test_simulation_produces_frames_and_disclaimer():
    a = client.post("/api/examples/" + client.get("/api/examples").json()["items"][0]["sample_id"] + "/analyze").json()
    r = client.post(
        f"/api/analyses/{a['id']}/simulate",
        json={"wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12},
    )
    assert r.status_code == 200, r.text
    sim = r.json()
    assert "SIMULACIÓN ILUSTRATIVA" in sim["disclaimer"]
    assert len(sim["frames"]) >= 12
    for fr in sim["frames"]:
        base64.b64decode(fr["png_b64"])  # decodifica sin error
    assert sim["frames"][0]["hour"] == 0
    assert sim["frames"][-1]["hour"] == pytest.approx(12.0, abs=0.05)
    assert sim["frames"][-1]["remaining_mass_pct"] <= 100.0 + 1e-6
    assert [st["hour"] for st in sim["stages"]] == [0, 3, 6, 12]  # etapas pedidas
    assert sim["drift"]["direction_es"] == "oeste"  # viento desde el este -> hacia el oeste
    assert sim["drift"]["end"][1] < sim["origin"][1]  # termina más al oeste que el origen
    assert sim["bounds"][0][1] < sim["footprint"][0][1]  # dominio ampliado
    base64.b64decode(sim["affected_zone"]["png_b64"])
    assert "hacia el oeste" in sim["narrative"] and "ilustrativo" in sim["narrative"]
    last = client.get(f"/api/analyses/{a['id']}/simulation")
    assert last.status_code == 200 and last.json()["simulation_id"] == sim["simulation_id"]


def test_simulation_rejects_out_of_range_and_missing_location():
    r = client.post("/api/analyze", files={"file": ("x.png", _val_png_bytes(1), "image/png")})
    no_loc = r.json()
    r = client.post(f"/api/analyses/{no_loc['id']}/simulate",
                    json={"mode": "weather", "hours": 6})
    assert r.status_code == 400
    assert "Sin ubicación no se puede consultar el clima real" in r.json()["detail"]
    r = client.post(f"/api/analyses/{no_loc['id']}/simulate",
                    json={"wind_speed_ms": 99, "wind_from_deg": 0, "hours": 6})
    assert r.status_code == 422


def test_report_is_self_contained_html():
    a = client.post("/api/examples/" + client.get("/api/examples").json()["items"][2]["sample_id"] + "/analyze").json()
    r = client.get(f"/api/analyses/{a['id']}/report")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Informe de análisis" in r.text
    assert r.text.count("data:image/png;base64,") >= 3  # 3 imágenes + capa del mapa
    assert "no sustituye" in r.text and "Limitaciones" in r.text


def test_history_dashboard_and_delete():
    dash = client.get("/api/dashboard").json()
    assert dash["counts"]["analyses"] >= 1
    assert dash["dataset"]["splits"] == {"train": 108, "val": 43, "test": 22}
    items = client.get("/api/analyses").json()["items"]
    target = items[0]["id"]
    assert client.get(f"/api/analyses/{target}").status_code == 200
    assert client.delete(f"/api/analyses/{target}").status_code == 200
    assert client.get(f"/api/analyses/{target}").status_code == 404
    assert client.get("/api/analyses/../../etc").status_code == 404


def test_plain_language_summary_and_honest_location():
    items = client.get("/api/examples").json()["items"]
    assert items[0]["scene"]["date"].startswith("2020-")  # metadato real del CSV del dataset
    a = client.post(f"/api/examples/{items[0]['sample_id']}/analyze").json()
    sm = a["summary"]
    assert sm["headline"] and "una persona debe confirmarlo" in sm["text"] or sm["level"] == "none"
    assert sm["location"]["state"] == "georef" and "Cantarell" not in sm["location"]["text"]  # sin texto del dataset
    assert "Sentinel-1" in sm["acquired"]
    assert sm["confidence"]["label"] in {"Alta", "Media", "Baja", "—"}
    assert "falsa alarma" in sm["confidence"]["text"] or sm["level"] == "none"
    slick = client.get(f"/api/analyses/{a['id']}/file/slick")
    assert slick.status_code == 200 and slick.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(slick.content)).mode == "RGBA"


def test_upload_without_location_is_not_placed_on_a_map():
    r = client.post("/api/analyze", files={"file": ("z.png", _val_png_bytes(3), "image/png")})
    a = r.json()
    assert a["bounds"] is None and a["center_lat"] is None
    assert a["summary"]["location"]["state"] == "none"
    assert "sin ubicación geográfica confirmada" in a["summary"]["location"]["text"].lower()
    rep = client.get(f"/api/analyses/{a['id']}/report").text
    assert "Sin ubicación geográfica confirmada" in rep and "L.map(" not in rep


def test_manual_location_is_labelled_approximate_and_report_has_map():
    r = client.post(
        "/api/analyze", files={"file": ("m.png", _val_png_bytes(4), "image/png")},
        data={"lat": "19.5", "lon": "-92.0"},
    )
    a = r.json()
    assert a["summary"]["location"]["state"] == "manual"
    assert "aproximada" in a["summary"]["location"]["text"]
    rep = client.get(f"/api/analyses/{a['id']}/report").text
    assert "L.map(" in rep and "no está verificada" in rep


def test_dashboard_has_activity_and_reliability():
    d = client.get("/api/dashboard").json()
    assert len(d["activity"]) == 14
    assert 0 < d["reliability"]["precision"] < 1
    assert "total_slick_area_km2" in d["counts"]


# ---------- clima real, costa, riesgo y recomendaciones ----------
from app import coast, context as ctxmod, weather  # noqa: E402


def _rows(winds, waves=None, wind_from=90.0):
    return [{"t": f"2026-01-01T{h:02d}:00Z", "wind_ms": w, "wind_from": wind_from, "gust_ms": w + 2,
             "wave_m": (waves[h] if waves else None), "wave_from": None, "current_ms": None, "current_to": None}
            for h, w in enumerate(winds)]


def test_real_coast_distance_near_cantarell():
    c = coast.nearest_coast(19.45, -92.0)
    assert c is not None and 40 < c["distance_km"] < 200  # offshore, decenas de km de la costa
    on_land = coast.nearest_coast(17.0, -92.5)  # tierra adentro: distancia a la costa grande pero finita
    assert on_land["distance_km"] > 20


def test_operational_windows_and_adverse_weather():
    rows = _rows([5, 5, 5, 5, 14, 14, 6, 6, 6, 6, 6, 6])
    w = ctxmod.operational_windows(rows)
    assert w[0]["hours"] == 6 and w[0]["start"].endswith("06:00Z")
    stats = ctxmod.weather_stats(rows)
    assert stats["wind_max_ms"] == 14 and stats["max_drift_kmh"] == pytest.approx(0.03 * 14 * 3.6, abs=0.01)
    assert ctxmod.operational_windows(_rows([15] * 8)) == []


def test_risk_levels_follow_declared_rules():
    small = {"slick_area_km2": 0.2}
    far = {"start": {"distance_km": 300}, "closest": {"distance_km": 300}, "closest_hour": 0, "approaching": False, "reaches_coast": False}
    assert ctxmod.risk_level(small, far, None)["label"] == "Baja"
    hit = {"start": {"distance_km": 15}, "closest": {"distance_km": 0.2}, "closest_hour": 9, "approaching": True, "reaches_coast": True}
    assert ctxmod.risk_level({"slick_area_km2": 8}, hit, None)["label"] == "Alta"


def test_manual_simulation_includes_context_and_containment_advice():
    sid = client.get("/api/examples").json()["items"][1]["sample_id"]
    a = client.post(f"/api/examples/{sid}/analyze").json()
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 6, "wind_from_deg": 45, "hours": 12})
    assert r.status_code == 200, r.text
    c = r.json()["context"]
    titles = [x["title"] for x in c["recommendations"]]
    assert "Confirmar antes de actuar" in titles and "Seguridad y decisiones técnicas" in titles
    assert c["risk"]["label"] in {"Baja", "Moderada", "Alta"}
    assert c["coast"]["start"]["distance_km"] > 0
    assert c["weather"]["mode"] == "manual" and "no es clima real" in c["weather"]["source"]


def test_weather_mode_uses_real_historical_wind_for_2020_scene():
    sid = client.get("/api/examples").json()["items"][0]["sample_id"]
    a = client.post(f"/api/examples/{sid}/analyze").json()
    r = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "weather", "hours": 12})
    if r.status_code == 503:  # sin internet: aviso comprensible, no un error opaco
        assert "clima" in r.json()["detail"].lower()
        pytest.skip("sin conexión al servicio meteorológico")
    assert r.status_code == 200, r.text
    sim = r.json()
    assert sim["mode"] == "weather" and sim["weather_mode"] == "historical"
    assert "reanálisis" in sim["forcing"]["source"] or "históricos" in sim["forcing"]["source"]
    assert "condiciones reales" in sim["narrative"]
    assert sim["context"]["weather"]["has_currents"] is False  # no hay corrientes históricas de 2020
    assert any("no un pronóstico" in n for n in sim["context"]["weather"]["notes"])
    ctx_r = client.get(f"/api/analyses/{a['id']}/context")
    assert ctx_r.status_code == 200 and ctx_r.json()["stats"]["wind_max_ms"] > 0


def test_context_requires_location():
    r = client.post("/api/analyze", files={"file": ("q.png", _val_png_bytes(5), "image/png")})
    assert client.get(f"/api/analyses/{r.json()['id']}/context").status_code == 400


def test_non_radar_image_is_rejected_with_a_friendly_message():
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (30, 120, 200)).save(buf, format="PNG")  # imagen a color, como una captura
    r = client.post("/api/analyze", files={"file": ("captura.png", buf.getvalue(), "image/png")},
                    data={"lat": "19.4", "lon": "-92.1"})
    assert r.status_code == 422
    assert r.json()["detail"] == config.MSG_NOT_RADAR and "Sube una imagen de radar" in r.json()["detail"]
    gray = client.post("/api/analyze", files={"file": ("g.png", _val_png_bytes(6), "image/png")}).json()
    assert gray["summary"]["warnings"] == []  # las imágenes del dataset no generan avisos



# ---------- imágenes propias: sin ubicación, GPS, escala ----------
import json as _json  # noqa: E402
import numpy as _np  # noqa: E402


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _first_slick_example():
    for it in client.get("/api/examples").json()["items"]:
        a = client.post(f"/api/examples/{it['sample_id']}/analyze").json()
        if a["slick_pct"] > 0:
            return a
    raise AssertionError("ningún ejemplo con derrame")


def test_unlocated_upload_gets_a_drift_scenario_on_the_image():
    r = client.post("/api/analyze", files={"file": ("sin.png", _val_png_bytes(7), "image/png")}, data={"scene_km": "12"})
    a = r.json()
    assert a["bounds"] is None and a["result"]["scale_assumed"] is False
    assert a["result"]["scale_m_per_px"] == pytest.approx(12000 / 800, rel=1e-3)
    assert a["slick_area_km2"] is not None  # área calculable gracias a la escala indicada
    sim = client.post(f"/api/analyses/{a['id']}/simulate",
                      json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12})
    assert sim.status_code == 200, sim.text
    s = sim.json()
    assert s["unlocated"] is True and s["image_size"] == {"width": 800, "height": 600}
    assert s["drift"]["direction_es"] == "oeste"
    ctx_ = s["context"]
    assert ctx_["coast"] is None and ctx_["weather"]["mode"] == "manual"
    assert any("Indicar dónde está" in x["title"] for x in ctx_["recommendations"])
    assert any("No se conoce la ubicación" in x for x in ctx_["impact"])


def test_scale_assumed_is_declared_when_not_given():
    a = client.post("/api/analyze", files={"file": ("e.png", _val_png_bytes(8), "image/png")}).json()
    assert a["result"]["scale_assumed"] is True
    if a["slick_area_km2"] is not None:
        assert a["result"]["scale_assumed"] is True


def test_gps_in_photo_exif_is_used_as_location():
    img = Image.new("RGB", (400, 300), (60, 60, 60))
    exif = img.getexif()
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (19.0, 27.0, 0.0), "W", (92.0, 0.0, 0.0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    a = client.post("/api/analyze", files={"file": ("gps.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert a["location_kind"] == "georreferencia del archivo (GPS de la foto)"
    assert a["center_lat"] == pytest.approx(19.45, abs=0.01) and a["center_lon"] == pytest.approx(-92.0, abs=0.01)
    assert "GPS guardado en la foto" in a["summary"]["location"]["text"]


def test_location_can_be_added_later_but_real_georef_is_protected():
    a = client.post("/api/analyze", files={"file": ("l.png", _val_png_bytes(9), "image/png")}).json()
    assert a["bounds"] is None
    r = client.post(f"/api/analyses/{a['id']}/location", json={"lat": 19.5, "lon": -92.1, "scene_km": 12})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["bounds"] is not None and b["summary"]["location"]["state"] == "manual"
    assert b["result"]["scale_assumed"] is False
    ex = _first_slick_example()
    assert client.post(f"/api/analyses/{ex['id']}/location", json={"lat": 1, "lon": 1}).status_code == 409


def test_wind_direction_scenarios_cover_eight_directions():
    ex = _first_slick_example()
    r = client.get(f"/api/analyses/{ex['id']}/scenarios", params={"wind_ms": 10, "hours": 12})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 8 and r.json()["located"] is True
    east = next(i for i in items if i["from_deg"] == 90)
    assert east["to_es"] == "oeste" and east["distance_km"] == pytest.approx(0.03 * 10 * 3.6 * 12, rel=0.05)
    assert all(i["closest_km"] is not None for i in items)  # con ubicación se evalúa la costa
    un = client.post("/api/analyze", files={"file": ("u.png", _val_png_bytes(10), "image/png")}).json()
    r2 = client.get(f"/api/analyses/{un['id']}/scenarios")
    assert r2.status_code == 200 and r2.json()["located"] is False
    assert all(i["closest_km"] is None for i in r2.json()["items"])


def test_geocode_validates_and_degrades_gracefully():
    assert client.get("/api/geocode", params={"q": "a"}).status_code == 400
    r = client.get("/api/geocode", params={"q": "Ciudad del Carmen"})
    if r.status_code == 503:
        assert "conexión" in r.json()["detail"]
        pytest.skip("sin conexión al servicio de geocodificación")
    assert r.status_code == 200 and isinstance(r.json()["items"], list)


# ---------- interfaz simple: semáforo, gravedad, dirección y campos nuevos ----------
from app import narrative, simple  # noqa: E402


def test_traffic_light_rule():
    red = narrative.TRAFFIC_RULES
    big = [{"area_px": red["red_min_component_px"]}]
    assert narrative.traffic_light({"slick_px": 0, "components": []}) == "green"
    assert narrative.traffic_light({"slick_px": 500, "components": [{"area_px": 500}], "mean_prob_slick": 0.99}) == "yellow"
    assert narrative.traffic_light({"slick_px": 5000, "components": big, "mean_prob_slick": 0.96}) == "yellow"
    assert narrative.traffic_light({"slick_px": 5000, "components": big, "mean_prob_slick": red["red_mean_prob"]}) == "red"


def test_severity_rule_low_medium_high():
    low = simple.severity(False, None, 10, 3.0)
    assert low["level"] == "low" and low["label"] == "baja"
    med = simple.severity(True, 40, 10, 3.0)           # llega a la costa tarde: 1 punto
    assert med["level"] == "medium"
    med2 = simple.severity(False, None, 80, 4.0)       # solo crece mucho
    assert med2["level"] == "medium"
    high = simple.severity(True, 18, 10, 14.0)         # costa en 18 h + viento fuerte
    assert high["level"] == "high"
    assert "llega a la costa en 18 horas" in high["why"] and "viento es fuerte" in high["why"]
    assert high["why"].startswith("Gravedad alta porque")


def test_direction_text_in_spanish():
    assert simple.direction_text(45) == "hacia el noreste"
    assert simple.direction_text(270) == "hacia el oeste"
    assert simple.direction_text(180) == "hacia el sur"
    assert simple.direction_text(None) == "sin desplazamiento apreciable"


def test_api_response_includes_new_simple_fields():
    a = _first_slick_example()
    s = a["summary"]
    assert s["traffic"]["color"] in ("yellow", "red")
    assert s["traffic"]["label"] in narrative.TRAFFIC_LABEL.values()
    assert s["traffic"]["area_km2"] is not None and s["traffic"]["where"]
    assert "una persona debe confirmarlo" in s["traffic"]["phrase"]
    assert {"threshold", "reliability", "disclaimer"} <= set(s["technical"])
    sim = client.post(f"/api/analyses/{a['id']}/simulate",
                      json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 48})
    assert sim.status_code == 200, sim.text
    simple_ = sim.json()["simple"]
    assert simple_["direction_text"] == "hacia el oeste"
    assert [x["hour"] for x in simple_["stages"]] == [6, 12, 24, 48]
    assert all(x["displacement_km"] >= 0 and "area_growth_pct" not in x for x in simple_["stages"])  # sin porcentajes
    lo, hi = simple_["affected_range_km2"]
    assert 0 <= lo <= hi
    assert simple_["severity"]["level"] in ("low", "medium", "high") and simple_["severity"]["why"]
    assert simple_["coast"] is not None and simple_["coast"]["distance_km"] >= 0


# ---------- ejemplos con y sin derrame (autosuficientes) y filtro gris-en-RGB ----------
def test_examples_groups_and_static_files():
    g = client.get("/api/examples").json()["groups"]
    assert len(g["con"]) >= 5 and len(g["sin"]) >= 3   # los ejemplos anteriores se conservan (y hay mas)
    assert all(e["etiqueta"].startswith("Sin derrame ") for e in g["sin"]) and all(e["etiqueta"].startswith("Con derrame ") for e in g["con"])
    man = (config.PLATFORM_DIR / "static" / "examples")
    assert all((man / f).exists() for f in ("manifest.json", "sin_ADR.jpg", "sin_JAV.jpg", "sin_VEN.jpg"))


def test_example_with_spill_and_example_without_spill():
    g = client.get("/api/examples").json()["groups"]
    con = client.post(f"/api/examples/{g['con'][0]['sample_id']}/analyze").json()
    assert con["slick_pct"] > 0 and con["summary"]["traffic"]["color"] in ("yellow", "red")
    for e in g["sin"]:
        a = client.post(f"/api/examples/{e['sample_id']}/analyze").json()
        assert a["slick_pct"] == 0 and a["summary"]["traffic"]["color"] == "green"
        assert a["summary"]["traffic"]["label"] == "No se detecta derrame"
        sim = client.post(f"/api/analyses/{a['id']}/simulate", json={"mode": "manual", "wind_speed_ms": 8, "wind_from_deg": 90, "hours": 12})
        assert sim.status_code == 400  # sin derrame no hay nada que simular


def test_gray_jpeg_with_equal_channels_is_accepted_but_real_color_is_rejected():
    gray = Image.open(config.PLATFORM_DIR / "static" / "examples" / "sin_ADR.jpg").convert("RGB")  # JPEG gris repetido en 3 canales
    arr = _np.asarray(gray)
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all()
    buf = io.BytesIO()
    gray.save(buf, format="JPEG")
    ok = client.post("/api/analyze", files={"file": ("csiro.jpg", buf.getvalue(), "image/jpeg")})
    assert ok.status_code == 200, ok.text
    col = _np.zeros((400, 400, 3), dtype=_np.uint8)
    col[..., 0], col[..., 1], col[..., 2] = 40, 110, 190
    cb = io.BytesIO()
    Image.fromarray(col).save(cb, format="JPEG")
    bad = client.post("/api/analyze", files={"file": ("foto.jpg", cb.getvalue(), "image/jpeg")})
    assert bad.status_code == 422 and "no parece de radar" in bad.json()["detail"]


def test_confidence_text_mentions_false_alarm_rate():
    a = client.post(f"/api/examples/{client.get('/api/examples').json()['groups']['con'][0]['sample_id']}/analyze").json()
    txt = a["summary"]["confidence"]["text"]
    assert "1 de cada 4 o 5" in txt and "unas 87" not in txt and "No se midieron" not in txt


# ---------- confianza en porcentaje (puntaje del modelo) ----------
def test_confidence_percent_with_and_without_spill():
    assert narrative.model_score_percent({"slick_px": 100, "mean_prob_slick": 0.912}) == 91
    assert narrative.model_score_percent({"slick_px": 0, "max_prob": 0.13}) == 87      # 1 - máximo
    assert narrative.model_score_percent({"slick_px": 0, "max_prob": 0.0}) == 100
    assert narrative.model_score_percent({"slick_px": 5, "mean_prob_slick": 1.0}) == 100
    assert narrative.model_score_percent({"slick_px": 0}) is None                      # análisis antiguo sin dato
    c = narrative.confidence({"slick_px": 0, "max_prob": 0.40, "components": []})
    assert c["title"] == "Confianza de que no hay derrame: 60 %" and c["weak_signal"] is True
    assert narrative.confidence({"slick_px": 0, "max_prob": 0.20, "components": []})["weak_signal"] is False
    assert narrative.confidence({"slick_px": 0, "max_prob": 0.99, "components": []})["weak_signal"] is False  # sobre el umbral: no es señal débil


def test_api_confidence_card_for_example_with_and_without_spill():
    g = client.get("/api/examples").json()["groups"]
    con = client.post(f"/api/examples/{g['con'][0]['sample_id']}/analyze").json()
    cc = con["summary"]["confidence"]
    assert 0 <= cc["percent"] <= 100 and cc["title"] == f"Confianza: {cc['percent']} %"
    assert con["summary"]["confidence_percent"] == cc["percent"]
    assert cc["text"] == narrative.FALSE_ALARM_LINE
    assert not any(w in cc["title"] + cc["text"] for w in ("Alta", "Media", "Baja"))
    sin = client.post(f"/api/examples/{g['sin'][0]['sample_id']}/analyze").json()
    cs = sin["summary"]["confidence"]
    assert 0 <= cs["percent"] <= 100 and cs["title"].startswith("Confianza de que no hay derrame: ")
    assert sin["result"]["max_prob"] is not None
    assert cs["percent"] == round((1 - sin["result"]["max_prob"]) * 100) or abs(cs["percent"] - (1 - sin["result"]["max_prob"]) * 100) < 1
    assert not any(w in cs["title"] + cs["text"] for w in ("Alta", "Media", "Baja"))
    rep_html = client.get(f"/api/analyses/{sin['id']}/report").text
    assert "Puntaje de confianza del modelo" in rep_html and f"{cs['percent']} %" in rep_html


# ---------- ejemplos ampliados (solo del conjunto de prueba, livianos, con fuente) ----------
def test_manifest_has_enough_light_examples_with_source():
    import json as _j
    d = config.PLATFORM_DIR / "static" / "examples"
    man = _j.loads((d / "manifest.json").read_text(encoding="utf-8"))
    nuevos_con = [e for e in man["con"] if e.get("conjunto") == "prueba"]
    nuevos_sin = [e for e in man["sin"] if e.get("conjunto") == "prueba"]
    assert len(nuevos_con) >= 10 and len(nuevos_sin) >= 10
    assert len(man["con"]) >= 15 and len(man["sin"]) >= 13  # los ejemplos anteriores se conservan
    assert sum(e["id"].startswith("SOS_") for e in nuevos_con) >= 3 and sum(e["id"].startswith("IMG_") for e in nuevos_con) >= 3
    assert len({e["region"] for e in nuevos_sin}) >= 3
    for e in man["con"] + man["sin"]:
        assert e.get("fuente"), e["id"]
        assert (d / e["file"]).exists() and (e["file"].endswith((".jpg", ".png")))
        assert (d / e["file"]).stat().st_size < 600_000
    for e in nuevos_con:
        assert (d / e["mask"]).exists()
        assert bool(e.get("zona")) == e["id"].startswith("IMG_")  # la zona solo donde el dataset la dice (originales); SOS: solo fuente
    assert all("zona" not in e for e in man["sin"])             # CSIRO: solo fuente y codigo de region
    assert sum(p.stat().st_size for p in d.iterdir()) < 8_000_000


def test_every_example_analyzes_and_groups_expose_source():
    g = client.get("/api/examples").json()["groups"]
    assert len(g["con"]) >= 15 and len(g["sin"]) >= 13
    assert all(i["etiqueta"] for i in g["con"] + g["sin"])
    for i in g["con"] + g["sin"]:
        r = client.post(f"/api/examples/{i['sample_id']}/analyze")
        assert r.status_code == 200, (i["sample_id"], r.text)
        assert r.json()["source"] == "ejemplo"


# ---------- etiquetas neutras: ningun nombre de dataset, region ni archivo visible ----------
import re as _re  # noqa: E402

_PROHIBIDO = _re.compile(r"\bSOS\b|CSIRO|dataset|fuente|INPE|\b(COR|EGY|GIB|NOR|ADR|JAV|VEN|GBR)\b|IMG_\d|SIN_|TILE|\.(png|jpg|tiff?)\b", _re.I)


def test_example_labels_are_neutral_everywhere_visible():
    g = client.get("/api/examples").json()["groups"]
    blob = _json.dumps(g, ensure_ascii=False)
    labels = [i["etiqueta"] for i in g["con"] + g["sin"]]
    assert labels == [f"Con derrame {k}" for k in range(1, len(g["con"]) + 1)] + [f"Sin derrame {k}" for k in range(1, len(g["sin"]) + 1)]
    for i in g["con"] + g["sin"]:
        assert set(i) == {"sample_id", "etiqueta"}                       # el origen no sale por la API de ejemplos
        assert not _PROHIBIDO.search(i["etiqueta"]), i["etiqueta"]
    assert not any(w in blob for w in ("fuente", "zona", "region", "CSIRO", "Cantarell"))
    for i in (g["con"][0], g["con"][-1], g["sin"][0], g["sin"][-1]):
        a = client.post(f"/api/examples/{i['sample_id']}/analyze").json()
        assert a["filename"] == i["etiqueta"] and not _PROHIBIDO.search(a["filename"])
        assert not _PROHIBIDO.search(a["summary"]["headline"] + a["summary"]["text"] + a["summary"]["traffic"]["phrase"])
        rep_html = client.get(f"/api/analyses/{a['id']}/report").text
        assert "dataset" not in rep_html.lower() and "TILE" not in rep_html and "CSIRO" not in rep_html
    nombres = [x["filename"] for x in client.get("/api/analyses").json()["items"] if x["source"] == "ejemplo"]
    assert nombres and all(re_ok(n) for n in nombres)


def re_ok(name: str) -> bool:
    return bool(_re.fullmatch(r"(Con derrame \d+|Sin derrame \d+|Ejemplo)", name))
