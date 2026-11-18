"""API y frontend de OCEAN GUARD AI: análisis de posibles derrames con la U-Net de referencia."""
from __future__ import annotations

import io
import json
import logging
import math
import re
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import analysis as A
from . import config
from . import context as ctx
from . import geo
from . import inference
from . import narrative
from . import scene_meta
from . import report as R
from . import simulation as S
from . import storage

import data as unet_data  # noqa: E402  (sys.path ya preparado por inference)

ID_PATTERN = re.compile(r"[0-9a-f]{12}")

log = logging.getLogger("oceanguard.api")


def _sanitize(obj, path: str = "$"):
    """Copia de `obj` con los números no finitos (NaN, ±inf) convertidos en None. Devuelve (copia, cuántos).

    Los valores ausentes o indefinidos van como `null`, nunca como cero. Cada hallazgo se registra para
    corregir el origen: esto es una red de seguridad, no una forma de ocultar el fallo.
    """
    bad = 0
    if isinstance(obj, float) or (hasattr(obj, "dtype") and getattr(obj, "shape", None) == ()):
        v = float(obj)
        if not math.isfinite(v):
            log.warning("valor no finito en la respuesta: %s", path)
            return None, 1
        return v, 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k], n = _sanitize(v, f"{path}.{k}")
            bad += n
        return out, bad
    if isinstance(obj, (list, tuple)):
        out = []
        for i, v in enumerate(obj):
            item, n = _sanitize(v, f"{path}[{i}]")
            out.append(item)
            bad += n
        return out, bad
    return obj, 0


class SafeJSONResponse(JSONResponse):
    """JSON estricto: nunca emite NaN/Infinity (no son JSON válido)."""

    def render(self, content) -> bytes:
        clean, _ = _sanitize(content)
        return json.dumps(clean, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


app = FastAPI(title="OCEAN GUARD AI", version="0.1.0", default_response_class=SafeJSONResponse)
STATIC = config.PLATFORM_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(Exception)
async def _unexpected(request, exc):  # los errores no esperados se muestran, no se ocultan
    return JSONResponse(status_code=500, content={"detail": f"Error interno: {type(exc).__name__}: {exc}"})


class SimulationRequest(BaseModel):
    mode: str = Field("manual", pattern="^(manual|weather)$", description="manual = viento constante; weather = clima real")
    wind_speed_ms: float | None = Field(None, ge=0, le=30, description="Velocidad del viento (m/s), modo manual")
    wind_from_deg: float | None = Field(None, ge=0, le=360, description="Dirección de origen del viento, modo manual")
    hours: float = Field(..., ge=1, le=72, description="Duración de la simulación (horas)")


def _model_info() -> dict | None:
    seg = inference.get_segmenter()
    return seg.info if seg else None


def _enrich(a: dict) -> dict:
    """Añade el resumen comprensible (titular, explicación, confianza, ubicación)."""
    a = dict(a)
    a["summary"] = narrative.describe(a)
    return a


def _summary(a: dict) -> dict:
    a = _normalize(a)
    out = {k: a.get(k) for k in (
        "id", "created_at", "source", "filename", "width", "height", "slick_pct",
        "n_components", "slick_area_km2", "location_kind", "center_lat", "center_lon", "bounds",
    )}
    d = narrative.describe(a)
    out.update({
        "level": d["level"],
        "headline": d["headline"],
        "location_state": d["location"]["state"],
        "acquired": d["acquired"],
    })
    return out


def _normalize(a: dict) -> dict:
    """Si la huella guardada no es válida (no finita, desordenada o absurda), el evento se trata como
    sin ubicación: no se coloca en un mapa una posición que no se puede sostener."""
    if a.get("bounds") and not geo.valid_bounds(a["bounds"]):
        log.warning("análisis %s con huella no válida %s: se trata como sin ubicación", a.get("id"), a["bounds"])
        a = dict(a, bounds=None, center_lat=None, center_lon=None,
                 location_kind="no indicada (la ubicación guardada no es válida)")
    return a


def _get_or_404(analysis_id: str) -> dict:
    if not ID_PATTERN.fullmatch(analysis_id):
        raise HTTPException(404, "Análisis no encontrado.")
    a = storage.get_analysis(analysis_id)
    if a is None:
        raise HTTPException(404, "Análisis no encontrado.")
    return _normalize(a)


async def _read_upload(file: UploadFile) -> bytes:
    ext = Path((file.filename or "").lower()).suffix
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Formato no admitido ({ext or 'sin extensión'}). Use PNG, JPG o TIFF.")
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El archivo supera el límite de 25 MB.")
    if not raw:
        raise HTTPException(400, "El archivo está vacío.")
    return raw


def _open_image(raw: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, "No se pudo leer la imagen. Verifique que el archivo no esté dañado.")
    if max(im.size) > config.MAX_SIDE_PX:
        raise HTTPException(413, f"La dimensión máxima admitida es {config.MAX_SIDE_PX} px por lado.")
    return im


def _run_pipeline(rgb: np.ndarray, *, source: str, filename: str | None, bounds, location_kind: str,
                  reference_mask: np.ndarray | None = None, extra_result: dict | None = None,
                  pixel_m: float | None = None, scale_assumed: bool = False,
                  manual_mask: np.ndarray | None = None) -> dict:
    if manual_mask is not None:
        mask = manual_mask.astype(bool)
        prob = mask.astype(np.float32)
    else:
        seg = inference.get_segmenter()
        if seg is None:
            raise HTTPException(503, f"El modelo no está disponible: {inference.load_error()}")
        prob = seg.predict_prob(rgb)
        mask = A.remove_small_components(prob > config.DEFAULT_THRESHOLD, config.MIN_COMPONENT_PX)
    H, W = mask.shape
    if bounds:
        pixel_km2 = geo.pixel_area_km2(bounds, W, H)
        pixel_m = geo.pixel_size_m(bounds, W, H)[0]
    else:
        pixel_km2 = (pixel_m / 1000.0) ** 2 if pixel_m else None
    result = A.summarize(mask, prob, pixel_km2)
    if manual_mask is not None:
        result["manual_mask"] = True
        result["mean_prob_slick"] = None
        result["warnings"] = [{"code": "manual", "text": "La zona fue marcada por usted; el sistema no verificó que sea petróleo."}]
    else:
        result["warnings"] = A.input_warnings(rgb) if source.startswith("carga del usuario") else []
    result["scale_m_per_px"] = round(pixel_m, 3) if pixel_m else None
    result["scale_assumed"] = bool(scale_assumed)
    if extra_result:
        result.update(extra_result)
    reference = A.binary_scores(mask, reference_mask) if reference_mask is not None else None

    analysis_id = uuid.uuid4().hex[:12]
    folder = storage.analysis_dir(analysis_id)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(folder / "image.png")
    A.mask_image(mask).save(folder / "mask.png")
    A.overlay(rgb, mask).save(folder / "overlay.png")
    A.slick_layer(mask).save(folder / "slick.png")

    center_lat = center_lon = None
    if bounds:
        center_lat = round((bounds[0][0] + bounds[1][0]) / 2, 6)
        center_lon = round((bounds[0][1] + bounds[1][1]) / 2, 6)

    storage.save_analysis({
        "id": analysis_id,
        "created_at": storage.now_iso(),
        "source": source,
        "filename": filename,
        "width": W,
        "height": H,
        "slick_pct": result["slick_pct"],
        "n_components": result["n_components"],
        "slick_area_km2": result["slick_area_km2"],
        "threshold": config.DEFAULT_THRESHOLD,
        "bounds_json": None if bounds is None else json.dumps(bounds),
        "location_kind": location_kind,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "reference_json": None if reference is None else json.dumps(reference),
        "result_json": json.dumps(result),
        "notes": None,
    })
    return _enrich(storage.get_analysis(analysis_id))


# ---------- Páginas y estado ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    seg = inference.get_segmenter()
    return {
        "model_loaded": seg is not None,
        "model": seg.info if seg else None,
        "model_error": None if seg else inference.load_error(),
        "dataset_present": config.DATASET_ROOT.exists(),
        "data_dir": str(config.DATA_DIR),
        "weather_recent_errors": ctx.weather.recent_errors()[-5:],
        "disclaimer": config.DISCLAIMER_DETECTION,
    }


@app.get("/api/dashboard")
def dashboard():
    counts = storage.dashboard_counts()
    counts["total_slick_area_km2"] = storage.total_slick_area_km2()
    return {
        "counts": counts,
        "activity": storage.activity_last_days(14),
        "recent": [_summary(a) for a in storage.list_analyses(6)],
        "reliability": config.VAL_RELIABILITY,
        "model": _model_info(),
        "dataset": {
            "present": config.DATASET_ROOT.exists(),
            "tiles": sum(unet_data.EXPECTED_COUNTS.values()),
            "splits": unet_data.EXPECTED_COUNTS,
        },
    }


# ---------- Análisis ----------

@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    pixel_m: float | None = Form(None),
    scene_km: float | None = Form(None),
    mask: UploadFile | None = File(None),
):
    raw = await _read_upload(file)
    im = _open_image(raw)
    rgb = A.to_rgb_uint8(im)
    H, W = rgb.shape[:2]

    manual = None
    if mask is not None and mask.filename:
        mraw = await mask.read(config.MAX_UPLOAD_BYTES + 1)
        try:
            mim = Image.open(io.BytesIO(mraw)).convert("L").resize((W, H), Image.NEAREST)
        except Exception:
            raise HTTPException(422, "No se pudo leer la zona marcada.")
        manual = np.asarray(mim) > 127
        if not manual.any():
            raise HTTPException(400, "La zona marcada está vacía: pinte sobre la imagen la mancha que quiere analizar.")

    georef = None
    try:
        georef = geo.read_georef(io.BytesIO(raw))
    except Exception:
        georef = None

    scale_assumed = False
    if georef is not None:
        bounds = geo.bounds_from_georef(georef, W, H)
        location_kind = "georreferencia del archivo (EPSG:4326)"
        pm = geo.pixel_size_m(bounds, W, H)[0]
    else:
        if scene_km is not None:
            if not (0.05 <= scene_km <= 5000):
                raise HTTPException(400, "El ancho de la escena debe estar entre 0,05 y 5000 km.")
            pm = scene_km * 1000.0 / W
        elif pixel_m is not None:
            if not (0.5 <= pixel_m <= 1000):
                raise HTTPException(400, "La resolución debe estar entre 0,5 y 1000 m por píxel.")
            pm = pixel_m
        else:
            pm = config.DEFAULT_PIXEL_M
            scale_assumed = True
        exif = geo.exif_gps(im)
        if lat is not None and lon is not None:
            try:
                bounds = geo.manual_bounds(lat, lon, W, H, pm)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            location_kind = f"manual: ubicación indicada, escala {'supuesta' if scale_assumed else 'indicada'} de {pm:.1f} m/px"
        elif exif:
            bounds = geo.manual_bounds(exif[0], exif[1], W, H, pm)
            location_kind = "georreferencia del archivo (GPS de la foto)"
        else:
            bounds = None
            location_kind = "no indicada"

    return _run_pipeline(
        rgb, source="carga del usuario (zona marcada a mano)" if manual is not None else "carga del usuario",
        filename=file.filename, bounds=bounds, location_kind=location_kind,
        pixel_m=pm, scale_assumed=scale_assumed, manual_mask=manual,
    )


class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    scene_km: float | None = Field(None, ge=0.05, le=5000, description="Ancho aproximado de la escena en km")


@app.post("/api/analyses/{analysis_id}/location")
def set_location(analysis_id: str, req: LocationRequest):
    """Indica (o corrige) la ubicación aproximada de un análisis que no la tiene o la tiene manual."""
    a = _get_or_404(analysis_id)
    kind = a.get("location_kind") or ""
    if a.get("bounds") and not kind.startswith("manual"):
        raise HTTPException(409, "Este análisis ya tiene una ubicación real tomada del archivo; no se sobrescribe.")
    result = dict(a.get("result") or {})
    W, H = a["width"], a["height"]
    assumed = bool(result.get("scale_assumed"))
    if req.scene_km is not None:
        pm, assumed = req.scene_km * 1000.0 / W, False
    else:
        pm = result.get("scale_m_per_px") or config.DEFAULT_PIXEL_M
    bounds = geo.manual_bounds(req.lat, req.lon, W, H, pm)
    area = geo.pixel_area_km2(bounds, W, H) * (result.get("slick_px") or 0)
    result["scale_m_per_px"] = round(pm, 3)
    result["scale_assumed"] = assumed
    new_kind = f"manual: ubicación indicada, escala {'supuesta' if assumed else 'indicada'} de {pm:.1f} m/px"
    storage.update_location(analysis_id, bounds, new_kind, round(req.lat, 6), round(req.lon, 6),
                            round(area, 4), result)
    return _enrich(storage.get_analysis(analysis_id))


@app.get("/api/geocode")
def geocode(q: str):
    q = q.strip()
    if len(q) < 2:
        raise HTTPException(400, "Escriba al menos 2 letras del lugar.")
    try:
        items = ctx.weather.geocode(q)
    except ctx.weather.WeatherError as exc:
        raise HTTPException(503, f"No se pudo buscar el lugar: {exc.message} Puede escribir las coordenadas.")
    return {"items": items}


@app.get("/api/examples")
def examples():
    try:
        samples = unet_data.read_split(config.DATASET_ROOT, "val")
    except Exception as exc:
        raise HTTPException(503, f"Dataset no disponible: {exc}")
    return {"split": "val", "count": len(samples), "items": [
        {"sample_id": s.sample_id, "scene": scene_meta.for_sample(s.sample_id)} for s in samples
    ]}


@app.post("/api/examples/{sample_id}/analyze")
def analyze_example(sample_id: str):
    """Ejemplo del split val con su máscara de referencia (solo lectura del dataset)."""
    try:
        samples = unet_data.read_split(config.DATASET_ROOT, "val")
    except Exception as exc:
        raise HTTPException(503, f"Dataset no disponible: {exc}")
    sample = next((s for s in samples if s.sample_id == sample_id), None)
    if sample is None:
        raise HTTPException(404, "Ejemplo no encontrado en el split de validación.")

    rgb = np.asarray(Image.open(sample.image_path).convert("RGB"))
    ref = unet_data.load_mask_binary(sample.mask_path).astype(bool)
    H, W = rgb.shape[:2]

    tif = config.DATASET_ROOT / "RASTER" / "IMG-TIFF" / "16-BIT" / f"{sample_id}.tiff"
    georef = geo.read_georef(tif) if tif.exists() else None
    if georef:
        bounds = geo.bounds_from_georef(georef, W, H)
        location_kind = "georreferencia del dataset (GeoTIFF EPSG:4326)"
    else:
        bounds, location_kind = None, "no indicada"

    return _run_pipeline(
        rgb, source="ejemplo del dataset (val)", filename=f"{sample_id}.png",
        bounds=bounds, location_kind=location_kind, reference_mask=ref,
        extra_result={"scene": scene_meta.for_sample(sample_id)},
    )


@app.get("/api/analyses")
def list_analyses():
    return {"items": [_summary(a) for a in storage.list_analyses()]}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    return _enrich(_get_or_404(analysis_id))


@app.get("/api/analyses/{analysis_id}/file/{kind}")
def analysis_file(analysis_id: str, kind: str):
    _get_or_404(analysis_id)
    if kind not in {"image", "mask", "overlay", "slick"}:
        raise HTTPException(404, "Archivo no encontrado.")
    path = storage.analysis_dir(analysis_id) / f"{kind}.png"
    if kind == "slick" and not path.exists() and (path.parent / "mask.png").exists():
        mask = np.asarray(Image.open(path.parent / "mask.png")) > 127
        A.slick_layer(mask).save(path)  # capa transparente para superponer sobre el mapa
    if not path.exists():
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(path, media_type="image/png")


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    _get_or_404(analysis_id)
    storage.delete_analysis(analysis_id)
    return {"deleted": analysis_id}


@app.get("/api/analyses/{analysis_id}/report", response_class=HTMLResponse)
def analysis_report(analysis_id: str):
    a = _get_or_404(analysis_id)
    folder = storage.analysis_dir(analysis_id)
    if not (folder / "slick.png").exists() and (folder / "mask.png").exists():
        A.slick_layer(np.asarray(Image.open(folder / "mask.png")) > 127).save(folder / "slick.png")
    sim = storage.latest_simulation(analysis_id)
    context = None
    if sim and sim.get("context"):
        context = sim["context"]
    elif a.get("bounds") and (a.get("slick_pct") or 0) > 0:
        try:
            context = ctx.build(a, np.asarray(Image.open(folder / "mask.png")) > 127, 48)
        except Exception:
            context = None
    html = R.build_report(_enrich(a), folder, _model_info(), sim, context)
    return HTMLResponse(html, headers={"Content-Disposition": f'inline; filename="informe_{analysis_id}.html"'})


# ---------- Simulación ----------

@app.post("/api/analyses/{analysis_id}/simulate")
def simulate(analysis_id: str, req: SimulationRequest):
    a = _get_or_404(analysis_id)
    located = bool(a.get("bounds"))
    W, H = a["width"], a["height"]
    pm = (a.get("result") or {}).get("scale_m_per_px") or config.DEFAULT_PIXEL_M
    bounds = a["bounds"] if located else geo.manual_bounds(0.0, 0.0, W, H, pm)  # marco local, no geográfico
    if not (a.get("slick_pct") or 0) > 0:
        raise HTTPException(400, "No hay derrame detectado en este análisis: nada que simular.")
    if not located and req.mode == "weather":
        raise HTTPException(400, "Sin ubicación no se puede consultar el clima real. Use el modo manual o indique dónde está.")
    mask = np.asarray(Image.open(storage.analysis_dir(analysis_id) / "mask.png")) > 127
    a2 = dict(a, bounds=bounds)
    hours_i = int(math.ceil(req.hours))
    forcing = None
    source = "viento constante indicado por el usuario"
    if req.mode == "weather":
        origin = ctx.origin_from_mask(mask, bounds)
        wx = ctx.weather.fetch_conditions(origin[0], origin[1], ctx.reference_time(a), hours_i + 1)
        if not wx.get("available"):
            raise HTTPException(503, wx["reason"])
        forcing = wx["hourly"]
        source = wx["source"]
    else:
        if req.wind_speed_ms is None or req.wind_from_deg is None:
            raise HTTPException(400, "Indique la velocidad y la dirección del viento.")
        wx = ctx.manual_conditions(req.wind_speed_ms, req.wind_from_deg, req.hours)
    try:
        result = S.simulate(mask, bounds, req.wind_speed_ms, req.wind_from_deg, req.hours,
                            forcing=forcing, forcing_source=source)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    try:
        result["context"] = ctx.build(a2, mask, hours_i + 1, wx=wx, located=located)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result["weather_notes"] = wx.get("notes", [])
    result["weather_mode"] = wx.get("mode")
    result["unlocated"] = not located
    result["image_size"] = {"width": W, "height": H}
    sim_id = uuid.uuid4().hex[:12]
    result["simulation_id"] = sim_id
    result["analysis_id"] = analysis_id
    storage.save_simulation(sim_id, analysis_id, result["params"], result)
    return result


@app.get("/api/analyses/{analysis_id}/simulation")
def latest_simulation(analysis_id: str):
    _get_or_404(analysis_id)
    sim = storage.latest_simulation(analysis_id)
    if sim is None:
        raise HTTPException(404, "Este análisis todavía no tiene simulaciones.")
    return sim


@app.get("/api/analyses/{analysis_id}/context")
def analysis_context(analysis_id: str, hours: int = 48):
    """Clima real, costa más cercana, nivel de atención y recomendaciones para un evento con ubicación."""
    a = _get_or_404(analysis_id)
    if not a.get("bounds"):
        raise HTTPException(400, "Este análisis no tiene ubicación; no se puede consultar el clima ni la costa.")
    if any(w["code"] == "color" for w in (a.get("result") or {}).get("warnings", [])):
        raise HTTPException(400, "La imagen no parece de radar (tiene color): no se calcula el contexto porque el resultado no es significativo.")
    if not (a.get("slick_pct") or 0) > 0:
        raise HTTPException(400, "No hay zona marcada en este análisis.")
    if not (6 <= hours <= 72):
        raise HTTPException(400, "El periodo debe estar entre 6 y 72 horas.")
    mask = np.asarray(Image.open(storage.analysis_dir(analysis_id) / "mask.png")) > 127
    try:
        return ctx.build(a, mask, hours)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/analyses/{analysis_id}/scenarios")
def scenarios(analysis_id: str, wind_ms: float = 8.0, hours: int = 12):
    """«¿Y si el viento viene de…?»: hacia dónde se desplazaría la mancha con viento de cada dirección."""
    a = _get_or_404(analysis_id)
    if not (a.get("slick_pct") or 0) > 0:
        raise HTTPException(400, "No hay zona marcada en este análisis.")
    if not (0 <= wind_ms <= 30) or not (1 <= hours <= 72):
        raise HTTPException(400, "Viento entre 0 y 30 m/s y duración entre 1 y 72 horas.")
    located = bool(a.get("bounds"))
    W, H = a["width"], a["height"]
    pm = (a.get("result") or {}).get("scale_m_per_px") or config.DEFAULT_PIXEL_M
    bounds = a["bounds"] if located else geo.manual_bounds(0.0, 0.0, W, H, pm)
    mask = np.asarray(Image.open(storage.analysis_dir(analysis_id) / "mask.png")) > 127
    origin = ctx.origin_from_mask(mask, bounds)
    items = []
    for from_deg in range(0, 360, 45):
        rows = S.constant_forcing(wind_ms, from_deg, hours)
        track = S.lagrangian_track(origin[0], origin[1], rows, hours)
        dist = S.haversine_km(origin, track[-1][:2])
        item = {"from_deg": from_deg, "from_es": geo.compass_es(from_deg),
                "to_deg": (from_deg + 180) % 360, "to_es": geo.compass_es(from_deg + 180),
                "distance_km": round(dist, 1), "reaches_coast": False, "closest_km": None, "closest_hour": None}
        if located and ctx.coast.available():
            ap = ctx.coast.approach(track)
            if ap:
                item.update({"reaches_coast": ap["reaches_coast"], "closest_km": ap["closest"]["distance_km"],
                             "closest_hour": ap["closest_hour"]})
        items.append(item)
    return {"wind_ms": wind_ms, "hours": hours, "located": located,
            "scale_assumed": bool((a.get("result") or {}).get("scale_assumed")),
            "scale_m_per_px": pm, "items": items}
