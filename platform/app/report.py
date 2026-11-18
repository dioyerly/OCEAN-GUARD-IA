"""Informe HTML de un análisis: resumen comprensible, imágenes, mapa y limitaciones.

Las imágenes van embebidas (base64). El mapa usa Leaflet desde cdnjs, por lo que necesita
internet para dibujarse; sin conexión, el informe muestra igualmente las coordenadas.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .config import DISCLAIMER_DETECTION, VAL_RELIABILITY


def _img(path: Path, caption: str) -> str:
    if not path.exists():
        return f"<p>{html.escape(caption)}: no disponible</p>"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _map_block(a: dict, folder: Path, sim: dict | None) -> str:
    state = a["summary"]["location"]["state"]
    bounds = a.get("bounds")
    if state == "none" or not bounds:
        return ("<div class='note'><strong>Sin ubicación geográfica confirmada.</strong> "
                "Esta escena no se muestra sobre un mapa para no sugerir una posición que no se conoce.</div>")
    slick = folder / "slick.png"
    slick_url = ("data:image/png;base64," + base64.b64encode(slick.read_bytes()).decode("ascii")) if slick.exists() else None
    sim_layer = None
    if sim and sim.get("frames"):
        sim_layer = {"url": "data:image/png;base64," + sim["frames"][-1]["png_b64"], "bounds": sim["bounds"],
                     "origin": sim.get("origin"), "end": (sim.get("drift") or {}).get("end")}
    cfg = json.dumps({"bounds": bounds, "slick": slick_url, "sim": sim_layer})
    warn = ("" if state == "georef" else
            "<p class='note'>Ubicación aproximada indicada por el usuario: no está verificada.</p>")
    return f"""{warn}
<div id="map" style="height:380px;border:1px solid #bcd;border-radius:8px"></div>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
(function () {{
  if (typeof L === 'undefined') {{ document.getElementById('map').innerHTML = '<p style="padding:1rem">El mapa necesita conexión a internet.</p>'; return; }}
  var cfg = {cfg};
  var map = L.map('map');
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{maxZoom: 18, attribution: 'Esri World Imagery'}}).addTo(map);
  L.control.scale().addTo(map);
  L.rectangle(cfg.bounds, {{color: '#00d1c1', weight: 2, fillOpacity: 0, dashArray: '6 4'}}).addTo(map);
  if (cfg.slick) L.imageOverlay(cfg.slick, cfg.bounds).addTo(map);
  if (cfg.sim) {{
    L.imageOverlay(cfg.sim.url, cfg.sim.bounds, {{opacity: 0.8}}).addTo(map);
    if (cfg.sim.origin && cfg.sim.end) L.polyline([cfg.sim.origin, cfg.sim.end], {{color: '#ffb703', weight: 3, dashArray: '8 6'}}).addTo(map);
  }}
  map.fitBounds(cfg.sim ? cfg.sim.bounds : cfg.bounds);
}})();
</script>"""


def build_report(a: dict, folder: Path, model_info: dict | None, sim: dict | None = None,
                 context: dict | None = None) -> str:
    s = a["summary"]
    result = a.get("result") or {}
    loc = s["location"]
    rows = [
        ("Identificador", a["id"]),
        ("Fecha del análisis (UTC)", a["created_at"]),
        ("Imagen analizada", a.get("filename") or "-"),
        ("Origen", a["source"]),
    ]
    if s.get("acquired"):
        rows.append(("Adquisición de la imagen", s["acquired"]))
    rows += [
        ("Superficie con posible derrame", f"{a['slick_pct']} % de la imagen"),
        ("Superficie estimada",
         f"{a['slick_area_km2']} km²" if a.get("slick_area_km2") is not None else "no calculada (sin ubicación ni resolución)"),
        ("Zonas detectadas", a.get("n_components")),
        ("Puntaje de confianza del modelo", s["confidence"]["title"].split(": ", 1)[1]),
        ("Ubicación", loc["text"]),
    ]
    table = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)

    ref = a.get("reference")
    ref_txt = ""
    if ref:
        ref_txt = (f"<h2>Comparación con la referencia de un experto</h2><p>"
                   f"Coincidencia de la zona marcada con la referencia: {round(ref['dice'] * 100)} % "
                   f"(detalle técnico: Dice {ref['dice']}, IoU {ref['iou']}).</p>")

    sim_txt = ""
    if sim:
        st = "".join(f"<tr><td>{x['hour']} h</td><td>{x['area_km2']} km²</td></tr>" for x in sim.get("stages", []))
        sim_txt = (f"<h2>Simulación de deriva (ilustrativa)</h2><p class='aviso'>{html.escape(sim['disclaimer'])}</p>"
                   f"<p>{html.escape(sim.get('narrative', ''))}</p>"
                   f"<table><tr><th>Instante</th><th>Área estimada</th></tr>{st}</table>")

    model_txt = ""
    if model_info:
        model_txt = (f"<p>Modelo: {html.escape(str(model_info.get('architecture')))} · época {model_info.get('epoch')} "
                     f"· coincidencia en validación (Dice) {model_info.get('val_dice')} con el umbral de entrenamiento.</p>")

    ctx_txt = ""
    if context:
        risk = context["risk"]
        wx = context.get("weather") or {}
        wx_line = (f"<p>Fuente del clima: {html.escape(wx.get('source', ''))}. "
                   f"{html.escape(' '.join(wx.get('notes', [])))}</p>") if wx.get("available") else (
            f"<p class='note'>{html.escape(wx.get('reason', 'Clima no disponible.'))}</p>")
        impact = "".join(f"<li>{html.escape(x)}</li>" for x in context.get("impact", []))
        recs = "".join(
            f"<li><strong>{html.escape(r['title'])}.</strong> {html.escape(r['text'])}"
            f"<br><em>Por qué: {html.escape(r['why'])}</em></li>" for r in context.get("recommendations", []))
        reasons = "".join(f"<li>{html.escape(x)}</li>" for x in risk["reasons"])
        ctx_txt = (f"<h2>Nivel de atención: {html.escape(risk['label'])} (orientativo)</h2><ul>{reasons}</ul>"
                   f"<h2>Qué podría verse afectado</h2><ul>{impact}</ul>{wx_line}"
                   f"<h2>Qué se puede hacer</h2><ol>{recs}</ol>"
                   f"<p class='aviso'>{html.escape(context.get('general_note', ''))}</p>")

    images = "".join([_img(folder / "image.png", "Imagen original"),
                      _img(folder / "mask.png", "Zonas marcadas (blanco)"),
                      _img(folder / "overlay.png", "Resultado resaltado")])
    limits = [
        "La detección es una estimación de un modelo de aprendizaje automático y no confirma la presencia de petróleo.",
        "Pueden confundirse con derrames otras manchas oscuras del mar (por ejemplo, zonas de calma o algas).",
        f"Fiabilidad medida en validación: ~{round(VAL_RELIABILITY['precision'] * 100)} % de lo marcado era derrame de referencia "
        f"y se encontró ~{round(VAL_RELIABILITY['recall'] * 100)} % del derrame de referencia. {VAL_RELIABILITY['note']}",
        "El modelo se entrenó con imágenes Sentinel-1 del Golfo de México; con otro tipo de imagen no está evaluado.",
        "La simulación, si existe, es ilustrativa y no incluye corrientes marinas, mareas ni datos meteorológicos.",
    ]
    limits_html = "".join(f"<li>{html.escape(x)}</li>" for x in limits)

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe {html.escape(a['id'])} · OCEAN GUARD AI</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; color: #12202e; background: #f4f8fb; }}
 header {{ background: linear-gradient(120deg,#06233b,#0b5d73); color: #fff; padding: 1.4rem 2rem; }}
 header h1 {{ margin: 0; font-size: 1.4rem; }} header p {{ margin: .3rem 0 0; opacity: .8; }}
 main {{ max-width: 980px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
 h2 {{ font-size: 1.15rem; margin-top: 2rem; color: #0b5d73; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }}
 th, td {{ border: 1px solid #d5e1ea; padding: .45rem .7rem; text-align: left; vertical-align: top; }}
 th {{ background: #eaf2f7; width: 34%; }}
 figure {{ display: inline-block; margin: .4rem; vertical-align: top; }}
 figure img {{ max-width: 300px; border: 1px solid #cdd9e2; border-radius: 6px; }}
 figcaption {{ font-size: .85rem; text-align: center; color: #52667a; }}
 .resumen {{ background: #fff; border-left: 5px solid #00b8a9; padding: 1rem 1.2rem; border-radius: 6px; }}
 .aviso, .note {{ background: #fff6df; border-left: 4px solid #e0a800; padding: .6rem .9rem; font-size: .92rem; }}
</style></head>
<body>
<header><h1>OCEAN GUARD AI · Informe de análisis</h1><p>Detección asistida de posibles derrames · demostración académica</p></header>
<main>
<div class="resumen"><h2 style="margin-top:0">{html.escape(s['headline'])}</h2><p>{html.escape(s['text'])}</p></div>
<h2>Imágenes</h2>{images}
<h2>Ubicación</h2>{_map_block(a, folder, sim)}
<h2>Datos del análisis</h2><table>{table}</table>
{ref_txt}
{sim_txt}
{ctx_txt}
<h2>Limitaciones</h2><ul>{limits_html}</ul>
<p class="aviso">{html.escape(DISCLAIMER_DETECTION)}</p>
{model_txt}
</main></body></html>"""
