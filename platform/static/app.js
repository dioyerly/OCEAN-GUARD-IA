/* OCEAN GUARD AI · Ocean Watch (JavaScript sin dependencias, salvo Leaflet para los mapas). */
"use strict";

const LEVEL_COLOR = { high: "#ff5a36", medium: "#ff9f1c", low: "#ffd166", none: "#6fe3d3" };
const LEVEL_LABEL = { high: "Señal clara", medium: "Señal moderada", low: "Señal débil", none: "Sin zonas marcadas" };
const RISK_COLOR = { low: "#2ee6d0", medium: "#ffc233", high: "#ff5a36" };
const COMPASS = ["norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste"];
const state = {
  maps: {}, mapGroups: null, dashGroup: null, dashLegend: null, analyses: [], selectedId: null, dashSel: null,
  resultMap: null, sim: null, simAnalysis: null, simFrameLayer: null, simGroup: null, simControls: [],
  simIndex: 0, simTimer: null, ctxCache: {}, simImg: null, simProj: null, simGroupMap: null, simPreselect: null,
};

// ---------- utilidades ----------
function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(v, d = 1) {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString("es", { maximumFractionDigits: d });
}
function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return esc(iso);
  return d.toLocaleDateString("es", { day: "numeric", month: "short", year: "numeric" }) + ", " +
    d.toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit" });
}
function fmtCoord(lat, lon) {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "O"}`;
}
const compassOf = (deg) => COMPASS[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
function utcLabel(iso) { return `${iso.slice(8, 10)}/${iso.slice(5, 7)} ${iso.slice(11, 16)}`; }
function showError(msg) {
  const b = document.getElementById("global-error");
  b.textContent = msg; b.hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function clearError() {
  const b = document.getElementById("global-error");
  b.hidden = true; b.textContent = "";
}
function setStatus(msg) {
  const b = document.getElementById("analyze-status");
  b.hidden = !msg; b.textContent = msg || "";
}
async function api(path, options = {}) {
  let r;
  try { r = await fetch(path, options); }
  catch (_) { throw new Error("No se pudo conectar con el servidor. Verifique que la plataforma esté en ejecución."); }
  if (!r.ok) {
    let detail = `Error ${r.status}`;
    try {
      const body = await r.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((x) => x.msg).join("; ");
    } catch (_) { /* sin JSON */ }
    throw new Error(detail);
  }
  return (r.headers.get("content-type") || "").includes("application/json") ? r.json() : r.text();
}
const locState = (a) => a.location_state || (a.summary && a.summary.location && a.summary.location.state) || "none";
const levelOf = (a) => a.level || (a.summary && a.summary.level) || "none";
function locChip(s) {
  if (s === "georef") return "";
  const m = { georef: ["georef", "Ubicación real del archivo"], manual: ["manual", "Ubicación aproximada"], none: ["none", "Sin ubicación confirmada"] }[s] || ["none", "Sin ubicación confirmada"];
  return `<span class="chip ${m[0]}">${m[1]}</span>`;
}
const levelChip = (l) => `<span class="chip ${esc(l)}">${esc(LEVEL_LABEL[l] || "")}</span>`;
function getContext(id) {
  if (!state.ctxCache[id]) {
    state.ctxCache[id] = api(`/api/analyses/${encodeURIComponent(id)}/context`).catch((e) => { delete state.ctxCache[id]; throw e; });
  }
  return state.ctxCache[id];
}

// ---------- navegación ----------
function setView(name) {
  stopSim();
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  clearError();
  if (name === "dashboard") loadDashboard();
  if (name === "map") loadMap();
  if (name === "simulate") loadSimulateOptions();
  if (name === "history") loadHistory();
  setTimeout(() => Object.values(state.maps).forEach((m) => m.invalidateSize && m.invalidateSize()), 120);
}

// ---------- mapas ----------
function baseLayers() {
  // Esri World Imagery solo tiene imagen real hasta el zoom ~11 sobre mar abierto: más allá se amplía (maxNativeZoom).
  const esri = "https://server.arcgisonline.com/ArcGIS/rest/services/";
  const sat = L.tileLayer(esri + "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 13, maxNativeZoom: 10, attribution: "Imágenes © Esri, Maxar, Earthstar Geographics" });
  const labels = L.tileLayer(esri + "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 13, maxNativeZoom: 10, attribution: "Referencias © Esri" });
  const dark = L.tileLayer(esri + "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 13, maxNativeZoom: 12, attribution: "Mapa base © Esri, HERE, Garmin, OpenStreetMap" });
  const darkLabels = L.tileLayer(esri + "Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 13, maxNativeZoom: 12, attribution: "Referencias © Esri" });
  return { sat: L.layerGroup([sat, labels]), dark: L.layerGroup([dark, darkLabels]) };
}
function addNorth(map) {
  const C = L.Control.extend({
    options: { position: "topleft" },
    onAdd() {
      const d = L.DomUtil.create("div", "north-arrow");
      d.innerHTML = '<svg viewBox="0 0 24 30" aria-hidden="true"><path d="M12 1 21 25 12 19.5 3 25z" fill="#2ee6d0" stroke="#fff" stroke-width="1"/></svg><span>N</span>';
      d.title = "Norte";
      return d;
    },
  });
  new C().addTo(map);
}
function addCoords(map) {
  let el = null;
  const C = L.Control.extend({
    options: { position: "bottomright" },
    onAdd() { el = L.DomUtil.create("div", "coords"); el.textContent = "Mueva el cursor sobre el mapa"; return el; },
  });
  new C().addTo(map);
  map.on("mousemove", (e) => { if (el) el.textContent = fmtCoord(e.latlng.lat, e.latlng.lng); });
}
function addHtmlControl(map, position, className, html) {
  const C = L.Control.extend({
    options: { position },
    onAdd() { const d = L.DomUtil.create("div", className); d.innerHTML = html; return d; },
  });
  const c = new C();
  c.addTo(map);
  return c;
}
function makeMap(el, view) {
  const map = L.map(el, { minZoom: 2, maxZoom: 13, worldCopyJump: true });
  const b = baseLayers();
  b.sat.addTo(map);
  L.control.layers({ "Satélite y costas": b.sat, "Mapa oscuro": b.dark }, null, { position: "topright" }).addTo(map);
  L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);
  addNorth(map);
  addCoords(map);
  map.setView(view || [20, -40], view ? 8 : 2);
  return map;
}
function ensureMap(id) {
  if (!state.maps[id]) state.maps[id] = makeMap(id);
  return state.maps[id];
}
function legendHtml(rows) {
  return `<b>LEYENDA</b>${rows.map(([cls, text]) => `<div><i class="${cls}"></i>${esc(text)}</div>`).join("")}`;
}

function drawEvent(groups, a, popup = true, onSelect = null) {
  const s = locState(a);
  if (!a.bounds) return null;
  const color = s === "georef" ? "#2ee6d0" : "#ffc233";
  if (groups.slick) L.imageOverlay(`/api/analyses/${a.id}/file/slick`, a.bounds, { opacity: 1 }).addTo(groups.slick);
  const rect = L.rectangle(a.bounds, { color, weight: 2, dashArray: "6 5", fillOpacity: 0.04 });
  if (groups.foot) rect.addTo(groups.foot);
  const center = [(a.bounds[0][0] + a.bounds[1][0]) / 2, (a.bounds[0][1] + a.bounds[1][1]) / 2];
  const marker = L.circleMarker(center, { radius: 8, color: "#fff", weight: 2, fillColor: LEVEL_COLOR[levelOf(a)] || "#ff6a3d", fillOpacity: 0.95 });
  if (popup) {
    marker.bindPopup(`<strong>${esc(a.headline || "Posible mancha")}</strong><br>
      ${esc(a.filename || a.id)} · ${fmtDate(a.created_at)}<br>${locChip(s)}<br>
      <span class="muted small">Superficie marcada: ${a.slick_area_km2 != null ? fmt(a.slick_area_km2, 1) + " km²" : fmt(a.slick_pct, 2) + " % de la imagen"}</span><br>
      <button class="primary small-btn" style="margin-top:.5rem" onclick="openAnalysis('${esc(a.id)}')">Ver análisis</button>`);
  }
  marker.on("click", () => (onSelect ? onSelect(a.id) : selectEvent(a.id, false)));
  marker.addTo(groups.marker || groups.foot || groups.slick);
  return { rect, marker, center };
}

// ---------- contexto: clima, impacto y qué hacer ----------
function windArrowSvg(toDeg, color = "#ffb703") {
  return `<svg viewBox="0 0 44 44" aria-hidden="true"><circle cx="22" cy="22" r="20" fill="#061b2e" stroke="#2a5a7c"/>
    <g transform="rotate(${toDeg} 22 22)"><path d="M22 6 30 30 22 25 14 30z" fill="${color}"/></g></svg>`;
}

function windLine(st) {
  if (!st) return "";
  const v = st.wind_mean_ms, lvl = v < 3 ? "suave" : v < 10 ? "moderado" : "fuerte";
  return `Viento: ${fmt(v, 0)} m/s del ${compassOf(st.wind_from_mean_deg)} (${lvl})`;
}
function contextHtml(c) {
  const wx = c.weather || {}, st = c.stats;
  const wind = wx.available
    ? `<p>${esc(windLine(st))}.</p>${(wx.notes || []).map((n) => `<p class="muted small">ℹ ${esc(n)}</p>`).join("")}`
    : `<div class="notice">${esc(wx.reason || "Clima no disponible.")}</div>`;
  const items = c.impact.filter((x) => !/^(Viento medio|Con ese viento)/.test(x));
  const impact = items.length ? `<ul class="impact">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  const steps = `<h3>Qué se puede hacer</h3><ol class="steps3"><li>Confirmar con otra fuente (imagen óptica, sobrevuelo o reporte en el lugar).</li>
    <li>Avisar a la autoridad marítima o ambiental con las coordenadas y la imagen.</li><li>Volver a revisar la zona en 6 a 12 horas.</li></ol>
    <p class="small muted">No se acerque sin equipo y personal capacitados.</p>`;
  return `<div class="ctx">${wind}${impact}${steps}</div>`;
}

async function loadAnalysisContext(a) {
  const box = document.getElementById("ctx-box");
  if (!box) return;
  try {
    const c = await getContext(a.id);
    box.innerHTML = contextHtml(c);
  } catch (err) {
    box.innerHTML = `<div class="notice">No se pudo calcular el contexto: ${esc(err.message)}</div>`;
  }
}

// ---------- Ocean Watch (dashboard) ----------
async function loadStatus() {
  const badge = document.getElementById("model-badge");
  try {
    const s = await api("/api/status");
    document.getElementById("disclaimer").textContent = s.disclaimer;
    if (s.model_loaded) { badge.className = "badge ok"; badge.textContent = "● Modelo listo"; }
    else {
      badge.className = "badge bad"; badge.textContent = "● Modelo no disponible";
      showError(`El modelo no está disponible: ${s.model_error || "error desconocido"}`);
    }
  } catch (err) {
    badge.className = "badge bad"; badge.textContent = "● Servidor no disponible";
    showError(err.message);
  }
}

function activitySvg(days) {
  const max = Math.max(1, ...days.map((d) => d.count));
  const w = 280, h = 80, bw = w / days.length;
  const bars = days.map((d, i) => {
    const bh = Math.max(d.count ? 6 : 2, (d.count / max) * (h - 14));
    return `<rect x="${(i * bw + 2).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${bh.toFixed(1)}" rx="3" opacity="${d.count ? 1 : 0.25}"><title>${esc(d.date)}: ${d.count}</title></rect>`;
  }).join("");
  return `<svg class="bars" viewBox="0 0 ${w} ${h + 12}" role="img" aria-label="Análisis por día">${bars}
    <text x="0" y="${h + 10}">${esc(days[0].date.slice(5))}</text><text x="${w}" y="${h + 10}" text-anchor="end">hoy</text></svg>`;
}

function eventCard(a, extra = "") {
  return `<button class="event" data-id="${esc(a.id)}" onclick="${extra || `openAnalysis('${esc(a.id)}')`}">
    <img src="/api/analyses/${esc(a.id)}/file/overlay" alt="Imagen analizada con zonas marcadas" loading="lazy">
    <div><div class="t">${esc(a.headline || "Análisis")}</div>
      <div class="m">${fmtDate(a.created_at)}${a.acquired ? " · " + esc(a.acquired.split(" · ")[1] || "") : ""}</div>
      <div>${locChip(locState(a))}${levelChip(levelOf(a))}</div></div></button>`;
}
function ribbonCard(a) {
  return `<button class="rcard" data-id="${esc(a.id)}" onclick="selectDashEvent('${esc(a.id)}', true)">
    <img src="/api/analyses/${esc(a.id)}/file/overlay" alt="Imagen analizada" loading="lazy">
    <div><div class="t"><i class="lvl" style="background:${LEVEL_COLOR[levelOf(a)]}"></i>${esc(a.filename || a.id)}</div>
      <div class="m">${fmtDate(a.created_at)}</div><div>${locChip(locState(a))}</div></div></button>`;
}

async function loadDashboard() {
  await loadStatus();
  try {
    const d = await api("/api/dashboard");
    const c = d.counts;
    const list = await api("/api/analyses");
    state.analyses = list.items;
    const located = list.items.filter((a) => a.bounds);

    // titular en lenguaje natural
    const head = document.getElementById("dash-headline"), sub = document.getElementById("dash-subline");
    if (!list.items.length) {
      head.textContent = "Aún no hay eventos en observación";
      sub.textContent = "Analice una imagen para empezar.";
    } else {
      const flagged = list.items.filter((a) => a.slick_pct > 0).length;
      head.textContent = flagged ? `${flagged} ${flagged === 1 ? "evento" : "eventos"} en observación` : "Sin zonas sospechosas por ahora";
      sub.textContent = `Último análisis: ${fmtDate(c.last_analysis_at)}. ` +
        (c.total_slick_area_km2 ? `En conjunto, unos ${fmt(c.total_slick_area_km2, 1)} km² marcados.` : "");
    }

    const kpis = [
      ["var(--blue)", fmt(c.analyses, 0), "", "Imágenes analizadas", "Cada imagen de satélite que el modelo ha revisado."],
      ["var(--slick)", fmt(c.with_slick, 0), "", "Con posible derrame", "Imágenes donde el modelo marcó alguna zona. No son confirmaciones."],
      ["var(--turq)", fmt(c.total_slick_area_km2, 1), " km²", "Superficie marcada", "Suma estimada; solo cuenta imágenes con ubicación y escala."],
      ["var(--sim)", fmt(c.simulations, 0), "", "Simulaciones", "Estimaciones de hacia dónde podría moverse una mancha."],
    ];
    document.getElementById("dash-cards").innerHTML = kpis.map(([col, n, u, l, e]) =>
      `<div class="kpi" style="--c:${col}"><div class="n">${n}<small>${u}</small></div><div class="l">${esc(l)}</div><div class="e">${esc(e)}</div></div>`).join("");
    document.getElementById("dash-activity").innerHTML = activitySvg(d.activity);

    const m = d.model, r = d.reliability;
    document.getElementById("dash-model").innerHTML = m ? `
      <p><b style="color:var(--turq)">Listo para analizar.</b> En pruebas, de cada 100 puntos que marcó como derrame, unos
        <b>${Math.round(r.precision * 100)}</b> lo eran, y encontró unos <b>${Math.round(r.recall * 100)}</b> de cada 100 puntos de derrame reales.</p>
      <details><summary>Detalles técnicos</summary><dl class="kv">
        <dt>Arquitectura</dt><dd>${esc(m.architecture)}</dd><dt>Entrenamiento</dt><dd>época ${esc(m.epoch)} (fase 1)</dd>
        <dt>Dice de validación</dt><dd>${fmt(m.val_dice, 4)} (umbral de entrenamiento)</dd><dt>Pesos verificados</dt><dd>${m.weights_sha256_ok ? "sí" : "no"}</dd>
        <dt>Dispositivo</dt><dd>${esc(m.device)}</dd></dl></details>` : "<p class=\"muted\">Modelo no cargado.</p>";

    drawDashMap(located);
    const ribbon = document.getElementById("dash-ribbon");
    ribbon.innerHTML = list.items.slice(0, 12).map(ribbonCard).join("");
    const first = located.find((a) => a.slick_pct > 0) || located[0];
    if (first) selectDashEvent(state.dashSel && located.find((a) => a.id === state.dashSel) ? state.dashSel : first.id, false);
    else document.getElementById("dash-focus").innerHTML = `<div class="focus-title">Nada seleccionado</div>
      <p class="muted small">Cuando haya un evento con ubicación, aquí verá su nivel de atención, el clima y qué hacer.</p>`;
  } catch (err) { showError(err.message); }
}

function drawDashMap(located) {
  const map = ensureMap("dash-map");
  if (state.dashGroup) state.dashGroup.clearLayers(); else state.dashGroup = L.layerGroup().addTo(map);
  const groups = { slick: state.dashGroup, foot: state.dashGroup, marker: state.dashGroup };
  located.forEach((a) => drawEvent(groups, a, true, (id) => selectDashEvent(id, false)));
  const empty = document.getElementById("dash-map-empty");
  if (located.length) {
    empty.hidden = true;
    map.fitBounds(located.map((a) => a.bounds).flat(), { padding: [40, 40], maxZoom: 9 });
    document.getElementById("dash-legend").innerHTML = [
      ["sw slick", "Posible mancha (estimación del modelo)"], ["sw foot", "Imagen con ubicación real"],
      ["sw foot-m", "Imagen con ubicación aproximada"], ["pin", "Evento"]].map(([c, t]) => `<span><i class="${c}"></i>${esc(t)}</span>`).join("");
  } else {
    empty.hidden = false;
    empty.innerHTML = "<div><b>Aún no hay eventos con ubicación.</b><p>Analice una imagen con ubicación para verlo aquí.</p></div>";
  }
}

async function selectDashEvent(id, fly) {
  state.dashSel = id;
  document.querySelectorAll("#dash-ribbon .rcard").forEach((e) => e.classList.toggle("sel", e.dataset.id === id));
  const a = state.analyses.find((x) => x.id === id);
  const box = document.getElementById("dash-focus");
  if (!a || !box) return;
  if (fly && a.bounds && state.maps["dash-map"]) state.maps["dash-map"].fitBounds(a.bounds, { padding: [60, 60], maxZoom: 11 });
  const head = `<div class="focus-title">${esc(a.headline)}</div>
    <div class="focus-sub">${esc(a.filename || a.id)} · ${fmtDate(a.created_at)}</div>
    <div>${locChip(locState(a))}${levelChip(levelOf(a))}</div>`;
  if (!a.bounds || !(a.slick_pct > 0)) {
    box.innerHTML = `${head}<p class="muted small" style="margin-top:.6rem">${!a.bounds
      ? "Esta imagen no tiene ubicación confirmada: no se consulta clima ni costa. Puede indicar dónde está o ver un escenario de deriva sobre la imagen."
      : "No hay zona marcada que evaluar."}</p><div class="actions"><button class="primary small-btn" onclick="openAnalysis('${esc(a.id)}')">Ver análisis</button>${a.slick_pct > 0 ? `<button class="secondary small-btn" onclick="simulateFrom('${esc(a.id)}')">Escenario de deriva</button>` : ""}</div>`;
    return;
  }
  box.innerHTML = `${head}<p class="skeleton">Consultando clima y costa…</p>`;
  try {
    const c = await getContext(id);
    if (state.dashSel !== id) return;
    const st = c.stats, wx = c.weather || {};
    const wind = st ? `<div class="wxline">${windArrowSvg((st.wind_from_mean_deg + 180) % 360)}<div><b>${fmt(st.wind_mean_ms, 0)} m/s desde el ${compassOf(st.wind_from_mean_deg)}</b>
      <span>${wx.mode === "historical" ? "Viento real de esa fecha" : wx.mode === "forecast" ? "Pronóstico próximas horas" : "Viento indicado"}${st.wave_max_m != null ? " · olas " + fmt(st.wave_max_m, 1) + " m" : ""}</span></div></div>`
      : `<p class="muted small">${esc(wx.reason || "Clima no disponible.")}</p>`;
    const coastTxt = c.coast ? `<p class="small" style="margin:.4rem 0">Costa más cercana: <b>${fmt(c.coast.start.distance_km, 0)} km</b> al ${esc(c.coast.start.direction_es)}.</p>` : "";
    box.innerHTML = `${head}
      <div class="gauge">${gaugeSvg(c.risk.level)}</div>
      <div class="gauge-label" style="color:${RISK_COLOR[c.risk.level]}">Atención: ${esc(c.risk.label)}<small>indicador orientativo</small></div>
      ${wind}${coastTxt}
      <ul class="reasons">${c.risk.reasons.slice(0, 3).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    <div class="actions"><button class="primary small-btn" onclick="openAnalysis('${esc(a.id)}')">Qué hacer y por qué</button>
        <button class="secondary small-btn" onclick="simulateFrom('${esc(a.id)}')">Simular</button></div>`;
  } catch (err) {
    if (state.dashSel === id) box.innerHTML = `${head}<div class="notice">No se pudo consultar el contexto: ${esc(err.message)}</div>`;
  }
}

// ---------- análisis ----------

// Franja de resumen (semáforo + frase + área) sobre el resultado; la regla del semáforo está en el backend (narrative.TRAFFIC_RULES).
const TRAFFIC_ICON = { green: "✓", yellow: "!", red: "!" };
function summaryStrip(s) {
  const t = s.traffic;
  if (!t) return "";
  return `<div class="sum-strip ${esc(t.color)}" role="status"><span class="lamp">${TRAFFIC_ICON[t.color]}</span>
    <div><b>${esc(t.label)}</b>${t.area_km2 != null ? ` · ${fmt(t.area_km2, 1)} km²` : ""}<br><span class="small">${esc(t.phrase)}</span></div></div>`;
}

function renderAnalysis(a) {
  const s = a.summary, ref = a.reference, conf = s.confidence, loc = s.location;
  const icon = s.level === "none" ? "✓" : "◉";
  const area = a.slick_area_km2 != null ? `${fmt(a.slick_area_km2, 1)} km²` : `${fmt(a.slick_pct, 2)} %`;
  const areaExtra = a.slick_area_km2 != null ? `Aproximadamente el ${fmt(a.slick_pct, 2)} % de la imagen.` : "No se puede pasar a km² sin ubicación ni escala conocidas.";
  const matchTxt = ref ? `<p>La zona marcada por el modelo coincide en un <b>${Math.round(ref.dice * 100)} %</b>
      con la que dibujó un experto.</p><p class="muted small">Detalle técnico: Dice ${fmt(ref.dice, 4)}, IoU ${fmt(ref.iou, 4)}.</p>` : "";
  const ctxBlock = (a.bounds && a.slick_pct > 0
    ? `<details class="card ctx-card fold"><summary><h2>Qué podría verse afectado y qué hacer</h2></summary><div id="ctx-box"><p class="skeleton">Consultando clima y costa reales…</p></div></details>`
    : "") + (a.slick_pct > 0 ? scenarioCard(a) : "");
  const locBlock = !a.bounds && a.slick_pct > 0 ? locationCard() : "";
  return `
    ${summaryStrip(s)}
    <div class="verdict ${esc(s.level)}">
      <div class="ico">${icon}</div>
      <div>
        <h2>${esc(s.headline)}</h2>
        <p>${esc(s.text)}</p>
        <div>${locChip(loc.state)}${s.acquired ? `<span class="chip none">${esc(s.acquired)}</span>` : ""}</div>
      </div>
    </div>
    <div class="result-grid">
      <div>
        <div class="compare" id="cmp" style="--pos:50%">
          <img src="/api/analyses/${esc(a.id)}/file/image" alt="Imagen satelital original">
          <div class="top"><img src="/api/analyses/${esc(a.id)}/file/overlay" alt="Imagen con las zonas marcadas resaltadas"></div>
          <div class="bar"></div>
          <span class="tag r">Con zonas marcadas</span><span class="tag l">Imagen original</span>
          <input type="range" min="0" max="100" value="50" aria-label="Comparar antes y después"
                 oninput="document.getElementById('cmp').style.setProperty('--pos', this.value + '%')">
        </div>
        <p class="muted small">Deslice para comparar la imagen original con el resultado. En rojo, las zonas que el modelo considera posible derrame; en amarillo, sus bordes.</p>
      </div>
      <div class="facts">
        <div class="fact"><div class="n">${area}</div><div class="l">Superficie estimada</div><div class="e">${areaExtra}</div></div>
        <div class="fact"><div class="n">${fmt(a.n_components, 0)}</div><div class="l">Zonas detectadas</div><div class="e">Manchas separadas que el modelo marcó.</div></div>
        <div class="fact wide"><div class="n">Confianza: ${esc(conf.label)}</div><div class="l">Qué tan fiable es este resultado</div><div class="e">${esc(conf.text)}</div></div>
        ${loc.state !== "georef" ? `<div class="fact wide"><div class="n" style="font-size:1.05rem">${locChip(loc.state)}</div><div class="e">${esc(loc.text)}</div></div>` : ""}
      </div>
    </div>
    ${a.bounds ? `<div class="mini-map"><h3>Dónde se encuentra</h3><div id="result-map" class="map"></div></div>`
      : `<div class="notice"><b>Sin ubicación geográfica confirmada.</b><br>No se muestra sobre un mapa para no sugerir una posición que no se conoce; por eso todavía no se consulta el clima real ni la costa.
         Abajo puede indicar dónde está, o ver hacia dónde se desplazaría con el viento que usted elija.</div>`}
    ${locBlock}
    ${ctxBlock}
    <details class="tech card"><summary>Detalles técnicos</summary>
      ${matchTxt}
      <dl class="kv"><dt>Tamaño</dt><dd>${a.width} × ${a.height} px</dd><dt>Umbral de detección</dt><dd>${fmt(a.threshold, 2)}</dd>
        <dt>Identificador</dt><dd>${esc(a.id)}</dd><dt>Origen</dt><dd>${esc(a.source)}</dd></dl>
    </details>
    ${a.slick_pct > 0 ? "" : `<p class="muted">Sin derrame detectado: no hay trayectoria que simular.</p>`}
    <p class="disclaimer">Detección asistida por IA: requiere validación humana. No sustituye vigilancia oficial.</p>
    <div class="actions">
      <a href="/api/analyses/${esc(a.id)}/report" target="_blank" rel="noopener"><button class="secondary" type="button">Abrir informe</button></a>
      ${a.slick_pct > 0 ? `<button class="secondary" type="button" onclick="simulateFrom('${esc(a.id)}')">Simular desplazamiento</button>` : ""}
      <button class="secondary" type="button" onclick="showOnMap('${esc(a.id)}')" ${a.bounds ? "" : "disabled"}>Ver en el mapa</button>
    </div>`;
}

function mountResultMap(a) {
  if (state.resultMap) { state.resultMap.remove(); state.resultMap = null; }
  const el = document.getElementById("result-map");
  if (!el || !a.bounds) return;
  const map = makeMap(el);
  state.resultMap = map;
  const g = L.layerGroup().addTo(map);
  drawEvent({ slick: g, foot: g, marker: g }, { ...a, headline: a.summary.headline, location_state: a.summary.location.state, level: a.summary.level }, false);
  map.fitBounds(a.bounds, { padding: [40, 40] });
  addHtmlControl(map, "bottomleft", "legend", legendHtml([
    ["sw slick", "Zona marcada por el modelo"], [a.summary.location.state === "georef" ? "sw foot" : "sw foot-m", "Contorno de la imagen"]]));
}

function showAnalysis(a) {
  const box = document.getElementById("analyze-result");
  box.innerHTML = renderAnalysis(a);
  setTimeout(() => mountResultMap(a), 40);
  loadAnalysisContext(a);
  wirePlaceSearch(box);
  const lg = document.getElementById("loc-go");
  if (lg) lg.addEventListener("click", () => submitLocation(a));
  if (a.slick_pct > 0) {
    ["scen-wind", "scen-hours"].forEach((id) => { const el = document.getElementById(id); if (el) el.addEventListener("change", () => loadScenarios(a)); });
    loadScenarios(a);
  }
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function openAnalysis(id) {
  setView("analyze");
  setStatus("Cargando análisis…");
  try { showAnalysis(await api(`/api/analyses/${encodeURIComponent(id)}`)); setStatus(""); }
  catch (err) { setStatus(""); showError(err.message); }
}

// ---------- mapa ----------
function mapGroupsEnsure() {
  const map = ensureMap("map");
  if (!state.mapGroups) {
    state.mapGroups = { slick: L.layerGroup(), foot: L.layerGroup(), sim: L.layerGroup(), marker: L.layerGroup() };
    Object.values(state.mapGroups).forEach((g) => g.addTo(map));
    addHtmlControl(map, "bottomleft", "legend", legendHtml([
      ["sw slick", "Zona marcada por el modelo"], ["sw foot", "Imagen con ubicación real"], ["sw foot-m", "Ubicación aproximada"],
      ["sw sim", "Simulación (estimada)"], ["pin", "Evento"]]));
    [["lyr-slick", "slick"], ["lyr-foot", "foot"], ["lyr-sim", "sim"]].forEach(([id, key]) => {
      document.getElementById(id).addEventListener("change", (e) => {
        if (e.target.checked) state.mapGroups[key].addTo(map); else map.removeLayer(state.mapGroups[key]);
      });
    });
  }
  return map;
}

async function loadMap() {
  const map = mapGroupsEnsure();
  Object.values(state.mapGroups).forEach((g) => g.clearLayers());
  try {
    const data = await api("/api/analyses");
    state.analyses = data.items;
    const located = data.items.filter((a) => a.bounds);
    located.forEach((a) => drawEvent(state.mapGroups, a));
    document.getElementById("map-list").innerHTML = located.length
      ? located.map((a) => eventCard(a, `selectEvent('${esc(a.id)}', true)`)).join("")
      : "<p class=\"muted\">No hay análisis con ubicación. Indique latitud y longitud al cargar una imagen.</p>";
    if (located.length) map.fitBounds(located.map((a) => a.bounds).flat(), { padding: [60, 60], maxZoom: 10 });
    if (state.selectedId) selectEvent(state.selectedId, false);
  } catch (err) { showError(err.message); }
}

async function selectEvent(id, fly) {
  state.selectedId = id;
  document.querySelectorAll("#map-list .event").forEach((e) => e.classList.toggle("sel", e.dataset.id === id));
  const a = state.analyses.find((x) => x.id === id);
  if (!a) return;
  const map = state.maps["map"];
  if (fly && map && a.bounds) map.fitBounds(a.bounds, { padding: [60, 60], maxZoom: 11 });
  const box = document.getElementById("map-detail");
  if (!box) return;
  box.innerHTML = `<h3>${esc(a.headline)}</h3>
    <p class="muted small">${fmtDate(a.created_at)} · ${esc(a.filename || "")}</p>
    <div>${locChip(locState(a))}${levelChip(levelOf(a))}</div>
    <dl class="kv"><dt>Superficie</dt><dd>${a.slick_area_km2 != null ? fmt(a.slick_area_km2, 1) + " km²" : fmt(a.slick_pct, 2) + " % de la imagen"}</dd>
      <dt>Zonas</dt><dd>${fmt(a.n_components, 0)}</dd>
      <dt>Coordenadas</dt><dd>${a.center_lat != null ? esc(fmtCoord(a.center_lat, a.center_lon)) : "—"}</dd>
      ${a.acquired ? `<dt>Imagen</dt><dd>${esc(a.acquired)}</dd>` : ""}</dl>
    <div class="actions"><button class="primary small-btn" onclick="openAnalysis('${esc(a.id)}')">Ver análisis</button>
      <button class="secondary small-btn" onclick="simulateFrom('${esc(a.id)}')" ${a.slick_pct > 0 ? "" : "disabled"}>Simular</button></div>
    <p class="muted small" id="map-sim-note"></p>`;
  if (state.mapGroups) {
    state.mapGroups.sim.clearLayers();
    try {
      const sim = await api(`/api/analyses/${encodeURIComponent(id)}/simulation`);
      const last = sim.frames[sim.frames.length - 1];
      L.imageOverlay(`data:image/png;base64,${last.png_b64}`, sim.bounds, { opacity: 0.85 }).addTo(state.mapGroups.sim);
      drawTrack(state.mapGroups.sim, sim);
      document.getElementById("map-sim-note").textContent = `Se muestra la última simulación (${fmt(sim.params.hours, 0)} h).`;
    } catch (_) { /* sin simulación: es normal */ }
  }
}

function showOnMap(id) { state.selectedId = id; setView("map"); }

// ---------- simulación ----------
function arrowIcon(bearing) {
  return L.divIcon({ className: "arrow-icon", html: `<div style="transform:rotate(${bearing}deg)">▲</div>`, iconSize: [26, 26], iconAnchor: [13, 13] });
}
function drawTrack(group, sim, proj = null) {
  const P = proj ? proj.pt : (p) => p;
  const track = (sim.track || []).map((p) => P([p[0], p[1]]));
  if (track.length > 1) L.polyline(track, { color: "#ffb703", weight: 3, dashArray: "8 6" }).addTo(group);
  (sim.track || []).forEach((p) => {
    if (p[2] > 0 && p[2] % 6 === 0) {
      L.circleMarker(P([p[0], p[1]]), { radius: 4, color: "#ffb703", weight: 2, fillColor: "#04121f", fillOpacity: 1 })
        .bindTooltip(`${p[2]} h`, { permanent: false }).addTo(group);
    }
  });
  if (sim.drift && sim.drift.end && sim.drift.bearing_to_deg != null) {
    L.marker(P(sim.drift.end), { icon: arrowIcon(sim.drift.bearing_to_deg), interactive: false }).addTo(group);
  }
}
function updateCompass() {
  const from = Number(document.getElementById("sim-dir").value);
  const to = (from + 180) % 360;
  document.getElementById("sim-compass").innerHTML = `<svg viewBox="0 0 120 120" aria-label="Dirección del viento">
    <circle cx="60" cy="60" r="54" fill="#061b2e" stroke="#2a5a7c"/>
    <text x="60" y="16" text-anchor="middle" fill="#8fb0c8" font-size="11">N</text><text x="60" y="114" text-anchor="middle" fill="#8fb0c8" font-size="11">S</text>
    <text x="108" y="64" text-anchor="middle" fill="#8fb0c8" font-size="11">E</text><text x="12" y="64" text-anchor="middle" fill="#8fb0c8" font-size="11">O</text>
    <g transform="rotate(${to} 60 60)"><path d="M60 22 72 62 60 54 48 62z" fill="#ffb703"/><rect x="57" y="54" width="6" height="30" rx="3" fill="#ffb703" opacity=".7"/></g></svg>
    <div class="muted small" style="text-align:center">El aceite se desplazaría hacia este lado</div>`;
}

async function loadSimulateOptions(preselect = state.simPreselect) {
  state.simPreselect = null;
  const sel = document.getElementById("sim-analysis");
  try {
    const data = await api("/api/analyses");
    state.analyses = data.items;
    const usable = data.items.filter((a) => a.slick_pct > 0);
    ensureMap("sim-map");
    updateCompass();
    if (!usable.length) {
      sel.innerHTML = "";
      document.getElementById("sim-summary").innerHTML = "<p class=\"muted\">No hay eventos con zona marcada. Suba su propia imagen.</p>";
      return;
    }
    sel.innerHTML = usable.map((a) => `<option value="${esc(a.id)}">${esc(a.filename || a.id)}${a.bounds ? "" : " · sin ubicación"} · ${fmtDate(a.created_at)}</option>`).join("");
    if (preselect) sel.value = preselect;
    await onSimEventChange();
  } catch (err) { showError(err.message); }
}

const sp = (p) => (state.simProj ? state.simProj.pt(p) : p);
const sb = (b) => (state.simProj ? state.simProj.bb(b) : b);
function projFrom(fp, W, H) {
  const [[south, west], [north, east]] = fp;
  const pt = ([la, lo]) => [((la - south) / (north - south)) * H, ((lo - west) / (east - west)) * W];
  return { pt, bb: ([a, b]) => [pt(a), pt(b)] };
}
function currentSimMap() { return state.simImg ? state.maps["sim-img"] : ensureMap("sim-map"); }

function simGroup() {
  const map = currentSimMap();
  if (!state.simGroup || state.simGroupMap !== map) { state.simGroup = L.layerGroup().addTo(map); state.simGroupMap = map; }
  state.simGroup.clearLayers();
  state.simControls.forEach((c) => c.remove && c.remove());
  state.simControls = [];
  state.simFrameLayer = null;
  return state.simGroup;
}

async function onSimEventChange() {
  stopSim();
  const id = document.getElementById("sim-analysis").value;
  const a = state.analyses.find((x) => x.id === id);
  document.getElementById("sim-controls").hidden = true;
  document.getElementById("sim-fit").hidden = true;
  document.getElementById("sim-context").innerHTML = "";
  document.getElementById("sim-simple").innerHTML = "";
  if (!a) return;
  state.simAnalysis = a;
  state.simImg = a.bounds ? null : { W: a.width, H: a.height };
  state.simProj = null;
  document.getElementById("sim-map").hidden = !!state.simImg;
  document.getElementById("sim-map-img").hidden = !state.simImg;
  const wRadio = document.querySelector("#form-sim input[name=mode][value=weather]");
  const mRadio = document.querySelector("#form-sim input[name=mode][value=manual]");
  wRadio.disabled = !!state.simImg;
  if (state.simImg) { mRadio.checked = true; document.getElementById("manual-fields").hidden = false; }
  let map;
  if (state.simImg) {
    if (state.maps["sim-img"]) { state.maps["sim-img"].remove(); delete state.maps["sim-img"]; }
    state.simGroup = null;
    map = L.map("sim-map-img", { crs: L.CRS.Simple, minZoom: -3, maxZoom: 3, zoomSnap: 0.25, attributionControl: false });
    state.maps["sim-img"] = map;
    const bb = [[0, 0], [a.height, a.width]];
    L.imageOverlay(`/api/analyses/${a.id}/file/image`, bb).addTo(map);
    L.imageOverlay(`/api/analyses/${a.id}/file/slick`, bb).addTo(map);
    map.fitBounds(bb);
    simGroup();
    state.simControls.push(addHtmlControl(map, "bottomleft", "legend", legendHtml([
      ["sw slick", "Mancha analizada"], ["pin", "Escenario sobre la imagen (sin ubicación geográfica)"]])));
    document.getElementById("sim-summary").innerHTML = "<p class=\"muted\">Este evento no tiene ubicación: la simulación se muestra sobre la propia imagen y usa el viento que usted elija (no se consulta el clima real). Indique dónde está en la sección Análisis para usar el clima real.</p>";
  } else {
    const g = simGroup();
    map = state.maps["sim-map"];
    L.imageOverlay(`/api/analyses/${a.id}/file/image`, a.bounds, { opacity: 0.9 }).addTo(g);
    drawEvent({ slick: g, foot: g, marker: g }, a, false);
    map.fitBounds(a.bounds, { padding: [60, 60], maxZoom: 12 });
    state.simControls.push(addHtmlControl(map, "bottomleft", "legend", legendHtml([
      ["sw slick", "Mancha detectada hoy"], ["sw foot", "Contorno de la imagen"]])));
    document.getElementById("sim-summary").innerHTML = "<p class=\"muted\">Elija las condiciones y pulse «Simular desplazamiento».</p>";
  }
  setTimeout(() => { const m = currentSimMap(); if (m && m.invalidateSize) m.invalidateSize(); }, 80);
  try { renderSim(await api(`/api/analyses/${encodeURIComponent(id)}/simulation`)); } catch (_) { /* sin simulación previa */ }
}

function simulateFrom(id) { state.simPreselect = id; setView("simulate"); }  // setView carga la lista una sola vez

function drawSimFrame(i) {
  const sim = state.sim;
  if (!sim) return;
  state.simIndex = i;
  if (state.simFrameLayer) state.simGroup.removeLayer(state.simFrameLayer);
  state.simFrameLayer = L.imageOverlay(`data:image/png;base64,${sim.frames[i].png_b64}`, sb(sim.bounds), { opacity: 0.88 });
  state.simFrameLayer.addTo(state.simGroup);
  document.getElementById("sim-hour").textContent = fmt(sim.frames[i].hour, 1);
  document.getElementById("sim-frame").value = String(i);
  document.querySelectorAll("#sim-stages button").forEach((b) => b.classList.toggle("on", Math.abs(Number(b.dataset.hour) - sim.frames[i].hour) < 0.06));
}
function stopSim() {
  if (state.simTimer) { clearInterval(state.simTimer); state.simTimer = null; }
  const b = document.getElementById("sim-play");
  if (b) b.textContent = "▶ Reproducir";
}
function toggleSim() {
  if (state.simTimer) { stopSim(); return; }
  if (!state.sim) return;
  if (state.simIndex >= state.sim.frames.length - 1) drawSimFrame(0);
  document.getElementById("sim-play").textContent = "❚❚ Pausar";
  state.simTimer = setInterval(() => {
    if (state.simIndex >= state.sim.frames.length - 1) { stopSim(); return; }
    drawSimFrame(state.simIndex + 1);
  }, 650);
}

// Encuadre del evento: la huella de la imagen y la trayectoria (no el dominio ampliado ni la costa lejana).
// Pura (sin mapa) para poder comprobarla: devuelve [[sur, oeste], [norte, este]] en las coordenadas del mapa.
function boundsOfPoints(points, padFrac = 0.25) {
  const lats = points.map((p) => p[0]), lons = points.map((p) => p[1]);
  let s = Math.min(...lats), n = Math.max(...lats), w = Math.min(...lons), e = Math.max(...lons);
  const dLat = Math.max(n - s, 1e-6), dLon = Math.max(e - w, 1e-6);
  return [[s - dLat * padFrac, w - dLon * padFrac], [n + dLat * padFrac, e + dLon * padFrac]];
}
function eventFitBounds(sim, proj = null, withCoast = false) {
  const P = proj ? proj.pt : (p) => p;
  const [[s, w], [n, e]] = sim.footprint;
  const pts = [P([s, w]), P([n, e]), P([s, e]), P([n, w])];
  (sim.track || []).forEach((p) => pts.push(P([p[0], p[1]])));
  if (sim.origin) pts.push(P(sim.origin));
  const coast = sim.context && sim.context.coast;
  if (withCoast && coast && !proj) pts.push(coast.start.point);
  return boundsOfPoints(pts, withCoast ? 0.1 : 0.25);
}
function fitSim(kind) {
  const sim = state.sim, map = currentSimMap();
  if (!sim || !map) return;
  const maxZoom = state.simImg ? 3 : 13;
  if (kind === "domain") map.fitBounds(sb(sim.bounds), { padding: [20, 20], maxZoom });
  else map.fitBounds(eventFitBounds(sim, state.simProj, kind === "coast"), { padding: [30, 30], maxZoom });
}

function renderSim(sim) {
  stopSim();
  state.sim = sim;
  const a = state.simAnalysis || state.analyses.find((x) => x.id === sim.analysis_id);
  state.simProj = state.simImg ? projFrom(sim.footprint, state.simImg.W, state.simImg.H) : null;
  const map = currentSimMap();
  const g = simGroup();
  if (a && !state.simImg) {
    L.imageOverlay(`/api/analyses/${a.id}/file/image`, a.bounds, { opacity: 0.9 }).addTo(g);  // la escena, en su sitio
    drawEvent({ slick: g, foot: g, marker: g }, a, false);
  }
  L.imageOverlay(`data:image/png;base64,${sim.affected_zone.png_b64}`, sb(sim.bounds), { opacity: 1 }).addTo(g);
  if (sim.origin) {
    L.circleMarker(sp(sim.origin), { radius: 7, color: "#fff", weight: 2, fillColor: "#ffb703", fillOpacity: 1 })
      .bindTooltip("Origen", { permanent: true, direction: "left", className: "origin-label" }).addTo(g);
  }
  drawTrack(g, sim, state.simProj);
  const coast = state.simImg ? null : (sim.context && sim.context.coast);
  if (coast && sim.origin) {
    L.polyline([sim.origin, coast.start.point], { color: "#ffffff", weight: 2, dashArray: "2 7", opacity: 0.8 }).addTo(g);
    L.circleMarker(coast.start.point, { radius: 5, color: "#fff", weight: 2, fillColor: "#6db7ff", fillOpacity: 1 })
      .bindTooltip(`Costa más cercana: ${fmt(coast.start.distance_km, 0)} km`, { permanent: true, direction: "top", className: "origin-label" }).addTo(g);
  }
  fitSim("event");
  document.getElementById("sim-fit").hidden = false;
  document.getElementById("fit-coast").hidden = !(coast && !state.simImg);

  const p = sim.params, f = sim.forcing;
  state.simControls.push(addHtmlControl(map, "topright", "wind-badge",
    `<div class="arr">${windArrowSvg(((p.wind_from_deg + 180) % 360))}</div>
     <div><b>Viento desde el ${esc(compassOf(p.wind_from_deg))}</b><br>
     <span class="muted">${fmt(p.wind_speed_ms, 0)} m/s · ${f.mode === "weather" ? "clima real" : "manual"}${sim.drift.bearing_to_deg != null ? " · deriva hacia el " + esc(sim.drift.direction_es) : ""}</span></div>`));
  state.simControls.push(addHtmlControl(map, "bottomleft", "legend", legendHtml([
    ["sw slick", "Mancha detectada hoy"], ["sw sim", "Mancha estimada"], ["sw zone", "Zona que podría verse afectada"],
    ["sw track", "Trayectoria estimada (marcas cada 6 h)"], ["pin", "Origen"]])));

  const slider = document.getElementById("sim-frame");
  slider.max = String(sim.frames.length - 1);
  const chips = document.getElementById("sim-stages");
  chips.innerHTML = sim.stages.map((s) =>
    `<button type="button" data-hour="${s.hour}">${s.hour === 0 ? "Inicio" : s.hour + " h"} · ${fmt(s.area_km2, 1)} km²</button>`).join("");
  chips.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    stopSim();
    const h = Number(b.dataset.hour);
    let best = 0;
    sim.frames.forEach((fr, i) => { if (Math.abs(fr.hour - h) < Math.abs(sim.frames[best].hour - h)) best = i; });
    drawSimFrame(best);
  }));
  document.getElementById("sim-controls").hidden = false;
  drawSimFrame(0);

  const as = sim.assumptions;
  document.getElementById("sim-summary").innerHTML = `
    <p>${esc(windLine({ wind_mean_ms: p.wind_speed_ms, wind_from_mean_deg: p.wind_from_deg }))}.</p>
    <details><summary>Supuestos de la simulación</summary><dl class="kv">
      <dt>Deriva</dt><dd>${fmt(as.drift_factor_of_wind * 100, 0)} % de la velocidad del viento${f.uses_currents ? " + corriente superficial" : ""}</dd>
      <dt>Difusión</dt><dd>${fmt(as.diffusivity_m2_s, 0)} m²/s</dd><dt>Degradación</dt><dd>${fmt(as.decay_per_hour * 100, 0)} % por hora</dd>
      <dt>No incluye</dt><dd>${esc(as.physics)}</dd><dt>Dominio</dt><dd>${esc(as.domain)}</dd></dl></details>`;
  document.getElementById("sim-context").innerHTML = sim.context
    ? `<details class="card ctx-card fold"><summary><h2>Qué podría verse afectado y qué hacer</h2></summary>${contextHtml(sim.context)}</details>` : "";
  document.getElementById("sim-simple").innerHTML = simpleSimBlock(sim);
}

// Texto simple de la simulación (cálculos en backend: app/simple.py). Sin porcentajes de crecimiento.
function simpleSimBlock(sim) {
  const s = sim.simple;
  if (!s) return "";
  const rows = s.stages.filter((r) => r.hour <= 12).map((r) => `<tr><td>${r.hour} horas</td><td>${fmt(r.displacement_km, 1)} km</td></tr>`).join("");
  const c = s.coast;
  const coast = !c ? "Sin ubicación confirmada no se puede evaluar la costa."
    : `La costa más cercana está a ${fmt(c.distance_km, 0)} km hacia el ${esc(c.direction_es)}. ` +
      (c.reaches ? `<b>La trayectoria la alcanza en unas ${fmt(c.hours, 0)} horas.</b>` : `En las ${s.horizon_hours} horas simuladas no la alcanza.`);
  const rg = s.affected_range_km2;
  return `<div class="card story"><h3>En palabras simples</h3>
    <p>La mancha se mueve <b>${esc(s.direction_text)}</b>, a ${esc(s.speed_text)}.</p>
    <table class="stage-table"><thead><tr><th>Pasado este tiempo</th><th>Se desplazó</th></tr></thead><tbody>${rows}</tbody></table>
    <p>${coast}</p>
    <p><span class="chip ${esc(s.severity.level)}">${esc(s.severity.label.toUpperCase())}</span> ${esc(s.severity.why)}</p>
    ${rg ? `<p>Zona que podría verse afectada (estimación aproximada): entre ${fmt(rg[0], 1)} y ${fmt(rg[1], 1)} km².</p>` : ""}</div>`;
}

async function runSimulation(event) {
  event.preventDefault();
  clearError();
  const f = event.target;
  const id = f.elements.analysis.value;
  if (!id) { showError("Seleccione un evento con zona marcada y ubicación conocida."); return; }
  const mode = (f.querySelector("input[name=mode]:checked") || {}).value || "weather";
  const body = { mode, hours: Number(f.elements.hours.value) };
  if (mode === "manual") {
    body.wind_speed_ms = Number(f.elements.wind_speed_ms.value);
    body.wind_from_deg = Number(f.elements.wind_from_deg.value);
  }
  const btn = f.querySelector("button[type=submit]");
  btn.disabled = true;
  document.getElementById("sim-summary").innerHTML = `<p class="muted">${mode === "weather" ? "Consultando el clima real y calculando la simulación…" : "Calculando la simulación…"}</p>`;
  try {
    const sim = await api(`/api/analyses/${encodeURIComponent(id)}/simulate`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    renderSim(sim);
    document.getElementById("sim-controls").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    document.getElementById("sim-summary").innerHTML = "";
    showError(err.message);
    if (mode === "weather") {  // el clima no está disponible: se ofrece el viento manual sin bloquear la plataforma
      const manual = f.querySelector("input[name=mode][value=manual]");
      manual.checked = true;
      document.getElementById("manual-fields").hidden = false;
      document.getElementById("sim-summary").innerHTML = "<div class=\"notice\"><b>No se pudo usar el clima real.</b> Ya está seleccionado el modo <b>Manual</b>: elija el viento y pulse «Simular desplazamiento» otra vez.</div>";
    }
  } finally { btn.disabled = false; }
}

// ---------- historial ----------
async function loadHistory() {
  const box = document.getElementById("history-table");
  try {
    const data = await api("/api/analyses");
    if (!data.items.length) {
      box.innerHTML = "<div class=\"card\"><p class=\"muted\">Todavía no hay análisis. Pruebe con un ejemplo en la sección Análisis.</p></div>";
      return;
    }
    box.innerHTML = data.items.map((a) => `
      <article class="hcard">
        <img src="/api/analyses/${esc(a.id)}/file/overlay" alt="Imagen analizada" loading="lazy">
        <div class="hbody">
          <div class="t">${esc(a.headline)}</div>
          <div class="m">${fmtDate(a.created_at)} · ${esc(a.filename || "")}</div>
          <div>${locChip(locState(a))}${levelChip(levelOf(a))}</div>
          <div class="m">Superficie: ${a.slick_area_km2 != null ? fmt(a.slick_area_km2, 1) + " km²" : fmt(a.slick_pct, 2) + " % de la imagen"} · ${fmt(a.n_components, 0)} zona(s)</div>
          <div class="m">${a.center_lat != null ? esc(fmtCoord(a.center_lat, a.center_lon)) + (locState(a) === "manual" ? " (aprox.)" : "") : "Ubicación desconocida"}</div>
          <div class="actions">
            <button class="primary small-btn" onclick="openAnalysis('${esc(a.id)}')">Abrir</button>
            <button class="secondary small-btn" onclick="window.open('/api/analyses/${esc(a.id)}/report','_blank')">Informe</button>
            <button class="secondary small-btn" onclick="simulateFrom('${esc(a.id)}')" ${a.slick_pct > 0 ? "" : "disabled"}>Simular</button>
            <button class="danger" onclick="deleteAnalysis('${esc(a.id)}')">Eliminar</button>
          </div>
        </div>
      </article>`).join("");
  } catch (err) { box.innerHTML = ""; showError(err.message); }
}

async function deleteAnalysis(id) {
  if (!confirm("¿Eliminar este análisis y sus archivos? Esta acción no se puede deshacer.")) return;
  try { await api(`/api/analyses/${encodeURIComponent(id)}`, { method: "DELETE" }); delete state.ctxCache[id]; loadHistory(); }
  catch (err) { showError(err.message); }
}


// ---------- ejemplos (con y sin derrame) ----------
async function loadExamples() {
  const sel = document.getElementById("example-select");
  try {
    const g = (await api("/api/examples")).groups;
    const opt = (i, txt) => `<option value="${esc(i.sample_id)}">${esc(txt)}</option>`;
    sel.innerHTML = `<optgroup label="Con derrame">${g.con.map((i) => opt(i, i.sample_id + (i.scene && i.scene.date ? " · " + i.scene.date : ""))).join("")}</optgroup>` +
      `<optgroup label="Sin derrame">${g.sin.map((i) => opt(i, `${i.label} · ${i.region}`)).join("")}</optgroup>`;
  } catch (_) { sel.innerHTML = ""; }
}

// ---------- buscar lugar (nombre -> coordenadas) ----------
function wirePlaceSearch(root) {
  const q = root.querySelector("[data-place-q]"), go = root.querySelector("[data-place-go]");
  const sel = root.querySelector("[data-place-results]"), note = root.querySelector("[data-place-note]");
  const lat = root.querySelector("[data-lat]"), lon = root.querySelector("[data-lon]");
  if (!q || !go || !sel || !lat || !lon || go.dataset.wired) return;
  go.dataset.wired = "1";
  const label = (i) => [i.name, i.admin1, i.country].filter(Boolean).join(", ");
  const choose = () => {
    const it = sel._items[Number(sel.value)];
    lat.value = it.lat.toFixed(4); lon.value = it.lon.toFixed(4);
    note.textContent = `Ubicación aproximada elegida: ${label(it)}. Podrá cambiarla escribiendo las coordenadas.`;
  };
  const run = async () => {
    const term = q.value.trim();
    if (term.length < 2) { note.textContent = "Escriba al menos 2 letras del lugar."; return; }
    note.textContent = "Buscando…";
    try {
      const data = await api(`/api/geocode?q=${encodeURIComponent(term)}`);
      if (!data.items.length) {
        sel.hidden = true;
        note.textContent = "No se encontró ese lugar. Pruebe con otro nombre o escriba las coordenadas (el mar abierto no tiene nombre: use coordenadas).";
        return;
      }
      sel._items = data.items;
      sel.innerHTML = data.items.map((i, k) => `<option value="${k}">${esc(label(i))} (${i.lat.toFixed(2)}, ${i.lon.toFixed(2)})</option>`).join("");
      sel.hidden = false;
      choose();
    } catch (err) { note.textContent = err.message; }
  };
  go.addEventListener("click", run);
  sel.addEventListener("change", choose);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); run(); } });
}

// ---------- ubicación posterior y escenarios por dirección del viento ----------
function locationCard() {
  return `<div class="card loc-card" id="loc-card"><h3>Indicar dónde está</h3>
    <p class="muted small">Con la ubicación se consulta el clima real, la costa más cercana y se coloca en el mapa (como ubicación aproximada indicada por usted).</p>
    <div class="row"><label>Buscar un lugar <input type="text" data-place-q placeholder="ej. Ciudad del Carmen"></label>
      <button type="button" class="secondary small-btn" data-place-go style="align-self:end;margin-bottom:.65rem">Buscar</button></div>
    <select data-place-results hidden aria-label="Resultados de la búsqueda"></select><p class="muted small" data-place-note></p>
    <div class="row"><label>Latitud <input type="number" data-lat step="any" min="-90" max="90"></label>
      <label>Longitud <input type="number" data-lon step="any" min="-180" max="180"></label>
      <label>Ancho de la imagen (km) <input type="number" id="loc-km" step="any" min="0.05" max="5000" placeholder="opcional"></label></div>
    <button type="button" class="primary" id="loc-go">Usar esta ubicación</button></div>`;
}
async function submitLocation(a) {
  const card = document.getElementById("loc-card");
  const lat = card.querySelector("[data-lat]").value, lon = card.querySelector("[data-lon]").value, km = card.querySelector("#loc-km").value;
  if (!lat || !lon) { showError("Indique el lugar: búsquelo por nombre o escriba latitud y longitud."); return; }
  clearError();
  const body = { lat: Number(lat), lon: Number(lon) };
  if (km) body.scene_km = Number(km);
  try {
    const upd = await api(`/api/analyses/${encodeURIComponent(a.id)}/location`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    delete state.ctxCache[a.id];
    showAnalysis(upd);
  } catch (err) { showError(err.message); }
}

function scenarioCard(a) {
  return `<details class="card ctx-card fold" id="scen-card"><summary><h2>¿Hacia dónde podría ir según el viento?</h2></summary>
    <p class="muted small">Para cada dirección posible del viento, hacia dónde se desplazaría la mancha. ${a.bounds
      ? "Como hay ubicación, también se evalúa si se acercaría a la costa."
      : "Sin ubicación no se consulta el clima real: elija usted la fuerza del viento."}</p>
    <div class="row"><label>Fuerza del viento <select id="scen-wind"><option value="3">Suave (≈ 11 km/h)</option><option value="8" selected>Moderado (≈ 29 km/h)</option>
      <option value="15">Fuerte (≈ 54 km/h)</option><option value="22">Muy fuerte (≈ 79 km/h)</option></select></label>
      <label>Horas <select id="scen-hours"><option value="6">6 horas</option><option value="12" selected>12 horas</option><option value="24">24 horas</option><option value="48">48 horas</option></select></label></div>
    <div id="scen-body"><p class="skeleton">Calculando…</p></div></details>`;
}
async function loadScenarios(a) {
  const body = document.getElementById("scen-body");
  if (!body) return;
  const wind = document.getElementById("scen-wind").value, hours = document.getElementById("scen-hours").value;
  try { body.innerHTML = roseHtml(await api(`/api/analyses/${encodeURIComponent(a.id)}/scenarios?wind_ms=${wind}&hours=${hours}`)); }
  catch (err) { body.innerHTML = `<div class="notice">${esc(err.message)}</div>`; }
}
function roseHtml(d) {
  const cx = 180, cy = 180, R = 105;
  const arrows = d.items.map((i) => {
    const ang = (i.to_deg * Math.PI) / 180, x2 = cx + R * Math.sin(ang), y2 = cy - R * Math.cos(ang);
    const col = i.reaches_coast ? "#ff5a36" : i.closest_km != null && i.closest_km < 30 ? "#ffb703" : "#2ee6d0";
    const lx = cx + (R + 30) * Math.sin(ang), ly = cy - (R + 30) * Math.cos(ang);
    return `<line x1="${cx}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="3" stroke-linecap="round"/>
      <g transform="translate(${x2.toFixed(1)} ${y2.toFixed(1)}) rotate(${i.to_deg})"><path d="M0 -7 6 6 0 3 -6 6z" fill="${col}"/></g>
      <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle">${esc(i.from_es)}</text>`;
  }).join("");
  const rows = d.items.map((i) => `<tr class="${i.reaches_coast ? "hit" : ""}"><td>${esc(i.from_es)}</td><td>${esc(i.to_es)}</td><td>${fmt(i.distance_km, 1)} km</td>
      ${d.located ? `<td>${i.reaches_coast ? `Podría llegar a la costa (≈ ${fmt(i.closest_hour, 0)} h)` : i.closest_km != null ? `${fmt(i.closest_km, 0)} km de la costa` : "—"}</td>` : ""}</tr>`).join("");
  return `<div class="rose"><div><svg viewBox="0 0 360 360" role="img" aria-label="Hacia dónde se desplazaría según la dirección del viento">
      <circle cx="${cx}" cy="${cy}" r="${R + 14}" fill="#061b2e" stroke="#2a5a7c"/><text x="${cx}" y="18" text-anchor="middle" style="fill:#2ee6d0;font-weight:700">N</text>
      ${arrows}<circle cx="${cx}" cy="${cy}" r="9" fill="#ff6a3d" stroke="#fff" stroke-width="2"/></svg>
      <p class="muted small" style="text-align:center">La etiqueta indica de dónde sopla el viento; la flecha, hacia dónde se movería la mancha.</p></div>
    <div><table class="rose-table"><thead><tr><th>Si el viento viene del</th><th>Se movería hacia el</th><th>En ${d.hours} h</th>${d.located ? "<th>Costa</th>" : ""}</tr></thead><tbody>${rows}</tbody></table>
      <p class="muted small">Con viento de ${fmt(d.wind_ms, 0)} m/s. Es un escenario simple (la mancha se mueve ≈ 3 % de la velocidad del viento), no una predicción.${d.located ? " Rojo: podría alcanzar la costa; ámbar: se acercaría a menos de 30 km." : ""}</p></div></div>`;
}

// ---------- eventos ----------
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

  document.getElementById("form-upload").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    const f = event.target;
    if (state.exMode) {  // modo «Ver un ejemplo»: mismo botón, otra acción
      const id = document.getElementById("example-select").value;
      if (!id) { showError("Seleccione un ejemplo."); return; }
      const eb = f.querySelector("button[type=submit]");
      eb.disabled = true;
      setStatus("Analizando el ejemplo… (la primera vez puede tardar unos segundos)");
      try { showAnalysis(await api(`/api/examples/${encodeURIComponent(id)}/analyze`, { method: "POST" })); }
      catch (err) { showError(err.message); }
      finally { eb.disabled = false; setStatus(""); }
      return;
    }
    const fd = new FormData();
    fd.append("file", f.elements.file.files[0]);
    if (f.elements.lat.value && f.elements.lon.value) {
      fd.append("lat", f.elements.lat.value);
      fd.append("lon", f.elements.lon.value);
    }
    if (f.elements.scene_km.value) fd.append("scene_km", f.elements.scene_km.value);
    const btn = f.querySelector("button[type=submit]");
    btn.disabled = true;
    setStatus("Analizando la imagen… (la primera vez puede tardar unos segundos)");
    try { showAnalysis(await api("/api/analyze", { method: "POST", body: fd })); }
    catch (err) { showError(err.message); }
    finally { btn.disabled = false; setStatus(""); }
  });


  wirePlaceSearch(document.getElementById("form-upload"));
  document.querySelectorAll(".mode-tabs button").forEach((b) => b.addEventListener("click", () => {
    state.exMode = b.dataset.mode === "example";
    document.querySelectorAll(".mode-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.getElementById("upload-box").hidden = state.exMode;
    document.getElementById("example-box").hidden = !state.exMode;
    document.getElementById("upload-file").required = !state.exMode;
  }));
  loadExamples();
  const form = document.getElementById("form-sim");
  form.addEventListener("submit", runSimulation);
  form.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
    document.getElementById("manual-fields").hidden = form.querySelector("input[name=mode]:checked").value !== "manual";
  }));
  document.getElementById("sim-analysis").addEventListener("change", onSimEventChange);
  document.getElementById("sim-dir").addEventListener("change", updateCompass);
  document.getElementById("sim-frame").addEventListener("input", (e) => { stopSim(); drawSimFrame(Number(e.target.value)); });
  document.getElementById("sim-play").addEventListener("click", toggleSim);
  document.getElementById("fit-event").addEventListener("click", () => fitSim("event"));
  document.getElementById("fit-domain").addEventListener("click", () => fitSim("domain"));
  document.getElementById("fit-coast").addEventListener("click", () => fitSim("coast"));

  setView("analyze");
});
