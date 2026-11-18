/* OCEAN GUARD AI · interfaz simple en 3 pasos (JavaScript sin dependencias, salvo Leaflet para los mapas). */
"use strict";

const state = { a: null, sim: null, map: null, history: [], shown: 10 };
const HISTORY_PAGE = 10;
const LIGHT = { green: "✓", yellow: "!", red: "!" };

// ---------- utilidades ----------
const $ = (id) => document.getElementById(id);
function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(v, d = 1) {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString("es", { maximumFractionDigits: d });
}
function showError(msg) { const b = $("global-error"); b.textContent = msg; b.hidden = false; window.scrollTo({ top: 0, behavior: "smooth" }); }
function clearError() { const b = $("global-error"); b.hidden = true; b.textContent = ""; }
function setStatus(msg) { const b = $("analyze-status"); b.hidden = !msg; b.textContent = msg || ""; }
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

// ---------- pasos ----------
function go(n) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `step-${n}`));
  document.querySelectorAll(".step").forEach((b) => b.classList.toggle("active", Number(b.dataset.step) === n));
  clearError();
  if (n === 3 && state.map) setTimeout(() => state.map.invalidateSize(), 100);
  window.scrollTo({ top: 0 });
}
function enableSteps(max) {
  document.querySelectorAll(".step").forEach((b) => { b.disabled = Number(b.dataset.step) > max; });
}

// ---------- paso 2: resultado en una sola tarjeta ----------
function technicalHtml(s) {
  const t = s.technical, rel = t.reliability, c = t.confidence;
  return `<details class="card"><summary>Detalles técnicos</summary>
    <p>${esc(t.details)}</p>
    <p><b>Confianza:</b> ${esc(c.label)}. ${esc(c.text)}</p>
    <p><b>Umbral del modelo:</b> ${fmt(t.threshold, 2)} · <b>zona mínima:</b> ${t.min_component_px} píxeles
       ${t.model_probability != null ? `· <b>probabilidad media en lo marcado:</b> ${fmt(t.model_probability, 3)}` : ""}</p>
    <p><b>Medido en validación:</b> precisión ${fmt(rel.precision, 2)}, cobertura ${fmt(rel.recall, 2)}, Dice ${fmt(rel.dice, 2)}.</p>
    <p><b>Regla del semáforo:</b> rojo si la probabilidad media ≥ ${t.traffic_rules.red_mean_prob} y la zona mayor ≥ ${t.traffic_rules.red_min_component_px} píxeles; amarillo si hay zona marcada pero menos segura; verde si no hay zonas.</p>
    <p class="fineprint">${esc(t.disclaimer)}</p></details>`;
}
function showResult(a) {
  state.a = a;
  state.sim = null;
  const s = a.summary, t = s.traffic;
  const warn = (s.warnings || []).map((w) => `<div class="notice">${esc(w)}</div>`).join("");
  const facts = [];
  if (t.area_km2 != null) facts.push(`<span>Área estimada: ${fmt(t.area_km2, 1)} km²${a.result && a.result.scale_assumed ? " (escala supuesta)" : ""}</span>`);
  if (t.where) facts.push(`<span>Dónde: ${esc(t.where)}</span>`);
  $("result").innerHTML = `
    <div class="verdict ${esc(t.color)}" role="status">
      <div class="light" aria-hidden="true">${LIGHT[t.color]}</div>
      <div><h1>${esc(t.label)}</h1><p>${esc(t.phrase)}</p>${facts.length ? `<div class="facts">${facts.join("")}</div>` : ""}</div>
    </div>
    ${warn}
    <div class="shotwrap"><img src="/api/analyses/${encodeURIComponent(a.id)}/file/image" alt="Imagen analizada">
      ${t.color !== "green" ? `<img class="over" src="/api/analyses/${encodeURIComponent(a.id)}/file/slick" alt="Zona marcada por el sistema">` : ""}</div>
    ${technicalHtml(s)}
    ${t.color === "green"
      ? `<p class="muted">No hay nada que simular. <button type="button" class="ghost" id="again">Analizar otra imagen</button></p>`
      : `<button type="button" class="primary" id="to-step3">Ver hacia dónde va →</button>`}`;
  if ($("to-step3")) $("to-step3").addEventListener("click", () => { prepareStep3(); go(3); });
  if ($("again")) $("again").addEventListener("click", () => go(1));
  enableSteps(t.color === "green" ? 2 : 3);
  prepareStep3();
  go(2);
}

// ---------- paso 3: hacia dónde va ----------
function locationCardHtml() {
  return `<div class="card" id="loc-card"><h2>Indique dónde está (opcional)</h2>
    <p class="muted">Sin ubicación se muestra un escenario sobre la propia imagen. Con ella se usa el viento real y se evalúa la costa.</p>
    <div class="row"><label>Buscar un lugar <input type="text" data-place-q placeholder="ej. Ciudad del Carmen"></label>
      <button type="button" class="ghost" data-place-go>Buscar</button></div>
    <select data-place-results hidden aria-label="Resultados de la búsqueda"></select><p class="muted" data-place-note></p>
    <div class="row"><label>Latitud <input type="number" data-lat step="any" min="-90" max="90"></label>
      <label>Longitud <input type="number" data-lon step="any" min="-180" max="180"></label></div>
    <label>Ancho de la imagen (km) <input type="number" id="loc-km" step="any" min="0.05" max="5000" placeholder="opcional"></label>
    <button type="button" class="secondary" id="loc-go">Usar esta ubicación</button></div>`;
}
function prepareStep3() {
  const a = state.a; if (!a) return;
  const located = !!a.bounds, f = $("form-sim");
  $("sim-result").innerHTML = "";
  if (state.map) { state.map.remove(); state.map = null; }
  $("loc-box").innerHTML = located ? "" : locationCardHtml();
  $("sim-mode-box").hidden = !located;
  f.elements.mode.value = located ? "weather" : "manual";
  $("manual-fields").hidden = located;
  $("manual-question").hidden = located;
  $("manual-question").textContent = "¿Y si el viento viene del…?";
  if (!located) {
    wirePlaceSearch($("loc-box"));
    $("loc-go").addEventListener("click", submitLocation);
  }
}
async function submitLocation() {
  const card = $("loc-card");
  const lat = card.querySelector("[data-lat]").value, lon = card.querySelector("[data-lon]").value, km = $("loc-km").value;
  if (!lat || !lon) { showError("Indique el lugar: búsquelo por nombre o escriba latitud y longitud."); return; }
  clearError();
  const body = { lat: Number(lat), lon: Number(lon) };
  if (km) body.scene_km = Number(km);
  try {
    const upd = await api(`/api/analyses/${encodeURIComponent(state.a.id)}/location`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    showResult(upd); go(3);
  } catch (err) { showError(err.message); }
}

const base = () => {
  const esri = "https://server.arcgisonline.com/ArcGIS/rest/services/";
  return L.layerGroup([
    L.tileLayer(esri + "World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 13, maxNativeZoom: 10, attribution: "Imágenes © Esri, Maxar" }),
    L.tileLayer(esri + "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", { maxZoom: 13, maxNativeZoom: 10 }),
  ]);
};

function drawSimMap(sim) {
  const a = state.a, located = !!a.bounds, el = $("sim-map");
  if (state.map) { state.map.remove(); state.map = null; }
  if (located) {
    const map = L.map(el, { minZoom: 2, maxZoom: 13 });
    base().addTo(map); L.control.scale({ imperial: false }).addTo(map);
    L.imageOverlay(`/api/analyses/${a.id}/file/slick`, a.bounds).addTo(map);
    const pts = [sim.origin];
    const track = (sim.track || []).map((p) => [p[0], p[1]]);
    if (track.length > 1) L.polyline(track, { color: "#f2b600", weight: 4, dashArray: "8 6" }).addTo(map);
    track.forEach((p) => pts.push(p));
    L.circleMarker(sim.origin, { radius: 7, color: "#fff", weight: 2, fillColor: "#b3261e", fillOpacity: 1 }).bindTooltip("Mancha hoy").addTo(map);
    const c = sim.simple.coast;
    if (c && c.point) {
      L.circleMarker(c.point, { radius: 7, color: "#fff", weight: 2, fillColor: "#1b5fd6", fillOpacity: 1 }).bindTooltip("Costa más cercana").addTo(map);
      L.polyline([sim.origin, c.point], { color: "#1b5fd6", weight: 2, dashArray: "2 6" }).addTo(map);
      pts.push(c.point);
    }
    state.map = map;
    map.fitBounds(L.latLngBounds(pts).pad(0.25), { maxZoom: 12 });
  } else {
    // sin ubicación: se dibuja sobre la propia imagen (coordenadas en píxeles)
    const W = a.width, H = a.height, [[s, w], [n, e]] = sim.footprint;
    const px = ([la, lo]) => [((la - s) / (n - s)) * H, ((lo - w) / (e - w)) * W];
    const map = L.map(el, { crs: L.CRS.Simple, minZoom: -3, maxZoom: 3, zoomSnap: 0.25, attributionControl: false });
    const bb = [[0, 0], [H, W]];
    L.imageOverlay(`/api/analyses/${a.id}/file/image`, bb).addTo(map);
    L.imageOverlay(`/api/analyses/${a.id}/file/slick`, bb).addTo(map);
    const track = (sim.track || []).map((p) => px([p[0], p[1]]));
    if (track.length > 1) L.polyline(track, { color: "#f2b600", weight: 4, dashArray: "8 6" }).addTo(map);
    L.circleMarker(px(sim.origin), { radius: 7, color: "#fff", weight: 2, fillColor: "#b3261e", fillOpacity: 1 }).bindTooltip("Mancha hoy").addTo(map);
    state.map = map;
    map.fitBounds(bb);
  }
}

function renderSim(sim) {
  state.sim = sim;
  const s = sim.simple, a = state.a, located = !!a.bounds;
  const rows = s.stages.map((r) => `<tr><td>${r.hour} h</td><td>${fmt(r.displacement_km, 1)} km</td>
    <td>${r.area_km2 != null ? `≈ ${fmt(r.area_km2, 1)} km²` : "—"}${r.area_growth_pct != null ? ` (${r.area_growth_pct >= 0 ? "+" : ""}${fmt(r.area_growth_pct, 0)} %)` : ""}</td></tr>`).join("");
  let coast;
  if (!located) coast = "No se conoce la ubicación, así que no se puede evaluar la costa. Indique dónde está arriba para hacerlo.";
  else if (!s.coast) coast = "No se pudo calcular la costa más cercana para esta zona.";
  else coast = `La costa más cercana está a ${fmt(s.coast.distance_km, 0)} km hacia el ${esc(s.coast.direction_es)}. ` +
    (s.coast.reaches ? `<b>La trayectoria la alcanza en unas ${fmt(s.coast.hours, 0)} horas.</b>`
      : `En las ${s.horizon_hours} horas simuladas la mancha no la alcanza.`);
  const notes = (sim.weather_notes || []).map((n) => `<div class="notice">${esc(n)}</div>`).join("");
  $("sim-result").innerHTML = `
    <div class="card">
      <h2>La mancha se mueve ${esc(s.direction_text)}</h2>
      <p>Velocidad aproximada: ${esc(s.speed_text)}.</p>
      <p><span class="badge ${esc(s.severity.level)}">${esc(s.severity.label.toUpperCase())}</span> ${esc(s.severity.why)}</p>
      <p>${coast}</p>
      ${notes}
    </div>
    <div id="sim-map" class="map" role="img" aria-label="Mapa con la mancha, su trayectoria y la costa más cercana"></div>
    <div class="card"><table><thead><tr><th>Pasado este tiempo</th><th>Se desplazó</th><th>Área (aprox.)</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="fineprint">Es una estimación simple: no incluye oleaje ni corrientes marinas. No es una predicción.
      ${sim.weather_mode === "weather" ? " Al consultar el viento real, a Open-Meteo solo se envían la latitud, la longitud y las fechas del evento." : ""}</p>`;
  drawSimMap(sim);
}

async function runSimulation(event) {
  event.preventDefault(); clearError();
  const f = event.target, a = state.a, located = !!a.bounds;
  const mode = located ? f.elements.mode.value : "manual";
  const body = { mode, hours: Number(f.elements.hours.value) };
  if (mode === "manual") { body.wind_speed_ms = Number(f.elements.wind_speed_ms.value); body.wind_from_deg = Number(f.elements.wind_from_deg.value); }
  const btn = f.querySelector("button[type=submit]"); btn.disabled = true;
  $("sim-result").innerHTML = `<p class="alert info">Calculando…</p>`;
  try {
    renderSim(await api(`/api/analyses/${encodeURIComponent(a.id)}/simulate`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
  } catch (err) {
    $("sim-result").innerHTML = "";
    if (mode === "weather") {  // si falla el clima real, se pasa sola al viento elegido por la persona
      f.elements.mode.value = "manual"; $("manual-fields").hidden = false;
      $("manual-question").hidden = false; $("manual-question").textContent = "Elija usted el viento:";
      showError(`No se pudo obtener el viento real (${err.message}). Elija el viento a mano y vuelva a pulsar el botón.`);
    } else showError(err.message);
  } finally { btn.disabled = false; }
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
    note.textContent = `Ubicación aproximada elegida: ${label(it)}.`;
  };
  const run = async () => {
    const term = q.value.trim();
    if (term.length < 2) { note.textContent = "Escriba al menos 2 letras del lugar."; return; }
    note.textContent = "Buscando…";
    try {
      const data = await api(`/api/geocode?q=${encodeURIComponent(term)}`);
      if (!data.items.length) { sel.hidden = true; note.textContent = "No se encontró ese lugar. Pruebe con otro nombre o escriba las coordenadas."; return; }
      sel._items = data.items;
      sel.innerHTML = data.items.map((i, k) => `<option value="${k}">${esc(label(i))}</option>`).join("");
      sel.hidden = false; choose();
    } catch (err) { note.textContent = err.message; }
  };
  go.addEventListener("click", run);
  sel.addEventListener("change", choose);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); run(); } });
}

// ---------- ejemplos e historial ----------
async function loadExamples() {
  try {
    const d = await api("/api/examples");
    $("example-select").innerHTML = d.items.map((i) => `<option value="${esc(i.sample_id)}">${esc(i.sample_id)}${i.scene && i.scene.date ? " · " + esc(i.scene.date) : ""}</option>`).join("");
  } catch (err) {
    $("example-select").innerHTML = "";
    $("form-example").querySelector("button").disabled = true;
    $("form-example").insertAdjacentHTML("beforeend", `<p class="muted">Los ejemplos no están disponibles (${esc(err.message)}).</p>`);
  }
}
async function loadHistory() {
  try { state.history = (await api("/api/analyses")).items; } catch (_) { state.history = []; }
  renderHistory();
}
function renderHistory() {
  const items = state.history, list = $("history-list");
  if (!items.length) { list.innerHTML = `<li class="muted">Todavía no hay análisis.</li>`; $("history-more").hidden = true; return; }
  list.innerHTML = items.slice(0, state.shown).map((i) => `<li>
    <i class="dot ${esc(i.traffic)}" aria-hidden="true"></i>
    <div class="t"><b>${esc(i.traffic_label)}</b><br><span class="muted">${esc(i.filename || i.id)} · ${new Date(i.created_at).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" })}</span></div>
    <button type="button" class="ghost" data-open="${esc(i.id)}">Abrir</button>
    <button type="button" class="ghost" data-del="${esc(i.id)}" aria-label="Eliminar este análisis">Eliminar</button></li>`).join("");
  $("history-more").hidden = items.length <= state.shown;
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".step").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) go(Number(b.dataset.step)); }));
  wirePlaceSearch($("form-upload"));
  $("form-upload").addEventListener("submit", async (event) => {
    event.preventDefault(); clearError();
    const f = event.target, fd = new FormData();
    fd.append("file", f.elements.file.files[0]);
    if (f.elements.lat.value && f.elements.lon.value) { fd.append("lat", f.elements.lat.value); fd.append("lon", f.elements.lon.value); }
    if (f.elements.scene_km.value) fd.append("scene_km", f.elements.scene_km.value);
    const btn = f.querySelector("button[type=submit]"); btn.disabled = true;
    setStatus("Analizando la imagen… (la primera vez puede tardar unos segundos)");
    try { showResult(await api("/api/analyze", { method: "POST", body: fd })); loadHistory(); }
    catch (err) { showError(err.message); }
    finally { btn.disabled = false; setStatus(""); }
  });
  $("form-example").addEventListener("submit", async (event) => {
    event.preventDefault(); clearError();
    const id = event.target.elements.sample.value;
    if (!id) { showError("Seleccione un ejemplo."); return; }
    const btn = event.target.querySelector("button"); btn.disabled = true;
    setStatus("Analizando el ejemplo… (la primera vez puede tardar unos segundos)");
    try { showResult(await api(`/api/examples/${encodeURIComponent(id)}/analyze`, { method: "POST" })); loadHistory(); }
    catch (err) { showError(err.message); }
    finally { btn.disabled = false; setStatus(""); }
  });
  $("form-sim").addEventListener("submit", runSimulation);
  $("form-sim").querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
    const manual = $("form-sim").elements.mode.value === "manual";
    $("manual-fields").hidden = !manual; $("manual-question").hidden = false;
    $("manual-question").textContent = "Elija usted el viento:";
  }));
  $("history-more").addEventListener("click", () => { state.shown += HISTORY_PAGE; renderHistory(); });
  $("history-list").addEventListener("click", async (e) => {
    const open = e.target.closest("[data-open]"), del = e.target.closest("[data-del]");
    try {
      if (open) showResult(await api(`/api/analyses/${encodeURIComponent(open.dataset.open)}`));
      if (del && confirm("¿Eliminar este análisis?")) { await api(`/api/analyses/${encodeURIComponent(del.dataset.del)}`, { method: "DELETE" }); loadHistory(); }
    } catch (err) { showError(err.message); }
  });
  loadExamples(); loadHistory();
});
