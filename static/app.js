/* ML Bench Studio front end — plain JS, no build step. Renders whatever the plugin contract describes. */
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = v => typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, {maximumFractionDigits: 4})) : esc(v);

const ICONS = {
  home: '<path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/>',
  trend: '<path d="M3 17 9 11l4 4 8-8"/><path d="M14 7h7v7"/>',
  layers: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
  server: '<rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><path d="M7 7h.01M7 17h.01"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="m21 16-5-5-9 9"/>',
  shield: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3Z"/><path d="m9 12 2 2 4-4"/>',
  cube: '<path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
};
const STATUS = {ready: ["ok", "ready"], "needs-artifacts": ["warn", "artifacts missing"], "needs-key": ["info", "needs key"]};
const PALETTE = ["#7C3AED", "#0891B2", "#DC2626", "#15803D", "#B45309", "#DB2777", "#4B5563", "#65A30D"];

let plugins = [], current = null, filter = "All";

// ---------- routing (hash) ----------
function go(id) { location.hash = id ? `#/${id}` : "#/"; }
window.addEventListener("hashchange", route);
function route() {
  const id = location.hash.replace(/^#\/?/, "");
  const p = plugins.find(x => x.id === id);
  p ? showModel(p) : showHome();
}

// ---------- theme ----------
const root = document.documentElement;
$("#theme").onclick = () => {
  const dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "light" : "dark"; localStorage.setItem("studio-theme", root.dataset.theme);
};
if (localStorage.getItem("studio-theme")) root.dataset.theme = localStorage.getItem("studio-theme");

// ---------- home ----------
function showHome() {
  current = null;
  $("#view-model").hidden = true; $("#view-home").hidden = false;
  $("#crumbs").innerHTML = "<li>Models</li>";
  const cats = ["All", ...new Set(plugins.map(p => p.category))];
  $("#filters").innerHTML = cats.map(c => `<button class="chip" role="tab" aria-selected="${c === filter}" data-c="${esc(c)}">${esc(c)}</button>`).join("");
  $("#filters").querySelectorAll(".chip").forEach(b => b.onclick = () => { filter = b.dataset.c; showHome(); });
  const list = plugins.filter(p => filter === "All" || p.category === filter);
  $("#grid").innerHTML = list.map(p => {
    const [cls, txt] = STATUS[p.status] || ["info", p.status];
    return `<button class="tile reveal" data-id="${esc(p.id)}" aria-label="Open ${esc(p.title)}">
      <span class="ico"><svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[p.icon] || ICONS.cube}</svg></span>
      <h3>${esc(p.title)}</h3><p>${esc(p.tagline)}</p>
      <span class="foot"><span>${esc(p.headline_metric)}</span><span class="pill ${cls}">${txt}</span></span></button>`;
  }).join("");
  $("#grid").querySelectorAll(".tile").forEach(t => t.onclick = () => go(t.dataset.id));
}

// ---------- model screen ----------
function showModel(p) {
  current = p;
  $("#view-home").hidden = true; $("#view-model").hidden = false;
  $("#crumbs").innerHTML = `<li><a href="#/">Models</a></li><li>${esc(p.category)}</li><li>${esc(p.title)}</li>`;
  $("#m-cat").textContent = p.category; $("#m-title").textContent = p.title; $("#m-desc").textContent = p.description || p.tagline;
  $("#m-metric").textContent = p.headline_metric; $("#m-repo").href = "file:///" + p.repo_path.replace(/\\/g, "/"); $("#m-repo").textContent = p.repo;
  $("#examples").innerHTML = p.examples.map((e, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(e.note || "")}">${esc(e.label)}</button>`).join("") || '<span class="muted small">No examples</span>';
  $("#examples").querySelectorAll(".chip").forEach(b => b.onclick = () => { fill(p.examples[+b.dataset.i].values); runModel(); });
  $("#fields").innerHTML = p.fields.map(renderField).join("");
  $("#out").hidden = true; $("#out").innerHTML = ""; $("#empty").hidden = false; $("#form-error").hidden = true; $("#timing").textContent = "";
  if (p.status === "needs-artifacts") showError("This model's artifacts are not built yet — run its training script first (see the repo README).");
  $("#m-title").focus?.();
}
function renderField(f) {
  const id = `f-${f.name}`, help = f.help ? `<span class="help" id="${id}-h">${esc(f.help)}</span>` : "", aria = f.help ? `aria-describedby="${id}-h"` : "";
  if (f.type === "checkbox") return `<div class="field check"><input type="checkbox" id="${id}" name="${esc(f.name)}" ${f.default ? "checked" : ""}><label for="${id}">${esc(f.label)}</label>${help}</div>`;
  if (f.type === "select") return `<div class="field"><label for="${id}">${esc(f.label)}</label><select id="${id}" name="${esc(f.name)}" ${aria}>${(f.options || []).map(o => `<option ${o === f.default ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>${help}</div>`;
  if (f.type === "textarea") return `<div class="field"><label for="${id}">${esc(f.label)}</label><textarea id="${id}" name="${esc(f.name)}" ${aria}>${esc(f.default ?? "")}</textarea>${help}</div>`;
  const t = f.type === "password" ? "password" : f.type === "number" ? "number" : "text";
  const attrs = [f.min != null ? `min="${f.min}"` : "", f.max != null ? `max="${f.max}"` : "", f.step != null ? `step="${f.step}"` : (t === "number" ? 'step="any"' : ""), t === "password" ? 'autocomplete="off"' : ""].join(" ");
  return `<div class="field"><label for="${id}">${esc(f.label)}${f.required === false ? ' <span class="muted small">(optional)</span>' : ""}</label><input type="${t}" id="${id}" name="${esc(f.name)}" value="${esc(f.default ?? "")}" ${attrs} ${aria}>${help}</div>`;
}
function fill(values) {
  for (const [k, v] of Object.entries(values)) {
    const el = $(`#f-${k}`); if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v ?? "";
  }
}
function collect() {
  const v = {};
  for (const f of current.fields) { const el = $(`#f-${f.name}`); if (!el) continue; v[f.name] = el.type === "checkbox" ? el.checked : el.value; }
  return v;
}
function showError(msg) { const e = $("#form-error"); e.textContent = msg; e.hidden = false; }

// ---------- run ----------
async function runModel() {
  if (!current) return;
  $("#form-error").hidden = true; $("#run").disabled = true;
  $("#empty").hidden = true; $("#out").hidden = true; $("#loading").hidden = false;
  $("#loading-text").textContent = current.heavy ? "Loading the model (first call can take a minute)…" : "Running…";
  const t0 = performance.now();
  try {
    const r = await fetch(`/api/run/${current.id}`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({values: collect()})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    render(d); $("#timing").textContent = `${Math.round(performance.now() - t0)} ms round-trip · ${d.latency_ms ?? "?"} ms model`;
  } catch (e) { showError(e.message); $("#empty").hidden = false; }
  $("#loading").hidden = true; $("#run").disabled = false;
  const pw = $('#fields input[type="password"]'); if (pw) pw.value = "";      // never keep a key in the DOM after use
}
$("#form").onsubmit = e => { e.preventDefault(); runModel(); };
$("#reset").onclick = () => showModel(current);
$("#back").onclick = () => go(""); $("#home").onclick = () => go("");

// ---------- renderers ----------
function render(d) {
  let h = "";
  if (d.summary) h += `<p class="summary">${esc(d.summary)}</p>`;
  if (d.metrics?.length) h += `<div class="metrics">${d.metrics.map(m => `<div class="metric"><div class="v">${esc(m.value)}</div><div class="l">${esc(m.label)}</div>${m.hint ? `<div class="h">${esc(m.hint)}</div>` : ""}</div>`).join("")}</div>`;
  if (d.text) h += `<div class="section"><h3>Answer</h3><div class="answer">${esc(d.text)}</div></div>`;
  if (d.citations?.length) h += `<div class="section"><h3>Citations</h3>${d.citations.map(c => `<div class="cite">[${c.n}] <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.source)}</a><div class="t">${esc(c.text)}</div></div>`).join("")}</div>`;
  if (d.image || d.image_plain) h += `<div class="section"><h3>Image</h3><div class="imgs">${d.image_plain ? `<figure><img src="${d.image_plain}" alt="Input image"><figcaption>Input</figcaption></figure>` : ""}${d.image ? `<figure><img src="${d.image}" alt="${esc(d.image_caption || "Model output image")}"><figcaption>${esc(d.image_caption || "")}</figcaption></figure>` : ""}</div></div>`;
  if (d.series) h += `<div class="section"><h3>${esc(d.series.title || "Series")}</h3>${lineChart(d.series)}</div>`;
  if (d.bars) h += `<div class="section"><h3>${esc(d.bars.label || "Bars")}</h3>${barChart(d.bars)}</div>`;
  if (d.table) h += `<div class="section"><h3>Table</h3><div class="table-wrap"><table><thead><tr>${d.table.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${d.table.rows.map(r => `<tr>${d.table.columns.map(c => `<td class="${r[c] === "yes" ? "yes" : r[c] === "no" ? "no" : ""}">${fmt(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>`;
  if (d.warnings?.length) h += d.warnings.map(w => `<div class="warn">${esc(w)}</div>`).join("");
  if (d.details) h += `<details><summary>Details / trace</summary><pre>${esc(JSON.stringify(d.details, null, 2))}</pre></details>`;
  const out = $("#out"); out.innerHTML = h; out.hidden = false; out.classList.remove("reveal"); void out.offsetWidth; out.classList.add("reveal");
}

function lineChart(s) {
  const W = 720, H = 300, P = {l: 52, r: 16, t: 12, b: 34};
  const xs = s.x, all = s.series.flatMap(q => q.y.filter(v => v != null));
  if (!all.length) return "";
  let lo = Math.min(...all), hi = Math.max(...all); if (lo === hi) { lo -= 1; hi += 1; } const pad = (hi - lo) * .06; lo -= pad; hi += pad;
  const X = i => P.l + (i / Math.max(xs.length - 1, 1)) * (W - P.l - P.r), Y = v => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);
  const ticks = 5, grid = Array.from({length: ticks + 1}, (_, k) => lo + (hi - lo) * k / ticks);
  let g = grid.map(v => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="currentColor" stroke-opacity=".12"/><text x="${P.l - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end" fill="currentColor" stroke="none">${fmt(+v.toPrecision(4))}</text>`).join("");
  const xt = Math.max(1, Math.round(xs.length / 8));
  g += xs.map((x, i) => i % xt ? "" : `<text x="${X(i)}" y="${H - 10}" font-size="11" text-anchor="middle" fill="currentColor" stroke="none">${esc(x)}</text>`).join("");
  const paths = s.series.map((q, k) => {
    let d = "", pen = false;
    q.y.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; });
    const dash = k === 0 ? "" : k === 1 ? 'stroke-dasharray="4 3"' : "";
    return `<path d="${d}" stroke="${PALETTE[k % PALETTE.length]}" stroke-width="${k < 2 ? 2.2 : 1.6}" fill="none" ${dash}><title>${esc(q.name)}</title></path>`;
  }).join("");
  const legend = s.series.map((q, k) => `<span style="--c:${PALETTE[k % PALETTE.length]}">${esc(q.name)}</span>`).join("");
  return `<div class="chart-wrap" style="aspect-ratio:${W}/${H}"><svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Line chart: ${esc(s.series.map(q => q.name).join(", "))}">${g}${paths}</svg></div><div class="legend">${legend}</div>`;
}
function barChart(b) {
  const W = 720, H = 240, P = {l: 52, r: 16, t: 12, b: 48}, n = b.labels.length;
  const hi = Math.max(...b.values, 0), lo = Math.min(...b.values, 0), span = (hi - lo) || 1;
  const bw = (W - P.l - P.r) / n, Y = v => P.t + (1 - (v - lo) / span) * (H - P.t - P.b);
  const bars = b.values.map((v, i) => `<rect x="${(P.l + i * bw + bw * .15).toFixed(1)}" y="${Math.min(Y(v), Y(0)).toFixed(1)}" width="${(bw * .7).toFixed(1)}" height="${Math.abs(Y(v) - Y(0)).toFixed(1)}" rx="3" fill="${PALETTE[0]}"><title>${esc(b.labels[i])}: ${fmt(v)}</title></rect>
     <text x="${(P.l + i * bw + bw / 2).toFixed(1)}" y="${(Math.min(Y(v), Y(0)) - 4).toFixed(1)}" font-size="10" text-anchor="middle" fill="currentColor" stroke="none">${fmt(+v.toPrecision(3))}</text>
     <text x="${(P.l + i * bw + bw / 2).toFixed(1)}" y="${H - 30}" font-size="10" text-anchor="middle" fill="currentColor" stroke="none" transform="rotate(${n > 8 ? -30 : 0} ${(P.l + i * bw + bw / 2).toFixed(1)} ${H - 30})">${esc(String(b.labels[i]).slice(0, 14))}</text>`).join("");
  return `<div class="chart-wrap" style="aspect-ratio:${W}/${H}"><svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Bar chart: ${esc(b.label || "")}"><line x1="${P.l}" x2="${W - P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="currentColor" stroke-opacity=".3"/>${bars}</svg></div>`;
}

// ---------- boot ----------
(async () => {
  try {
    const d = await (await fetch("/api/plugins")).json();
    plugins = d.plugins.sort((a, b) => a.category.localeCompare(b.category) || a.title.localeCompare(b.title));
    $("#status").textContent = `${plugins.length} models`; $("#status").className = "pill ok";
    const errs = Object.entries(d.load_errors || {});
    if (errs.length) $("#load-errors").textContent = "Plugins that failed to load: " + errs.map(([k, v]) => `${k} (${v})`).join("; ");
    route();
  } catch (e) { $("#status").textContent = "server unreachable"; $("#status").className = "pill bad"; }
})();
