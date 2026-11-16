"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const qs = o => new URLSearchParams(o).toString();
const FRIENDLY = {dense: "meaning", bm25: "exact words", graph: "concept map"};
const MODES = {standard: ["dense"], hybrid: ["dense", "bm25"], graph: ["dense", "bm25", "graph"]};
let CORPORA = [], QUESTIONS = [], STATUS = null, MAP = null, SELECTED = null, HIDDEN = new Set(), PANEL = {};

/* ---------------------------------------------------------------- helpers */
function banner(msg, info) {
  const b = $("banner");
  b.textContent = msg || "";
  b.className = "banner" + (msg ? " show" : "") + (info ? " info" : "");
}
async function call(url, body, method) {
  try {
    const opts = body !== undefined ? {method: method || "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : (method ? {method} : {});
    const r = await fetch(url, opts);
    let d = null;
    try { d = await r.json(); } catch (e) { /* no body */ }
    if (!r.ok) {
      const msg = d && d.detail ? (typeof d.detail === "string" ? d.detail : "Some value is not valid: " + d.detail.map(x => (x.loc || []).slice(-1) + " " + x.msg).join("; ")) : "Error " + r.status;
      return {ok: false, error: msg, trace_id: d && d.trace_id};
    }
    return {ok: true, data: d};
  } catch (e) {
    return {ok: false, error: "Cannot reach the server. Is it still running?"};
  }
}
async function get(url) { const r = await call(url); if (!r.ok) banner(r.error); else banner(""); return r; }
const cfg = () => ({corpus: $("corpus").value, chunk_size: +$("cs").value, overlap_percent: +$("ov").value});
const retr = () => ({retrievers: MODES[$("mode").value], candidates: +$("cand").value, rrf_k: +$("rrfk").value, weights: {graph: +$("gw").value}});
const fmtMs = ms => ms == null ? "" : ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
const fmtCost = c => c == null ? "" : c === 0 ? "$0" : c < 0.0001 ? "< $0.0001" : "$" + c.toFixed(4);
const plural = (n, w) => n + " " + w + (n === 1 ? "" : "s");
const traceLink = id => `<a data-trace="${esc(id)}">Details ▸ trace ${esc(id)}</a>`;
const chunkLink = id => `<a data-chunk="${esc(id)}">${esc(id)}</a>`;
const nice = s => String(s).toLowerCase().replace(/_/g, " ");
const pretty = o => Object.entries(o || {}).map(([k, n]) => plural(n, k.toLowerCase())).join(", ");

/* ---------------------------------------------------------------- tabs */
const TABS = ["ask", "docs", "map", "history", "eval", "words"];
function show(tab) {
  TABS.forEach(t => $("t-" + t).hidden = t !== tab);
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.t === tab));
  banner("");
  if (tab === "docs") loadDocs();
  if (tab === "map") loadMap();
  if (tab === "history") loadHistory();
  if (tab === "words") vocab();
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => show(b.dataset.t));

/* ---------------------------------------------------------------- corpora and status */
async function loadCorpora(keep) {
  const r = await get("/api/corpora");
  if (!r.ok) return;
  CORPORA = r.data;
  const cur = keep || $("corpus").value;
  $("corpus").innerHTML = CORPORA.map(c => `<option value="${esc(c.name)}">${esc(c.name)} (${plural(c.documents, "document")})</option>`).join("");
  if (CORPORA.some(c => c.name === cur)) $("corpus").value = cur;
  await onCorpus();
}
async function onCorpus() {
  const r = await call("/api/questions?" + qs({corpus: $("corpus").value}));
  QUESTIONS = r.ok ? r.data : [];
  const opts = '<option value="">Try an example question…</option>' + QUESTIONS.map(q => `<option value="${esc(q.id)}">${esc(q.question)}</option>`).join("");
  $("qpick").innerHTML = opts;
  $("qpick").style.display = QUESTIONS.length ? "" : "none";
  $("wpick").innerHTML = '<option value="">- pick a question from the set -</option>' + QUESTIONS.map(q => `<option value="${esc(q.id)}">${esc(q.id)} ${esc(q.question)}</option>`).join("");
  await refreshStatus();
}
async function refreshStatus() {
  const r = await call(`/api/corpora/${encodeURIComponent($("corpus").value)}/status?` + qs({chunk_size: +$("cs").value, overlap_percent: +$("ov").value}));
  STATUS = r.ok ? r.data : null;
  const hint = $("ask-hint");
  let msg = "";
  if (STATUS && !STATUS.documents.length) msg = "This collection has no documents yet. Add some in the “My documents” tab.";
  else if (STATUS && !STATUS.dense_ready) msg = "These documents are not prepared yet, so the first question will take longer (the search index is built once). You can prepare them now in “My documents”.";
  hint.textContent = msg;
  hint.className = "banner info" + (msg ? " show" : "");
  const g = $("mode").querySelector('[value="graph"]');
  const hasGraph = STATUS && STATUS.graph.nodes > 0;
  g.disabled = !hasGraph;
  g.textContent = "meaning + exact words + concept map" + (hasGraph ? "" : " (build the concept map first)");
  if (!hasGraph && $("mode").value === "graph") $("mode").value = "hybrid";
}
$("corpus").onchange = onCorpus;
$("cs").onchange = $("ov").onchange = refreshStatus;
$("qpick").onchange = () => { const q = QUESTIONS.find(x => x.id === $("qpick").value); if (q) $("q").value = q.question; };

/* ---------------------------------------------------------------- ask */
let askTimer = null;
async function ask() {
  const q = $("q").value.trim();
  if (!q) { banner("Type a question first."); return; }
  banner("");
  $("ask-btn").disabled = true;
  const t0 = Date.now();
  $("ask-out").innerHTML = '<div class="card muted">Searching your documents… <span id="el">0</span> s</div>';
  askTimer = setInterval(() => { const e = $("el"); if (e) e.textContent = Math.round((Date.now() - t0) / 1000); }, 500);
  const r = await call("/api/ask", {...cfg(), ...retr(), question: q, k: +$("k").value});
  clearInterval(askTimer);
  $("ask-btn").disabled = false;
  if (!r.ok) {
    $("ask-out").innerHTML = `<div class="card"><b class="bad">No answer this time.</b> ${esc(r.error)} ${r.trace_id ? "<div class='small'>" + traceLink(r.trace_id) + "</div>" : ""}</div>`;
    return;
  }
  const d = r.data, s = d.summary || {};
  $("ask-out").innerHTML = `
   <div class="card answer"><h3 style="margin-top:0">Answer</h3>
    ${d.in_documents ? "" : '<span class="chip warn">The documents do not contain the answer</span>'}
    <p>${esc(d.answer)}</p>
    ${Object.keys(d.privacy || {}).length ? `<span class="chip ok">🔒 Personal data hidden from the AI: ${esc(pretty(d.privacy))}</span>` : ""}
   </div>
   <div class="small muted">${plural(s.llm_calls || 0, "AI call")} · ${(s.tokens_in || 0) + (s.tokens_out || 0)} tokens · ≈ ${fmtCost(s.cost_usd)} · ${fmtMs(s.duration_ms)} · ${traceLink(d.trace_id)}</div>
   <h2>Sources</h2><div class="card">${d.chunks.map(sourceHtml).join("") || '<span class="muted">No passage found.</span>'}</div>`;
}
function sourceHtml(c) {
  const how = Object.entries(c.found_by || {}).map(([n, r]) => `<span class="chip">${FRIENDLY[n] || n} #${r}</span>`).join("");
  const path = c.path ? `<div class="small muted"><b>concept path</b><br>${c.path.map(esc).join("<br>")}</div>` : "";
  return `<div class="src"><b>${esc(c.source)}</b> <span class="muted small">passage ${c.chunk_index}</span> ${how}
   <details><summary>${esc(c.text.replace(/\s+/g, " ").slice(0, 150))}…</summary><pre>${esc(c.text)}</pre>${path}</details></div>`;
}
$("ask-btn").onclick = ask;
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); });

/* ---------------------------------------------------------------- my documents */
async function loadDocs() {
  const r = await get("/api/corpora");
  if (!r.ok) return;
  CORPORA = r.data;
  const q = qs({chunk_size: +$("cs").value, overlap_percent: +$("ov").value});
  const sts = await Promise.all(CORPORA.map(c => call(`/api/corpora/${encodeURIComponent(c.name)}/status?${q}`)));
  $("docs-out").innerHTML = CORPORA.map((c, i) => docsCard(c, sts[i].ok ? sts[i].data : null)).join("");
}
function docsCard(c, st) {
  if (!st) return `<div class="card"><b>${esc(c.name)}</b> <span class="muted">(status unavailable)</span></div>`;
  const g = st.graph, need = Math.max(0, st.chunks - g.chunks_done);
  const chips = `<span class="chip">${plural(st.documents.length, "document")}</span><span class="chip">${plural(st.chunks, "passage")}</span>
   <span class="chip ${st.dense_ready ? "ok" : ""}">meaning index ${st.dense_ready ? "ready" : "not built"}</span>
   <span class="chip ${st.bm25_ready ? "ok" : ""}">exact-words index ${st.bm25_ready ? "ready" : "not built"}</span>
   <span class="chip ${g.nodes ? "ok" : ""}">concept map ${g.nodes ? `${plural(g.nodes, "concept")}, ${plural(g.edges, "link")}` : "not built"}${g.failed ? `, ${g.failed} passage(s) failed` : ""}</span>`;
  const docs = st.documents.map(f => `<li>${esc(f)} ${st.user_corpus ? `<a class="small bad" data-rm="${esc(c.name)}|${esc(f)}">remove</a>` : ""}</li>`).join("") || '<li class="muted">no document yet</li>';
  const upload = st.user_corpus ? `<div class="drop" data-drop="${esc(c.name)}">Drop .md or .txt files here or <label class="link"><input type="file" multiple accept=".md,.txt,text/plain,text/markdown" data-file="${esc(c.name)}" hidden>choose files</label></div>` : '<div class="muted small">Provided collection: read-only.</div>';
  return `<div class="card" data-corpus="${esc(c.name)}"><div class="row" style="justify-content:space-between"><b>${esc(c.name)}</b>
    ${st.user_corpus ? `<a class="small bad" data-delcorpus="${esc(c.name)}">delete collection</a>` : ""}</div>
   <div>${chips}</div><ul>${docs}</ul>${upload}
   <div class="row" style="margin-top:10px">
    <button class="btn" data-index="${esc(c.name)}" ${st.documents.length ? "" : "disabled"}>Prepare for search</button>
    <label class="small"><input type="checkbox" data-withgraph="${esc(c.name)}" ${need ? "" : "disabled"}> also build the concept map ${need ? `(uses the AI: up to ${plural(need, "call")}, one per passage)` : st.chunks ? "(already built)" : ""}</label></div>
   <div data-job="${esc(c.name)}" style="margin-top:8px"></div></div>`;
}
async function uploadFiles(name, files) {
  let ok = 0, errs = [];
  for (const f of files) {
    const r = await call(`/api/corpora/${encodeURIComponent(name)}/documents`, {filename: f.name, content: await f.text()}, "PUT");
    if (r.ok) ok++; else errs.push(f.name + ": " + r.error);
  }
  banner(errs.length ? errs.join(" · ") : `${plural(ok, "document")} added. Prepare the collection for search to use ${ok === 1 ? "it" : "them"}.`, !errs.length);
  await loadDocs(); await loadCorpora(name);
}
async function runIndex(name) {
  const card = document.querySelector(`[data-corpus="${CSS.escape(name)}"]`);
  const box = card.querySelector(`[data-job]`), withGraph = card.querySelector("[data-withgraph]").checked;
  card.querySelector("[data-index]").disabled = true;
  const r = await call(`/api/corpora/${encodeURIComponent(name)}/index`, {chunk_size: +$("cs").value, overlap_percent: +$("ov").value, graph: withGraph});
  if (!r.ok) { banner(r.error); card.querySelector("[data-index]").disabled = false; return; }
  const poll = async () => {
    const j = await call("/api/jobs/" + r.data.job_id);
    if (!j.ok) { box.innerHTML = `<span class="bad">${esc(j.error)}</span>`; return; }
    const d = j.data, pct = d.total ? Math.round(100 * d.done / d.total) : (d.status === "done" ? 100 : 5);
    if (d.status === "running") {
      box.innerHTML = `<div class="small">${esc(d.stage)}${d.total ? ` — ${d.done}/${d.total}` : "…"}</div><div class="progress"><i style="width:${pct}%"></i></div>`;
      setTimeout(poll, 700);
    } else if (d.status === "error") {
      box.innerHTML = `<span class="bad">Indexing stopped: ${esc(d.error)}</span> ${d.trace_id ? traceLink(d.trace_id) : ""}`;
      card.querySelector("[data-index]").disabled = false;
    } else {
      box.innerHTML = `<span class="ok">${d.status === "partial" ? "Done, but some passages failed: run it again to retry only those." : "Ready."}</span> ${traceLink(d.trace_id)}`;
      await refreshStatus();
      setTimeout(loadDocs, 1500);
    }
  };
  poll();
}
$("create-btn").onclick = async () => {
  const r = await call("/api/corpora", {name: $("newname").value});
  if (!r.ok) { banner(r.error); return; }
  $("newname").value = "";
  banner(`Collection “${r.data.name}” created. Add documents to it below.`, true);
  await loadCorpora(r.data.name); await loadDocs();
};

/* ---------------------------------------------------------------- trace viewer ("Details") */
async function openTrace(id) {
  const r = await call("/api/traces/" + encodeURIComponent(id));
  if (!r.ok) { banner(r.error); return; }
  const t = r.data, a = t.attrs || {}, cfgs = a.config || {};
  PANEL = {corpus: a.corpus && a.corpus.split("@")[0], cs: cfgs.chunk_size || 800, ov: cfgs.chunk_overlap || 20};
  const spans = t.spans, total = t.duration_ms || 1;
  const by = {};
  spans.forEach(s => by[s.id] = {...s, kids: []});
  const roots = [];
  spans.forEach(s => (s.parent && by[s.parent] ? by[s.parent].kids : roots).push(by[s.id]));
  const tt = t.totals;
  $("panel").innerHTML = `
   <div class="row" style="justify-content:space-between"><h2 style="margin:0">Details · ${esc(t.kind)} <span class="muted small">trace ${esc(t.trace_id)}</span></h2><button class="btn ghost" data-close>Close</button></div>
   <div style="margin:6px 0"><span class="chip ${["answered", "found", "indexed", "shown", "ok"].includes(t.outcome) ? "ok" : t.outcome === "abstained" || t.outcome === "partial" ? "warn" : "bad"}">${esc(t.outcome)}</span>
    <span class="chip">${fmtMs(t.duration_ms)}</span><span class="chip">${esc(new Date(t.started_at).toLocaleString())}</span>
    ${a.fingerprint ? `<span class="chip" title="Same fingerprint = same pipeline configuration">config ${esc(a.fingerprint)}</span>` : ""}</div>
   ${a.question ? `<div class="card"><div class="small muted">Question (personal data replaced)</div>${esc(a.question)}</div>` : ""}
   ${a.answer ? `<div class="card"><div class="small muted">Answer as written by the model</div>${esc(a.answer)}</div>` : ""}
   ${a.error ? `<div class="card bad">${esc(a.error)}</div>` : ""}
   <div class="kpis"><div class="kpi"><b>${tt.llm_calls}</b>AI calls</div><div class="kpi"><b>${tt.tokens_in} / ${tt.tokens_out}</b>tokens in / out</div><div class="kpi"><b>${fmtCost(tt.cost_usd)}</b>estimated cost</div></div>
   <h3>Steps (each bar shows when the step ran and how long it took)</h3>
   <div>${roots.map(n => spanHtml(n, total)).join("") || '<span class="muted">No step recorded.</span>'}</div>
   ${(a.retrieved || []).length ? `<h3>Passages given to the model</h3>${a.retrieved.map(chunkLink).join(" · ")}` : ""}
   <div id="chunkview"></div>
   <details style="margin-top:14px"><summary>Configuration recorded with this trace</summary><pre class="passage">${esc(JSON.stringify(cfgs, null, 1))}</pre></details>`;
  document.body.classList.add("panel");
}
function spanHtml(n, total) {
  const left = Math.min(99, 100 * n.start_ms / total), w = Math.max(0.5, Math.min(100 - left, 100 * (n.duration_ms || 0) / total));
  return `<div class="span ${n.status === "error" ? "err" : ""}"><div><span class="nm">${esc(n.name)}</span> ${n.status === "error" ? "⚠" : ""}
     <div class="small muted">${summarize(n)}</div></div>
     <div><div class="small" style="text-align:right">${fmtMs(n.duration_ms)}</div><div class="bar"><i style="left:${left}%;width:${w}%"></i></div></div></div>` +
    (n.kids.length ? `<div class="kids">${kidsHtml(n.kids, total)}</div>` : "");
}
function kidsHtml(kids, total) {
  if (kids.length > 8 && kids.every(k => k.name === kids[0].name)) {
    const hit = kids.filter(k => k.attrs.cache_hit).length, retried = kids.filter(k => k.attrs.retried).length,
      none = kids.filter(k => k.attrs.note === "no triples").length, failed = kids.filter(k => k.status === "error").length;
    return `<details><summary>${esc(kids[0].name)} × ${kids.length}: ${hit} reused from cache, ${retried} retried, ${none} with no triples, ${failed} failed</summary>${kids.map(k => spanHtml(k, total)).join("")}</details>`;
  }
  return kids.map(k => spanHtml(k, total)).join("");
}
function summarize(n) {
  const a = n.attrs || {}, top = (l) => (l || []).slice(0, 3).map(x => `${chunkLink(x.id)}${x.score != null ? " (" + x.score + ")" : ""}`).join(", ");
  if (n.name.startsWith("retrieve.")) return `${a.n_results} passages found · best: ${top(a.top)} <span class="muted">(${esc(a.score_name)})</span>`;
  if (n.name === "retrieve") return `${esc((a.retrievers || []).map(r => FRIENDLY[r] || r).join(" + "))} · top ${a.k} of ${a.candidates} candidates`;
  if (n.name === "fusion") return `RRF k=${a.rrf_k} → ` + (a.fused || []).map(f => `${chunkLink(f.id)} [${Object.entries(f.found_by).map(([r, k]) => r + "#" + k).join(" ")}]`).join(", ");
  if (n.name === "llm.call") return `asked ${esc(a["gen_ai.request.model"])} → answered by ${esc(a["gen_ai.response.model"] || "?")} · tokens in ${a["gen_ai.usage.input_tokens"] ?? "?"} / out ${a["gen_ai.usage.output_tokens"] ?? "?"} · ${fmtCost(a.cost_usd)} · finish: ${esc(a.finish_reason || "?")}${a.error ? " · " + esc(a.error) : ""}`;
  if (n.name === "prompt.build") return `${(a.chunk_ids || []).length} passages, ${a.prompt_chars} characters, prompt ${esc(a.prompt_version)}${Object.keys(a.personal_data_hidden || {}).length ? " · 🔒 hidden: " + esc(pretty(a.personal_data_hidden)) : ""}`;
  if (n.name === "load+chunk") return `${a.documents} documents → ${a.chunks} passages`;
  if (n.name === "graph.chunk") return `${chunkLink(a.chunk)} · ${a.cache_hit ? "reused from cache" : `${a.triples ?? 0} triples, ${a.llm_calls ?? 0} call(s)${a.retried ? ", retried" : ""}${a.dropped ? ", " + a.dropped + " dropped" : ""}`}${a.note ? " · <b>" + esc(a.note) + "</b>" : ""}${a.error ? " · " + esc(a.error) : ""}`;
  if (n.name === "graph.extract") return `${esc(a.profile || "")} schema · ${a.nodes ?? "?"} concepts, ${a.edges ?? "?"} links, ${a.failed ?? 0} passages failed`;
  const skip = new Set(["error"]), kv = Object.entries(a).filter(([k]) => !skip.has(k)).map(([k, v]) => `${esc(k)}=${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(" · ");
  return kv + (a.error ? ` <span class="bad">${esc(a.error)}</span>` : "");
}
async function openChunk(id) {
  const r = await call("/api/chunk?" + qs({corpus: PANEL.corpus, chunk_id: id, chunk_size: PANEL.cs, overlap_percent: PANEL.ov}));
  const box = document.body.classList.contains("panel") ? $("chunkview") : $("mapchunk");
  if (!box) return;
  box.innerHTML = r.ok ? `<h3>${esc(id)}</h3><pre class="passage">${esc(r.data.text)}</pre>` : `<p class="bad">${esc(r.error)}</p>`;
  box.scrollIntoView({block: "nearest"});
}
const closePanel = () => document.body.classList.remove("panel");

/* ---------------------------------------------------------------- history and metrics */
async function loadHistory() {
  const [m, t] = await Promise.all([call("/api/metrics?days=7"), call("/api/traces?" + qs({limit: 50, kind: $("tkind").value}))]);
  if (m.ok) {
    const d = m.data, o = d.outcomes;
    $("metrics").innerHTML = `<h2>Last 7 days</h2><div class="kpis">
      <div class="kpi"><b>${d.questions}</b>questions</div>
      <div class="kpi"><b>${o.answered || 0}</b>answered</div><div class="kpi"><b>${o.abstained || 0}</b>abstained</div>
      <div class="kpi"><b>${(o.error || 0) + (o.invalid_output || 0)}</b>failed or invalid</div>
      <div class="kpi"><b>${fmtMs(d.latency_ms.p50) || "–"}</b>median time (p50)</div><div class="kpi"><b>${fmtMs(d.latency_ms.p95) || "–"}</b>p95 time</div>
      <div class="kpi"><b>${d.index_runs}</b>indexing runs</div></div>
     <details><summary>Time per step and tokens per day (computed from the traces)</summary>
      <table><tr><th>step</th><th>runs</th><th>p50</th><th>p95</th></tr>${Object.entries(d.steps).filter(([n]) => !n.startsWith("graph.chunk")).map(([n, s]) => `<tr><td>${esc(n)}</td><td>${s.n}</td><td>${fmtMs(s.p50)}</td><td>${fmtMs(s.p95)}</td></tr>`).join("")}</table>
      <table style="margin-top:10px"><tr><th>day</th><th>requests</th><th>tokens</th><th>estimated cost</th></tr>${Object.entries(d.per_day).map(([day, s]) => `<tr><td>${day}</td><td>${s.requests}</td><td>${s.tokens}</td><td>${fmtCost(s.cost_usd)}</td></tr>`).join("")}</table></details>`;
  }
  if (t.ok) {
    $("traces").innerHTML = t.data.length ? `<table><tr><th>when</th><th>kind</th><th>what</th><th>result</th><th>time</th><th>tokens</th><th></th></tr>${t.data.map(x =>
      `<tr><td>${esc(new Date(x.started_at).toLocaleString())}</td><td>${esc(x.kind)}</td><td>${esc((x.title || "").slice(0, 70))}</td><td>${esc(x.outcome)}</td><td>${fmtMs(x.duration_ms)}</td><td>${x.totals.tokens_in + x.totals.tokens_out}</td><td>${traceLink(x.trace_id)}</td></tr>`).join("")}</table>` : '<p class="muted">Nothing yet: ask a question, and its trace appears here.</p>';
  }
}
$("tkind").onchange = loadHistory;

/* ---------------------------------------------------------------- concept map explorer */
const PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#9d755d", "#72b7b2", "#eeca3b"];
const color = (types, t) => PALETTE[types.indexOf(t) % PALETTE.length];
async function loadMap() {
  const r = await get("/api/graph/map?" + qs(cfg()));
  if (!r.ok) { MAP = null; $("mapgrid").hidden = true; $("mapempty").innerHTML = ""; $("maplegend").innerHTML = ""; $("mapinfo").textContent = ""; return; }
  if (!r.data.available) {
    MAP = null;
    $("mapgrid").hidden = true; $("maplegend").innerHTML = ""; $("mapinfo").textContent = "";
    $("mapempty").innerHTML = `<div class="card"><b>No concept map for these documents yet.</b><p class="muted">${esc(r.data.message)} The map shows what the documents talk about (services, teams, products, people…) and how those things are linked.</p><button class="btn" data-tab="docs">Go to My documents</button></div>`;
    return;
  }
  MAP = r.data.map; MAP.types = [...new Set(MAP.nodes.map(n => n.type))].sort(); MAP.stats = r.data.stats;
  SELECTED = null;
  $("mapempty").innerHTML = ""; $("mapgrid").hidden = false;
  const s = MAP.stats;
  $("mapinfo").textContent = `${s.nodes} concepts, ${s.edges} links, built from ${s.chunks_done} of ${s.chunks_total} passages${MAP.total_nodes > MAP.shown_nodes ? ` · showing the ${MAP.shown_nodes} best connected` : ""}`;
  $("maplegend").innerHTML = MAP.types.map(t => `<span data-type="${esc(t)}" class="${HIDDEN.has(t) ? "off" : ""}"><i class="dot" style="background:${color(MAP.types, t)}"></i>${esc(t)}</span>`).join("") + '<span class="muted">click a type to hide it</span>';
  layout(); drawMap();
  $("mapside").innerHTML = '<span class="muted">Click a concept to see what it is linked to and where each link comes from.</span>';
}
function components() {
  const parent = {}, find = x => parent[x] === x ? x : (parent[x] = find(parent[x]));
  MAP.nodes.forEach(n => parent[n.id] = n.id);
  MAP.edges.forEach(e => { parent[find(e.src)] = find(e.dst); });
  const groups = {};
  MAP.nodes.forEach(n => (groups[find(n.id)] = groups[find(n.id)] || []).push(n.id));
  return Object.values(groups).sort((a, b) => b.length - a.length);
}
function forceLayout(ids, w, h) {
  // Fruchterman-Reingold inside a w x h box: linked concepts attract, all concepts repel
  const pos = {}, n = ids.length, set = new Set(ids), edges = MAP.edges.filter(e => set.has(e.src) && set.has(e.dst)), k = Math.sqrt(w * h / n) * 0.8;
  ids.forEach((id, i) => { const a = 2 * Math.PI * i / n; pos[id] = {x: w / 2 + Math.cos(a) * w * 0.3, y: h / 2 + Math.sin(a) * h * 0.3, dx: 0, dy: 0}; });
  if (n === 1) { pos[ids[0]] = {x: w / 2, y: h / 2}; return pos; }
  let temp = w / 6;
  for (let it = 0; it < 250; it++) {
    for (const id of ids) { pos[id].dx = 0; pos[id].dy = 0; }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const p = pos[ids[i]], q = pos[ids[j]]; const dx = p.x - q.x, dy = p.y - q.y, d = Math.max(0.5, Math.hypot(dx, dy)), f = k * k / d;
      p.dx += dx / d * f; p.dy += dy / d * f; q.dx -= dx / d * f; q.dy -= dy / d * f;
    }
    for (const e of edges) {
      const p = pos[e.src], q = pos[e.dst]; const dx = p.x - q.x, dy = p.y - q.y, d = Math.max(0.5, Math.hypot(dx, dy)), f = d * d / k;
      p.dx -= dx / d * f; p.dy -= dy / d * f; q.dx += dx / d * f; q.dy += dy / d * f;
    }
    for (const id of ids) {
      const p = pos[id]; p.dx += (w / 2 - p.x) * 0.03; p.dy += (h / 2 - p.y) * 0.03;
      const d = Math.max(0.5, Math.hypot(p.dx, p.dy)), m = Math.min(d, temp);
      p.x += p.dx / d * m; p.y += p.dy / d * m;
    }
    temp *= 0.985;
  }
  const xs = ids.map(i => pos[i].x), ys = ids.map(i => pos[i].y), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  for (const id of ids) {  // stretch to fill the box
    pos[id].x = 25 + (pos[id].x - x0) / ((x1 - x0) || 1) * (w - 50);
    pos[id].y = 25 + (pos[id].y - y0) / ((y1 - y0) || 1) * (h - 50);
  }
  return pos;
}
function layout() {
  // each cluster of linked concepts is laid out on its own, then the clusters are packed in rows (the position on the page
  // has no meaning, only the links do)
  const W = 900, H = 540, GAP = 40, ROW = 1000, boxes = components().map(ids => {
    const side = Math.sqrt(ids.length), w = Math.max(210, side * 120), h = Math.max(110, side * 85);
    return {ids, w, h, pos: forceLayout(ids, w, h)};
  });
  let x = 0, y = 0, rowH = 0, maxW = 0;
  for (const b of boxes) {
    if (x > 0 && x + b.w > ROW) { x = 0; y += rowH + GAP; rowH = 0; }
    b.x = x; b.y = y; x += b.w + GAP; rowH = Math.max(rowH, b.h); maxW = Math.max(maxW, x - GAP);
  }
  const totalH = y + rowH, s = Math.min((W - 80) / maxW, (H - 50) / totalH, 1.6), ox = (W - maxW * s) / 2, oy = (H - totalH * s) / 2;
  MAP.pos = {};
  for (const b of boxes) for (const id of b.ids) MAP.pos[id] = {x: ox + (b.x + b.pos[id].x) * s, y: oy + (b.y + b.pos[id].y) * s};
}
function drawMap() {
  const q = $("mapfind").value.trim().toLowerCase(), vis = n => !HIDDEN.has(n.type), N = MAP.nodes.filter(vis), ids = new Set(N.map(n => n.id));
  const nb = new Set(); if (SELECTED) { nb.add(SELECTED); MAP.edges.forEach(e => { if (e.src === SELECTED) nb.add(e.dst); if (e.dst === SELECTED) nb.add(e.src); }); }
  const showLabel = n => N.length <= 60 || n.degree >= 3 || nb.has(n.id) || (q && n.name.toLowerCase().includes(q));
  let svg = "";
  for (const e of MAP.edges) {
    if (!ids.has(e.src) || !ids.has(e.dst)) continue;
    const p = MAP.pos[e.src], r = MAP.pos[e.dst], hot = SELECTED && (e.src === SELECTED || e.dst === SELECTED);
    svg += `<line class="edge ${hot ? "hot" : ""} ${SELECTED && !hot ? "dim" : ""}" x1="${p.x}" y1="${p.y}" x2="${r.x}" y2="${r.y}"></line>`;
    if (hot) svg += `<text x="${(p.x + r.x) / 2}" y="${(p.y + r.y) / 2}" text-anchor="middle" style="font-size:12px;fill:var(--accent)">${esc(nice(e.rel))}</text>`;
  }
  for (const n of N) {
    const p = MAP.pos[n.id], match = q && n.name.toLowerCase().includes(q), r = 9 + Math.min(12, n.degree * 1.4);
    svg += `<circle data-node="${esc(n.id)}" cx="${p.x}" cy="${p.y}" r="${r}" fill="${color(MAP.types, n.type)}" class="${n.id === SELECTED ? "sel" : ""} ${(SELECTED && !nb.has(n.id)) || (q && !match) ? "dim" : ""}"><title>${esc(n.name)} (${esc(n.type)})</title></circle>`;
    if (showLabel(n)) svg += `<text x="${p.x}" y="${p.y + r + 11}" text-anchor="middle" class="${(SELECTED && !nb.has(n.id)) || (q && !match) ? "dim" : ""}">${esc(n.name.length > 22 ? n.name.slice(0, 21) + "…" : n.name)}</text>`;
  }
  $("mapsvg").innerHTML = svg;
}
async function selectNode(id) {
  SELECTED = id; drawMap();
  const r = await call("/api/graph/node?" + qs({...cfg(), node_id: id}));
  if (!r.ok) { $("mapside").innerHTML = `<span class="bad">${esc(r.error)}</span>`; return; }
  const d = r.data; PANEL = {corpus: cfg().corpus, cs: +$("cs").value, ov: +$("ov").value};
  const links = d.links.map(l => `<li>${l.direction === "out" ? `<b>${esc(nice(l.relation))}</b> → ` : `← <b>${esc(nice(l.relation))}</b> from `}<a data-node="${esc(l.other.id)}">${esc(l.other.name)}</a> <span class="muted small">(${esc(l.other.type)})</span>
     <details><summary class="small">where this comes from (${plural(l.evidence.length, "passage")})</summary>${l.evidence.map(e => `<div class="small">“${esc(e.quote)}” — ${chunkLink(e.chunk_id)}</div>`).join("")}</details></li>`).join("");
  const byDoc = {}; d.chunks.forEach(c => (byDoc[c.source] = byDoc[c.source] || []).push(c));
  $("mapside").innerHTML = `<h3 style="margin-top:0">${esc(d.name)}</h3><span class="chip" style="background:${color(MAP.types, d.type)};color:#fff">${esc(d.type)}</span>
   ${d.also_called.length ? `<div class="small muted">also called: ${d.also_called.map(esc).join(", ")}</div>` : ""}
   <h3>Linked to</h3><ul style="padding-left:18px;margin:0">${links || '<li class="muted">no link</li>'}</ul>
   <h3>Documents that mention it</h3>${Object.entries(byDoc).map(([s, cs]) => `<details><summary>${esc(s)} <span class="muted small">(${plural(cs.length, "passage")})</span></summary>${cs.map(c => `<pre class="passage">${esc(c.text)}</pre>`).join("")}</details>`).join("") || '<span class="muted">none</span>'}
   <div class="row" style="margin-top:12px"><button class="btn" data-askabout="${esc(d.name)}">Ask a question about it</button></div><div id="mapchunk"></div>`;
}
$("mapsvg").addEventListener("click", e => { const c = e.target.closest("[data-node]"); if (c) selectNode(c.dataset.node); });
$("mapfind").oninput = () => { if (MAP) drawMap(); };
$("mapfind").onkeydown = e => { if (e.key === "Enter" && MAP) { const q = $("mapfind").value.trim().toLowerCase(), n = MAP.nodes.find(x => x.name.toLowerCase().includes(q)); if (n) selectNode(n.id); } };
$("maplegend").addEventListener("click", e => { const s = e.target.closest("[data-type]"); if (!s) return; HIDDEN.has(s.dataset.type) ? HIDDEN.delete(s.dataset.type) : HIDDEN.add(s.dataset.type); s.classList.toggle("off"); drawMap(); });

/* ---------------------------------------------------------------- evaluation and words (V2/V3 tools) */
const table = (rows, head) => `<table><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`;
$("eval-btn").onclick = async () => {
  $("eval-out").innerHTML = "Running…";
  const r = await call("/api/eval/run", {...cfg(), retrievers: $("evmode").value.split(","), candidates: +$("cand").value, rrf_k: +$("rrfk").value, weights: {graph: +$("gw").value}, with_answers: $("wa").checked, answer_k: +$("k").value});
  if (!r.ok) { $("eval-out").innerHTML = ""; banner(r.error); return; }
  const d = r.data, ks = d.meta.ks, s = d.summary, c = s.cost_per_question;
  const rows = [["OVERALL", s.overall], ...Object.entries(s.by_type)].map(([n, b]) => [n, b.n, ...ks.flatMap(k => [b[k].hit_rate, b[k].recall])]);
  let h = `<p>retrievers: <b>${d.meta.retrievers.join(" + ")}</b> · ${d.meta.n_chunks} chunks · ${d.meta.n_scored} scored questions · saved to <code>${esc(d.saved_to)}</code></p>
   <p>cost per question: retrieval ${c.retrieval_ms} ms · LLM calls ${c.llm_calls} · tokens ${c.tokens} · answer time ${c.answer_ms} ms</p>` + table(rows, ["type", "n", ...ks.flatMap(k => ["hit@" + k, "rec@" + k])]);
  if (d.unreachable_evidence.length) h += `<p class="bad">${d.unreachable_evidence.length} evidence quote(s) cut by a chunk boundary: ${d.unreachable_evidence.map(u => u.id).join(", ")}</p>`;
  h += "<h3>Per question (top 10)</h3>" + table(d.questions.map(q => [q.id, q.type, esc(q.question), q.answerable ? q.retrieved.map(x => x.relevant ? "●" : "○").join("") : '<span class="muted">unanswerable</span>', q.answer ? esc(q.answer.text || q.answer.error) : ""]), ["id", "type", "question", "top-10 (● relevant)", "answer"]);
  $("eval-out").innerHTML = h;
};
async function vocab() {
  const r = await call("/api/words/vocabulary?" + qs({...cfg(), contains: $("vf").value}));
  if (!r.ok) { banner(r.error); return; }
  const d = r.data;
  $("vinfo").textContent = `${d.distinct_words} distinct words, ${d.total_words} in total (showing ${d.shown.length})`;
  $("vocab-out").innerHTML = table(d.shown.map(x => [`<a data-word="${esc(x.word)}">${esc(x.word)}</a>`, x.count, x.chunks]), ["word", "count", "in how many passages"]);
}
async function wordChunks(w) {
  if (!w) return; $("wq").value = w;
  const r = await call("/api/words/word?" + qs({...cfg(), word: w}));
  if (!r.ok) { banner(r.error); return; }
  const d = r.data;
  $("word-out").innerHTML = d.chunks.length ? `<p><b>${esc(d.word)}</b>: ${d.total} times in ${d.chunks.length} passages</p>` + d.chunks.map(c => `<div class="card"><b>${esc(c.source)}</b> #${c.chunk_index} · ${c.count}×<pre class="passage">${esc(c.text)}</pre></div>`).join("") : `<p class="bad">“${esc(d.word)}” does not exist in the documents.</p>`;
}
async function qWords() {
  const r = await call("/api/words/question", {...cfg(), question: $("wtext").value});
  if (!r.ok) { banner(r.error); return; }
  const rows = r.data.words.filter(x => !($("hs").checked && x.stopword)).map(x => [
    x.exists ? `<a data-word="${esc(x.word)}">${esc(x.word)}</a>` : `<span class="bad">${esc(x.word)}</span>`,
    x.exists ? '<span class="ok">exists</span>' : '<span class="bad">ABSENT</span>', x.total, x.n_chunks,
    x.chunks.slice(0, 6).map(c => esc(c.id) + " (" + c.count + ")").join(", ") + (x.chunks.length > 6 ? " …" : "")]);
  $("qwords-out").innerHTML = table(rows, ["word", "in the documents?", "times", "passages", "which passages"]);
}
$("vf").oninput = vocab; $("wfind").onclick = () => wordChunks($("wq").value);
$("qwords-btn").onclick = qWords; $("hs").onchange = qWords;
$("wpick").onchange = () => { const q = QUESTIONS.find(x => x.id === $("wpick").value); if (q) { $("wtext").value = q.question; qWords(); } };

/* ---------------------------------------------------------------- clicks on dynamic elements */
document.addEventListener("click", async e => {
  const t = e.target.closest("[data-trace],[data-chunk],[data-node],[data-tab],[data-close],[data-index],[data-rm],[data-delcorpus],[data-askabout],[data-word]");
  if (!t) { if (e.target.id === "overlay") closePanel(); return; }
  const d = t.dataset;
  if (d.trace) openTrace(d.trace);
  else if (d.chunk) openChunk(d.chunk);
  else if (d.node && !t.closest("#mapsvg")) selectNode(d.node);
  else if (d.tab) show(d.tab);
  else if (d.close !== undefined) closePanel();
  else if (d.index) runIndex(d.index);
  else if (d.word) { show("words"); wordChunks(d.word); }
  else if (d.askabout) { $("q").value = `What do the documents say about ${d.askabout}?`; show("ask"); ask(); }
  else if (d.rm) { const [c, f] = d.rm.split("|"); const r = await call(`/api/corpora/${encodeURIComponent(c)}/documents/${encodeURIComponent(f)}`, undefined, "DELETE"); if (!r.ok) banner(r.error); await loadDocs(); await loadCorpora(c); }
  else if (d.delcorpus) { if (confirm(`Delete the collection “${d.delcorpus}” and its documents?`)) { const r = await call(`/api/corpora/${encodeURIComponent(d.delcorpus)}`, undefined, "DELETE"); if (!r.ok) banner(r.error); await loadCorpora(); await loadDocs(); } }
});
document.addEventListener("change", e => { if (e.target.dataset && e.target.dataset.file) uploadFiles(e.target.dataset.file, [...e.target.files]); });
document.addEventListener("dragover", e => { const z = e.target.closest("[data-drop]"); if (z) { e.preventDefault(); z.classList.add("over"); } });
document.addEventListener("dragleave", e => { const z = e.target.closest("[data-drop]"); if (z) z.classList.remove("over"); });
document.addEventListener("drop", e => { const z = e.target.closest("[data-drop]"); if (z) { e.preventDefault(); z.classList.remove("over"); uploadFiles(z.dataset.drop, [...e.dataTransfer.files]); } });
document.addEventListener("keydown", e => { if (e.key === "Escape") closePanel(); });

/* ---------------------------------------------------------------- start */
(async function start() {
  const c = await call("/api/config");
  if (c.ok) $("foot").textContent = `Model: ${c.data.model} · server: ${c.data.server.replace(/^https?:\/\//, "").split("/")[0]} · personal data hidden before it is sent: ${c.data.personal_data_hidden ? "on" : "OFF"} · traces stay on this computer for ${c.data.trace_retention_days} days`;
  await loadCorpora();
})();
