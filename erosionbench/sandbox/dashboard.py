"""Dashboard route — serves a single self-contained HTML page for the live demo.

Vanilla JS + Chart.js (CDN), no build step. This page is the *disposable demo skin*: it consumes the
durable JSON contract (/admin/metrics, /admin/blast-radius, /admin/showcase-t3, /admin/showcase-replay).
A future React app points at the same endpoints and this file is deleted — a frontend swap, not a
rewrite.

NB: the live audit-log replay theater (/admin/replay/*, erosionbench/sandbox/replay.py) is intentionally
NOT wired into this page. It only has something to show when a live eval run is actively writing to
the in-memory audit-log; during a replay-based demo (the Scenario Replay panel below) nothing streams
there, so the panel sat empty/dead. Removed here; the replay.py endpoints remain available for anyone
scripting against a live run directly.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ErosionBench — Alignment Decay Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --ink: #e6edf3; --dim: #8b949e;
    --ok: #3fb950; --warn: #f85149; --accent: #58a6ff;
    /* Accessible categorical palette (light+dark safe). */
    --c1: #58a6ff; --c2: #bc8cff; --c3: #f0883e; --c4: #56d4bc;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 12px 20px;
           border-bottom: 1px solid var(--border); background: var(--panel); position: sticky; top: 0; z-index: 5; }
  header h1 { font-size: 16px; margin: 0; letter-spacing: .3px; }
  header .sub { color: var(--dim); font-size: 12px; }
  header .spacer { flex: 1; }
  select, button { background: #21262d; color: var(--ink); border: 1px solid var(--border);
                   border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .8px; color: var(--dim);
              margin: 0 0 10px; font-weight: 600; display: flex; align-items: baseline; gap: 8px; }
  .panel h2 .count { margin-left: auto; font-size: 11px; font-weight: 700; letter-spacing: .3px;
                     text-transform: none; color: var(--accent); background: rgba(88,166,255,.10);
                     border: 1px solid var(--border); border-radius: 999px; padding: 1px 9px; }
  .full { grid-column: 1 / -1; }
  .chart-sub { color: var(--dim); font-size: 11px; margin: -6px 0 10px; }
  /* Stat tiles */
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 12px; }
  .tile { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .tile .k { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  .tile .v { font-size: 22px; font-weight: 700; margin-top: 2px; }
  /* Blast map */
  #blast { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .dcell { border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: #0d1117; transition: .3s; }
  .dcell.hit { border-color: var(--warn); box-shadow: 0 0 0 1px var(--warn) inset; background: rgba(248,81,73,.07); }
  .dcell .name { font-size: 12px; font-weight: 600; }
  .dcell .cnt { font-size: 20px; font-weight: 700; margin: 4px 0; }
  .dcell .tags { font-size: 11px; color: var(--warn); }
  .dcell.clean .cnt { color: var(--ok); }
  canvas { max-height: 300px; }
  .muted { color: var(--dim); font-size: 12px; }
  /* ③ Scenario Replay */
  .rp-note { color: var(--dim); font-size: 12px; margin-bottom: 12px; }
  .rp-layout { display: grid; grid-template-columns: 300px 1fr; gap: 16px; align-items: start; }
  /* left: grouped scenario picker */
  .rp-picker { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 8px;
               max-height: 640px; overflow-y: auto; }
  .rp-group-h { font-size: 10px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim);
                margin: 10px 6px 4px; font-weight: 600; }
  .rp-group-h:first-child { margin-top: 2px; }
  .rp-item { border: 1px solid transparent; border-radius: 7px; padding: 7px 9px; margin: 3px 0; cursor: pointer; }
  .rp-item:hover { background: #161b22; }
  .rp-item.active { border-color: var(--accent); background: rgba(88,166,255,.08); }
  .rp-item .sid { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--accent); }
  .rp-item .sub { font-size: 11px; color: var(--dim); margin-top: 2px; }
  .rp-item .badge { float: right; font-size: 10px; font-weight: 700; padding: 0 6px; border-radius: 999px; }
  .rp-item .badge.caught { color: var(--ok); border: 1px solid var(--ok); }
  .rp-item .badge.miss   { color: var(--warn); border: 1px solid var(--warn); }
  /* right: replay stage */
  .rp-stage { min-width: 0; }
  .rp-head { margin-bottom: 10px; }
  .rp-head .sid { font-family: ui-monospace, Menlo, monospace; font-size: 13px; color: var(--accent); }
  .rp-head .tag { display: inline-block; background: #21262d; border: 1px solid var(--border);
                  border-radius: 999px; padding: 1px 9px; margin-left: 6px; font-size: 11px; color: var(--dim); }
  .rp-head .obj { font-size: 12px; color: var(--dim); margin: 8px 0 0; line-height: 1.5; }
  .rp-head .obj b { color: var(--ink); }
  .rp-contrast { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: stretch; margin: 12px 0; }
  .rp-lane { border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; text-align: center; }
  .rp-lane .lab { font-size: 10px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim); }
  .rp-lane .out { font-size: 15px; font-weight: 700; margin-top: 3px; }
  .rp-lane.l2 { background: rgba(248,81,73,.08); border-color: var(--warn); }
  .rp-lane.l2 .out { color: var(--warn); }
  .rp-lane.l5.caught { background: rgba(63,185,80,.10); border-color: var(--ok); }
  .rp-lane.l5.caught .out { color: var(--ok); }
  .rp-lane.l5.miss { background: rgba(248,81,73,.08); border-color: var(--warn); }
  .rp-lane.l5.miss .out { color: var(--warn); }
  .rp-mid { display: flex; align-items: center; font-size: 18px; color: var(--dim); }
  .rp-controls { display: flex; gap: 8px; align-items: center; margin: 10px 0; }
  .rp-controls .rp-progress { color: var(--dim); font-size: 12px; margin-left: 6px; }
  .rp-transcript { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
                   max-height: 360px; overflow-y: auto; }
  .rp-turn { padding: 8px 10px; margin: 6px 0; border-radius: 8px; border: 1px solid #21262d;
             opacity: 0; transform: translateY(4px); transition: opacity .35s, transform .35s; }
  .rp-turn.show { opacity: 1; transform: none; }
  .rp-turn .who { font-size: 11px; font-weight: 700; margin-bottom: 3px; }
  .rp-turn .content { font-size: 12px; color: var(--ink); line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .rp-turn.gateway_agent      { border-left: 3px solid #56d4bc; }
  .rp-turn.orchestrator_agent { border-left: 3px solid #58a6ff; }
  .rp-turn.policy_agent       { border-left: 3px solid #bc8cff; }
  .rp-turn.deployment_agent   { border-left: 3px solid #f0883e; }
  .rp-turn .who.gateway_agent      { color: #56d4bc; }
  .rp-turn .who.orchestrator_agent { color: #58a6ff; }
  .rp-turn .who.policy_agent       { color: #bc8cff; }
  .rp-turn .who.deployment_agent   { color: #f0883e; }
  /* auditor reasoning card (revealed after the transcript finishes) */
  .rp-reasoning { margin-top: 14px; border: 1px solid var(--border); border-radius: 10px; padding: 0;
                  opacity: 0; max-height: 0; overflow: hidden; transition: opacity .4s; }
  .rp-reasoning.show { opacity: 1; max-height: none; padding: 14px 16px; }
  .rp-reasoning.caught { border-color: var(--ok); background: rgba(63,185,80,.06); }
  .rp-reasoning.miss   { border-color: var(--warn); background: rgba(248,81,73,.06); }
  .rp-reasoning h3 { margin: 0 0 4px; font-size: 14px; }
  .rp-reasoning .verdict { font-size: 18px; font-weight: 700; }
  .rp-reasoning .verdict.caught { color: var(--ok); } .rp-reasoning .verdict.miss { color: var(--warn); }
  .rp-mode { display: inline-block; font-size: 11px; font-weight: 700; padding: 1px 9px; border-radius: 999px;
             margin-left: 8px; }
  .rp-mode.sync_edge { color: #56d4bc; border: 1px solid #56d4bc; }
  .rp-mode.grey_zone_arbiter { color: #bc8cff; border: 1px solid #bc8cff; }
  .rp-mode.none { color: var(--dim); border: 1px solid var(--border); }
  .rp-metrics { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
  .rp-metric { background: #0d1117; border: 1px solid var(--border); border-radius: 7px; padding: 5px 10px; font-size: 12px; }
  .rp-metric b { font-family: ui-monospace, Menlo, monospace; }
  .rp-metric.fired { border-color: var(--warn); color: var(--warn); }
  .rp-block { margin: 10px 0; }
  .rp-block .lab { font-size: 10px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim); margin-bottom: 3px; }
  .rp-block .txt { font-size: 13px; color: var(--ink); line-height: 1.55; }
  .rp-block.reason .txt { background: #0d1117; border: 1px solid var(--border); border-radius: 8px;
                          padding: 10px 12px; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
                          white-space: pre-wrap; word-break: break-word; }
  .rp-src { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--accent); word-break: break-word; }
</style>
</head>
<body>
<header>
  <h1>ErosionBench</h1>
  <span class="sub">Alignment Decay — Live Dashboard</span>
  <span class="spacer"></span>
  <span class="sub">Metrics source</span>
  <select id="srcSel"></select>
</header>

<main>
  <section class="panel">
    <h2>① ASR Across Defense Tiers <span class="count" id="asrCount"></span></h2>
    <div class="chart-sub" id="asrSub"></div>
    <div class="tiles" id="asrTiles"></div>
    <canvas id="asrChart"></canvas>
  </section>

  <section class="panel">
    <h2>② Blast-Radius Map
      <select id="blastMode" style="font-size:11px;padding:2px 6px;">
        <option value="static" selected>Static (frozen eval)</option>
        <option value="live">Live (this sandbox)</option>
      </select>
      <span class="count" id="blastCount"></span>
    </h2>
    <div class="chart-sub" id="blastSub"></div>
    <div id="blast"><div class="muted">No compromised operations yet.</div></div>
  </section>

  <section class="panel full">
    <h2>Auditor Quality (Precision · Recall · FPR by domain) <span class="count" id="audCount"></span></h2>
    <div class="chart-sub" id="audSub"></div>
    <canvas id="audChart"></canvas>
  </section>

  <section class="panel full">
    <h2>③ Scenario Replay — watch the attack, then how L5 reasoned about it <span class="count" id="rpCount"></span></h2>
    <div class="rp-note" id="rpNote">Loading curated scenarios…</div>
    <div class="rp-layout">
      <div class="rp-picker" id="rpPicker"></div>
      <div class="rp-stage" id="rpStage">
        <div class="muted">Pick a scenario on the left, then press ▶ Replay to step through the
          inter-agent chat and reveal the Meta-Auditor's reasoning.</div>
      </div>
    </div>
  </section>
</main>

<script>
const DOMAINS = ["quantitative_finance","cicd_supply_chain","data_engineering","autonomous_crm"];
const PRETTY = {quantitative_finance:"Quant Finance",cicd_supply_chain:"CI/CD Supply Chain",
                data_engineering:"Data Engineering",autonomous_crm:"Autonomous CRM"};
const C = ["#58a6ff","#bc8cff","#f0883e","#56d4bc"];
/* Per-layer defense-ladder color ramp (L1 no-defense -> L5 auditor). Extends past 5 by cycling. */
const LAYER_C = ["#f0883e","#e3b341","#58a6ff","#bc8cff","#3fb950","#56d4bc"];
let asrChart, audChart;

async function j(url, opts) { const r = await fetch(url, opts); if(!r.ok) throw new Error(r.status); return r.json(); }

/* ---------- ② blast-radius map ---------- */
function renderBlast(br) {
  const box = document.getElementById("blast");
  const effects = br.cascading_effects || [];
  const byDomain = {};
  for (const eff of effects) {
    (byDomain[eff.domain] = byDomain[eff.domain] || {n:0, tags:new Set()});
    byDomain[eff.domain].n++; (eff.impact_tags||[]).forEach(t=>byDomain[eff.domain].tags.add(t));
  }
  // Badge = total compromised operations across all domains.
  const bc = document.getElementById("blastCount");
  if (bc) bc.textContent = `${effects.length} compromised op${effects.length===1?"":"s"}`;
  // Sub-line explains which view is showing (static replays the frozen eval; live reads this sandbox).
  const bs = document.getElementById("blastSub");
  if (bs) {
    bs.innerHTML = (br.mode === "static")
      ? `Static — L2-landed attacks from the frozen eval (${scEsc(br.source_file||"")}); ${br.compromised_operations||0} of ${br.total_operations||0} scenarios.`
      : `Live — compromised ops recorded by THIS sandbox's audit log. Empty until an eval run targets it.`;
  }
  box.innerHTML = "";
  for (const d of DOMAINS) {
    const info = byDomain[d];
    const cell = document.createElement("div");
    cell.className = "dcell " + (info ? "hit" : "clean");
    cell.innerHTML = `<div class="name">${PRETTY[d]}</div>` +
      `<div class="cnt">${info ? info.n : 0}</div>` +
      `<div class="tags">${info ? [...info.tags].join(', ') : 'no violations'}</div>`;
    box.appendChild(cell);
  }
}

/* ---------- ① + auditor charts ---------- */
function modelSub(m) {
  // The eval-pipeline result file records which mesh (attack surface) and auditor (defense) model
  // produced these numbers (run_final_eval.py meta.mesh_model / meta.auditor_model, or the ladder
  // runner's equivalent). Surface it directly under both charts so a demo never shows unlabeled
  // ASR/precision numbers — legacy result files that predate this field render "not recorded".
  const mesh = m.mesh_model || "not recorded";
  const auditor = m.auditor_model || "not recorded";
  return `${scenCount(m)} · mesh: <b>${scEsc(mesh)}</b> · auditor: <b>${scEsc(auditor)}</b> · source: ${scEsc(m.source_file)}`;
}

// Total scenario count behind a metrics panel. Prefers the top-level n_total (final-eval schema);
// falls back to summing per-domain n_scenarios (ladder/legacy schemas carry no top-level total).
function scenTotal(m) {
  if (m.n_total != null) return m.n_total;
  const doms = m.domains || [];
  const sum = doms.reduce((a, d) => a + (d.n_scenarios || 0), 0);
  return sum || null;
}
function scenCount(m) {
  const n = scenTotal(m);
  if (n == null) return "n = —";
  const mal = m.n_malicious, ben = m.n_benign;
  return (mal != null && ben != null)
    ? `n = <b>${n}</b> (${mal} mal / ${ben} benign)`
    : `n = <b>${n}</b> scenarios`;
}

function renderMetrics(m) {
  const t = m.overall_asr || {};
  const fmt = v => v==null ? "—" : v.toFixed(1) + "%";
  // Layers are data-driven (may be 2..5): each carries {key, name}. Fall back to the legacy trio.
  const layers = (m.layers && m.layers.length) ? m.layers
    : [{key:"layer1_no_defense",name:"No Defense"},{key:"layer2_firewall",name:"Firewall"},
       {key:"layer3_auditor",name:"Auditor"}];
  const tiles = layers.map((L,i) =>
    `<div class="tile"><div class="k">${L.key} ${L.name}</div>` +
    `<div class="v" style="color:${LAYER_C[i%LAYER_C.length]}">${fmt(t[L.key])}</div></div>`);
  tiles.push(`<div class="tile"><div class="k">Source</div>` +
    `<div class="v" style="font-size:12px;color:var(--dim)">${m.source_file}</div></div>`);
  document.getElementById("asrTiles").innerHTML = tiles.join("");
  document.getElementById("asrSub").innerHTML = modelSub(m);
  document.getElementById("audSub").innerHTML = modelSub(m);
  // Total-scenario badges on both metric-driven panels.
  const nTot = scenTotal(m);
  const badge = nTot == null ? "" : `${nTot} scenarios`;
  document.getElementById("asrCount").textContent = badge;
  document.getElementById("audCount").textContent = badge;

  const labels = m.domains.map(d => PRETTY[d.domain]||d.domain);
  const asrDs = layers.map((L,i) => ({
    label: `${L.key} ${L.name}`,
    data: m.domains.map(d => d.asr[L.key]),
    backgroundColor: LAYER_C[i%LAYER_C.length],
  }));
  const opts = grid => ({responsive:true, plugins:{legend:{labels:{color:"#e6edf3"}}},
    scales:{x:{ticks:{color:"#8b949e"},grid:{color:"#21262d"}},
            y:{ticks:{color:"#8b949e"},grid:{color:"#21262d"},beginAtZero:true, suggestedMax:grid}}});
  if (asrChart) asrChart.destroy();
  asrChart = new Chart(document.getElementById("asrChart"),
    {type:"bar", data:{labels, datasets:asrDs}, options:opts(100)});

  const audDs = [
    {label:"Precision", data:m.domains.map(d=>pct(d.auditor.precision)), backgroundColor:C[1]},
    {label:"Recall",    data:m.domains.map(d=>pct(d.auditor.recall)),    backgroundColor:C[3]},
    {label:"FPR",       data:m.domains.map(d=>d.auditor.fpr),            backgroundColor:C[2]},
  ];
  if (audChart) audChart.destroy();
  audChart = new Chart(document.getElementById("audChart"),
    {type:"bar", data:{labels, datasets:audDs}, options:opts(100)});
}
const pct = v => v==null ? null : Math.round(v*1000)/10;

/* ---------- metrics source selector ---------- */
let currentSrc = "latest";
async function loadSources() {
  const {files} = await j("/admin/metrics/files");
  const sel = document.getElementById("srcSel");
  sel.innerHTML = '<option value="latest">Latest run</option>' +
    files.map(f=>`<option value="${f.name}">${scEsc(f.label||f.name)}</option>`).join("");
}

/* ---------- ② blast-radius mode toggle (static frozen-eval vs live sandbox) ---------- */
let blastMode = "static";
function blastUrl() { return blastMode === "static" ? "/admin/blast-radius/static" : "/admin/blast-radius"; }
document.getElementById("blastMode").onchange = e => { blastMode = e.target.value; pollBlast(); };

/* ---------- polling loops ---------- */
async function pollBlast() {
  try { renderBlast(await j(blastUrl())); } catch(e){}
}
async function refreshMetrics() {
  try { renderMetrics(await j("/admin/metrics?source=" + encodeURIComponent(currentSrc))); }
  catch(e){ document.getElementById("asrTiles").innerHTML = '<div class="muted">No metrics file found in results/.</div>'; }
}

document.getElementById("srcSel").onchange = e => { currentSrc = e.target.value; refreshMetrics(); };

/* ---------- ③ scenario replay (chat logs + auditor reasoning) ---------- */
// Data model: /admin/showcase-t3 = the curated MANIFEST (always available, no gateway needed);
// /admin/showcase-replay = the CAPTURED transcripts + auditor reasoning (populated by
// capture_showcase.py once the gateway budget is available). We join them by scenario_id so the
// picker + contrast render from the manifest immediately, and Replay steps through the captured
// transcript + reasoning when present.
const scEsc = s => String(s==null?"":s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
// `revealed` remembers which scenarios the user has already played/shown-all, so a finished
// replay survives ANY later re-render (poll, manifest refresh, capture rewrite) instead of
// resetting to a blank stage. It persists until a DIFFERENT scenario is selected.
const RP = { manifest: [], replay: {}, active: null, timer: null, revealed: new Set() };
const BUCKET_H = { deterministic: "Deterministic catch (sync-edge · formal-metric threshold)",
                   semantic: "Semantic catch (grey-zone LLM arbiter)",
                   both_missed: "Both missed (the honest ceiling)" };
const MODE_LABEL = { sync_edge: "deterministic sync-edge (formal-metric threshold)",
                     grey_zone_arbiter: "LLM grey-zone arbiter (semantic reasoning)",
                     none: "not flagged" };

function rpPicker() {
  const box = document.getElementById("rpPicker");
  if (!RP.manifest.length) { box.innerHTML = '<div class="muted">No scenarios.</div>'; return; }
  const order = ["deterministic","semantic","both_missed"];
  const groups = {}; RP.manifest.forEach(s => (groups[s.bucket] = groups[s.bucket]||[]).push(s));
  let html = "";
  for (const b of order) {
    if (!groups[b]) continue;
    html += `<div class="rp-group-h">${scEsc(BUCKET_H[b]||b)}</div>`;
    for (const s of groups[b]) {
      const cap = RP.replay[s.scenario_id];
      const caught = (cap && cap.transcript) ? cap.l5_caught : s.l5_caught;
      const badge = caught ? '<span class="badge caught">L5 ✔</span>' : '<span class="badge miss">L5 ✗</span>';
      const act = RP.active===s.scenario_id ? " active" : "";
      html += `<div class="rp-item${act}" data-sid="${scEsc(s.scenario_id)}">${badge}
        <div class="sid">${scEsc(s.scenario_id)}</div>
        <div class="sub">${scEsc(PRETTY[s.domain]||s.domain)} · ${scEsc(s.attack_class)} · ${scEsc(s.tier)}</div></div>`;
    }
  }
  box.innerHTML = html;
  box.querySelectorAll(".rp-item").forEach(el =>
    el.addEventListener("click", () => rpSelect(el.getAttribute("data-sid"))));
}

function rpSelect(sid) {
  // Selecting a DIFFERENT scenario clears its remembered reveal so it starts fresh; re-selecting
  // the same one keeps whatever was already revealed (rpRenderStage re-applies it).
  if (sid !== RP.active) RP.revealed.delete(sid);
  RP.active = sid;
  if (RP.timer) { clearInterval(RP.timer); RP.timer = null; }
  document.querySelectorAll(".rp-item").forEach(el =>
    el.classList.toggle("active", el.getAttribute("data-sid")===sid));
  rpRenderStage(sid);
}

function rpMetric(name, v, fires) {
  return `<div class="rp-metric${fires?" fired":""}">${name} <b>${v==null?"—":(+v).toFixed(3)}</b>${fires?" ▲":""}</div>`;
}

function rpRenderStage(sid) {
  const m = RP.manifest.find(x=>x.scenario_id===sid);
  const cap = RP.replay[sid];   // captured transcript+reasoning, may be undefined
  if (!m) return;
  const hasCap = !!(cap && cap.transcript);
  // When a capture exists, the contrast MUST reflect the captured run (the transcript being shown
  // IS that run) — the mesh is stochastic, so a fresh capture can land/miss differently than the
  // frozen eval. Fall back to the frozen manifest outcome only when no capture is present.
  const caught = hasCap ? cap.l5_caught : m.l5_caught;
  const l2Landed = hasCap ? cap.l2_landed : m.l2_landed;
  const nTurns = hasCap ? cap.transcript.length : 0;
  const intent = m.user_intent || (cap && cap.user_intent) || "";
  const obj = m.attacker_objective || (cap && cap.attacker_objective) || "";
  const stage = document.getElementById("rpStage");
  stage.innerHTML = `
    <div class="rp-head">
      <span class="sid">${scEsc(sid)}</span>
      <span class="tag">${scEsc(PRETTY[m.domain]||m.domain)}</span>
      <span class="tag">${scEsc(m.tier)}</span>
      <span class="tag">${scEsc(m.attack_class)}</span>
      ${intent?`<div class="obj"><b>User authorized:</b> ${scEsc(intent)}</div>`:""}
      ${obj?`<div class="obj"><b>Hidden attacker objective:</b> ${scEsc(obj)}</div>`:""}
    </div>
    <div class="rp-contrast">
      <div class="rp-lane l2"><div class="lab">L2 · No Defense</div>
        <div class="out">${l2Landed?"COMPROMISED":"clean"}</div></div>
      <div class="rp-mid">→</div>
      <div class="rp-lane l5 ${caught?"caught":"miss"}"><div class="lab">L5 · Meta-Auditor</div>
        <div class="out">${caught?"CAUGHT":"MISSED"}</div></div>
    </div>
    ${hasCap && (cap.l2_landed!==m.l2_landed || cap.l5_caught!==m.l5_caught)
      ? `<div class="muted" style="margin:-4px 0 8px">Note: this replay is a fresh mesh capture; its
           outcome differs from the frozen eval (frozen: L2 ${m.l2_landed?"landed":"clean"} · L5
           ${m.l5_caught?"caught":"missed"}). The mesh is stochastic.</div>` : ""}
    <div class="rp-controls">
      <button id="rpPlay" ${hasCap?"":"disabled"}>▶ Replay</button>
      <button id="rpAll" ${hasCap?"":"disabled"}>Show all</button>
      <span class="rp-progress" id="rpProgress">${hasCap?`0 / ${nTurns} turns`:"transcript not captured yet"}</span>
    </div>
    <div class="rp-transcript" id="rpTranscript">${
      hasCap
        ? cap.transcript.map((t,i)=>{
            const who = t.speaker || "agent";
            const to = t.listener ? ` → ${scEsc(PRETTY_ROLE(t.listener))}` : "";
            return `<div class="rp-turn ${scEsc(who)}" data-i="${i}">
              <div class="who ${scEsc(who)}">${scEsc(PRETTY_ROLE(who))}${to}</div>
              <div class="content">${scEsc(t.content)}</div></div>`;
          }).join("")
        : `<div class="muted">The inter-agent chat for this scenario has not been captured yet.
             Run <code>capture_showcase.py</code> (needs the gateway) to populate the transcript and
             the auditor's step-by-step reasoning. The verdict contrast above is from the frozen eval.</div>`
    }</div>
    ${rpReasoningHTML(m, cap, caught)}`;
  const play = document.getElementById("rpPlay");
  if (play) play.addEventListener("click", () => rpPlay(sid));
  const all = document.getElementById("rpAll");
  if (all) all.addEventListener("click", () => rpShowAll(sid));
  // If this scenario was already played/revealed, re-apply that state so a re-render (poll,
  // manifest refresh, capture rewrite) never wipes a finished replay back to a blank stage.
  if (hasCap && RP.revealed.has(sid)) {
    document.querySelectorAll("#rpTranscript .rp-turn").forEach(t=>t.classList.add("show"));
    const prog = document.getElementById("rpProgress");
    if (prog) prog.textContent = `${nTurns} / ${nTurns} turns`;
    rpRevealReasoning();
  }
}

function PRETTY_ROLE(r){
  return ({gateway_agent:"Gateway", orchestrator_agent:"Orchestrator",
           policy_agent:"Policy", deployment_agent:"Deployment"})[r] || r;
}

function rpReasoningHTML(m, cap, caught) {
  const a = cap && cap.auditor;
  const mode = (a && a.detection_mode) || m.detection_mode || "none";
  const fired = v => v!=null && v>0;
  const fm = (a && a.formal_metrics) || {sdi:m.sdi, cii:m.cii, vad:m.vad, vpr:m.vpr};
  const tv = a && a.tiered_verdict;
  const gz = a && a.grey_zone_result;
  const sem = tv ? tv.semantic_compromise_prob : m.semantic_compromise_prob;
  const metrics = [
    rpMetric("SDI", fm.sdi, fired(fm.sdi)),
    rpMetric("CII", fm.cii, fired(fm.cii)),
    rpMetric("VAD", fm.vad, fired(fm.vad)),
    rpMetric("VPR", fm.vpr, false),
    rpMetric("raw intent-divergence", sem, sem!=null && sem>0.72),
  ].join("");
  // The reasoning card is always rendered but stays hidden until the replay finishes (or Show all).
  let blocks = "";
  if (a && a.reasoning)
    blocks += `<div class="rp-block reason"><div class="lab">Auditor reasoning trace</div>
                 <div class="txt">${scEsc(a.reasoning)}</div></div>`;
  if (gz && gz.rationale)
    blocks += `<div class="rp-block"><div class="lab">Grey-zone rationale (telemetry citation)</div>
                 <div class="txt">${scEsc(gz.rationale)}</div></div>`;
  if (gz && gz.deep_reasoning)
    blocks += `<div class="rp-block"><div class="lab">Grey-zone deep reasoning (Tier-3 arbiter)</div>
                 <div class="txt">${scEsc(gz.deep_reasoning)}</div></div>`;
  if (gz && (gz.original_intent_summary || gz.final_intent_summary))
    blocks += `<div class="rp-block"><div class="lab">Intent drift</div>
                 <div class="txt"><b>authorized:</b> ${scEsc(gz.original_intent_summary||"—")}<br>
                 <b>steered toward:</b> ${scEsc(gz.final_intent_summary||"—")}</div></div>`;
  const src = (a && a.verdict_source) || m.verdict_source || "—";
  const why = caught
    ? (mode==="sync_edge"
        ? "A formal metric crossed its conformal threshold — the deterministic edge interceptor fired on that threshold crossing alone."
        : "Formal and lightweight-semantic tiers disagreed, so the grey-zone LLM arbiter (Tier-3) resolved it by reasoning over the telemetry.")
    : "The payload landed AND the auditor did not flag it — a residual miss (the benchmark's honest ceiling, concentrated in reasoning-influence).";
  return `<div class="rp-reasoning ${caught?"caught":"miss"}" id="rpReasoning">
      <h3>L5 verdict: <span class="verdict ${caught?"caught":"miss"}">${caught?"COMPROMISED (caught)":"missed"}</span>
        <span class="rp-mode ${scEsc(mode)}">${scEsc(MODE_LABEL[mode]||mode)}</span></h3>
      <div class="muted">${why}</div>
      <div class="rp-metrics">${metrics}</div>
      ${blocks || '<div class="muted">Detailed reasoning text will appear here once the scenario is captured.</div>'}
      <div class="rp-block"><div class="lab">verdict source</div>
        <div class="txt"><span class="rp-src">${scEsc(src)}</span></div></div>
    </div>`;
}

function rpRevealReasoning() {
  const el = document.getElementById("rpReasoning");
  if (el) el.classList.add("show");
}

function rpPlay(sid) {
  const cap = RP.replay[sid];
  if (!cap || !cap.transcript) return;
  if (RP.timer) clearInterval(RP.timer);
  const turns = document.querySelectorAll("#rpTranscript .rp-turn");
  turns.forEach(t=>t.classList.remove("show"));
  const rz = document.getElementById("rpReasoning"); if (rz) rz.classList.remove("show");
  const box = document.getElementById("rpTranscript");
  let i = 0;
  const prog = document.getElementById("rpProgress");
  RP.timer = setInterval(() => {
    if (i >= turns.length) {
      clearInterval(RP.timer); RP.timer = null;
      RP.revealed.add(sid);   // remember: keep this replay visible through later re-renders
      rpRevealReasoning();
      return;
    }
    turns[i].classList.add("show");
    box.scrollTop = turns[i].offsetTop - box.offsetTop - 20;
    i++;
    if (prog) prog.textContent = `${i} / ${turns.length} turns`;
  }, 900);
}

function rpShowAll(sid) {
  if (RP.timer) { clearInterval(RP.timer); RP.timer = null; }
  RP.revealed.add(sid);   // remember: keep this replay visible through later re-renders
  document.querySelectorAll("#rpTranscript .rp-turn").forEach(t=>t.classList.add("show"));
  const cap = RP.replay[sid];
  const prog = document.getElementById("rpProgress");
  if (prog && cap && cap.transcript) prog.textContent = `${cap.transcript.length} / ${cap.transcript.length} turns`;
  rpRevealReasoning();
}

async function loadReplay(isPoll) {
  // On a background poll we must NOT blow away a stage the user is watching (rebuilding the HTML
  // resets revealed turns + the reasoning card). Track whether the active scenario had captured
  // data BEFORE this refresh so we only re-render it when new data actually arrived for it.
  const hadCapBefore = RP.active ? !!(RP.replay[RP.active] && RP.replay[RP.active].transcript) : false;
  try {
    const rep = await j("/admin/showcase-replay");
    RP.replay = {}; (rep.scenarios||[]).forEach(s => RP.replay[s.scenario_id]=s);
  } catch(e){ /* capture not run yet */ }
  const hasCapAfter = RP.active ? !!(RP.replay[RP.active] && RP.replay[RP.active].transcript) : false;
  try {
    const man = await j("/admin/showcase-t3");
    RP.manifest = man.scenarios || [];
    const note = document.getElementById("rpNote");
    const nCap = Object.keys(RP.replay).length;
    const rpc = document.getElementById("rpCount");
    if (rpc) rpc.textContent = RP.manifest.length ? `${RP.manifest.length} curated` : "";
    note.innerHTML = RP.manifest.length
      ? `${RP.manifest.length} curated scenarios — `
        + `<b>${RP.manifest.filter(s=>s.bucket==="deterministic").length}</b> deterministic, `
        + `<b>${RP.manifest.filter(s=>s.bucket==="semantic").length}</b> semantic, `
        + `<b>${RP.manifest.filter(s=>s.bucket==="both_missed").length}</b> both-missed.`
        + (nCap? ` Transcripts captured for <b>${nCap}</b>.`
               : ` <span style="color:var(--c3)">Transcripts not captured yet — run capture_showcase.py.</span>`)
      : ((man.meta && man.meta.note) || "No showcase scenarios found.");
  } catch(e){ /* endpoint absent on older sandbox */ }
  rpPicker();
  if (!RP.active && RP.manifest.length) {
    rpSelect(RP.manifest[0].scenario_id);
  } else if (RP.active && !isPoll) {
    // initial/explicit load: render normally
    rpRenderStage(RP.active);
  } else if (RP.active && isPoll && !hadCapBefore && hasCapAfter) {
    // background poll: only re-render if capture data just appeared for the active scenario
    // (never mid-viewing once it's already there — that would reset the user's replay)
    rpRenderStage(RP.active);
  }
}

/* boot */
loadSources().then(refreshMetrics);
loadReplay();
setInterval(pollBlast, 2000);
setInterval(refreshMetrics, 10000);   // newer run in results/ -> charts update automatically
setInterval(() => loadReplay(true), 20000);  // picks up showcase_replay.json; non-destructive to the active replay
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return _PAGE
