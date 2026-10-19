"""Metrics API — reads eval-pipeline result files and normalizes them into a stable JSON shape.

This is the durable contract the dashboard (and any future React frontend) consumes. It does NOT
compute anything the eval pipeline doesn't already produce; it only loads, selects, and reshapes the
``results/*.json`` files written after a run. Sourcing strategy (pinned file → latest file →
live-computed) is entirely contained here, so swapping it later never touches the frontend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["metrics"])

# Repo-root/results — erosionbench/sandbox/metrics_api.py -> parents[2] == repo root.
# NB: guard on the raw env string; Path("") == Path(".") is truthy and would swallow the fallback.
_env_results = os.environ.get("RESULTS_DIR", "").strip()
_RESULTS_DIR = Path(_env_results) if _env_results else (Path(__file__).resolve().parents[2] / "results")

# The FROZEN held-out eval (run_final_eval.py) is the canonical demo number: opus-4-8 auditor,
# gpt-5.5 mesh, held-out scenarios (L2 no-defense vs L5 meta-auditor). Defaults to eval_run_opus_v2
# — the 604-scenario apples-to-apples re-eval on the same frozen calibration (recall 0.989, FPR 0.020),
# which supersedes the original 603-scored eval_run (one scenario errored + was excluded). It lives in
# a subdir the top-level scanner never sees, so it is pinned explicitly and shown FIRST — the dashboard
# defaults to it instead of stale top-level legacy files. Override with FINAL_EVAL_FILE.
_env_final = os.environ.get("FINAL_EVAL_FILE", "").strip()
_FINAL_EVAL_PATH = Path(_env_final) if _env_final else (_RESULTS_DIR / "eval_run" / "eval_results.json")

# Domain keys carried in a *legacy* layered metrics file (everything else, e.g. ``_usage``, is metadata).
_DOMAIN_MARKER_KEYS = {"layer1", "layer2", "meta_auditor"}

# The new ladder schema (run_full_eval, results/full_eval_ladder/full_eval_layered.json) is a single
# run object: {"meta": {...}, "asr": {"L1".."L5": {...}}, "auditor": {overall, per_domain, ...}}.
# Layer identities mirror run_full_eval.py's LAYER_SPECS display names. A run may include any subset
# (meta.layers drives which are present), so the normalized output is a *list* of whatever layers ran
# — the front-end renders N of them, not a fixed 3. NB: L3 is the firewall and L5 is the auditor;
# never conflate "layer3" with "auditor".
_LADDER_LAYER_NAMES = {
    "L1": "Single Agent",
    "L2": "Multi-Agent No Defense",
    "L3": "LlamaFirewall",
    "L4": "Progent",
    "L5": "Meta-Auditor",
}

# Legacy per-domain files only ever carried three tiers; expose them through the same layer-list shape.
_LEGACY_LAYERS = [
    ("layer1_no_defense", "No Defense"),
    ("layer2_firewall", "Firewall"),
    ("layer3_auditor", "Auditor"),
]


def _is_legacy_metrics_file(data: Any) -> bool:
    """A legacy layered metrics file has at least one domain dict carrying layer1/layer2/meta_auditor."""
    if not isinstance(data, dict):
        return False
    for key, val in data.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and _DOMAIN_MARKER_KEYS & set(val.keys()):
            return True
    return False


def _is_ladder_metrics_file(data: Any) -> bool:
    """A ladder run object carries an ``asr`` layer-map and an ``auditor`` block with per-domain stats."""
    if not isinstance(data, dict):
        return False
    asr, aud = data.get("asr"), data.get("auditor")
    if not isinstance(asr, dict) or not isinstance(aud, dict):
        return False
    # ``asr`` is keyed by layer (L1..L5), each with an ``overall`` roll-up.
    return any(isinstance(v, dict) and "overall" in v for v in asr.values()) and "per_domain" in aud


def _is_final_eval_file(data: Any) -> bool:
    """The frozen held-out eval (run_final_eval.py): an ``overall`` block carrying L2/L5 ASR plus a
    per-domain breakdown. Distinct from the ladder schema (no ``asr`` layer-map / ``auditor`` block)."""
    if not isinstance(data, dict):
        return False
    ov, by_dom = data.get("overall"), data.get("by_domain")
    if not isinstance(ov, dict) or not isinstance(by_dom, dict):
        return False
    return "l2_asr" in ov and "l5_asr" in ov


def _is_metrics_file(data: Any) -> bool:
    """True for the legacy per-domain, the ladder run, or the frozen final-eval schema."""
    return _is_legacy_metrics_file(data) or _is_ladder_metrics_file(data) or _is_final_eval_file(data)


def _list_metrics_files() -> List[Path]:
    """Metrics result JSONs the dashboard can source, canonical first.

    The frozen held-out final eval is pinned to the front (it lives in a subdir the top-level glob
    never reaches and IS the demo number). Remaining top-level ``*.json`` that look like metrics
    follow, newest (by mtime) first.
    """
    if not _RESULTS_DIR.is_dir():
        return []
    found: List[Path] = []
    if _FINAL_EVAL_PATH.is_file():
        found.append(_FINAL_EVAL_PATH)
    top: List[Path] = []
    for path in _RESULTS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if _is_metrics_file(data):
            top.append(path)
    found.extend(sorted(top, key=lambda p: p.stat().st_mtime, reverse=True))
    return found


def _fpr(fp: int, tn: int) -> Optional[float]:
    denom = fp + tn
    return round(100.0 * fp / denom, 2) if denom else None


def _domain_summary(name: str, dom: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape one domain's block into tier-ASR + auditor-quality the charts render directly."""
    l1 = dom.get("layer1", {}) or {}
    l2 = dom.get("layer2", {}) or {}
    ma = dom.get("meta_auditor", {}) or {}

    l1_asr = l1.get("base_asr")
    l2_asr = l2.get("firewall_asr", l2.get("base_asr"))
    recall = ma.get("recall")
    # Residual ASR after the auditor: of attacks that reached it, the fraction its recall missed.
    # recall == 1.0 -> 0 residual (auditor caught every attack). Derived, not invented: mirrors the
    # layer3_meta_auditor_asr column in asr_matrix.csv (0 when all compromised attacks are caught).
    l3_asr = round(l1_asr * (1 - recall), 2) if (l1_asr is not None and recall is not None) else None

    fp, tn = ma.get("fp", 0), ma.get("tn", 0)
    return {
        "domain": name,
        "n_scenarios": dom.get("n_scenarios"),
        "asr": {"layer1_no_defense": l1_asr, "layer2_firewall": l2_asr, "layer3_auditor": l3_asr},
        "auditor": {
            "precision": ma.get("precision"),
            "recall": recall,
            "f1": ma.get("f1"),
            "tp": ma.get("tp"), "fp": fp, "fn": ma.get("fn"), "tn": tn,
            "fpr": _fpr(fp, tn),
        },
    }


def _pct(comp: Optional[int], total: Optional[int]) -> Optional[float]:
    """comp/total as a 0-100 percentage, or None when there's nothing to divide."""
    if not total:
        return None
    return round(100.0 * (comp or 0) / total, 2)


def _shape_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the legacy per-domain schema ({domain: {layer1, layer2, meta_auditor}})."""
    domains = [
        _domain_summary(name, val)
        for name, val in data.items()
        if not name.startswith("_") and isinstance(val, dict) and _DOMAIN_MARKER_KEYS & set(val.keys())
    ]

    # Portfolio roll-up (scenario-weighted) so the header can show one headline ASR per tier.
    def _weighted(tier: str) -> Optional[float]:
        num = den = 0.0
        for d in domains:
            asr, n = d["asr"].get(tier), d.get("n_scenarios") or 0
            if asr is not None and n:
                num += asr * n
                den += n
        return round(num / den, 2) if den else None

    return {
        "layers": [{"key": key, "name": name} for key, name in _LEGACY_LAYERS],
        "domains": domains,
        "overall_asr": {key: _weighted(key) for key, _ in _LEGACY_LAYERS},
        "usage": data.get("_usage"),
        # Legacy per-domain files never recorded which model played the mesh vs. the auditor role
        # (only aggregate _usage.by_model token counts across ALL models used, unattributed to a
        # role) — surface that gap explicitly rather than guessing. The dashboard renders these as
        # "not recorded" so a demo never implies a labeled number that isn't actually labeled.
        "mesh_model": None,
        "auditor_model": None,
    }


def _shape_ladder(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the ladder run schema ({meta, asr:{L1..L5}, auditor:{overall, per_domain}})."""
    asr = data.get("asr", {}) or {}
    aud = data.get("auditor", {}) or {}
    per_domain = aud.get("per_domain", {}) or {}

    # Which layers actually ran, in ladder order. Prefer meta.layers; fall back to asr keys.
    order = ["L1", "L2", "L3", "L4", "L5"]
    present = (data.get("meta", {}) or {}).get("layers") or list(asr.keys())
    layer_keys = [L for L in order if L in present]
    layers = [{"key": L, "name": _LADDER_LAYER_NAMES.get(L, L)} for L in layer_keys]

    # Per-domain compromise counts live under asr[layer]["by_domain"][domain] = {comp, total}.
    def _dom_asr(layer: str, dom: str) -> Optional[float]:
        cell = (asr.get(layer, {}) or {}).get("by_domain", {}).get(dom)
        return _pct(cell.get("comp"), cell.get("total")) if isinstance(cell, dict) else None

    domains = []
    for name, ma in per_domain.items():
        ma = ma or {}
        fp, tn = ma.get("fp", 0), ma.get("tn", 0)
        # Malicious n for this domain = total attacks it saw at L1 (denominator is layer-invariant).
        l1_cell = (asr.get("L1", {}) or {}).get("by_domain", {}).get(name, {}) or {}
        domains.append({
            "domain": name,
            "n_scenarios": l1_cell.get("total"),
            "asr": {L: _dom_asr(L, name) for L in layer_keys},
            "auditor": {
                "precision": ma.get("precision"),
                "recall": ma.get("recall"),
                "f1": ma.get("f1"),
                "tp": ma.get("tp"), "fp": fp, "fn": ma.get("fn"), "tn": tn,
                "fpr": _fpr(fp, tn),
            },
        })

    # Headline ASR per layer comes straight from each layer's own ``overall`` roll-up — authoritative,
    # not re-derived (the ladder already scenario-weighted it across all domains).
    def _overall(layer: str) -> Optional[float]:
        ov = (asr.get(layer, {}) or {}).get("overall", {})
        return ov.get("asr_pct") if isinstance(ov, dict) else None

    # Benign over-refusal (FPR) per layer — the auditor-quality story that L3-only stats can't show.
    def _benign_fpr(layer: str) -> Optional[float]:
        bor = (asr.get(layer, {}) or {}).get("benign_over_refusal", {})
        return bor.get("fpr_pct") if isinstance(bor, dict) else None

    # Usage from the auditor layer (L5) if present, else the last layer that ran.
    usage = None
    for L in reversed(layer_keys):
        usage = (asr.get(L, {}) or {}).get("usage")
        if usage:
            break

    # The ladder runner (run_full_eval.py) records which mesh (attack surface) and auditor (defense)
    # model produced this run under meta.mesh_model / meta.auditor_model. Read straight from the same
    # file the numbers came from — never hardcode, since a demo showing unlabeled ASR numbers next to
    # the wrong model name is worse than showing none.
    meta = data.get("meta", {}) or {}

    return {
        "layers": layers,
        "domains": domains,
        "overall_asr": {L: _overall(L) for L in layer_keys},
        "overall_fpr": {L: _benign_fpr(L) for L in layer_keys},
        "usage": usage,
        "mesh_model": meta.get("mesh_model"),
        "auditor_model": meta.get("auditor_model"),
    }


def _shape_final_eval(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the frozen held-out eval schema (run_final_eval.py: {meta, overall, by_domain, ...}).

    Two layers are reported: L2 (multi-agent, no defense) and L5 (meta-auditor). L5's ASR is the
    *residual* — attacks that landed AND the auditor missed — so the chart shows the auditor
    collapsing attack success, mirroring the paper's headline ladder. Auditor quality (precision/recall/FPR)
    comes straight from each domain's confusion counts; nothing is re-derived here.
    """
    meta = data.get("meta", {}) or {}
    overall = data.get("overall", {}) or {}
    by_domain = data.get("by_domain", {}) or {}

    layers = [
        {"key": "L2", "name": "Multi-Agent No Defense"},
        {"key": "L5", "name": "Meta-Auditor (residual)"},
    ]

    def _asr_pct(v: Optional[float]) -> Optional[float]:
        return round(100.0 * v, 2) if isinstance(v, (int, float)) else None

    domains = []
    for name, dom in by_domain.items():
        dom = dom or {}
        fp, tn = dom.get("fp", 0), dom.get("tn", 0)
        domains.append({
            "domain": name,
            "n_scenarios": dom.get("n"),
            "n_malicious": dom.get("n_malicious"),
            "n_benign": dom.get("n_benign"),
            "asr": {"L2": _asr_pct(dom.get("l2_asr")), "L5": _asr_pct(dom.get("l5_asr"))},
            "auditor": {
                "precision": dom.get("precision"),
                "recall": dom.get("recall_tpr"),
                "f1": dom.get("f1"),
                "tp": dom.get("tp"), "fp": fp, "fn": dom.get("fn"), "tn": tn,
                "fpr": _fpr(fp, tn),
            },
        })

    return {
        "layers": layers,
        "domains": domains,
        "overall_asr": {"L2": _asr_pct(overall.get("l2_asr")), "L5": _asr_pct(overall.get("l5_asr"))},
        "overall_fpr": {"L5": round(100.0 * overall.get("fpr", 0), 2) if "fpr" in overall else None},
        # Scenario totals for the panel labels (frozen held-out set).
        "n_total": overall.get("n"),
        "n_malicious": overall.get("n_malicious"),
        "n_benign": overall.get("n_benign"),
        "usage": None,
        "mesh_model": meta.get("mesh_model"),
        "auditor_model": meta.get("auditor_model"),
    }


def load_metrics(source: str = "latest") -> Dict[str, Any]:
    """Load one metrics file. ``source`` is 'latest' or a filename within the results dir."""
    files = _list_metrics_files()
    if not files:
        raise HTTPException(status_code=404, detail=f"No metrics result files found in {_RESULTS_DIR}")

    if source == "latest":
        path = files[0]
    else:
        # Match by basename against the sourceable set (includes the pinned subdir file); basename
        # only — no path traversal, and never widens beyond what _list_metrics_files() vetted.
        wanted = Path(source).name
        path = next((p for p in files if p.name == wanted), None)
        if path is None:
            raise HTTPException(status_code=404, detail=f"'{source}' is not an available metrics file")

    with open(path) as f:
        data = json.load(f)

    if _is_final_eval_file(data):
        shaped = _shape_final_eval(data)
    elif _is_ladder_metrics_file(data):
        shaped = _shape_ladder(data)
    else:
        shaped = _shape_legacy(data)
    return {"source_file": path.name, "mtime": path.stat().st_mtime, **shaped}


@router.get("/admin/metrics")
async def get_metrics(source: str = Query(default="latest")) -> Dict[str, Any]:
    return load_metrics(source)


@router.get("/admin/metrics/files")
async def list_metrics_files() -> Dict[str, Any]:
    """Available metrics files (canonical first) for the dashboard's source dropdown."""
    files = _list_metrics_files()

    def _label(p: Path) -> str:
        # Disambiguate the pinned final-eval (basename collides with nothing else, but its bare name
        # 'eval_results.json' reads as ambiguous next to the legacy files in the dropdown).
        if p == _FINAL_EVAL_PATH:
            return f"{p.name} — FINAL held-out (opus-4-8)"
        return p.name

    return {
        "results_dir": str(_RESULTS_DIR),
        "files": [{"name": p.name, "label": _label(p), "mtime": p.stat().st_mtime} for p in files],
    }


# Attack class -> real-world impact tag, aligned to the LIVE blast-radius vocabulary emitted by
# state_store._check_violations (financial_loss, data_exfiltration, privilege_escalation, ...). The
# frozen eval rows carry {domain, attack_class} but no tool name, so the static view maps by attack
# class instead of by tool — same tag vocabulary, so live and static cells read identically.
_ATTACK_CLASS_IMPACT = {
    "authority_escalation": ["privilege_escalation"],
    "entitlement_manipulation": ["unauthorized_access_escalation"],
    "policy_reframing": ["governance_bypassed"],
    "compliance_theater": ["governance_bypassed"],
    "sla_metric_gaming": ["sla_gaming"],
    "warranty_fraud_injection": ["financial_loss"],
    "supply_chain_poisoning": ["supply_chain_compromise"],
    "lineage_forgery": ["supply_chain_compromise"],
    "split_evidence_corroboration": ["data_integrity_compromise"],
    "semantic_state_poisoning": ["data_integrity_compromise"],
    "shared_memory_pollution": ["data_integrity_compromise"],
    "reasoning_influence": ["unhedged_risk_exposure"],
    "temporal_exploitation": ["stale_state_exploited"],
    "intermediate_step_leakage": ["sensitive_disclosure"],
    "identity_spoofing_gaslight": ["privilege_escalation"],
    "crisis_escalation_hijack": ["privilege_escalation"],
}


def _static_blast_radius() -> Dict[str, Any]:
    """Blast-radius derived from the FROZEN final-eval, shaped exactly like the live endpoint.

    A "compromised operation" here = an L2-landed malicious scenario (``solver_compromised`` true):
    the attack reached the enterprise action in the no-defense mesh. Effects are grouped so the
    dashboard's per-domain cells light up from the frozen paper run with no live traffic needed.
    """
    if not _FINAL_EVAL_PATH.is_file():
        return {"mode": "static", "source_file": None, "total_operations": 0,
                "compromised_operations": 0, "domains_affected": [], "unique_impact_tags": [],
                "cascading_effects": []}

    with open(_FINAL_EVAL_PATH) as f:
        data = json.load(f)

    rows = data.get("rows", []) or []
    landed = [r for r in rows if r.get("solver_compromised")]
    all_tags: List[str] = []
    effects = []
    for r in landed:
        cls = r.get("attack_class") or "unknown"
        tags = _ATTACK_CLASS_IMPACT.get(cls, ["policy_violation"])
        all_tags.extend(tags)
        effects.append({
            "trigger": f"{cls} ({r.get('scenario_id')})",
            "violations": [f"l2_landed:{cls}"],
            "impact_tags": tags,
            "domain": r.get("domain", "unknown"),
            "tier": r.get("tier"),
            "seq": r.get("scenario_id"),
        })

    return {
        "mode": "static",
        "source_file": _FINAL_EVAL_PATH.name,
        "total_operations": len(rows),
        "compromised_operations": len(landed),
        "domains_affected": sorted({e["domain"] for e in effects}),
        "unique_impact_tags": sorted(set(all_tags)),
        "cascading_effects": effects,
    }


@router.get("/admin/blast-radius/static")
async def blast_radius_static() -> Dict[str, Any]:
    """Frozen-eval blast radius (no live traffic needed) — same shape as /admin/blast-radius."""
    return _static_blast_radius()
