# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scenario-compare API — serves per-scenario L2-vs-L5 records for the dashboard panel.

Read-only. Reshapes the per-scenario rows written by ``run_final_eval.py`` into the stable JSON
shape the dashboard's Scenario Compare panel consumes. It reads whichever source is freshest:

  1. A finished run's ``eval_results.json`` (the ``rows`` array), if present, OR
  2. The live-updating ``checkpoint.jsonl`` (one JSON row per line) while a run is in flight.

Both carry the same per-scenario record schema, so the panel works during and after a run.
Sourcing lives entirely here; the panel never learns the on-disk layout. Defaults to the FINAL
eval run dir; override with the ``SCENARIO_COMPARE_DIR`` env var.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Query

router = APIRouter(tags=["scenario-compare"])

# Repo-root/results/eval_run (run_final_eval.py's default EVAL_OUT_DIR) — scenario_compare_api.py -> parents[2] == repo root.
_env_dir = os.environ.get("SCENARIO_COMPARE_DIR", "").strip()
_EVAL_DIR = Path(_env_dir) if _env_dir else (
    Path(__file__).resolve().parents[2] / "results" / "eval_run"
)

# Per-scenario fields carried by run_final_eval.py rows (checkpoint.jsonl / eval_results.json).
_KEEP = (
    "scenario_id", "is_malicious", "tier", "domain", "attack_class",
    "solver_compromised", "auditor_compromised", "detection_mode", "verdict_source",
    "grey_zone_triggered", "semantic_compromise_prob", "calibrated_confidence",
    "formal_verdict", "semantic_verdict", "sdi", "cii", "vad", "vpr",
)


def _load_rows() -> tuple[List[Dict[str, Any]], str]:
    """Return (rows, source_label). Prefer the finished results file; fall back to the live checkpoint."""
    results = _EVAL_DIR / "eval_results.json"
    if results.is_file():
        try:
            data = json.loads(results.read_text())
            rows = data.get("rows", []) if isinstance(data, dict) else []
            if rows:
                return rows, "eval_results.json (final)"
        except (OSError, ValueError):
            pass
    ckpt = _EVAL_DIR / "checkpoint.jsonl"
    if ckpt.is_file():
        rows = []
        with open(ckpt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        if rows:
            return rows, "checkpoint.jsonl (in-progress)"
    return [], "none"


def _project(r: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields the panel renders (drops solver-internal noise, tolerant to missing keys)."""
    return {k: r.get(k) for k in _KEEP}


@router.get("/admin/scenario-compare")
async def scenario_compare(source: str = Query(default="latest")) -> Dict[str, Any]:
    """Per-scenario L2 (solver, no-defense) vs L5 (meta-auditor) records for the compare panel."""
    rows, label = _load_rows()
    return {
        "source_file": label,
        "eval_dir": str(_EVAL_DIR),
        "n": len(rows),
        "rows": [_project(r) for r in rows],
    }


# Showcase manifest + captured replay live in the pool root (parent of eval_run/), written by
# build_showcase_manifest.py and capture_showcase.py respectively. Look there first, but tolerate
# a copy sitting inside eval_run/ too.
def _find_pool_file(name: str) -> Path | None:
    for cand in (_EVAL_DIR.parent / name, _EVAL_DIR / name):
        if cand.is_file():
            return cand
    return None


@router.get("/admin/showcase-t3")
async def showcase_t3() -> Dict[str, Any]:
    """Curated exemplars (with attack narrative) where L2 was compromised but L5 caught it.

    Reads the pre-built ``showcase_t3.json`` manifest (a small hand-picked join of eval outcomes +
    dataset narrative). Falls back to an empty payload if absent so the panel degrades gracefully.
    """
    path = _find_pool_file("showcase_t3.json")
    if path is not None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    return {"meta": {"title": "Showcase", "note": "showcase_t3.json not found", "n": 0},
            "scenarios": []}


@router.get("/admin/showcase-replay")
async def showcase_replay() -> Dict[str, Any]:
    """Pre-captured per-scenario transcripts + auditor reasoning for the replay panel.

    Reads ``showcase_replay.json`` (written by ``capture_showcase.py``). Each scenario carries the
    inter-agent ``transcript`` (turn-by-turn chat), the ``ledger``, L2/L5 outcomes, and the full
    ``auditor`` reasoning (structured reasoning string, verdict_source, detection_mode, tiered
    verdict, formal metrics, and the grey-zone arbiter's rationale/deep_reasoning when it fired).

    Falls back to an empty payload (with a hint) if the capture has not been run yet — the panel
    then shows the manifest metadata without the playable transcript.
    """
    path = _find_pool_file("showcase_replay.json")
    if path is not None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    return {"meta": {"note": "showcase_replay.json not found — run capture_showcase.py to populate "
                             "transcripts + auditor reasoning (needs the gateway).", "n_captured": 0},
            "scenarios": []}
