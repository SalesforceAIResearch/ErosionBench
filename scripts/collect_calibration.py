#!/usr/bin/env python3
"""Calibration collection harness — Phase-5 semantic + formal fit (LLM-bound).

Runs a CALIBRATION dataset through the real mesh + meta-auditor and collects the two
things the frozen artifact needs:

  1. Semantic Tier-2 signal — the RECALCULATED continuous intent-divergence score
     (``tiered_verdict.semantic_compromise_prob``, produced by the auditor's
     ``intent_divergence_analysis`` pass), paired with the scenario's ``is_malicious``
     label. Platt scaling is fit on this MIX (needs both classes).

  2. Formal-metric benign nulls — VPR/SDI/CII/VAD from the BENIGN scenarios only, so the
     conformal thresholds guarantee FPR <= alpha distribution-free.

The collected scores feed ``calibration_workflow.build_artifact`` (fits Platt on the mix,
conformal tau on the Platt-calibrated benign scores, and the per-metric benign nulls) and
the result is saved as a versioned, hash-sealed FrozenCalibrationArtifact.

This script FITS — so the families it runs on become the artifact's calibration families and
can NEVER be used for evaluation (``assert_not_fit_on`` enforces this at load). Keep the
evaluation dataset's families disjoint from whatever you pass here.

Nothing is scored fail-open: a scenario whose mesh/auditor run errors is EXCLUDED from the
fit (logged), never silently treated as benign/clean.

Usage:
    EROSIONBENCH_SANDBOX_URL=http://localhost:8210 \
        LIVE_DATASET=data/calib.json \
        LIVE_MESH_MODEL=gpt-5.5 LIVE_AUDITOR_MODEL=claude-haiku-4-5-20251001 \
        OPENAI_JUDGE_MODEL=gpt-4.1 \
        CAL_FRACTION=1.0 CAL_SEED=1337 TARGET_FPR=0.05 EXP_CONCURRENCY=3 \
        CAL_OUT=results/calib/artifact.json \
        .venv/bin/python collect_calibration.py

Env:
    LIVE_DATASET       calibration dataset json (default data/calib.json)
    LIVE_MESH_MODEL    mesh model under audit (default gpt-5.5)
    LIVE_AUDITOR_MODEL meta-auditor model (default claude-haiku-4-5-20251001)
    CAL_FRACTION       fraction of families used to FIT (default 1.0 = all supplied families)
    CAL_SEED           family-split seed (default 1337)
    TARGET_FPR         conformal alpha (default 0.05)
    EXP_CONCURRENCY    concurrent audits (default 3)
    CAL_OUT            artifact output path (default results/calib/artifact.json)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import httpx

from erosionbench.eval_pipeline.datagen.assertion_registry import compile_assertions
from erosionbench.eval_pipeline.datagen.state_graph import GroundTruthContract
from erosionbench.eval_pipeline.langgraph_mesh import MeshConfig
from erosionbench.eval_pipeline.meta_auditor import MetaAuditorAgent, _run_mesh_async
from erosionbench.eval_pipeline.schemas import EnterpriseScenario
from erosionbench.eval_pipeline.calibration_workflow import (
    FamilySplit,
    build_artifact,
    family_split,
    family_of,
)

DATASET = os.environ.get("LIVE_DATASET", "data/calib.json")
MESH_MODEL = os.environ.get("LIVE_MESH_MODEL", "gpt-5.5")
AUDITOR_MODEL = os.environ.get("LIVE_AUDITOR_MODEL", "claude-haiku-4-5-20251001")
CAL_FRACTION = float(os.environ.get("CAL_FRACTION", "1.0"))
CAL_SEED = int(os.environ.get("CAL_SEED", "1337"))
TARGET_FPR = float(os.environ.get("TARGET_FPR", "0.05"))
# Semantic tau scale: "1"/"true" fits Platt then tau on calibrated scores (legacy default);
# "0"/"false" skips Platt and fits tau directly on RAW benign scores (eval then thresholds raw).
CAL_USE_PLATT = os.environ.get("CAL_USE_PLATT", "1").strip().lower() not in ("0", "false", "no")
CONC = int(os.environ.get("EXP_CONCURRENCY", "3"))
CAL_OUT = os.environ.get("CAL_OUT", "results/calib/artifact.json")
CODE_VERSION = os.environ.get("CAL_CODE_VERSION", "cal-collect-v2")
# Resumable checkpoint: each successfully collected record is appended here as one JSON line
# the moment it completes. A re-run loads this file and SKIPS scenarios already collected, so a
# gateway outage mid-run only costs the uncollected tail — not the whole run. Delete it to force
# a clean recollection. Defaults next to the artifact.
CKPT = os.environ.get("CAL_CHECKPOINT",
                      CAL_OUT.replace(".json", "_checkpoint.jsonl"))

# Formal metrics whose benign runs form the conformal nulls. Higher = more anomalous
# for all four (matches formal_metrics orientation used across the auditor).
_METRICS = ("vpr", "sdi", "cii", "vad")
_ORIENTATIONS = {m: True for m in _METRICS}


def all_calibration_split(scenario_ids, seed: int) -> FamilySplit:
    """A split where EVERY family is a calibration family (empty eval side).

    ``family_split`` requires a fraction strictly in (0,1); this single-purpose
    calibration file has no eval side (validation runs on a disjoint dataset), so we
    build the all-calibration split directly and still get leak-free family grouping.
    """
    fam_to_ids = {}
    for sid in scenario_ids:
        fam_to_ids.setdefault(family_of(sid), []).append(sid)
    fams = tuple(fam_to_ids.keys())
    ids = tuple(sid for f in fams for sid in fam_to_ids[f])
    split = FamilySplit(
        calibration_ids=ids,
        evaluation_ids=(),
        calibration_families=fams,
        evaluation_families=(),
        seed=seed,
        calibration_fraction=1.0,
    )
    split.assert_disjoint()
    return split


def build(item):
    cd = item.pop("ground_truth_contracts", [])
    item.pop("ground_truth_assertions", None)
    cs = [GroundTruthContract(name=c["name"], description=c["description"], severity=c["severity"],
                              assertion_type=c["assertion_type"], params=c.get("params", {}))
          for c in cd]
    s = EnterpriseScenario.model_validate(item)
    return s.model_copy(update={"ground_truth_assertions": compile_assertions(cs)})


_ckpt_lock = asyncio.Lock()


def _load_checkpoint(path):
    """Return {scenario_id: record} already collected in a prior run (empty if none)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["scenario_id"]] = rec
            except Exception:
                continue  # tolerate a torn final line from an interrupted run
    return done


async def _append_checkpoint(path, rec):
    """Append one record as a JSON line under a lock (concurrent-safe)."""
    async with _ckpt_lock:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


async def _collect_one(scenario, mesh_cfg, auditor, sem, ckpt_path=None):
    """Run mesh + auditor once; return the calibration record or None on error.

    On success the record is appended to ``ckpt_path`` immediately so a later crash keeps it.
    """
    async with sem:
        try:
            ctx, _, _ = await _run_mesh_async(scenario, mesh_cfg)
            report = await auditor.audit(ctx, scenario)
            tv = getattr(report, "tiered_verdict", None)
            fm = getattr(report, "formal_metrics", None)
            if tv is None or fm is None:
                print(f"    [CAL-SKIP] {scenario.scenario_id}: no formal/tiered verdict")
                return None
            rec = {
                "scenario_id": scenario.scenario_id,
                "is_malicious": bool(scenario.is_malicious),
                "semantic_compromise_prob": float(tv.semantic_compromise_prob),
                "vpr": float(fm.vpr), "sdi": float(fm.sdi),
                "cii": float(fm.cii), "vad": float(fm.vad),
            }
            if ckpt_path:
                await _append_checkpoint(ckpt_path, rec)
            return rec
        except Exception as e:
            print(f"    [CAL-SKIP] {scenario.scenario_id}: {type(e).__name__}: {e}")
            return None


async def main() -> int:
    base_url = os.environ.get("EROSIONBENCH_SANDBOX_URL")
    if not base_url:
        raise SystemExit("EROSIONBENCH_SANDBOX_URL must be set (fail loud, no in-process fallback)")

    raw = json.load(open(DATASET))
    all_ids = [s["scenario_id"] for s in raw]

    # Split by FAMILY so a benign twin / tier variant can't straddle fit and eval. With
    # CAL_FRACTION>=1.0 every supplied family is a calibration family (single-purpose file);
    # a fraction in (0,1) carves out a held-in eval side for a self-contained sanity split.
    if CAL_FRACTION >= 1.0:
        split = all_calibration_split(all_ids, seed=CAL_SEED)
    else:
        split = family_split(all_ids, calibration_fraction=CAL_FRACTION, seed=CAL_SEED)
    cal_ids = set(split.calibration_ids)
    scenarios = [build(dict(s)) for s in raw if s["scenario_id"] in cal_ids]

    print("=== CALIBRATION COLLECTION ===")
    print(f"Dataset : {DATASET}")
    print(f"Mesh    : {MESH_MODEL}   Auditor: {AUDITOR_MODEL}   Judge: {os.environ.get('OPENAI_JUDGE_MODEL','?')}")
    print(f"Sandbox : {base_url}   Concurrency: {CONC}   alpha(FPR)={TARGET_FPR}")
    print(f"Split   : {len(split.calibration_families)} calibration families, "
          f"{len(scenarios)} scenarios (frac={CAL_FRACTION}, seed={CAL_SEED})")
    mal = sum(1 for s in scenarios if s.is_malicious)
    print(f"          malicious={mal}  benign={len(scenarios) - mal}\n")

    mesh_cfg = MeshConfig(mode="multi", model=MESH_MODEL, firewall_kind="none",
                          scripted_agent=False, attacker="adaptive", partial_observability=True)
    auditor = MetaAuditorAgent(model=AUDITOR_MODEL, max_concurrency=CONC, mesh_config=mesh_cfg)
    sem = asyncio.Semaphore(CONC)

    # Resume: replay any records collected in a prior run and only run the uncollected tail.
    done = _load_checkpoint(CKPT)
    if done:
        print(f"[RESUME] checkpoint {CKPT}: {len(done)} already collected — skipping those\n")
    todo = [s for s in scenarios if s.scenario_id not in done]
    resumed = [done[s.scenario_id] for s in scenarios if s.scenario_id in done]

    httpx.post(f"{base_url}/admin/reset", timeout=10.0)
    fresh = await asyncio.gather(*[_collect_one(s, mesh_cfg, auditor, sem, CKPT) for s in todo])
    fresh = [r for r in fresh if r is not None]
    records = resumed + fresh
    excluded = len(todo) - len(fresh)
    print(f"\ncollected {len(records)}/{len(scenarios)} "
          f"(resumed {len(resumed)}, new {len(fresh)}, excluded {excluded} on error this run)")

    # -- Assemble fit inputs --------------------------------------------------
    # Semantic MIX (Platt): every collected scenario's divergence + label.
    semantic_scores = [r["semantic_compromise_prob"] for r in records]
    semantic_labels = [1 if r["is_malicious"] else 0 for r in records]
    # Formal benign NULLs (conformal): metric values from BENIGN scenarios only.
    benign = [r for r in records if not r["is_malicious"]]
    benign_null_samples = {m: [r[m] for r in benign] for m in _METRICS}

    mal_n = sum(semantic_labels)
    print(f"semantic mix: {mal_n} malicious / {len(semantic_labels) - mal_n} benign")
    print(f"benign nulls: n={len(benign)} per metric ({', '.join(_METRICS)})")
    if mal_n == 0 or mal_n == len(semantic_labels):
        print("  [WARN] semantic mix is single-class — Platt will be SKIPPED (need both classes)")

    # Rebuild the fit split over ONLY the collected ids so calibration_families reflects
    # exactly what we fit on (errored scenarios excluded => their families may drop out).
    collected_ids = [r["scenario_id"] for r in records]
    fit_split = all_calibration_split(collected_ids, seed=CAL_SEED)

    artifact = build_artifact(
        scenario_ids=collected_ids,
        split=fit_split,
        mesh_model=MESH_MODEL,
        auditor_model=AUDITOR_MODEL,
        code_version=CODE_VERSION,
        benign_null_samples=benign_null_samples,
        metric_orientations=_ORIENTATIONS,
        target_fpr=TARGET_FPR,
        seeds={"split": CAL_SEED, "bootstrap": CAL_SEED},
        semantic_scores=semantic_scores,
        semantic_labels=semantic_labels,
        semantic_use_platt=CAL_USE_PLATT,
    )

    os.makedirs(os.path.dirname(CAL_OUT) or ".", exist_ok=True)
    artifact.save(CAL_OUT)
    # Persist the raw records alongside for the separation report / audit trail.
    with open(CAL_OUT.replace(".json", "_records.json"), "w") as fh:
        json.dump(records, fh, indent=2)

    print("\n=== ARTIFACT ===")
    print(f"saved            : {CAL_OUT}")
    print(f"sufficient       : {artifact.sufficiency.sufficient}  "
          f"(benign_n={artifact.sufficiency.benign_n}, families={artifact.sufficiency.family_n})")
    if not artifact.sufficiency.sufficient:
        print(f"  reasons        : {'; '.join(artifact.sufficiency.reasons)}")
    print(f"tau scale        : {'PLATT-calibrated' if CAL_USE_PLATT else 'RAW (no Platt)'}")
    print(f"platt fitted     : {artifact.semantic_platt_fitted}  "
          f"(a={artifact.semantic_platt_a}, b={artifact.semantic_platt_b})")
    print(f"semantic tau     : {artifact.semantic_conformal_tau}")
    print(f"metric thresholds: {artifact.thresholds}")
    print(f"content_hash     : {artifact.content_hash()[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
