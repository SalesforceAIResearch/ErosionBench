#!/usr/bin/env python3
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Held-out validation of a frozen calibration artifact (Phase-5 gate).

Loads a FrozenCalibrationArtifact, builds a meta-auditor via ``from_artifact`` (which
runs ``assert_not_fit_on`` — refusing if the eval families overlap the calibration
families), runs a HELD-OUT dataset through mesh + calibrated auditor, and reports:

  * Tier-1 SEPARATION — do the deterministic metrics (VPR/SDI/CII/VAD) still separate
    malicious from benign on our attacks? (the "verify the metrics still align" check).
  * Tier-2 SEPARATION — does the recalculated intent-divergence separate the classes?
  * ACHIEVED FPR — fraction of BENIGN held-out scenarios the calibrated system flags,
    vs the artifact's target alpha. This is the number that says "calibrated", not the
    fit stats (which are in-sample by construction).

Errored scenarios are EXCLUDED (never scored clean). Fails loud without a sandbox URL.

Usage:
    EROSIONBENCH_SANDBOX_URL=http://localhost:8210 \
        CAL_ARTIFACT=results/calib/artifact.json \
        EVAL_DATASET=results/coord_fix2/dataset.json \
        LIVE_MESH_MODEL=gpt-5.5 OPENAI_JUDGE_MODEL=gpt-4.1 \
        EXP_CONCURRENCY=3 \
        .venv/bin/python validate_calibration.py
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
    FrozenCalibrationArtifact,
    SplitLeakageError,
)

ARTIFACT = os.environ.get("CAL_ARTIFACT", "results/calib/artifact.json")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "results/coord_fix2/dataset.json")
MESH_MODEL = os.environ.get("LIVE_MESH_MODEL", "gpt-5.5")
CONC = int(os.environ.get("EXP_CONCURRENCY", "3"))
# Machine-readable validation summary (separation report + achieved FPR/TPR + per-scenario
# records). Defaults next to the artifact so an overnight run leaves a durable result file.
VAL_OUT = os.environ.get("VAL_OUT", os.path.join(os.path.dirname(ARTIFACT) or ".",
                                                 "validation_results.json"))

_METRICS = ("vpr", "sdi", "cii", "vad")


def build(item):
    cd = item.pop("ground_truth_contracts", [])
    item.pop("ground_truth_assertions", None)
    cs = [GroundTruthContract(name=c["name"], description=c["description"], severity=c["severity"],
                              assertion_type=c["assertion_type"], params=c.get("params", {}))
          for c in cd]
    s = EnterpriseScenario.model_validate(item)
    return s.model_copy(update={"ground_truth_assertions": compile_assertions(cs)})


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


async def _eval_one(scenario, mesh_cfg, auditor, sem):
    async with sem:
        try:
            ctx, _, _ = await _run_mesh_async(scenario, mesh_cfg)
            report = await auditor.audit(ctx, scenario)
            tv = getattr(report, "tiered_verdict", None)
            fm = getattr(report, "formal_metrics", None)
            if tv is None or fm is None:
                print(f"    [VAL-SKIP] {scenario.scenario_id}: no formal/tiered verdict")
                return None
            return {
                "scenario_id": scenario.scenario_id,
                "is_malicious": bool(scenario.is_malicious),
                "flagged": bool(report.is_compromised),
                "semantic_compromise_prob": float(tv.semantic_compromise_prob),
                "calibrated_confidence": float(tv.calibrated_confidence),
                "vpr": float(fm.vpr), "sdi": float(fm.sdi),
                "cii": float(fm.cii), "vad": float(fm.vad),
            }
        except Exception as e:
            print(f"    [VAL-SKIP] {scenario.scenario_id}: {type(e).__name__}: {e}")
            return None


async def main() -> int:
    base_url = os.environ.get("EROSIONBENCH_SANDBOX_URL")
    if not base_url:
        raise SystemExit("EROSIONBENCH_SANDBOX_URL must be set (fail loud, no in-process fallback)")

    artifact = FrozenCalibrationArtifact.load(ARTIFACT)
    raw = json.load(open(EVAL_DATASET))
    scenarios = [build(dict(s)) for s in raw]
    eval_ids = [s.scenario_id for s in scenarios]

    print("=== CALIBRATION VALIDATION (held-out) ===")
    print(f"Artifact : {ARTIFACT}  (hash {artifact.content_hash()[:16]}...)")
    print(f"Eval set : {EVAL_DATASET}  ({len(scenarios)} scenarios)")
    print(f"Mesh     : {MESH_MODEL}   Auditor: {artifact.auditor_model}")
    print(f"Sufficient: {artifact.sufficiency.sufficient}   target FPR(alpha)={artifact.target_fpr}")
    print(f"Platt    : fitted={artifact.semantic_platt_fitted} a={artifact.semantic_platt_a} "
          f"b={artifact.semantic_platt_b}   semantic tau={artifact.semantic_conformal_tau}\n")

    # No-leakage gate: from_artifact refuses if any eval family was a calibration family.
    try:
        auditor = MetaAuditorAgent.from_artifact(
            artifact, eval_ids, model=artifact.auditor_model,
            max_concurrency=CONC,
            mesh_config=MeshConfig(mode="multi", model=MESH_MODEL, firewall_kind="none",
                                   scripted_agent=False, attacker="adaptive",
                                   partial_observability=True),
        )
    except SplitLeakageError as e:
        print("REFUSED — evaluation set overlaps the artifact's calibration families:")
        print(f"  {e}")
        print("Use a held-out dataset whose families are disjoint from the calibration set.")
        return 2

    mesh_cfg = auditor._mesh_config
    sem = asyncio.Semaphore(CONC)
    httpx.post(f"{base_url}/admin/reset", timeout=10.0)
    recs = await asyncio.gather(*[_eval_one(s, mesh_cfg, auditor, sem) for s in scenarios])
    recs = [r for r in recs if r is not None]
    excluded = len(scenarios) - len(recs)

    mal = [r for r in recs if r["is_malicious"]]
    ben = [r for r in recs if not r["is_malicious"]]
    print(f"\nscored {len(recs)}/{len(scenarios)} (excluded {excluded}); "
          f"malicious={len(mal)} benign={len(ben)}")

    # -- Tier-1 metric separation --------------------------------------------
    print("\n=== TIER-1 METRIC SEPARATION (mean; malicious vs benign) ===")
    print(f"{'metric':6s} {'malicious':>12s} {'benign':>12s} {'gap':>10s} {'threshold':>12s}")
    for m in _METRICS:
        mm, mb = _mean([r[m] for r in mal]), _mean([r[m] for r in ben])
        thr = artifact.thresholds.get(m)
        thr_s = f"{thr:.4f}" if thr is not None else "None"
        print(f"{m:6s} {mm:12.4f} {mb:12.4f} {mm - mb:+10.4f} {thr_s:>12s}")

    # -- Tier-2 semantic separation ------------------------------------------
    print("\n=== TIER-2 SEMANTIC DIVERGENCE SEPARATION ===")
    print(f"  raw divergence   malicious={_mean([r['semantic_compromise_prob'] for r in mal]):.4f} "
          f"benign={_mean([r['semantic_compromise_prob'] for r in ben]):.4f}")
    print(f"  calibrated conf  malicious={_mean([r['calibrated_confidence'] for r in mal]):.4f} "
          f"benign={_mean([r['calibrated_confidence'] for r in ben]):.4f}")

    # -- Achieved FPR / TPR (the calibration claim) --------------------------
    fp = sum(1 for r in ben if r["flagged"])
    tp = sum(1 for r in mal if r["flagged"])
    fpr = fp / len(ben) if ben else float("nan")
    tpr = tp / len(mal) if mal else float("nan")
    print("\n=== ACHIEVED OPERATING POINT (calibrated auditor) ===")
    print(f"  benign FPR : {fpr:.4f}  ({fp}/{len(ben)})   target alpha={artifact.target_fpr}")
    print(f"  malicious TPR (recall): {tpr:.4f}  ({tp}/{len(mal)})")
    within = (fpr <= artifact.target_fpr) if ben else False
    print(f"  FPR within target: {'YES' if within else 'NO'}")
    if not artifact.sufficiency.sufficient:
        print("  NOTE: artifact.sufficient=False — this is a diagnostic run, "
              "NOT a validated FPR guarantee.")

    # -- Persist a machine-readable summary + per-scenario records -------------
    summary = {
        "artifact": ARTIFACT,
        "artifact_hash": artifact.content_hash(),
        "eval_dataset": EVAL_DATASET,
        "mesh_model": MESH_MODEL,
        "auditor_model": artifact.auditor_model,
        "sufficient": bool(artifact.sufficiency.sufficient),
        "target_fpr": artifact.target_fpr,
        "semantic_platt_fitted": artifact.semantic_platt_fitted,
        "semantic_platt_a": artifact.semantic_platt_a,
        "semantic_platt_b": artifact.semantic_platt_b,
        "semantic_conformal_tau": artifact.semantic_conformal_tau,
        "scored": len(recs), "excluded": excluded,
        "n_malicious": len(mal), "n_benign": len(ben),
        "tier1_separation": {
            m: {"malicious_mean": _mean([r[m] for r in mal]),
                "benign_mean": _mean([r[m] for r in ben]),
                "threshold": artifact.thresholds.get(m)}
            for m in _METRICS
        },
        "tier2_separation": {
            "raw_divergence_malicious": _mean([r["semantic_compromise_prob"] for r in mal]),
            "raw_divergence_benign": _mean([r["semantic_compromise_prob"] for r in ben]),
            "calibrated_conf_malicious": _mean([r["calibrated_confidence"] for r in mal]),
            "calibrated_conf_benign": _mean([r["calibrated_confidence"] for r in ben]),
        },
        "achieved": {"benign_fpr": fpr, "malicious_tpr": tpr,
                     "false_positives": fp, "true_positives": tp,
                     "fpr_within_target": bool(within)},
        "records": recs,
    }
    os.makedirs(os.path.dirname(VAL_OUT) or ".", exist_ok=True)
    with open(VAL_OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[VAL] results written -> {VAL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
