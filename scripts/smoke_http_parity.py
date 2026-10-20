#!/usr/bin/env python3
"""End-to-end smoke test: in-process vs live (HTTP) document read path.

Runs a few DOCUMENT-GROUNDED scenarios through the mesh in scripted mode (no API keys) twice:
  1. In-process  (EROSIONBENCH_SANDBOX_URL unset)  -> DocumentServer splices locally.
  2. Live        (EROSIONBENCH_SANDBOX_URL set)     -> HttpDocumentServer serves reads over HTTP.

Because both paths use the one canonical splice_needle and the sandbox serves the same uploaded
corpus, the solver verdict and all four formal metrics must be identical. This exercises the full
Step-3 wiring (corpus upload, needle set/clear, HTTP fetch) against a real running sandbox.

Usage:
    # in-process only (default): python smoke_http_parity.py
    # live half needs a sandbox already running at $EROSIONBENCH_SANDBOX_URL
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from erosionbench.eval_pipeline.datagen.assertion_registry import compile_assertions
from erosionbench.eval_pipeline.datagen.state_graph import GroundTruthContract
from erosionbench.eval_pipeline.formal_metrics import compute_formal_metrics
from erosionbench.eval_pipeline.langgraph_mesh import LangGraphMesh, MeshConfig
from erosionbench.eval_pipeline.schemas import EnterpriseScenario
from erosionbench.eval_pipeline.solver import DeterministicMeshSolver
from erosionbench.eval_pipeline.watermark_scrubber import scrub_scenario

DATASET = "data/sample/sample_dataset.json"
MAX_SCENARIOS = 3


def load_doc_grounded(path: str, limit: int):
    with open(path) as f:
        raw = json.load(f)
    scenarios = []
    for item in raw:
        if not item.get("document_manifest"):
            continue
        contracts_data = item.pop("ground_truth_contracts", [])
        item.pop("ground_truth_assertions", None)
        contracts = [
            GroundTruthContract(
                name=c["name"], description=c["description"],
                severity=c["severity"], assertion_type=c["assertion_type"],
                params=c.get("params", {}),
            ) for c in contracts_data
        ]
        assertions = compile_assertions(contracts)
        scenario = EnterpriseScenario.model_validate(item)
        scenarios.append(scenario.model_copy(update={"ground_truth_assertions": assertions}))
        if len(scenarios) >= limit:
            break
    return scenarios


def run_once(scenarios):
    """Run all scenarios scripted; return {scenario_id: (verdict, metrics_tuple)}."""
    config = MeshConfig(
        mode="multi", model="mock", firewall_kind="none",
        scripted_agent=True, attacker="scripted", partial_observability=True,
    )
    solver = DeterministicMeshSolver()
    out = {}
    for scenario in scenarios:
        scrubbed = scrub_scenario(scenario)
        mesh = LangGraphMesh(config)
        context, _trace = mesh.run(scrubbed)
        result = solver.evaluate(scenario, context)
        m = compute_formal_metrics(scenario, context)
        out[scenario.scenario_id] = (
            result.is_compromised,
            (round(m.vpr, 6), round(m.sdi, 6), round(m.cii, 6), round(m.vad, 6), m.all_anomalous),
        )
    return out


def main() -> int:
    scenarios = load_doc_grounded(DATASET, MAX_SCENARIOS)
    if not scenarios:
        print("No document-grounded scenarios found in", DATASET)
        return 1
    print(f"Loaded {len(scenarios)} document-grounded scenarios:",
          ", ".join(s.scenario_id for s in scenarios))

    url = os.environ.get("EROSIONBENCH_SANDBOX_URL")

    # In-process pass (force unset regardless of the ambient env).
    saved = os.environ.pop("EROSIONBENCH_SANDBOX_URL", None)
    print("\n=== IN-PROCESS pass ===")
    inproc = run_once(scenarios)
    for sid, (v, m) in inproc.items():
        print(f"  {sid:<40} compromised={v}  metrics(vpr,sdi,cii,vad,all_anom)={m}")

    if not url:
        if saved:
            os.environ["EROSIONBENCH_SANDBOX_URL"] = saved
        print("\nEROSIONBENCH_SANDBOX_URL not set — skipping live pass. "
              "Start the sandbox and re-run to compare.")
        return 0

    # Live pass.
    os.environ["EROSIONBENCH_SANDBOX_URL"] = url
    print(f"\n=== LIVE pass (EROSIONBENCH_SANDBOX_URL={url}) ===")
    live = run_once(scenarios)
    for sid, (v, m) in live.items():
        print(f"  {sid:<40} compromised={v}  metrics(vpr,sdi,cii,vad,all_anom)={m}")

    print("\n=== PARITY ===")
    ok = True
    for sid in inproc:
        if inproc[sid] != live.get(sid):
            ok = False
            print(f"  MISMATCH {sid}: in-process={inproc[sid]} live={live.get(sid)}")
        else:
            print(f"  OK       {sid}")
    print("\nRESULT:", "PARITY HOLDS" if ok else "PARITY BROKEN")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
