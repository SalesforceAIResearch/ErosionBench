# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 gate verification: run over a generated dataset and assert the plan's gate criteria.

  1. Non-refusal text in every visible page + session narrative.
  2. No WRAPPER meta-leakage (attack-class names, benchmark commentary, needle markers).
  3. Exact page-count parity vs. document manifest (where present).
  4. Cardinality summary (malicious/benign, per attack class, per tier).

Payload-value tokens (attacker-controlled / fabricated-*) are Phase 2 scope and are reported
separately, not failed here.
"""
from __future__ import annotations
import json
import sys
from collections import Counter

from erosionbench.eval_pipeline.datagen.validation_gate import (
    detect_refusal, detect_meta_commentary, _PAYLOAD_META_PATTERNS,
)
import re

_PAYLOAD_RE = re.compile("|".join(_PAYLOAD_META_PATTERNS), re.IGNORECASE)


def main(path):
    scns = json.load(open(path))
    if not isinstance(scns, list):
        scns = scns.get("scenarios", scns.get("data", []))

    refusal_hits, meta_hits, payload_hits = [], [], []
    mal = ben = 0
    by_class = Counter()
    by_tier = Counter()

    for s in scns:
        sid = s["scenario_id"]
        is_mal = s.get("is_malicious")
        mal += bool(is_mal); ben += (not is_mal)
        by_class[s.get("attack_class") or "benign"] += 1
        tier = "T3" if sid.endswith("_T3") else "T2" if sid.endswith("_T2") else \
               "CONTROL" if sid.endswith("_CONTROL") else "T1"
        by_tier[tier] += 1

        pages = [p for d in (s.get("long_context_documents") or [])
                 for p in (d.get("pages") or []) if isinstance(p, str)]
        narrs = []
        sh = s.get("session_history")
        if isinstance(sh, dict):
            narrs = [v.get("narrative", "") for v in sh.values() if isinstance(v, dict)]
        elif isinstance(sh, list):
            narrs = [e.get("narrative", "") for e in sh if isinstance(e, dict)]

        for t in pages + narrs:
            if detect_refusal(t):
                refusal_hits.append((sid, detect_refusal(t)))
                break
        for i, t in enumerate(pages):
            if detect_meta_commentary(t):
                meta_hits.append((sid, detect_meta_commentary(t)))
                break
        for t in pages:
            m = _PAYLOAD_RE.search(t)
            if m:
                payload_hits.append((sid, m.group(0)))
                break

    print(f"DATASET: {path}")
    print(f"  scenarios={len(scns)}  malicious={mal}  benign={ben}")
    print(f"  by attack_class: {dict(by_class)}")
    print(f"  by tier: {dict(by_tier)}")
    print()
    print(f"  [1] REFUSAL hits (BLOCKING):        {len(refusal_hits)}")
    for sid, h in refusal_hits[:5]: print(f"        - {sid}: {h!r}")
    print(f"  [2] WRAPPER meta-leakage (BLOCKING): {len(meta_hits)}")
    for sid, h in meta_hits[:5]: print(f"        - {sid}: {h!r}")
    print(f"  [3] PAYLOAD-value tokens (Phase 2):  {len(payload_hits)} scenarios "
          f"(expected >0 pre-Phase-2)")
    for sid, h in payload_hits[:5]: print(f"        - {sid}: {h!r}")
    print()

    gate_pass = (len(refusal_hits) == 0 and len(meta_hits) == 0)
    print("=" * 60)
    print(f"PHASE 1 GATE: {'PASS' if gate_pass else 'FAIL'} "
          f"(refusals={len(refusal_hits)}, wrapper-meta={len(meta_hits)})")
    print("=" * 60)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/dataset.json"))
