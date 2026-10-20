#!/usr/bin/env python3
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the FULL 4-domain verification_manifest.json — OFFLINE, no gateway.

The manifest shipped with the frozen pool covered only `autonomous_crm` (domains:
["autonomous_crm"]). This rebuilds it across ALL four domains by REUSING the exact same
extraction + writer the generator uses:

  * `_load_checkpoint(out, domain, ...)`  -> the recovered-payload rows, straight from the
    frozen `.ckpt_<domain>.jsonl` checkpoints (source of truth, byte-frozen at generation).
  * `_write_verification_manifest(...)`   -> re-hashes each domain's corpus (PDF bytes +
    page text) and writes the manifest in the identical schema.

Because it reuses the generator's own functions, the output is format-identical to a fresh
run's manifest — just with all four domains present. No LLM calls, no sandbox, no contention
with anything else running.

Usage:
    .venv/bin/python regen_verification_manifest.py
      [--pool results/full_eval] [--corpus-dir results/shared_corpus]
"""
from __future__ import annotations

import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_dataset as gd

ALL_DOMAINS = ("autonomous_crm", "cicd_supply_chain", "data_engineering", "quantitative_finance")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/full_eval",
                    help="Dir holding .ckpt_<domain>.jsonl + the dataset (also the manifest out dir).")
    ap.add_argument("--corpus-dir", default="results/shared_corpus")
    ap.add_argument("--out", default=None,
                    help="Dataset path used only to locate checkpoints (default <pool>/dataset.json).")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.pool, "dataset.json")
    tiers = [1, 2, 3]

    # _load_checkpoint validates a config fingerprint against the checkpoint's .meta.json. Pull the
    # frozen config out of each meta so this run matches (seed/count/model) and never trips the guard.
    import json
    metas = {}
    for dom in ALL_DOMAINS:
        _, meta_path = gd._ckpt_paths(out_path, dom)
        if not os.path.exists(meta_path):
            print(f"ERROR: missing checkpoint meta for {dom}: {meta_path}", file=sys.stderr)
            return 2
        metas[dom] = json.load(open(meta_path))

    # All domains were generated in one run -> shared seed/count/model. Build the args shim from
    # the first meta and assert the rest agree (fail loud on any drift).
    ref = metas[ALL_DOMAINS[0]]
    for dom, m in metas.items():
        for k in ("seed", "count", "model"):
            if m.get(k) != ref.get(k):
                print(f"ERROR: checkpoint meta drift on {k!r}: {dom}={m.get(k)} vs "
                      f"{ALL_DOMAINS[0]}={ref.get(k)}", file=sys.stderr)
                return 3

    shim = types.SimpleNamespace(
        out=out_path,
        corpus_dir=args.corpus_dir,
        seed=ref.get("seed"),
        count=ref.get("count"),
        model=ref.get("model"),
        tiers=",".join(str(t) for t in tiers),
        mock=bool(ref.get("mock", False)),
    )

    all_payload_rows = []
    per_domain = {}
    for dom in ALL_DOMAINS:
        _done, _rows, prows = gd._load_checkpoint(out_path, dom, shim, tiers)
        per_domain[dom] = len(prows)
        all_payload_rows.extend(prows)

    print("Recovered payload rows per domain:")
    for dom, n in per_domain.items():
        print(f"  {dom:24} {n}")
    print(f"  {'TOTAL':24} {len(all_payload_rows)}")

    out_dir = os.path.dirname(out_path) or "."
    gd._write_verification_manifest(
        args.corpus_dir, all_payload_rows, list(ALL_DOMAINS), shim, out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
