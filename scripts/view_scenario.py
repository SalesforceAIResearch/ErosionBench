#!/usr/bin/env python3
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Human-readable browser for generated scenarios.

Usage:
    .venv/bin/python view_scenario.py                       # list all scenario ids
    .venv/bin/python view_scenario.py CR_SLA_METR_0003      # show one in full
    .venv/bin/python view_scenario.py --grep sla            # filter ids by substring
    .venv/bin/python view_scenario.py --file data/eval.json CR_...
"""
from __future__ import annotations

import argparse
import json
import textwrap


def _short(v, n=220):
    if v is None:
        return "None"
    return textwrap.shorten(" ".join(str(v).split()), n)


def show(s):
    print("=" * 80)
    print(f'{s["scenario_id"]}   ({s.get("name")})')
    print("=" * 80)
    for k in ("is_malicious", "attack_class", "threat_category", "owasp_llm_category",
              "target_agent", "enterprise_context"):
        print(f"  {k:20s}: {s.get(k)}")
    print(f'  {"user_intent":20s}: {_short(s.get("user_intent"))}')
    print(f'  {"attacker_objective":20s}: {_short(s.get("attacker_objective"))}')

    es = s.get("execution_spec", {}) or {}
    print(f'  {"tool":20s}: {es.get("tool")}')
    print(f'  {"base_args":20s}: {json.dumps(es.get("base_args", {}))[:200]}')
    if es.get("conditioning"):
        print(f'  {"conditioning":20s}: {_short(es.get("conditioning"))}')

    sh = s.get("session_history", {}) or {}
    print(f'  {"session_history":20s}: {len(sh)} sessions')
    for k, v in list(sh.items())[:6]:
        print(f"       [{k}] {json.dumps(v)[:200]}")

    docs = s.get("long_context_documents", []) or []
    print(f'  {"documents":20s}: {len(docs)}')
    for dcm in docs[:3]:
        pages = dcm.get("pages", [])
        print(f'       - {dcm.get("title")}  ({len(pages)} page(s))')

    man = s.get("document_manifest", {}) or {}
    if man.get("needle"):
        print(f'  {"needle":20s}: {json.dumps(man["needle"])[:200]}')

    gc = s.get("ground_truth_contracts", []) or []
    print(f'  {"gt_contracts":20s}: {[c.get("name") for c in gc]}')
    for c in gc:
        print(f'       - {c.get("name")} [{c.get("severity")}/{c.get("assertion_type")}]: '
              f'{_short(c.get("description"), 120)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_id", nargs="?", help="scenario id to show in full")
    ap.add_argument("--file", default="data/dataset.json")
    ap.add_argument("--grep", help="filter listed ids by substring (case-insensitive)")
    ap.add_argument("--full", action="store_true", help="dump raw JSON of the scenario")
    args = ap.parse_args()

    data = json.load(open(args.file))
    by_id = {s["scenario_id"]: s for s in data}

    if args.scenario_id:
        s = by_id.get(args.scenario_id)
        if not s:
            print(f"no scenario {args.scenario_id!r} in {args.file}")
            return 1
        if args.full:
            print(json.dumps(s, indent=2))
        else:
            show(s)
        return 0

    # list mode
    g = (args.grep or "").lower()
    ids = [sid for sid in by_id if g in sid.lower()]
    print(f"{len(ids)}/{len(by_id)} scenarios in {args.file}"
          + (f" matching {args.grep!r}" if g else ""))
    for sid in sorted(ids):
        s = by_id[sid]
        tag = "MAL" if s["is_malicious"] else "ben"
        print(f"  [{tag}] {sid:34s} {s.get('attack_class') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
