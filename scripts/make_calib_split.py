#!/usr/bin/env python3
"""Carve ONE generated pool into two FAMILY-DISJOINT dataset files.

The calibration artifact FITS on the calibration side, so its families can never be
used for evaluation (``assert_not_fit_on`` enforces this at load). Rather than generate
a separate file (which re-mints the SAME seed-independent family IDs and would collide),
we split a single pool by family: every base/_CONTROL/_T2/_T3 of a family lands wholly on
one side. The eval side is then a genuinely held-out, leak-free validation set.

Usage:
    .venv/bin/python make_calib_split.py \
        --pool data/dataset.json \
        --calib-out results/split/calib.json \
        --eval-out results/split/eval.json \
        --calibration-fraction 0.65 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from erosionbench.eval_pipeline.calibration_workflow import family_of, family_split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="Generated pool dataset.json")
    ap.add_argument("--calib-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--calibration-fraction", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    all_ids = [s["scenario_id"] for s in pool]
    split = family_split(all_ids, calibration_fraction=args.calibration_fraction, seed=args.seed)
    split.assert_disjoint()

    calib_ids = set(split.calibration_ids)
    eval_ids = set(split.evaluation_ids)
    calib = [s for s in pool if s["scenario_id"] in calib_ids]
    evalset = [s for s in pool if s["scenario_id"] in eval_ids]

    # Hard invariant: no family straddles the two files.
    calib_fams = {family_of(s["scenario_id"]) for s in calib}
    eval_fams = {family_of(s["scenario_id"]) for s in evalset}
    overlap = calib_fams & eval_fams
    assert not overlap, f"LEAK: families on both sides: {sorted(overlap)[:5]}"

    def _mal(rows):
        return sum(1 for s in rows if s.get("is_malicious"))

    os.makedirs(os.path.dirname(args.calib_out) or ".", exist_ok=True)
    json.dump(calib, open(args.calib_out, "w"), indent=2)
    json.dump(evalset, open(args.eval_out, "w"), indent=2)

    print(f"pool           : {len(pool)} scenarios / {len(calib_fams | eval_fams)} families")
    print(f"calibration    : {len(calib):3d} scenarios / {len(calib_fams):3d} families "
          f"(malicious={_mal(calib)}, benign={len(calib) - _mal(calib)})  -> {args.calib_out}")
    print(f"held-out eval  : {len(evalset):3d} scenarios / {len(eval_fams):3d} families "
          f"(malicious={_mal(evalset)}, benign={len(evalset) - _mal(evalset)})  -> {args.eval_out}")
    print(f"family overlap : {len(overlap)} (must be 0)")
    print(f"benign(calib)  : {len(calib) - _mal(calib)}  (need >=50 for alpha=0.05 conformal sufficiency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
