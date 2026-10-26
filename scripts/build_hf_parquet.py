#!/usr/bin/env python3
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the parquet ``load_dataset`` view of the released splits for the Hugging Face repo.

The canonical split files (``calib.json`` / ``eval.json`` / ``eval_clean_control.json``) are
what the harness reads and what ``PROVENANCE_hashes.txt`` pins, so they are never modified.
Two fields in them are not Arrow-typeable, which breaks ``datasets.load_dataset`` and the Hub
Dataset Viewer:

* ``ground_truth_contracts[].params.threshold`` mixes float / bool / the string sentinel
  ``"__never_matches__"`` across records;
* ``session_history`` is deeply nested and type-unstable across records;
* ``execution_spec.base_args`` holds per-tool arguments (heterogeneous across tools, and
  includes always-empty objects such as ``feature_spec`` that parquet cannot store).

This script writes parquet copies with exactly those fields JSON-encoded as strings
(``json.loads`` recovers the original value); every other field is passed through as-is.

Usage (needs ``pyarrow``, e.g. ``pip install -e ".[images]"``)
-----
    python scripts/build_hf_parquet.py --src data --out data/parquet
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SPLITS = ("calib", "eval", "eval_clean_control")


def to_hf_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``rec`` with the non-Arrow-typeable fields JSON-encoded."""
    out = copy.deepcopy(rec)
    out["session_history"] = json.dumps(rec.get("session_history"), ensure_ascii=False)
    spec = out.get("execution_spec")
    if isinstance(spec, dict) and "base_args" in spec:
        spec["base_args"] = json.dumps(spec["base_args"], ensure_ascii=False)
    for contract in out.get("ground_truth_contracts") or []:
        params = contract.get("params")
        if isinstance(params, dict) and "threshold" in params:
            params["threshold"] = json.dumps(params["threshold"], ensure_ascii=False)
    return out


def build(src: Path, out: Path) -> List[Path]:
    """Write ``<out>/<split>.parquet`` for every split JSON in ``src``; return the paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        records = json.loads((src / f"{split}.json").read_text())
        table = pa.Table.from_pylist([to_hf_record(r) for r in records])
        path = out / f"{split}.parquet"
        pq.write_table(table, path)
        print(f"{path}: {table.num_rows} rows")
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("data"), help="dir with the split JSONs")
    ap.add_argument("--out", type=Path, default=Path("data/parquet"), help="output dir")
    args = ap.parse_args()
    build(args.src, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
