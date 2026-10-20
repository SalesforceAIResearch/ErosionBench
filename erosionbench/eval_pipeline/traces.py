# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Append-only execution-trace persistence.

Every evaluation run appends one self-contained JSON record (LangGraph trajectory, full
multi-turn dialogue, the verifiable ledger transaction chain, firewall decisions, tool
calls, token consumption, and the final verdict) to ``evaluation_traces.json``.

The file is JSON Lines (one JSON object per line) precisely because it is *append-only*:
new runs never rewrite earlier records, so the trace log itself is tamper-evident and
safe to accumulate across many sessions for paper-grade reproducibility.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

__all__ = ["DEFAULT_TRACE_PATH", "append_trace"]

DEFAULT_TRACE_PATH = os.path.join(os.getcwd(), "evaluation_traces.json")


def append_trace(record: Dict[str, Any], path: str = DEFAULT_TRACE_PATH) -> None:
    """Append one run record to the JSONL trace file (created if absent)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
