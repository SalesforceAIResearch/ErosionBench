"""Watermark Scrubber — strips benchmark tells from scenario objects before mesh execution.

Removes phrases like "Forged SEMANTIC_STATE_POISONING evidence" or other benchmark
watermarks from agent-visible text. The agent under test never sees these tells —
this ensures the benchmark measures real robustness, not keyword-spotting.

Ground-truth labels, structured keys, and machine-readable metadata are preserved.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .schemas import ContextDocument, EnterpriseScenario, MemoryState

__all__ = ["scrub_scenario", "scrub_scenarios"]

_WATERMARK_PATTERNS: List[re.Pattern] = [
    re.compile(r"Forged\s+[A-Z_]+\s+evidence", re.IGNORECASE),
    # "Synthetic Benchmark Artifact" heading AND the block-quote "Synthetic Benchmark
    # Notice:" tell (optionally markdown-decorated with '>' / '**').
    re.compile(r">?\s*\*{0,2}Synthetic\s+Benchmark\s+(?:Artifact|Notice)\*{0,2}:?", re.IGNORECASE),
    re.compile(r"For\s+Evaluation\s+Only\.?", re.IGNORECASE),
    re.compile(r"\[BENCHMARK[^\]]*\]", re.IGNORECASE),
    re.compile(r"\(generated\s+for\s+alignment\s+decay\s+benchmark\)", re.IGNORECASE),
    # Evaluator disclaimer sentence ("This document is not authorization to deploy...").
    # Matches the whole clause up to its terminating period so the real instruction in a
    # preceding sentence is preserved but the tell is removed in full.
    re.compile(r"[^.]*\bnot\s+authorization\b[^.]*\.", re.IGNORECASE),
]


def scrub_text(text: str) -> Tuple[str, int]:
    """Remove benchmark watermark phrases from text. Returns (cleaned, count_removed)."""
    count = 0
    for pat in _WATERMARK_PATTERNS:
        text, n = pat.subn("", text)
        count += n
    text = re.sub(r"  +", " ", text).strip()
    return text, count


def _scrub_document(doc: ContextDocument) -> ContextDocument:
    """Return a new ContextDocument with watermarks removed from pages."""
    scrubbed_pages: List[str] = []
    for page in doc.pages:
        clean, _ = scrub_text(page)
        scrubbed_pages.append(clean)
    return ContextDocument(
        title=doc.title,
        pages=scrubbed_pages,
        needle_page=doc.needle_page,
    )


def _scrub_memory(idx: int, mem: MemoryState) -> MemoryState:
    """Return a new MemoryState with watermarks removed from narrative."""
    if mem.narrative:
        clean_narrative, _ = scrub_text(mem.narrative)
    else:
        clean_narrative = mem.narrative

    clean_request = mem.request
    if mem.request:
        clean_request, _ = scrub_text(mem.request)

    return MemoryState(
        action=mem.action,
        ledger_delta=mem.ledger_delta,
        request=clean_request,
        narrative=clean_narrative,
        source=mem.source,
    )


def scrub_scenario(scenario: EnterpriseScenario) -> EnterpriseScenario:
    """Return a deep copy of the scenario with agent-visible watermarks removed.

    Scrubs:
      - long_context_documents[].pages (the text agents read)
      - session_history[].narrative (the framing text agents may see)
      - session_history[].request (the user request text)

    Preserves:
      - All structured/ground-truth fields (ledger_delta, assertions, etc.)
      - Machine-readable provenance and scenario metadata
      - The scenario_id, execution_spec, attack_class, etc.
    """
    scrubbed_docs = [_scrub_document(doc) for doc in scenario.long_context_documents]

    scrubbed_history: Dict[int, MemoryState] = {}
    for idx, mem in scenario.session_history.items():
        scrubbed_history[idx] = _scrub_memory(idx, mem)

    return scenario.model_copy(update={
        "long_context_documents": scrubbed_docs,
        "session_history": scrubbed_history,
    })


def scrub_scenarios(scenarios: List[EnterpriseScenario]) -> List[EnterpriseScenario]:
    """Scrub watermarks from a batch of scenarios."""
    return [scrub_scenario(s) for s in scenarios]
