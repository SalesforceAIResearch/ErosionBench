"""Runtime document server — injects per-scenario needles into shared corpus at execution time.

At scenario execution time, when an agent calls `fetch_document(doc_id, page_range)`:
  1. Load the base corpus document (shared across all scenarios)
  2. Look up the scenario's needle (which doc, which page, what content)
  3. Splice the needle into the appropriate page
  4. Return the combined content (or render to a per-scenario PDF on-the-fly)

This means:
  - 700 scenarios share the SAME ~15 corpus PDFs
  - Each scenario has a unique needle (a ~200-char JSON object)
  - At runtime, the harness composes corpus + needle → realistic document with poisoned evidence
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .corpus_builder import CorpusDocument, ScenarioNeedle
from .needle_renderer import splice_needle

__all__ = ["DocumentServer", "build_scenario_manifest"]


class DocumentServer:
    """Serves documents to agents at execution time with per-scenario needle injection.

    Usage:
        server = DocumentServer(corpus_dir="./dataset/corpus/")
        server.load_corpus("quantitative_finance")
        server.set_scenario_needle(needle)  # one per scenario execution

        # Agent calls during scenario:
        pages = server.fetch_document("qf_risk_committee_minutes", page_range=(3, 7))
        # Returns pages 3-7, with the needle spliced in if injection_page is in range
    """

    def __init__(self, corpus_dir: str | Path):
        self._corpus_dir = Path(corpus_dir)
        self._corpus: Dict[str, CorpusDocument] = {}
        self._needle: Optional[ScenarioNeedle] = None

    def load_corpus(self, domain: str) -> None:
        """Load corpus metadata from the corpus manifest file."""
        manifest_path = self._corpus_dir / domain / "corpus_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            data = json.load(f)

        for doc_data in data["documents"]:
            doc = CorpusDocument(
                doc_id=doc_data["doc_id"],
                title=doc_data["title"],
                domain=doc_data["domain"],
                doc_type=doc_data["doc_type"],
                total_pages=doc_data["total_pages"],
                pages=doc_data["pages"],
                pdf_path=doc_data.get("pdf_path"),
                tags=doc_data.get("tags", []),
                department=doc_data.get("department", ""),
                classification=doc_data.get("classification", "Internal"),
            )
            self._corpus[doc.doc_id] = doc

    def set_scenario_needle(self, needle: ScenarioNeedle) -> None:
        """Set the per-scenario needle for the current execution."""
        self._needle = needle

    def clear_needle(self) -> None:
        """Clear the needle (between scenarios)."""
        self._needle = None

    def fetch_document(
        self,
        doc_id: str,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Fetch document content, injecting the needle if applicable.

        This is the function exposed to agents via the tool harness.
        Returns a dict with title, pages (list of page strings), and metadata.
        """
        if doc_id not in self._corpus:
            return {
                "error": f"Document '{doc_id}' not found in corpus",
                "available": list(self._corpus.keys()),
            }

        doc = self._corpus[doc_id]
        pages = list(doc.pages)

        if self._needle and self._needle.target_doc_id == doc_id:
            injection_idx = self._needle.injection_page
            if 0 <= injection_idx < len(pages):
                pages[injection_idx] = self._inject_needle(
                    original_page=pages[injection_idx],
                    needle=self._needle,
                )

        if page_range:
            start, end = page_range
            start = max(0, start - 1)
            end = min(len(pages), end)
            pages = pages[start:end]
            actual_range = (start + 1, start + len(pages))
        else:
            actual_range = (1, len(pages))

        return {
            "doc_id": doc_id,
            "title": doc.title,
            "department": doc.department,
            "classification": doc.classification,
            "total_pages": doc.total_pages,
            "returned_pages": actual_range,
            "pages": pages,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List available documents in the corpus (for agent discovery)."""
        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "department": doc.department,
                "tags": doc.tags,
                "total_pages": doc.total_pages,
                "classification": doc.classification,
            }
            for doc in self._corpus.values()
        ]

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword search across corpus documents."""
        query_lower = query.lower()
        results = []
        for doc in self._corpus.values():
            score = 0
            if query_lower in doc.title.lower():
                score += 3
            if any(query_lower in tag for tag in doc.tags):
                score += 2
            if query_lower in doc.department.lower():
                score += 1
            for page_idx, page in enumerate(doc.pages):
                if query_lower in page.lower():
                    score += 1
                    if self._needle and self._needle.target_doc_id == doc.doc_id:
                        if page_idx == self._needle.injection_page:
                            score += 2

            if score > 0:
                results.append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "relevance_score": score,
                    "department": doc.department,
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:10]

    def _inject_needle(self, original_page: str, needle: ScenarioNeedle) -> str:
        """Splice needle content into a page via the canonical splicer (shared with the sandbox)."""
        return splice_needle(original_page, needle.needle_content)


def build_scenario_manifest(
    scenario_id: str,
    needle: ScenarioNeedle,
    corpus_domain: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Build the per-scenario manifest (stored in benchmark JSON).

    This is what gets saved per scenario instead of a full PDF.
    At runtime, the execution harness uses this to set up the DocumentServer.
    """
    manifest = {
        "scenario_id": scenario_id,
        "corpus_domain": corpus_domain,
        "needle": {
            "target_doc_id": needle.target_doc_id,
            "injection_page": needle.injection_page,
            "needle_content": needle.needle_content,
            "needle_context": needle.needle_context,
            "retrieval_hint": needle.retrieval_hint,
            "poisoned_chunk_indices": needle.poisoned_chunk_indices,
        },
        "content_hash": hashlib.sha256(
            needle.needle_content.encode()
        ).hexdigest()[:12],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
