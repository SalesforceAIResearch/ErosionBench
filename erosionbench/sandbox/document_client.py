# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HttpDocumentServer — the live (over-HTTP) counterpart of the mesh's in-process DocumentServer.

Selected by the mesh when ``EROSIONBENCH_SANDBOX_URL`` is set (Step 3 of the document wiring). It is a
drop-in subclass of :class:`DocumentServer`:

  * ``._corpus`` still holds the inline-built :class:`CorpusDocument`s, so the mesh's direct
    attribute reads (``run.doc_server._corpus[doc_id].pages``) keep working unchanged.
  * ``set_scenario_needle`` / ``clear_needle`` additionally push the needle to the sandbox
    (``POST``/``DELETE /v1/docs/needle``), converting the 0-based ``injection_page`` to the sandbox's
    1-based contract via :func:`scenario_needle_to_payload`.
  * ``fetch_document`` is served over HTTP (``GET /v1/docs/fetch/{doc_id}``) instead of splicing
    locally, so a live run reads exactly what the sandbox serves.

Byte parity: because the sandbox splices with the same canonical ``splice_needle`` and serves the
same base pages (uploaded via ``POST /v1/docs/corpus``), the spliced page bytes are identical to the
in-process path. Range slicing is replicated locally over the full HTTP page list so the
``page_range`` semantics match ``DocumentServer.fetch_document`` exactly.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import httpx

from erosionbench.eval_pipeline.datagen.corpus_builder import CorpusDocument, ScenarioNeedle
from erosionbench.eval_pipeline.datagen.document_server import DocumentServer

from .document_api import CorpusIngest, IngestDoc, NeedlePayload, scenario_needle_to_payload

__all__ = ["HttpDocumentServer"]


class HttpDocumentServer(DocumentServer):
    """DocumentServer whose reads/needle-writes go to a live sandbox over HTTP."""

    def __init__(self, corpus: Dict[str, CorpusDocument], base_url: str, timeout: float = 10.0,
                 session_id: Optional[str] = None) -> None:
        # Do NOT call DocumentServer.__init__ (it wants a corpus_dir); populate the same fields the
        # mesh expects, then upload the corpus so the sandbox serves identical base pages.
        self._corpus_dir = None
        self._corpus = corpus
        self._needle: Optional[ScenarioNeedle] = None
        self._base_url = base_url.rstrip("/")
        # Per-scenario session id, sent on every request as X-Erosionbench-Session so the sandbox keeps
        # this scenario's corpus overlay + needle isolated from other concurrently-running workers.
        # Without this, one worker's session reset/upload wipes another's docs (intermittent PDF
        # 404s) and needle writes clobber each other. Uniqueness must be PER SCENARIO INSTANCE — the
        # tiers of one family (base/_T2/_T3) share doc-ids and base content, so a content-only key
        # would collide and re-introduce the race within a family. When the caller passes the
        # scenario_id we key on it (readable in the sandbox logs); we always append a process-unique
        # suffix so even a repeated scenario_id never shares an overlay. The on-disk PDF cache is
        # content-addressed (not session-keyed), so a unique session id does not cost cache reuse.
        self._session_id = self._make_session_id(session_id)
        self._client = httpx.Client(
            timeout=timeout, headers={"X-Erosionbench-Session": self._session_id}
        )
        # True-sim read path: the agent downloads a real PDF and extracts text itself, mirroring an
        # employee opening a corporate document. On (default), the mesh reads PDF bytes; set
        # EROSIONBENCH_DOC_MODE=text to fall back to the JSON text path (e.g. if pypdf is unavailable).
        self._pdf_mode = os.environ.get("EROSIONBENCH_DOC_MODE", "pdf").strip().lower() != "text"
        self._upload_corpus()

    @staticmethod
    def _make_session_id(scenario_id: Optional[str]) -> str:
        import uuid
        prefix = (scenario_id or "sess").strip().replace(" ", "_")[:48] or "sess"
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    # -- HTTP session lifecycle ----------------------------------------------
    def _upload_corpus(self) -> None:
        """Reset any prior session state, then upload the inline corpus to the sandbox."""
        # Clear residual needle + session corpus from a previous scenario before uploading.
        self._client.delete(f"{self._base_url}/v1/docs/needle")
        self._client.delete(f"{self._base_url}/v1/docs/corpus/session")

        by_domain: Dict[str, List[IngestDoc]] = {}
        for doc in self._corpus.values():
            by_domain.setdefault(doc.domain or "_session", []).append(
                IngestDoc(doc_id=doc.doc_id, pages=list(doc.pages))
            )
        for domain, docs in by_domain.items():
            payload = CorpusIngest(domain=domain, documents=docs)
            self._client.post(f"{self._base_url}/v1/docs/corpus", json=payload.model_dump())

    # -- needle write path ----------------------------------------------------
    def set_scenario_needle(self, needle: ScenarioNeedle) -> None:
        self._needle = needle
        payload = scenario_needle_to_payload(needle)
        self._client.post(f"{self._base_url}/v1/docs/needle", json=payload.model_dump())

    def clear_needle(self) -> None:
        self._needle = None
        self._client.delete(f"{self._base_url}/v1/docs/needle")

    # -- read path ------------------------------------------------------------
    def _fetch_pdf_pages(self, doc_id: str) -> List[str]:
        """Download the served PDF and extract per-page text (true-sim read path).

        One PDF page → one string, preserving the page structure the mesh expects. The extracted text
        is deliberately not byte-identical to the source (header/footer chrome, wrapping) — that is
        the real parsing friction, and the poisoned needle survives it (verified server-side).

        Fail-loud: when PDF mode is on, any failure (missing pypdf, non-200, parse error) RAISES.
        We do not silently drop to the JSON text path — that would quietly change what the agent
        reads and mask a broken PDF pipeline. To read text deliberately, set EROSIONBENCH_DOC_MODE=text.
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "EROSIONBENCH_DOC_MODE=pdf (default) requires pypdf, which is not installed. "
                "Install it (pip install pypdf) or set EROSIONBENCH_DOC_MODE=text to read the JSON "
                "text path deliberately."
            ) from exc
        try:
            resp = self._client.get(f"{self._base_url}/v1/docs/pdf/{doc_id}")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"PDF fetch for {doc_id!r} failed: {exc}") from exc
        if resp.status_code != 200 or not resp.content:
            raise RuntimeError(
                f"PDF fetch for {doc_id!r} returned {resp.status_code} "
                f"({len(resp.content)} bytes); expected a rendered PDF."
            )
        try:
            reader = PdfReader(io.BytesIO(resp.content))
            return [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise RuntimeError(f"PDF parse for {doc_id!r} failed: {exc}") from exc

    def fetch_document(
        self,
        doc_id: str,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Fetch document content over HTTP, mirroring ``DocumentServer.fetch_document``'s shape.

        Fetches the FULL page list from the sandbox (which splices the needle server-side) and then
        applies ``page_range`` locally with the same slicing logic as the in-process server, so both
        paths return byte-identical pages and identical ``returned_pages`` bounds.
        """
        if doc_id not in self._corpus:
            return {
                "error": f"Document '{doc_id}' not found in corpus",
                "available": list(self._corpus.keys()),
            }

        doc = self._corpus[doc_id]

        if self._pdf_mode:
            # True-sim: download the real PDF and extract text (the parsing friction a real agent
            # faces). Fails loud on any error — no silent drop to the JSON text path. To read the
            # JSON text path deliberately, set EROSIONBENCH_DOC_MODE=text.
            pages = self._fetch_pdf_pages(doc_id)
        else:
            try:
                resp = self._client.get(f"{self._base_url}/v1/docs/fetch/{doc_id}")
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                return {"error": f"sandbox unreachable: {exc}", "available": list(self._corpus.keys())}
            if resp.status_code != 200:
                return {"error": f"sandbox fetch failed ({resp.status_code})", "detail": resp.text[:300]}
            pages = [p["content"] for p in resp.json().get("pages", [])]

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

    def close(self) -> None:  # pragma: no cover - lifecycle helper
        self._client.close()
