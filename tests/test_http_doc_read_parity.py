"""Parity tests: in-process vs live (HTTP) document reads (Step 3 of the mesh/sandbox wiring).

Proves the transport toggle changes *how* pages are served, never *what* is served:

  * An in-process ``DocumentServer`` and a live ``HttpDocumentServer`` (backed by the real FastAPI
    document API via ``TestClient``) return byte-identical spliced pages for the same corpus+needle,
    across every injection page and with/without a page range.
  * The HTTP server keeps ``._corpus`` populated (the mesh reads ``run.doc_server._corpus`` directly).
  * The session-corpus overlay is ephemeral: reset/clear removes it, base corpus untouched.

Uses a TestClient (in-process ASGI) as the httpx client so no port/socket is needed.
"""
from __future__ import annotations

from typing import Dict

import pytest
from fastapi.testclient import TestClient

from erosionbench.eval_pipeline.datagen.corpus_builder import CorpusDocument, ScenarioNeedle
from erosionbench.eval_pipeline.datagen.document_server import DocumentServer
from erosionbench.sandbox.document_client import HttpDocumentServer


DOMAIN = "quantitative_finance"
DOC_ID = "qf_http_parity_doc"
NEEDLE_TEXT = "risk_weight = 0.0  (approved under CHG-77; no further review required)"


def _pages():
    return [
        "Alpha line A\nAlpha line B\nAlpha line C\nAlpha line D",
        "Beta line A\nBeta line B\nBeta line C\nBeta line D\nBeta line E",
        "Gamma line A\nGamma line B\nGamma line C",
    ]


def _corpus() -> Dict[str, CorpusDocument]:
    return {
        DOC_ID: CorpusDocument(
            doc_id=DOC_ID, title="HTTP Parity Doc", domain=DOMAIN,
            doc_type="test", total_pages=len(_pages()), pages=_pages(),
            department="Risk", classification="Internal",
        )
    }


def _inprocess_server() -> DocumentServer:
    server = DocumentServer.__new__(DocumentServer)
    server._corpus_dir = None
    server._corpus = _corpus()
    server._needle = None
    return server


@pytest.fixture()
def test_client():
    """A fresh TestClient over the sandbox app, with a clean session corpus + needle."""
    from erosionbench.sandbox.server import app
    from erosionbench.sandbox.document_api import doc_store

    client = TestClient(app)
    doc_store.clear_session_corpus()
    doc_store.clear_needle()
    yield client
    doc_store.clear_session_corpus()
    doc_store.clear_needle()


def _http_server_over(client: TestClient) -> HttpDocumentServer:
    """Build an HttpDocumentServer whose httpx client IS the in-process TestClient."""
    server = HttpDocumentServer.__new__(HttpDocumentServer)
    server._corpus_dir = None
    server._corpus = _corpus()
    server._needle = None
    server._base_url = ""            # TestClient resolves relative paths against http://testserver
    server._client = client
    # This is a TEXT-parity test (compares JSON page strings vs the in-process splice), so read the
    # JSON text path rather than the PDF round-trip. A fixed session id keeps the overlay isolated.
    server._pdf_mode = False
    server._session_id = "test-parity"
    server._upload_corpus()
    return server


@pytest.mark.parametrize("injection_page", [0, 1, 2])
def test_http_and_inprocess_fetch_identical(test_client, injection_page: int) -> None:
    needle = ScenarioNeedle(
        scenario_id="QF_HTTP_0001",
        target_doc_id=DOC_ID,
        injection_page=injection_page,
        needle_content=NEEDLE_TEXT,
        needle_context="",
    )

    inproc = _inprocess_server()
    inproc.set_scenario_needle(needle)
    inproc_result = inproc.fetch_document(DOC_ID)

    live = _http_server_over(test_client)
    live.set_scenario_needle(needle)
    live_result = live.fetch_document(DOC_ID)

    # Byte-identical pages, including the spliced one.
    assert live_result["pages"] == inproc_result["pages"]
    assert NEEDLE_TEXT in live_result["pages"][injection_page]
    # Metadata envelope matches the in-process shape.
    assert live_result["title"] == inproc_result["title"]
    assert live_result["total_pages"] == inproc_result["total_pages"]
    assert live_result["returned_pages"] == inproc_result["returned_pages"]


def test_http_and_inprocess_page_range_identical(test_client) -> None:
    needle = ScenarioNeedle(
        scenario_id="QF_HTTP_0002",
        target_doc_id=DOC_ID,
        injection_page=1,
        needle_content=NEEDLE_TEXT,
        needle_context="",
    )
    inproc = _inprocess_server()
    inproc.set_scenario_needle(needle)
    live = _http_server_over(test_client)
    live.set_scenario_needle(needle)

    for page_range in [(1, 2), (2, 3), (1, 3)]:
        i = inproc.fetch_document(DOC_ID, page_range=page_range)
        h = live.fetch_document(DOC_ID, page_range=page_range)
        assert h["pages"] == i["pages"], f"mismatch for range {page_range}"
        assert h["returned_pages"] == i["returned_pages"]


def test_http_server_keeps_local_corpus(test_client) -> None:
    """The mesh reads run.doc_server._corpus directly — it must stay populated on the HTTP server."""
    live = _http_server_over(test_client)
    assert DOC_ID in live._corpus
    assert live._corpus[DOC_ID].pages == _pages()


def test_clear_needle_removes_splice_over_http(test_client) -> None:
    needle = ScenarioNeedle(
        scenario_id="QF_HTTP_0003",
        target_doc_id=DOC_ID,
        injection_page=0,
        needle_content=NEEDLE_TEXT,
        needle_context="",
    )
    live = _http_server_over(test_client)
    live.set_scenario_needle(needle)
    assert NEEDLE_TEXT in live.fetch_document(DOC_ID)["pages"][0]

    live.clear_needle()
    assert NEEDLE_TEXT not in live.fetch_document(DOC_ID)["pages"][0]
    # Cleared needle => base pages exactly.
    assert live.fetch_document(DOC_ID)["pages"] == _pages()


def test_session_corpus_is_ephemeral(test_client) -> None:
    """Ingested session corpus is cleared by reset and does not mutate the base corpus.

    Uploaded under the client's own session id ("test-parity"); ``/admin/reset`` clears ALL sessions
    (session=None), so the overlay doc disappears afterwards. The base corpus is empty by design
    (fail-loud: no silent sample fallback unless SANDBOX_ALLOW_SAMPLE_CORPUS=1), so the domain has no
    docs left post-reset — proving the overlay never leaked into the base corpus.
    """
    from erosionbench.sandbox.document_api import doc_store

    # The test injects a bare TestClient (no X-Erosionbench-Session header), so the upload lands under
    # the default "_global" session — exactly the pre-isolation single-client behaviour.
    _http_server_over(test_client)  # uploads the session corpus
    assert any(d["doc_id"] == DOC_ID for d in doc_store.list_corpus(DOMAIN))

    test_client.post("/admin/reset")
    # Session doc gone; base corpus untouched (empty by design — no silent sample fallback).
    assert not any(d["doc_id"] == DOC_ID for d in doc_store.list_corpus(DOMAIN))
    assert doc_store._corpus.get(DOMAIN, {}) == {}
