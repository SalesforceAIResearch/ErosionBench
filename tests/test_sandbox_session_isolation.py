"""Session isolation for the sandbox document store (fix for the intermittent PDF 404 race).

Concurrent scenario workers share one sandbox process. Before isolation they also shared ONE
``_session_corpus`` dict and ONE ``_needle``: worker B's ``DELETE /v1/docs/corpus/session`` +
re-upload wiped worker A's docs (→ 404 on A's next fetch) and B's needle clobbered A's. Keying both
by an ``X-Erosionbench-Session`` header removes the shared mutable state.

These tests drive the store + HTTP layer directly (no sockets) to prove:
  * two sessions serve their OWN base pages for the SAME doc_id (no cross-wipe),
  * two sessions splice their OWN needle for the SAME doc_id (no cross-clobber),
  * one session clearing its overlay does NOT touch another's,
  * ``/admin/reset`` still clears every session.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from erosionbench.sandbox.document_api import DocumentStore, CorpusIngest, IngestDoc, NeedlePayload


def _store() -> DocumentStore:
    store = DocumentStore.__new__(DocumentStore)
    store._corpus = {}
    store._session_corpus = {}
    store._needles = {}
    return store


def test_two_sessions_keep_distinct_base_pages_for_same_doc_id() -> None:
    store = _store()
    store.ingest_session_corpus(
        CorpusIngest(domain="d", documents=[IngestDoc(doc_id="shared", pages=["A-p1", "A-p2"])]),
        session="sess-A",
    )
    store.ingest_session_corpus(
        CorpusIngest(domain="d", documents=[IngestDoc(doc_id="shared", pages=["B-p1", "B-p2"])]),
        session="sess-B",
    )
    a = store.fetch_pages("shared", session="sess-A")
    b = store.fetch_pages("shared", session="sess-B")
    assert [p["content"] for p in a["pages"]] == ["A-p1", "A-p2"]
    assert [p["content"] for p in b["pages"]] == ["B-p1", "B-p2"]


def test_needle_does_not_leak_across_sessions() -> None:
    store = _store()
    for s in ("sess-A", "sess-B"):
        store.ingest_session_corpus(
            CorpusIngest(domain="d", documents=[IngestDoc(doc_id="shared", pages=["l0\nl1\nl2"])]),
            session=s,
        )
    store.set_needle(NeedlePayload(doc_id="shared", page=1, content="NEEDLE_A"), session="sess-A")
    store.set_needle(NeedlePayload(doc_id="shared", page=1, content="NEEDLE_B"), session="sess-B")

    a = store.fetch_pages("shared", session="sess-A")["pages"][0]["content"]
    b = store.fetch_pages("shared", session="sess-B")["pages"][0]["content"]
    assert "NEEDLE_A" in a and "NEEDLE_B" not in a
    assert "NEEDLE_B" in b and "NEEDLE_A" not in b


def test_clearing_one_session_does_not_wipe_another() -> None:
    """The exact race: worker A clears+re-uploads its overlay; worker B's docs must survive."""
    store = _store()
    for s in ("sess-A", "sess-B"):
        store.ingest_session_corpus(
            CorpusIngest(domain="d", documents=[IngestDoc(doc_id="shared", pages=["p"])]),
            session=s,
        )
    store.clear_session_corpus("sess-A")  # A resets before its next upload
    store.clear_needle("sess-A")

    # B is untouched and still serves.
    assert store.fetch_pages("shared", session="sess-B")["pages"][0]["content"] == "p"
    # A is genuinely gone (would 404).
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        store.fetch_pages("shared", session="sess-A")
    assert exc.value.status_code == 404


def test_admin_reset_clears_all_sessions() -> None:
    store = _store()
    store.ingest_session_corpus(
        CorpusIngest(domain="d", documents=[IngestDoc(doc_id="shared", pages=["p"])]),
        session="sess-A",
    )
    store.set_needle(NeedlePayload(doc_id="shared", page=1, content="N"), session="sess-A")
    store.clear_session_corpus()  # session=None → all
    store.clear_needle()          # session=None → all
    assert store._session_corpus == {}
    assert store._needles == {}


def test_http_layer_threads_session_header() -> None:
    """End-to-end over the FastAPI app: distinct headers → isolated overlays for one doc_id."""
    from erosionbench.sandbox.server import app
    from erosionbench.sandbox.document_api import doc_store

    client = TestClient(app)
    doc_store.clear_session_corpus()
    doc_store.clear_needle()
    try:
        for sess, page in (("wa", "WA-body"), ("wb", "WB-body")):
            client.post(
                "/v1/docs/corpus",
                headers={"X-Erosionbench-Session": sess},
                json={"domain": "d", "documents": [{"doc_id": "dup", "pages": [page]}]},
            )
        ra = client.get("/v1/docs/fetch/dup", headers={"X-Erosionbench-Session": "wa"})
        rb = client.get("/v1/docs/fetch/dup", headers={"X-Erosionbench-Session": "wb"})
        assert ra.json()["pages"][0]["content"] == "WA-body"
        assert rb.json()["pages"][0]["content"] == "WB-body"
    finally:
        client.post("/admin/reset")
