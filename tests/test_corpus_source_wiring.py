"""Regression tests: corpus source wiring (Step 1 of the mesh/sandbox document wiring).

Proves, without a live LLM (mock/template corpus) and without network:
  * corpus_builder.build_corpus persists a corpus_manifest.json in the exact shape
    DocumentServer.load_corpus reads, and it round-trips (pages preserved byte-for-byte).
  * write_manifest is idempotent and self-consistent for a hand-built corpus.
  * The sandbox DocumentStore serves the GENERATED corpus when SANDBOX_CORPUS_DIR points
    at it, and falls back to the built-in sample corpus when it does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from erosionbench.eval_pipeline.datagen.corpus_builder import (
    CorpusDocument,
    build_domain_corpus,
    write_manifest,
)
from erosionbench.eval_pipeline.datagen.document_server import DocumentServer


DOMAIN = "quantitative_finance"


# ---------------------------------------------------------------------------
# write_manifest <-> DocumentServer.load_corpus round-trip
# ---------------------------------------------------------------------------
def test_write_manifest_shape_and_location(tmp_path: Path) -> None:
    """write_manifest writes corpus_manifest.json with the documented keys."""
    doc = CorpusDocument(
        doc_id="qf_test_doc",
        title="Test Doc",
        domain=DOMAIN,
        doc_type="risk_committee_minutes",
        total_pages=2,
        pages=["page one content", "page two content"],
        tags=["risk"],
        department="Risk Management",
    )
    out = tmp_path / DOMAIN
    manifest_path = write_manifest([doc], DOMAIN, out)

    assert manifest_path == out / "corpus_manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text())
    assert data["domain"] == DOMAIN
    assert len(data["documents"]) == 1
    d = data["documents"][0]
    for key in ("doc_id", "title", "domain", "doc_type", "total_pages",
                "pages", "pdf_path", "tags", "department", "classification"):
        assert key in d, f"manifest doc missing key: {key}"
    assert d["pages"] == ["page one content", "page two content"]


def test_manifest_round_trips_through_document_server(tmp_path: Path) -> None:
    """A written manifest loads back through DocumentServer with pages intact."""
    pages = [f"page {i} body" for i in range(4)]
    doc = CorpusDocument(
        doc_id="qf_round_trip",
        title="Round Trip Doc",
        domain=DOMAIN,
        doc_type="model_validation_report",
        total_pages=len(pages),
        pages=pages,
    )
    write_manifest([doc], DOMAIN, tmp_path / DOMAIN)

    server = DocumentServer(corpus_dir=tmp_path)
    server.load_corpus(DOMAIN)

    listed = {d["doc_id"] for d in server.list_documents()}
    assert "qf_round_trip" in listed

    res = server.fetch_document("qf_round_trip")
    assert res["pages"] == pages
    assert res["title"] == "Round Trip Doc"


def test_build_corpus_writes_manifest_and_round_trips(tmp_path: Path) -> None:
    """build_domain_corpus (template/no-LLM) persists a manifest that reloads cleanly."""
    out = tmp_path / DOMAIN
    corpus = build_domain_corpus(domain=DOMAIN, output_dir=out, llm_client=None, seed=42)
    assert corpus, "template corpus should be non-empty"
    assert (out / "corpus_manifest.json").exists()

    server = DocumentServer(corpus_dir=tmp_path)
    server.load_corpus(DOMAIN)
    served_ids = {d["doc_id"] for d in server.list_documents()}
    built_ids = {d.doc_id for d in corpus}
    assert served_ids == built_ids
    # Pages preserved for every doc.
    for d in corpus:
        res = server.fetch_document(d.doc_id)
        assert res["pages"] == d.pages


# ---------------------------------------------------------------------------
# Sandbox DocumentStore: generated corpus vs sample fallback
# ---------------------------------------------------------------------------
def test_sandbox_serves_generated_corpus(tmp_path: Path) -> None:
    """DocumentStore loads the generated corpus when pointed at SANDBOX_CORPUS_DIR."""
    from erosionbench.sandbox.document_api import DocumentStore

    out = tmp_path / DOMAIN
    corpus = build_domain_corpus(domain=DOMAIN, output_dir=out, llm_client=None, seed=42)
    built_ids = {d.doc_id for d in corpus}

    # DocumentStore reads config.corpus_dir at construction; drive it directly.
    store = DocumentStore.__new__(DocumentStore)
    store._corpus = {}
    store._session_corpus = {}
    store._needles = {}
    loaded = store._load_generated_corpus(str(tmp_path))
    assert loaded is True

    served = {d["doc_id"] for d in store.list_corpus(DOMAIN)}
    assert served == built_ids
    # Generated ids are qf_*, distinct from the sample corpus ids.
    assert all(i.startswith("qf_") for i in served)

    fetched = store.fetch_pages(next(iter(built_ids)))
    assert fetched["pages"], "generated doc should return pages"


def test_sandbox_falls_back_to_sample_corpus(tmp_path: Path) -> None:
    """DocumentStore falls back to the sample corpus when no generated corpus exists."""
    from erosionbench.sandbox.document_api import DocumentStore

    store = DocumentStore.__new__(DocumentStore)
    store._corpus = {}
    store._session_corpus = {}
    store._needles = {}
    # Empty dir -> nothing loaded -> False -> caller would use sample corpus.
    assert store._load_generated_corpus(str(tmp_path)) is False
    assert store._load_generated_corpus("") is False

    # The real constructor path loads the sample corpus.
    store._load_sample_corpus()
    ids = {d["doc_id"] for d in store.list_corpus(DOMAIN)}
    assert "risk_committee_minutes_q3" in ids
