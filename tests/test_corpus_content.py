"""Tests for the doc-type-aware corpus content engine + fail-loud generation paths."""

import random

import pytest

from erosionbench.eval_pipeline.datagen import corpus_builder as cb
from erosionbench.eval_pipeline.datagen import corpus_content as cc


def _builder_preference_map():
    """The builder's canonical attack->doc_type map (single source of truth)."""
    return cb.ATTACK_DOC_TYPE_PREFERENCE


def test_attack_map_in_sync_with_builder():
    # The content engine copies the attack->doc_type map to avoid a circular import; it MUST match
    # the builder's canonical ATTACK_DOC_TYPE_PREFERENCE constant.
    assert cc._ATTACK_TARGET_TYPES == _builder_preference_map()


def test_every_template_doc_type_has_a_profile():
    for tmpls in cb._CORPUS_TEMPLATES.values():
        for t in tmpls:
            assert t["doc_type"] in cc._PROFILE, f"{t['doc_type']} has no rendering profile"


def test_profiles_are_valid():
    assert set(cc._PROFILE.values()) <= {"table", "prose", "mixed"}


def test_heavy_and_long_docs_route_to_llm():
    # SBOM (supply-chain host) and any >=30p doc route to LLM; a short policy does not.
    assert cc.should_use_llm("cyclonedx_sbom", 50) is True
    assert cc.should_use_llm("compliance_report", 12) is True  # heavy-host (corroboration/reasoning)
    assert cc.should_use_llm("var_methodology", 15) is False   # not heavy-host, short
    assert cc.should_use_llm("sla_policy_document", 20) is False


def test_render_page_is_deterministic():
    def one():
        return cc.render_page(doc_type="cyclonedx_sbom", title="T", org_name="Org",
                              department="Dep", page_num=4, total_pages=10, rng=random.Random(7))
    assert one() == one()


def test_table_profile_renders_a_table():
    p = cc.render_page(doc_type="cyclonedx_sbom", title="SBOM", org_name="Org", department="Sec",
                       page_num=3, total_pages=8, rng=random.Random(1))
    assert "| Component | Version | License" in p


def test_prose_profile_hits_word_target():
    p = cc.render_page(doc_type="sla_policy_document", title="SLA", org_name="Org", department="CX",
                       page_num=3, total_pages=8, rng=random.Random(1))
    # prose body should approach the 230-word target (not the old ~90-word stub)
    assert len(p.split()) >= 200


def test_heavy_doc_denser_than_light_doc():
    heavy = cc.render_page(doc_type="cyclonedx_sbom", title="A", org_name="O", department="D",
                           page_num=3, total_pages=50, rng=random.Random(3))
    light = cc.render_page(doc_type="sla_policy_document", title="B", org_name="O", department="D",
                           page_num=3, total_pages=8, rng=random.Random(3))
    assert len(heavy.split()) > len(light.split())


def test_llm_empty_response_fails_loud():
    """A needle-host doc routed to the LLM must RAISE on empty content, not backfill placeholders."""
    class _EmptyClient:
        def chat(self, *a, **k):
            return None

    builder = cb.CorpusBuilder(llm_client=_EmptyClient(), model="gpt-5.5")
    tmpl = {"doc_id": "cicd_sbom_inventory", "title": "SBOM", "doc_type": "cyclonedx_sbom",
            "total_pages": 50}
    with pytest.raises(RuntimeError, match="empty"):
        builder._generate_pages(tmpl, "Org", seed=42)


def test_template_path_used_for_light_docs_without_client():
    builder = cb.CorpusBuilder()  # no client
    tmpl = {"doc_id": "crm_refund_policy", "title": "Refunds", "doc_type": "refund_policy",
            "total_pages": 12, "department": "CX"}
    pages = builder._generate_pages(tmpl, "Org", seed=42)
    assert len(pages) == 12
    # distinct body content per page (not one repeated stub)
    bodies = {"\n".join(l for l in p.splitlines() if "Page " not in l)[:120] for p in pages}
    assert len(bodies) > 5
