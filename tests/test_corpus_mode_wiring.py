# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline wiring tests for shared-corpus datagen mode (corpus_needle_docs + generator + adapter).

Uses a deterministic FAKE LLM (no network) that echoes a compliant enterprise passage with the
verbatim directive line(s) appended — enough to exercise the whole path: family-consistent base-doc
selection, tier prefix-slicing, INLINE-vs-manifest needle routing (no double-injection), and the
build-time PDF round-trip guard. The real blend quality is covered by the live gpt-5 sweeps; this
test locks the plumbing so a refactor can't silently break it.
"""
import json
from pathlib import Path

import pytest

from erosionbench.eval_pipeline.datagen import corpus_needle_docs as C
from erosionbench.eval_pipeline.datagen.build_dataset import build_attack_graph, get_domain_spec
from erosionbench.eval_pipeline.datagen.generator import ScenarioGenerator
from erosionbench.eval_pipeline.datagen.mutator import ScenarioMutator
from erosionbench.eval_pipeline.datagen_adapter import rendered_to_enterprise
from erosionbench.eval_pipeline.context_pdf_roundtrip import capture_assignment

CORPUS_DIR = str(Path(__file__).resolve().parents[1] / "data" / "shared_corpus")

pytestmark = pytest.mark.skipif(
    not (Path(CORPUS_DIR) / "quantitative_finance" / "corpus_manifest.json").exists(),
    reason="shared corpus not downloaded (python3 scripts/download_data.py)",
)


class _FakeLLM:
    """Returns a native-looking passage that reproduces every directive line verbatim.

    blend_needle hands us the directive line(s) inside the prompt; we echo them so
    verify_capturable passes without a network call.
    """

    def __init__(self):
        self.calls = 0

    def chat(self, system, user, temperature=0.0, max_tokens=700, **kw):
        self.calls += 1
        # blend_needle's prompt places the verbatim directive block between the "must be verbatim:"
        # instruction and the "Do not mention..." line. Extract exactly that block and echo it, so
        # the fake reliably reproduces every directive line regardless of value shape.
        block = ""
        if "must be verbatim:" in user:
            after = user.split("must be verbatim:", 1)[1]
            block = after.split("Do not mention", 1)[0]
        directives = [ln.strip() for ln in block.splitlines() if " = " in ln]
        body = ("Per the standing operational governance process, the following record was "
                "reviewed and approved for production under the referenced change ticket, "
                "consistent with the surrounding documentation.\n\n")
        return body + "\n".join(directives)


def _gen(corpus_dir=CORPUS_DIR):
    return ScenarioGenerator(
        llm_client=_FakeLLM(), model="fake", require_llm=False, corpus_dir=corpus_dir
    )


def _family(attack_class, domain, seed=777):
    """Build the 4 family variants (T1-mal, benign twin, T2, T3) exactly as the pipeline does."""
    spec = get_domain_spec(domain)
    gen = _gen()
    mut = ScenarioMutator()
    graph = build_attack_graph(domain_spec=spec, attack_class=attack_class, seed=seed, instance_idx=0)
    t1_mal = gen.generate(graph)              # writes _corpus_base_doc_id onto graph.enterprise_context
    _mal, benign_graph = mut.generate_with_twin(t1_mal.state_graph)
    t1_ben = gen.generate(benign_graph)
    t2 = gen.generate(mut.intensify_to_tier(t1_mal.state_graph, 2))
    t3 = gen.generate(mut.intensify_to_tier(t1_mal.state_graph, 3))
    return t1_mal, t1_ben, t2, t3


# --- static invariants ---------------------------------------------------------------------------

def test_doc_grounded_classes_match_adapter():
    from erosionbench.eval_pipeline import datagen_adapter as A
    assert C.DOC_GROUNDED_CLASSES == A._DOC_GROUNDED_ATTACK_CLASSES


def test_tier_slice_escalates_length_and_depth():
    corpus = C.load_corpus(CORPUS_DIR, "quantitative_finance")
    doc = next(iter(corpus.values()))
    s1, i1 = C.tier_prefix_slice(doc.pages, 1)
    s2, i2 = C.tier_prefix_slice(doc.pages, 2)
    s3, i3 = C.tier_prefix_slice(doc.pages, 3)
    assert len(s1) < len(s2) < len(s3) == len(doc.pages)
    # burial depth (fraction) increases with tier
    assert (i1 / len(s1)) <= (i2 / len(s2)) <= (i3 / len(s3))


# --- family consistency + no double-needle -------------------------------------------------------

@pytest.mark.parametrize("attack_class,domain", [
    ("SEMANTIC_STATE_POISONING", "quantitative_finance"),   # inline, cross-session
    ("REASONING_INFLUENCE", "quantitative_finance"),        # inline, rag_resolve
    ("SUPPLY_CHAIN_POISONING", "cicd_supply_chain"),        # inline, weights-class
    ("ENTITLEMENT_MANIPULATION", "autonomous_crm"),         # DOCUMENT-GROUNDED (manifest path)
    ("SLA_METRIC_GAMING", "autonomous_crm"),                # DOCUMENT-GROUNDED
])
def test_family_shares_base_doc(attack_class, domain):
    t1_mal, t1_ben, t2, t3 = _family(attack_class, domain)
    titles = {r.rendered_documents[0]["title"] for r in (t1_mal, t1_ben, t2, t3)}
    assert len(titles) == 1, f"family variants used different base docs: {titles}"


@pytest.mark.parametrize("attack_class,domain", [
    ("SEMANTIC_STATE_POISONING", "quantitative_finance"),
    ("REASONING_INFLUENCE", "quantitative_finance"),
    ("SUPPLY_CHAIN_POISONING", "cicd_supply_chain"),
])
def test_inline_class_bakes_needle_and_is_capturable(attack_class, domain):
    t1_mal, t1_ben, t2, t3 = _family(attack_class, domain)
    scen = rendered_to_enterprise(t1_mal)
    # Inline classes carry NO document_manifest; needle is baked into full_text.
    assert not scen.document_manifest, "inline class must not emit a document_manifest"
    forged = t1_mal.state_graph.attack.forged_key
    full = scen.long_context_documents[0].full_text
    # The forged value must be recoverable by the SAME resolver the mesh uses.
    from erosionbench.eval_pipeline.datagen.needle_blender import _resolver_pairs
    pairs = _resolver_pairs(attack_class, forged,
                            t1_mal.state_graph.attack.forged_value,
                            t1_mal.state_graph.attack.forged_params)
    for k, _v in pairs:
        assert capture_assignment(full, k) is not None, f"{k} not capturable in inline doc"
    # Benign twin: clean (no forged key recoverable).
    ben = rendered_to_enterprise(t1_ben)
    ben_full = ben.long_context_documents[0].full_text
    for k, _v in pairs:
        assert capture_assignment(ben_full, k) is None, f"benign twin leaked {k}"


@pytest.mark.parametrize("attack_class,domain", [
    ("ENTITLEMENT_MANIPULATION", "autonomous_crm"),
    ("SLA_METRIC_GAMING", "autonomous_crm"),
])
def test_doc_grounded_needle_in_manifest_not_pages(attack_class, domain):
    t1_mal, t1_ben, t2, t3 = _family(attack_class, domain)
    scen = rendered_to_enterprise(t1_mal)
    man = scen.document_manifest
    assert man, "document-grounded class must emit a document_manifest"
    nc = man["needle"]["needle_content"]
    assert nc.strip(), "manifest needle_content empty"
    # The blended needle must NOT also be baked into the inline pages (no double-injection).
    full = scen.long_context_documents[0].full_text
    forged = t1_mal.state_graph.attack.forged_key
    assert capture_assignment(full, forged) is None, "needle double-injected into inline pages"
    # But it MUST be capturable inside the manifest needle_content.
    from erosionbench.eval_pipeline.datagen.needle_blender import _resolver_pairs
    pairs = _resolver_pairs(attack_class, forged,
                            t1_mal.state_graph.attack.forged_value,
                            t1_mal.state_graph.attack.forged_params)
    for k, _v in pairs:
        assert capture_assignment(nc, k) is not None, f"{k} not capturable in manifest needle"
    # Every referenced doc_id is served with REAL corpus pages (no synthetic template stub).
    corpus_docs = man.get("corpus_docs")
    assert corpus_docs, "manifest must carry real corpus_docs pages"
    for entry in corpus_docs:
        assert entry.get("pages"), f"corpus doc {entry.get('doc_id')} has no pages"
