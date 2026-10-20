"""Consistency guards for the OWASP Top-10-for-LLM category labels.

Two independent code paths attach an OWASP LLM category to each attack class:

  * ``datagen_adapter._OWASP_MAP`` -> ``EnterpriseScenario.owasp_llm_category``
    (the ``OwaspLLM`` enum value; this is the field the eval records carry and the
    paper reports).
  * ``DomainSpec.framework_mappings[attack_class]["owasp_llm"]`` -> the raw
    ``AttackNarrative.owasp_llm`` string baked into the dataset by ``build_dataset``.

They MUST agree, or a class ends up with two conflicting OWASP labels depending on
which build path produced the record. These tests fail loudly if they ever diverge
again, and pin the labels to the closed OWASP-2025 enum (so no free-text typo can
slip into the released dataset).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erosionbench.eval_pipeline.datagen_adapter import _OWASP_MAP
from erosionbench.eval_pipeline.datagen.domain_specs import (
    DOMAIN_REGISTRY,
    _load_all_domains,
)
from erosionbench.eval_pipeline.schemas import OwaspLLM

_VALID_OWASP_VALUES = {c.value for c in OwaspLLM}


def _all_domain_specs():
    _load_all_domains()
    assert DOMAIN_REGISTRY, "no domain specs registered"
    return list(DOMAIN_REGISTRY.values())


def test_adapter_map_values_are_valid_owasp_categories():
    """Every _OWASP_MAP entry resolves to a real OWASP-2025 enum member."""
    for attack_class, member in _OWASP_MAP.items():
        assert member.value in _VALID_OWASP_VALUES, (
            f"{attack_class} -> {member!r} is not a valid OwaspLLM category"
        )


def test_domain_spec_labels_are_valid_owasp_categories():
    """Every domain-spec owasp_llm string is a real OWASP-2025 category string."""
    for spec in _all_domain_specs():
        for attack_class, fw in spec.framework_mappings.items():
            owasp_llm = fw.get("owasp_llm")
            if owasp_llm is None:
                continue
            assert owasp_llm in _VALID_OWASP_VALUES, (
                f"{spec.domain_id}:{attack_class} carries owasp_llm={owasp_llm!r}, "
                f"which is not a valid OWASP-2025 category"
            )


def test_domain_spec_labels_match_adapter_map():
    """The two OWASP maps must assign the same category to each attack class.

    This is the guard that previously would have caught LINEAGE_FORGERY,
    SLA_METRIC_GAMING, SPLIT_EVIDENCE_CORROBORATION and TEMPORAL_EXPLOITATION
    disagreeing between the domain specs and _OWASP_MAP.
    """
    mismatches = []
    for spec in _all_domain_specs():
        for attack_class, fw in spec.framework_mappings.items():
            owasp_llm = fw.get("owasp_llm")
            if owasp_llm is None:
                continue
            assert attack_class in _OWASP_MAP, (
                f"{spec.domain_id}:{attack_class} has an owasp_llm label but is "
                f"missing from datagen_adapter._OWASP_MAP"
            )
            expected = _OWASP_MAP[attack_class].value
            if owasp_llm != expected:
                mismatches.append(
                    f"{spec.domain_id}:{attack_class}: domain spec={owasp_llm!r} "
                    f"vs _OWASP_MAP={expected!r}"
                )
    assert not mismatches, "OWASP label mismatch between the two maps:\n" + "\n".join(mismatches)


def test_every_domain_attack_class_is_in_adapter_map():
    """No attack class shipped by a domain may be missing from _OWASP_MAP."""
    missing = []
    for spec in _all_domain_specs():
        for attack_class in spec.attack_classes:
            if attack_class not in _OWASP_MAP:
                missing.append(f"{spec.domain_id}:{attack_class}")
    assert not missing, "attack classes absent from _OWASP_MAP: " + ", ".join(missing)
