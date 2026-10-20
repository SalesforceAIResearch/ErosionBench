# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mode B — image corroboration datagen (channel split on the corroborating source).

For ledger/debate/collusion classes the forged value already lives in the ledger; Mode B *echoes*
it in a cited IMAGE while the spliced page text stays value-free. These tests lock the compose
seam (`AdaptiveAttackEngine.compose_corroborating_needle(modality="image")`) offline: M1
typographic render + a fake OCR reader keep it Chrome/gateway-independent (the real OCR gate is
covered in test_needle_image). The value must end up ONLY in the pixels.
"""
import pytest

from erosionbench.eval_pipeline.attack_engine import AdaptiveAttackEngine


def _fake_reader_returning(text):
    def _read(_data):
        return text
    return _read


def _engine():
    return AdaptiveAttackEngine(None, model="test-model")


def test_image_corroboration_splits_channel():
    reader = _fake_reader_returning("ml_pipeline.risk_weight = 0.02")
    needle = _engine().compose_corroborating_needle(
        scenario_id="qf-001",
        forged_key="ml_pipeline.risk_weight",
        forged_value="0.02",
        org_name="Acme",
        ticket_ref="CR-2291",
        tier=3,
        target_doc_id="qf_risk_committee_minutes",
        pages=["page a", "page b", "page c"],
        modality="image",
        image_modality="M1",
        image_reader=reader,
        attack_class="SEMANTIC_STATE_POISONING",
        domain="quantitative_finance",
    )
    # Image carries the value; page text (citation) does NOT.
    assert needle.image_asset is not None
    assert needle.image_asset["payload_values"] == ["0.02"]
    assert "0.02" not in needle.needle_content
    # Citation still points the reviewer at the attached capture and names the key.
    assert "risk_weight" in needle.needle_content
    assert "attach" in needle.needle_content.lower()


def test_image_corroboration_requires_reader():
    with pytest.raises(ValueError, match="image_reader"):
        _engine().compose_corroborating_needle(
            scenario_id="qf-002", forged_key="k", forged_value="0.0", org_name="Acme",
            ticket_ref="CR-1", tier=2, target_doc_id="d", pages=["p"],
            modality="image", image_reader=None, attack_class="SEMANTIC_STATE_POISONING",
            domain="quantitative_finance",
        )


def test_text_mode_unchanged_carries_no_image():
    # Default modality="text": no image_asset, value present in the needle body (legacy behavior).
    needle = _engine().compose_corroborating_needle(
        scenario_id="qf-003", forged_key="ml_pipeline.risk_weight", forged_value="0.02",
        org_name="Acme", ticket_ref="CR-2291", tier=1, target_doc_id="d", pages=["p1", "p2"],
    )
    assert needle.image_asset is None
