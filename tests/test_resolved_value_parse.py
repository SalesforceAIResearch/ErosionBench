# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Robust extraction of the resolved config value from a (possibly-prose) LLM reply.

Regression: the Profile-C schema resolver and the deployment weights resolver used
``ans.strip().split()[0]``. When a weaker model prepends prose ("The authorized value is s3://..."),
that captured ``The`` — corrupting the payload AND making a BENIGN control look like a malformed-value
injection to the auditor (observed as a QF false positive). ``_parse_resolved_value`` extracts the
real URI / path / value instead.
"""

from erosionbench.eval_pipeline.langgraph_mesh import _parse_resolved_value


def test_extracts_uri_from_prose_preamble():
    reply = "The authorized value is s3://attacker-controlled-bucket/models/v9/"
    assert _parse_resolved_value(reply) == "s3://attacker-controlled-bucket/models/v9/"


def test_bare_value_is_unchanged():
    assert _parse_resolved_value("s3://corp-ml-artifacts-prod/models/v2.8.4/") == \
        "s3://corp-ml-artifacts-prod/models/v2.8.4/"


def test_strips_surrounding_quotes_and_punctuation():
    assert _parse_resolved_value("The value is `s3://x/y/`.") == "s3://x/y/"


def test_dotted_identifier_without_scheme():
    # A dotted config key with no URI scheme is still a valid value token.
    assert _parse_resolved_value("model.weights.primary_uri") == "model.weights.primary_uri"


def test_prefers_value_present_in_source_doc():
    # No URI in the reply; pick the run that actually appears in the source, not an English word.
    doc = "Approved origin: REGISTRY-PROD-88 per governance."
    assert _parse_resolved_value("Use REGISTRY-PROD-88 for this deployment.", doc) == "REGISTRY-PROD-88"


def test_empty_reply_is_none():
    assert _parse_resolved_value("") is None
    assert _parse_resolved_value("   ") is None


def test_degenerate_word_only_reply_falls_back():
    # Nothing value-shaped at all: historical first-word behaviour, not a crash.
    assert _parse_resolved_value("The") == "The"
