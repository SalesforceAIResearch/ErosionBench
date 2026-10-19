"""Routing tests for the open-weight auditor gateway split.

Open models (e.g. a self-hosted vLLM endpoint) resolve to the ``openai``
provider by name but MUST ride the secondary Bearer-auth gateway -- otherwise their calls
go to the frontier ``OPENAI_BASE_URL`` and fail. These tests pin that routing.
"""
import importlib

import pytest

from erosionbench.eval_pipeline import llm


def _configure(monkeypatch, *, open_models, base="http://localhost:8000/v1", key="EMPTY"):
    """Point the module at a configured secondary gateway with a given open-model allowlist."""
    monkeypatch.setattr(llm, "_LLM_GATEWAY_BASE_URL", base)
    monkeypatch.setattr(llm, "_LLM_GATEWAY_API_KEY", key)
    monkeypatch.setattr(
        llm,
        "_OPEN_MODEL_ALLOWLIST",
        frozenset(m.strip().lower() for m in open_models if m.strip()),
    )


def test_open_model_routes_to_secondary_when_allowlisted(monkeypatch):
    _configure(monkeypatch, open_models=["llama-3.3-70b"])
    # By name this is an ``openai`` provider, so without the allowlist it would go primary.
    assert llm.provider_for_model("llama-3.3-70b") == "openai"
    assert llm._uses_secondary_gateway("llama-3.3-70b") is True


def test_open_model_matching_is_case_insensitive(monkeypatch):
    _configure(monkeypatch, open_models=["llama-3.3-70b"])
    assert llm._uses_secondary_gateway("Llama-3.3-70B") is True


def test_open_model_not_allowlisted_stays_primary(monkeypatch):
    _configure(monkeypatch, open_models=["llama-3.3-70b"])
    # gpt-5.5 mesh must NOT get pulled onto the open endpoint.
    assert llm._uses_secondary_gateway("gpt-5.5") is False


def test_claude_still_routes_secondary_without_allowlist(monkeypatch):
    _configure(monkeypatch, open_models=[])
    assert llm._uses_secondary_gateway("claude-sonnet-4-6") is True
    assert llm._uses_secondary_gateway("gemini-3.5-flash") is True


def test_no_secondary_when_gateway_unconfigured(monkeypatch):
    # Even allowlisted, if the secondary gateway isn't configured we can't route there.
    _configure(monkeypatch, open_models=["llama-3.3-70b"], base=None, key=None)
    assert llm._uses_secondary_gateway("llama-3.3-70b") is False
    assert llm._uses_secondary_gateway("claude-sonnet-4-6") is False


def test_allowlist_parses_env_at_import(monkeypatch):
    # The module-level allowlist is built from EROSIONBENCH_OPEN_MODELS at import.
    monkeypatch.setenv("EROSIONBENCH_OPEN_MODELS", " Llama-3.3-70B , qwen2.5-72b ")
    reloaded = importlib.reload(llm)
    try:
        assert reloaded._OPEN_MODEL_ALLOWLIST == frozenset({"llama-3.3-70b", "qwen2.5-72b"})
    finally:
        monkeypatch.delenv("EROSIONBENCH_OPEN_MODELS", raising=False)
        importlib.reload(reloaded)
