"""Routing tests for native per-provider keys (no-gateway path).

An outside user should be able to drop in their own OpenAI / Anthropic / Google keys and run
with NO gateway: claude-* -> ANTHROPIC_API_KEY against api.anthropic.com, gemini-* ->
GEMINI_API_KEY against the Google OpenAI-compat endpoint, everything else -> OPENAI_API_KEY
against the public OpenAI API. These tests pin that fallback AND verify the gateway/segment
path still takes precedence when configured, so the internal setup is unaffected.
"""
import pytest

from erosionbench.eval_pipeline import llm

PUBLIC_OPENAI = "https://api.openai.com/v1"


def _clean_public_env(monkeypatch):
    """Simulate a no-gateway public install: public OpenAI base, no overrides/gateway."""
    monkeypatch.setattr(llm, "OPENAI_BASE_URL", PUBLIC_OPENAI)
    monkeypatch.setattr(llm, "_X_API_KEY", None)
    monkeypatch.setattr(llm, "_LLM_GATEWAY_BASE_URL", None)
    monkeypatch.setattr(llm, "_LLM_GATEWAY_API_KEY", None)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)


def test_native_base_urls_without_gateway(monkeypatch):
    _clean_public_env(monkeypatch)
    assert llm.base_url_for_model("gpt-5.5") == PUBLIC_OPENAI
    assert llm.base_url_for_model("claude-opus-4-8") == "https://api.anthropic.com/v1/"
    assert (
        llm.base_url_for_model("gemini-3.5-flash")
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def test_native_api_key_selected_by_provider(monkeypatch):
    _clean_public_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real")
    monkeypatch.setattr(llm, "_ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setattr(llm, "_GEMINI_API_KEY", "gemini-real")
    assert llm._api_key_for_model("gpt-5.5") == "sk-openai-real"
    assert llm._api_key_for_model("claude-opus-4-8") == "sk-ant-real"
    assert llm._api_key_for_model("gemini-3.5-flash") == "gemini-real"


def test_falls_back_to_openai_key_when_native_unset(monkeypatch):
    _clean_public_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real")
    monkeypatch.setattr(llm, "_ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(llm, "_GEMINI_API_KEY", None)
    # No native key -> claude still resolves a key (OPENAI_API_KEY) rather than crashing.
    assert llm._api_key_for_model("claude-opus-4-8") == "sk-openai-real"


def test_gateway_segment_swap_wins_over_native(monkeypatch):
    # When a gateway base URL with a /openai/ segment is configured, the segment-swap convention
    # must win -- native endpoints are NOT substituted. This preserves the internal setup.
    monkeypatch.setattr(llm, "OPENAI_BASE_URL", "https://gw.example/openai/v1/")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    assert llm.base_url_for_model("claude-opus-4-8") == "https://gw.example/anthropic/v1/"
    assert llm.base_url_for_model("gemini-3.5-flash") == "https://gw.example/google/v1/"


def test_explicit_override_wins_over_native(monkeypatch):
    _clean_public_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://my-proxy.example/anthropic/v1/")
    assert llm.base_url_for_model("claude-opus-4-8") == "https://my-proxy.example/anthropic/v1/"


def test_native_key_satisfies_creds_gate(monkeypatch):
    monkeypatch.setattr(llm, "_X_API_KEY", None)
    monkeypatch.setattr(llm, "_LLM_GATEWAY_API_KEY", None)
    monkeypatch.setattr(llm, "_ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setattr(llm, "_GEMINI_API_KEY", None)
    monkeypatch.setenv("OPENAI_API_KEY", "")  # no OpenAI key at all
    assert llm._have_creds() is True


def test_placeholder_native_key_is_ignored(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-anthropic-key-here")
    assert llm._real_key("ANTHROPIC_API_KEY") is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert llm._real_key("ANTHROPIC_API_KEY") == "sk-ant-real"
