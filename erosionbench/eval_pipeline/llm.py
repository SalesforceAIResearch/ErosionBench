"""Shared LLM client plumbing for the evaluation pipeline.

A thin, gateway-aware wrapper over the OpenAI SDK so the agent backends
(:mod:`backends`) and the LLM-backed firewalls (:mod:`firewalls`) authenticate
identically. It mirrors the conventions already used elsewhere in ErosionBench
(``erosionbench.mesh.config`` / ``unit_baseline``):

* honors an optional ``X-Api-Key`` gateway header (Salesforce Research gateway),
* clamps temperature for reasoning-class models (gpt-5.x / o1 / o3 / o4) that reject
  any explicit temperature other than the default,
* degrades gracefully: if the SDK or credentials are missing, :pyattr:`available`
  is ``False`` and callers fall back to deterministic behavior.

This module never raises on a failed call -- it returns ``None`` so the caller can fall
back. That keeps a benchmark run alive even if the gateway hiccups mid-corpus.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI as _OpenAI

    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

__all__ = [
    "LLMClient",
    "LLMClientError",
    "OPENAI_BASE_URL",
    "AGENT_MODEL",
    "JUDGE_MODEL",
    "make_raw_openai_client",
    "patch_scanner_client",
    "temperature_for",
    "provider_for_model",
    "base_url_for_model",
    "_system_role",
]

OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
AGENT_MODEL: str = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")
JUDGE_MODEL: str = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Multi-provider routing (OpenAI / Anthropic / Google), all via the same gateway
# ---------------------------------------------------------------------------
# Every provider is reached through the SAME OpenAI-compatible gateway and the SAME
# X-Api-Key auth; only the provider path segment of the base URL differs. The gateway
# normalizes Claude / Gemini to the OpenAI chat-completions schema, so the OpenAI SDK is
# used uniformly. Provider is inferred from the model name (``claude-*`` -> anthropic,
# ``gemini-*`` -> google, everything else -> openai). The default base URL for a provider
# is derived by swapping the ``/openai/`` segment of ``OPENAI_BASE_URL`` for the provider's
# segment; set ``ANTHROPIC_BASE_URL`` / ``GEMINI_BASE_URL`` to override explicitly.
#: Gateway path segment for each provider (overridable via EROSIONBENCH_<P>_PATH_SEGMENT).
_PROVIDER_PATH_SEGMENT: Dict[str, str] = {
    "openai": "openai",
    "anthropic": os.getenv("EROSIONBENCH_ANTHROPIC_PATH_SEGMENT", "anthropic"),
    "google": os.getenv("EROSIONBENCH_GOOGLE_PATH_SEGMENT", "google"),
}

#: Each provider's own OpenAI-compatible endpoint. Used as the base URL when NO gateway is
#: configured (``OPENAI_BASE_URL`` has no ``/openai/`` segment to swap) so a native per-provider
#: key works directly. NOTE: these are the providers' OpenAI-compat shims (beta on some
#: providers) -- convenient for outside users, but not guaranteed bit-identical to the dedicated
#: gateway the paper numbers were validated on. Override with ANTHROPIC_BASE_URL / GEMINI_BASE_URL.
_NATIVE_OPENAI_COMPAT_BASE_URL: Dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


def provider_for_model(model: Optional[str]) -> str:
    """Infer the provider family from a model name.

    ``claude*`` -> ``anthropic``; ``gemini*`` / ``google*`` -> ``google``; everything else
    (gpt-*, o1/o3/o4, etc.) -> ``openai``.
    """
    m = (model or "").lower()
    # Allow an explicit ``provider:model`` prefix to win.
    if m.startswith("anthropic:") or m.startswith("claude") or "claude" in m:
        return "anthropic"
    if m.startswith("google:") or m.startswith("gemini") or "gemini" in m:
        return "google"
    return "openai"


def _derive_base_url(provider: str) -> str:
    """Best-effort base URL for ``provider`` derived from ``OPENAI_BASE_URL``.

    Honors an explicit per-provider override first; otherwise swaps the ``/openai/`` path
    segment of the configured OpenAI base URL for the provider's segment. Falls back to the
    OpenAI base URL unchanged when no ``/openai/`` segment is present (e.g. api.openai.com).
    """
    if provider == "openai":
        return OPENAI_BASE_URL
    override = os.getenv("ANTHROPIC_BASE_URL") if provider == "anthropic" else os.getenv("GEMINI_BASE_URL")
    if override:
        return override
    seg = _PROVIDER_PATH_SEGMENT.get(provider, "openai")
    # Swap the bracketed provider segment (e.g. ``/openai/`` -> ``/anthropic/``) if present.
    if "/openai/" in OPENAI_BASE_URL:
        return OPENAI_BASE_URL.replace("/openai/", f"/{seg}/", 1)
    # No gateway segment to swap (e.g. the public api.openai.com default): fall back to the
    # provider's own OpenAI-compatible endpoint so a native per-provider key works directly.
    return _NATIVE_OPENAI_COMPAT_BASE_URL.get(provider, OPENAI_BASE_URL)


def base_url_for_model(model: Optional[str]) -> str:
    """Resolve the gateway base URL to use for ``model`` (provider-aware)."""
    return _derive_base_url(provider_for_model(model))

_X_API_KEY: Optional[str] = os.getenv("X_API_KEY")
_PLACEHOLDER_KEYS = {
    "", "your_api_key_here", "dummy",
    # Placeholders used in .env.example -- never treat these as real auth.
    "sk-your-openai-key-here", "sk-ant-your-anthropic-key-here", "your-gemini-key-here",
}


def _real_key(name: str) -> Optional[str]:
    """Return the env var ``name`` only if set to a real (non-placeholder) value."""
    val = os.getenv(name, "")
    return val if val and val not in _PLACEHOLDER_KEYS else None


# Native per-provider keys (optional). When set, an outside user can plug in their own
# OpenAI / Anthropic / Google keys and run with NO gateway at all: claude-* uses
# ANTHROPIC_API_KEY, gemini-* uses GEMINI_API_KEY, everything else uses OPENAI_API_KEY.
# The internal gateway path (X_API_KEY / LLM_GATEWAY_*) still takes precedence and is
# unaffected -- these only matter on the primary path when no gateway is configured.
_ANTHROPIC_API_KEY: Optional[str] = _real_key("ANTHROPIC_API_KEY")
_GEMINI_API_KEY: Optional[str] = _real_key("GEMINI_API_KEY")


def _api_key_for_model(model: Optional[str]) -> str:
    """Resolve the primary-path api_key for ``model``, preferring a native per-provider key.

    So a user with their own OpenAI / Anthropic / Google keys can run directly: claude-* ->
    ANTHROPIC_API_KEY, gemini-* -> GEMINI_API_KEY, else OPENAI_API_KEY (or ``dummy`` when the
    X-Api-Key header carries the real auth).
    """
    provider = provider_for_model(model)
    if provider == "anthropic" and _ANTHROPIC_API_KEY:
        return _ANTHROPIC_API_KEY
    if provider == "google" and _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    return os.getenv("OPENAI_API_KEY") or "dummy"

# Optional secondary gateway for providers the primary gateway does not serve
# (e.g. Anthropic / Google). This gateway uses native ``Authorization: Bearer``
# auth (the OpenAI SDK ``api_key``) rather than the ``X-Api-Key`` header. When set,
# claude-* / gemini-* models are routed here instead of the derived primary URL.
_LLM_GATEWAY_BASE_URL: Optional[str] = os.getenv("LLM_GATEWAY_BASE_URL")
_LLM_GATEWAY_API_KEY: Optional[str] = os.getenv("LLM_GATEWAY_API_KEY")

#: Self-hosted open-weight models (e.g. a vLLM endpoint) served over an
#: OpenAI-compatible schema. These resolve to the ``openai`` provider by name, but must ride
#: the secondary (Bearer-auth) gateway -- otherwise their calls go to the frontier
#: ``OPENAI_BASE_URL`` and fail. Set ``EROSIONBENCH_OPEN_MODELS`` to a comma-separated list of
#: served-model-names (case-insensitive), e.g. ``llama-3.3-70b,qwen2.5-72b``.
_OPEN_MODEL_ALLOWLIST: frozenset = frozenset(
    m.strip().lower()
    for m in (os.getenv("EROSIONBENCH_OPEN_MODELS", "") or "").split(",")
    if m.strip()
)


def _uses_secondary_gateway(model: Optional[str]) -> bool:
    """True if ``model`` should route through the secondary (Bearer-auth) gateway."""
    if not (_LLM_GATEWAY_BASE_URL and _LLM_GATEWAY_API_KEY):
        return False
    if (model or "").lower() in _OPEN_MODEL_ALLOWLIST:
        return True
    return provider_for_model(model) in ("anthropic", "google")


def _secondary_base_url() -> str:
    """Normalized chat-completions base URL for the secondary gateway."""
    base = (_LLM_GATEWAY_BASE_URL or "").rstrip("/")
    if base.endswith("/v1"):
        return base + "/"
    return base + "/v1/"


# The secondary gateway is an internal Salesforce host whose TLS chain is signed by a
# private CA not present in certifi's public bundle. We build a combined bundle
# (certifi + Salesforce CA) once and hand the secondary client an httpx client that
# verifies against it -- without disturbing the public certs the primary gateway needs.
#: Candidate locations for the Salesforce private CA bundle (first existing wins).
_SF_CA_CANDIDATES = [
    os.path.expanduser("~/.claude/certs/salesforce-ca-bundle.pem"),
    os.getenv("EROSIONBENCH_EXTRA_CA_BUNDLE", ""),
]


def _combined_ca_bundle() -> Optional[str]:
    """Path to a certifi+Salesforce combined CA bundle, or None if unavailable.

    Cached on disk next to the package's working dir so it is built at most once.
    Returns None (caller falls back to default verification) on any failure.
    """
    sf_ca = next((p for p in _SF_CA_CANDIDATES if p and os.path.exists(p)), None)
    if not sf_ca:
        return None
    try:
        import certifi

        out = os.path.join(os.getcwd(), ".certs_combined.pem")
        # Rebuild if missing or stale relative to either source.
        sources_mtime = max(os.path.getmtime(certifi.where()), os.path.getmtime(sf_ca))
        if not os.path.exists(out) or os.path.getmtime(out) < sources_mtime:
            with open(out, "wb") as fh:
                for src in (certifi.where(), sf_ca):
                    with open(src, "rb") as s:
                        fh.write(s.read())
                        fh.write(b"\n")
        return out
    except Exception:
        return None


def _secondary_http_client():
    """An httpx client that trusts the combined CA bundle, or None to use defaults.

    The secondary gateway is connection-sensitive under load; bound the pool so concurrent
    mesh+auditor calls queue rather than racing into connection resets, and keep generous
    timeouts so a queued request waits instead of failing the (fail-closed) run.
    """
    bundle = _combined_ca_bundle()
    if not bundle:
        return None
    try:
        import httpx

        limits = httpx.Limits(
            max_connections=int(os.getenv("EROSIONBENCH_SECONDARY_MAX_CONN", "4") or 4),
            max_keepalive_connections=2,
        )
        # Generous per-request timeout for the secondary (auditor) gateway. The auditor reads the
        # full transcript + documents, so large-context Claude calls can run long under load; a
        # queued request should wait, not fail the (fail-closed) run. Defaults high, env-overridable
        # to match the primary gateway's EROSIONBENCH_LLM_REQUEST_TIMEOUT knob.
        sec_timeout = float(os.getenv("EROSIONBENCH_SECONDARY_REQUEST_TIMEOUT", "300") or 300)
        return httpx.Client(verify=bundle, timeout=httpx.Timeout(sec_timeout), limits=limits)
    except Exception:
        return None

# llamafirewall / the SDK want OPENAI_API_KEY to exist even when the real auth is a
# gateway header; give it a harmless placeholder so construction never fails.
if _X_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy"


def _have_creds() -> bool:
    """True if we can authenticate (gateway header or a real bearer key)."""
    if _X_API_KEY:
        return True
    if _LLM_GATEWAY_API_KEY:
        return True
    if _ANTHROPIC_API_KEY or _GEMINI_API_KEY:
        return True
    return os.getenv("OPENAI_API_KEY", "") not in _PLACEHOLDER_KEYS


def make_raw_openai_client(model: Optional[str] = None):
    """Build a raw, gateway-aware ``OpenAI`` client (or ``None`` if unavailable).

    Routing is provider-aware:
      * ``claude-*`` / ``gemini-*`` -> secondary gateway (``LLM_GATEWAY_BASE_URL``) with
        native ``Authorization: Bearer`` auth, when configured.
      * everything else -> primary gateway (``OPENAI_BASE_URL``) with ``X-Api-Key`` auth.

    Passing ``model=None`` keeps the default OpenAI base URL (used by the LlamaFirewall
    scanner repointing in :func:`patch_scanner_client`).
    """
    if not (_HAS_OPENAI and _have_creds()):
        return None
    try:
        if _uses_secondary_gateway(model):
            # Secondary gateway: Bearer-auth, OpenAI-compatible chat schema. Verify against
            # the combined CA bundle (certifi + Salesforce private CA) so the internal host's
            # TLS chain validates without overriding the public certs the primary gateway needs.
            sec_kwargs: Dict[str, Any] = {
                "api_key": _LLM_GATEWAY_API_KEY,
                "base_url": _secondary_base_url(),
            }
            http_client = _secondary_http_client()
            if http_client is not None:
                sec_kwargs["http_client"] = http_client
            return _OpenAI(**sec_kwargs)
        kwargs: Dict[str, Any] = {
            "api_key": _api_key_for_model(model),
            "base_url": base_url_for_model(model) if model else OPENAI_BASE_URL,
            # Bound the per-request timeout so a hung gateway fails FAST into our exponential
            # back-off instead of blocking the SDK default (~600s). Under gateway degradation a
            # 10-min hang per worker stalls the whole run; a tight timeout + retry recovers.
            "timeout": _REQUEST_TIMEOUT,
            "max_retries": 0,  # we own retries/back-off in chat(); don't double up.
        }
        if _X_API_KEY:
            kwargs["default_headers"] = {"X-Api-Key": _X_API_KEY}
        return _OpenAI(**kwargs)
    except Exception:
        return None


def patch_scanner_client(scanner):
    """Repoint a LlamaFirewall LLM scanner's client at our gateway-aware client."""
    try:
        scanner.llm.client = make_raw_openai_client()
    except Exception:
        pass
    return scanner


def _requires_default_temperature(model: str) -> bool:
    """Reasoning-class models reject an explicit ``temperature`` (only the default is allowed).

    Two provider families are affected:
      * OpenAI gpt-5.x / o1 / o3 / o4 — reject any explicit non-default temperature.
      * Anthropic Claude Opus 4.8 and the Claude 5 generation — ``temperature`` is DEPRECATED
        and the gateway returns a 400 ("`temperature` is deprecated for this model") when it is
        sent at all. Verified live against ``claude-opus-4-8`` on 2026-07-25.
    For all of these we must omit the temperature kwarg entirely (see ``chat``); the same flag
    also grants reasoning headroom in ``_completion_budget``, which these extended-thinking
    models benefit from. System-role selection is handled separately in ``_system_role`` (Claude
    stays on the ``system`` role), so widening this predicate does not affect it.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    # Anthropic models that deprecated the temperature parameter.
    if "claude" in m and ("opus-4-8" in m or "opus-5" in m or "sonnet-5" in m):
        return True
    return False


#: Extra completion budget granted to reasoning-class models on top of the caller's
#: intended *visible-answer* size. For gpt-5.x / o-series, ``max_completion_tokens``
#: is consumed by hidden reasoning tokens *before* any visible text is emitted, so the
#: small caps the agents/judges request (8-400 tokens, tuned for gpt-4o) would be spent
#: entirely on reasoning and return an empty, ``length``-truncated message. We add this
#: headroom so the model has room to think *and* answer in a single call. Override via
#: ``EROSIONBENCH_REASONING_HEADROOM``.
_REASONING_HEADROOM: int = int(os.getenv("EROSIONBENCH_REASONING_HEADROOM", "4000") or 4000)

#: Number of attempts a single ``chat`` / ``chat_json`` call makes before giving up and
#: returning ``None`` (which the caller's fail-closed guard turns into an aborted run).
#: Gateways occasionally return an empty/non-JSON body for an otherwise-valid request
#: (transient hiccup, content-filter blip, rate-limit). Retrying that one call keeps a
#: long multi-scenario live run alive without weakening the fail-closed guarantee: a
#: genuinely-broken model still returns ``None`` after the retries are exhausted.
#: Override via ``EROSIONBENCH_LLM_MAX_ATTEMPTS``.
_LLM_MAX_ATTEMPTS: int = max(1, int(os.getenv("EROSIONBENCH_LLM_MAX_ATTEMPTS", "3") or 3))

#: Base back-off (seconds) between retry attempts; grows EXPONENTIALLY (base * 2**attempt)
#: and is capped at :data:`_LLM_RETRY_BACKOFF_MAX`. A linear back-off is useless against a
#: gateway that is unreachable for minutes (the overnight failure mode); exponential lets a
#: long run ride out a transient outage without weakening the fail-closed guarantee.
#: Override via ``EROSIONBENCH_LLM_RETRY_BACKOFF`` / ``EROSIONBENCH_LLM_RETRY_BACKOFF_MAX``.
_LLM_RETRY_BACKOFF: float = float(os.getenv("EROSIONBENCH_LLM_RETRY_BACKOFF", "0.75") or 0.75)
_LLM_RETRY_BACKOFF_MAX: float = float(os.getenv("EROSIONBENCH_LLM_RETRY_BACKOFF_MAX", "60") or 60)

#: Substrings identifying a CONNECTION-class failure (gateway down / DNS / TLS / reset) as
#: opposed to a request-level rejection. These are worth waiting out with a longer back-off:
#: the request is fine, the endpoint is briefly gone. Matched case-insensitively against the
#: exception type name + message.
_CONN_ERROR_MARKERS: tuple = (
    "apiconnectionerror", "connection error", "connection reset", "connection refused",
    "connection aborted", "timeout", "timed out", "temporarily unavailable",
    "name resolution", "getaddrinfo", "max retries exceeded",
)
#: Multiplier applied to the (already exponential) back-off when the error looks
#: connection-class. Override via ``EROSIONBENCH_LLM_CONN_BACKOFF_MULT``.
_LLM_CONN_BACKOFF_MULT: float = float(os.getenv("EROSIONBENCH_LLM_CONN_BACKOFF_MULT", "4") or 4)

#: Per-request timeout (seconds) for the PRIMARY OpenAI-gateway client. Without this the SDK
#: default (~600s) lets a single hung request block a worker for 10 minutes under gateway
#: degradation; bounding it turns a hang into a fast APITimeoutError that our back-off retries.
#: Override via ``EROSIONBENCH_LLM_REQUEST_TIMEOUT``.
_REQUEST_TIMEOUT: float = float(os.getenv("EROSIONBENCH_LLM_REQUEST_TIMEOUT", "120") or 120)


def _is_connection_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient endpoint outage rather than a bad request."""
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(m in blob for m in _CONN_ERROR_MARKERS)


def _retry_sleep(attempt: int, *, connection_error: bool = False) -> float:
    """Exponential back-off for ``attempt`` (0-indexed), capped, longer for connection errors."""
    delay = _LLM_RETRY_BACKOFF * (2 ** attempt)
    if connection_error:
        delay *= _LLM_CONN_BACKOFF_MULT
    return min(delay, _LLM_RETRY_BACKOFF_MAX)

#: Minimum completion budget for ``chat_json``. The mesh's JSON calls request small caps
#: (~120 tokens) tuned for GPT-4o, but verbose models (Claude / Gemini) preface the object
#: with reasoning and need room to reach it. Floor + escalation keep provider behavior
#: comparable without altering the decision the model makes. Override via env.
_JSON_MIN_TOKENS: int = int(os.getenv("EROSIONBENCH_JSON_MIN_TOKENS", "1024") or 1024)
_JSON_MAX_TOKENS: int = int(os.getenv("EROSIONBENCH_JSON_MAX_TOKENS", "4096") or 4096)

#: Appended to the system prompt for JSON calls so verbose models lead with the object.
_JSON_ONLY_SUFFIX: str = (
    "IMPORTANT: Respond with ONLY the JSON object requested. No preamble, no explanation, "
    "no markdown commentary before or after -- the JSON object must be the entire response."
)


def _completion_budget(model: str, max_tokens: int) -> int:
    """Resolve the ``max_completion_tokens`` to actually send for ``model``.

    Non-reasoning models keep the caller's ``max_tokens`` verbatim. Reasoning-class
    models get :data:`_REASONING_HEADROOM` added on top so hidden reasoning tokens do
    not starve the visible answer (the caller's value still governs the *answer* length).
    """
    if max_tokens <= 0:
        return max_tokens
    if _requires_default_temperature(model):
        return max_tokens + _REASONING_HEADROOM
    return max_tokens


def _system_role(model: str) -> str:
    """Return the correct role name for system-level instructions.

    GPT-5+, o1, o3, and o4 models use 'developer' instead of 'system'.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "developer"
    return "system"


def temperature_for(model: str, desired: float = 0.0) -> float:
    """Clamp ``desired`` to a value ``model`` accepts.

    Reasoning-class models (gpt-5.x / o-series) reject any explicit temperature other than
    the default and return a 400 for ``temperature=0``; for those we return ``1.0``. All other
    models keep ``desired`` (0.0 by default, for deterministic judging).
    """
    return 1.0 if _requires_default_temperature(model) else desired


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot be constructed or a call fails in strict mode."""
    pass


# ---------------------------------------------------------------------------
# Global (process-wide) usage accumulator
# ---------------------------------------------------------------------------
# Client instances are created deep inside the mesh/auditor per scenario, so a
# top-level runner cannot read a single client's `.usage`. Every client also
# reports into this process-wide registry, keyed by model, so a runner can
# `reset_global_usage()` before a phase and `snapshot_global_usage()` after to
# attribute token counts + latency per model regardless of instance count.
_GLOBAL_USAGE_LOCK = threading.Lock()
_GLOBAL_USAGE: Dict[str, Dict[str, Any]] = {}


def reset_global_usage() -> None:
    """Clear the process-wide token/latency registry (call before a phase)."""
    with _GLOBAL_USAGE_LOCK:
        _GLOBAL_USAGE.clear()


def _accumulate_global_usage(
    model: str, prompt: int, completion: int, total: int, latency: float
) -> None:
    with _GLOBAL_USAGE_LOCK:
        entry = _GLOBAL_USAGE.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_latency_s": 0.0,
                "_latencies": [],
            },
        )
        entry["calls"] += 1
        entry["prompt_tokens"] += prompt
        entry["completion_tokens"] += completion
        entry["total_tokens"] += total
        entry["total_latency_s"] += latency
        entry["_latencies"].append(latency)


def snapshot_global_usage() -> Dict[str, Dict[str, Any]]:
    """Return per-model token + latency stats accumulated since the last reset."""
    with _GLOBAL_USAGE_LOCK:
        out: Dict[str, Dict[str, Any]] = {}
        for model, e in _GLOBAL_USAGE.items():
            lat = sorted(e["_latencies"])
            calls = e["calls"]
            out[model] = {
                "calls": calls,
                "prompt_tokens": e["prompt_tokens"],
                "completion_tokens": e["completion_tokens"],
                "total_tokens": e["total_tokens"],
                "total_latency_s": round(e["total_latency_s"], 2),
                "avg_latency_s": round(e["total_latency_s"] / calls, 2) if calls else 0.0,
                "p95_latency_s": round(lat[int(len(lat) * 0.95)], 2) if lat else 0.0,
            }
        return out


class LLMClient:
    """Gateway-aware chat client with text and JSON helpers.

    Parameters
    ----------
    model:
        Model name to call (e.g. ``gpt-5.5``). Defaults to ``OPENAI_AGENT_MODEL``.
    strict:
        If True, raise LLMClientError on construction failure or call failure instead
        of returning None. Use this when fallback behavior is not acceptable.
    """

    def __init__(self, model: Optional[str] = None, strict: bool = False) -> None:
        self.model: str = model or AGENT_MODEL
        self.provider: str = provider_for_model(self.model)
        self.strict: bool = strict
        self._client = self._build_client()
        if strict and self._client is None:
            raise LLMClientError(
                f"Cannot construct LLM client for model '{self.model}' "
                f"(provider={self.provider}). Check credentials: "
                f"X_API_KEY, OPENAI_API_KEY, or LLM_GATEWAY_API_KEY must be set."
            )
        self._lock = threading.Lock()
        #: Cumulative token consumption across all calls (for trace metrics).
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0
        #: Latency tracking (seconds).
        self.total_latency: float = 0.0
        self._latencies: List[float] = []

    def reset_usage(self) -> None:
        """Zero the cumulative token counters and latency stats."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.total_latency = 0.0
        self._latencies = []

    def _record_usage(self, resp: Any, latency: float = 0.0) -> None:
        prompt = completion = total = 0
        try:
            usage = resp.usage
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            total = int(getattr(usage, "total_tokens", 0) or 0)
        except Exception:
            pass
        with self._lock:
            self.call_count += 1
            self.total_latency += latency
            self._latencies.append(latency)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
        _accumulate_global_usage(self.model, prompt, completion, total, latency)

    @property
    def usage(self) -> Dict[str, Any]:
        """A serializable snapshot of cumulative token consumption and latency."""
        avg_latency = (self.total_latency / self.call_count) if self.call_count else 0.0
        p95_latency = sorted(self._latencies)[int(len(self._latencies) * 0.95)] if self._latencies else 0.0
        return {
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_s": round(avg_latency, 2),
            "p95_latency_s": round(p95_latency, 2),
        }

    # -- construction ----------------------------------------------------------
    def _build_client(self):
        return make_raw_openai_client(self.model)

    @property
    def available(self) -> bool:
        """True if a live chat call can be attempted."""
        return self._client is not None

    # -- calls -----------------------------------------------------------------
    def _create(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """One raw completion call. Returns stripped content, or ``None`` if empty."""
        t0 = time.time()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        self._record_usage(resp, latency)
        content = resp.choices[0].message.content
        return content.strip() if content else None

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        images: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Single-shot chat completion. Returns stripped text, or ``None`` on failure.

        Retries up to :data:`_LLM_MAX_ATTEMPTS` times on a transient empty body or a
        gateway exception (rate-limit, timeout, content-filter blip). Within each attempt
        an empty response is also retried once with the ``max_completion_tokens`` cap
        dropped (covers reasoning-token starvation). Returns ``None`` only after every
        attempt is exhausted, so the caller's fail-closed guard still fires for a
        genuinely-broken model.

        ``images`` (opt-in, multimodal): a list of base64-encoded PNG payloads (or full
        ``data:``/``http`` URLs). When present, the user turn is sent as an OpenAI-schema
        multi-part content array (``text`` + one ``image_url`` part per image); the gateway
        normalizes this to each provider's native vision format. Omitting ``images`` keeps the
        legacy plain-string content byte-for-byte, so the text path is unchanged. This is the
        seam the image-modality (Direction-1) needle rides at eval time — a vision-capable
        agent reads the forged value out of the pixels.
        """
        if self._client is None:
            if self.strict:
                raise LLMClientError(
                    f"LLM client not available for model '{self.model}' — cannot make call."
                )
            return None
        user_content: Any = user
        if images:
            user_content = [{"type": "text", "text": user}]
            for img in images:
                url = img if img.startswith(("data:", "http://", "https://")) else \
                    f"data:image/png;base64,{img}"
                user_content.append({"type": "image_url", "image_url": {"url": url}})
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": _system_role(self.model), "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": _completion_budget(self.model, max_tokens),
        }
        # Reasoning models reject explicit non-default temperatures.
        if not _requires_default_temperature(self.model):
            kwargs["temperature"] = temperature

        for attempt in range(_LLM_MAX_ATTEMPTS):
            conn_error = False
            try:
                result = self._create(dict(kwargs))
                if result:
                    return result
                # Empty body: retry once this attempt with the token cap removed.
                uncapped = {k: v for k, v in kwargs.items() if k != "max_completion_tokens"}
                result = self._create(uncapped)
                if result:
                    return result
                print(
                    f"[LLMClient] {self.model} returned empty content "
                    f"(attempt {attempt + 1}/{_LLM_MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
            except Exception as e:
                conn_error = _is_connection_error(e)
                print(
                    f"[LLMClient] {self.model} call failed "
                    f"(attempt {attempt + 1}/{_LLM_MAX_ATTEMPTS}): {type(e).__name__}: {e}"
                    + ("  [connection-class: longer back-off]" if conn_error else ""),
                    file=sys.stderr,
                )
            if attempt + 1 < _LLM_MAX_ATTEMPTS:
                time.sleep(_retry_sleep(attempt, connection_error=conn_error))
        if self.strict:
            raise LLMClientError(
                f"LLM call failed after {_LLM_MAX_ATTEMPTS} attempts for model '{self.model}'. "
                f"No fallback allowed in strict mode."
            )
        return None

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        images: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Chat completion expected to return a JSON object; tolerant of code fences.

        If a non-empty response comes back but is not parseable as a JSON object (the
        model occasionally answers in prose), the generation is retried up to
        :data:`_LLM_MAX_ATTEMPTS` times before giving up.

        Verbose models (Claude / Gemini) tend to emit reasoning *before* the JSON, which
        a tight ``max_tokens`` (tuned for GPT-4o's terseness) truncates before any object
        is produced. To keep the comparison fair across providers -- without changing what
        the model is free to decide -- JSON calls are given a floor of headroom and a
        strict "JSON only" instruction, and a non-JSON parse escalates the token budget on
        the next attempt rather than failing closed on a formatting artifact.
        """
        # Floor the budget so a verbose preamble cannot starve the object itself.
        budget = max(max_tokens, _JSON_MIN_TOKENS)
        json_system = f"{system}\n\n{_JSON_ONLY_SUFFIX}"
        for attempt in range(_LLM_MAX_ATTEMPTS):
            raw = self.chat(json_system, user, temperature=temperature, max_tokens=budget,
                            images=images)
            if raw is None:
                if self.strict:
                    raise LLMClientError(
                        f"LLM JSON call failed for model '{self.model}' — chat() returned None."
                    )
                return None
            parsed = _extract_json_object(raw)
            if parsed is not None:
                return parsed
            print(
                f"[LLMClient] {self.model} returned non-JSON content "
                f"(attempt {attempt + 1}/{_LLM_MAX_ATTEMPTS})",
                file=sys.stderr,
            )
            # Escalate headroom: a truncated preamble is the most common cause.
            budget = min(budget * 2, _JSON_MAX_TOKENS)
            if attempt + 1 < _LLM_MAX_ATTEMPTS:
                time.sleep(_retry_sleep(attempt))
        if self.strict:
            raise LLMClientError(
                f"LLM JSON call failed after {_LLM_MAX_ATTEMPTS} attempts for model "
                f"'{self.model}' — could not parse JSON from response."
            )
        return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from a model response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
