"""Regression: infrastructure-errored scenarios must be EXCLUDED from ASR, not scored compromised.

The old fail-closed behavior scored any scenario that raised during evaluation as
baseline/firewall/final_compromised=True (verdict_source="error"), which inflated the headline
ASR with transient connectivity/rate-limit noise rather than genuine agent behavior. The fix
retries transient failures and, if a scenario still fails, marks it errored=True with all
compromise flags False and drops it from the ASR denominator.
"""
from __future__ import annotations

import asyncio

import erosionbench.eval_pipeline.meta_auditor as ma
from erosionbench.eval_pipeline.meta_auditor import (
    ScenarioASRDetail,
    run_asr_benchmark_suite,
)
from erosionbench.eval_pipeline.schemas import EnterpriseScenario


def _mal(sid: str) -> EnterpriseScenario:
    return EnterpriseScenario(scenario_id=sid, name=sid, user_intent="do", is_malicious=True)


def _detail(sid: str, *, compromised: bool) -> ScenarioASRDetail:
    return ScenarioASRDetail(
        scenario_id=sid,
        scenario_name=sid,
        attack_class=None,
        baseline_compromised=compromised,
        firewall_compromised=compromised,
        auditor_caught=False,
        final_compromised=compromised,
        intent_alignment_score=0.0,
        alignment_decay_rate=0.0,
        provenance_trust_ratio=0.0,
        causal_session=None,
        verdict_source="test",
        payload_actually_compromised=compromised,
    )


def test_errored_scenario_excluded_from_asr(monkeypatch):
    """One always-failing scenario + one clean-compromised scenario -> ASR=100% over n=1."""
    calls: dict[str, int] = {}

    async def fake_eval(scenario, *a, **k):
        calls[scenario.scenario_id] = calls.get(scenario.scenario_id, 0) + 1
        if scenario.scenario_id == "BOOM":
            raise ConnectionError("simulated transient failure")
        return _detail(scenario.scenario_id, compromised=True)

    monkeypatch.setattr(ma, "_evaluate_scenario", fake_eval)
    # Neutralize the retry backoff so the test is instant.
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(ma.asyncio, "sleep", _no_sleep)

    result = asyncio.run(
        run_asr_benchmark_suite(
            [_mal("GOOD"), _mal("BOOM")],
            mesh_model="stub",
            auditor_model="stub",
            layer=None,
        )
    )

    # Denominator excludes the errored scenario.
    assert result.total_malicious_runs == 1
    assert result.base_asr == 100.0
    assert result.final_system_asr == 100.0

    # The errored row is retried (3 attempts) and retained for transparency.
    assert calls["BOOM"] == 3
    per = {d.scenario_id: d for d in result.per_scenario}
    assert per["BOOM"].errored is True
    assert per["BOOM"].verdict_source == "error"
    # Critically: an errored scenario is NOT counted as compromised.
    assert per["BOOM"].baseline_compromised is False
    assert per["BOOM"].final_compromised is False
    assert per["GOOD"].errored is False


def test_transient_error_recovers_on_retry(monkeypatch):
    """A scenario that fails once then succeeds is scored normally, not errored."""
    attempts: dict[str, int] = {}

    async def flaky_eval(scenario, *a, **k):
        n = attempts.get(scenario.scenario_id, 0) + 1
        attempts[scenario.scenario_id] = n
        if n == 1:
            raise TimeoutError("first attempt times out")
        return _detail(scenario.scenario_id, compromised=False)

    monkeypatch.setattr(ma, "_evaluate_scenario", flaky_eval)
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(ma.asyncio, "sleep", _no_sleep)

    result = asyncio.run(
        run_asr_benchmark_suite([_mal("FLAKY")], mesh_model="stub", auditor_model="stub")
    )

    assert result.total_malicious_runs == 1
    per = {d.scenario_id: d for d in result.per_scenario}
    assert per["FLAKY"].errored is False
    assert result.base_asr == 0.0
    assert attempts["FLAKY"] == 2
