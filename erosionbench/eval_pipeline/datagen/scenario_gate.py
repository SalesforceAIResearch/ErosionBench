"""Stage 4: Live Execution + 6-Baseline Validation + HuggingFace Export.

Compiles RenderedScenario into EnterpriseScenario, executes with live adaptive
debate across all 6 baselines, validates ground-truth assertions, and exports
to HuggingFace Parquet format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .assertion_registry import compile_assertions
from .domain_specs import DomainSpec, get_domain_spec
from .state_graph import EnterpriseStateGraph, RenderedScenario

if TYPE_CHECKING:
    from ..schemas import (
        AgentRole,
        ContextDocument,
        EvaluationContext,
        GroundTruthAssertion,
        MemoryState,
        SessionAction,
        Verdict,
    )


__all__ = [
    "BaselineConfig",
    "ValidationResult",
    "ScenarioValidator",
    "export_to_parquet",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for one defense baseline run."""
    name: str
    mode: str  # "single" | "multi"
    firewall_kind: str
    attacker: str = "adaptive"
    description: str = ""


BASELINES: List[BaselineConfig] = [
    BaselineConfig(
        name="single_agent_no_defense",
        mode="single",
        firewall_kind="none",
        description="Single Agent — isolated orchestrator, no firewall",
    ),
    BaselineConfig(
        name="multi_agent_no_defense",
        mode="multi",
        firewall_kind="none",
        description="Multi-Agent No Defense — full mesh, no firewall",
    ),
    BaselineConfig(
        name="llamafirewall",
        mode="multi",
        firewall_kind="llamafirewall",
        description="LlamaFirewall — Meta PromptGuard + AlignmentCheck",
    ),
    BaselineConfig(
        name="llm_guard",
        mode="multi",
        firewall_kind="llm-guard",
        description="LLM Guard — Protect AI DeBERTa classifier",
    ),
    BaselineConfig(
        name="sync_edge_interceptor",
        mode="multi",
        firewall_kind="progent",
        description="Synchronous Edge-Interceptors — Progent privilege-control",
    ),
    BaselineConfig(
        name="async_meta_auditor",
        mode="multi",
        firewall_kind="none",
        attacker="adaptive",
        description="Asynchronous Meta-Auditor — post-hoc detection",
    ),
]


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result of executing one scenario under one baseline."""
    baseline: str
    verdict: str
    trajectory: List[str] = field(default_factory=list)
    debate_transcript: List[Dict[str, Any]] = field(default_factory=list)
    tool_fired: bool = False
    tool_args: Optional[Dict[str, Any]] = None
    suppressed_by: Optional[str] = None
    assertions_fired: List[str] = field(default_factory=list)
    alignment_decay_turn: Optional[int] = None
    policy_consensus: Optional[str] = None


@dataclass
class ValidationResult:
    """Full validation result for one scenario across all baselines."""
    scenario_id: str
    domain: str
    tier: int
    is_malicious: bool
    results: Dict[str, BaselineResult] = field(default_factory=dict)
    validation_passed: bool = True
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Scenario Validator
# ---------------------------------------------------------------------------

class ScenarioValidator:
    """Validates scenarios through live multi-baseline execution."""

    def __init__(
        self,
        mesh_factory: Any = None,
        solver_factory: Any = None,
        max_retries: int = 3,
        debate_mode: str = "adaptive",
    ):
        self._mesh_factory = mesh_factory
        self._solver_factory = solver_factory
        self._max_retries = max_retries
        self._debate_mode = debate_mode

    def validate(
        self,
        rendered: RenderedScenario,
        baselines: Optional[List[BaselineConfig]] = None,
    ) -> ValidationResult:
        """Validate a rendered scenario across all baselines."""
        if baselines is None:
            baselines = BASELINES

        graph = rendered.state_graph
        spec = graph.domain_spec or get_domain_spec(graph.domain)
        scenario = self._compile_scenario(rendered, spec)
        assertions = compile_assertions(graph.assertions)

        result = ValidationResult(
            scenario_id=graph.graph_id,
            domain=graph.domain,
            tier=graph.tier,
            is_malicious=graph.is_malicious,
        )

        for baseline in baselines:
            baseline_result = self._run_baseline(scenario, baseline, assertions, spec)
            result.results[baseline.name] = baseline_result

        result.validation_passed = self._check_validation(result, graph.is_malicious)
        return result

    def _run_baseline(
        self,
        scenario: Any,
        baseline: BaselineConfig,
        assertions: Any,
        spec: DomainSpec,
    ) -> BaselineResult:
        """Execute one baseline and evaluate assertions."""
        if self._mesh_factory is None:
            return self._mock_baseline_result(baseline)

        from ..schemas import Verdict

        try:
            mesh_config = {
                "mode": baseline.mode,
                "firewall_kind": baseline.firewall_kind,
                "attacker": baseline.attacker if baseline.attacker else self._debate_mode,
                "domain_spec": spec,
            }
            mesh = self._mesh_factory(mesh_config)
            context = mesh.run(scenario)

            fired: List[str] = []
            for assertion in assertions:
                if assertion.predicate(context):
                    fired.append(assertion.name)

            verdict = Verdict.COMPROMISED if fired else Verdict.SECURE

            return BaselineResult(
                baseline=baseline.name,
                verdict=verdict.value,
                trajectory=[r.value for r in context.runtime_state.trajectory],
                tool_fired=context.tool_fired,
                tool_args=context.tool_log[-1].arguments if context.tool_log else None,
                suppressed_by=context.runtime_state.execution_suppressed_by,
                assertions_fired=fired,
                alignment_decay_turn=context.runtime_state.alignment_decay_turn,
                policy_consensus=context.runtime_state.policy_consensus,
            )
        except Exception as e:
            logger.warning(f"Baseline {baseline.name} failed: {e}")
            return BaselineResult(baseline=baseline.name, verdict="ERROR")

    def _mock_baseline_result(self, baseline: BaselineConfig) -> BaselineResult:
        """Produce a placeholder result when no mesh factory is configured."""
        return BaselineResult(
            baseline=baseline.name,
            verdict="PENDING",
            trajectory=[],
        )

    def _compile_scenario(self, rendered: RenderedScenario, spec: DomainSpec) -> Any:
        """Compile a RenderedScenario into a format the mesh can execute."""
        from ..schemas import ContextDocument, MemoryState, SessionAction

        graph = rendered.state_graph

        session_history: Dict[int, MemoryState] = {}
        for i, session in enumerate(rendered.session_history):
            action = SessionAction.WRITE if session.get("action") == "write" else SessionAction.EXECUTE
            session_history[i] = MemoryState(
                action=action,
                ledger_delta=session.get("ledger_delta", {}),
                request=session.get("request"),
                narrative=session.get("narrative"),
                source=session.get("source"),
            )

        documents = [
            ContextDocument(
                title=doc.get("title", ""),
                pages=doc.get("pages", []),
            )
            for doc in rendered.rendered_documents
        ]

        return {
            "scenario_id": graph.graph_id,
            "domain": graph.domain,
            "session_history": session_history,
            "long_context_documents": documents,
            "execution_spec": rendered.execution_spec,
            "agent_prompts": rendered.agent_prompts,
            "user_intent": rendered.execute_request,
            "is_malicious": graph.is_malicious,
        }

    def _check_validation(self, result: ValidationResult, is_malicious: bool) -> bool:
        """Check if validation criteria are met."""
        if is_malicious:
            no_defense = result.results.get("multi_agent_no_defense")
            if no_defense and no_defense.verdict == "SECURE":
                return False
        else:
            for baseline_result in result.results.values():
                if baseline_result.assertions_fired:
                    return False
        return True


# ---------------------------------------------------------------------------
# HuggingFace Parquet Export
# ---------------------------------------------------------------------------

def export_to_parquet(
    results: List[ValidationResult],
    rendered_scenarios: Dict[str, RenderedScenario],
    output_path: str,
    push_to_hub: Optional[str] = None,
) -> Path:
    """Export validation results to HuggingFace Parquet format."""
    rows: List[Dict[str, Any]] = []

    for vr in results:
        rendered = rendered_scenarios.get(vr.scenario_id)
        graph = rendered.state_graph if rendered else None

        base_row = {
            "scenario_id": vr.scenario_id,
            "domain": vr.domain,
            "tier": vr.tier,
            "is_malicious": vr.is_malicious,
            "attack_class": graph.attack.attack_class if graph and graph.attack else None,
            "owasp_asi": graph.attack.owasp_asi if graph and graph.attack else None,
            "mastrike_tactic": graph.attack.mastrike_tactic if graph and graph.attack else None,
            "benign_twin_id": graph.benign_twin_id if graph else None,
            "validation_passed": vr.validation_passed,
            "session_history": rendered.session_history if rendered else [],
            "rendered_documents": rendered.rendered_documents if rendered else [],
            "agent_prompts": rendered.agent_prompts if rendered else {},
            "execute_request": rendered.execute_request if rendered else "",
        }

        if vr.results:
            for baseline_name, br in vr.results.items():
                row = {
                    **base_row,
                    "baseline": baseline_name,
                    "verdict": br.verdict,
                    "trajectory": br.trajectory,
                    "debate_transcript": br.debate_transcript,
                    "tool_fired": br.tool_fired,
                    "tool_args": br.tool_args,
                    "suppressed_by": br.suppressed_by,
                    "assertions_fired": br.assertions_fired,
                    "alignment_decay_turn": br.alignment_decay_turn,
                    "policy_consensus": br.policy_consensus,
                }
                rows.append(row)
        else:
            rows.append(base_row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import datasets
        ds = datasets.Dataset.from_list(rows)
        ds.to_parquet(str(output))

        if push_to_hub:
            ds.push_to_hub(push_to_hub)
            logger.info(f"Pushed dataset to HuggingFace Hub: {push_to_hub}")

    except ImportError:
        json_path = output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.warning(
            f"'datasets' package not installed. Exported as JSON: {json_path}"
        )
        return json_path

    logger.info(f"Exported {len(rows)} rows to {output}")
    return output
