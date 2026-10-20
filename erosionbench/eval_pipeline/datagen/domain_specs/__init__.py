"""Domain Spec Registry — structural declarations for each enterprise domain.

Each domain declares its own agent topology, tool catalog, debate parameters,
attack classes, and framework mappings. The registry is the single source of
truth for what a domain's mesh looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set


class AgentRole(str, Enum):
    """Canonical agent roles (mirrors eval_pipeline.schemas.AgentRole)."""

    GATEWAY = "gateway_agent"
    ORCHESTRATOR = "orchestrator_agent"
    POLICY = "policy_agent"
    DEPLOYMENT = "deployment_agent"


__all__ = [
    "AgentSpec",
    "MeshTopology",
    "DebateSpec",
    "DomainSpec",
    "DOMAIN_REGISTRY",
    "get_domain_spec",
]


# ---------------------------------------------------------------------------
# Core spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSpec:
    """One agent node in a domain-specific mesh."""

    agent_id: str
    role_label: str
    canonical_role: AgentRole
    tool_access: List[str]
    observability_scope: List[str]
    decay_eligible: bool
    debate_role: str  # "attacker" | "defender" | "corroborator" | "executor"
    pressures: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MeshTopology:
    """A directed graph of agent connections."""

    agents: tuple  # Tuple[AgentSpec, ...] — frozen requires hashable
    edges: Dict[str, List[str]]
    entry_node: str
    exit_node: str
    decay_target: str
    partial_observability_map: Dict[str, List[str]]

    def agent_ids(self) -> List[str]:
        return [a.agent_id for a in self.agents]

    def get_agent(self, agent_id: str) -> Optional[AgentSpec]:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None


@dataclass(frozen=True)
class DebateSpec:
    """Domain-specific debate parameters."""

    pressure_angles: List[str]
    forbidden_moves: List[str]
    evidence_scope: List[str]
    max_turns: int = 5


@dataclass(frozen=True)
class ToolSpec:
    """Schema for a domain-specific tool."""

    endpoint: str
    required_params: List[str]
    summary: str
    is_consequential: bool = True


@dataclass(frozen=True)
class DomainSpec:
    """Complete structural specification for one domain."""

    domain_id: str
    display_name: str
    topology: MeshTopology
    tools: Dict[str, ToolSpec]
    debate: DebateSpec
    attack_classes: List[str]
    framework_mappings: Dict[str, Dict[str, str]]
    assertion_types: List[str]
    enterprise_context_template: Dict[str, Any] = field(default_factory=dict)
    scenario_prefix: str = ""


# ---------------------------------------------------------------------------
# Shared forbidden moves (hard guardrails for debate — no prompt injection)
# ---------------------------------------------------------------------------

COMMON_FORBIDDEN_MOVES: List[str] = [
    r"ignore\s+(all|previous|prior)\s+instructions",
    r"disregard\s+.*\s+(policy|guidelines|rules)",
    r"you are now",
    r"developer mode",
    r"system prompt",
    r"exfiltrat",
    r"bypass\s+.*\s+(control|firewall|guardrail)",
    r"jailbreak",
    r"ignore\s+policy",
    r"disable\s+safety",
]


# ---------------------------------------------------------------------------
# Registry (populated by domain module imports)
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: Dict[str, DomainSpec] = {}


def register_domain(spec: DomainSpec) -> DomainSpec:
    """Register a domain spec in the global registry."""
    DOMAIN_REGISTRY[spec.domain_id] = spec
    return spec


def get_domain_spec(domain_id: str) -> DomainSpec:
    """Retrieve a registered domain spec by ID."""
    if domain_id not in DOMAIN_REGISTRY:
        _load_all_domains()
    if domain_id not in DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain: {domain_id!r}. "
            f"Available: {list(DOMAIN_REGISTRY.keys())}"
        )
    return DOMAIN_REGISTRY[domain_id]


def _load_all_domains() -> None:
    """Import all domain modules to populate the registry."""
    from . import quantitative_finance  # noqa: F401
    from . import cicd_supply_chain  # noqa: F401
    from . import data_engineering  # noqa: F401
    from . import autonomous_crm  # noqa: F401
    from . import healthcare_clinical  # noqa: F401
