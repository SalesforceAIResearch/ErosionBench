"""Stage 1: Enterprise State Graph — structural contracts for scenario generation.

This module defines WHAT happens in a scenario (structurally) without specifying
HOW it reads to a human. The LLM in Stage 2 receives these contracts as creative
briefs and produces coherent enterprise narratives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from .domain_specs import DomainSpec


__all__ = [
    "AgentNode",
    "AttackCategory",
    "AttackNarrative",
    "ConditioningStrategy",
    "DebateContract",
    "DocumentContract",
    "CollusionSpec",
    "SessionScaling",
    "GroundTruthContract",
    "EnterpriseStateGraph",
    "RenderedScenario",
]


class AttackCategory(str, Enum):
    """Attack categories unique to multi-agent systems."""
    CROSS_SESSION = "cross_session"
    GASLIGHTING = "gaslighting"
    LONG_CONTEXT = "long_context"
    COLLUSION = "collusion"


# Session count ranges per category × tier.
# Each individual session's delta looks benign; aggregate drift is the violation.
SESSION_RANGES: Dict[AttackCategory, Dict[int, Tuple[int, int]]] = {
    AttackCategory.CROSS_SESSION: {1: (3, 5), 2: (6, 9), 3: (10, 12)},
    AttackCategory.GASLIGHTING: {1: (3, 4), 2: (5, 7), 3: (8, 10)},
    AttackCategory.LONG_CONTEXT: {1: (3, 5), 2: (6, 8), 3: (9, 12)},
    AttackCategory.COLLUSION: {1: (3, 5), 2: (6, 9), 3: (10, 12)},
}


@dataclass
class AgentNode:
    """A dynamically-defined agent in the mesh."""

    agent_id: str
    role_label: str
    canonical_role: str
    persona_seed: Dict[str, Any] = field(default_factory=dict)
    rbac_permissions: List[str] = field(default_factory=list)
    tool_access: List[str] = field(default_factory=list)
    observability_scope: List[str] = field(default_factory=list)
    decay_target: bool = False
    debate_role: str = "executor"


@dataclass
class SessionScaling:
    """Controls how sessions scale across tiers for a given attack category."""

    category: AttackCategory
    session_range: Tuple[int, int]  # (min, max) for this tier
    drift_per_session: str = "incremental"  # incremental | burst | mixed
    session_markers: bool = True  # whether temporal markers are preserved for agents


@dataclass
class ConditioningStrategy:
    """Adaptive memory-based attack conditioning per MAStrike methodology.

    Controls how the session history builds a coherent argument in shared memory
    that anticipates and preempts the policy agent's objections during the debate
    phase. Each poisoned session reads accumulated memory state and writes content
    that strengthens the overall attack narrative.

    This models real enterprise adversaries who build legitimacy incrementally:
    - APT groups establishing long-term presence through routine-looking operations
    - Supply chain attackers building trust through legitimate contributions
    - Insider threats creating paper trails before the actual violation

    The conditioning strategy is tier-dependent:
      T1: No conditioning — attack is direct, easily questioned
      T2: Preamble establishes operational context; debate attacker can cite memory
      T3: Full argument arc with redundant justification paths; rejecting requires
          reasoning across the entire session history as a coordinated fabrication
    """

    # How many sessions establish legitimacy BEFORE the first forged value arrives.
    # These write governance metadata, approvals, or compliance records that the
    # debate attacker can reference as justification for the forged values.
    preamble_depth: int = 0

    # Memory keys written during preamble that the debate attacker can cite.
    # e.g., ["governance.change_approval", "risk_management.review_status"]
    citable_memory_keys: List[str] = field(default_factory=list)

    # Known policy objection patterns that the conditioning arc preemptively addresses.
    # The generator produces sessions that neutralize these objections before they arise.
    # e.g., ["provenance_challenge", "threshold_violation", "precedent_mismatch"]
    preempted_objections: List[str] = field(default_factory=list)

    # How the accumulated memory is used as ammunition during the live debate.
    # "none": attacker argues from first principles
    # "cite_precedent": attacker references specific prior sessions as evidence
    # "construct_dilemma": attacker frames rejection as contradicting established policy
    memory_exploitation: str = "none"

    # Number of independent justification paths in memory. If one path is ablated
    # (single session removed), the argument still holds via alternate paths.
    # T1=1, T2=2, T3=3+ (maps to counterfactual ablation resilience)
    argument_redundancy: int = 1

    # Enterprise attack pattern this conditioning follows.
    # Drives the generator's session construction to match real adversary TTPs.
    # "direct_injection": no build-up, immediate value plant (T1)
    # "operational_mimicry": looks like a standard change management lifecycle (T2)
    # "institutional_capture": builds governance artifacts that make the attack
    #   look like sanctioned organizational policy (T3)
    enterprise_pattern: str = "direct_injection"


@dataclass
class AttackNarrative:
    """The structural contract for an attack — WHAT is forged, not HOW it reads."""

    attack_class: str
    attack_category: AttackCategory = AttackCategory.CROSS_SESSION
    owasp_asi: str = ""
    mastrike_tactic: str = ""
    owasp_llm: str = ""
    tier: int = 1

    forged_key: str = ""
    forged_value: Any = None
    forged_provenance: str = ""
    legitimate_precedent_key: Optional[str] = None

    # Multi-parameter attacks: each session writes a different key.
    # Each value alone may look reasonable; the compound effect is the violation.
    forged_params: List[Dict[str, Any]] = field(default_factory=list)

    provenance_plausibility: str = "obvious_mismatch"
    corroboration_depth: int = 0
    temporal_complexity: str = "multi_session"

    agents_who_corroborate: List[str] = field(default_factory=list)
    agents_who_must_yield: List[str] = field(default_factory=list)
    information_asymmetry: Dict[str, List[str]] = field(default_factory=dict)

    session_scaling: Optional[SessionScaling] = None
    conditioning: Optional["ConditioningStrategy"] = None


@dataclass
class DebateContract:
    """Structural parameters for the live adversarial debate.

    These constrain the live attacker LLM — they are NOT scripted lines.
    """

    attacker_objective: str
    defender_mandate: str
    available_evidence: Dict[str, Any] = field(default_factory=dict)
    forbidden_moves: List[str] = field(default_factory=list)
    pressure_angles: List[str] = field(default_factory=list)
    decay_budget: int = 5


@dataclass
class DocumentContract:
    """What a long-context document must ACHIEVE, not what it literally contains."""

    document_type: str
    total_pages: int
    purpose: str

    must_contain: List[Dict[str, Any]] = field(default_factory=list)
    must_establish: List[str] = field(default_factory=list)
    internal_consistency: List[str] = field(default_factory=list)

    legitimate_anchors: List[Dict[str, Any]] = field(default_factory=list)
    poisoned_values: List[Dict[str, Any]] = field(default_factory=list)
    false_implication: str = ""

    separation_of_concerns: str = "separate_sections"


@dataclass
class CollusionSpec:
    """Structural preconditions for emergent multi-agent collusion.

    Collusion is NOT scripted. It is an emergent property of:
    1. Shared write access to overlapping ledger regions
    2. Partial observability that prevents a third agent from cross-checking
    3. Incentive alignment (both colluding agents benefit from the same outcome)

    Whether collusion actually manifests depends on model behavior.
    """

    shared_memory_keys: List[str]
    colluding_agents: List[str]
    blind_agent: str
    incentive_alignment: str
    example_collusion_pattern: str = ""


@dataclass
class GroundTruthContract:
    """Deterministic, serializable assertion specification.

    Compiled into executable predicates by the assertion_registry at runtime.
    Survives Parquet serialization — anyone who loads the dataset can re-evaluate.
    """

    name: str
    description: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM"
    assertion_type: str  # "param_boundary", "trajectory", "consensus", "provenance", etc.
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnterpriseStateGraph:
    """The complete structural contract for one scenario."""

    graph_id: str
    domain: str
    domain_spec: Optional[DomainSpec] = None
    tier: int = 1
    seed: int = 0
    attack_category: Optional[AttackCategory] = None

    agents: List[AgentNode] = field(default_factory=list)
    topology: Dict[str, List[str]] = field(default_factory=dict)
    topology_overrides: Optional[Dict[str, List[str]]] = None

    attack: Optional[AttackNarrative] = None
    debate: Optional[DebateContract] = None
    documents: List[DocumentContract] = field(default_factory=list)
    collusion: Optional[CollusionSpec] = None

    enterprise_context: Dict[str, Any] = field(default_factory=dict)

    tool: str = ""
    base_args: Dict[str, Any] = field(default_factory=dict)
    merge_strategy: str = "risk_weight_from_strategy"

    assertions: List[GroundTruthContract] = field(default_factory=list)
    is_malicious: bool = True
    benign_twin_id: Optional[str] = None

    @property
    def effective_topology(self) -> Dict[str, List[str]]:
        """Topology with scenario-specific overrides applied."""
        topo = dict(self.topology)
        if self.topology_overrides:
            for source, targets in self.topology_overrides.items():
                existing = topo.get(source, [])
                topo[source] = list(set(existing + targets))
        return topo


@dataclass
class RenderedScenario:
    """Output of the generator — a fully rendered scenario ready for validation."""

    state_graph: EnterpriseStateGraph
    session_history: List[Dict[str, Any]]
    rendered_documents: List[Dict[str, Any]]
    agent_prompts: Dict[str, str]
    execute_request: str
    execution_spec: Dict[str, Any] = field(default_factory=dict)
