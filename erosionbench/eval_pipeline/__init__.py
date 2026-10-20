# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Financial-ML multi-agent security evaluation pipeline (LangGraph + LlamaFirewall).

A modular, threat-agnostic data pipeline (AgentDojo-inspired goal verification) for evaluating
stateful, multi-turn attacks against multi-agent systems. Standardized on a real LangGraph
mesh and a live LlamaFirewall baseline.

Module map
----------
* :mod:`schemas`        -- core typed abstractions (``EnterpriseScenario``, assertions, telemetry).
* :mod:`scenario_pool`  -- the dataset-as-code factory (Financial-ML Profiles A / B / C).
* :mod:`ledger`         -- append-only, hash-chained shared state ledger.
* :mod:`tools`          -- explicit per-agent Python tool-execution permissions.
* :mod:`langgraph_mesh` -- the real LangGraph StateGraph (single-agent + multi-agent loops).
* :mod:`firewalls`      -- the LlamaFirewall perimeter (PromptGuard input + AlignmentCheck output).
* :mod:`solver`         -- deterministic ground-truth contracts + aggregate security metrics.
* :mod:`traces`         -- append-only ``evaluation_traces.json`` persistence.
* :mod:`main`           -- the orchestrator and summary report.
"""

from __future__ import annotations

from .firewalls import (
    Firewall,
    LLMGuardFirewall,
    PromptGuardFirewall,
    ProtectAILLMGuardFirewall,
    build_firewall,
)
from .progent import ProgentBaseline
from .langgraph_mesh import LangGraphMesh, MeshConfig
from .ledger import LedgerEntry, StateLedger
from .llm import LLMClient
from .scenario_pool import (
    ScenarioPool,
    build_default_pool,
    build_single_agent_pool,
    build_standard_pool,
)
from .schemas import (
    AgentRole,
    EnterpriseScenario,
    EvaluationContext,
    GroundTruthAssertion,
    ThreatCategory,
    Verdict,
)
from .solver import (
    AggregateMetrics,
    DeterministicMeshSolver,
    ScenarioResult,
    assert_parameter_boundary,
    assert_topology_invariant,
)
from .tools import ToolBoundary

__all__ = [
    # schemas
    "AgentRole",
    "EnterpriseScenario",
    "EvaluationContext",
    "GroundTruthAssertion",
    "ThreatCategory",
    "Verdict",
    # dataset
    "ScenarioPool",
    "build_standard_pool",
    "build_default_pool",
    "build_single_agent_pool",
    # runtime
    "LangGraphMesh",
    "MeshConfig",
    "StateLedger",
    "LedgerEntry",
    "ToolBoundary",
    # firewalls (defense)
    "Firewall",
    "LLMGuardFirewall",
    "PromptGuardFirewall",
    "ProtectAILLMGuardFirewall",
    "ProgentBaseline",
    "build_firewall",
    # llm plumbing
    "LLMClient",
    # solver + contracts
    "DeterministicMeshSolver",
    "ScenarioResult",
    "AggregateMetrics",
    "assert_topology_invariant",
    "assert_parameter_boundary",
]
