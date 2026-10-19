"""Data-driven agent persona catalog for the LangGraph mesh.

No system prompt is hardcoded in the mesh logic. Personas (system prompts, reasoning prompts,
KPI pressures) and the adaptive-debate configuration are loaded from an editable YAML catalog
(:data:`DEFAULT_PROFILE_PATH`), so every agent persona is auditable, explainable, and A/B-testable
for the benchmark. Point :meth:`MeshProfiles.load` at an alternate file (e.g. via
``--agent-profiles``) to swap safety framings or KPI pressures without touching code.

The loaded :class:`MeshProfiles` is the single source of truth the mesh consults via
:meth:`MeshProfiles.system_for` / :meth:`MeshProfiles.reasoning_for`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import yaml

from .schemas import AgentRole

__all__ = ["AgentProfile", "MeshProfiles", "DEFAULT_PROFILE_PATH"]

#: Packaged default persona catalog (editable YAML).
DEFAULT_PROFILE_PATH: str = os.path.join(
    os.path.dirname(__file__), "configs", "agent_mesh.yaml"
)

#: Maps the YAML agent keys onto the canonical :class:`AgentRole` enum.
_ROLE_KEYS: Dict[str, AgentRole] = {
    "gateway_agent": AgentRole.GATEWAY,
    "orchestrator_agent": AgentRole.ORCHESTRATOR,
    "policy_agent": AgentRole.POLICY,
    "deployment_agent": AgentRole.DEPLOYMENT,
}


@dataclass(frozen=True)
class AgentProfile:
    """One agent's editable persona (pure data)."""

    role: AgentRole
    title: str
    system_prompt: str
    role_alias: str = ""
    reasoning_prompt: Optional[str] = None
    pressures: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MeshProfiles:
    """The full, data-driven persona catalog the mesh consults at runtime."""

    agents: Dict[AgentRole, AgentProfile]
    debate_orchestrator_system: str
    debate_tactics: Tuple[str, ...]
    debate_forbidden_patterns: Tuple[str, ...]
    source_path: str = ""
    version: int = 1

    # -- accessors -------------------------------------------------------------
    def profile(self, role: AgentRole) -> AgentProfile:
        return self.agents[role]

    def system_for(self, role: AgentRole) -> str:
        """The terse system prompt for ``role``."""
        return self.agents[role].system_prompt

    def reasoning_for(self, role: AgentRole) -> str:
        """The reasoned-verdict prompt for ``role`` (falls back to the terse one)."""
        prof = self.agents[role]
        return prof.reasoning_prompt or prof.system_prompt

    # -- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "MeshProfiles":
        """Build a catalog from a YAML file (defaults to the packaged catalog)."""
        resolved = path or DEFAULT_PROFILE_PATH
        with open(resolved, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw, source_path=resolved)

    @classmethod
    def from_dict(cls, raw: Dict, *, source_path: str = "") -> "MeshProfiles":
        agents_raw = raw.get("agents", {}) or {}
        agents: Dict[AgentRole, AgentProfile] = {}
        for key, role in _ROLE_KEYS.items():
            spec = agents_raw.get(key, {}) or {}
            system_prompt = (spec.get("system_prompt") or "").strip()
            if not system_prompt:
                raise ValueError(
                    f"agent_mesh profile missing 'system_prompt' for '{key}' (source={source_path})"
                )
            reasoning = spec.get("reasoning_prompt")
            agents[role] = AgentProfile(
                role=role,
                title=spec.get("title", key),
                role_alias=spec.get("role_alias", ""),
                system_prompt=system_prompt,
                reasoning_prompt=(reasoning.strip() if isinstance(reasoning, str) else None),
                pressures=tuple(spec.get("pressures", []) or ()),
            )
        debate = raw.get("debate", {}) or {}
        orch_system = (debate.get("orchestrator_system") or "").strip()
        if not orch_system:
            raise ValueError(f"agent_mesh profile missing debate.orchestrator_system (source={source_path})")
        return cls(
            agents=agents,
            debate_orchestrator_system=orch_system,
            debate_tactics=tuple(debate.get("tactics", []) or ()),
            debate_forbidden_patterns=tuple(debate.get("forbidden_patterns", []) or ()),
            source_path=source_path,
            version=int(raw.get("version", 1) or 1),
        )
