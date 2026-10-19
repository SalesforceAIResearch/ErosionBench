"""Domain-specific ToolBoundary factory.

Extends the base ToolBoundary with domain-specific consequential tools
without breaking the existing tool execution flow for legacy scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from .domain_specs import DomainSpec, ToolSpec

# Lazy import to avoid triggering eval_pipeline.__init__ (which requires langgraph)
_DEPLOYMENT_TOOLS: frozenset | None = None


def _get_deployment_tools() -> frozenset:
    global _DEPLOYMENT_TOOLS
    if _DEPLOYMENT_TOOLS is None:
        from ..schemas import DEPLOYMENT_TOOLS
        _DEPLOYMENT_TOOLS = DEPLOYMENT_TOOLS
    return _DEPLOYMENT_TOOLS


__all__ = [
    "get_deployment_tools_for_domain",
    "get_tool_schemas_for_domain",
]


def get_deployment_tools_for_domain(domain_spec: DomainSpec) -> frozenset:
    """Return the full set of consequential tools for a domain.

    Merges base DEPLOYMENT_TOOLS with domain-specific consequential tools.
    """
    domain_consequential: Set[str] = {
        name for name, spec in domain_spec.tools.items()
        if spec.is_consequential
    }
    return _get_deployment_tools() | frozenset(domain_consequential)


def get_tool_schemas_for_domain(domain_spec: DomainSpec) -> Dict[str, Dict[str, Any]]:
    """Build a tool schemas dict for InProcessSimulator from a domain spec."""
    schemas: Dict[str, Dict[str, Any]] = {}
    for name, spec in domain_spec.tools.items():
        schemas[name] = {
            "endpoint": spec.endpoint,
            "required": spec.required_params,
            "summary": spec.summary,
            "consequential": spec.is_consequential,
        }
    return schemas
