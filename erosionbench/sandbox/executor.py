# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LiveHttpToolExecutor — concrete implementation of the HttpToolExecutor seam."""

from __future__ import annotations

import re
from typing import Any, Dict

import httpx


TOOL_ENDPOINTS = {
    "trigger_pipeline_execution": "POST /v1/pipelines/{pipeline_id}/execute",
    "update_feature_matrix_registry": "POST /v1/feature-registry/{matrix_id}",
    "deploy_inference_pipeline": "POST /v1/deployments/{model_name}",
    "execute_privileged_override": "POST /v1/admin/overrides/execute",
    "expose_database_endpoint": "POST /v1/network/endpoints",
    "execute_etl_sink": "POST /v1/etl/sinks/{pipeline_id}/execute",
    "execute_etl_stage": "POST /v1/etl/stages/{pipeline_id}/{stage_id}/execute",
    "process_refund": "POST /v1/billing/refunds",
    "apply_sla_waiver": "POST /v1/sla/waivers",
    "override_entitlement": "POST /v1/entitlements/overrides",
    "ship_replacement": "POST /v1/fulfillment/ship",
    "authorize_cross_region_fulfillment": "POST /v1/fulfillment/cross-region",
    "place_medication_order": "POST /v1/cpoe/orders",
    "release_phi": "POST /v1/phi/release",
    "override_formulary_restriction": "POST /v1/formulary/override",
}

_PATH_PARAM_RE = re.compile(r"\{(\w+)\}")


class LiveHttpToolExecutor:
    """Send real HTTP requests to the sandbox server.

    Wire-compatible with InProcessSimulator — the response envelope has
    the same top-level keys the solver expects.
    """

    def __init__(self, base_url: str = "http://localhost:8199") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def execute(self, tool_name: str, arguments: Dict[str, Any], signature: Dict[str, Any]) -> Dict[str, Any]:
        template = TOOL_ENDPOINTS.get(tool_name)
        if not template:
            return {
                "status": 404,
                "ok": False,
                "endpoint": f"POST /v1/{tool_name}",
                "resource_id": "",
                "side_effect": "unknown_tool",
                "echo": dict(arguments),
                "simulated": False,
                "latency_ms": 0,
            }

        method, path_template = template.split(" ", 1)
        path = self._interpolate(path_template, arguments)
        url = f"{self.base_url}{path}"

        try:
            response = self._client.request(
                method=method,
                url=url,
                json={"arguments": arguments, "signature": signature},
            )
            if response.status_code == 200:
                return response.json()
            return {
                "status": response.status_code,
                "ok": False,
                "endpoint": template,
                "resource_id": "",
                "side_effect": "error",
                "echo": dict(arguments),
                "simulated": False,
                "latency_ms": 0,
                "error": response.text[:500],
            }
        except httpx.ConnectError:
            return {
                "status": 503,
                "ok": False,
                "endpoint": template,
                "resource_id": "",
                "side_effect": "sandbox_unreachable",
                "echo": dict(arguments),
                "simulated": True,
                "latency_ms": 0,
                "error": f"Cannot connect to sandbox at {self.base_url}",
            }

    def _interpolate(self, path_template: str, arguments: Dict[str, Any]) -> str:
        """Fill ``{param}`` path segments from ``arguments``.

        A missing OR empty path param would collapse the URL to a trailing slash (e.g.
        ``/v1/deployments/``) which FastAPI does not route -> a spurious 404 that the in-process
        simulator never produces (it ignores path params). Substitute a stable ``unspecified``
        placeholder so the route matches and the live path stays consistent with in-process. The
        placeholder only fills the URL segment; the recorded ``arguments`` (which the solver and the
        sandbox violation rules inspect) are untouched.
        """
        def replacer(match: re.Match) -> str:
            key = match.group(1)
            value = arguments.get(key)
            if value is None or str(value) == "":
                return "unspecified"
            return str(value)
        return _PATH_PARAM_RE.sub(replacer, path_template)

    def close(self) -> None:
        self._client.close()
