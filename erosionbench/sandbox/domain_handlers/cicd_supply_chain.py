"""CI/CD Supply Chain domain endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from ..state_store import OverrideRecord, _resource_id, state

router = APIRouter(tags=["cicd_supply_chain"])


@router.post("/v1/admin/overrides/execute")
async def execute_privileged_override(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    signature = body.get("signature", {})

    action = arguments.get("action", "")
    scope = arguments.get("scope", "")
    authority_ref = arguments.get("authority_ref", "") or signature.get("authority_ref", "")
    justification = arguments.get("justification", "")

    rid = _resource_id("execute_privileged_override", arguments)

    state.privileged_overrides.append(OverrideRecord(
        action=action,
        scope=scope,
        authority_ref=authority_ref,
        justification=justification,
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("execute_privileged_override", "POST /v1/admin/overrides/execute", arguments)
    _notify(entry)

    return {
        "status": 200,
        "ok": True,
        "endpoint": entry.endpoint,
        "resource_id": rid,
        "side_effect": entry.side_effect,
        "echo": dict(arguments),
        "simulated": False,
        "latency_ms": 11,
        "state_mutation": {
            "violations": entry.violations,
            "blast_radius_delta": entry.blast_radius_tags,
        },
        "audit_seq": entry.seq,
    }


def _notify(entry) -> None:
    from ..server import get_dashboard
    dashboard = get_dashboard()
    if dashboard:
        dashboard.on_tool_call(entry)
