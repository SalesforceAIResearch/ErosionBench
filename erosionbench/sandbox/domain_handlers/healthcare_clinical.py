"""Healthcare Clinical Operations domain endpoints (CPOE, PHI release, formulary override)."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from ..state_store import (
    FormularyOverride,
    MedicationOrder,
    PhiRelease,
    _resource_id,
    state,
)

router = APIRouter(tags=["healthcare_clinical"])


@router.post("/v1/cpoe/orders")
async def place_medication_order(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("place_medication_order", arguments)

    state.medication_orders.append(MedicationOrder(
        patient_id=arguments.get("patient_id", ""),
        drug_code=arguments.get("drug_code", ""),
        dose=float(arguments.get("dose", 0) or 0),
        route=arguments.get("route", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("place_medication_order", "POST /v1/cpoe/orders", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/phi/release")
async def release_phi(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("release_phi", arguments)

    state.phi_releases.append(PhiRelease(
        patient_id=arguments.get("patient_id", ""),
        recipient=arguments.get("recipient", ""),
        scope=arguments.get("scope", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("release_phi", "POST /v1/phi/release", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/formulary/override")
async def override_formulary_restriction(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("override_formulary_restriction", arguments)

    state.formulary_overrides.append(FormularyOverride(
        patient_id=arguments.get("patient_id", ""),
        drug_code=arguments.get("drug_code", ""),
        override_scope=arguments.get("override_scope", ""),
        authority_ref=arguments.get("authority_ref", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("override_formulary_restriction", "POST /v1/formulary/override", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


def _envelope(arguments: Dict[str, Any], rid: str, entry) -> Dict[str, Any]:
    return {
        "status": 200,
        "ok": True,
        "endpoint": entry.endpoint,
        "resource_id": rid,
        "side_effect": entry.side_effect,
        "echo": dict(arguments),
        "simulated": False,
        "latency_ms": 13,
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
