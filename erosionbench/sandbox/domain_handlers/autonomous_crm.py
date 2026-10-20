"""Autonomous CRM domain endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from ..state_store import (
    CrossRegionAuth,
    EntitlementOverride,
    RefundRecord,
    ShipmentRecord,
    SlaWaiverRecord,
    _resource_id,
    state,
)

router = APIRouter(tags=["autonomous_crm"])


@router.post("/v1/billing/refunds")
async def process_refund(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("process_refund", arguments)

    state.refunds.append(RefundRecord(
        ticket_id=arguments.get("ticket_id", ""),
        amount=float(arguments.get("amount", 0)),
        reason_code=arguments.get("reason_code", ""),
        entitlement_ref=arguments.get("entitlement_ref", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("process_refund", "POST /v1/billing/refunds", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/sla/waivers")
async def apply_sla_waiver(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("apply_sla_waiver", arguments)

    state.sla_waivers.append(SlaWaiverRecord(
        ticket_id=arguments.get("ticket_id", ""),
        sla_metric=arguments.get("sla_metric", ""),
        waiver_justification=arguments.get("waiver_justification", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("apply_sla_waiver", "POST /v1/sla/waivers", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/entitlements/overrides")
async def override_entitlement(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    signature = body.get("signature", {})

    rid = _resource_id("override_entitlement", arguments)

    state.entitlement_overrides.append(EntitlementOverride(
        customer_id=arguments.get("customer_id", ""),
        entitlement_id=arguments.get("entitlement_id", ""),
        override_scope=arguments.get("override_scope", ""),
        authority_ref=arguments.get("authority_ref", "") or signature.get("authority_ref", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("override_entitlement", "POST /v1/entitlements/overrides", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/fulfillment/ship")
async def ship_replacement(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("ship_replacement", arguments)

    state.shipments.append(ShipmentRecord(
        ticket_id=arguments.get("ticket_id", ""),
        product_sku=arguments.get("product_sku", ""),
        shipping_priority=arguments.get("shipping_priority", "standard"),
        warehouse_id=arguments.get("warehouse_id", ""),
        warranty_ref=arguments.get("warranty_ref", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("ship_replacement", "POST /v1/fulfillment/ship", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/fulfillment/cross-region")
async def authorize_cross_region_fulfillment(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})

    rid = _resource_id("authorize_cross_region_fulfillment", arguments)

    state.cross_region_auths.append(CrossRegionAuth(
        ticket_id=arguments.get("ticket_id", ""),
        source_warehouse=arguments.get("source_warehouse", ""),
        destination_region=arguments.get("destination_region", ""),
        override_reason=arguments.get("override_reason", ""),
        billing_authorization=arguments.get("billing_authorization", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("authorize_cross_region_fulfillment", "POST /v1/fulfillment/cross-region", arguments)
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
        "latency_ms": 10,
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
