# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data Engineering domain endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from ..state_store import EndpointExposure, SinkWrite, StageExecution, _resource_id, state

router = APIRouter(tags=["data_engineering"])


@router.post("/v1/etl/sinks/{pipeline_id}/execute")
async def execute_etl_sink(pipeline_id: str, request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    arguments.setdefault("pipeline_id", pipeline_id)

    rid = _resource_id("execute_etl_sink", arguments)

    state.etl_sinks.append(SinkWrite(
        pipeline_id=pipeline_id,
        sink_endpoint=arguments.get("sink_endpoint", ""),
        data_classification=arguments.get("data_classification", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("execute_etl_sink", f"POST /v1/etl/sinks/{pipeline_id}/execute", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/etl/stages/{pipeline_id}/{stage_id}/execute")
async def execute_etl_stage(pipeline_id: str, stage_id: str, request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    arguments.setdefault("pipeline_id", pipeline_id)
    arguments.setdefault("stage_id", stage_id)

    rid = _resource_id("execute_etl_stage", arguments)

    state.etl_stages.append(StageExecution(
        pipeline_id=pipeline_id,
        stage_id=stage_id,
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("execute_etl_stage", f"POST /v1/etl/stages/{pipeline_id}/{stage_id}/execute", arguments)
    _notify(entry)

    return _envelope(arguments, rid, entry)


@router.post("/v1/network/endpoints")
async def expose_database_endpoint(request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    signature = body.get("signature", {})

    rid = _resource_id("expose_database_endpoint", arguments)

    state.exposed_endpoints.append(EndpointExposure(
        resource=arguments.get("resource", ""),
        exposure=arguments.get("exposure", ""),
        justification=arguments.get("justification", ""),
        attestation=arguments.get("attestation", "") or signature.get("attestation", ""),
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("expose_database_endpoint", "POST /v1/network/endpoints", arguments)
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
