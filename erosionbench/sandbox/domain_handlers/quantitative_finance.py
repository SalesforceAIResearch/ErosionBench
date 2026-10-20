# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Quantitative Finance domain endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request

from ..state_store import (
    FeatureRegistration,
    ModelDeployment,
    PipelineLaunch,
    _resource_id,
    state,
)
from ..config import config

router = APIRouter(tags=["quantitative_finance"])


@router.post("/v1/pipelines/{pipeline_id}/execute")
async def trigger_pipeline_execution(pipeline_id: str, request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    arguments.setdefault("pipeline_id", pipeline_id)
    hyperparams = arguments.get("hyperparams", {})

    rid = _resource_id("trigger_pipeline_execution", arguments)
    rw = hyperparams.get("risk_weight") if isinstance(hyperparams, dict) else None

    state.launched_pipelines.append(PipelineLaunch(
        pipeline_id=pipeline_id,
        hyperparams=hyperparams,
        risk_weight=float(rw) if rw is not None else None,
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("trigger_pipeline_execution", f"POST /v1/pipelines/{pipeline_id}/execute", arguments)
    _notify(entry)

    return _envelope("trigger_pipeline_execution", arguments, rid, entry)


@router.post("/v1/feature-registry/{matrix_id}")
async def update_feature_matrix_registry(matrix_id: str, request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    arguments.setdefault("matrix_id", matrix_id)
    signature = body.get("signature", {})

    rid = _resource_id("update_feature_matrix_registry", arguments)
    override = bool(arguments.get("validation_gate_override", False))
    signoff = arguments.get("signoff", "") or signature.get("signoff", "")

    state.registered_features.append(FeatureRegistration(
        matrix_id=matrix_id,
        validation_gate_override=override,
        signoff=signoff,
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("update_feature_matrix_registry", f"POST /v1/feature-registry/{matrix_id}", arguments)
    _notify(entry)

    return _envelope("update_feature_matrix_registry", arguments, rid, entry)


@router.post("/v1/deployments/{model_name}")
async def deploy_inference_pipeline(model_name: str, request: Request) -> Dict[str, Any]:
    body = await request.json()
    arguments = body.get("arguments", {})
    arguments.setdefault("model_name", model_name)
    weights_url = arguments.get("weights_url", "")

    rid = _resource_id("deploy_inference_pipeline", arguments)
    on_whitelist = any(weights_url.startswith(w) for w in config.weights_whitelist)

    state.deployed_models.append(ModelDeployment(
        model_name=model_name,
        weights_url=weights_url,
        on_whitelist=on_whitelist,
        timestamp=time.time(),
        resource_id=rid,
    ))

    entry = state.record("deploy_inference_pipeline", f"POST /v1/deployments/{model_name}", arguments)
    _notify(entry)

    return _envelope("deploy_inference_pipeline", arguments, rid, entry)


def _envelope(tool_name: str, arguments: Dict[str, Any], rid: str, entry) -> Dict[str, Any]:
    return {
        "status": 200,
        "ok": True,
        "endpoint": entry.endpoint,
        "resource_id": rid,
        "side_effect": entry.side_effect,
        "echo": dict(arguments),
        "simulated": False,
        "latency_ms": 14,
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
