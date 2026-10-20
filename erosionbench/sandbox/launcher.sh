#!/usr/bin/env bash
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
# Launch the ErosionBench Sandbox execution environment.
# Usage: bash erosionbench/sandbox/launcher.sh [--port 8199]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${1:-8199}"
if [[ "${1:-}" == "--port" ]]; then
    PORT="${2:-8199}"
fi

export EROSIONBENCH_SANDBOX_URL="http://localhost:${PORT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "╭──────────────────────────────────────────────────────╮"
echo "│         ErosionBench Sandbox Execution Environment      │"
echo "╰──────────────────────────────────────────────────────╯"
echo ""
echo "  Server:   http://localhost:${PORT}"
echo "  Admin:    http://localhost:${PORT}/admin/health"
echo "  Docs:     http://localhost:${PORT}/docs (OpenAPI)"
echo ""
echo "  To connect the pipeline:"
echo "    export EROSIONBENCH_SANDBOX_URL=http://localhost:${PORT}"
echo ""
echo "  Endpoints:"
echo "    POST /v1/pipelines/{id}/execute      (QF)"
echo "    POST /v1/feature-registry/{id}       (QF)"
echo "    POST /v1/deployments/{name}          (QF)"
echo "    POST /v1/admin/overrides/execute     (CI/CD)"
echo "    POST /v1/etl/sinks/{id}/execute      (DE)"
echo "    POST /v1/etl/stages/{id}/{sid}/exec  (DE)"
echo "    POST /v1/network/endpoints           (DE)"
echo "    POST /v1/billing/refunds             (CRM)"
echo "    POST /v1/sla/waivers                 (CRM)"
echo "    POST /v1/entitlements/overrides      (CRM)"
echo "    POST /v1/fulfillment/ship            (CRM)"
echo "    POST /v1/fulfillment/cross-region    (CRM)"
echo ""
echo "  Document API:"
echo "    GET  /v1/docs/corpus/{domain}"
echo "    GET  /v1/docs/fetch/{doc_id}?page_start=N&page_end=M"
echo "    POST /v1/docs/needle"
echo "    GET  /v1/docs/search?q=query"
echo ""
echo "Starting server..."
echo ""

python -m erosionbench.sandbox.server --host 0.0.0.0 --port "${PORT}" --log-level info
