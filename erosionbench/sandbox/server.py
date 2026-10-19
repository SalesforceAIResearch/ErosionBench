"""FastAPI server — assembles domain routers, document API, and admin endpoints."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import config
from .dashboard import router as dashboard_router
from .document_api import doc_store, router as doc_router
from .domain_handlers.autonomous_crm import router as crm_router
from .domain_handlers.cicd_supply_chain import router as cicd_router
from .domain_handlers.data_engineering import router as de_router
from .domain_handlers.healthcare_clinical import router as healthcare_router
from .domain_handlers.quantitative_finance import router as qf_router
from .metrics_api import router as metrics_router
from .observability import SandboxDashboard
from .replay import router as replay_router
from .scenario_compare_api import router as scenario_compare_router
from .state_store import state

_dashboard: Optional[SandboxDashboard] = None


def get_dashboard() -> Optional[SandboxDashboard]:
    return _dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dashboard
    _dashboard = SandboxDashboard(state, port=config.port)
    _dashboard.start()
    yield
    _dashboard.stop()
    _dashboard = None


app = FastAPI(
    title="ErosionBench Sandbox",
    description="Lightweight execution environment for the alignment decay benchmark",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive so a future React frontend on another port can call these endpoints
# unchanged. The endpoints are read-only telemetry + demo replay; no auth surface is exposed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount domain routers
app.include_router(qf_router)
app.include_router(cicd_router)
app.include_router(de_router)
app.include_router(crm_router)
app.include_router(healthcare_router)
app.include_router(doc_router)

# Dashboard + demo telemetry routers (additive; the durable JSON contract the frontend consumes)
app.include_router(metrics_router)
app.include_router(replay_router)
app.include_router(scenario_compare_router)
app.include_router(dashboard_router)


# Admin endpoints
@app.post("/admin/reset")
async def admin_reset() -> Dict[str, str]:
    state.reset()
    doc_store.clear_needle()
    doc_store.clear_session_corpus()
    return {"status": "reset", "message": "All state cleared"}


@app.get("/admin/state")
async def admin_state() -> Dict[str, Any]:
    return state.get_state_summary()


@app.get("/admin/audit-log")
async def admin_audit_log() -> Dict[str, Any]:
    entries = [
        {
            "seq": e.seq,
            "timestamp": e.timestamp,
            "tool_name": e.tool_name,
            "endpoint": e.endpoint,
            "arguments": e.arguments,
            "resource_id": e.resource_id,
            "side_effect": e.side_effect,
            "violations": e.violations,
            "blast_radius_tags": e.blast_radius_tags,
            "domain": e.domain,
        }
        for e in state.audit_log
    ]
    return {"total": len(entries), "entries": entries}


@app.get("/admin/blast-radius")
async def admin_blast_radius() -> Dict[str, Any]:
    # Live view: derived from THIS server's in-memory audit log (empty until an eval run hits it).
    # The static counterpart (/admin/blast-radius/static) replays the frozen final-eval instead.
    return {"mode": "live", **state.get_blast_radius()}


@app.get("/admin/health")
async def admin_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - _start_time,
        "operations_recorded": len(state.audit_log),
        "sandbox_version": "0.1.0",
    }


_start_time = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="ErosionBench Sandbox Server")
    parser.add_argument("--host", default=config.host, help="Bind host")
    parser.add_argument("--port", type=int, default=config.port, help="Bind port")
    parser.add_argument("--log-level", default=config.log_level, help="Log level")
    parser.add_argument("--corpus-dir", default=config.corpus_dir, help="Corpus directory")
    args = parser.parse_args()

    config.host = args.host
    config.port = args.port
    config.log_level = args.log_level
    config.corpus_dir = args.corpus_dir

    # doc_store is built at module import, when config.corpus_dir only reflects SANDBOX_CORPUS_DIR.
    # The --corpus-dir flag is parsed here, AFTER that, so load it into the existing store now — else
    # the flag is a silent no-op and a standalone `--corpus-dir` boot serves 404s for every base doc.
    # (The eval harness never hit this: it feeds docs via the per-scenario session overlay.)
    if args.corpus_dir:
        from .document_api import doc_store
        if not doc_store._load_generated_corpus(args.corpus_dir):
            raise SystemExit(
                f"ERROR: --corpus-dir={args.corpus_dir!r} has no usable corpus_manifest.json under it "
                f"(expected <dir>/<domain>/corpus_manifest.json). Generate the corpus first, or drop "
                f"the flag to rely on the per-scenario session overlay."
            )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
