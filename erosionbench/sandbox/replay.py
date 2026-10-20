"""Replay API — paced playback of a captured audit-log for the demo's live-attack theater.

The live audit-log is in-memory and resets per eval run, so it's empty between runs. This module
captures a snapshot of the current live log (or accepts one) and replays it into an ISOLATED buffer
at a demo-friendly pace. The replay buffer is entirely separate from the real ``state`` object, so
playback never mutates live eval state — it is a read-only theater over a copy.

The dashboard's stream polls ``/admin/replay/feed``, which returns the LIVE audit-log when no replay
is active and the paced replay buffer when one is running. That single feed endpoint is the frontend
contract; the source switch lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .state_store import state

router = APIRouter(tags=["replay"])


def _entry_to_dict(e: Any) -> Dict[str, Any]:
    return {
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


@dataclass
class ReplaySession:
    """A paced playback of captured entries. ``elapsed`` maps to how many entries are 'revealed'."""

    entries: List[Dict[str, Any]]
    rate_per_sec: float = 2.0
    started_at: Optional[float] = None
    paused_reveal: int = 0  # entries revealed at the moment of the last pause
    is_paused: bool = False

    def start(self) -> None:
        self.started_at = time.time()
        self.paused_reveal = 0
        self.is_paused = False

    def _revealed_count(self) -> int:
        if self.started_at is None:
            return 0
        if self.is_paused:
            return self.paused_reveal
        n = self.paused_reveal + int((time.time() - self.started_at) * self.rate_per_sec)
        return min(n, len(self.entries))

    def revealed(self) -> List[Dict[str, Any]]:
        return self.entries[: self._revealed_count()]

    def status(self) -> Dict[str, Any]:
        revealed = self._revealed_count()
        return {
            "active": True,
            "paused": self.is_paused,
            "rate_per_sec": self.rate_per_sec,
            "revealed": revealed,
            "total": len(self.entries),
            "complete": revealed >= len(self.entries),
        }


# Module-level singleton — one replay at a time is plenty for a demo.
_session: Optional[ReplaySession] = None
# Captured snapshots the operator can replay, keyed by a label.
_snapshots: Dict[str, List[Dict[str, Any]]] = {}


class CaptureRequest(BaseModel):
    label: str = "capture"


class StartRequest(BaseModel):
    label: str = "capture"
    rate_per_sec: float = 2.0


@router.post("/admin/replay/capture")
async def capture_live() -> Dict[str, Any]:
    """Snapshot the CURRENT live audit-log so it can be replayed later, even after a reset."""
    entries = [_entry_to_dict(e) for e in state.audit_log]
    if not entries:
        raise HTTPException(status_code=400, detail="Live audit-log is empty — nothing to capture")
    label = f"capture_{len(_snapshots) + 1}"
    _snapshots[label] = entries
    return {"status": "captured", "label": label, "entries": len(entries)}


@router.get("/admin/replay/list")
async def list_snapshots() -> Dict[str, Any]:
    return {
        "snapshots": [{"label": k, "entries": len(v)} for k, v in _snapshots.items()],
        "live_entries": len(state.audit_log),
    }


@router.post("/admin/replay/start")
async def start_replay(req: StartRequest) -> Dict[str, Any]:
    global _session
    entries = _snapshots.get(req.label)
    if entries is None:
        raise HTTPException(status_code=404, detail=f"No snapshot labelled '{req.label}' (capture one first)")
    _session = ReplaySession(entries=list(entries), rate_per_sec=max(0.1, req.rate_per_sec))
    _session.start()
    return {"status": "replay_started", **_session.status()}


@router.post("/admin/replay/stop")
async def stop_replay() -> Dict[str, str]:
    global _session
    _session = None
    return {"status": "replay_stopped"}


@router.post("/admin/replay/pause")
async def pause_replay() -> Dict[str, Any]:
    if _session is None:
        raise HTTPException(status_code=400, detail="No active replay")
    if not _session.is_paused:
        _session.paused_reveal = _session._revealed_count()
        _session.is_paused = True
    return _session.status()


@router.post("/admin/replay/resume")
async def resume_replay() -> Dict[str, Any]:
    if _session is None:
        raise HTTPException(status_code=400, detail="No active replay")
    if _session.is_paused:
        _session.started_at = time.time()
        _session.is_paused = False
    return _session.status()


@router.get("/admin/replay/feed")
async def feed() -> Dict[str, Any]:
    """The stream the dashboard polls. Replay buffer when active; live audit-log otherwise."""
    if _session is not None:
        return {"mode": "replay", "status": _session.status(), "entries": _session.revealed()}
    return {
        "mode": "live",
        "status": {"active": False, "total": len(state.audit_log)},
        "entries": [_entry_to_dict(e) for e in state.audit_log],
    }
