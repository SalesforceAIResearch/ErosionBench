"""Append-only, cryptographically traceable shared state ledger.

The mesh's "memory" is a persistent key/value store that survives across sessions.
For peer-review-grade verifiability (AgentDojo-inspired paradigm) we never mutate state in
place: every write is recorded as an immutable :class:`LedgerEntry` whose
``state_hash`` chains to its parent (a Merkle-style hash chain). That gives us two
guarantees a benchmark reviewer cares about:

* **Tamper evidence** -- :meth:`StateLedger.verify` recomputes the whole chain, so a
  silently edited transaction is detectable.
* **Provenance** -- every state key can be traced to the exact session UUID and
  parent timestamp that introduced it.

The poisoning (WRITE) sessions of an :class:`~.schemas.EnterpriseScenario` are
replayed into this ledger *before* the clean EXECUTE session runs, so the agents
operate as confused deputies over attacker-seeded — but fully auditable — state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = ["LedgerEntry", "StateLedger"]

_GENESIS = "0" * 64


def _canonical(delta: Dict[str, Any]) -> str:
    """Stable JSON encoding so identical deltas always hash identically."""
    return json.dumps(delta, sort_keys=True, default=str, separators=(",", ":"))


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable, hash-chained transaction in the state ledger."""

    seq: int
    session_uuid: str
    parent_hash: str
    timestamp: str
    delta: Dict[str, Any]
    narrative: str
    state_hash: str
    #: System-attested write channel (provenance). Part of the signed chain so it is
    #: tamper-evident: an attacker cannot retroactively relabel a poison write as control-plane.
    source: str = ""

    def recompute_hash(self) -> str:
        payload = "|".join(
            [
                str(self.seq),
                self.session_uuid,
                self.parent_hash,
                self.timestamp,
                self.source,
                _canonical(self.delta),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "session_uuid": self.session_uuid,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "delta": self.delta,
            "narrative": self.narrative,
            "source": self.source,
            "state_hash": self.state_hash,
        }


class StateLedger:
    """Append-only key/value store with a verifiable transaction chain."""

    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []
        self._state: Dict[str, Any] = {}

    # -- writes ----------------------------------------------------------------
    def apply(
        self,
        delta: Dict[str, Any],
        *,
        session_uuid: Optional[str] = None,
        narrative: str = "",
        source: str = "",
    ) -> LedgerEntry:
        """Record ``delta`` as a new chained transaction and fold it into state.

        Each call gets its own ``session_uuid`` (unless one is supplied) and a parent
        timestamp/hash linking it to the previous entry, so the full provenance of any
        key is reconstructable from :pyattr:`transactions`. ``source`` records the
        system-attested write channel and is folded into the signed hash chain.
        """
        parent_hash = self._entries[-1].state_hash if self._entries else _GENESIS
        seq = len(self._entries)
        sid = session_uuid or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        payload = "|".join([str(seq), sid, parent_hash, ts, source, _canonical(delta)])
        state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        entry = LedgerEntry(
            seq=seq,
            session_uuid=sid,
            parent_hash=parent_hash,
            timestamp=ts,
            delta=dict(delta),
            narrative=narrative,
            state_hash=state_hash,
            source=source,
        )
        self._entries.append(entry)
        self._state.update(delta)
        return entry

    # -- reads -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """A flat, read-only view of current effective state (last-writer-wins)."""
        return dict(self._state)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    @property
    def transactions(self) -> List[LedgerEntry]:
        return list(self._entries)

    def provenance(self, key: str) -> List[Dict[str, Any]]:
        """Every transaction that touched ``key`` (oldest first), for auditing."""
        return [e.to_dict() for e in self._entries if key in e.delta]

    def source_for(self, key: str) -> Optional[str]:
        """The system-attested write channel of the latest transaction touching ``key``."""
        for entry in reversed(self._entries):
            if key in entry.delta:
                return entry.source or None
        return None

    def all_sources_for(self, key: str) -> List[str]:
        """All write channels that have ever touched ``key`` (oldest first)."""
        return [entry.source for entry in self._entries if key in entry.delta and entry.source]

    # -- integrity -------------------------------------------------------------
    def verify(self) -> bool:
        """Recompute the chain end-to-end; True iff no entry has been tampered with."""
        parent = _GENESIS
        for entry in self._entries:
            if entry.parent_hash != parent:
                return False
            if entry.recompute_hash() != entry.state_hash:
                return False
            parent = entry.state_hash
        return True

    @property
    def head_hash(self) -> str:
        return self._entries[-1].state_hash if self._entries else _GENESIS

    def to_serializable(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]
