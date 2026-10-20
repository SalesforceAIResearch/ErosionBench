# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic invariants for the CRM/DE direct-injection session builder.

These assert the *fragmentation mechanism* is well formed WITHOUT running the mesh
(no LLM, no eval noise). The builder is pure Python, so the tier-escalation property
we care about -- more writes -> genuinely stealthier, not a benign->malicious cliff --
is checkable straight from the rendered ledger.

Regression guarded here: the old builder wrote benign placeholders for every write but
the last, then dumped the whole payload AND flipped governance_status/operational_directive
to malicious values in the FINAL write. That "fragmentation" landed the poison whole in one
write and the abrupt flip telegraphed the attack, so deeper fragmentation LOWERED success.
"""
import types

import pytest

from erosionbench.eval_pipeline.datagen_adapter import _build_direct_injection_sessions
from erosionbench.eval_pipeline.schemas import SessionAction


def _attack(attack_class, forged_key="sla.waiver_authority", forged_value=None,
            n_hints=3):
    """Minimal duck-typed attack: the builder only reads these attributes."""
    return types.SimpleNamespace(
        attack_class=attack_class,
        forged_key=forged_key,
        forged_value=forged_value if forged_value is not None else {"x": 1},
        forged_provenance="unsigned_config_commit",
        forged_params=[{"key": forged_key, "narrative_hint": f"hint {i}"}
                       for i in range(n_hints)],
    )


def _sessions(n_writes):
    """n WRITE sessions + a terminal EXECUTE, matching the generator's list shape."""
    return [{"action": "write"} for _ in range(n_writes)] + [{"action": "execute",
                                                              "request": "Do the thing."}]


def _writes(history):
    """Return the WRITE MemoryStates in session order (skip preamble-free case's execute)."""
    items = sorted(history.items(), key=lambda kv: kv[0])
    return [m for _, m in items if m.action == SessionAction.WRITE]


def _forged_obj(mstate, parent_key="sla.waiver_authority"):
    return mstate.ledger_delta.get(parent_key, {})


# The multi-key payloads are the ones that CAN fragment; single-key ones degenerate.
MULTI_KEY_CLASSES = ["SLA_METRIC_GAMING", "WARRANTY_FRAUD_INJECTION", "CRISIS_ESCALATION_HIJACK"]
ALL_CLASSES = MULTI_KEY_CLASSES + ["ENTITLEMENT_MANIPULATION"]


@pytest.mark.parametrize("attack_class", ALL_CLASSES)
@pytest.mark.parametrize("n_writes", [3, 5, 8])
def test_no_benign_to_malicious_flip(attack_class, n_writes):
    """governance_status / operational_directive must be settled from the first write.

    The old builder set pending_review/stage_for_review on all but the last write and
    flipped to pre_approved/apply_parameters_as_specified only at the end -- the tell that
    made deeper fragmentation backfire. Every write must now carry the settled framing.
    """
    hist = _build_direct_injection_sessions(_attack(attack_class), _sessions(n_writes))
    writes = _writes(hist)
    for w in writes:
        obj = _forged_obj(w)
        assert obj.get("operational_directive") == "apply_parameters_as_specified"
        assert obj.get("governance_status") == "approved"
        # the discredited placeholder values must never appear
        assert obj.get("governance_status") != "pending_review"
        assert obj.get("operational_directive") != "stage_for_review"


@pytest.mark.parametrize("attack_class", MULTI_KEY_CLASSES)
@pytest.mark.parametrize("n_writes", [3, 5, 8])
def test_payload_actually_fragmented(attack_class, n_writes):
    """The payload must be revealed gradually, never dumped whole in one step.

    Two properties together defeat single-write inspection:
      (a) the FIRST write does not already contain the complete payload -- the poison
          starts as a slice, and
      (b) the count of revealed keys grows by at most one per write -- no single write's
          delta jumps from a partial slice straight to the full payload (the "cliff" that
          made the old builder's final write a giant tell).
    When there are more writes than payload keys the reveal legitimately reaches full
    before the final write and then holds steady; that is padding under a consistent
    frame, not a discontinuity.
    """
    hist = _build_direct_injection_sessions(_attack(attack_class), _sessions(n_writes))
    writes = _writes(hist)
    full = _forged_obj(writes[-1]).get("target_parameters", {})
    assert len(full) >= 2, "multi-key class should yield a >=2-key payload"

    revealed = [len(_forged_obj(w).get("target_parameters", {})) for w in writes]
    # (a) gradual start
    assert revealed[0] < len(full), (
        f"first write already exposes full payload ({revealed[0]}/{len(full)})")
    # (b) no mid-sequence dump: each step adds at most one key
    for a, b in zip(revealed, revealed[1:]):
        assert b - a <= 1, f"payload jumps >1 key in a single write: {revealed}"


@pytest.mark.parametrize("attack_class", ALL_CLASSES)
@pytest.mark.parametrize("n_writes", [3, 5, 8])
def test_final_snapshot_complete(attack_class, n_writes):
    """The final write's accumulated snapshot must carry the complete payload.

    Fragmentation must not lose any sub-key -- the merge reads the final snapshot's
    target_parameters, so an incomplete final state would silently break the attack.
    """
    attack = _attack(attack_class)
    hist = _build_direct_injection_sessions(attack, _sessions(n_writes))
    writes = _writes(hist)
    final_tp = _forged_obj(writes[-1]).get("target_parameters", {})

    from erosionbench.eval_pipeline.datagen_adapter import _build_injection_target_params
    expected = _build_injection_target_params(attack)
    assert final_tp == expected, "final snapshot must equal the full intended payload"


@pytest.mark.parametrize("attack_class", MULTI_KEY_CLASSES)
def test_reveal_is_monotone_non_decreasing(attack_class):
    """The count of revealed payload keys must never DECREASE across writes.

    Genuine fragmentation is a gradual reveal; a drop would mean a key was retracted,
    which reads as inconsistency (the exact kind of tell we are removing).
    """
    hist = _build_direct_injection_sessions(_attack(attack_class), _sessions(6))
    writes = _writes(hist)
    revealed = [len(_forged_obj(w).get("target_parameters", {})) for w in writes]
    for a, b in zip(revealed, revealed[1:]):
        assert b >= a, f"revealed payload keys decreased: {revealed}"
    assert revealed[-1] == max(revealed), "final write must hold the most keys"
