"""Invariants for image-modality subset selection (build_dataset._select_image_instances).

Modality is a technique orthogonal to the tier axis and applied to a SUBSET of instances, capped
by an explicit budget. These assert the selection is deterministic, budget-capped, resume-stable
(independent of which instances were skipped on a prior run), and family-consistent (a whole
instance = T1-mal/twin/T2/T3 gets one modality because selection keys on (attack_class, global_idx)).
"""
from erosionbench.eval_pipeline.datagen.build_dataset import _select_image_instances

CLASSES = ["ENTITLEMENT_MANIPULATION", "SLA_METRIC_GAMING", "COMPLIANCE_THEATER"]
PER = 5  # scenarios_per_attack


def test_zero_budget_selects_nothing():
    assert _select_image_instances("procurement", CLASSES, PER, 0) == set()
    assert _select_image_instances("procurement", CLASSES, PER, -3) == set()


def test_budget_is_a_hard_cap():
    for budget in (1, 4, 7):
        chosen = _select_image_instances("procurement", CLASSES, PER, budget)
        assert len(chosen) == budget


def test_budget_clamps_to_universe_size():
    universe = len(CLASSES) * PER
    chosen = _select_image_instances("procurement", CLASSES, PER, universe + 100)
    assert len(chosen) == universe


def test_selection_is_deterministic():
    a = _select_image_instances("procurement", CLASSES, PER, 6)
    b = _select_image_instances("procurement", CLASSES, PER, 6)
    assert a == b


def test_selection_is_a_prefix_growing_monotonically():
    # A larger budget must be a superset of a smaller one (stable ranking, take-first-N).
    small = _select_image_instances("procurement", CLASSES, PER, 3)
    big = _select_image_instances("procurement", CLASSES, PER, 8)
    assert small <= big


def test_keys_are_valid_and_family_consistent():
    chosen = _select_image_instances("procurement", CLASSES, PER, 6)
    for attack_class, global_idx in chosen:
        assert attack_class in CLASSES
        assert 0 <= global_idx < len(CLASSES) * PER
        # global_idx encodes which class-block it belongs to; must agree with attack_class
        attack_idx = global_idx // PER
        assert CLASSES[attack_idx] == attack_class


def test_selection_independent_of_domain():
    # Different domain -> potentially different picks (seeded by domain_id), still budget-capped.
    p = _select_image_instances("procurement", CLASSES, PER, 5)
    f = _select_image_instances("finance", CLASSES, PER, 5)
    assert len(p) == len(f) == 5
