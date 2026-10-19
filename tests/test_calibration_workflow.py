"""Phase 5 regression tests: frozen calibration workflow (deterministic, offline).

Covers the plan's Phase-5 required checks, none of which need a live model:
  * Family-split leakage prevention: no family (base / _CONTROL / _T2 / _T3) can
    straddle the calibration/evaluation boundary; split is deterministic in seed.
  * Empirical benign-null CDF behaviour: one-sided p-value with the conformal +1
    correction; refuses to resolve an FPR the sample is too small to support;
    does NOT fall back to a 1-score pseudo-p-value.
  * Artifact schema / version / hash checks: round-trips, detects tampering,
    rejects a wrong schema_version.
  * No evaluation-time fitting: an artifact refuses to be applied to an eval split
    that overlaps its calibration families.
  * Deterministic artifact loading: load == build (same content hash).
  * Refusal to claim guarantees without sufficient calibration data.
"""
from __future__ import annotations

import math

import pytest

from erosionbench.eval_pipeline.calibration_workflow import (
    ARTIFACT_SCHEMA_VERSION,
    EmpiricalNull,
    FrozenCalibrationArtifact,
    InsufficientCalibrationError,
    MIN_BENIGN_SAMPLES,
    MIN_FAMILIES,
    SplitLeakageError,
    assess_sufficiency,
    bootstrap_ci,
    build_artifact,
    dataset_hash,
    family_of,
    family_split,
)


# ---------------------------------------------------------------------------
# family_of
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sid,fam", [
    ("DE_SHARED_M_0000", "DE_SHARED_M_0000"),
    ("DE_SHARED_M_0000_CONTROL", "DE_SHARED_M_0000"),
    ("DE_SHARED_M_0000_T2", "DE_SHARED_M_0000"),
    ("DE_SHARED_M_0000_T3", "DE_SHARED_M_0000"),
    ("DE_SHARED_M_0000_T2_CONTROL", "DE_SHARED_M_0000"),
    ("FOO_0001_BENIGN", "FOO_0001"),
    ("FOO_0001_TWIN", "FOO_0001"),
])
def test_family_of_collapses_variants(sid, fam):
    assert family_of(sid) == fam


# ---------------------------------------------------------------------------
# family_split — leakage prevention + determinism
# ---------------------------------------------------------------------------
def _dataset(n_families: int = 20):
    ids = []
    for i in range(n_families):
        base = f"FAM_{i:04d}"
        ids += [base, f"{base}_CONTROL", f"{base}_T2", f"{base}_T3"]
    return ids


def test_split_is_leak_free_across_all_variants():
    ids = _dataset(20)
    split = family_split(ids, calibration_fraction=0.5, seed=7)
    split.assert_disjoint()  # would raise on any straddling family
    calib_fams = set(split.calibration_families)
    eval_fams = set(split.evaluation_families)
    assert not (calib_fams & eval_fams)
    # Every id lands on exactly one side, and with its whole family.
    for sid in ids:
        fam = family_of(sid)
        on_calib = sid in split.calibration_ids
        on_eval = sid in split.evaluation_ids
        assert on_calib ^ on_eval
        assert (fam in calib_fams) == on_calib
        assert (fam in eval_fams) == on_eval


def test_split_is_deterministic_in_seed():
    ids = _dataset(20)
    a = family_split(ids, seed=3)
    b = family_split(ids, seed=3)
    assert a.split_hash == b.split_hash
    assert a.calibration_ids == b.calibration_ids
    # A different seed generally reassigns families -> different hash.
    c = family_split(ids, seed=99)
    assert c.split_hash != a.split_hash


def test_split_order_independent():
    ids = _dataset(20)
    shuffled = list(reversed(ids))
    a = family_split(ids, seed=1)
    b = family_split(shuffled, seed=1)
    # Same families each side regardless of input order.
    assert set(a.calibration_families) == set(b.calibration_families)
    assert a.split_hash == b.split_hash


def test_split_never_empties_a_side():
    ids = _dataset(2)
    split = family_split(ids, calibration_fraction=0.5, seed=0)
    assert split.calibration_families and split.evaluation_families


def test_manually_corrupted_split_is_caught():
    from erosionbench.eval_pipeline.calibration_workflow import FamilySplit
    bad = FamilySplit(
        calibration_ids=("A", "A_T2"),
        evaluation_ids=("A_CONTROL",),
        calibration_families=("A",),
        evaluation_families=("A",),  # same family both sides == leak
        seed=0,
        calibration_fraction=0.5,
    )
    with pytest.raises(SplitLeakageError):
        bad.assert_disjoint()


# ---------------------------------------------------------------------------
# EmpiricalNull — CDF behaviour, conformal correction, refusal
# ---------------------------------------------------------------------------
def test_empirical_pvalue_conformal_correction():
    null = EmpiricalNull(tuple(float(i) for i in range(100)), higher_is_anomalous=True)
    # A value at/above the max: only itself in the tail -> (0+1)/(100+1).
    assert null.p_value(1000.0) == pytest.approx(1.0 / 101.0)
    # A value at/below the min: whole sample in the tail -> (100+1)/(101).
    assert null.p_value(-1.0) == pytest.approx(1.0)


def test_empirical_pvalue_is_not_one_minus_score():
    # Benign scores clustered near 0; a raw score of 0.3 must NOT map to 1-0.3=0.7.
    null = EmpiricalNull((0.0, 0.0, 0.0, 0.0, 0.1), higher_is_anomalous=True)
    p = null.p_value(0.3)
    assert p == pytest.approx(1.0 / 6.0)  # only the fresh point in the tail
    assert abs(p - (1.0 - 0.3)) > 0.4     # decisively different from pseudo-p


def test_empirical_null_vpr_orientation():
    # For VPR lower==more anomalous: a low VPR should get a SMALL p-value.
    null = EmpiricalNull((0.9, 0.95, 1.0, 0.85, 0.92), higher_is_anomalous=False)
    p_low = null.p_value(0.1)
    p_high = null.p_value(1.0)
    assert p_low < p_high


def test_empirical_null_zero_samples_refuses():
    null = EmpiricalNull((), higher_is_anomalous=True)
    with pytest.raises(InsufficientCalibrationError):
        null.p_value(0.5)


def test_threshold_for_fpr_refuses_when_unresolvable():
    # 14 benign samples cannot resolve alpha=0.05 (1/15 = 0.067 > 0.05).
    null = EmpiricalNull(tuple(float(i) for i in range(14)), higher_is_anomalous=True)
    assert null.threshold_for_fpr(0.05) is None
    # 100 samples can (1/101 < 0.05).
    null2 = EmpiricalNull(tuple(float(i) for i in range(100)), higher_is_anomalous=True)
    assert null2.threshold_for_fpr(0.05) is not None


def test_threshold_controls_fpr_on_benign_sample():
    scores = tuple(float(i) / 100.0 for i in range(100))
    null = EmpiricalNull(scores, higher_is_anomalous=True)
    tau = null.threshold_for_fpr(0.05)
    assert tau is not None
    fp = sum(1 for s in scores if s > tau) / len(scores)
    assert fp <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
def test_bootstrap_ci_deterministic_and_bracketing():
    vals = [0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.18, 0.22]
    lo1, hi1 = bootstrap_ci(vals, seed=42)
    lo2, hi2 = bootstrap_ci(vals, seed=42)
    assert (lo1, hi1) == (lo2, hi2)          # seeded => reproducible
    assert lo1 <= sum(vals) / len(vals) <= hi1
    assert lo1 <= hi1


def test_bootstrap_ci_empty_is_nan():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------------------
# Sufficiency gate
# ---------------------------------------------------------------------------
def test_sufficiency_rejects_small_sample():
    rep = assess_sufficiency(benign_n=14, family_n=14)
    assert rep.sufficient is False
    assert rep.reasons
    assert rep.finest_resolvable_fpr == pytest.approx(1.0 / 15.0)


def test_sufficiency_accepts_large_sample():
    rep = assess_sufficiency(benign_n=MIN_BENIGN_SAMPLES, family_n=MIN_FAMILIES)
    assert rep.sufficient is True
    assert rep.reasons == ()


# ---------------------------------------------------------------------------
# Artifact — build / serialize / integrity / no-eval-fitting
# ---------------------------------------------------------------------------
def _build_sufficient_artifact():
    ids = _dataset(40)  # 40 families
    split = family_split(ids, calibration_fraction=0.5, seed=5)
    # Enough benign nulls to be sufficient and resolve alpha=0.05.
    benign = {
        "sdi": [0.0] * 60,
        "cii": [0.0] * 60,
        "vad": [0.0] * 60,
        "vpr": [1.0] * 60,
    }
    orient = {"sdi": True, "cii": True, "vad": True, "vpr": False}
    return ids, split, build_artifact(
        scenario_ids=ids,
        split=split,
        mesh_model="gpt-4o",
        auditor_model="claude-sonnet-4-6",
        code_version="testsha",
        benign_null_samples=benign,
        metric_orientations=orient,
        target_fpr=0.05,
        seeds={"split": 5, "bootstrap": 0},
    )


def test_artifact_roundtrip_and_hash(tmp_path):
    ids, split, art = _build_sufficient_artifact()
    path = tmp_path / "cal.json"
    art.save(str(path))
    loaded = FrozenCalibrationArtifact.load(str(path))
    assert loaded.content_hash() == art.content_hash()
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.dataset_hash == dataset_hash(ids)
    assert loaded.split_hash == split.split_hash


def test_artifact_detects_tampering():
    ids, split, art = _build_sufficient_artifact()
    d = art.to_dict(include_hash=True)
    d["target_fpr"] = 0.5  # tamper after hashing
    with pytest.raises(ValueError, match="content_hash mismatch"):
        FrozenCalibrationArtifact.from_dict(d)


def test_artifact_rejects_wrong_schema_version():
    ids, split, art = _build_sufficient_artifact()
    d = art.to_dict(include_hash=True)
    d["schema_version"] = "cal-artifact-v0"
    with pytest.raises(ValueError, match="schema mismatch"):
        FrozenCalibrationArtifact.from_dict(d)


def test_artifact_refuses_fitting_on_overlapping_eval_split():
    ids, split, art = _build_sufficient_artifact()
    # Using the artifact on its OWN calibration families is the leak we forbid.
    calib_ids = list(split.calibration_ids)
    with pytest.raises(SplitLeakageError):
        art.assert_not_fit_on(calib_ids)
    # On the held-out eval families it is fine.
    art.assert_not_fit_on(list(split.evaluation_ids))


def test_sufficient_artifact_yields_threshold():
    ids, split, art = _build_sufficient_artifact()
    assert art.sufficiency.sufficient is True
    tau = art.guarantee_threshold("sdi")
    assert tau is not None


def test_insufficient_artifact_refuses_guarantee():
    ids = _dataset(14)  # 14 families, small
    split = family_split(ids, seed=1)
    benign = {"sdi": [0.0] * 14}
    art = build_artifact(
        scenario_ids=ids,
        split=split,
        mesh_model="gpt-4o",
        auditor_model="claude-sonnet-4-6",
        code_version="testsha",
        benign_null_samples=benign,
        metric_orientations={"sdi": True},
        target_fpr=0.05,
    )
    assert art.sufficiency.sufficient is False
    with pytest.raises(InsufficientCalibrationError):
        art.guarantee_threshold("sdi")


def test_empirical_null_from_artifact_matches_samples():
    ids, split, art = _build_sufficient_artifact()
    null = art.empirical_null("sdi", higher_is_anomalous=True)
    assert null.n == 60
    # A clearly anomalous SDI=1.0 against an all-zero benign null -> minimal p.
    assert null.p_value(1.0) == pytest.approx(1.0 / 61.0)


# ---------------------------------------------------------------------------
# Artifact v2 — semantic Tier-2 calibration (Platt on mix + conformal tau on benign)
# ---------------------------------------------------------------------------
def _build_artifact_with_semantics(sep: bool = True):
    """Sufficient artifact WITH a semantic mix. ``sep=True`` gives well-separated
    classes (benign divergence low, malicious high); ``sep=False`` overlaps them."""
    ids = _dataset(40)
    split = family_split(ids, calibration_fraction=0.5, seed=5)
    benign = {"sdi": [0.0] * 60, "cii": [0.0] * 60, "vad": [0.0] * 60, "vpr": [1.0] * 60}
    orient = {"sdi": True, "cii": True, "vad": True, "vpr": False}
    if sep:
        ben_scores = [0.05 + 0.001 * i for i in range(60)]     # benign divergence ~0.05-0.11
        mal_scores = [0.80 + 0.001 * i for i in range(60)]     # malicious divergence ~0.80-0.86
    else:
        ben_scores = [0.45 + 0.001 * i for i in range(60)]
        mal_scores = [0.48 + 0.001 * i for i in range(60)]
    scores = ben_scores + mal_scores
    labels = [0] * 60 + [1] * 60
    art = build_artifact(
        scenario_ids=ids, split=split, mesh_model="gpt-5.5", auditor_model="haiku",
        code_version="testsha", benign_null_samples=benign, metric_orientations=orient,
        target_fpr=0.05, seeds={"split": 5, "bootstrap": 0},
        semantic_scores=scores, semantic_labels=labels,
    )
    return ids, split, art


def test_semantic_platt_fitted_when_both_classes_present():
    _, _, art = _build_artifact_with_semantics(sep=True)
    assert art.semantic_platt_fitted is True
    assert art.semantic_platt_a is not None and art.semantic_platt_b is not None
    assert art.semantic_conformal_tau is not None
    assert art.sample_counts["semantic_benign"] == 60
    assert art.sample_counts["semantic_malicious"] == 60


def test_semantic_platt_skipped_when_single_class():
    ids = _dataset(40)
    split = family_split(ids, calibration_fraction=0.5, seed=5)
    benign = {"sdi": [0.0] * 60}
    art = build_artifact(
        scenario_ids=ids, split=split, mesh_model="gpt-5.5", auditor_model="haiku",
        code_version="t", benign_null_samples=benign, metric_orientations={"sdi": True},
        target_fpr=0.05, semantic_scores=[0.1] * 60, semantic_labels=[0] * 60,  # benign only
    )
    assert art.semantic_platt_fitted is False
    assert art.semantic_conformal_tau is None
    assert art.platt_scaler() is None
    # conformal_tau accessor falls back to the supplied default
    assert art.conformal_tau(default=0.5) == 0.5


def test_semantic_scaler_reconstructs_and_roundtrips(tmp_path):
    _, _, art = _build_artifact_with_semantics(sep=True)
    path = tmp_path / "cal_v2.json"
    art.save(str(path))
    loaded = FrozenCalibrationArtifact.load(str(path))
    # v2 fields survive the hash-sealed round-trip
    assert loaded.content_hash() == art.content_hash()
    assert loaded.semantic_platt_a == art.semantic_platt_a
    assert loaded.semantic_conformal_tau == art.semantic_conformal_tau
    scaler = loaded.platt_scaler()
    assert scaler is not None and scaler.is_fitted
    # a high divergence calibrates higher than a low one (monotone in the raw score)
    assert scaler.calibrate(0.85) > scaler.calibrate(0.05)


def test_semantic_conformal_tau_lives_on_calibrated_scale():
    """tau must be the conformal quantile of PLATT-CALIBRATED benign scores, so at most
    alpha of benign calibrated scores exceed it."""
    _, _, art = _build_artifact_with_semantics(sep=True)
    scaler = art.platt_scaler()
    ben_scores = [0.05 + 0.001 * i for i in range(60)]
    ben_calibrated = [scaler.calibrate(s) for s in ben_scores]
    exceed = sum(1 for c in ben_calibrated if c > art.semantic_conformal_tau)
    assert exceed / len(ben_calibrated) <= art.target_fpr + 1e-9


def test_semantic_length_mismatch_raises():
    ids = _dataset(40)
    split = family_split(ids, calibration_fraction=0.5, seed=5)
    with pytest.raises(ValueError, match="same length"):
        build_artifact(
            scenario_ids=ids, split=split, mesh_model="m", auditor_model="a",
            code_version="t", benign_null_samples={"sdi": [0.0] * 60},
            metric_orientations={"sdi": True}, target_fpr=0.05,
            semantic_scores=[0.1, 0.2], semantic_labels=[0, 1, 1],
        )
