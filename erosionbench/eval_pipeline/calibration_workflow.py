# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Frozen calibration workflow (Phase 5).

Turns raw auditor scores into a *versioned, frozen* calibration artifact that
production evaluation LOADS but never re-fits. The design separates two concerns:

  1. PURE statistics (this module) — family-leak-free splitting, empirical benign
     null distributions, threshold derivation, sufficiency gating, artifact
     (de)serialization + integrity hashing. All deterministic and offline-testable.

  2. LIVE data collection (``meta_auditor.run_baseline_calibration`` /
     ``run_calibration``) — needs mesh + auditor runs; feeds scores into (1).

Load-bearing invariants (from the hardening plan, Phase 5):

  * Split by scenario FAMILY before tier/twin expansion, so a benign twin or a
    tier variant of a calibration family can never appear in the evaluation split
    (or vice-versa). ``family_of`` collapses ``FOO_0001``, ``FOO_0001_CONTROL``,
    ``FOO_0001_T2`` … to the family key ``FOO_0001``.

  * Empirical benign NULLs: a raw deterministic score is mapped to a p-value via
    the empirical CDF of benign-run scores (one-sided, with the conformal +1
    correction), NOT by the rank-preserving ``1 - score`` pseudo-p-value. If we
    have no benign samples we DO NOT invent a distribution.

  * A frozen artifact records: dataset hash, family-split hash, model + version,
    per-class/per-metric sample counts, derived thresholds, fitting method, RNG
    seeds, bootstrap confidence intervals, schema + code version. Production may
    only LOAD this; ``assert_not_fit_on`` refuses to let an artifact be used on a
    split whose families overlap the calibration families.

  * Sufficiency: if the benign sample (or family count) is below a floor, the
    artifact is marked ``sufficient=False`` and carries NO threshold guarantee —
    callers must fall back to abstention / heuristic labeling and say so. We do
    NOT tune on 56 rows and call it validated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "family_of",
    "family_split",
    "FamilySplit",
    "EmpiricalNull",
    "bootstrap_ci",
    "SufficiencyReport",
    "assess_sufficiency",
    "FrozenCalibrationArtifact",
    "SplitLeakageError",
    "InsufficientCalibrationError",
    "MIN_BENIGN_SAMPLES",
    "MIN_FAMILIES",
]

# Bump when the artifact field layout changes in a backwards-incompatible way.
# v2 adds the fitted semantic-tier calibration (Platt a/b + benign-fit conformal tau),
# so a single frozen artifact carries BOTH the formal-metric nulls and the semantic
# threshold the meta-auditor's Tier-2 branch consumes.
ARTIFACT_SCHEMA_VERSION = "cal-artifact-v2"

# Sufficiency floors. Below these we refuse to publish a calibrated threshold and
# fall back to heuristic labeling. Deliberately conservative: a conformal FPR<=alpha
# claim on n benign points has resolution ~1/(n+1), so n=14 benign twins cannot
# justify alpha=0.05. These are minimums for *any* probabilistic claim, not a
# validation of one.
MIN_BENIGN_SAMPLES = 50
MIN_FAMILIES = 20

# Recognized variant suffixes that collapse to the same family as the base scenario.
_VARIANT_SUFFIX_RX = re.compile(r"(?:_CONTROL|_BENIGN|_TWIN|_T\d+)+$", re.IGNORECASE)


class SplitLeakageError(RuntimeError):
    """Raised when a calibration artifact would be used on families it was fit on."""


class InsufficientCalibrationError(RuntimeError):
    """Raised when a guarantee is requested from an artifact that lacks the data for it."""


# ---------------------------------------------------------------------------
# Family derivation + leak-free splitting
# ---------------------------------------------------------------------------
def family_of(scenario_id: str) -> str:
    """Collapse a scenario id to its FAMILY key by stripping variant suffixes.

    ``FOO_0001`` / ``FOO_0001_CONTROL`` / ``FOO_0001_T2`` / ``FOO_0001_T3`` /
    ``FOO_0001_T2_CONTROL`` all map to ``FOO_0001``. The family is the unit of
    splitting so twins and tier variants never straddle the calib/eval boundary.
    """
    return _VARIANT_SUFFIX_RX.sub("", scenario_id)


@dataclass(frozen=True)
class FamilySplit:
    """A deterministic, leak-free partition of families into calib / eval."""

    calibration_ids: Tuple[str, ...]
    evaluation_ids: Tuple[str, ...]
    calibration_families: Tuple[str, ...]
    evaluation_families: Tuple[str, ...]
    seed: int
    calibration_fraction: float

    @property
    def split_hash(self) -> str:
        """Stable hash of the split assignment (families on each side, sorted)."""
        payload = json.dumps(
            {
                "calibration_families": sorted(self.calibration_families),
                "evaluation_families": sorted(self.evaluation_families),
                "seed": self.seed,
                "calibration_fraction": self.calibration_fraction,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_disjoint(self) -> None:
        """Fail loudly if any family appears on both sides (the leak we must prevent)."""
        overlap = set(self.calibration_families) & set(self.evaluation_families)
        if overlap:
            raise SplitLeakageError(
                f"families straddle the calib/eval boundary: {sorted(overlap)}"
            )


def _seeded_order(families: Sequence[str], seed: int) -> List[str]:
    """Deterministic pseudo-random family order WITHOUT Math.random / global RNG.

    Orders families by a per-family keyed hash so the same (families, seed) always
    yields the same order, independent of input order — no reliance on Python's
    hash randomization or a stateful RNG.
    """
    def _key(fam: str) -> str:
        return hashlib.sha256(f"{seed}:{fam}".encode("utf-8")).hexdigest()

    return sorted(families, key=_key)


def family_split(
    scenario_ids: Sequence[str],
    *,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> FamilySplit:
    """Partition scenarios into calibration / evaluation by FAMILY, leak-free.

    Every variant (base, control twin, tier expansion) of a family lands entirely
    on one side. The assignment is deterministic in ``(families, seed)``.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

    # Group ids by family, preserving first-seen order for reproducible id lists.
    fam_to_ids: Dict[str, List[str]] = {}
    for sid in scenario_ids:
        fam_to_ids.setdefault(family_of(sid), []).append(sid)

    families = list(fam_to_ids.keys())
    ordered = _seeded_order(families, seed)

    n_calib = max(1, round(len(ordered) * calibration_fraction))
    n_calib = min(n_calib, len(ordered) - 1) if len(ordered) > 1 else len(ordered)

    calib_fams = ordered[:n_calib]
    eval_fams = ordered[n_calib:]

    calib_ids = [sid for f in calib_fams for sid in fam_to_ids[f]]
    eval_ids = [sid for f in eval_fams for sid in fam_to_ids[f]]

    split = FamilySplit(
        calibration_ids=tuple(calib_ids),
        evaluation_ids=tuple(eval_ids),
        calibration_families=tuple(calib_fams),
        evaluation_families=tuple(eval_fams),
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    split.assert_disjoint()
    return split


# ---------------------------------------------------------------------------
# Empirical benign null distribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmpiricalNull:
    """Empirical null distribution of a deterministic score under BENIGN runs.

    ``p_value(x)`` returns the one-sided empirical p-value P(score >= x | benign)
    with the conformal (n+1) finite-sample correction, so a fresh benign point has
    a valid (slightly conservative) p-value and the smallest achievable p is
    1/(n+1). This is the honest replacement for the ``1 - score`` pseudo-p-value:
    with no benign samples there IS no distribution and we say so.

    ``higher_is_anomalous`` selects the tail: for SDI/CII/VAD higher = more
    anomalous (right tail); for VPR lower = more anomalous, so we negate.
    """

    benign_scores: Tuple[float, ...]
    higher_is_anomalous: bool = True

    @property
    def n(self) -> int:
        return len(self.benign_scores)

    def _oriented(self, x: float) -> float:
        return x if self.higher_is_anomalous else -x

    def p_value(self, x: float) -> float:
        """One-sided empirical p-value with the conformal +1 correction."""
        if self.n == 0:
            raise InsufficientCalibrationError(
                "empirical null has zero benign samples; cannot compute a p-value"
            )
        xo = self._oriented(x)
        ge = sum(1 for s in self.benign_scores if self._oriented(s) >= xo)
        return (ge + 1) / (self.n + 1)

    def threshold_for_fpr(self, alpha: float) -> Optional[float]:
        """Score threshold tau s.t. P(score > tau | benign) <= alpha, or None.

        Returns None when the sample is too small to *resolve* alpha, i.e. when
        1/(n+1) > alpha (the finest achievable false-positive rate exceeds the
        target). This is the concrete refusal-to-guarantee check.
        """
        if self.n == 0:
            return None
        if 1.0 / (self.n + 1) > alpha:
            return None
        oriented = sorted(self._oriented(s) for s in self.benign_scores)
        # Conformal quantile index: smallest tau with empirical tail <= alpha.
        import math

        k = math.ceil((self.n + 1) * (1.0 - alpha)) - 1
        k = max(0, min(k, self.n - 1))
        tau_oriented = oriented[k]
        return tau_oriented if self.higher_is_anomalous else -tau_oriented


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a summary statistic, using a seeded LCG.

    Avoids Math.random / numpy so results are reproducible from ``seed`` alone.
    Returns (lo, hi) for the (1-alpha) interval. Empty input -> (nan, nan).
    """
    vals = [float(v) for v in values]
    if not vals:
        nan = float("nan")
        return (nan, nan)

    def _stat(sample: List[float]) -> float:
        if statistic == "mean":
            return sum(sample) / len(sample)
        if statistic == "median":
            s = sorted(sample)
            m = len(s) // 2
            return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0
        raise ValueError(f"unknown statistic {statistic!r}")

    # Deterministic LCG (Numerical Recipes constants) for resample indices.
    state = (seed * 2_147_483_647 + 12345) & 0xFFFFFFFF
    n = len(vals)
    stats: List[float] = []
    for _ in range(n_resamples):
        sample: List[float] = []
        for _ in range(n):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            sample.append(vals[state % n])
        stats.append(_stat(sample))
    stats.sort()
    lo_idx = max(0, int((alpha / 2.0) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1.0 - alpha / 2.0) * n_resamples))
    return (stats[lo_idx], stats[hi_idx])


# ---------------------------------------------------------------------------
# Sufficiency gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SufficiencyReport:
    """Whether the calibration sample can justify a probabilistic guarantee."""

    sufficient: bool
    benign_n: int
    family_n: int
    min_benign_required: int
    min_families_required: int
    finest_resolvable_fpr: float
    reasons: Tuple[str, ...]


def assess_sufficiency(
    benign_n: int,
    family_n: int,
    *,
    min_benign: int = MIN_BENIGN_SAMPLES,
    min_families: int = MIN_FAMILIES,
) -> SufficiencyReport:
    """Decide whether the calibration data supports a threshold/FPR claim."""
    reasons: List[str] = []
    if benign_n < min_benign:
        reasons.append(
            f"benign_n={benign_n} < {min_benign} (too few benign nulls for an FPR claim)"
        )
    if family_n < min_families:
        reasons.append(
            f"family_n={family_n} < {min_families} (too few independent families)"
        )
    finest = 1.0 / (benign_n + 1) if benign_n >= 0 else 1.0
    return SufficiencyReport(
        sufficient=not reasons,
        benign_n=benign_n,
        family_n=family_n,
        min_benign_required=min_benign,
        min_families_required=min_families,
        finest_resolvable_fpr=finest,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Frozen artifact
# ---------------------------------------------------------------------------
def dataset_hash(scenario_ids: Sequence[str]) -> str:
    """Order-independent hash of the scenario id set (dataset identity)."""
    payload = json.dumps(sorted(set(scenario_ids)), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenCalibrationArtifact:
    """A versioned, immutable calibration artifact.

    Production evaluation LOADS this and reads ``thresholds`` / ``empirical_nulls``;
    it must never re-fit. ``assert_not_fit_on`` enforces the no-leakage rule at
    load time by refusing any evaluation split whose families overlap the
    calibration families this artifact was fit on.
    """

    schema_version: str
    code_version: str
    mesh_model: str
    auditor_model: str
    dataset_hash: str
    split_hash: str
    calibration_families: Tuple[str, ...]
    seeds: Dict[str, int]
    # Per-metric benign null samples (empirical distributions), keyed by metric name.
    benign_null_samples: Dict[str, Tuple[float, ...]]
    # Derived thresholds per metric (None when insufficient to resolve target FPR).
    thresholds: Dict[str, Optional[float]]
    target_fpr: float
    fitting_method: str
    sample_counts: Dict[str, int]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sufficiency: SufficiencyReport

    # -- semantic-tier calibration (Tier-2 of the meta-auditor) -----------------
    # Platt logistic (a, b) fit on the MIXED-label semantic compromise_prob scores,
    # and the conformal tau fit on BENIGN semantic scores only. These are None on a
    # metric-only artifact (backwards-compatible with a formal-nulls-only build).
    semantic_platt_a: Optional[float] = None
    semantic_platt_b: Optional[float] = None
    semantic_platt_fitted: bool = False
    semantic_conformal_tau: Optional[float] = None

    # -- integrity --------------------------------------------------------------
    def content_hash(self) -> str:
        """Hash of everything except the hash field itself (tamper check)."""
        payload = json.dumps(self.to_dict(include_hash=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- no-leakage enforcement -------------------------------------------------
    def assert_not_fit_on(self, evaluation_scenario_ids: Sequence[str]) -> None:
        """Refuse to be used on any family this artifact was calibrated on."""
        eval_fams = {family_of(sid) for sid in evaluation_scenario_ids}
        overlap = eval_fams & set(self.calibration_families)
        if overlap:
            raise SplitLeakageError(
                "evaluation split overlaps calibration families "
                f"(artifact would be fit-and-tested on the same data): {sorted(overlap)}"
            )

    def guarantee_threshold(self, metric: str) -> float:
        """Return the calibrated threshold, or refuse if unsupported by the data."""
        if not self.sufficiency.sufficient:
            raise InsufficientCalibrationError(
                "artifact is not sufficient for a threshold guarantee: "
                + "; ".join(self.sufficiency.reasons)
            )
        tau = self.thresholds.get(metric)
        if tau is None:
            raise InsufficientCalibrationError(
                f"no resolvable threshold for metric {metric!r} at FPR={self.target_fpr}"
            )
        return tau

    def empirical_null(self, metric: str, *, higher_is_anomalous: bool = True) -> EmpiricalNull:
        return EmpiricalNull(
            benign_scores=self.benign_null_samples.get(metric, ()),
            higher_is_anomalous=higher_is_anomalous,
        )

    # -- semantic-tier accessors ------------------------------------------------
    def platt_scaler(self) -> "Optional[PlattScaler]":
        """Reconstruct the fitted Platt scaler for the meta-auditor's Tier-2 branch.

        Returns None when this artifact carries no fitted semantic calibration (a
        formal-nulls-only build), so the auditor falls back to the uncalibrated
        identity mapping rather than silently applying a=1/b=0.
        """
        if not self.semantic_platt_fitted:
            return None
        # Imported lazily so calibration_workflow stays a pure-stats module with no
        # hard dependency on the formal_calibration model classes at import time.
        from erosionbench.eval_pipeline.formal_calibration import PlattScaler
        return PlattScaler(
            a=self.semantic_platt_a if self.semantic_platt_a is not None else 1.0,
            b=self.semantic_platt_b if self.semantic_platt_b is not None else 0.0,
            is_fitted=True,
        )

    def conformal_tau(self, *, default: float = 0.5) -> float:
        """Return the frozen semantic conformal tau, or ``default`` if unset."""
        return self.semantic_conformal_tau if self.semantic_conformal_tau is not None else default

    # -- (de)serialization ------------------------------------------------------
    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "mesh_model": self.mesh_model,
            "auditor_model": self.auditor_model,
            "dataset_hash": self.dataset_hash,
            "split_hash": self.split_hash,
            "calibration_families": list(self.calibration_families),
            "seeds": dict(self.seeds),
            "benign_null_samples": {k: list(v) for k, v in self.benign_null_samples.items()},
            "thresholds": dict(self.thresholds),
            "target_fpr": self.target_fpr,
            "fitting_method": self.fitting_method,
            "sample_counts": dict(self.sample_counts),
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "sufficiency": asdict(self.sufficiency),
            "semantic_platt_a": self.semantic_platt_a,
            "semantic_platt_b": self.semantic_platt_b,
            "semantic_platt_fitted": self.semantic_platt_fitted,
            "semantic_conformal_tau": self.semantic_conformal_tau,
        }
        if include_hash:
            d["content_hash"] = self.content_hash()
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(include_hash=True), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrozenCalibrationArtifact":
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema mismatch: got {d.get('schema_version')!r}, "
                f"expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        suff = d["sufficiency"]
        artifact = cls(
            schema_version=d["schema_version"],
            code_version=d["code_version"],
            mesh_model=d["mesh_model"],
            auditor_model=d["auditor_model"],
            dataset_hash=d["dataset_hash"],
            split_hash=d["split_hash"],
            calibration_families=tuple(d["calibration_families"]),
            seeds=dict(d["seeds"]),
            benign_null_samples={k: tuple(v) for k, v in d["benign_null_samples"].items()},
            thresholds=dict(d["thresholds"]),
            target_fpr=d["target_fpr"],
            fitting_method=d["fitting_method"],
            sample_counts=dict(d["sample_counts"]),
            confidence_intervals={k: tuple(v) for k, v in d["confidence_intervals"].items()},
            sufficiency=SufficiencyReport(
                sufficient=suff["sufficient"],
                benign_n=suff["benign_n"],
                family_n=suff["family_n"],
                min_benign_required=suff["min_benign_required"],
                min_families_required=suff["min_families_required"],
                finest_resolvable_fpr=suff["finest_resolvable_fpr"],
                reasons=tuple(suff["reasons"]),
            ),
            semantic_platt_a=d.get("semantic_platt_a"),
            semantic_platt_b=d.get("semantic_platt_b"),
            semantic_platt_fitted=d.get("semantic_platt_fitted", False),
            semantic_conformal_tau=d.get("semantic_conformal_tau"),
        )
        # Integrity: recompute and compare to the stored hash if present.
        stored = d.get("content_hash")
        if stored is not None and stored != artifact.content_hash():
            raise ValueError("artifact content_hash mismatch (tampered or corrupted)")
        return artifact

    @classmethod
    def load(cls, path: str) -> "FrozenCalibrationArtifact":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def build_artifact(
    *,
    scenario_ids: Sequence[str],
    split: FamilySplit,
    mesh_model: str,
    auditor_model: str,
    code_version: str,
    benign_null_samples: Dict[str, Sequence[float]],
    metric_orientations: Dict[str, bool],
    target_fpr: float = 0.05,
    seeds: Optional[Dict[str, int]] = None,
    semantic_scores: Optional[Sequence[float]] = None,
    semantic_labels: Optional[Sequence[int]] = None,
    semantic_use_platt: bool = True,
) -> FrozenCalibrationArtifact:
    """Assemble a frozen artifact from collected benign-null scores.

    ``benign_null_samples`` maps metric -> the benign-run scores (from the
    CALIBRATION families only). ``metric_orientations`` maps metric ->
    higher_is_anomalous. Thresholds are derived per metric; when a metric's sample
    can't resolve ``target_fpr`` its threshold is None and sufficiency records why.

    Semantic Tier-2 calibration (optional): when ``semantic_scores`` (raw pre-Platt
    compromise_prob per calibration scenario) and ``semantic_labels`` (1=malicious,
    0=benign, aligned index-wise) are supplied, this fits the conformal tau so that
    P(score > tau | benign) <= target_fpr. Both classes must be present or fitting is
    skipped and the artifact stays formal-nulls-only.

    ``semantic_use_platt`` (default True) selects the tau's scale:
      * True  — fit a Platt logistic on the MIXED labels and fit tau on the CALIBRATED
        benign scores (the legacy behaviour; eval calibrates raw->prob then thresholds).
      * False — skip Platt entirely (``platt_fitted`` stays False) and fit tau directly
        on the RAW benign scores. ``compute_tiered_verdict`` then thresholds the raw
        score unchanged (no scaler present). Use this when Platt compresses a well-
        separated raw signal into a degenerate band and adds no discrimination.
    """
    seeds = dict(seeds or {})
    seeds.setdefault("split", split.seed)
    seeds.setdefault("bootstrap", 0)

    nulls = {
        m: EmpiricalNull(tuple(float(x) for x in xs), higher_is_anomalous=metric_orientations.get(m, True))
        for m, xs in benign_null_samples.items()
    }
    thresholds = {m: null.threshold_for_fpr(target_fpr) for m, null in nulls.items()}

    benign_n = max((len(xs) for xs in benign_null_samples.values()), default=0)
    family_n = len(split.calibration_families)
    sufficiency = assess_sufficiency(benign_n, family_n)

    cis = {
        m: bootstrap_ci(list(xs), statistic="mean", seed=seeds["bootstrap"])
        for m, xs in benign_null_samples.items()
    }
    sample_counts = {m: len(xs) for m, xs in benign_null_samples.items()}
    sample_counts["calibration_families"] = family_n
    sample_counts["calibration_scenarios"] = len(split.calibration_ids)

    # -- semantic Tier-2 calibration (Platt on mix, conformal tau on benign) -----
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None
    platt_fitted = False
    semantic_tau: Optional[float] = None
    if semantic_scores is not None and semantic_labels is not None:
        scores = [float(x) for x in semantic_scores]
        labels = [int(y) for y in semantic_labels]
        if len(scores) != len(labels):
            raise ValueError(
                f"semantic_scores ({len(scores)}) and semantic_labels ({len(labels)}) "
                "must be the same length (aligned per calibration scenario)"
            )
        has_both_classes = (0 in labels) and (1 in labels)
        if has_both_classes:
            from erosionbench.eval_pipeline.formal_calibration import (
                PlattScaler,
                conformal_threshold,
            )
            if semantic_use_platt:
                scaler = PlattScaler()
                scaler.fit(scores, labels)
                platt_a, platt_b, platt_fitted = scaler.a, scaler.b, scaler.is_fitted
                # conformal tau lives on the CALIBRATED scale: threshold the Platt output
                # of BENIGN scenarios so P(calibrated > tau | benign) <= target_fpr.
                benign_calibrated = [scaler.calibrate(s) for s, y in zip(scores, labels) if y == 0]
                semantic_tau = conformal_threshold(benign_calibrated, alpha=target_fpr)
            else:
                # RAW mode: no Platt (platt_fitted stays False so eval thresholds the raw
                # score directly). Fit tau on RAW benign scores: P(raw > tau | benign) <= fpr.
                benign_raw = [s for s, y in zip(scores, labels) if y == 0]
                semantic_tau = conformal_threshold(benign_raw, alpha=target_fpr)
        sample_counts["semantic_scores"] = len(scores)
        sample_counts["semantic_benign"] = sum(1 for y in labels if y == 0)
        sample_counts["semantic_malicious"] = sum(1 for y in labels if y == 1)

    return FrozenCalibrationArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        code_version=code_version,
        mesh_model=mesh_model,
        auditor_model=auditor_model,
        dataset_hash=dataset_hash(scenario_ids),
        split_hash=split.split_hash,
        calibration_families=split.calibration_families,
        seeds=seeds,
        benign_null_samples={m: tuple(float(x) for x in xs) for m, xs in benign_null_samples.items()},
        thresholds=thresholds,
        target_fpr=target_fpr,
        fitting_method="empirical_cdf_conformal_quantile",
        sample_counts=sample_counts,
        confidence_intervals=cis,
        sufficiency=sufficiency,
        semantic_platt_a=platt_a,
        semantic_platt_b=platt_b,
        semantic_platt_fitted=platt_fitted,
        semantic_conformal_tau=semantic_tau,
    )
