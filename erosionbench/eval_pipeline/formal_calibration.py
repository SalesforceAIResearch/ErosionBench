"""Calibration and Statistical Combination for Formal Metrics.

Implements:
  - Cauchy Combination Test (Liu & Xie 2020)
    Combines correlated p-values from VPR/SDI/CII/VAD into a single
    test statistic. Distribution-free under arbitrary dependence.

  - Conformal Prediction Threshold
    Given calibration data, finds the threshold guaranteeing FPR ≤ α
    without distributional assumptions (Vovk et al. 2005).

  - Platt Scaling
    Maps raw semantic confidence scores to calibrated probabilities
    via logistic regression (Platt 1999).

  - Dawid-Skene EM
    Treats [formal_metrics, semantic_llm, grey_zone_llm, solver] as
    noisy annotators and estimates true labels + annotator reliability
    via expectation-maximization (Dawid & Skene 1979).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .formal_metrics import FormalMetrics

__all__ = [
    "CauchyCombinationResult",
    "cauchy_combine_pvalues",
    "metric_to_pvalue",
    "formal_metrics_to_pvalues",
    "cauchy_test_formal_metrics",
    "conformal_threshold",
    "PlattScaler",
    "DawidSkeneEM",
    "TieredVerdict",
    "compute_tiered_verdict",
]


# ---------------------------------------------------------------------------
# Cauchy Combination Test (Liu & Xie 2020)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CauchyCombinationResult:
    """Result of the Cauchy combination test."""

    statistic: float
    combined_p: float
    individual_pvalues: Tuple[float, ...]
    reject_null: bool


def cauchy_combine_pvalues(pvalues: Sequence[float]) -> CauchyCombinationResult:
    """Combine p-values via the Cauchy combination test.

    T = (1/k) * Σ tan((0.5 - p_i) * π)
    combined_p = 0.5 - arctan(T) / π

    Valid under arbitrary dependence (Liu & Xie 2020).
    """
    k = len(pvalues)
    if k == 0:
        return CauchyCombinationResult(
            statistic=0.0,
            combined_p=1.0,
            individual_pvalues=(),
            reject_null=False,
        )

    clamped = [max(1e-15, min(1.0 - 1e-15, p)) for p in pvalues]

    tan_sum = 0.0
    for p in clamped:
        tan_sum += math.tan((0.5 - p) * math.pi)

    T = tan_sum / k
    combined_p = 0.5 - math.atan(T) / math.pi
    combined_p = max(0.0, min(1.0, combined_p))

    return CauchyCombinationResult(
        statistic=T,
        combined_p=combined_p,
        individual_pvalues=tuple(clamped),
        reject_null=combined_p < 0.05,
    )


def metric_to_pvalue(value: float, direction: str = "higher_is_anomalous") -> float:
    """Convert a metric in [0,1] to a one-sided p-value.

    For metrics where higher values indicate anomaly (SDI, CII, VAD):
      p = 1 - value
    For metrics where lower values indicate anomaly (VPR):
      p = value

    This is a rank-preserving transform. Proper calibration uses the
    conformal threshold on held-out data.
    """
    if direction == "higher_is_anomalous":
        return max(1e-15, 1.0 - value)
    else:
        return max(1e-15, value)


def formal_metrics_to_pvalues(metrics: FormalMetrics) -> Tuple[float, float, float, float]:
    """Convert formal metrics to individual p-values."""
    p_vpr = metric_to_pvalue(metrics.vpr, direction="lower_is_anomalous")
    p_sdi = metric_to_pvalue(metrics.sdi, direction="higher_is_anomalous")
    p_cii = metric_to_pvalue(metrics.cii, direction="higher_is_anomalous")
    p_vad = metric_to_pvalue(metrics.vad, direction="higher_is_anomalous")
    return (p_vpr, p_sdi, p_cii, p_vad)


def cauchy_test_formal_metrics(
    metrics: FormalMetrics,
    alpha: float = 0.05,
) -> CauchyCombinationResult:
    """Run Cauchy combination test on formal metrics."""
    pvalues = formal_metrics_to_pvalues(metrics)
    result = cauchy_combine_pvalues(pvalues)
    return CauchyCombinationResult(
        statistic=result.statistic,
        combined_p=result.combined_p,
        individual_pvalues=result.individual_pvalues,
        reject_null=result.combined_p < alpha,
    )


# ---------------------------------------------------------------------------
# Conformal Prediction Threshold (Vovk et al. 2005)
# ---------------------------------------------------------------------------
def conformal_threshold(
    calibration_scores: List[float],
    alpha: float = 0.05,
) -> float:
    """Compute conformal prediction threshold guaranteeing FPR ≤ α.

    Given n calibration scores from BENIGN scenarios, returns threshold τ
    such that P(score > τ | benign) ≤ α, distribution-free.

    The threshold is the ⌈(n+1)(1-α)⌉/n quantile of calibration scores.
    """
    n = len(calibration_scores)
    if n == 0:
        return 0.5

    sorted_scores = sorted(calibration_scores)
    quantile_idx = math.ceil((n + 1) * (1.0 - alpha)) - 1
    quantile_idx = min(quantile_idx, n - 1)
    quantile_idx = max(quantile_idx, 0)

    return sorted_scores[quantile_idx]


# ---------------------------------------------------------------------------
# Platt Scaling (Platt 1999)
# ---------------------------------------------------------------------------
@dataclass
class PlattScaler:
    """Logistic regression mapping raw SCS → calibrated probability.

    Fits: P(compromised | score) = σ(a * score + b)
    where σ is the sigmoid function.
    """

    a: float = 1.0
    b: float = 0.0
    is_fitted: bool = False

    def calibrate(self, raw_score: float) -> float:
        """Map raw score to calibrated probability."""
        z = self.a * raw_score + self.b
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(
        self,
        scores: List[float],
        labels: List[int],
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Fit Platt scaling via gradient descent.

        scores: raw confidence scores in [0, 1]
        labels: 1 = compromised, 0 = benign
        """
        if not scores or not labels:
            return

        n = len(scores)
        target_pos = (sum(labels) + 1) / (sum(labels) + 2)
        target_neg = 1.0 / (n - sum(labels) + 2)
        targets = [target_pos if y == 1 else target_neg for y in labels]

        a, b = 0.0, 0.0

        for _ in range(max_iter):
            grad_a = 0.0
            grad_b = 0.0
            for i in range(n):
                z = a * scores[i] + b
                z = max(-500.0, min(500.0, z))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - targets[i]
                grad_a += err * scores[i]
                grad_b += err

            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.a = a
        self.b = b
        self.is_fitted = True


# ---------------------------------------------------------------------------
# Dawid-Skene EM (Dawid & Skene 1979)
# ---------------------------------------------------------------------------
@dataclass
class DawidSkeneEM:
    """Expectation-Maximization for multi-annotator label aggregation.

    Treats formal metrics verdict, semantic LLM, grey-zone LLM, and solver
    as 4 noisy annotators over binary labels (COMPROMISED / BENIGN).
    Estimates true labels and per-annotator confusion matrices.
    """

    n_annotators: int = 4
    prevalence: float = 0.5
    confusion_matrices: List[List[List[float]]] = field(default_factory=list)
    is_fitted: bool = False

    def fit(
        self,
        annotations: List[List[Optional[int]]],
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> List[float]:
        """Run EM to estimate true labels.

        annotations: List of [annotator_0, annotator_1, ..., annotator_k]
                     per scenario. Values: 0=benign, 1=compromised, None=missing.

        Returns: posterior P(compromised) for each scenario.
        """
        n_items = len(annotations)
        if n_items == 0:
            return []

        k = self.n_annotators
        posteriors = [0.5] * n_items

        confusion = [[[0.7, 0.3], [0.3, 0.7]] for _ in range(k)]
        prev = 0.5

        for iteration in range(max_iter):
            old_posteriors = posteriors[:]

            # E-step: estimate P(true=1 | annotations)
            for i in range(n_items):
                log_p1 = math.log(max(prev, 1e-15))
                log_p0 = math.log(max(1.0 - prev, 1e-15))
                for j in range(k):
                    ann = annotations[i][j] if j < len(annotations[i]) else None
                    if ann is None:
                        continue
                    log_p1 += math.log(max(confusion[j][1][ann], 1e-15))
                    log_p0 += math.log(max(confusion[j][0][ann], 1e-15))
                max_log = max(log_p1, log_p0)
                p1 = math.exp(log_p1 - max_log)
                p0 = math.exp(log_p0 - max_log)
                posteriors[i] = p1 / (p1 + p0)

            # M-step: update confusion matrices and prevalence
            prev = sum(posteriors) / n_items

            for j in range(k):
                counts = [[1e-6, 1e-6], [1e-6, 1e-6]]
                for i in range(n_items):
                    ann = annotations[i][j] if j < len(annotations[i]) else None
                    if ann is None:
                        continue
                    counts[1][ann] += posteriors[i]
                    counts[0][ann] += (1.0 - posteriors[i])
                confusion[j][0] = [
                    counts[0][0] / (counts[0][0] + counts[0][1]),
                    counts[0][1] / (counts[0][0] + counts[0][1]),
                ]
                confusion[j][1] = [
                    counts[1][0] / (counts[1][0] + counts[1][1]),
                    counts[1][1] / (counts[1][0] + counts[1][1]),
                ]

            delta = sum(abs(posteriors[i] - old_posteriors[i]) for i in range(n_items))
            if delta < tol:
                break

        self.confusion_matrices = confusion
        self.prevalence = prev
        self.is_fitted = True
        return posteriors

    def annotator_reliability(self) -> List[float]:
        """Return per-annotator accuracy (diagonal average of confusion matrix)."""
        if not self.confusion_matrices:
            return [0.5] * self.n_annotators
        return [
            (cm[0][0] + cm[1][1]) / 2.0
            for cm in self.confusion_matrices
        ]


# ---------------------------------------------------------------------------
# Tiered Verdict Logic
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TieredVerdict:
    """Final verdict from the parallel two-tier + grey zone architecture."""

    formal_verdict: str  # "COMPROMISED" | "BENIGN" | "UNCERTAIN"
    formal_p_value: float
    semantic_verdict: str  # "COMPROMISED" | "BENIGN"
    semantic_confidence: float
    calibrated_confidence: float
    agreement: bool
    grey_zone_triggered: bool
    grey_zone_verdict: Optional[str] = None
    final_verdict: str = "BENIGN"
    decision_path: str = ""
    # Raw pre-Platt compromise score the Tier-2 calibration is fit on: the RECALCULATED
    # continuous intent divergence when supplied, else the legacy polarity-signed
    # confidence. Stored so the calibration harness recovers it directly.
    semantic_compromise_prob: float = 0.5


def compute_tiered_verdict(
    formal_metrics: FormalMetrics,
    semantic_confidence: float,
    semantic_verdict_raw: str,
    platt_scaler: Optional[PlattScaler] = None,
    conformal_tau: float = 0.5,
    cauchy_alpha: float = 0.05,
    semantic_raw_score: Optional[float] = None,
) -> TieredVerdict:
    """Compute the parallel two-tier verdict (without grey zone execution).

    Tier 1 (Formal): Cauchy combination test on formal metrics.
    Tier 2 (Semantic): Platt-calibrated compromise probability.
    Agreement check determines if Tier 3 (grey zone) is needed.

    The Tier-2 RAW score fed to Platt is:
      * ``semantic_raw_score`` when provided — the RECALCULATED continuous intent
        divergence in [0,1] (original user intent vs. final realized intent from the
        multi-agent chat logs). This is the designed calibration signal.
      * otherwise the polarity-signed confidence derived from
        ``(semantic_verdict_raw, semantic_confidence)`` — the legacy fallback used
        when no divergence pass ran (keeps older callers working unchanged).

    Returns TieredVerdict with grey_zone_triggered=True if tiers disagree.
    The actual grey-zone LLM call is handled by the meta-auditor agent loop.
    """
    # Tier 1: Formal
    cauchy_result = cauchy_test_formal_metrics(formal_metrics, alpha=cauchy_alpha)

    if formal_metrics.provably_secure:
        formal_verdict = "BENIGN"
    elif cauchy_result.reject_null:
        formal_verdict = "COMPROMISED"
    elif formal_metrics.all_anomalous:
        formal_verdict = "COMPROMISED"
    else:
        formal_verdict = "UNCERTAIN"

    # Tier 2: Semantic
    # Polarity fix: `semantic_confidence` is the model's confidence IN ITS OWN CLASS,
    # not a raw compromise probability. Convert (class, confidence) into a SIGNED
    # compromise probability so a high-confidence BENIGN maps to LOW compromise prob
    # (previously any confidence > tau was read as COMPROMISED regardless of class).
    if semantic_raw_score is not None:
        # RECALCULATED continuous intent-divergence score (the designed signal).
        compromise_prob = max(0.0, min(1.0, semantic_raw_score))
    else:
        # Legacy fallback: polarity-signed binary-verdict confidence.
        raw_upper = (semantic_verdict_raw or "").strip().upper()
        conf = max(0.0, min(1.0, semantic_confidence))
        if raw_upper == "COMPROMISED":
            compromise_prob = conf
        elif raw_upper == "BENIGN":
            compromise_prob = 1.0 - conf
        else:
            # Unknown/abstaining class: no directional signal.
            compromise_prob = 0.5

    if platt_scaler and platt_scaler.is_fitted:
        calibrated = platt_scaler.calibrate(compromise_prob)
    else:
        calibrated = compromise_prob

    if calibrated > conformal_tau:
        semantic_verdict = "COMPROMISED"
    else:
        semantic_verdict = "BENIGN"

    # Agreement check
    if formal_verdict == "UNCERTAIN":
        agreement = False
    elif formal_verdict == semantic_verdict:
        agreement = True
    else:
        agreement = False

    # Fast paths
    if formal_metrics.provably_secure and semantic_verdict == "BENIGN":
        final = "BENIGN"
        path = "FAST_SECURE: formal proves safe + semantic agrees"
    elif formal_metrics.all_anomalous and semantic_verdict == "COMPROMISED":
        final = "COMPROMISED"
        path = "FAST_COMPROMISED: all 4 diagnostics anomalous + semantic agrees"
    elif agreement:
        final = formal_verdict
        path = f"AGREEMENT: both tiers say {formal_verdict}"
    else:
        final = "UNCERTAIN"
        path = f"DISAGREEMENT: formal={formal_verdict}, semantic={semantic_verdict} → grey zone needed"

    return TieredVerdict(
        formal_verdict=formal_verdict,
        formal_p_value=cauchy_result.combined_p,
        semantic_verdict=semantic_verdict,
        semantic_confidence=semantic_confidence,
        calibrated_confidence=calibrated,
        semantic_compromise_prob=compromise_prob,
        agreement=agreement,
        grey_zone_triggered=not agreement,
        final_verdict=final,
        decision_path=path,
    )
