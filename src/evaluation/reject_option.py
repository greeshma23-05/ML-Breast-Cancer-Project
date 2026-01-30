from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class RejectMetrics:
    coverage: float
    error_rate: float
    false_negative_rate: float
    threshold: float


def evaluate_reject_option(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    confidence_threshold: float,
) -> RejectMetrics:
    """
    Reject predictions with confidence < confidence_threshold.
    Confidence = max(p, 1-p)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    confidence = np.maximum(y_prob, 1.0 - y_prob)
    keep_mask = confidence >= confidence_threshold

    if keep_mask.sum() == 0:
        return RejectMetrics(0.0, float("nan"), float("nan"), confidence_threshold)

    y_kept = y_true[keep_mask]
    p_kept = y_prob[keep_mask]
    y_pred = (p_kept >= 0.5).astype(int)

    errors = (y_pred != y_kept).mean()

    # False negatives
    fn = ((y_pred == 0) & (y_kept == 1)).sum()
    positives = (y_kept == 1).sum()
    fn_rate = fn / positives if positives > 0 else float("nan")

    coverage = keep_mask.mean()

    return RejectMetrics(
        coverage=float(coverage),
        error_rate=float(errors),
        false_negative_rate=float(fn_rate),
        threshold=float(confidence_threshold),
    )
