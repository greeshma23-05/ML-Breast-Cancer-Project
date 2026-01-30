from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    bin_edges: list[float]
    bin_acc: list[float]
    bin_conf: list[float]
    bin_counts: list[int]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> CalibrationReport:
    """
    Compute ECE and reliability diagram summary stats.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    # Bin by predicted probability
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc = []
    bin_conf = []
    bin_counts = []

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_counts.append(count)

        if count == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            continue

        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        bin_acc.append(acc)
        bin_conf.append(conf)

        ece += (count / n) * abs(acc - conf)

    return CalibrationReport(
        ece=float(ece),
        bin_edges=bin_edges.tolist(),
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        bin_counts=bin_counts,
    )
