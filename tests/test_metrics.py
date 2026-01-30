import numpy as np
from src.evaluation.metrics import evaluate_binary


def test_evaluate_binary_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.10, 0.90, 0.99])

    m = evaluate_binary(y_true, y_prob, threshold=0.5)

    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert m.confusion_matrix == [[2, 0], [0, 2]]
    assert m.sensitivity == 1.0
    assert m.specificity == 1.0
