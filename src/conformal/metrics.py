from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim != 2:
        raise ValueError("Predicted probabilities must be a 2D array.")
    y_true_onehot = np.zeros_like(y_prob)
    for idx, label in enumerate(y_true):
        y_true_onehot[idx, int(label)] = 1.0
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if y_prob.ndim != 2:
        raise ValueError("Predicted probabilities must be a 2D array.")
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        mask = (confidences >= lo) & (confidences < hi if idx < n_bins - 1 else confidences <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(predictions[mask] == y_true[mask])
        conf = np.mean(confidences[mask])
        ece += np.sum(mask) / total * abs(acc - conf)
    return float(ece)
