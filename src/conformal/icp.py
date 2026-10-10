from __future__ import annotations

from typing import Any

import numpy as np


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.1, method: str = "adaptive") -> None:
        self.alpha = alpha
        self.method = method
        self.threshold_ = None
        self.classes_ = None

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, y_prob_cal: np.ndarray) -> "SplitConformalClassifier":
        if X_cal.shape[0] != y_cal.shape[0] or X_cal.shape[0] != y_prob_cal.shape[0]:
            raise ValueError("Calibration inputs must have consistent lengths.")

        self.classes_ = np.unique(y_cal)
        scores = self._nonconformity_scores(y_cal, y_prob_cal)
        q_level = np.ceil((1 - self.alpha) * len(scores)) / len(scores)
        self.threshold_ = np.quantile(scores, q_level, method="higher")
        return self

    def _nonconformity_scores(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        if y_prob.ndim != 2:
            raise ValueError("Predicted probabilities must be a 2D array with one column per class.")
        y_true_idx = np.asarray(y_true, dtype=int)
        true_class_probs = y_prob[np.arange(len(y_true_idx)), y_true_idx]
        scores = 1.0 - true_class_probs
        return scores

    def predict(self, X: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise ValueError("The conformal classifier must be fit before prediction.")
        if y_prob.ndim != 2:
            raise ValueError("Predicted probabilities must be a 2D array with one column per class.")

        valid = np.sum(y_prob, axis=1) > 0
        set_sizes = []
        predictions = []
        for i, proba in enumerate(y_prob):
            if not valid[i]:
                predictions.append(np.array([], dtype=int))
                set_sizes.append(0)
                continue
            candidate_classes = np.where(proba >= 1.0 - self.threshold_)[0]
            if candidate_classes.size == 0:
                candidate_classes = np.array([np.argmax(proba)])
            predictions.append(candidate_classes)
            set_sizes.append(candidate_classes.size)
        return np.asarray(predictions, dtype=object)

    def coverage(self, y_true: np.ndarray, prediction_sets: np.ndarray) -> float:
        coverages = []
        for actual, prediction_set in zip(y_true, prediction_sets):
            coverages.append(int(actual in prediction_set))
        return float(np.mean(coverages)) if coverages else 0.0
