from __future__ import annotations

import numpy as np

from src.conformal.icp import SplitConformalClassifier


def test_split_conformal_classifier_fits_and_predicts() -> None:
    X_cal = np.zeros((10, 3), dtype=float)
    y_cal = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    y_prob_cal = np.array(
        [
            [0.95, 0.05], [0.10, 0.90], [0.85, 0.15], [0.15, 0.85], [0.80, 0.20],
            [0.20, 0.80], [0.90, 0.10], [0.10, 0.90], [0.88, 0.12], [0.12, 0.88],
        ],
        dtype=float,
    )

    conformal = SplitConformalClassifier(alpha=0.1).fit(X_cal, y_cal, y_prob_cal)
    assert conformal.threshold_ is not None

    y_prob_test = np.array(
        [
            [0.99, 0.01],
            [0.04, 0.96],
            [0.70, 0.30],
            [0.45, 0.55],
        ],
        dtype=float,
    )
    prediction_sets = conformal.predict(np.zeros((4, 3)), y_prob_test)
    assert len(prediction_sets) == 4
    assert np.all([len(prediction_set) >= 1 for prediction_set in prediction_sets])


def test_conformal_coverage_is_between_zero_and_one() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=int)
    prediction_sets = np.array([
        np.array([0]),
        np.array([1, 0]),
        np.array([1]),
        np.array([0, 1]),
    ], dtype=object)

    conformal = SplitConformalClassifier(alpha=0.1)
    coverage = conformal.coverage(y_true, prediction_sets)
    assert 0.0 <= coverage <= 1.0
