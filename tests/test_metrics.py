from __future__ import annotations

import numpy as np

from src.conformal.metrics import accuracy_score, brier_score, expected_calibration_error


def test_metric_helpers_return_expected_ranges() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0], dtype=int)
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.8, 0.2],
        ],
        dtype=float,
    )

    accuracy = accuracy_score(y_true, y_pred)
    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)

    assert accuracy == 0.75
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0
