from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.isotonic import IsotonicRegression


def calibrate_predictions(model: Any, X_cal: Any, y_cal: Any, method: str = "sigmoid") -> Any:
    method = method.lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unsupported calibration method: {method}")

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
    )
    calibrator.fit(X_cal, y_cal)
    return calibrator


def fit_isotonic_regression(probas: np.ndarray, y_true: np.ndarray) -> Any:
    if probas.shape[1] < 2:
        raise ValueError("Isotonic regression requires at least two classes.")
    regressors = []
    for class_index in range(probas.shape[1]):
        target = (y_true == class_index).astype(float)
        reg = IsotonicRegression(out_of_bounds="clip")
        reg.fit(probas[:, class_index], target)
        regressors.append(reg)
    return regressors
