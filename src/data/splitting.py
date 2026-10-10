from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def train_calibration_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train_fraction + calibration_fraction must be less than 1.")

    n_samples = len(X)
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(n_samples)
    X = np.asarray(X)
    y = np.asarray(y)

    train_end = int(train_fraction * n_samples)
    cal_end = train_end + int(calibration_fraction * (n_samples - train_end))

    train_idx = shuffled_indices[:train_end]
    cal_idx = shuffled_indices[train_end:cal_end]
    test_idx = shuffled_indices[cal_end:]

    if len(cal_idx) == 0 or len(test_idx) == 0:
        raise ValueError("The split produced an empty calibration or test set. Increase the sample size or adjust the fractions.")

    return (
        X[train_idx],
        X[cal_idx],
        X[test_idx],
        y[train_idx],
        y[cal_idx],
        y[test_idx],
    )
