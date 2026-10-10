from __future__ import annotations

from typing import Any

import numpy as np


def add_noise(X: np.ndarray, noise_level: float = 0.05, random_state: int | None = None) -> np.ndarray:
    if noise_level <= 0:
        return X
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, noise_level, size=X.shape)
    return X + noise


def add_missingness(X: np.ndarray, missing_rate: float = 0.05, random_state: int | None = None) -> np.ndarray:
    if missing_rate <= 0:
        return X
    rng = np.random.default_rng(random_state)
    mask = rng.random(X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing


def apply_covariate_shift(
    X: np.ndarray, shift_strength: float = 0.5, random_state: int | None = None
) -> np.ndarray:
    if shift_strength <= 0:
        return X
    rng = np.random.default_rng(random_state)
    shift = rng.normal(0.0, shift_strength, size=X.shape)
    return X + shift
