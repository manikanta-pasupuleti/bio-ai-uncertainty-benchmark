from __future__ import annotations

import pytest

from src.data.loader import load_dataset


def test_load_breast_cancer_dataset() -> None:
    X, y = load_dataset("breast_cancer")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
    assert set(y.unique()).issubset({0, 1})


def test_invalid_dataset_raises() -> None:
    with pytest.raises(ValueError):
        load_dataset("not_a_dataset")
