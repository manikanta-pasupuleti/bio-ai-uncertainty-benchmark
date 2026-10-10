from __future__ import annotations

from src.experiments.benchmark import run_benchmark


def test_benchmark_smoke() -> None:
    result = run_benchmark("breast_cancer", n_trials=1, alpha=0.1)
    assert list(result.columns) == [
        "dataset",
        "trial",
        "model",
        "alpha",
        "accuracy",
        "brier",
        "ece",
        "calibrated_accuracy",
        "calibrated_brier",
        "calibrated_ece",
        "conformal_coverage",
        "mean_set_size",
    ]
    assert set(result["model"]) == {"logistic_regression", "random_forest", "svm"}
    assert len(result) == 3
