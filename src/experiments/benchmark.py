from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import make_preprocessor
from src.data.splitting import train_calibration_test_split
from src.models.calibration import calibrate_predictions
from src.models.classifiers import build_pipeline
from src.conformal.icp import SplitConformalClassifier
from src.conformal.metrics import accuracy_score, brier_score, expected_calibration_error


def run_benchmark(
    dataset_name: str = "breast_cancer",
    n_trials: int = 1,
    alpha: float = 0.1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    X, y = load_dataset(dataset_name)
    rows = []
    for trial in range(n_trials):
        X_train, X_cal, X_test, y_train, y_cal, y_test = train_calibration_test_split(
            X, y, train_fraction=0.5, calibration_fraction=0.2, random_state=42 + trial
        )

        preprocessor = make_preprocessor(pd.DataFrame(X_train), scale_numeric=True)
        model = build_pipeline("logistic_regression", preprocessor, random_state=42 + trial)
        model.fit(X_train, y_train)

        y_prob_test = model.predict_proba(X_test)
        y_prob_cal = model.predict_proba(X_cal)

        uncalibrated_acc = accuracy_score(y_test, y_prob_test.argmax(axis=1))
        uncalibrated_brier = brier_score(y_test, y_prob_test)
        uncalibrated_ece = expected_calibration_error(y_test, y_prob_test)

        calibrated = calibrate_predictions(model, X_cal, y_cal, method="sigmoid")
        cal_probs = calibrated.predict_proba(X_test)

        conformal = SplitConformalClassifier(alpha=alpha)
        conformal.fit(np.asarray(X_cal), np.asarray(y_cal), np.asarray(y_prob_cal))
        prediction_sets = conformal.predict(np.asarray(X_test), np.asarray(cal_probs))

        rows.append(
            {
                "dataset": dataset_name,
                "trial": trial,
                "alpha": alpha,
                "accuracy": uncalibrated_acc,
                "brier": uncalibrated_brier,
                "ece": uncalibrated_ece,
                "calibrated_accuracy": accuracy_score(y_test, cal_probs.argmax(axis=1)),
                "calibrated_brier": brier_score(y_test, cal_probs),
                "calibrated_ece": expected_calibration_error(y_test, cal_probs),
                "conformal_coverage": conformal.coverage(y_test, prediction_sets),
                "mean_set_size": float(np.mean([len(s) for s in prediction_sets])),
            }
        )
        if progress_callback is not None:
            progress_callback(trial + 1, n_trials)

    result = pd.DataFrame(rows)
    output_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / f"{dataset_name}_benchmark.csv", index=False)
    return result
