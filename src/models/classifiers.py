from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_classifier(name: str, random_state: int = 42) -> Any:
    model_name = name.lower()
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    if model_name == "svm":
        return SVC(probability=True, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(name: str, preprocessor: Any, random_state: int = 42) -> Pipeline:
    clf = get_classifier(name, random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
