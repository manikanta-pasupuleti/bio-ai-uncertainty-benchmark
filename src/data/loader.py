import io
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer


HEART_DISEASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def load_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    return data.frame, data.target


def load_heart_disease_dataset(data_dir: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    target_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "heart_disease.csv"

    if target_path.exists():
        df = pd.read_csv(target_path)
    else:
        try:
            response = requests.get(HEART_DISEASE_URL, timeout=30)
            response.raise_for_status()
            raw = response.text
        except requests.RequestException as exc:  # pragma: no cover - relies on network availability
            raise RuntimeError(
                "The UCI Heart Disease dataset could not be downloaded. "
                "Please ensure network access is available or use a local copy."
            ) from exc

        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(io.StringIO(raw), header=None, names=columns)
        df.to_csv(target_path, index=False)

    if "target" not in df.columns:
        raise ValueError("Heart disease dataset is missing the target column.")

    X = df.drop(columns=["target"])
    y = df["target"].apply(lambda value: 1 if value > 0 else 0)
    return X, y


def load_dataset(name: str, data_dir: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    name = name.lower().replace(" ", "_")
    if name == "breast_cancer":
        return load_breast_cancer_dataset()
    if name in {"heart_disease", "heart-disease", "uci_heart_disease"}:
        return load_heart_disease_dataset(data_dir=data_dir)
    raise ValueError(f"Unknown dataset: {name}")
