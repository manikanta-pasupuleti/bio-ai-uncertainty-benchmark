from __future__ import annotations

import streamlit as st

from src.data.loader import load_dataset
from src.data.preprocessing import make_preprocessor
from src.data.splitting import train_calibration_test_split
from src.models.classifiers import build_pipeline

@st.cache_data(show_spinner="Loading prediction demo...")
def build_prediction_demo(dataset_name: str):
    X, y = load_dataset(dataset_name)
    X_train, X_cal, X_test, y_train, y_cal, y_test = train_calibration_test_split(
        X, y, train_fraction=0.6, calibration_fraction=0.2, random_state=42
    )
    preprocessor = make_preprocessor(X_train, scale_numeric=True)
    model = build_pipeline("logistic_regression", preprocessor, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

st.title("Prediction Demo")
selected = st.selectbox("Dataset", ["breast_cancer", "heart_disease"])

try:
    model, X_test, y_test = build_prediction_demo(selected)
    y_pred = model.predict(X_test)
    st.write(f"Test accuracy: {((y_pred == y_test).mean()):.3f}")
    st.dataframe(
        {
            "actual": y_test[:10].to_numpy(),
            "predicted": y_pred[:10],
        }
    )
except Exception as exc:  # pragma: no cover - UI-only guard
    st.warning(f"Demo could not run: {exc}")
