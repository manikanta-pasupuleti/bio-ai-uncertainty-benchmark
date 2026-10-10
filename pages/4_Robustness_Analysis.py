from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from src.conformal.icp import SplitConformalClassifier
from src.data.loader import load_dataset
from src.data.preprocessing import make_preprocessor
from src.data.splitting import train_calibration_test_split
from src.experiments.robustness import add_noise, apply_covariate_shift
from src.models.classifiers import build_pipeline

st.title("Robustness Analysis")
st.caption("Stress-test empirical coverage and accuracy under controlled feature perturbations.")
st.markdown(
    """
    This benchmark studies how limited-data settings degrade calibration, predictive confidence, and coverage.

    Planned stress tests include:
    - feature noise injection
    - missingness amplification
    - covariate shift between calibration and test samples
    - reduced calibration-set sizes for split conformal prediction

    Important caveat: split conformal methods retain finite-sample marginal coverage only under exchangeability.
    Under distribution shift, empirical coverage may drift away from the nominal target,
    so the benchmark reports observed coverage rather than assuming the guarantee still holds.
    """
)

dataset_name = st.selectbox("Dataset", ["breast_cancer"], key="robustness_dataset")
stress_type = st.selectbox("Stress test", ["Feature noise", "Covariate shift"])
levels = st.slider("Stress level", 0.0, 1.0, (0.0, 0.5), 0.05)
alpha = st.slider("Conformal alpha", 0.01, 0.3, 0.1, 0.01)

if st.button("Run robustness sweep"):
    with st.spinner("Running controlled stress test..."):
        X, y = load_dataset(dataset_name)
        X_train, X_cal, X_test, y_train, y_cal, y_test = train_calibration_test_split(
            X, y, train_fraction=0.5, calibration_fraction=0.2, random_state=42
        )
        preprocessor = make_preprocessor(X_train, scale_numeric=True)
        model = build_pipeline("logistic_regression", preprocessor, random_state=42)
        model.fit(X_train, y_train)
        cal_prob = model.predict_proba(X_cal)
        conformal = SplitConformalClassifier(alpha=alpha).fit(
            np.asarray(X_cal), np.asarray(y_cal), np.asarray(cal_prob)
        )

        rows = []
        for level in np.arange(levels[0], levels[1] + 0.001, 0.05):
            stressed = X_test.to_numpy(dtype=float)
            if stress_type == "Feature noise":
                stressed = add_noise(stressed, noise_level=float(level), random_state=42)
            else:
                stressed = apply_covariate_shift(stressed, shift_strength=float(level), random_state=42)
            probabilities = model.predict_proba(pd.DataFrame(stressed, columns=X_test.columns))
            sets = conformal.predict(stressed, probabilities)
            rows.append(
                {
                    "stress_level": round(float(level), 2),
                    "accuracy": float((probabilities.argmax(axis=1) == y_test.to_numpy()).mean()),
                    "coverage": conformal.coverage(y_test.to_numpy(), sets),
                    "mean_set_size": float(np.mean([len(prediction_set) for prediction_set in sets])),
                }
            )
        results = pd.DataFrame(rows)
    st.success("Robustness sweep completed.")
    st.line_chart(results.set_index("stress_level")[["accuracy", "coverage"]])
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
        "Download robustness results",
        data=results.to_csv(index=False),
        file_name="robustness_sweep.csv",
        mime="text/csv",
    )
