from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Benchmark Results")
st.caption("Compare predictive quality, calibration, and uncertainty-set behavior across completed trials.")
results_dir = Path(__file__).resolve().parents[1] / "results" / "tables"

@st.cache_data(show_spinner=False)
def read_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

csv_files = sorted(results_dir.glob("*_benchmark.csv")) if results_dir.exists() else []
if not csv_files:
    st.info("No benchmark CSV files are available yet. Run the benchmark script to generate them.")
else:
    selected_file = st.selectbox("Result file", [path.name for path in csv_files])
    df = read_results(str(results_dir / selected_file))
    if "model" not in df.columns:
        df["model"] = "logistic_regression"
    model_labels = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "svm": "SVM",
    }
    df["Model"] = df["model"].map(model_labels).fillna(df["model"])
    st.dataframe(df, use_container_width=True)
    target_coverage = 1 - float(df["alpha"].iloc[0])
    model_summary = (
        df.groupby("Model", sort=False)[
            ["accuracy", "calibrated_accuracy", "conformal_coverage", "mean_set_size"]
        ]
        .mean()
        .rename(
            columns={
                "accuracy": "Accuracy",
                "calibrated_accuracy": "Calibrated accuracy",
                "conformal_coverage": "Conformal coverage",
                "mean_set_size": "Mean set size",
            }
        )
    )
    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Calibrated accuracy", "Conformal coverage", "Mean set size"],
            "Value": [
                df["accuracy"].mean(),
                df["calibrated_accuracy"].mean(),
                df["conformal_coverage"].mean(),
                df["mean_set_size"].mean(),
            ],
        }
    )
    st.subheader("Average performance")
    st.dataframe(summary.style.format({"Value": "{:.3f}"}), hide_index=True, use_container_width=True)

    st.subheader("Model comparison")
    st.bar_chart(model_summary[["Accuracy", "Calibrated accuracy", "Conformal coverage"]])
    st.caption("Higher is better for accuracy and coverage. Mean prediction-set size is shown separately because lower is more concise.")
    st.bar_chart(model_summary[["Mean set size"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Coverage by trial")
        coverage = df.pivot(index="trial", columns="Model", values="conformal_coverage")
        coverage["Nominal target"] = target_coverage
        st.line_chart(coverage)
    with right:
        st.subheader("Calibration error")
        calibration = (
            df.groupby("Model", sort=False)[["ece", "calibrated_ece"]]
            .mean()
            .rename(columns={"ece": "Uncalibrated ECE", "calibrated_ece": "Calibrated ECE"})
        )
        st.bar_chart(calibration)
    st.download_button(
        "Download this result table",
        data=df.to_csv(index=False),
        file_name=selected_file,
        mime="text/csv",
    )
