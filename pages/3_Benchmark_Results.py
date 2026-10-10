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
    st.dataframe(df, use_container_width=True)
    target_coverage = 1 - float(df["alpha"].iloc[0])
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

    left, right = st.columns(2)
    with left:
        st.subheader("Coverage by trial")
        coverage = df.set_index("trial")[["conformal_coverage"]].rename(columns={"conformal_coverage": "Observed coverage"})
        coverage["Nominal target"] = target_coverage
        st.line_chart(coverage)
    with right:
        st.subheader("Calibration error")
        st.bar_chart(
            df[["ece", "calibrated_ece"]].mean().rename(
                {"ece": "Uncalibrated ECE", "calibrated_ece": "Calibrated ECE"}
            )
        )
    st.download_button(
        "Download this result table",
        data=df.to_csv(index=False),
        file_name=selected_file,
        mime="text/csv",
    )
