from __future__ import annotations

import streamlit as st

from src.experiments.benchmark import run_benchmark

st.set_page_config(page_title="Medical Uncertainty Benchmark", page_icon="🩺", layout="wide")
st.title("Robustness & Uncertainty Quantification in Small-Data Biomedical Classification")
st.caption("Benchmarking calibration methods and split conformal prediction on limited biomedical datasets.")

with st.sidebar:
    st.header("Benchmark controls")
    dataset_name = st.selectbox("Dataset", ["breast_cancer", "heart_disease"])
    n_trials = st.slider("Trials", min_value=1, max_value=10, value=2)
    alpha = st.slider("Alpha", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
    run_button = st.button("Run benchmark")

if run_button:
    try:
        with st.spinner("Running benchmark..."):
            result = run_benchmark(dataset_name=dataset_name, n_trials=n_trials, alpha=alpha)
        st.success("Benchmark completed.")
        st.dataframe(result, use_container_width=True)
    except Exception as exc:  # pragma: no cover - UI path only
        st.error(f"The benchmark failed: {exc}")
else:
    st.info("Select a dataset and execute the benchmark to view empirical accuracy, calibration, and conformal coverage.")
    st.markdown(
        """
        Small-data biomedical classification is sensitive to limited calibration samples and distribution shift.
        This project compares conventional calibration baselines to split conformal prediction while explicitly
        reporting empirical coverage, not assuming nominal guarantees under arbitrary shift.
        """
    )
