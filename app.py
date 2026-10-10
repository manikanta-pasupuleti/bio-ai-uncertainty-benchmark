from __future__ import annotations

import streamlit as st

from src.experiments.benchmark import run_benchmark

st.set_page_config(page_title="Medical Uncertainty Benchmark", page_icon="🩺", layout="wide")
st.title("🩺 Medical Uncertainty Benchmark")
st.subheader("Robustness and uncertainty quantification for small-data biomedical classification")
st.caption("A reproducible capstone study of calibration, split conformal prediction, and distribution shift.")

with st.expander("About this capstone", expanded=True):
    st.markdown(
        """
        **Research question:** When biomedical datasets are small, noisy, or shifted, how reliably do
        calibrated probabilities and conformal prediction sets communicate uncertainty?

        This project compares logistic regression with conventional probability calibration and split
        conformal prediction. It reports empirical accuracy, Brier score, expected calibration error,
        coverage, and prediction-set size rather than assuming that nominal guarantees survive distribution shift.
        """
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Datasets", "2", help="Breast Cancer Wisconsin and UCI Heart Disease")
metric_columns[1].metric("Models", "3", help="Logistic regression, random forest, and SVM")
metric_columns[2].metric("Primary method", "Split ICP")
metric_columns[3].metric("Nominal coverage", "90%", help="Default alpha = 0.10")

with st.sidebar:
    st.header("Run an experiment")
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
        st.download_button(
            "Download benchmark CSV",
            data=result.to_csv(index=False),
            file_name=f"{dataset_name}_benchmark.csv",
            mime="text/csv",
        )
    except Exception as exc:  # pragma: no cover - UI path only
        st.error(f"The benchmark failed: {exc}")
else:
    st.info("Use the sidebar to run the benchmark, or open the pages for dataset exploration and robustness analysis.")
    st.markdown("### How to interpret the results")
    st.markdown(
        """
        - **Calibration:** lower Brier score and ECE indicate better probability quality.
        - **Conformal coverage:** the fraction of test examples whose true label appears in the prediction set.
        - **Mean set size:** smaller sets are more informative, provided coverage remains acceptable.
        - **Scientific caveat:** split conformal coverage is justified under exchangeability; shift can reduce
          empirical coverage, which is why robustness experiments are included.
        """
    )
