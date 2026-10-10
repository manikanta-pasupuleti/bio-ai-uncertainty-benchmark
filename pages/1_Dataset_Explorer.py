from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data.loader import load_dataset

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_cached(dataset_name: str):
    return load_dataset(dataset_name)

st.title("Dataset Explorer")
st.caption("Inspect sample size, feature structure, and class balance before interpreting model results.")

selected = st.selectbox("Dataset", ["breast_cancer", "heart_disease"])

try:
    X, y = load_dataset_cached(selected)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{X.shape[0]:,}")
    metric_columns[1].metric("Features", f"{X.shape[1]:,}")
    metric_columns[2].metric("Missing values", f"{int(X.isna().sum().sum()):,}")
    metric_columns[3].metric("Classes", f"{y.nunique():,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Target distribution")
        distribution = y.value_counts().sort_index().rename("count")
        st.bar_chart(distribution)
    with right:
        st.subheader("Feature types")
        type_counts = X.dtypes.astype(str).value_counts().rename("count")
        st.dataframe(pd.DataFrame({"dtype": type_counts.index, "count": type_counts.values}), hide_index=True)

    st.subheader("Sample records")
    st.dataframe(X.head())
except Exception as exc:  # pragma: no cover - UI-only guard
    st.error(f"Dataset could not be loaded: {exc}")
