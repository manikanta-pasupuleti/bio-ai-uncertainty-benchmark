from __future__ import annotations

import streamlit as st

from src.data.loader import load_dataset

st.title("Dataset Explorer")

selected = st.selectbox("Dataset", ["breast_cancer", "heart_disease"])

try:
    X, y = load_dataset(selected)
    st.write(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")
    st.write("Target distribution:")
    st.bar_chart(y.value_counts().sort_index())
    st.dataframe(X.head())
except Exception as exc:  # pragma: no cover - UI-only guard
    st.warning(f"Dataset could not be loaded: {exc}")
