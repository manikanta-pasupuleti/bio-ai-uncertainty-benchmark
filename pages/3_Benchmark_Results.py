from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Benchmark Results")
results_dir = Path(__file__).resolve().parents[1] / "results" / "tables"

csv_files = sorted(results_dir.glob("*_benchmark.csv")) if results_dir.exists() else []
if not csv_files:
    st.info("No benchmark CSV files are available yet. Run the benchmark script to generate them.")
else:
    selected_file = st.selectbox("Result file", [path.name for path in csv_files])
    df = pd.read_csv(results_dir / selected_file)
    st.dataframe(df, use_container_width=True)
    st.line_chart(df.set_index("trial")["conformal_coverage"])
