from __future__ import annotations

import streamlit as st

st.title("Robustness Analysis")
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
