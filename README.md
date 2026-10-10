# Robustness & Uncertainty Quantification in Small-Data Bio-AI

## Objective
This project benchmarks inductive conformal prediction (ICP) against conventional probability calibration methods on small, biomedical classification datasets. It explicitly studies how limited training data, calibration-set size, feature noise, missing data, and distribution shift alter coverage, set size, classification accuracy, and calibration quality.

## Scientific note
Split conformal prediction provides finite-sample marginal coverage under exchangeability assumptions. Distribution shift can invalidate these assumptions; this project measures the actual empirical coverage and warns when nominal guarantees may no longer hold.

## Quick start
```bash
cd C:/23eg107b46/ML-project
python -m pip install -r requirements.txt
python scripts/run_benchmark.py --dataset both --n_trials 2
streamlit run app.py
```

## Datasets
- Breast Cancer Wisconsin Diagnostic: loaded from scikit-learn.
- UCI Heart Disease: downloaded from the official UCI archive when available.
- If external data retrieval fails, the loader raises a clear error and the rest of the project continues to use the remaining working dataset.

## Project structure
- `src/data`: loaders, preprocessing and splitting helpers
- `src/models`: classifiers and calibration wrappers
- `src/conformal`: split conformal prediction and metrics
- `src/experiments`: benchmark and robustness analyses
- `src/visualization`: plotting utilities
- `pages`: Streamlit pages
- `scripts`: benchmark execution entry points
- `tests`: validation tests
