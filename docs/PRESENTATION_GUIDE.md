# Capstone Presentation and Viva Guide

## Suggested 8-minute presentation

### 1. Problem and motivation — 1 minute

Small biomedical datasets can produce confident predictions even when the model is
uncertain. Accuracy alone does not show whether probabilities are trustworthy or
whether a prediction should be treated as ambiguous.

### 2. Research question — 45 seconds

> When biomedical datasets are small, noisy, or shifted, how reliably do calibrated
> probabilities and conformal prediction sets communicate uncertainty?

### 3. Data and models — 1 minute

- Breast Cancer Wisconsin Diagnostic dataset
- UCI Heart Disease dataset
- Logistic regression, random forest, and SVM model support
- Separate training, calibration, and test partitions

### 4. Methodology — 2 minutes

1. Load and preprocess the selected biomedical dataset.
2. Train the classifier only on the training partition.
3. Use a separate calibration partition to estimate uncertainty thresholds.
4. Evaluate probabilities using Brier score and expected calibration error.
5. Build split conformal prediction sets.
6. Measure accuracy, empirical coverage, and mean prediction-set size.
7. Repeat the experiment across trials and stress the test data with noise or shift.

### 5. Demo — 2 minutes

1. Open the [live Streamlit app](https://bio-ai-uncertainty-benchmark.streamlit.app/).
2. Show the capstone overview and research question.
3. Open **Dataset Explorer** and explain rows, features, missing values, and class balance.
4. Use the sidebar to run a two-trial breast-cancer benchmark.
5. Open **Benchmark Results** and explain observed versus nominal coverage.
6. Open **Robustness Analysis**, select feature noise, and run a sweep.
7. Point out that empirical coverage can move under shift.

### 6. Findings and limitations — 1 minute

The project is designed to compare uncertainty behavior, not to claim clinical
validity. Split conformal coverage relies on exchangeability; distribution shift
can weaken that guarantee. The datasets are also small and may not represent
current clinical populations.

## Demo checklist

- Confirm the Streamlit app is awake before presenting.
- Start with the homepage rather than the benchmark page.
- Use two trials for a quick demo; use more trials for final reported results.
- Keep alpha at `0.10` unless explaining how nominal coverage changes.
- Download the CSV from Benchmark Results if a table is needed for slides.
- Have a local fallback:

```powershell
streamlit run app.py
```

## Viva questions and answers

### Why did you choose conformal prediction?

It produces prediction sets rather than only a single label and gives a
finite-sample marginal coverage statement under exchangeability. This makes the
uncertainty output easier to evaluate empirically than a raw confidence score.

### What does alpha mean?

Alpha is the allowed error rate in the nominal conformal target. For example,
`alpha = 0.10` corresponds to a nominal coverage target of `1 - alpha = 0.90`.

### Why is the calibration set separate from the test set?

The calibration set estimates the nonconformity threshold, while the test set
provides an independent evaluation of coverage and prediction-set size. Reusing
the same data for both would make the evaluation less reliable.

### What is calibration?

Calibration describes whether predicted probabilities match observed frequencies.
This project uses Brier score and expected calibration error. Lower values are
better for both metrics.

### What is the difference between accuracy and coverage?

Accuracy asks whether the single most likely class is correct. Coverage asks
whether the true class appears anywhere in the conformal prediction set. A model
can have reasonable accuracy but poor uncertainty coverage.

### Why can prediction sets contain multiple classes?

When the model is uncertain, more than one class may pass the conformal
threshold. Larger sets improve the chance of containing the truth but provide
less specific information, so coverage and mean set size should be considered
together.

### Does conformal prediction guarantee coverage under distribution shift?

No. The standard split conformal guarantee assumes exchangeability between
calibration and test examples. The robustness page intentionally reports observed
coverage under noise and shift instead of claiming the guarantee still holds.

### Why use multiple trials?

Small datasets can produce results that depend strongly on the split. Multiple
randomized trials show how stable the measured accuracy, calibration, and coverage
are across different train/calibration/test partitions.

### Is this system suitable for clinical diagnosis?

No. It is an educational and research benchmark. The datasets are small, the
experiments are not clinically validated, and the output must not be used as
medical advice or a clinical decision-support system.

### What would you improve next?

I would add repeated cross-validation, external clinical datasets, subgroup
fairness analysis, stronger missing-data experiments, confidence intervals, and
prospective clinical validation before considering any real-world use.
