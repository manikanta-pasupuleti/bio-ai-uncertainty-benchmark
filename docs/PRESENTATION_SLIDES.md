# Presentation Slides

## Slide 1 — Title

**Medical Uncertainty Benchmark**

Robustness and uncertainty quantification for small-data biomedical classification

Presenter: Manikanta Pasupuleti

**Speaker note:** This project evaluates whether model probabilities and
conformal prediction sets remain informative when biomedical data is limited or
shifted.

## Slide 2 — Motivation

- Accuracy alone does not measure confidence quality.
- Small biomedical datasets are sensitive to data splits.
- Overconfident predictions can be unsafe.
- Uncertainty should be measured and stress-tested.

**Speaker note:** The project treats uncertainty as an observable model behavior,
not as a hidden assumption.

## Slide 3 — Research question

> When biomedical datasets are small, noisy, or shifted, how reliably do
> calibrated probabilities and conformal prediction sets communicate uncertainty?

## Slide 4 — Datasets and models

- Breast Cancer Wisconsin Diagnostic
- UCI Heart Disease
- Logistic regression baseline
- Random forest and SVM support
- Separate training, calibration, and test partitions

## Slide 5 — Methodology

1. Load and preprocess data.
2. Fit the model on training data.
3. Estimate calibration and conformal thresholds using calibration data.
4. Evaluate probabilities and prediction sets on held-out test data.
5. Repeat across five randomized trials.
6. Add synthetic feature noise for robustness analysis.

## Slide 6 — Evaluation metrics

- Accuracy
- Brier score
- Expected calibration error
- Empirical conformal coverage
- Mean prediction-set size

**Speaker note:** Coverage and set size must be interpreted together: higher
coverage is useful, but very large sets are less informative.

## Slide 7 — Final benchmark results

| Dataset | Accuracy | Coverage | Mean set size |
|---|---:|---:|---:|
| Breast Cancer | 0.9982 | 0.9947 | 1.0000 |
| Heart Disease | 0.8279 | 0.9361 | 1.4033 |

**Speaker note:** The breast-cancer task is easier. Heart disease produces lower
accuracy and larger sets, showing more uncertainty.

## Slide 8 — Robustness result

Breast-cancer feature-noise stress test:

- Noise `0.00`: accuracy `1.0000`, coverage `1.0000`
- Noise `0.25`: accuracy `0.6360`, coverage `0.6360`
- Noise `0.50`: accuracy `0.5833`, coverage `0.5833`
- Noise `1.00`: accuracy `0.5702`, coverage `0.5702`

**Speaker note:** The nominal 0.90 coverage target does not survive this
distribution shift. This is why the dashboard reports observed coverage.

## Slide 9 — Streamlit demonstration

1. Homepage and research question
2. Dataset Explorer
3. Run benchmark
4. Benchmark Results charts and CSV export
5. Robustness Analysis sweep
6. Prediction Demo

## Slide 10 — Limitations and responsible use

- Educational/research benchmark only
- No clinical validation
- Small datasets
- Synthetic robustness perturbations
- Exchangeability is required for standard conformal coverage
- Not medical advice

## Slide 11 — Conclusion

- Uncertainty quality requires more than accuracy.
- Conformal prediction can provide useful prediction sets.
- Coverage can degrade under shift.
- Robustness experiments expose this failure mode.

## Slide 12 — Questions

Contact:

- manikantapasupuleti17@gmail.com
- linkedin.com/in/pmanii
