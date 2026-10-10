# Medical Uncertainty Benchmark

## Abstract

This capstone evaluates uncertainty quantification for small-data biomedical
classification. It compares conventional probability calibration with split
inductive conformal prediction on the Breast Cancer Wisconsin Diagnostic and UCI
Heart Disease datasets. The benchmark measures accuracy, Brier score, expected
calibration error (ECE), empirical conformal coverage, and mean prediction-set
size. It also stress-tests predictions with feature noise. In five-trial runs at
alpha `0.10`, breast-cancer classification was highly accurate, while the heart
disease task was more difficult and produced larger prediction sets. Noise
experiments showed that empirical accuracy and coverage can deteriorate sharply,
demonstrating why nominal conformal coverage should not be assumed under
distribution shift.

## 1. Introduction

Biomedical machine-learning systems must communicate uncertainty, not only
produce a class label. This is especially important for small datasets, where
random train/test splits can materially change results and overconfident
probabilities can be misleading.

The project asks:

> When biomedical datasets are small, noisy, or shifted, how reliably do
> calibrated probabilities and conformal prediction sets communicate uncertainty?

The goal is an educational and reproducible benchmark, not a clinical diagnostic
system.

## 2. Experimental design

### Datasets

1. **Breast Cancer Wisconsin Diagnostic** — 569 samples and 30 measured features,
   loaded through scikit-learn.
2. **UCI Heart Disease** — retrieved from the official UCI archive and converted
   to a binary target, with 303 samples and 13 input features.

Each dataset is split into training, calibration, and test partitions. The
calibration partition is not used to fit the base classifier.

### Models and metrics

The project supports logistic regression, random forest, and SVM pipelines. The
current benchmark uses logistic regression as the primary baseline. It reports:

- Accuracy: fraction of correct top-label predictions
- Brier score: squared probability error, where lower is better
- ECE: expected calibration error, where lower is better
- Conformal coverage: fraction of test labels included in prediction sets
- Mean set size: average number of labels in each prediction set

### Conformal method

The split conformal classifier computes calibration nonconformity scores from
the probability assigned to the true label. A threshold is selected from the
calibration scores, and test labels whose probability passes the threshold are
included in the prediction set.

The standard finite-sample marginal coverage statement depends on exchangeability.
It does not automatically apply under arbitrary distribution shift.

## 3. Results

The following results were produced with five trials and alpha `0.10`, giving a
nominal coverage target of `0.90`.

| Dataset | Accuracy | Calibrated accuracy | Brier | Calibrated Brier | ECE | Calibrated ECE | Coverage | Mean set size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 0.9982 | 0.9947 | 0.0037 | 0.0170 | 0.0123 | 0.0508 | 0.9947 | 1.0000 |
| Heart Disease | 0.8279 | 0.8164 | 0.2530 | 0.2660 | 0.0564 | 0.0557 | 0.9361 | 1.4033 |

### Interpretation

The breast-cancer task was nearly perfectly separable for this split protocol.
Its conformal prediction sets were always singletons in the five-trial summary,
while empirical coverage was above the 0.90 nominal target.

The heart-disease task was more challenging. Accuracy was approximately 0.83,
coverage averaged 0.9361, and prediction sets averaged 1.4033 labels. The
larger set size indicates that the method expressed more uncertainty on this
dataset.

Calibration did not improve every metric in this small benchmark. This is an
important result: calibration should be measured rather than assumed to improve
accuracy or every probability metric.

### Feature-noise stress test

For one reproducible breast-cancer split, adding Gaussian feature noise produced
the following observed behavior:

| Noise level | Accuracy | Coverage | Mean set size |
|---:|---:|---:|---:|
| 0.00 | 1.0000 | 1.0000 | 1.0000 |
| 0.25 | 0.6360 | 0.6360 | 1.0000 |
| 0.50 | 0.5833 | 0.5833 | 1.0000 |
| 1.00 | 0.5702 | 0.5702 | 1.0000 |

Coverage fell well below the nominal target as noise increased. This is
consistent with the exchangeability caveat: the stressed test distribution no
longer behaves like the calibration distribution.

## 4. Limitations

- The datasets are small and may not represent current clinical populations.
- The experiment uses a single primary model configuration.
- Results depend on the split protocol and the number of trials.
- The noise experiment is synthetic and does not represent every clinical shift.
- There is no external validation, subgroup analysis, or prospective evaluation.
- The application must not be used for diagnosis or treatment decisions.

## 5. Conclusion

The benchmark demonstrates the value of reporting uncertainty with multiple
metrics. Split conformal prediction provides useful prediction sets under
exchangeability, but the robustness experiment shows that empirical coverage can
degrade when the test distribution is perturbed. For small biomedical datasets,
accuracy, calibration, coverage, and set size should be interpreted together.

## 6. Future work

Future versions should add repeated cross-validation, confidence intervals,
stronger missing-data experiments, subgroup fairness analysis, additional
external datasets, and clinical validation with domain experts.
