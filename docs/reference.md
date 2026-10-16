# Technical reference

This document describes what the toolkit computes and how a verdict is reached. How to run it on your own files is in the [README](../README.md). Defaults live in `Thresholds` and `DriftConfig` in `population_drift/analysis.py` and are copied into each run’s metadata.

## Question and what is not identified

The question is whether a boosted tree model fitted on a training population can be applied to a new population that has the model’s features and no labels.

 Every performance number below is an estimate under a stated assumption. The verdict is a decision rule on a list of signals. It is not a level-α test of “the populations are the same”, and it does not control a family-wise error rate across that list.

## Which pair each test uses

Without `--test`, every test uses the labelled file passed as `--train` against the new file. The toolkit does not split that file.

With `--test`, `--train` is the fit set and `--test` is the evaluation set. Each test uses one pair:

| Tests | Pair |
|---|---|
| Continuous and categorical per-feature distributions, equivalence, missingness, domain classifier, maximum mean discrepancy (MMD), energy distance, correlations, range, isolation forest | Fit set vs new |
| Score distribution, positive rate, TreeSHAP (Shapley additive explanations), importance weighting, confidence-based performance estimation (CBPE), black-box shift estimation (BBSE) | Evaluation set vs new |
| Calibration | Evaluation set |

The weighting classifier is fit on the evaluation set versus the new population, because those weights reweight evaluation rows. It is stored separately from the fit-set domain classifier. Levels are `severe`, `moderate`, and `mild`. A reference shorter than `n_new + 30` raises before the analysis starts. Each reference has its own size-matched baseline for the tests that use it.

## Null

Each baseline round permutes one reference (the fit set, or the evaluation set). The first `subset_size` rows stand in for the new population; the rest stand in for that reference. The same point statistics are computed. Baseline rounds do not run bootstrap or permutation tests.

A statistic exceeds the baseline when it is larger than the baseline 95th percentile (`Thresholds.baseline_percentile`). The empirical p-value is `(1 + count of rounds at least as large) / (1 + rounds)`. With 30 rounds the smallest p-value is 1/31.

`subset_size` must match the new population’s row count. Population Stability Index (PSI), Kolmogorov–Smirnov statistic (KS D), and domain-classifier area under the curve (AUC) shrink as sample size grows. The code warns when the new size is outside half to twice the baseline subset size, and it refuses a baseline built on a different feature set.

This null is sampling noise under exchangeability inside that reference. It is not the noise of drift already present inside the reference window. If that period itself moved, the baseline is wide and a further shift can look ordinary.

## Feature severity

Benjamini–Hochberg (BH) adjustment is applied inside a test family, across features, at α = 0.05 (`DriftConfig.alpha`, `correction_method`). It is not applied across the signal list.

A feature is severe only when three conditions hold: the test is significant after that adjustment, the statistic exceeds its baseline 95th percentile, and the effect is large. A moderate level uses the same significance and baseline conditions with a moderate effect. Otherwise the feature is mild. The feature’s level is the worse of its distribution level and its missingness level. Without a model, every feature is treated as important.

| Type | Significant | Beyond baseline on | Large | Moderate |
|---|---|---|---|---|
| Continuous | Kolmogorov–Smirnov (KS) or Anderson–Darling (AD) adjusted p < α | Kolmogorov–Smirnov statistic (KS D), Population Stability Index (PSI), Wasserstein distance / standard deviation (SD) | Population Stability Index (PSI) ≥ 0.25 or Kolmogorov–Smirnov statistic (KS D) ≥ 0.20 or \|standardized mean difference (SMD)\| ≥ 0.50 | Population Stability Index (PSI) ≥ 0.10 or Kolmogorov–Smirnov statistic (KS D) ≥ 0.10 or \|standardized mean difference (SMD)\| ≥ 0.10 |
| Categorical | chi-square adjusted p < α | Cramér’s V, Population Stability Index (PSI), Jensen–Shannon (JS) | Population Stability Index (PSI) ≥ 0.25 or V ≥ 0.30 | Population Stability Index (PSI) ≥ 0.10 or V ≥ 0.10 or unseen-level share > 1% |
| Missingness | two-proportion z adjusted p < α | Cohen’s h | \|h\| ≥ 0.50 | \|h\| ≥ 0.20 |

The effect-size cutoffs are conventions: Population Stability Index (PSI) 0.10 / 0.25 (Siddiqi 2006), standardized mean difference (SMD) and Cohen’s h (Cohen 1988; Austin 2009 for the 0.10 mean-difference benchmark), Cramér’s V 0.10 / 0.30. They are not estimated from the user’s data. Change them in `Thresholds` before looking at the new population, not after.

Low-importance features are capped at moderate in the signal list, so a large shift in a feature the model barely uses cannot by itself force a severe verdict. The feature table still shows that feature’s own severity.

Model-important features are the smallest set whose mean absolute TreeSHAP (Shapley additive explanations) on the reference used in that comparison reaches 80% of the total (`important_feature_cumulative_share`). On the fit set those contributions are in-sample.

## Signals

`info` signals are reported and are not counted in the verdict.

| Signal | Rule |
|---|---|
| Feature drift, model-important features | Worst severity among important features. This signal can be severe. |
| Feature drift, other features | Moderate if any non-important feature is flagged. Never severe. |
| Missing-value rates | Moderate if any \|h\| ≥ 0.2, significant, and beyond baseline. Severe if \|h\| ≥ 0.5 on an important feature. |
| Unseen category levels | Moderate if any feature has an unseen level in more than 1% of new rows. |
| Domain classifier model | Mild if within baseline or area under the curve (AUC) ≤ 0.55. Moderate if area under the curve (AUC) ≤ 0.80. Severe if area under the curve (AUC) > 0.80. |
| Kernel and energy tests | Moderate if either test has permutation p < α/2 and exceeds its baseline. Cannot be severe. |
| Dependence | Moderate if the Spearman Frobenius norm is significant and beyond baseline, or an important pair has \|Δρ\| > 0.20 with adjusted p < α. |
| Out-of-support rows | Mild if within baseline or share < 2%. Moderate up to 5%. Severe above 5%. |
| Values outside the training range | Moderate if an important feature has more than twice the expected share outside the training 0.5%–99.5% quantiles, adjusted p < α, and beyond baseline. |
| Score distribution | Mild if within baseline or Population Stability Index (PSI) < 0.10. Moderate if Population Stability Index (PSI) ≤ 0.25. Severe if Population Stability Index (PSI) > 0.25. |
| Predicted positive rate | Moderate if \|Cohen’s h\| ≥ 0.20 and the two-proportion p-value < α, at `decision_threshold`. |
| Shapley additive explanations (SHAP) attributions | Moderate if the rank correlation of mean \|Shapley additive explanations (SHAP)\| is below 0.80, or an important feature has contribution Kolmogorov–Smirnov (KS) ≥ 0.10, adjusted p < α, and beyond baseline. |
| Effective sample size | Mild if the Kish effective sample size (ESS) ratio > 0.50. Moderate from 0.10 to 0.50. Severe below 0.10. |
| Importance-weighted performance and confidence-based performance estimation (CBPE) | Severe only if the whole 95% confidence interval (CI) is beyond `performance_tolerance` (default 0.02 absolute area under the curve (AUC), or relative error for regression). Moderate if the point estimate is beyond tolerance, or the interval crosses it and the degradation exceeds half the tolerance. Confidence-based performance estimation (CBPE) is downgraded to `info` when source expected calibration error (ECE) > 0.05. |
| Class prior (BBSE) | Moderate if the interval excludes the training prevalence. Valid only if P(x\|y) is unchanged. |
| Source calibration | `info`. expected calibration error (ECE) and calibration slope on the labelled training file. |

Verdict:

- **severe** — “Different: do not use the model as is” if any counted signal is severe.
- **mild** — “Similar: the model can be used (with monitoring)” if there is no severe signal, at most two moderate signals, and no moderate signal with `important=True`. “Monitoring” means a later rerun of this comparison. The repository does not schedule one.
- **moderate** — “Use with caution” otherwise.

On 10 synthetic pairs drawn from the same distribution (2,500 vs 1,000 rows, 10-round baselines) the verdict was never severe, and 3 of 10 were moderate because one weak signal crossed its threshold. That check is not the operating configuration. Treat a single weak moderate signal as a review item, not as a demonstrated shift.

## Methods

One function per question. A failing task is logged in `metadata["failed_tasks"]` and does not stop the others. Random draws use `DriftConfig.seed` (default 42).

| Question | Procedure | Why this procedure |
|---|---|---|
| Did a continuous per-feature distribution change? | Two-sample Kolmogorov–Smirnov (KS), Anderson–Darling (AD) k-sample, Cramér–von Mises (CvM). Benjamini–Hochberg (BH) on each p-value family. | The three tests weight the distribution differently. Kolmogorov–Smirnov statistic (KS D) is also the effect size in the severity rule. |
| Did the mean change by a practical amount, and are the means equivalent? | Standardized mean difference (SMD), pooled standard deviation (SD) with an analytic 95% confidence interval (CI). Two one-sided tests (TOST), unequal variance, margin 0.10 pooled standard deviation (SD). | Standardized mean difference (SMD) is the scale used in the severity rule. Two one-sided tests (TOST) are evidence for similar means. It says nothing about variance or shape. |
| Did scale or bin masses change? | Variance ratio, Wasserstein distance divided by the source standard deviation (SD), Population Stability Index (PSI) on source-quantile bins (10 bins). Bootstrap intervals on the comparison run (200 resamples). | Population Stability Index (PSI) is the conventional stability summary. It is not used as a p-value. |
| Did a categorical per-feature distribution change? | Chi-square after merging levels with expected count below 5. Cramér’s V, Jensen–Shannon (JS) (base 2), Population Stability Index (PSI). Labels `1` and `1.0` are treated as the same level. | Chi-square is the test of a shared discrete distribution. V does not grow with sample size. |
| Will the model see an untrained level? | Share of new rows whose level is absent from training. | LightGBM sends an unseen level down a default path. |
| Did missingness change? | Two-proportion z-test and Cohen’s h per feature. Separately, chi-square and Jensen–Shannon (JS) on the joint missingness pattern. | Observed values can match while the missing rate changes, and LightGBM’s missing branch then applies to a different fraction of rows. |
| Can any function of the features tell the populations apart? | Domain classifier model: LightGBM, 200 trees, learning rate 0.05, 31 leaves, row and column subsampling 0.8, L2 penalty 1. Label is population membership, never the business target. Metric is stratified 5-fold out-of-fold (OOF) receiver operating characteristic (ROC) area under the curve (AUC). | Per-feature tests miss a shift that exists only as a combination. The area under the curve (AUC) does not depend on a cutoff and stays 0.5 when the score distributions match, which matters because the group sizes differ. Out-of-fold scoring stops the trees being graded on rows they just fitted. Hyperparameters are fixed, so the area under the curve (AUC) is not the winner of a model search (Friedman 2003; Lopez-Paz and Oquab 2017). |
| Could that area under the curve (AUC) arise under no shift? | Two nulls. Shuffle the population label and refit the cross-validation (`n_classifier_permutations`, default 20). And the training-split baseline above. A Mann–Whitney test on the held-out scores is stored; for fixed scores it is the same comparison as area under the curve (AUC) (Hanley and McNeil 1982) but it treats the scores as given, so the verdict uses the permutation p-value. | The permutation null breaks only the population label. The baseline asks whether the same area under the curve (AUC) appears when both groups are random pieces of the training population. |
| Which inputs drive that separation? | Gain (total loss reduction) from one fit on all rows, as a share of total gain. | Descriptive of that fit. Not the business model’s Shapley additive explanations (SHAP), and not a causal ranking. |
| Do two nonparametric distances agree? | Unbiased squared maximum mean discrepancy (MMD²), radial basis function (RBF) kernel, median-heuristic bandwidth; energy distance. Permutation p-values. Continuous columns are source-median-imputed and source-standardized. Categoricals are one-hot of the top 10 source levels. Each group is capped at 2,000 rows. | These do not depend on trees. The row cap limits power. They cannot turn the verdict severe. |
| Did rank correlations change? | Spearman matrices. Permutation test on the Frobenius norm of the upper triangle. Per pair, Fisher z with variance 1.06/(n−3), then Benjamini–Hochberg (BH). Capped at 20,000 rows. | A global norm can hide one large pair change, and one pair can be a term the model uses. |
| Are values outside the training tails? | Share of new values outside the source 0.5%–99.5% quantiles, one-sided binomial test against the expected tail share, Benjamini–Hochberg (BH) across features. Also the share beyond the source min and max. | Quantiles, so one training outlier does not define the support. |
| What share of new rows has no training neighbors? | Isolation Forest fitted on 70% of source rows. The threshold is the 1% quantile of scores on the held-out 30%, so the null share is 1% by construction. One-sided binomial test. Capped at 100,000 rows. | A product of per-feature ranges can still leave holes. |
| Did the score distribution change? | Kolmogorov–Smirnov (KS) and Population Stability Index (PSI) on predicted probabilities. Two-proportion z-test and Cohen’s h on the positive rate at the operating threshold, and the same test for the share of scores within ±0.05 of that threshold. | A small location shift can move the approval rate without a large Population Stability Index (PSI). The two questions are separate signals. |
| Did the reasons for the scores change? | TreeSHAP (Shapley additive explanations) via `pred_contrib`, at most 5,000 rows per population. Kolmogorov–Smirnov (KS) on contributions. Spearman correlation of mean absolute Shapley additive explanations (SHAP). | A similar score histogram can be produced by different features. |

### Performance without labels

If these rows disagree, do not quote one of them as the performance on the new population.

| Estimator | What it computes | Assumption |
|---|---|---|
| Calibration | Brier, expected calibration error (ECE) on 10 equal-width bins, slope and intercept of a logistic regression of the label on logit(score). | Computed on the evaluation set in a dual run. Optimistic if those rows were used to fit the model. Expected calibration error (ECE) > 0.05 downgrades confidence-based performance estimation (CBPE) to `info`. |
| Importance weighting | Domain-classifier out-of-fold (OOF) scores become weights `w = p/(1−p) × n_source/n_target`, clipped at the 0.99 quantile. Kish effective sample size (ESS). Weighted area under the curve (AUC), Brier, and log loss (binary) or root mean squared error (RMSE) and mean absolute error (MAE) (regression), with a bootstrap interval, compared with the unweighted metrics on the same rows. | Covariate shift only: P(y\|x) is unchanged and the domain classifier estimates the density ratio. The factor `n_source/n_target` converts classifier odds into a density ratio when the groups differ in size. |
| Confidence-based performance estimation (CBPE) | Expected area under the curve (AUC) from pairs of predicted probabilities (ties count as 0.5). The reference is the same functional on the source scores, reported next to the realized source area under the curve (AUC). | Calibrated probabilities and an unchanged map from features to labels. Using the same functional on both populations avoids calling a calibration defect a population shift. |
| Black-box shift estimation (BBSE) | Source confusion matrix C, new prediction rates μ, weights C⁻¹μ, prior renormalized. Interval by resampling both populations. | P(x\|y) is unchanged (label shift). Under covariate shift this estimator is the wrong model. |

## Configuration a user is expected to set

| Field | Default | Set it from |
|---|---|---|
| `decision_threshold` | 0.5 | The cutoff the deployed model uses |
| `performance_tolerance` | 0.02 | The area under the curve (AUC) drop, or relative error increase, that should count as material |
| `problem_type` | `binary` | `binary` or `regression` |
| `n_rounds` | 30 | Leave at 30 or higher for a decision |
| `subset_size` | `n_new` inside `compare` | Row count of the new population. With `--test`, each reference is matched separately. |
| `seed` | 42 | Fix and record it. |
| `n_classifier_permutations` | 20 | Smallest permutation p-value is 1/21. Raising this refits the domain classifier model each time |
| `n_bootstrap` / `n_permutations` | 200 / 100 | Comparison run only; baseline rounds set these to 0 |

Row caps: `max_kernel_rows` 2,000, `max_correlation_rows` 20,000, `max_classifier_rows` 200,000, `max_shap_rows` 5,000, `max_iforest_rows` 100,000. A shift that is visible only above a cap can be missed by that task.

## Outputs

`runs/baseline/baseline_statistics.csv` columns: `round`, `task`, `table`, `feature`, `metric`, `value`.

`runs/baseline/baseline_summary.csv` columns: `task`, `table`, `feature`, `metric`, `n_rounds`, `mean`, `std`, `p50`, `p90`, `p95`, `p99`, `max`.

`runs/comparison/baseline_comparison.csv` adds, for each observed statistic, the baseline median, the 95th-percentile threshold, the maximum, the empirical p-value, and `exceeds_baseline`.

`runs/comparison/decision.json` holds `verdict`, `verdict_level` (`severe`, `moderate`, or `mild`), `verdict_rule`, `signals` (each with a `pair`), `task_pairs`, `n_severe`, `n_moderate`, `warnings`, and the baseline path.

`runs/comparison/tables/<task>__<table>.csv` is the full table for that task. Global rows use `feature="__global__"`. Registered comparison metrics are oriented so that larger means more drift (absolute standardized mean difference (SMD), absolute log variance ratio, `1 − ρ`, `1 − effective sample size (ESS) ratio`, degradation).

`metadata.json` records sizes, the feature list and types, whether a model and a label were supplied, the full config, failed tasks, elapsed time, and library versions.

## References

- Austin, P. C. (2009). Using the standardized difference to compare the prevalence of a binary variable between two groups. *Communications in Statistics — Simulation and Computation*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Friedman, J. (2003). On multivariate goodness-of-fit and two-sample testing.
- Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic curve. *Radiology*.
- Lopez-Paz, D. and Oquab, M. (2017). Revisiting classifier two-sample tests. International Conference on Learning Representations (ICLR).
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Population Stability Index (PSI) conventions 0.10 and 0.25.
