# Technical reference

This document describes what the toolkit computes and how a verdict is reached. How to run it on your own files is in the [README](../README.md). Defaults live in `Thresholds` and `DriftConfig` in `population_drift/analysis.py` and are copied into each run’s metadata.

## Question and what is not identified

The question is whether a LightGBM model fitted on a training population can be applied to a new population that has the model’s features and no labels.

Concept shift, a change in P(y|x), is not identifiable from features alone. Every performance number below is an estimate under a stated assumption. The verdict is a decision rule on a list of signals. It is not a level-α test of “the populations are the same”, and it does not control a family-wise error rate across that list.

## Null

Each baseline round permutes the training rows. The first `subset_size` rows stand in for the new population; the rest stand in for the training population. The same point statistics are computed. Baseline rounds do not run bootstrap or permutation tests.

A statistic exceeds the baseline when it is larger than the baseline 95th percentile (`Thresholds.baseline_percentile`). The empirical p-value is `(1 + count of rounds at least as large) / (1 + rounds)`. With 30 rounds the smallest p-value is 1/31.

`subset_size` must match the new population’s row count. PSI, Kolmogorov–Smirnov D, and domain-classifier AUC shrink as sample size grows. The code warns when the new size is outside half to twice the baseline subset size, and it refuses a baseline built on a different feature set.

This null is sampling noise under exchangeability inside the training population. It is not the noise of drift that is already present inside the training window. If the training period itself moved, the baseline is wide and a further shift can look ordinary.

## Feature severity

Benjamini–Hochberg adjustment is applied inside a test family, across features, at α = 0.05 (`DriftConfig.alpha`, `correction_method`). It is not applied across the signal list.

A feature is red only when three conditions hold: the test is significant after that adjustment, the statistic exceeds its baseline 95th percentile, and the effect is large. Amber uses the same significance and baseline conditions with a moderate effect. Otherwise the feature is green. The feature’s level is the worse of its distribution level and its missingness level. Without a model, every feature is treated as important.

| Type | Significant | Beyond baseline on | Large | Moderate |
|---|---|---|---|---|
| Continuous | KS or Anderson–Darling adjusted p < α | KS D, PSI, Wasserstein/SD | PSI ≥ 0.25 or KS D ≥ 0.20 or \|SMD\| ≥ 0.50 | PSI ≥ 0.10 or KS D ≥ 0.10 or \|SMD\| ≥ 0.10 |
| Categorical | chi-square adjusted p < α | Cramér’s V, PSI, Jensen–Shannon | PSI ≥ 0.25 or V ≥ 0.30 | PSI ≥ 0.10 or V ≥ 0.10 or unseen-level share > 1% |
| Missingness | two-proportion z adjusted p < α | Cohen’s h | \|h\| ≥ 0.50 | \|h\| ≥ 0.20 |

The effect-size cutoffs are conventions: PSI 0.10 / 0.25 (Siddiqi 2006), standardized mean difference and Cohen’s h (Cohen 1988; Austin 2009 for the 0.10 mean-difference benchmark), Cramér’s V 0.10 / 0.30. They are not estimated from the user’s data. Change them in `Thresholds` before looking at the new population, not after.

Low-importance features are capped at amber in the signal list, so a large shift in a feature the model barely uses cannot by itself force a red verdict. The feature table still shows that feature’s own severity.

Model-important features are the smallest set whose mean absolute TreeSHAP on the training population reaches 80% of the total (`important_feature_cumulative_share`).

## Signals

`info` signals are reported and are not counted in the verdict.

| Signal | Rule |
|---|---|
| Feature drift, model-important features | Worst severity among important features. This signal can be red. |
| Feature drift, other features | Amber if any non-important feature is flagged. Never red. |
| Missing-value rates | Amber if any \|h\| ≥ 0.2, significant, and beyond baseline. Red if \|h\| ≥ 0.5 on an important feature. |
| Unseen category levels | Amber if any feature has an unseen level in more than 1% of new rows. |
| Domain classifier model | Green if within baseline or AUC ≤ 0.55. Amber if AUC ≤ 0.80. Red if AUC > 0.80. |
| Kernel and energy tests | Amber if either test has permutation p < α/2 and exceeds its baseline. Cannot be red. |
| Dependence | Amber if the Spearman Frobenius norm is significant and beyond baseline, or an important pair has \|Δρ\| > 0.20 with adjusted p < α. |
| Out-of-support rows | Green if within baseline or share < 2%. Amber up to 5%. Red above 5%. |
| Values outside the training range | Amber if an important feature has more than twice the expected share outside the training 0.5%–99.5% quantiles, adjusted p < α, and beyond baseline. |
| Score distribution | Green if within baseline or PSI < 0.10. Amber if PSI ≤ 0.25. Red if PSI > 0.25. |
| Predicted positive rate | Amber if \|Cohen’s h\| ≥ 0.20 and the two-proportion p-value < α, at `decision_threshold`. |
| SHAP attributions | Amber if the rank correlation of mean \|SHAP\| is below 0.80, or an important feature has contribution KS ≥ 0.10, adjusted p < α, and beyond baseline. |
| Effective sample size | Green if the Kish ESS ratio > 0.50. Amber from 0.10 to 0.50. Red below 0.10. |
| Importance-weighted performance and CBPE | Red only if the whole 95% interval is beyond `performance_tolerance` (default 0.02 absolute AUC, or relative error for regression). Amber if the point estimate is beyond tolerance, or the interval crosses it and the degradation exceeds half the tolerance. CBPE is downgraded to `info` when source ECE > 0.05. |
| Class prior (BBSE) | Amber if the interval excludes the training prevalence. Valid only if P(x\|y) is unchanged. |
| Source calibration | `info`. ECE and calibration slope on the labelled training file. |

Verdict:

- **red** — “Different: do not use the model as is” if any counted signal is red.
- **green** — “Similar: the model can be used (with monitoring)” if there is no red signal, at most two amber signals, and no amber signal with `important=True`. “Monitoring” means a later rerun of this comparison. The repository does not schedule one.
- **amber** — “Use with caution” otherwise.

On 10 synthetic pairs drawn from the same distribution (2,500 vs 1,000 rows, 10-round baselines) the verdict was never red, and 3 of 10 were amber because one weak signal crossed its threshold. That check is not the operating configuration. Treat a single weak amber signal as a review item, not as a demonstrated shift.

## Methods

One function per question. A failing task is logged in `metadata["failed_tasks"]` and does not stop the others. Random draws use `DriftConfig.seed` (default 42).

| Question | Procedure | Why this procedure |
|---|---|---|
| Did a continuous margin change? | Two-sample KS, Anderson–Darling k-sample, Cramér–von Mises. BH on each p-value family. | The three tests weight the distribution differently. KS D is also the effect size in the severity rule. |
| Did the mean change by a practical amount, and are the means equivalent? | Standardized mean difference (pooled SD) with an analytic 95% interval. TOST, unequal variance, margin 0.10 pooled SD. | SMD is the scale used in the severity rule. TOST is evidence for similar means. It says nothing about variance or shape. |
| Did scale or bin masses change? | Variance ratio, Wasserstein distance divided by the source SD, PSI on source-quantile bins (10 bins). Bootstrap intervals on the comparison run (200 resamples). | PSI is the conventional stability summary. It is not used as a p-value. |
| Did a categorical margin change? | Chi-square after merging levels with expected count below 5. Cramér’s V, Jensen–Shannon (base 2), PSI. Labels `1` and `1.0` are treated as the same level. | Chi-square is the test of a shared discrete distribution. V does not grow with sample size. |
| Will the model see an untrained level? | Share of new rows whose level is absent from training. | LightGBM sends an unseen level down a default path. |
| Did missingness change? | Two-proportion z-test and Cohen’s h per feature. Separately, chi-square and Jensen–Shannon on the joint missingness pattern. | Observed values can match while the missing rate changes, and LightGBM’s missing branch then applies to a different fraction of rows. |
| Can any function of the features tell the populations apart? | Domain classifier model: LightGBM, 200 trees, learning rate 0.05, 31 leaves, row and column subsampling 0.8, L2 penalty 1. Label is population membership, never the business target. Metric is stratified 5-fold out-of-fold ROC AUC. | Marginal tests miss a shift that exists only as a combination. AUC does not depend on a cutoff and stays 0.5 when the score distributions match, which matters because the group sizes differ. Out-of-fold scoring stops the trees being graded on rows they just fitted. Hyperparameters are fixed, so the AUC is not the winner of a model search (Friedman 2003; Lopez-Paz and Oquab 2017). |
| Could that AUC arise under no shift? | Two nulls. Shuffle the population label and refit the cross-validation (`n_classifier_permutations`, default 20). And the training-split baseline above. A Mann–Whitney test on the held-out scores is stored; for fixed scores it is the same comparison as AUC (Hanley and McNeil 1982) but it treats the scores as given, so the verdict uses the permutation p-value. | The permutation null breaks only the population label. The baseline asks whether the same AUC appears when both groups are random pieces of the training population. |
| Which inputs drive that separation? | Gain (total loss reduction) from one fit on all rows, as a share of total gain. | Descriptive of that fit. Not the business model’s SHAP, and not a causal ranking. |
| Do two nonparametric distances agree? | Unbiased MMD², radial-basis kernel, median-heuristic bandwidth; energy distance. Permutation p-values. Continuous columns are source-median-imputed and source-standardized. Categoricals are one-hot of the top 10 source levels. Each group is capped at 2,000 rows. | These do not depend on trees. The row cap limits power. They cannot turn the verdict red. |
| Did rank correlations change? | Spearman matrices. Permutation test on the Frobenius norm of the upper triangle. Per pair, Fisher z with variance 1.06/(n−3), then BH. Capped at 20,000 rows. | A global norm can hide one large pair change, and one pair can be a term the model uses. |
| Are values outside the training tails? | Share of new values outside the source 0.5%–99.5% quantiles, one-sided binomial test against the expected tail share, BH across features. Also the share beyond the source min and max. | Quantiles, so one training outlier does not define the support. |
| What share of new rows has no training neighbours? | Isolation Forest fitted on 70% of source rows. The threshold is the 1% quantile of scores on the held-out 30%, so the null share is 1% by construction. One-sided binomial test. Capped at 100,000 rows. | A product of marginal ranges can still leave holes. |
| Did the score distribution change? | KS and PSI on predicted probabilities. Two-proportion z-test and Cohen’s h on the positive rate at the operating threshold, and the same test for the share of scores within ±0.05 of that threshold. | A small location shift can move the approval rate without a large PSI. The two questions are separate signals. |
| Did the reasons for the scores change? | TreeSHAP via `pred_contrib`, at most 5,000 rows per population. KS on contributions. Spearman correlation of mean absolute SHAP. | A similar score histogram can be produced by different features. |

### Performance without labels

If these rows disagree, do not quote one of them as the performance on the new population.

| Estimator | What it computes | Assumption |
|---|---|---|
| Calibration | Brier, ECE on 10 equal-width bins, slope and intercept of a logistic regression of the label on logit(score). | Computed on the labelled training file. Optimistic if those rows were used to fit the model. ECE > 0.05 downgrades CBPE to `info`. |
| Importance weighting | Domain-classifier out-of-fold scores become weights `w = p/(1−p) × n_source/n_target`, clipped at the 0.99 quantile. Kish effective sample size. Weighted AUC, Brier, and log loss (binary) or RMSE and MAE (regression), with a bootstrap interval, compared with the unweighted metrics on the same rows. | Covariate shift only: P(y\|x) is unchanged and the domain classifier estimates the density ratio. The factor `n_source/n_target` converts classifier odds into a density ratio when the groups differ in size. |
| CBPE | Expected AUC from pairs of predicted probabilities (ties count as 0.5). The reference is the same functional on the source scores, reported next to the realized source AUC. | Calibrated probabilities and no concept shift. Using the same functional on both populations avoids calling a calibration defect a population shift. |
| BBSE | Source confusion matrix C, new prediction rates μ, weights C⁻¹μ, prior renormalized. Interval by resampling both populations. | P(x\|y) is unchanged (label shift). Under covariate shift this estimator is the wrong model. |

## Configuration a user is expected to set

| Field | Default | Set it from |
|---|---|---|
| `decision_threshold` | 0.5 | The cutoff the deployed model uses |
| `performance_tolerance` | 0.02 | The AUC drop, or relative error increase, that should count as material |
| `problem_type` | `binary` | `binary` or `regression` |
| `n_rounds` | 30 | Leave at 30 or higher for a decision |
| `subset_size` | required for a matched baseline | Row count of the new population |
| `seed` | 42 | Fix and record it |
| `n_classifier_permutations` | 20 | Smallest permutation p-value is 1/21. Raising this refits the domain classifier model each time |
| `n_bootstrap` / `n_permutations` | 200 / 100 | Comparison run only; baseline rounds set these to 0 |

Row caps: `max_kernel_rows` 2,000, `max_correlation_rows` 20,000, `max_classifier_rows` 200,000, `max_shap_rows` 5,000, `max_iforest_rows` 100,000. A shift that is visible only above a cap can be missed by that task.

## Outputs

`runs/baseline/baseline_statistics.csv` columns: `round`, `task`, `table`, `feature`, `metric`, `value`.

`runs/baseline/baseline_summary.csv` columns: `task`, `table`, `feature`, `metric`, `n_rounds`, `mean`, `std`, `p50`, `p90`, `p95`, `p99`, `max`.

`runs/comparison/baseline_comparison.csv` adds, for each observed statistic, the baseline median, the 95th-percentile threshold, the maximum, the empirical p-value, and `exceeds_baseline`.

`runs/comparison/decision.json` holds `verdict`, `verdict_level`, `verdict_rule`, `signals`, `warnings`, and the baseline path, round count, and subset size.

`runs/comparison/tables/<task>__<table>.csv` is the full table for that task. Global rows use `feature="__global__"`. Registered comparison metrics are oriented so that larger means more drift (absolute SMD, absolute log variance ratio, `1 − ρ`, `1 − ESS ratio`, degradation).

`metadata.json` records sizes, the feature list and types, whether a model and a label were supplied, the full config, failed tasks, elapsed time, and library versions.

## References

- Austin, P. C. (2009). Using the standardized difference to compare the prevalence of a binary variable between two groups. *Communications in Statistics — Simulation and Computation*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Friedman, J. (2003). On multivariate goodness-of-fit and two-sample testing.
- Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic curve. *Radiology*.
- Lopez-Paz, D. and Oquab, M. (2017). Revisiting classifier two-sample tests. ICLR.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. PSI conventions 0.10 and 0.25.
