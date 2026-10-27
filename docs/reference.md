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
| Continuous and categorical per-feature distributions, equivalence, missingness, domain classifier, MMD, energy distance, correlations, range, isolation forest | Fit set vs new |
| Score distribution, positive rate, TreeSHAP, importance weighting, CBPE, BBSE | Evaluation set vs new |
| Calibration | Evaluation set |

The weighting classifier is fit on the evaluation set versus the new population, because those weights reweight evaluation rows. It is stored separately from the fit-set domain classifier. Levels are `severe`, `moderate`, and `mild`. A reference shorter than `n_new + 30` raises before the analysis starts. Each reference has its own size-matched baseline for the tests that use it.

## Null

Each baseline round permutes one reference (the fit set, or the evaluation set). The first `subset_size` rows stand in for the new population; the rest stand in for that reference. The same point statistics are computed. Baseline rounds do not run bootstrap or permutation tests.

A statistic exceeds the baseline when it is larger than the baseline 95th percentile (`Thresholds.baseline_percentile`). The empirical p-value is `(1 + count of rounds at least as large) / (1 + rounds)`. With 30 rounds the smallest p-value is 1/31.

`subset_size` must match the new population’s row count. Population Stability Index (PSI), Kolmogorov–Smirnov statistic (KS D), and domain-classifier area under the curve (AUC) shrink as sample size grows. The code warns when the new size is outside half to twice the baseline subset size, and it refuses a baseline built on a different feature set.

This null is sampling noise under exchangeability inside that reference. Exchangeability means any row in the reference can stand in for any other. The two sides of a random split are then not systematically distinguishable, and the 95th percentile is the distinguishability that the split alone still produces. The new population exceeds the baseline when it is more distinguishable from the reference than those two pieces are from each other. This is not the noise of drift already present inside the reference window. If that period itself moved, the baseline is wide and a further shift can look ordinary.

## Feature severity

Benjamini–Hochberg (BH) adjustment is applied inside a test family, across features, at α = 0.05 (`DriftConfig.alpha`, `correction_method`). It is not applied across the signal list.

A feature is severe only when three conditions hold: the test is significant after that adjustment, the statistic exceeds its baseline 95th percentile, and the effect is large. A moderate level uses the same significance and baseline conditions with a moderate effect. Otherwise the feature is mild. The feature’s level is the worse of its distribution level and its missingness level. Without a model, every feature is treated as important.

| Type | Significant | Beyond baseline on | Large | Moderate |
|---|---|---|---|---|
| Continuous | KS or AD adjusted p < α | KS D, PSI, Wasserstein/SD | PSI ≥ 0.25 or KS D ≥ 0.20 or \|SMD\| ≥ 0.50 | PSI ≥ 0.10 or KS D ≥ 0.10 or \|SMD\| ≥ 0.10 |
| Categorical | chi-square adjusted p < α | Cramér’s V, PSI, JS | PSI ≥ 0.25 or V ≥ 0.30 | PSI ≥ 0.10 or V ≥ 0.10 or unseen-level share > 1% |
| Missingness | two-proportion z adjusted p < α | Cohen’s h | \|h\| ≥ 0.50 | \|h\| ≥ 0.20 |

The effect-size cutoffs are conventions: PSI 0.10 / 0.25 (Siddiqi 2006), SMD and Cohen’s h (Cohen 1988; Austin 2009 for the 0.10 mean-difference benchmark), Cramér’s V 0.10 / 0.30. They are not estimated from the user’s data. Change them in `Thresholds` before looking at the new population, not after.

Low-importance features are capped at moderate in the signal list, so a large shift in a feature the model barely uses cannot by itself force a severe verdict. The feature table still shows that feature’s own severity.

Model-important features are the smallest set whose mean absolute TreeSHAP on the reference used in that comparison reaches 80% of the total (`important_feature_cumulative_share`). On the fit set those contributions are in-sample.

## Signals

`info` signals are reported and are not counted in the verdict.

| Signal | Rule |
|---|---|
| Feature drift, model-important features | Worst severity among important features. This signal can be severe. |
| Feature drift, other features | Moderate if any non-important feature is flagged. Never severe. |
| Missing-value rates | Moderate if any \|h\| ≥ 0.2, significant, and beyond baseline. Severe if \|h\| ≥ 0.5 on an important feature. |
| Unseen category levels | Moderate if any feature has an unseen level in more than 1% of new rows. |
| Domain classifier model | Mild if within baseline or AUC ≤ 0.55. Moderate if AUC ≤ 0.80. Severe if AUC > 0.80. |
| Kernel and energy tests | Moderate if either test has permutation p < α/2 and exceeds its baseline. Cannot be severe. |
| Dependence | Moderate if the Spearman Frobenius norm is significant and beyond baseline, or an important pair has \|Δρ\| > 0.20 with adjusted p < α. |
| Out-of-support rows | Mild if within baseline or share < 2%. Moderate up to 5%. Severe above 5%. |
| Values outside the training range | Moderate if an important feature has more than twice the expected share outside the training 0.5%–99.5% quantiles, adjusted p < α, and beyond baseline. |
| Score distribution | Mild if within baseline or PSI < 0.10. Moderate if PSI ≤ 0.25. Severe if PSI > 0.25. |
| Predicted positive rate | Moderate if \|Cohen’s h\| ≥ 0.20 and the two-proportion p-value < α, at `decision_threshold`. |
| SHAP attributions | Moderate if the rank correlation of mean \|SHAP\| is below 0.80, or an important feature has contribution KS ≥ 0.10, adjusted p < α, and beyond baseline. |
| ESS | Mild if the Kish ESS ratio > 0.50. Moderate from 0.10 to 0.50. Severe below 0.10. |
| Importance-weighted performance and CBPE | Severe only if the whole 95% CI is beyond `performance_tolerance` (default 0.02 absolute AUC, or relative error for regression). Moderate if the point estimate is beyond tolerance, or the interval crosses it and the degradation exceeds half the tolerance. CBPE is downgraded to `info` when source ECE > 0.05. |
| Class prior (BBSE) | Moderate if the interval excludes the training prevalence. Valid only if P(x\|y) is unchanged. |
| Source calibration | `info`. ECE and calibration slope on the labelled training file. |

Verdict:

- **severe** — “Different: do not use the model as is” if any counted signal is severe.
- **mild** — “Similar: the model can be used (with monitoring)” if there is no severe signal, at most two moderate signals, and no moderate signal with `important=True`. “Monitoring” means a later rerun of this comparison. The repository does not schedule one.
- **moderate** — “Use with caution” otherwise.

On 10 synthetic pairs drawn from the same distribution (2,500 vs 1,000 rows, 10-round baselines) the verdict was never severe, and 3 of 10 were moderate because one weak signal crossed its threshold. That check is not the operating configuration. Treat a single weak moderate signal as a review item, not as a demonstrated shift.

## Methods

One function per question. A failing task is logged in `metadata["failed_tasks"]` and does not stop the others. Random draws use `DriftConfig.seed` (default 42).

| Question | Procedure | Why this procedure |
|---|---|---|
| Did a continuous per-feature distribution change? | Two-sample KS, AD k-sample, CvM. BH on each p-value family. | The three tests weight the distribution differently. KS D is also the effect size in the severity rule. |
| Did the mean change by a practical amount, and are the means equivalent? | SMD, pooled SD with an analytic 95% CI. TOST, unequal variance, margin 0.10 pooled SD. | SMD is the scale used in the severity rule. TOST is evidence for similar means. It says nothing about variance or shape. |
| Did scale or bin masses change? | Variance ratio, Wasserstein distance divided by the source SD, PSI on source-quantile bins (10 bins). Bootstrap intervals on the comparison run (200 resamples). | PSI is the conventional stability summary. It is not used as a p-value. |
| Did a categorical per-feature distribution change? | Chi-square after merging levels with expected count below 5. Cramér’s V, JS (base 2), PSI. Labels `1` and `1.0` are treated as the same level. | Chi-square is the test of a shared discrete distribution. V does not grow with sample size. |
| Will the model see an untrained level? | Share of new rows whose level is absent from training. | LightGBM sends an unseen level down a default path. |
| Did missingness change? | Two-proportion z-test and Cohen’s h per feature. Separately, chi-square and JS on the joint missingness pattern. | Observed values can match while the missing rate changes, and LightGBM’s missing branch then applies to a different fraction of rows. |
| Can any function of the features tell the populations apart? | Domain classifier model: LightGBM, 200 trees, learning rate 0.05, 31 leaves, row and column subsampling 0.8, L2 penalty 1. Label is population membership, never the business target. Stratified 5-fold OOF scores. Report pooled OOF ROC AUC (`auc`, used for baseline and verdict) and the mean of fold AUCs (`cv_auc`) with a LeDell et al. (2015) influence-curve 95% CI (see below). | Per-feature tests miss a shift that exists only as a combination. The AUC does not depend on a cutoff and stays 0.5 when the score distributions match, which matters because the group sizes differ. OOF scoring stops the trees being graded on rows they just fitted. Hyperparameters are fixed, so the AUC is not the winner of a model search (Friedman 2003; Lopez-Paz and Oquab 2017). |
| Could that AUC arise under no shift? | Two nulls. Shuffle the population label and refit the cross-validation (`n_classifier_permutations`, default 20). And the training-split baseline above. A Mann–Whitney test on the held-out scores is stored; for fixed scores it is the same comparison as AUC (Hanley and McNeil 1982) but it treats the scores as given, so the verdict uses the permutation p-value. | The permutation null breaks only the population label. The baseline asks whether the same AUC appears when both groups are random pieces of the training population. |
| Which inputs drive that separation? | Gain (total loss reduction) from one fit on all rows, as a share of total gain. | Descriptive of that fit. Not the business model’s SHAP, and not a causal ranking. |
| Do two nonparametric distances agree? | Unbiased MMD², RBF kernel, median-heuristic bandwidth; energy distance. Permutation p-values. Continuous columns are source-median-imputed and source-standardized. Categoricals are one-hot of the top 10 source levels. Each group is capped at 2,000 rows. | These do not depend on trees. The row cap limits power. They cannot turn the verdict severe. |
| Did rank correlations change? | Spearman matrices. Permutation test on the Frobenius norm of the upper triangle. Per pair, Fisher z with variance 1.06/(n−3), then BH. Capped at 20,000 rows. | A global norm can hide one large pair change, and one pair can be a term the model uses. |
| Are values outside the training tails? | Share of new values outside the source 0.5%–99.5% quantiles, one-sided binomial test against the expected tail share, BH across features. Also the share beyond the source min and max. | Quantiles, so one training outlier does not define the support. |
| What share of new rows has no training neighbors? | Isolation Forest fitted on 70% of source rows. The threshold is the 1% quantile of scores on the held-out 30%, so the null share is 1% by construction. One-sided binomial test. Capped at 100,000 rows. | A product of per-feature ranges can still leave holes. |
| Did the score distribution change? | KS and PSI on predicted probabilities. Two-proportion z-test and Cohen’s h on the positive rate at the operating threshold, and the same test for the share of scores within ±0.05 of that threshold. | A small location shift can move the approval rate without a large PSI. The two questions are separate signals. |
| Did the reasons for the scores change? | TreeSHAP via `pred_contrib`, at most 5,000 rows per population. KS on contributions. Spearman correlation of mean absolute SHAP. | A similar score histogram can be produced by different features. |

### Performance without labels

If these rows disagree, do not quote one of them as the performance on the new population.

| Estimator | What it computes | Assumption |
|---|---|---|
| Calibration | Brier, ECE on 10 equal-width bins, slope and intercept of a logistic regression of the label on logit(score). | Computed on the evaluation set in a dual run. Optimistic if those rows were used to fit the model. ECE > 0.05 downgrades CBPE to `info`. |
| Importance weighting | Domain-classifier OOF scores become weights `w = p/(1−p) × n_source/n_target`, clipped at the 0.99 quantile. Kish ESS. Weighted AUC, Brier, and log loss (binary) or RMSE and MAE (regression), with a bootstrap interval, compared with the unweighted metrics on the same rows. | Covariate shift only: P(y\|x) is unchanged and the domain classifier estimates the density ratio. The factor `n_source/n_target` converts classifier odds into a density ratio when the groups differ in size. |
| CBPE | Expected AUC from pairs of predicted probabilities (ties count as 0.5). The reference is the same functional on the source scores, reported next to the realized source AUC. | Calibrated probabilities and an unchanged map from features to labels. Using the same functional on both populations avoids calling a calibration defect a population shift. |
| BBSE | Source confusion matrix C, new prediction rates μ, weights C⁻¹μ, prior renormalized. Interval by resampling both populations. | P(x\|y) is unchanged (label shift). Under covariate shift this estimator is the wrong model. |

### Domain-classifier AUC and the influence-curve interval

The domain classifier produces one out-of-fold (OOF) score per row. Two AUC summaries are stored:

| Column | Definition | Role |
|---|---|---|
| `auc` | ROC AUC on the pooled OOF scores | Compared to the baseline 95th percentile and to the mild/moderate/severe bands. Unchanged from earlier releases. |
| `cv_auc` | Mean of the per-fold AUCs | Point estimate for the LeDell et al. (2015) interval (`cv_auc_se`, `cv_auc_ci_low`, `cv_auc_ci_high`). |

Under stratified folds of similar size the two points usually agree to the third decimal. The interval is attached only to `cv_auc`, because that is the estimator the derivation targets.

**What an influence curve is.** An influence curve (also called an influence function) is a function of a single observation that says how much that observation pulls an estimator. For many estimators used in statistics and machine learning, the estimation error is asymptotically a sample average of those pulls: after scaling by √n, the error converges to a normal law whose variance is the variance of the influence curve. Estimating that variance from the observed influence values on the data gives a standard error without refitting the model and without bootstrap resampling. One observation with a large influence value moves the estimate more; the spread of the influence values across the sample is the SE.

**What LeDell et al. compute.** They derive the influence curve of fold-wise AUC and average it across the V validation folds. Class rates use the full sample; ranking fractions that enter the influence curve stay inside each fold (each fold has its own fitted scorer). Ties count as 0.5 so the influence values match `sklearn.metrics.roc_auc_score`. The 95% interval is the mean fold AUC ± 1.96 × SE, clipped to [0, 1], with SE = √(mean of fold-wise mean(IC²) / n). This replaces the earlier stratified bootstrap of frozen OOF scores, which treated the scores as fixed and therefore answered a different question (uncertainty of the Mann–Whitney statistic given the scores, not uncertainty of the cross-validated AUC). Coverage in their simulations is about 0.92–0.93 at n = 1000 and about 0.94–0.95 at n ≥ 5000. The interval does not use `n_bootstrap`. Importance-weighted AUC and CBPE keep their bootstrap intervals; this influence curve does not cover those functionals.

## Configuration a user is expected to set

| Field | Default | Set it from |
|---|---|---|
| `decision_threshold` | 0.5 | The cutoff the deployed model uses |
| `performance_tolerance` | 0.02 | The AUC drop, or relative error increase, that should count as material |
| `problem_type` | `binary` | `binary` or `regression` |
| `n_rounds` | 30 | Leave at 30 or higher for a decision |
| `subset_size` | `n_new` inside `compare` | Row count of the new population. With `--test`, each reference is matched separately. |
| `seed` | 42 | Fix and record it. |
| `n_classifier_permutations` | 20 | Smallest permutation p-value is 1/21. Raising this refits the domain classifier model each time |
| `n_bootstrap` / `n_permutations` | 200 / 100 | Comparison run only; baseline rounds set these to 0. `n_bootstrap` drives PSI/Wasserstein, weighted performance, CBPE, and BBSE intervals — not the domain-classifier `cv_auc` CI (that uses the influence curve). |

Row caps: `max_kernel_rows` 2,000, `max_correlation_rows` 20,000, `max_classifier_rows` 200,000, `max_shap_rows` 5,000, `max_iforest_rows` 100,000. A shift that is visible only above a cap can be missed by that task.

## Outputs

`runs/baseline/baseline_statistics.csv` columns: `round`, `task`, `table`, `feature`, `metric`, `value`.

`runs/baseline/baseline_summary.csv` columns: `task`, `table`, `feature`, `metric`, `n_rounds`, `mean`, `std`, `p50`, `p90`, `p95`, `p99`, `max`.

`runs/comparison/baseline_comparison.csv` adds, for each observed statistic, the baseline median, the 95th-percentile threshold, the maximum, the empirical p-value, and `exceeds_baseline`.

`runs/comparison/decision.json` holds `verdict`, `verdict_level` (`severe`, `moderate`, or `mild`), `verdict_rule`, `signals` (each with a `pair`), `task_pairs`, `n_severe`, `n_moderate`, `warnings`, and the baseline path.

`runs/comparison/tables/<task>__<table>.csv` is the full table for that task. Global rows use `feature="__global__"`. Registered comparison metrics are oriented so that larger means more drift (absolute SMD, absolute log variance ratio, `1 − ρ`, `1 − ESS ratio`, degradation).

`metadata.json` records sizes, the feature list and types, whether a model and a label were supplied, the full config, failed tasks, elapsed time, and library versions.

## References

- Austin, P. C. (2009). Using the standardized difference to compare the prevalence of a binary variable between two groups. *Communications in Statistics — Simulation and Computation*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Friedman, J. (2003). On multivariate goodness-of-fit and two-sample testing.
- Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic curve. *Radiology*.
- LeDell, E., Petersen, M., and van der Laan, M. (2015). Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates. *Electronic Journal of Statistics*, 9, 1583–1607. doi:10.1214/15-EJS1035.
- Lopez-Paz, D. and Oquab, M. (2017). Revisiting classifier two-sample tests. International Conference on Learning Representations (ICLR).
- Siddiqi, N. (2006). *Credit Risk Scorecards*. PSI conventions 0.10 and 0.25.

## Appendix

### Tests

| Acronym | Meaning | Reading |
|---|---|---|
| KS | Kolmogorov–Smirnov | Two-sample test of one continuous distribution. Adjusted p below α: the distributions differ. Size is KS D. |
| AD | Anderson–Darling | Two-sample test with more weight in the tails than KS. Adjusted p below α: the tails differ. Not a severity size. |
| CvM | Cramér–von Mises | Two-sample test of the full ECDF gap, not only its maximum. Adjusted p below α: the distributions differ. Not a severity size. |
| TOST | two one-sided tests | Equivalence test for two means, margin 0.10 SD. p below α: means agree. Not a test of variance or shape. |
| BH | Benjamini–Hochberg | False-discovery adjustment inside one p-value family. Significant means adjusted p below α. |

| Test | Reading |
|---|---|
| Chi-square | Test that two samples share one discrete distribution. Levels with expected count below 5 are merged. Adjusted p below α: the category shares differ. Also used on the joint missingness pattern. Size is Cramér's V. |
| Two-proportion z-test | Test that two samples share a rate. Used for missingness, the predicted positive rate, and the share of scores within 0.05 of the threshold. Adjusted p below α: the rates differ. Size is Cohen's h. |
| Permutation test | Shuffle the population label and recompute the statistic. p = (1 + shuffles at least as large) / (1 + shuffles). Used for domain-classifier AUC, MMD², energy distance, and the Spearman-matrix Frobenius norm. |
| Mann–Whitney | Test that domain-classifier scores are higher for new rows than for training rows. For fixed scores this is the same comparison as AUC. The verdict uses the permutation p, because the scores were fitted on these rows. |
| Fisher z | Test that one Spearman correlation differs between the samples. The z variance is 1.06/(n−3) in each sample (Fieller, Hartley, and Pearson 1957). Adjusted p below α: that pair changed. |
| Binomial test | One-sided test that the new share outside the training 0.5–99.5% range, or the isolation-forest outlier share, exceeds the share expected from the training rule. Adjusted p below α: more new rows fall outside support. |
| Baseline empirical p | Share of random reference splits at least as large as the new population: (1 + such rounds) / (1 + rounds). Beyond baseline means the statistic exceeds that reference's 95th percentile. |

### Magnitudes

| Acronym | Meaning | Reading |
|---|---|---|
| KS D | Kolmogorov–Smirnov statistic | Largest vertical gap between the two ECDFs, from 0 to 1. 0.20 means the cumulative shares differ by 20 percentage points at that gap. Distance only: no sign, and it does not separate a mean shift from a spread or tail change. ≥ 0.10 moderate, ≥ 0.20 large, and only with adjusted p below α and a value above the baseline 95th percentile. The same reading applies to score KS D and SHAP KS D. |
| PSI | Population Stability Index | Change in bin shares. Below 0.10 stable, 0.10–0.25 monitor, above 0.25 large. |
| SMD | standardized mean difference | Mean gap divided by pooled SD. Absolute SMD ≥ 0.10 small, ≥ 0.50 large. |
| JS | Jensen–Shannon | Distance between category shares. 0 means the same frequencies; larger means less overlap. |
| MMD | maximum mean discrepancy | Kernel distance between the two samples. Larger means the joint distributions differ. Cannot be severe. |
| MMD² | squared maximum mean discrepancy | Square of MMD, the stored number. Same reading as MMD. |
| RBF | radial basis function | Kernel used for MMD. Bandwidth is the median distance between rows. |
| ECE | expected calibration error | Mean gap between predicted probability and observed rate. Above 0.05, probabilities are off and CBPE is info. |
| ESS | effective sample size | Kish ratio after weighting. Above 0.50 usable, 0.10–0.50 caution, below 0.10 the weights have collapsed. |
| SD | standard deviation | Spread of one column. SMD and Wasserstein/SD are in these units. |

### Scores

| Acronym | Meaning | Reading |
|---|---|---|
| AUC | area under the curve | Area under the ROC curve. 0.5 means no separation. Domain classifier pooled OOF AUC: ≤ 0.55 mild, ≤ 0.80 moderate, above 0.80 severe. |
| CV AUC | cross-validated area under the curve | Mean of the per-fold OOF AUCs (`cv_auc`). Target of the LeDell influence-curve CI. |
| ROC | receiver operating characteristic | True-positive rate against false-positive rate across cutoffs. AUC is its area. |
| SHAP | Shapley additive explanations | A feature's additive contribution to one score. Mean absolute SHAP rank correlation below 0.80: the ranking of reasons changed. |
| TreeSHAP | Shapley additive explanations from trees | SHAP for a tree model, from LightGBM pred_contrib. |
| OOF | out-of-fold | Score from a model that did not train on that row. |
| IC | influence curve | Function of one observation that gives the first-order effect of that row on an estimator. For domain-classifier CV AUC, the empirical variance of the IC yields the LeDell et al. (2015) standard error. |

### Estimates

| Acronym | Meaning | Reading |
|---|---|---|
| CBPE | confidence-based performance estimation | Expected metric if labels were drawn from the predicted probabilities. Requires calibrated scores. |
| BBSE | black-box shift estimation | Estimated new class prior. Usable only if P(x given y) is unchanged. |
| CI | confidence interval | Range for an estimate. Domain-classifier `cv_auc` uses an influence-curve CI; weighted performance, CBPE, and BBSE use bootstrap CIs. Performance is severe only when the whole interval lies past the tolerance. |
| RMSE | root mean squared error | Square root of mean squared error. Larger is worse. |
| MAE | mean absolute error | Average absolute error. Larger is worse. |
