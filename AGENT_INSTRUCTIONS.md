# Agent build instructions: population drift analysis for model transferability

You are building (or extending) a Python toolkit that decides whether a LightGBM model trained on a
**training population** can be used on a **new population** whose target labels are **not available**.
The reference implementation in this folder (`population_drift.py`, `drift_report.py`) follows this
specification; keep the two in sync when you change either.

---

## 1. Problem statement and constraints

- Model: LightGBM (Booster or sklearn API), about 20 input features, binary classification or regression.
- Data: training population (features, optionally labels); new population (features only).
- Goal: compare the two populations on the model's features with statistical rigor, and produce a
  defensible **similar / use with caution / different** decision plus a readable HTML report.
- Without labels, **concept shift (change in P(y|x)) is undetectable**. Every output must say so, and all
  performance estimates must state their assumption.
- Every comparison must be positioned against a **baseline null**: the same statistic computed between
  random subsets of the training population. Fixed rules of thumb are secondary.

## 2. Deliverables

| File | Content |
|---|---|
| `population_drift.py` | Config and result classes, one function per task, orchestrator, baseline pipeline, comparison pipeline, interpretation, console summary, CLI (`baseline`, `compare`) |
| `drift_report.py` | Report builder: comparison report and baseline-only report; CLI |
| `make_synthetic_data.py` | Synthetic populations with planted shifts and a model for demos/tests |
| `tests/test_population_drift.py` | Unit tests of statistical building blocks and end-to-end tests |
| `requirements.txt`, `README.md` | Pinned dependencies, usage |

Dependencies: numpy, pandas, scipy, scikit-learn, lightgbm, statsmodels, matplotlib, joblib, pyarrow,
pytest. Do not add heavy optional dependencies (shap, alibi-detect, nannyml): TreeSHAP comes from
LightGBM's `pred_contrib=True`, and MMD/CBPE are implemented directly.

## 3. Code standards

- Python ≥ 3.10, type hints everywhere, Google-style docstrings with Args/Returns/Raises for public
  functions, module-level `logging` (no `print` except the console summary and the CLI).
- Validate inputs early and raise `ValueError` / `FileNotFoundError` with actionable messages.
- **One function per task**, each with the signature pattern
  `task(source, target, columns, [feature_types], config) -> TaskResult`
  (model tasks take scores or model-ready frames instead). No function may run more than one task.
- Every random operation takes its seed from `DriftConfig.seed` (derived seeds per round/permutation).
- A failing task must not abort the analysis: the orchestrator logs the error and records it in
  `metadata["failed_tasks"]`.

## 4. Data contracts

### 4.1 Configuration

`Thresholds` (dataclass) holds every rule-of-thumb threshold; `DriftConfig` (dataclass) holds inference
settings and a `Thresholds` instance. Validate in `__post_init__`. Required fields and defaults:

| Field | Default | Meaning |
|---|---|---|
| `alpha`, `correction_method` | 0.05, `fdr_bh` | significance level; BH / Holm / Bonferroni across features |
| `n_bootstrap`, `n_permutations`, `n_classifier_permutations` | 200, 100, 20 | resampling budgets |
| `cv_folds` | 5 | domain classifier CV |
| `psi_bins`, `discrete_max_unique`, `min_expected_count` | 10, 10, 5 | binning, categorical inference, chi-square merging |
| `equivalence_margin_smd` | 0.10 | TOST margin in pooled SD |
| `range_quantiles`, `ood_quantile` | (0.005, 0.995), 0.01 | support checks |
| `max_kernel_rows`, `max_correlation_rows`, `max_classifier_rows`, `max_shap_rows`, `max_iforest_rows` | 2000, 20000, 200000, 5000, 100000 | subsampling caps |
| `weight_clip_quantile` | 0.99 | density-ratio clipping |
| `decision_threshold`, `problem_type`, `performance_tolerance` | 0.5, `binary`, 0.02 | model settings; tolerance is absolute AUC (binary) or relative error (regression) |

`DriftConfig.for_baseline()` returns a copy with all bootstrap/permutation budgets set to 0; baseline
rounds only need point statistics.

### 4.2 Results

- `TaskResult(name, tables: dict[str, DataFrame], plot_data: dict, arrays: dict, notes: list[str])`.
  `tables["main"]` is the primary table. Per-feature tables have a `feature` column; global tables have
  one row with `feature="__global__"`. `plot_data` must be JSON-serialisable and compact (quantiles and
  histogram counts, never raw rows). `arrays` hold in-memory values reused by later tasks and are not saved.
- `DriftAnalysisResult(tasks, metadata)` with `table(task, name="main")` returning an empty frame when absent.
- `COMPARABLE_METRICS`: registry `task -> table -> (key column, metric columns)` of the statistics compared
  with the baseline. **Every registered metric is oriented so that larger = more drift** (use |SMD|,
  |log variance ratio|, 1 − ρ, 1 − ESS ratio, degradation).

### 4.3 Files written

```
<baseline_dir>/baseline_statistics.csv  (round, task, table, feature, metric, value)
<baseline_dir>/baseline_summary.csv     (task, table, feature, metric, n_rounds, mean, std, p50, p90, p95, p99, max)
<baseline_dir>/baseline_metadata.json   (n_rounds, subset_size, n_source, feature_cols, feature_types, config, ...)
<comparison_dir>/tables/<task>__<table>.csv
<comparison_dir>/plot_data.json, metadata.json
<comparison_dir>/baseline_comparison.csv (observed, baseline median/threshold/max, empirical_p, exceeds_baseline)
<comparison_dir>/feature_assessment.csv
<comparison_dir>/decision.json           (verdict, verdict_level, verdict_rule, signals, warnings, baseline info)
```

Read `task`, `table`, `feature`, `metric` columns back as strings.

## 5. Task functions (one each)

| # | Function | Inputs | Method and required outputs |
|---|---|---|---|
| 1 | `validate_inputs` | source, target, features | Raise on empty data, duplicate or missing columns. Return a schema table: dtypes, dtype compatibility, n unique, missing rates, constant flag. |
| 2 | `infer_feature_types` | source, features, declared categoricals | Categorical if declared, non-numeric/bool, or numeric with ≤ `discrete_max_unique` levels (KS is invalid with heavy ties). |
| 3 | `compare_continuous_distributions` | continuous features | Per feature: n, means, medians, SDs; SMD (pooled SD) with analytic 95% CI; variance ratio and abs log ratio; KS (D, p); Anderson–Darling k-sample (stat, p; SciPy caps p to [0.001, 0.25]); Cramér–von Mises (stat, p); Wasserstein / source SD; PSI on source-quantile bins; bootstrap CIs for Wasserstein and PSI when `n_bootstrap > 0`. Add `<p>_adj` columns. Store 101 quantiles per population for ECDF plots. |
| 4 | `run_equivalence_tests` | continuous features | TOST (`statsmodels.ttost_ind`, unequal variance) with margin `equivalence_margin_smd × pooled SD`; `equivalent = p < alpha`. Do not name it `test_*` (pytest would collect it). |
| 5 | `compare_categorical_distributions` | categorical features | Normalise labels so 1 and 1.0 match; chi-square homogeneity after merging levels with expected count < `min_expected_count`; Cramér's V; Jensen–Shannon distance (base 2); PSI over levels (ε = 1e-4); unseen-level count and share; top levels. Store top-12 level frequencies (+ "other"). |
| 6 | `compare_missingness` | all features | Missing rates, difference, Cohen's h, two-proportion z-test (p = 1 when degenerate), adjusted p. |
| 7 | `compare_missingness_patterns` | all features | Joint missing patterns (bit-packed rows), chi-square with merging, JS distance, complete-case shares, top-20 pattern table. |
| 8 | `classifier_two_sample_test` | all features + types | LightGBM domain classifier (200 trees, lr 0.05, 31 leaves, subsample/colsample 0.8), stratified K-fold **out-of-fold** AUC; stratified bootstrap CI; Mann–Whitney p on OOF scores; permutation p with `n_classifier_permutations` full CV refits, p = (1 + #null ≥ obs)/(1 + B); gain importance of a full fit. `arrays`: source row positions, source OOF scores, group sizes (for density-ratio weights). Categoricals as pandas categoricals with shared categories. |
| 9 | `mmd_test` | all features + types | Encode (continuous: source-median imputation and source standardisation; categorical: one-hot of top-10 source levels incl. missing). Subsample to `max_kernel_rows` per group. RBF kernel, median-heuristic bandwidth, unbiased MMD², permutation p. Compute each permutation with one matrix-vector product (block sums), not by re-indexing the kernel. |
| 10 | `energy_distance_test` | all features + types | Same encoding; energy distance 2E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖ (U-statistic), scaled statistic, permutation p. |
| 11 | `compare_correlation_structure` | continuous features | Spearman matrices; global Frobenius norm of the upper-triangle difference with permutation p; per pair: ρ in each population, Δρ, Fisher z-test with Spearman variance 1.06/(n−3), adjusted p. Store both matrices. |
| 12 | `check_range_coverage` | continuous features | Share of new values below/above the source [0.5%, 99.5%] quantiles; expected share; ratio to expected; share beyond source min/max; one-sided binomial test; adjusted p. |
| 13 | `detect_joint_outliers` | all features + types | Isolation Forest fitted on 70% of source rows; threshold = `ood_quantile` of scores on the held-out 30%; new-population OOD share; one-sided binomial test vs `ood_quantile`. Store score histograms and the threshold. |
| 14 | `compare_prediction_distributions` | source/target scores | KS, PSI, means; binary: positive rate at threshold (Cohen's h, z-test) and share of predictions within ±0.05 of the threshold (z-test). Store histograms. |
| 15 | `compare_shap_distributions` | model, model-ready frames | TreeSHAP via `pred_contrib=True` (drop the bias column) on ≤ `max_shap_rows` rows per population; per feature mean \|SHAP\|, importance share and ranks, relative change, KS on contributions (adjusted); `important` = top features reaching 80% cumulative share; summary: Spearman ρ of importances and 1 − ρ. |
| 16 | `assess_calibration` | source scores and labels (binary) | Brier, ECE (10 equal-width bins), calibration slope/intercept (logistic regression on logit p), quantile reliability table. |
| 17 | `compute_density_ratio_weights` | domain OOF scores | w = p/(1−p) × n_source/n_target, clipped at `weight_clip_quantile`; Kish ESS, ESS ratio, ESS deficit, clipping info. |
| 18 | `estimate_importance_weighted_performance` | source scores, labels, domain scores | Unweighted reference vs weighted estimate for AUC/Brier/log-loss (binary) or RMSE/MAE (regression); bootstrap CIs; `degradation` oriented positive = worse; summary with ESS. |
| 19 | `estimate_performance_cbpe` | source scores (+labels), target scores | Expected AUC (ratio of expectations over pairs, self-pairs removed, ties = 0.5) and expected accuracy/precision/recall from probabilities. Reference = CBPE on the source (like-for-like); also report the realised source metric and the calibration gap; bootstrap CI on the target. |
| 20 | `estimate_label_shift_bbse` | source scores, labels, target scores | Black Box Shift Estimation: confusion matrix C on source, target prediction rates μ, w = C⁻¹μ, target prior q = w ⊙ p_source (clipped, normalised); bootstrap CI resampling both populations; `prior_shift_detected` when the CI excludes the source prevalence; condition number of C. |

Helpers to provide: `load_table` (csv/parquet/feather, sentinel → NaN), `load_model` (Booster `.txt`/`.model`,
joblib/pickle), `prepare_model_input` (model feature order, object → category; LightGBM re-maps pandas
categories to training categories), `predict_scores`, `predict_contributions`, `adjust_pvalues`,
`population_stability_index`.

## 6. Orchestrator: `population_drift_analyzer`

`population_drift_analyzer(source, target, feature_cols, categorical_cols=None, model=None, label_col=None,
config=None) -> DriftAnalysisResult`

1. Validate inputs (also that `label_col` exists and is not a feature) and infer feature types.
2. Run tasks 3–13 on every call.
3. If a model is supplied: predict scores on both populations, run tasks 14–15; if labels are supplied run
   16, 18 (with weights from task 8's arrays) and 20; for binary problems run 19.
4. Metadata: timestamps, sizes, features and types, label column, model flag, full config, failed tasks,
   task notes, elapsed time, library versions.

## 7. Baseline pipeline: `build_baseline`

`build_baseline(source, feature_cols, categorical_cols, model, label_col, n_rounds=30, subset_size=None,
subset_fraction=0.3, config=None, output_dir=None) -> BaselineResult`

- Each round: random permutation of the training rows; the first `subset_size` rows play the **new**
  population, the rest play the training population; run the analyzer with `config.for_baseline()` and a
  round-specific seed; collect `COMPARABLE_METRICS` into the long table.
- Validate 30 ≤ subset_size ≤ n − 30 and n_rounds ≥ 2. Recommend `subset_size` ≈ expected new-population
  size: statistics such as PSI, KS D and classifier AUC are biased upward in small samples.
- Summarise per (task, table, feature, metric): count, mean, sd, p50, p90, p95, p99, max. Save statistics,
  summary and metadata (including failed-task counts).

## 8. Comparison pipeline: `run_comparison`

- Load the baseline (or build a size-matched one in `<out>/baseline` when none is given). Raise if the
  baseline feature set differs. Warn when new size / baseline subset size is outside [0.5, 2], when the
  training population is too small to match the size, and when model availability differs.
- Run the analyzer with the full config, save everything, then `compare_to_baseline`: for each observed
  statistic, the baseline threshold (`baseline_percentile`, default 95), median, max, empirical
  p = (1 + #null ≥ observed)/(1 + rounds), and `exceeds_baseline`.
- Interpret (Section 9), write `decision.json` and `feature_assessment.csv`, print the console summary.

## 9. Interpretation rules (`interpret_results`)

### 9.1 Feature severity (`assess_features`)

For each feature, with "beyond baseline" = any listed metric exceeds its baseline threshold (True if no
baseline exists):

| Type | Significant | Beyond baseline on | Large (red) | Moderate (amber) |
|---|---|---|---|---|
| Continuous | KS or AD adjusted p < α | KS D, PSI, Wasserstein | PSI ≥ 0.25 or KS D ≥ 0.20 or \|SMD\| ≥ 0.5 | PSI ≥ 0.10 or KS D ≥ 0.10 or \|SMD\| ≥ 0.10 |
| Categorical | chi-square adjusted p < α | Cramér's V, PSI, JS | PSI ≥ 0.25 or V ≥ 0.30 | PSI ≥ 0.10 or V ≥ 0.10 or unseen share > 1% |
| Missingness (any) | z-test adjusted p < α | Cohen's h | \|h\| ≥ 0.5 | \|h\| ≥ 0.2 |

red = significant ∧ beyond ∧ large; amber = significant ∧ beyond ∧ moderate; green otherwise.
A feature takes the worst of its distribution and missingness levels. Record human-readable reasons,
TOST equivalence and model importance. Without a model, treat every feature as important.

### 9.2 Signals

Each signal: `{signal, level ∈ green/amber/red/info, value, detail, important}`.

| Signal | Rule |
|---|---|
| Feature drift, important features | worst severity among important features (`important=True`) |
| Feature drift, other features | amber if any flagged (never red) |
| Missing-value rates | amber if any \|h\| ≥ 0.2, significant and beyond baseline; red if \|h\| ≥ 0.5 on an important feature |
| Unseen category levels | amber if any share > 1% |
| Classifier two-sample test | green if within baseline or AUC ≤ 0.55; amber ≤ 0.80; red > 0.80 |
| Kernel/energy tests | amber if any test has p < α / (number of tests) and is beyond baseline |
| Dependence structure | amber if the Frobenius norm is beyond baseline and significant, or an important pair has \|Δρ\| > 0.2 with adjusted p < α |
| Out-of-support rows | green if within baseline or < 2%; amber ≤ 5%; red > 5% |
| Values outside training range | amber if an important feature has > 2× expected share, adjusted p < α, beyond baseline |
| Model score distribution | green if within baseline or PSI < 0.10; amber ≤ 0.25; red > 0.25 (`important=True`) |
| Predicted positive rate | amber if \|h\| ≥ 0.2 and p < α |
| SHAP attributions | amber if rank ρ < 0.8, or an important feature has SHAP KS ≥ 0.1, adjusted p < α, beyond baseline |
| Effective sample size | green > 0.5, amber 0.1–0.5, red < 0.1 |
| Estimated performance (weighting, CBPE) | red if the whole 95% CI is beyond tolerance; amber if the point estimate is beyond tolerance, or the CI crosses it and degradation > tolerance/2; else green. CBPE becomes `info` when source ECE > 0.05 |
| Class prior (BBSE) | amber if the CI excludes the source prevalence (state that it assumes P(x\|y) unchanged) |
| Source calibration | `info` |

Multiplicity, the baseline and minimum effect sizes are combined deliberately: with large samples nearly
every test is significant, and with small samples the baseline is wide. Check the false-alarm rate on
same-distribution data whenever you change a rule (Section 11).

### 9.3 Verdict

- **red**, "Different: do not use the model as is": any red signal.
- **green**, "Similar: the model can be used (with monitoring)": no red, ≤ 2 amber, and no amber signal
  with `important=True`.
- **amber**, "Use with caution: mitigation and verification needed": otherwise.

`info` signals are shown but not counted. The rule text is stored in `verdict_rule`.

## 10. Report requirements (`drift_report.py`)

`build_report(output_path, comparison_dir=None, baseline_dir=None, title=None, top_k=6)`; one standalone
HTML file (inline CSS, PNG plots embedded as base64, no external resources), with a sticky table of contents,
traffic-light badges and collapsible `<details>` for full tables. The report should be comprehensive and
concise: key plots and 2–4 sentences of generated interpretation per section, full tables collapsed.

Comparison report sections, in order:

1. **Executive summary**: verdict banner and rule, context (sizes, feature types, model, labels, baseline
   rounds and subset size), signal table sorted by severity, key findings (flagged features with reasons,
   joint separability, OOD share, performance estimates, prior), caveats (concept shift, label rows,
   run warnings), how-to-read box.
2. **Feature-level drift**: PSI bar chart coloured by severity with baseline-p95 ticks and 0.10/0.25 lines;
   drift-vs-importance scatter (if model); feature assessment table.
3. **Continuous features**: ECDF overlays for the top-k drifted features; full table with SMD CI, variance
   ratio, KS/AD/CvM adjusted p, Wasserstein, PSI CI, baseline p95, TOST.
4. **Categorical features**: grouped frequency bars for the top-k; full table.
5. **Missing values**: missing-rate bars (features with any missing values); tables of rates and patterns.
6. **Joint distribution**: domain-score histograms, AUC vs baseline null histogram, top separation drivers;
   MMD/energy observed-vs-null panels; tables.
7. **Dependence structure**: Δρ heatmap (annotated when ≤ 16 features); top pair changes.
8. **Support**: Isolation Forest score histograms with threshold; out-of-range share bars with expected line.
9. **Model behaviour**: score histograms with threshold; mean |SHAP| bars (important marked); tables.
10. **Estimated performance**: reference-vs-estimate CI panels per metric; reliability diagram; text stating
    each method's assumption; tables.
11. **Baseline reference**: grid of observed-vs-null histograms for all global statistics; full baseline
    comparison table.
12. **Methods and thresholds**: compact table built from the run's config.

Baseline-only report: overview banner, global-statistic null histograms, per-feature PSI p50/p95 chart,
summary table, methods. Explain the next step (`compare --baseline`).

Plot rules: training = blue `#2563eb`, new = orange `#ea580c`, severity colours green/amber/red; readable
tick counts (`MaxNLocator`), figure-level legends for grids, no math-text rendering.

## 11. Testing and acceptance criteria

Unit tests (fast, deterministic seeds):

- PSI ≈ 0 for identical distributions and > 0.25 for a 1 SD shift.
- Expected AUC from calibrated probabilities matches realised AUC within 0.01 (n = 40k).
- BBSE recovers a prior shift 0.3 → 0.6 within 0.03.
- Density-ratio weights are uniform (ESS ratio 1) when domain scores equal the base rate.
- MMD and energy tests reject a 0.8 SD shift and do not reject identical distributions.
- Continuous, categorical and missingness tasks detect the planted shifts of `make_synthetic_data.py`.
- Validation raises on missing columns; low-cardinality numeric columns are categorical.

End-to-end tests: baseline (≥ 4 rounds) + comparison + both reports, for (a) a shifted population →
verdict red, f01 red, f07 not green; (b) a same-distribution population → no red feature, verdict green or
amber.

Acceptance:

- `pytest -q` passes.
- Demo (`make_synthetic_data.py` → baseline 20 rounds → compare) flags f01, c01 (red), f02 (variance) and
  f07 (missingness), c02 (unseen level), f05–f06 (correlation), and does **not** flag f10 (negligible shift).
- False-alarm check: over ≥ 10 seeds of same-distribution data, no red verdict and no red feature.
- Reports render without external resources; every plot has a title and axis labels; no table cell
  wraps to more than a few lines (wide text columns get a minimum width).

## 12. Extension points

- Additional problem types (multiclass): extend `predict_scores`, `_metric_specs`, CBPE and BBSE (k × k C).
- Time-aware baseline: compare training sub-periods where the model was known to perform acceptably and
  register the result as an "operational tolerance" next to the random-split baseline.
- Labelled verification sample: when some new-population labels arrive, add a task that measures realised
  performance with bootstrap CIs and calibration, and compares it to the label-free estimates.
- Monitoring: schedule `compare` on each new batch with a fixed baseline and alert on signal changes.
