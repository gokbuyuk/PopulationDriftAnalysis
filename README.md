# Population drift toolkit

Decide whether a LightGBM model trained on one population can be used on a new population **without
target labels** for the new population. The toolkit compares the two populations feature by feature and
jointly, checks out-of-support rows, measures how the model's outputs and attributions change, estimates
performance without labels, and positions every statistic against a **baseline**: the same statistic
computed between random subsets of the training population (pure sampling noise).

| File | Purpose |
|---|---|
| `population_drift.py` | One function per test, the orchestrator `population_drift_analyzer`, the baseline pipeline `build_baseline`, the comparison pipeline `run_comparison`, interpretation, CLI |
| `drift_report.py` | Builds a standalone HTML report (embedded plots, interpretations, full tables) |
| `make_synthetic_data.py` | Demo data with planted shifts and a LightGBM model |
| `tests/` | Unit and end-to-end tests (`pytest -q`) |
| `AGENT_INSTRUCTIONS.md` | Specification an agent can use to build or extend the toolkit |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

**Step 1: baseline (training population only).** Run once; set `--subset-size` to the expected size of
the new population, because PSI, KS D and classifier AUC shrink as sample size grows.

```bash
python population_drift.py baseline \
  --train train.parquet --features-file features.txt --categorical c01,c02 \
  --label-col target --model model.txt \
  --subset-size 5000 --n-rounds 30 \
  --out runs/baseline --report runs/baseline_report.html
```

**Step 2: comparison (training vs new population).**

```bash
python population_drift.py compare \
  --train train.parquet --new new.parquet --features-file features.txt --categorical c01,c02 \
  --label-col target --model model.txt \
  --baseline runs/baseline --out runs/comparison --report runs/report.html
```

Without `--baseline`, a size-matched baseline is built automatically in `runs/comparison/baseline`.
The report can be regenerated at any time:

```bash
python drift_report.py --comparison runs/comparison --out runs/report.html
python drift_report.py --baseline runs/baseline --out runs/baseline_report.html
```

**Demo** (about 2 minutes):

```bash
python make_synthetic_data.py --out demo_data
COMMON="--train demo_data/train.csv --features-file demo_data/features.txt --categorical c01,c02,c03,c04 --label-col target --model demo_data/model.txt"
python population_drift.py baseline $COMMON --subset-size 2500 --n-rounds 20 --out runs/baseline
python population_drift.py compare $COMMON --new demo_data/new.csv --baseline runs/baseline --out runs/comparison --report runs/report.html
```

### Inputs

- `--train`, `--new`: CSV, Parquet or Feather. Use `--missing-sentinels -999` if missing values are coded.
- `--features` / `--features-file`: the model's input features. If omitted, all shared columns except
  `--label-col` and `--exclude` are used. Exclude IDs and timestamps.
- `--categorical`: categorical features. Non-numeric columns and numeric columns with ≤ 10 distinct values
  are treated as categorical automatically.
- `--model` (optional): LightGBM Booster file (`.txt`) or pickled/joblib `LGBMClassifier`/`LGBMRegressor`.
  Enables score drift, SHAP drift and label-free performance estimation. Binary classification and
  regression are supported (`--problem-type`).
- `--label-col` (optional): label in the training data. Enables calibration, importance-weighted
  performance and BBSE. **Pass a labelled holdout of the training population** (rows the model was not fitted
  on); on training rows these checks are optimistic.

### Outputs

```
runs/baseline/
  baseline_statistics.csv   long table: round, task, table, feature, metric, value
  baseline_summary.csv      mean, sd, p50/p90/p95/p99 per statistic
  baseline_metadata.json
runs/comparison/
  decision.json             verdict, traffic-light signals, warnings
  feature_assessment.csv    per-feature severity with reasons
  baseline_comparison.csv   every statistic vs its baseline (threshold, empirical p, exceeds)
  tables/<task>__<table>.csv  full results of every task
  plot_data.json, metadata.json
```

The comparison also prints a console summary (verdict, signals, flagged features).

## What is tested

| Area | Function | Methods |
|---|---|---|
| Continuous features | `compare_continuous_distributions` | KS, Anderson–Darling, Cramér–von Mises (BH-adjusted); SMD with CI, variance ratio, Wasserstein/SD, PSI with bootstrap CI |
| Equivalence | `run_equivalence_tests` | TOST on means with margin 0.1 SD (evidence *for* similarity) |
| Categorical features | `compare_categorical_distributions` | Chi-square (sparse levels merged), Cramér's V, Jensen–Shannon, PSI, unseen-level share |
| Missingness | `compare_missingness`, `compare_missingness_patterns` | Two-proportion z-test, Cohen's h; joint pattern chi-square / JS |
| Joint distribution | `classifier_two_sample_test`, `mmd_test`, `energy_distance_test` | LightGBM domain classifier (OOF AUC, bootstrap CI, permutation p, drivers); MMD and energy distance with permutation p |
| Dependence | `compare_correlation_structure` | Spearman Frobenius-norm permutation test; per-pair Fisher z |
| Support | `check_range_coverage`, `detect_joint_outliers` | Out-of-range share + binomial test; Isolation Forest calibrated on held-out training rows |
| Model behaviour | `compare_prediction_distributions`, `compare_shap_distributions` | Score KS/PSI, positive-rate z-test, borderline share; TreeSHAP KS, importance rank ρ |
| Performance (no labels) | `assess_calibration`, `estimate_importance_weighted_performance`, `estimate_performance_cbpe`, `estimate_label_shift_bbse` | Calibration (ECE, slope); density-ratio weighting with Kish ESS; CBPE; BBSE prior estimate |

## How the decision is made

A feature is **red** when its difference is significant after multiple-testing correction, larger than the
baseline 95th percentile, and large (PSI ≥ 0.25, KS D ≥ 0.2, |SMD| ≥ 0.5 or Cramér's V ≥ 0.3, or a missing-rate
change with |h| ≥ 0.5). **Amber** requires the same evidence with a moderate effect. Global signals (classifier AUC,
kernel tests, out-of-support share, score PSI, ESS, estimated performance, prior shift) follow the thresholds in
`Thresholds`, and each is checked against the baseline as well.

Verdict: **red** if any signal is red; **green** if there is no red, at most two amber, and none involving
model-important features (top 80% of mean |SHAP|); **amber** otherwise. All thresholds are in
`DriftConfig`/`Thresholds` and are recorded in the outputs.

## Python API

```python
import population_drift as pdrift

config = pdrift.DriftConfig(problem_type="binary", decision_threshold=0.5)
result = pdrift.population_drift_analyzer(train, new, features, categorical, model, "target", config)
result.table("continuous")                     # any task table

baseline = pdrift.build_baseline(train, features, categorical, model, "target", n_rounds=30,
                                 subset_size=len(new), config=config, output_dir="runs/baseline")
decision = pdrift.run_comparison(train, new, features, categorical, model, "target", config,
                                 baseline_dir="runs/baseline", output_dir="runs/comparison")
```

## Limitations

- **Concept shift** (a change in P(y|x)) cannot be detected without labels. Label a small random sample of
  the new population to confirm the conclusion.
- Importance weighting assumes covariate shift only; CBPE assumes a calibrated model; BBSE assumes P(x|y) is
  unchanged. When they disagree, treat the performance estimate as inconclusive.
- Baseline empirical p-values have a resolution of 1/(rounds + 1); use ≥ 30 rounds for decisions.
- Rule-of-thumb thresholds are conventions; adjust them to your domain's tolerance (`--performance-tolerance`,
  `Thresholds`).
- False alarms: on 10 synthetic pairs drawn from the same distribution (2,500 vs 1,000 rows,
  10-round baselines) the verdict was never red, but 3 of 10 were amber because a single chance signal crossed its
  threshold. Treat an amber verdict caused by one weak signal as "investigate", not as evidence of shift.
- Heavy tasks subsample (`max_kernel_rows`, `max_classifier_rows`, `max_shap_rows`, `max_iforest_rows`).
