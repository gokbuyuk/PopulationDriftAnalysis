# Population drift toolkit

Decide whether a LightGBM model fitted on one population can be applied to a new population that has the same features and **no labels**.

The comparison is scored against a baseline: the same statistics computed on random splits of the training population. A fixed cutoff such as Population Stability Index (PSI) ≥ 0.10 is not, by itself, the decision.

Run the commands below from the repository root, with the virtual environment activated. The whole toolkit is the `src/population_drift/` directory. `PYTHONPATH=src` makes `python -m population_drift` import that package. Outputs under `data/`, `demo_data/`, `runs/`, and `reports/` are gitignored. Put your own files there, or anywhere else outside the paths you commit. A sample of each report type is committed under `examples/sample_reports/`.

Technical detail (tests, null, thresholds, output columns) is in [docs/reference.md](docs/reference.md).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Inside a larger project

Copy `src/population_drift/` into the other project's `src/`. Tests, docs, and example data stay in this repository. They are not part of the drop-in directory.

```
other_project/
  src/
    population_drift/     # this toolkit
    other_package/
  data/
```

From `other_project/`, with `PYTHONPATH=src`, `python -m population_drift` is the analysis command, `python -m population_drift.run_drift` is the path runner, and `python -m population_drift.batch_score` scores a new table. Relative paths in `run_drift.py` and the notebook resolve from `other_project/` (the parent of `src/`). The package must be a direct child of `src/`.

## Compare your own populations

Open `src/population_drift/run_drift.ipynb`, edit the paths in the settings cell, and run all cells. The same fields can be edited at the top of `src/population_drift/run_drift.py` and run with `PYTHONPATH=src python -m population_drift.run_drift`.

The new table has the same features and no labels. Levels are **severe**, **moderate**, and **mild**.

Without `--test`, every test uses `--train` against the new table. The toolkit does not split `--train`.

With `--test`, `--train` is the fit set and `--test` is the evaluation set:

| Tests | Pair |
|---|---|
| Per-feature distributions, missingness, domain classifier, kernel tests, correlations, support | Fit set vs new |
| Scores, positive rate, TreeSHAP, weighting, CBPE, BBSE | Evaluation set vs new |
| Calibration | Evaluation set |

Each reference needs at least `n_new + 30` rows, and the new table needs at least 30. Use 30 rounds for a decision (the empirical p-value cannot be smaller than `1/(rounds + 1)`).

CSV, Parquet, and Feather are accepted. Pass the fitted LightGBM estimator as a pickle with `--model` (`.pkl`, `.pickle`, or `.joblib`). If that file cannot be loaded, feature-distribution tests still run and the load error is recorded on `decision.json` warnings; model-based tests are skipped. Pass missing codes with `--missing-sentinels` (for example `-999`). Name categorical columns with `--categorical`. A numeric column with at most 10 distinct training values is treated as categorical anyway, because the Kolmogorov–Smirnov (KS) statistic is not valid under heavy ties. Leave identifiers, the time column, and anything built from the label or the score out of the feature list.

```bash
PYTHONPATH=src python -m population_drift compare \
  --train data/labeled.parquet \
  --new data/new.parquet \
  --features-file data/features.txt \
  --categorical country,channel \
  --label-col target \
  --model data/model.pkl \
  --missing-sentinels -999 \
  --threshold 0.5 \
  --performance-tolerance 0.02 \
  --n-rounds 30 \
  --out runs/comparison \
  --report runs/report.html
```

`--threshold` is the operating cutoff for the predicted-positive rate. Set it to the cutoff you deploy. `--performance-tolerance` is the absolute area under the curve (AUC) drop (binary) or relative error increase (regression) that counts as material. `--rate-tolerance` is the absolute band around the reference operating rate; omit it to show the rate without letting it set the expectation headline. `--problem-type regression` switches those metrics.

When the fit set and the evaluation set are separate files:

```bash
PYTHONPATH=src python -m population_drift compare \
  --train data/fit.parquet \
  --test data/evaluation.parquet \
  --new data/new.parquet \
  --features-file data/features.txt \
  --label-col target \
  --model data/model.pkl \
  --out runs/comparison \
  --report runs/report.html
```

Regenerate the HTML and markdown reports from a finished directory:

```bash
PYTHONPATH=src python -m population_drift.report --comparison runs/comparison --out runs/report.html
```

That writes the detailed report (`runs/report.html`, `runs/report.md`, plots under `runs/report_plots/`) and a short decision brief (`runs/report_short.html`, `runs/report_short.md`, plots under `runs/report_short_plots/`).

`baseline` builds one source-vs-source null. Pass it to `compare` with `--baseline` when there is no `--test`. With `--test`, each reference gets its own baseline inside the output directory.

### What to open

| Path | Contents |
|---|---|
| `runs/report_expectation.html` | Page to show stakeholders and other data scientists: whether estimated performance on the new data stays inside the pre-set tolerance, the operating rate, and the model-important features that moved. |
| `runs/report_expectation.md` | Same page as markdown. Plots are PNG files under `runs/report_expectation_plots/`. |
| `runs/report_short.html` | Drift screen: verdict, counted signals, flagged features, and the essential plots. |
| `runs/report_short.md` | Same brief as markdown. Plots are PNG files under `runs/report_short_plots/`. |
| `runs/report.html` | Full report. Each section names the pair that test used. Plots are embedded. |
| `runs/report.md` | Same full report as markdown. Plots are PNG files under `runs/report_plots/`. |
| `runs/comparison/decision.json` | `verdict_level` (`severe`, `moderate`, `mild`), signals, and `task_pairs`. |

`report_expectation.html` is the page to circulate. Its headline is whether estimated performance stays inside the tolerance. `severe` means do not use the model as is. `moderate` means use only with review. `mild` means no severe signal, at most two moderate signals, and no moderate signal on a model-important feature. Model-important features are the smallest set reaching 80% of mean absolute TreeSHAP on the population used for that comparison.

A mild label is not a measurement of accuracy on the new population. The run does not test whether a given feature vector still maps to the same label. The performance numbers are estimates under the assumptions in [docs/reference.md](docs/reference.md).

## Score a new dataset

`batch_score` is separate from the comparison. It scores every row of a table that has the model's features and writes the scores, the score distribution, and TreeSHAP feature importance. Call it whenever a new file needs scores.

Edit [examples/batch_score_config.sh](examples/batch_score_config.sh), or a copy of it, and pass that file. It is shell assignment syntax (`name="value"`). The scorer reads the assignments and does not execute the file. Leave a value empty to keep the default. A flag on the command line replaces the same field.

```bash
PYTHONPATH=src python -m population_drift.batch_score --config examples/batch_score_config.sh
```

| Field | Meaning |
|---|---|
| `model_path` | Fitted LightGBM model (`.pkl`, `.pickle`, `.joblib`, or Booster text). |
| `data_path` | Table to score (`.csv`, `.parquet`, or `.feather`). |
| `scores_path` | Per-row score file. The suffix selects the format. |
| `output_dir` | Directory for the distribution, importance, plot, and `scoring.json`. |
| `id_columns` | Columns copied onto the score file, separated by commas. Empty copies none. |
| `feature_columns` | Comma-separated feature list. Empty uses the names stored on the model. |
| `features_file` | Text file with one feature name per line. Do not set this and `feature_columns`. |
| `problem_type` | `binary` or `regression`. |
| `threshold` | Positive-class cutoff for a binary model. |
| `missing_sentinels` | Codes treated as missing, separated by commas, for example `-999`. |
| `max_shap_rows` | Maximum rows used for TreeSHAP importance. |
| `n_bins` | Histogram bins when the scores are not constant. |
| `seed` | Seed for the SHAP subsample. |
| `log_level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

The same fields can be passed as flags instead of a file:

```bash
PYTHONPATH=src python -m population_drift.batch_score \
  --model data/model.pkl \
  --data data/new.parquet \
  --scores runs/scores.parquet \
  --out runs/score_summary \
  --id-cols customer_id
```

`--scores` is the per-row file (`row_index`, any `--id-cols`, and `score`). CSV, Parquet, and Feather are accepted for `--data` and `--scores`. `--out` receives:

| File | Contents |
|---|---|
| `score_distribution.csv` | Row count, mean, standard deviation, and quantiles. Binary runs also record the positive rate at `--threshold`. |
| `score_histogram.csv` | Equal-width bins (`bin_left`, `bin_right`, `count`, `share`). A constant score is one bin. |
| `score_distribution.png` | Histogram of the scores. Binary runs mark the threshold. |
| `feature_importance.csv` | Mean absolute TreeSHAP, its share of the total, and rank. Rank 1 is the largest share. |
| `scoring.json` | Paths, row counts, and the settings used. |

TreeSHAP uses `shap.TreeExplainer` on at most `--max-shap-rows` rows (default 5000, seed `--seed`). Every input row is still scored. `--features` or `--features-file` is only required when the model file has no feature names. `--missing-sentinels` uses the same codes as `compare`.

```python
from population_drift.batch_score import request_from_config, run_batch_score

result = run_batch_score(request_from_config("examples/batch_score_config.sh"))
result.scores
```

## Example data

`examples/make_synthetic_data.py` is not part of the analysis. The package does not import it. It only writes files so the commands above can be run without a real dataset. Output goes to `demo_data/`, which is gitignored.

`train.csv` is the labelled slice the example model was fitted on. `evaluation.csv` is a later labelled slice the model did not see. `new.csv` has no label. The command below uses `train.csv` for every test. Add `--test demo_data/evaluation.csv` to try the two-file assignment. Ten rounds are enough to see the pipeline; a decision uses 30.

```bash
python examples/make_synthetic_data.py --out demo_data
```

Reports from that generator at seed 7, using `--test` and 10 baseline rounds, are committed under `examples/sample_reports/`. Performance tolerance is 0.02. No operating-rate tolerance was set. The expectation page is within that band. The detailed report and the short drift screen are severe.

| Path | Type |
|---|---|
| `examples/sample_reports/report.html` | Detailed report. Plots are embedded. |
| `examples/sample_reports/report.md` | Same report. PNGs are in `report_plots/`. |
| `examples/sample_reports/report_short.html` | Drift screen. |
| `examples/sample_reports/report_short.md` | Same brief. PNGs are in `report_short_plots/`. |
| `examples/sample_reports/report_expectation.html` | Performance-expectation page. |
| `examples/sample_reports/report_expectation.md` | Same page. PNGs are in `report_expectation_plots/`. |

The notebook defaults already point at `demo_data/`. Open `src/population_drift/run_drift.ipynb` and run all cells, or:

```bash
PYTHONPATH=src python -m population_drift compare \
  --train demo_data/train.csv \
  --new demo_data/new.csv \
  --features-file demo_data/features.txt \
  --categorical c01,c02,c03,c04 \
  --label-col target \
  --model demo_data/model.pkl \
  --n-rounds 10 \
  --out runs/comparison \
  --report runs/report.html
```

## Layout

| Path | Role |
|---|---|
| `src/population_drift/` | Whole toolkit. Copy this directory into another project's `src/`. |
| `src/population_drift/run_drift.ipynb` | Edit paths and run a comparison in Jupyter |
| `src/population_drift/run_drift.py` | Same runner (`python -m population_drift.run_drift`) |
| `src/population_drift/batch_score.py` | Score a new table (`python -m population_drift.batch_score`) |
| `examples/batch_score_config.sh` | Paths and settings for that scoring script |
| `src/population_drift/analysis.py` | Tasks, baseline, comparison, CLI (`python -m population_drift`) |
| `src/population_drift/report.py` | HTML report (`python -m population_drift.report`) |
| `examples/make_synthetic_data.py` | Example files only. Not imported by the toolkit. |
| `examples/sample_reports/` | Committed sample of the detailed, short, and expectation reports. |
| `docs/reference.md` | Methods, decision rule, configuration, output columns |

## Python API

```python
import population_drift as pdrift

model = pdrift.load_model("data/model.pkl")
config = pdrift.DriftConfig(problem_type="binary", decision_threshold=0.5)
result = pdrift.population_drift_analyzer(train, new, features, categorical, model, "target", config)
result.table("continuous")

baseline = pdrift.build_baseline(
    train, features, categorical, model, "target",
    n_rounds=30, subset_size=len(new), config=config, output_dir="runs/baseline",
)
decision = pdrift.run_comparison(
    labeled, new, features, categorical, model, "target", config,
    baseline_dir="runs/baseline", output_dir="runs/comparison",
)
paired = pdrift.run_paired_comparison(
    fit_set, evaluation, new, features, categorical, model, "target", config,
    output_dir="runs/comparison",
)
```

`run_comparison` uses one labelled table for every test. `run_paired_comparison` assigns each test to the fit set or the evaluation set. `decision["verdict_level"]` is `severe`, `moderate`, or `mild`.

## Appendix

### Tests

| Acronym | Meaning | Reading |
|---|---|---|
| KS | Kolmogorov–Smirnov | Two-sample test of one continuous distribution. Adjusted p below α: the distributions differ. Size is KS D. |

| Test | Reading |
|---|---|
| Chi-square | Test that two samples share one discrete distribution. Levels with expected count below 5 are merged. Adjusted p below α: the category shares differ. Also used on the joint missingness pattern. Size is Cramér's V. |
| Two-proportion z-test | Test that two samples share a rate. Used for missingness, the predicted positive rate, and the share of scores within 0.05 of the threshold. Adjusted p below α: the rates differ. Size is Cohen's h. |
| Permutation test | Shuffle the population label and recompute the statistic. p = (1 + shuffles at least as large) / (1 + shuffles). Used for domain-classifier AUC, MMD², energy distance, and the Spearman-matrix Frobenius norm. |
| Mann–Whitney | Test that domain-classifier scores are higher for new rows than for training rows. For fixed scores this is the same comparison as AUC. The verdict uses the permutation p, because the scores were fitted on these rows. |
| Fisher z | Test that one Spearman correlation differs between the samples. The z variance is 1.06/(n−3) in each sample (Fieller, Hartley, and Pearson 1957). Adjusted p below α: that pair changed. |
| Binomial test | One-sided test that the new share outside the training 0.5–99.5% range, or the isolation-forest outlier share, exceeds the share expected from the training rule. Adjusted p below α: more new rows fall outside support. |
| Wilks profile likelihood | Interval for one confusion-matrix metric. The profile log-likelihood ratio is compared with a chi-square distribution with one degree of freedom (Urlus et al. 2023). Used for precision, recall, F1, and the false-positive rate on the labelled set, and for each predicted-positive rate. |
| Baseline empirical p | Share of random reference splits at least as large as the new population: (1 + such rounds) / (1 + rounds). Beyond baseline means the statistic exceeds that reference's 95th percentile. |

### Magnitudes

| Acronym | Meaning | Reading |
|---|---|---|
| PSI | Population Stability Index | Change in bin shares. Below 0.10 stable, 0.10–0.25 monitor, above 0.25 large. |

### Scores

| Acronym | Meaning | Reading |
|---|---|---|
| AUC | area under the curve | Area under the ROC curve. 0.5 means no separation. Domain classifier pooled out-of-fold (OOF) AUC: ≤ 0.55 mild, ≤ 0.80 moderate, above 0.80 severe. |
| CV AUC | cross-validated area under the curve | Mean of the per-fold OOF AUCs. Target of the LeDell influence-curve confidence interval (CI). |
| OOF | out-of-fold | Score from a model that did not train on that row. |
| IC | influence curve | How much one observation pulls an estimator. The spread of those pulls is the asymptotic standard error for domain-classifier CV AUC (LeDell et al. 2015). |
| TreeSHAP | Shapley additive explanations from trees | SHAP for a tree model, from `shap.TreeExplainer`. |

### Estimates

| Acronym | Meaning | Reading |
|---|---|---|
| CBPE | confidence-based performance estimation | Expected metric if labels were drawn from the predicted probabilities. Requires calibrated scores. |
| BBSE | black-box shift estimation | Estimated new class prior. Usable only if P(x given y) is unchanged. |
| CI | confidence interval | Domain-classifier CV AUC uses an influence-curve CI. Precision, recall, F1, the false-positive rate, and each predicted-positive rate use a Wilks profile-likelihood CI (Urlus et al. 2023). Weighted performance, CBPE, and BBSE use bootstrap CIs. |
