# Population drift toolkit

Decide whether a LightGBM model fitted on one population can be applied to a new population that has the same features and **no labels**.

The comparison is scored against a baseline: the same statistics computed on random splits of the training population. A fixed cutoff such as PSI ≥ 0.10 is not, by itself, the decision.

Run the commands below from the repository root, with the virtual environment activated. Outputs under `data/`, `demo_data/`, `runs/`, and `reports/` are gitignored. Put your own files there, or anywhere else outside the paths you commit.

Technical detail (tests, null, thresholds, output columns) is in [docs/reference.md](docs/reference.md).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Compare your own populations

You need three inputs, and a model file if you want score, attribution, and performance checks.

| File | What it must contain |
|---|---|
| Training table | Feature columns, plus the label if you want calibration and label-free performance. Use a **holdout**: rows the model was not fitted on. On rows used for fitting, calibration and the reference AUC are optimistic. |
| New table | The same feature columns. Labels are not read. Extra columns are ignored once you pass a feature list. |
| Feature list | One model input per line. Leave out identifiers, timestamps, and anything built from the label or from the decision. |
| Model (optional) | LightGBM Booster (`.txt`) or a joblib/pickle `LGBMClassifier` / `LGBMRegressor`. |

CSV, Parquet, and Feather are accepted. If missing values are stored as codes, pass them with `--missing-sentinels` (for example `-999`). Name categorical columns with `--categorical`. A numeric column with at most 10 distinct values is treated as categorical anyway, because the Kolmogorov–Smirnov statistic is not valid under heavy ties.

`--subset-size` must be the new table’s row count. PSI, KS D, and the domain-classifier AUC shrink as the sample grows, so a baseline built at the wrong size is not a reference. The training table must be at least `subset-size + 30` rows. Use 30 baseline rounds for a decision (the empirical p-value cannot be smaller than `1/(rounds + 1)`).

```bash
python -m population_drift baseline \
  --train data/train.parquet \
  --features-file data/features.txt \
  --categorical country,channel \
  --label-col target \
  --model data/model.txt \
  --missing-sentinels -999 \
  --subset-size 5000 \
  --n-rounds 30 \
  --out runs/baseline

python -m population_drift compare \
  --train data/train.parquet \
  --new data/new.parquet \
  --features-file data/features.txt \
  --categorical country,channel \
  --label-col target \
  --model data/model.txt \
  --missing-sentinels -999 \
  --threshold 0.5 \
  --performance-tolerance 0.02 \
  --baseline runs/baseline \
  --out runs/comparison \
  --report runs/report.html
```

`--threshold` is the operating cutoff used for the predicted-positive rate. Set it to the cutoff you actually deploy, not 0.5, if those differ. `--performance-tolerance` is the absolute AUC drop (binary) or relative error increase (regression) that counts as material. `--problem-type regression` switches the performance metrics.

Omit `--baseline` and a size-matched baseline is built under `runs/comparison/baseline`. Rebuild the baseline yourself when the new population’s size changes.

Regenerate the HTML report from a finished run:

```bash
python -m population_drift.report --comparison runs/comparison --out runs/report.html
python -m population_drift.report --baseline runs/baseline --out runs/baseline_report.html
```

### What to open

The compare command prints a verdict, the signals, and the flagged features. The same content is in:

| Path | Contents |
|---|---|
| `runs/report.html` | Report. Open it in a browser. |
| `runs/comparison/decision.json` | `verdict`, `verdict_level` (`red`, `amber`, `green`), and one record per signal. |
| `runs/comparison/feature_assessment.csv` | Per-feature severity and the reason string. |
| `runs/comparison/baseline_comparison.csv` | Every recorded statistic against its baseline percentile and empirical p-value. |
| `runs/comparison/tables/` | One CSV per task. |

`red` means do not use the model as is: at least one signal is red. `amber` means use only with review. `green` means no red signal, at most two amber signals, and none of those amber signals on a model-important feature. Model-important features are the smallest set reaching 80% of mean absolute TreeSHAP on the training population.

A green label is not a measurement of accuracy on the new population. Concept shift, a change in P(y|x), cannot be seen without labels. The performance numbers in the report are estimates under the assumptions in [docs/reference.md](docs/reference.md).

## Example data

`examples/make_synthetic_data.py` is not part of the analysis. The package does not import it. It only writes files so the commands below can be run without a real dataset. Output goes to `demo_data/`, which is gitignored.

`train.csv` is the labelled slice the example model was fitted on. `new.csv` has 700 unlabelled rows by default, so the baseline subset size below matches that file. `evaluation.csv` is a later labelled slice and is not used by this command.

```bash
python examples/make_synthetic_data.py --out demo_data
COMMON="--train demo_data/train.csv --features-file demo_data/features.txt --categorical c01,c02,c03,c04 --label-col target --model demo_data/model.txt"
python -m population_drift baseline $COMMON --subset-size 700 --n-rounds 20 --out runs/baseline
python -m population_drift compare $COMMON --new demo_data/new.csv --baseline runs/baseline --out runs/comparison --report runs/report.html
```

## Layout

| Path | Role |
|---|---|
| `population_drift/analysis.py` | Tasks, baseline, comparison, CLI (`python -m population_drift`) |
| `population_drift/report.py` | HTML report (`python -m population_drift.report`) |
| `examples/make_synthetic_data.py` | Example files only. Not imported by the toolkit. |
| `docs/reference.md` | Methods, decision rule, configuration, output columns |

## Python API

```python
import population_drift as pdrift

config = pdrift.DriftConfig(problem_type="binary", decision_threshold=0.5)
result = pdrift.population_drift_analyzer(train, new, features, categorical, model, "target", config)
result.table("continuous")

baseline = pdrift.build_baseline(
    train, features, categorical, model, "target",
    n_rounds=30, subset_size=len(new), config=config, output_dir="runs/baseline",
)
decision = pdrift.run_comparison(
    train, new, features, categorical, model, "target", config,
    baseline_dir="runs/baseline", output_dir="runs/comparison",
)
```
