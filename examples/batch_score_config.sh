# Scoring paths and settings for python -m population_drift.batch_score.
# Edit the values, then run:
#   PYTHONPATH=src python -m population_drift.batch_score --config examples/batch_score_config.sh
# Leave a value empty to keep the default. Paths are relative to the working directory.

model_path="data/model.pkl"
data_path="data/new.parquet"
scores_path="runs/scores.parquet"
output_dir="runs/score_summary"
id_columns=""
feature_columns=""
features_file=""
problem_type="binary"
threshold="0.5"
missing_sentinels=""
max_shap_rows="5000"
n_bins="20"
seed="42"
log_level="INFO"
