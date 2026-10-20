"""Run a population comparison from editable path settings.

The working directory is set to this file's directory before anything else runs,
so relative paths stay valid when this toolkit is a subdirectory of a larger project.

Edit the constants below and run ``python run_drift.py``, or edit the same
fields in ``run_drift.ipynb`` and run all cells.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from population_drift.analysis import main as run_compare

logger = logging.getLogger("population_drift.run")

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    os.chdir(ROOT)

# Edit these. A relative path is resolved from ROOT (the directory containing this file).
# Leave TEST_PATH empty to use the training table for every test.
TRAIN_PATH = "demo_data/train.csv"
TEST_PATH = "demo_data/evaluation.csv"
NEW_PATH = "demo_data/new.csv"

FEATURES_FILE = "demo_data/features.txt"
CATEGORICAL = "c01,c02,c03,c04"
LABEL_COL = "target"
MODEL_PATH = "demo_data/model.pkl"
MISSING_SENTINELS = ""
OUTPUT_DIR = "runs/comparison"
REPORT_PATH = "runs/report.html"
N_ROUNDS = 10
THRESHOLD = 0.5
PERFORMANCE_TOLERANCE = 0.02
PROBLEM_TYPE = "binary"


@dataclass
class DriftRunSettings:
    """Paths and comparison options used by ``DriftRun`` and ``run_drift.ipynb``.

    Args:
        train_path: Labelled fit table.
        test_path: Evaluation table, or blank to use ``train_path`` for every test.
        new_path: Unlabelled new population.
        features_file: One feature name per line, or blank to infer.
        categorical: Comma-separated categorical columns.
        label_col: Label column on the labelled table(s).
        model_path: Pickled LightGBM estimator (``.pkl``), or blank for feature-only checks.
        missing_sentinels: Comma-separated missing codes.
        output_dir: Comparison output directory.
        report_path: HTML report path.
        n_rounds: Baseline rounds. Must be >= 2.
        threshold: Operating cutoff for the predicted-positive rate.
        performance_tolerance: Absolute AUC drop (binary) or relative error increase.
        problem_type: ``binary`` or ``regression``.
    """

    train_path: str
    test_path: str
    new_path: str
    features_file: str
    categorical: str
    label_col: str
    model_path: str
    missing_sentinels: str
    output_dir: str
    report_path: str
    n_rounds: int
    threshold: float
    performance_tolerance: float
    problem_type: str

    def __post_init__(self) -> None:
        """Reject settings that the comparison CLI cannot run.

        Raises:
            ValueError: If a numeric or choice field is invalid.
            TypeError: If a path field is not a string.
        """
        for name in (
            "train_path", "test_path", "new_path", "features_file", "categorical",
            "label_col", "model_path", "missing_sentinels", "output_dir", "report_path",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string.")
        if isinstance(self.n_rounds, bool) or not isinstance(self.n_rounds, int) or self.n_rounds < 2:
            raise ValueError("n_rounds must be an integer >= 2.")
        if self.problem_type not in {"binary", "regression"}:
            raise ValueError("problem_type must be 'binary' or 'regression'.")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise ValueError("threshold must be a number.")
        if isinstance(self.performance_tolerance, bool) or not isinstance(self.performance_tolerance, (int, float)):
            raise ValueError("performance_tolerance must be a number.")
        if self.performance_tolerance < 0:
            raise ValueError("performance_tolerance must be >= 0.")


class DriftRun:
    """Resolve ``DriftRunSettings`` and run ``population_drift.analysis.main``."""

    def __init__(self, settings: DriftRunSettings, root: Path | None = None) -> None:
        """Store settings and the directory used to resolve relative paths.

        Args:
            settings: Paths and comparison options.
            root: Toolkit root. Defaults to the directory containing this file.

        Raises:
            TypeError: If ``settings`` is not a ``DriftRunSettings`` instance.
        """
        if not isinstance(settings, DriftRunSettings):
            raise TypeError(f"settings must be DriftRunSettings, got {type(settings).__name__}.")
        self.settings = settings
        self.root = (root or ROOT).expanduser().resolve()

    def build_argv(self) -> list[str]:
        """Build the compare arguments from ``settings``.

        Returns:
            Argument vector for ``population_drift.analysis.main``.

        Raises:
            ValueError: If a required path setting is empty.
            FileNotFoundError: If a set path is not an existing file.
        """
        argv = [
            "compare",
            "--train", str(self._require_file("train_path", self.settings.train_path)),
            "--new", str(self._require_file("new_path", self.settings.new_path)),
            "--out", str(self._resolve_setting("output_dir", self.settings.output_dir)),
            "--report", str(self._resolve_setting("report_path", self.settings.report_path)),
            "--n-rounds", str(self.settings.n_rounds),
            "--threshold", str(self.settings.threshold),
            "--performance-tolerance", str(self.settings.performance_tolerance),
            "--problem-type", self.settings.problem_type,
        ]
        self._append_optional(argv)
        return argv

    def run(self) -> int:
        """Run the comparison and log the verdict.

        Returns:
            Process exit code. ``2`` means a path or setting is invalid.
        """
        try:
            argv = self.build_argv()
        except (ValueError, FileNotFoundError) as exc:
            logger.error("%s", exc)
            return 2
        logger.info("Working directory is %s", Path.cwd())
        code = run_compare(argv)
        if code == 0:
            self._log_verdict(Path(argv[argv.index("--out") + 1]))
        return code

    def _append_optional(self, argv: list[str]) -> None:
        """Add optional flags whose settings are non-empty.

        Args:
            argv: Argument vector being built.

        Raises:
            FileNotFoundError: If a non-blank optional path is not a file.
        """
        test = self._optional_file("test_path", self.settings.test_path)
        features = self._optional_file("features_file", self.settings.features_file)
        model = self._optional_file("model_path", self.settings.model_path)
        if test is not None:
            argv.extend(["--test", str(test)])
        if features is not None:
            argv.extend(["--features-file", str(features)])
        if model is not None:
            argv.extend(["--model", str(model)])
        self._append_text(argv, "--categorical", self.settings.categorical)
        self._append_text(argv, "--label-col", self.settings.label_col)
        self._append_text(argv, "--missing-sentinels", self.settings.missing_sentinels)

    def _require_file(self, label: str, raw: str) -> Path:
        """Resolve a required data path and check that the file exists.

        Args:
            label: Setting name, used in the error.
            raw: Path text from the settings.

        Returns:
            Absolute path of an existing file.

        Raises:
            ValueError: If ``raw`` is empty.
            FileNotFoundError: If the path is not a file.
        """
        if not raw.strip():
            raise ValueError(f"Set {label} in the run settings.")
        path = self._resolve(raw)
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
        return path

    def _optional_file(self, label: str, raw: str) -> Path | None:
        """Resolve a path that may be left blank.

        Args:
            label: Setting name, used in the error.
            raw: Path text from the settings.

        Returns:
            Absolute path, or ``None`` when ``raw`` is blank.

        Raises:
            FileNotFoundError: If a non-blank path is not a file.
        """
        if not raw.strip():
            return None
        return self._require_file(label, raw)

    def _resolve_setting(self, label: str, raw: str) -> Path:
        """Resolve an output path. The file or directory does not need to exist yet.

        Args:
            label: Setting name, used in the error.
            raw: Path text from the settings.

        Returns:
            Absolute path.

        Raises:
            ValueError: If ``raw`` is empty.
        """
        if not str(raw).strip():
            raise ValueError(f"Set {label} in the run settings.")
        return self._resolve(str(raw))

    def _resolve(self, raw: str) -> Path:
        """Resolve ``raw`` against ``self.root`` when it is not absolute.

        Args:
            raw: Path text.

        Returns:
            Absolute path.
        """
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _log_verdict(self, output_dir: Path) -> None:
        """Log the verdict written by the comparison.

        Args:
            output_dir: Comparison directory that contains ``decision.json``.
        """
        decision_path = output_dir / "decision.json"
        if not decision_path.is_file():
            logger.error("Comparison finished without %s", decision_path)
            return
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        logger.info("Verdict: %s", decision.get("verdict_level"))

    @staticmethod
    def _append_text(argv: list[str], flag: str, value: str) -> None:
        """Append ``flag`` when ``value`` has text.

        Args:
            argv: Argument vector being built.
            flag: Command-line flag, including the leading dashes.
            value: Setting text. Blank text is skipped.
        """
        text = value.strip()
        if text:
            argv.extend([flag, text])


def set_working_directory(root: Path = ROOT) -> Path:
    """Make the toolkit directory the process working directory.

    Args:
        root: Directory that contains the ``population_drift`` package.

    Returns:
        The directory now in use.

    Raises:
        FileNotFoundError: If ``root`` does not contain the package.
        TypeError: If ``root`` is not a ``Path``.
    """
    if not isinstance(root, Path):
        raise TypeError("root must be a Path.")
    if not (root / "population_drift").is_dir():
        raise FileNotFoundError(f"population_drift package not found under {root}.")
    os.chdir(root)
    return root.resolve()


def settings_from_constants() -> DriftRunSettings:
    """Build settings from the editable constants at the top of this file.

    Returns:
        Settings matching ``TRAIN_PATH`` and the other module-level constants.
    """
    return DriftRunSettings(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        new_path=NEW_PATH,
        features_file=FEATURES_FILE,
        categorical=CATEGORICAL,
        label_col=LABEL_COL,
        model_path=MODEL_PATH,
        missing_sentinels=MISSING_SENTINELS,
        output_dir=OUTPUT_DIR,
        report_path=REPORT_PATH,
        n_rounds=N_ROUNDS,
        threshold=THRESHOLD,
        performance_tolerance=PERFORMANCE_TOLERANCE,
        problem_type=PROBLEM_TYPE,
    )


def main() -> int:
    """Resolve the paths in this file and run the comparison.

    Returns:
        Process exit code. ``2`` means a path or setting in this file is invalid.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return DriftRun(settings_from_constants()).run()


if __name__ == "__main__":
    set_working_directory()
    raise SystemExit(main())
