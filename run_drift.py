"""Run a population comparison from paths edited in this file.

The working directory is set to this file's directory before anything else runs,
so relative paths stay valid when this toolkit is a subdirectory of a larger project.
"""

from __future__ import annotations

import json
import logging
import os
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
MODEL_PATH = "demo_data/model.txt"
MISSING_SENTINELS = ""
OUTPUT_DIR = "runs/comparison"
REPORT_PATH = "runs/report.html"
N_ROUNDS = 10
THRESHOLD = 0.5
PERFORMANCE_TOLERANCE = 0.02
PROBLEM_TYPE = "binary"


def set_working_directory(root: Path = ROOT) -> Path:
    """Make the toolkit directory the process working directory.

    Args:
        root: Directory that contains the ``population_drift`` package.

    Returns:
        The directory now in use.

    Raises:
        FileNotFoundError: If ``root`` does not contain the package.
    """
    if not (root / "population_drift").is_dir():
        raise FileNotFoundError(f"population_drift package not found under {root}.")
    os.chdir(root)
    return root.resolve()


def build_argv(root: Path) -> list[str]:
    """Build the compare arguments from the constants in this file.

    Args:
        root: Toolkit root. Relative paths are resolved from here.

    Returns:
        Argument vector for ``population_drift.analysis.main``.

    Raises:
        ValueError: If a required setting is missing or invalid.
        FileNotFoundError: If a set path is not an existing file.
    """
    _validate_settings()
    argv = [
        "compare",
        "--train", str(_require_file("TRAIN_PATH", TRAIN_PATH, root)),
        "--new", str(_require_file("NEW_PATH", NEW_PATH, root)),
        "--out", str(_resolve_setting("OUTPUT_DIR", OUTPUT_DIR, root)),
        "--report", str(_resolve_setting("REPORT_PATH", REPORT_PATH, root)),
        "--n-rounds", str(N_ROUNDS),
        "--threshold", str(THRESHOLD),
        "--performance-tolerance", str(PERFORMANCE_TOLERANCE),
        "--problem-type", PROBLEM_TYPE,
    ]
    _append_optional(argv, root)
    return argv


def main() -> int:
    """Resolve the paths in this file and run the comparison.

    Returns:
        Process exit code. ``2`` means a path or setting in this file is invalid.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        argv = build_argv(ROOT)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Working directory is %s", Path.cwd())
    code = run_compare(argv)
    if code == 0:
        _log_verdict(Path(argv[argv.index("--out") + 1]))
    return code


def _validate_settings() -> None:
    """Check numeric and choice settings before any file is read.

    Raises:
        ValueError: If a setting is the wrong type or out of range.
    """
    if isinstance(N_ROUNDS, bool) or not isinstance(N_ROUNDS, int) or N_ROUNDS < 2:
        raise ValueError("N_ROUNDS must be an integer >= 2.")
    if PROBLEM_TYPE not in {"binary", "regression"}:
        raise ValueError("PROBLEM_TYPE must be 'binary' or 'regression'.")
    if isinstance(THRESHOLD, bool) or not isinstance(THRESHOLD, (int, float)):
        raise ValueError("THRESHOLD must be a number.")
    if isinstance(PERFORMANCE_TOLERANCE, bool) or not isinstance(PERFORMANCE_TOLERANCE, (int, float)):
        raise ValueError("PERFORMANCE_TOLERANCE must be a number.")
    if PERFORMANCE_TOLERANCE < 0:
        raise ValueError("PERFORMANCE_TOLERANCE must be >= 0.")


def _append_optional(argv: list[str], root: Path) -> None:
    """Add optional flags whose constants are non-empty.

    Args:
        argv: Argument vector being built.
        root: Toolkit root used to resolve file paths.

    Raises:
        FileNotFoundError: If a non-blank optional path is not a file.
    """
    test = _optional_file("TEST_PATH", TEST_PATH, root)
    features = _optional_file("FEATURES_FILE", FEATURES_FILE, root)
    model = _optional_file("MODEL_PATH", MODEL_PATH, root)
    if test is not None:
        argv.extend(["--test", str(test)])
    if features is not None:
        argv.extend(["--features-file", str(features)])
    if model is not None:
        argv.extend(["--model", str(model)])
    _append_text(argv, "--categorical", CATEGORICAL)
    _append_text(argv, "--label-col", LABEL_COL)
    _append_text(argv, "--missing-sentinels", MISSING_SENTINELS)


def _append_text(argv: list[str], flag: str, value: str) -> None:
    """Append ``flag`` when ``value`` has text.

    Args:
        argv: Argument vector being built.
        flag: Command-line flag, including the leading dashes.
        value: Constant text. Blank text is skipped.
    """
    text = value.strip()
    if text:
        argv.extend([flag, text])


def _require_file(label: str, raw: str, root: Path) -> Path:
    """Resolve a required data path and check that the file exists.

    Args:
        label: Constant name, used in the error.
        raw: Path text from this file.
        root: Toolkit root.

    Returns:
        Absolute path of an existing file.

    Raises:
        ValueError: If ``raw`` is empty.
        FileNotFoundError: If the path is not a file.
    """
    if not raw.strip():
        raise ValueError(f"Set {label} at the top of run_drift.py.")
    path = _resolve(raw, root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _optional_file(label: str, raw: str, root: Path) -> Path | None:
    """Resolve a path that may be left blank.

    Args:
        label: Constant name, used in the error.
        raw: Path text from this file.
        root: Toolkit root.

    Returns:
        Absolute path, or ``None`` when ``raw`` is blank.

    Raises:
        FileNotFoundError: If a non-blank path is not a file.
    """
    if not raw.strip():
        return None
    return _require_file(label, raw, root)


def _resolve_setting(label: str, raw: str, root: Path) -> Path:
    """Resolve an output path. The file or directory does not need to exist yet.

    Args:
        label: Constant name, used in the error.
        raw: Path text from this file.
        root: Toolkit root.

    Returns:
        Absolute path.

    Raises:
        ValueError: If ``raw`` is empty.
    """
    if not str(raw).strip():
        raise ValueError(f"Set {label} at the top of run_drift.py.")
    return _resolve(str(raw), root)


def _resolve(raw: str, root: Path) -> Path:
    """Resolve ``raw`` against ``root`` when it is not absolute.

    Args:
        raw: Path text.
        root: Toolkit root.

    Returns:
        Absolute path.
    """
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _log_verdict(output_dir: Path) -> None:
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


if __name__ == "__main__":
    set_working_directory()
    raise SystemExit(main())
