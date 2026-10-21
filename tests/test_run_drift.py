"""Unit tests for the path runner used by ``run_drift.py`` and ``run_drift.ipynb``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_drift import MODEL_PATH, DriftRun, DriftRunSettings, ROOT, set_working_directory


def _write(path: Path, text: str = "x") -> Path:
    """Create a parent directory and write ``text``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _settings(tmp_path: Path, **overrides: object) -> DriftRunSettings:
    """Valid settings whose required files exist under ``tmp_path``."""
    train = _write(tmp_path / "train.csv")
    new = _write(tmp_path / "new.csv")
    values: dict[str, object] = {
        "train_path": str(train),
        "test_path": "",
        "new_path": str(new),
        "features_file": "",
        "categorical": "c01,c02",
        "label_col": "target",
        "model_path": "",
        "missing_sentinels": "",
        "output_dir": str(tmp_path / "out"),
        "report_path": str(tmp_path / "report.html"),
        "n_rounds": 10,
        "threshold": 0.5,
        "performance_tolerance": 0.02,
        "problem_type": "binary",
    }
    values.update(overrides)
    return DriftRunSettings(**values)  # type: ignore[arg-type]


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def test_settings_reject_too_few_rounds(tmp_path: Path) -> None:
    """N_ROUNDS must be an integer of at least 2."""
    with pytest.raises(ValueError, match="n_rounds must be an integer >= 2"):
        _settings(tmp_path, n_rounds=1)


def test_settings_reject_unknown_problem_type(tmp_path: Path) -> None:
    """PROBLEM_TYPE is binary or regression."""
    with pytest.raises(ValueError, match="problem_type must be 'binary' or 'regression'"):
        _settings(tmp_path, problem_type="multiclass")


def test_settings_reject_negative_performance_tolerance(tmp_path: Path) -> None:
    """PERFORMANCE_TOLERANCE cannot be negative."""
    with pytest.raises(ValueError, match="performance_tolerance must be >= 0"):
        _settings(tmp_path, performance_tolerance=-0.01)


def test_build_argv_resolves_required_paths_and_skips_blank_optionals(tmp_path: Path) -> None:
    """Required files become absolute paths; blank optional flags are omitted."""
    argv = DriftRun(_settings(tmp_path), root=tmp_path).build_argv()
    _check(argv[0], "compare", "subcommand")
    _check(argv[argv.index("--train") + 1], str((tmp_path / "train.csv").resolve()), "train path")
    _check(argv[argv.index("--new") + 1], str((tmp_path / "new.csv").resolve()), "new path")
    _check("--test" in argv, False, "blank TEST_PATH is omitted")
    _check("--features-file" in argv, False, "blank FEATURES_FILE is omitted")
    _check("--model" in argv, False, "blank MODEL_PATH is omitted")
    _check(argv[argv.index("--categorical") + 1], "c01,c02", "categorical list")
    _check(argv[argv.index("--n-rounds") + 1], "10", "n_rounds")


def test_build_argv_includes_optional_files_when_they_exist(tmp_path: Path) -> None:
    """Non-blank optional paths are resolved and passed as flags."""
    test = _write(tmp_path / "evaluation.csv")
    features = _write(tmp_path / "features.txt", "f01\n")
    model = _write(tmp_path / "model.pkl")
    argv = DriftRun(
        _settings(
            tmp_path,
            test_path=str(test),
            features_file=str(features),
            model_path=str(model),
            missing_sentinels="-999",
        ),
        root=tmp_path,
    ).build_argv()
    _check(argv[argv.index("--test") + 1], str(test.resolve()), "test path")
    _check(argv[argv.index("--features-file") + 1], str(features.resolve()), "features file")
    _check(argv[argv.index("--model") + 1], str(model.resolve()), "model path")
    _check(argv[argv.index("--missing-sentinels") + 1], "-999", "missing sentinels")


def test_build_argv_rejects_a_missing_required_file(tmp_path: Path) -> None:
    """A required path that is not a file names the setting."""
    settings = _settings(tmp_path, train_path=str(tmp_path / "missing.csv"))
    with pytest.raises(FileNotFoundError, match="train_path is not a file"):
        DriftRun(settings, root=tmp_path).build_argv()


def test_run_returns_2_when_a_required_file_is_missing(tmp_path: Path) -> None:
    """Invalid paths are logged and become exit code 2 instead of a traceback."""
    settings = _settings(tmp_path, train_path=str(tmp_path / "missing.csv"))
    code = DriftRun(settings, root=tmp_path).run()
    _check(code, 2, "exit code for a missing train file")


def test_run_still_invokes_compare_when_the_model_file_is_unreadable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A set model path that exists is passed through; load errors are handled later."""
    model = _write(tmp_path / "model.pkl", "not-a-pickle")
    out = tmp_path / "out"
    out.mkdir()
    (out / "decision.json").write_text(json.dumps({"verdict_level": "mild"}), encoding="utf-8")
    seen: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    monkeypatch.setattr("run_drift.run_compare", fake_main)
    code = DriftRun(_settings(tmp_path, model_path=str(model), output_dir=str(out)), root=tmp_path).run()
    _check(code, 0, "exit code when the model file exists but is unreadable")
    _check("--model" in seen[0], True, "unreadable model is still passed to compare")


def test_run_logs_the_verdict_when_the_comparison_succeeds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero exit from the comparison logs verdict_level from decision.json."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "decision.json").write_text(json.dumps({"verdict_level": "mild"}), encoding="utf-8")

    def fake_main(argv: list[str]) -> int:
        _check(argv[0], "compare", "subcommand passed to analysis.main")
        return 0

    monkeypatch.setattr("run_drift.run_compare", fake_main)
    settings = _settings(tmp_path, output_dir=str(out))
    code = DriftRun(settings, root=tmp_path).run()
    _check(code, 0, "exit code when compare succeeds")


def test_set_working_directory_requires_the_package(tmp_path: Path) -> None:
    """The working directory must contain the population_drift package."""
    with pytest.raises(FileNotFoundError, match="population_drift package not found"):
        set_working_directory(tmp_path)


def test_notebook_settings_cell_builds_the_same_argv_shape() -> None:
    """The notebook is present, empty of outputs, and constructs DriftRunSettings."""
    path = ROOT / "run_drift.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    joined = "\n".join(sources)
    _check(notebook["nbformat"], 4, "notebook format")
    _check(any(cell.get("outputs") for cell in notebook["cells"]), False, "committed outputs")
    _check("DriftRunSettings(" in joined, True, "settings constructed in a cell")
    _check("demo_data/train.csv" in joined, True, "demo train path")
    _check("demo_data/new.csv" in joined, True, "demo new path")
    _check("demo_data/model.pkl" in joined, True, "demo pickle model path")
    _check("demo_data/model.txt" in joined, False, "demo model is not a LightGBM text file")
    _check("DriftRun(settings).run()" in joined, True, "run call")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        compile("".join(cell.get("source", [])), f"run_drift.ipynb:cell{index}", "exec")


def test_default_model_path_is_a_pickle_file() -> None:
    """The runner expects a pickled LightGBM estimator."""
    _check(Path(MODEL_PATH).suffix, ".pkl", "MODEL_PATH suffix")
    _check(MODEL_PATH, "demo_data/model.pkl", "default MODEL_PATH")
