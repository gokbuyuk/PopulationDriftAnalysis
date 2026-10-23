# Toolkit HTML Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create concise and detailed HTML documentation for the current toolkit without changing the existing HTML documents.

**Architecture:** Add two self-contained HTML files. The proposal summarizes the approval decision; the assessment is the complete technical reference. Both derive claims from the README, reference, implementation, and tests rather than from the old documents.

**Tech Stack:** HTML5, CSS, Python 3.10+ standard library, pytest, Git.

## Global Constraints

- Preserve `docs/proposal.html` and `reports/population_shift_assessment.html` byte-for-byte.
- Create only `docs/proposal_v2.html` and `reports/population_shift_assessment_v2.html`.
- Keep both outputs self-contained; add no dependencies or external assets.
- Use `severe`, `moderate`, and `mild` exactly as the toolkit does.
- Treat `README.md`, `docs/reference.md`, `src/population_drift/`, and `tests/` as the sources of truth.
- Run verification with the existing `.venv`.

## File Structure

- `docs/proposal_v2.html`: short decision proposal for technical approvers.
- `reports/population_shift_assessment_v2.html`: detailed operational and statistical reference.
- Existing source and HTML files: read-only inputs.

---

### Task 1: Concise Decision Proposal

**Files:**
- Create: `docs/proposal_v2.html`
- Read: `README.md`
- Read: `docs/reference.md`
- Read: `src/population_drift/analysis.py`
- Read: `tests/test_comparison.py`

**Interfaces:**
- Consumes: current CLI workflow, pair assignment, defaults, verdict logic, and output contract.
- Produces: a standalone approval document linked to no external resources.

- [ ] **Step 1: Confirm the new output does not exist**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; assert Path('docs/proposal_v2.html').exists()"
```

Expected: failure with `AssertionError`.

- [ ] **Step 2: Create the proposal**

Create valid HTML5 with the restrained print-friendly style of
`docs/proposal.html`, using these sections and content:

1. **Purpose:** decide whether the deployed LightGBM model may score an
   unlabelled population with the same feature contract.
2. **Required inputs:** fit set, optional independent evaluation set, new
   unlabelled set, exact model feature list, categorical declarations, missing
   sentinels, deployed model, label column, operating threshold, and
   performance tolerance.
3. **Protocol:** validate schema; build target-size-matched source baselines;
   run feature, joint, support, model-output, attribution, and label-free
   performance checks; aggregate the resulting signals.
4. **Reference assignment:** without `--test`, every comparison uses the
   labelled set; with `--test`, feature/joint/support tasks use fit versus new,
   score/attribution/performance tasks use evaluation versus new, and
   calibration uses evaluation only.
5. **Decision rule:** severe if any counted signal is severe; mild if no signal
   is severe, at most two are moderate, and no moderate signal is
   model-important; moderate otherwise.
6. **Outputs:** `decision.json`, task tables, baseline summaries, metadata, and
   the generated report.
7. **Limitation:** the run measures distribution differences and estimates
   performance under explicit assumptions; it cannot identify concept shift
   or measure performance on the new population without labels.
8. **Approval defaults:** seed 42, at least 30 baseline rounds, decision
   threshold 0.5 unless production differs, and performance tolerance 0.02
   unless a domain-specific tolerance is approved.

Keep the body short enough to print in approximately two pages. Do not include
method-by-method tables, code sketches, or unimplemented recommendations.

- [ ] **Step 3: Verify structure and required claims**

Run:

```bash
.venv/bin/python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

path = Path("docs/proposal_v2.html")
text = path.read_text(encoding="utf-8")
HTMLParser().feed(text)
required = [
    "severe", "moderate", "mild", "30", "42", "0.02",
    "concept shift", "decision.json",
]
assert all(value in text for value in required)
assert len(text.splitlines()) <= 220
PY
```

Expected: exit code 0.

- [ ] **Step 4: Commit the proposal**

```bash
git add docs/proposal_v2.html
git commit -m "docs: add concise toolkit proposal"
```

### Task 2: Detailed Population Shift Assessment

**Files:**
- Create: `reports/population_shift_assessment_v2.html`
- Read: `README.md`
- Read: `docs/reference.md`
- Read: `src/population_drift/analysis.py`
- Read: `src/population_drift/report.py`
- Read: `tests/test_comparison.py`

**Interfaces:**
- Consumes: implemented analyses, configuration defaults, failure behavior,
  output schemas, CLI commands, and Python API.
- Produces: a navigable standalone technical reference with citations.

- [ ] **Step 1: Confirm the new output does not exist**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; assert Path('reports/population_shift_assessment_v2.html').exists()"
```

Expected: failure with `AssertionError`.

- [ ] **Step 2: Create the assessment**

Create valid HTML5 with responsive navigation and print styles. Include these
sections:

1. **Question, scope, and non-identifiability:** distinguish covariate shift,
   label shift, and concept shift; identify all label-free performance values
   as estimates.
2. **Input contract and validation:** supported file formats, feature parity,
   categorical handling, low-cardinality numeric handling, excluded columns,
   missing sentinels, row minimums, model/label requirements, and fail-fast
   input errors.
3. **Single and paired workflows:** reproduce the exact task-to-reference map
   from `task_pair_map`; explain separate fit and evaluation baselines.
4. **Baseline:** target-size-matched random partitions, add-one empirical
   p-value, 95th percentile, exchangeability assumption, 30-round minimum
   resolution, feature-set validation, and size warnings.
5. **Feature tests:** continuous, categorical, equivalence, and missingness
   methods; Benjamini-Hochberg scope; effect-size thresholds; model-importance
   severity cap.
6. **Joint and support tests:** out-of-fold domain classifier and permutation
   refits, MMD, energy distance, correlation structure, range coverage, and
   Isolation Forest, including row caps.
7. **Model-centric tests:** score KS/PSI, predicted-positive rate, near-cutoff
   share, TreeSHAP contribution drift, and the 80% cumulative importance rule.
8. **Performance estimates:** source calibration, importance weighting and
   Kish effective sample size, CBPE, and BBSE. State each assumption,
   confidence interval behavior, and disagreement handling.
9. **Signals and verdict:** reproduce every signal rule and the exact severe /
   moderate / mild aggregation. Mark informational signals as non-counting.
10. **Configuration and reproducibility:** seed 42; 30 baseline rounds; 20
    classifier permutations; 200 bootstraps; 100 task permutations; all row
    caps; dataset hash and code commit expectations.
11. **CLI and Python API:** include single-reference and paired CLI commands,
    report regeneration, and calls to `DriftConfig`, `build_baseline`,
    `run_comparison`, and `run_paired_comparison`.
12. **Outputs and failures:** document baseline CSVs, comparison CSV, task
    tables, metadata, `decision.json`, generated reports, warnings, and
    `metadata["failed_tasks"]`.
13. **Interpretation and limitations:** explain what each verdict authorizes,
    why mild is not measured target accuracy, baseline limitations, row-cap
    power limits, and the need for labels to test concept shift.
14. **References:** retain citations used by the implementation and threshold
    conventions: Austin, Cohen, Friedman, Hanley and McNeil, Lopez-Paz and
    Oquab, and Siddiqi.

Use tables for pair assignment, signal rules, configuration defaults, and
outputs. Remove old illustrative Python implementations and any library or
procedure the toolkit does not use.

- [ ] **Step 3: Verify navigation and required coverage**

Run:

```bash
.venv/bin/python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
import re

path = Path("reports/population_shift_assessment_v2.html")
text = path.read_text(encoding="utf-8")
HTMLParser().feed(text)
ids = set(re.findall(r'id="([^"]+)"', text))
links = set(re.findall(r'href="#([^"]+)"', text))
assert links <= ids
required = [
    "run_paired_comparison", "metadata.json", "failed_tasks",
    "1/31", "200,000", "100,000", "20,000", "5,000", "2,000",
    "severe", "moderate", "mild", "concept shift",
]
assert all(value in text for value in required)
PY
```

Expected: exit code 0.

- [ ] **Step 4: Commit the assessment**

```bash
git add reports/population_shift_assessment_v2.html
git commit -m "docs: add detailed toolkit assessment"
```

### Task 3: Cross-Source Verification

**Files:**
- Verify: `docs/proposal_v2.html`
- Verify: `reports/population_shift_assessment_v2.html`
- Preserve: `docs/proposal.html`
- Preserve: `reports/population_shift_assessment.html`

**Interfaces:**
- Consumes: both new HTML documents and all project sources of truth.
- Produces: evidence that the documents are consistent, parseable, and
  non-destructive.

- [ ] **Step 1: Audit documented names and numeric defaults**

Compare every command, public API name, output path, reference pair,
threshold, row cap, and verdict condition in the new HTML files against
`README.md`, `docs/reference.md`, `src/population_drift/analysis.py`,
`src/population_drift/report.py`, and `tests/test_comparison.py`. Correct any
discrepancy in the new files only.

- [ ] **Step 2: Run the test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify both HTML files and unchanged originals**

Run:

```bash
.venv/bin/python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

for name in (
    "docs/proposal_v2.html",
    "reports/population_shift_assessment_v2.html",
):
    text = Path(name).read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    HTMLParser().feed(text)
PY
git diff --exit-code HEAD~2 -- docs/proposal.html reports/population_shift_assessment.html
git diff --check HEAD~2..HEAD
```

Expected: all commands exit 0; the originals have no diff.

- [ ] **Step 4: Record final status**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected: no uncommitted changes; the two documentation commits and the design
commit are visible.
