# Toolkit HTML Documentation Design

## Goal

Create updated standalone HTML documents that accurately describe the current
population drift toolkit while preserving the existing documents unchanged.

## Sources of Truth

Content must agree with:

- `README.md` for supported workflows and commands.
- `docs/reference.md` for methods, assumptions, thresholds, and outputs.
- `src/population_drift/` for implemented behavior and defaults.
- `tests/` for validated decision and input-handling behavior.

If prose conflicts with code, the code and tests control. Unsupported methods
or recommendations must not be presented as toolkit capabilities.

## Outputs

- `docs/proposal_v2.html`: a short approval document for technical
  decision-makers. It will state the objective, required assumptions, analysis
  protocol, verdict rule, deliverables, and label-free limitation.
- `reports/population_shift_assessment_v2.html`: the detailed technical
  assessment. It will cover input contracts, single and paired reference
  assignment, baseline construction, implemented tests, performance
  estimators and assumptions, thresholds, verdict aggregation, CLI and Python
  usage, output files, reproducibility, failure handling, and limitations.

Both files will be self-contained HTML with no external assets. The existing
`docs/proposal.html` and `reports/population_shift_assessment.html` will remain
unchanged.

## Content Rules

- Use the toolkit's `severe`, `moderate`, and `mild` terminology.
- Distinguish measurements from assumption-dependent estimates.
- State that label-free analysis cannot identify concept shift or measure
  target performance.
- Describe each test's actual reference pair in single and paired workflows.
- Include only implemented tasks, defaults, commands, APIs, and output paths.
- Keep citations for non-obvious statistical methods and conventional
  thresholds.
- Keep the proposal concise; place operational and statistical detail in the
  assessment.

## Verification

- Parse both files as HTML and confirm internal navigation targets resolve.
- Check commands, API names, defaults, thresholds, pair assignments, verdict
  logic, and output paths against the four sources of truth.
- Run the existing test suite with the repository's `.venv`.
- Confirm with Git that the two original HTML files are unchanged.
