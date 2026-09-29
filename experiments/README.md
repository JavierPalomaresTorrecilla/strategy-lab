# experiments/

Every research experiment (a hypothesis tested against data with a specific
configuration) should be recorded in `registry.csv` for provenance and audit.

## Why

Recording experiments prevents silent parameter mining: if only "winning" runs
are remembered, the apparent success rate of a research process is meaningless.
Recording every run, including failures, is what makes results falsifiable.

## What to record

At minimum, per experiment: date, hypothesis, strategy/config identifier,
data range (train/validation/test), and outcome/notes. See `registry.csv` header.

`git_commit` should be the commit the experiment was actually run against, so
a result can be reproduced exactly later. Since only the human operator
creates commits, this column is filled in once a commit exists for the run
being recorded — leave it blank for work in progress rather than guessing or
fabricating a value.

## Milestone 3 schema (`record_status`)

Starting with Milestone 3, every row also carries a `record_status` of
either `finalized` or `dev`:

- `finalized` requires `git_commit`, `working_tree_clean=true`,
  `processed_snapshot_sha256`, `source_raw_sha256`, `config_sha256`,
  `params_json`, `cost_scenario`, `window_ids`, `metrics_report_path`, and
  `dependency_versions_json` to all be populated together — this is the
  only status that may be presented as a fully reproducible record.
  `strategy_lab.validation.registry` enforces this as an all-or-nothing
  contract; it is never partially true. `dependency_versions_json` must
  additionally parse as a JSON object containing at least `python`,
  `pandas`, `pyarrow`, and `vectorbt` keys.
- `dev` covers everything else: work in progress, a dirty working tree, or
  (after migration) any pre-Milestone-3 legacy row, which predates this
  contract and can never retroactively prove Git/SHA lineage.

`working_tree_clean` means **the repository working tree was clean
immediately before this experiment's output artifacts were written** —
not "is clean right now." `scripts/run_temporal_validation.py` captures
`git_commit`, `working_tree_clean`, and `config_sha256` from a
**read-only** Git inspection (current commit + `git status --porcelain`)
immediately after resolving/validating the run's inputs and strictly
before writing the report JSON or appending the registry row, so the
run's own output writes can never self-pollute this measurement. The
script never stages, commits, or otherwise mutates Git state itself. A
working tree that was dirty at that moment always produces a `dev`
record, never a falsely-labeled `finalized` one.

Each row's `metrics_report_path` points at a separate JSON artifact under
`reports/generated/` (the repository's existing, gitignored
generated-research-artifact location — see `reports/README.md`) holding
the full per-window/per-cell breakdown; the CSV row itself stays a thin
index. Because that file is gitignored and regenerable, `dependency_versions_json`
duplicates the exact dependency versions directly into the tracked CSV row,
so a `finalized` record's reproducibility never depends on the generated
report file still existing on disk.

## TEST-set governance

TEST (2023 onward) is methodologically sealed. `strategy_family_id`
identifies the hypothesis/logic under test (e.g. `ema_crossover`) —
**not** a specific parameter setting; changing `(fast, slow)` does not
create a new family. Once TEST has been evaluated for a family
(`test_evaluated=true`) and any subsequent change is made to that
family's logic or parameters in response, TEST is no longer a valid,
untouched holdout for that family. A genuinely new family requires
materially distinct signal logic, not a renamed parameter variant.

No Milestone 3 script evaluates TEST performance. TEST evaluation remains
a distinct, future, explicitly human-invoked action outside this
milestone's scope — never something `pytest`, CI, or a walk-forward/
robustness script does automatically.
