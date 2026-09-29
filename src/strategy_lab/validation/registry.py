"""Experiment registry: schema, finalized/dev contract, CSV I/O.

Extends the existing ``experiments/registry.csv`` (see
``experiments/README.md``) rather than introducing a database. All CSV I/O
goes through the stdlib ``csv`` module (``DictReader``/``DictWriter``) so
fields containing commas/quotes (e.g. ``params_json``) round-trip
correctly -- never manual string concatenation.

Generated report JSON artifacts (``reports/generated/``) are intentionally
gitignored and regenerable from a row's recorded snapshot/config/parameter
identity -- they are not a durable provenance store. Exact dependency
versions are therefore also captured directly in the registry row itself
(``dependency_versions_json``), not only inside the regenerable report, so
a ``finalized`` record's reproducibility does not depend on that file
still existing.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

OLD_HEADER: tuple[str, ...] = (
    "experiment_id",
    "date",
    "git_commit",
    "hypothesis",
    "strategy_id",
    "data_start",
    "data_end",
    "split",
    "params",
    "outcome",
    "notes",
)

NEW_HEADER: tuple[str, ...] = (
    "experiment_id",
    "date",
    "git_commit",
    "working_tree_clean",
    "record_status",
    "hypothesis",
    "strategy_family_id",
    "strategy_id",
    "data_start",
    "data_end",
    "split",
    "processed_snapshot_sha256",
    "source_raw_sha256",
    "config_sha256",
    "window_ids",
    "params_json",
    "cost_scenario",
    "outcome",
    "metrics_report_path",
    "dependency_versions_json",
    "test_evaluated",
    "test_evaluated_date",
    "test_evaluation_experiment_id",
    "notes",
)

# A "finalized" record must have all of these populated, plus
# working_tree_clean == True and a valid dependency_versions_json (checked
# separately below).
FINALIZED_REQUIRED_FIELDS: tuple[str, ...] = (
    "git_commit",
    "processed_snapshot_sha256",
    "source_raw_sha256",
    "config_sha256",
    "params_json",
    "cost_scenario",
    "window_ids",
    "metrics_report_path",
    "dependency_versions_json",
)

# Keys `dependency_versions_json` must contain for a "finalized" record.
REQUIRED_DEPENDENCY_KEYS: tuple[str, ...] = ("python", "pandas", "pyarrow", "vectorbt")

RECORD_STATUSES = ("finalized", "dev")


def dependency_versions_json(versions: dict[str, str | None]) -> str:
    """Deterministic (sorted-key, compact) JSON serialization of a
    dependency-version mapping, for the registry's
    ``dependency_versions_json`` column."""
    return json.dumps(versions, sort_keys=True, separators=(",", ":"))


class RegistryError(ValueError):
    """Base class for registry schema/record problems."""


class RegistryRecordError(RegistryError):
    """Raised when a ``RegistryRecord`` violates the finalized/dev
    contract."""


class RegistryMigrationError(RegistryError):
    """Raised when the existing registry file's header doesn't match what
    the migration expects."""


@dataclass(frozen=True)
class RegistryRecord:
    experiment_id: str
    date: str
    git_commit: str
    working_tree_clean: bool | None
    record_status: str
    hypothesis: str
    strategy_family_id: str
    strategy_id: str
    data_start: str
    data_end: str
    split: str
    processed_snapshot_sha256: str
    source_raw_sha256: str
    config_sha256: str
    window_ids: str
    params_json: str
    cost_scenario: str
    outcome: str
    metrics_report_path: str
    dependency_versions_json: str
    test_evaluated: bool
    test_evaluated_date: str
    test_evaluation_experiment_id: str
    notes: str

    def __post_init__(self) -> None:
        if self.record_status not in RECORD_STATUSES:
            raise RegistryRecordError(
                f"record_status must be one of {RECORD_STATUSES}, got {self.record_status!r}"
            )
        if self.record_status == "finalized":
            missing = [field for field in FINALIZED_REQUIRED_FIELDS if not getattr(self, field)]
            if self.working_tree_clean is not True:
                missing.append("working_tree_clean(must be True)")
            if missing:
                raise RegistryRecordError(
                    "record_status='finalized' requires all of "
                    f"{FINALIZED_REQUIRED_FIELDS} plus working_tree_clean=True; "
                    f"missing/falsy: {missing}"
                )
            try:
                parsed_versions = json.loads(self.dependency_versions_json)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RegistryRecordError(
                    "record_status='finalized' requires dependency_versions_json to be "
                    f"valid JSON: {exc}"
                ) from exc
            if not isinstance(parsed_versions, dict):
                raise RegistryRecordError(
                    "record_status='finalized' requires dependency_versions_json to be a "
                    f"JSON object, got {type(parsed_versions).__name__}"
                )
            missing_keys = [key for key in REQUIRED_DEPENDENCY_KEYS if key not in parsed_versions]
            if missing_keys:
                raise RegistryRecordError(
                    f"record_status='finalized' requires dependency_versions_json to contain "
                    f"{REQUIRED_DEPENDENCY_KEYS}; missing: {missing_keys}"
                )

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        row["working_tree_clean"] = (
            "" if self.working_tree_clean is None else str(self.working_tree_clean).lower()
        )
        row["test_evaluated"] = str(self.test_evaluated).lower()
        return {key: ("" if value is None else value) for key, value in row.items()}


def append_record(registry_path: Path, record: RegistryRecord) -> None:
    """Append ``record`` as a new row to ``registry_path``, writing the
    header first if the file doesn't already exist."""
    file_exists = registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NEW_HEADER))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record.to_row())


def git_commit_and_clean(repo_root: Path) -> tuple[str, bool]:
    """Read-only Git inspection: the current commit hash, and whether the
    working tree was clean *at the moment this function is called*.

    Callers must call this -- and capture ``config_sha256`` -- before
    writing any experiment output (report JSON, registry row, or any other
    repository-side artifact), so the returned ``working_tree_clean`` means
    "the repository working tree was clean immediately before this
    experiment's output artifacts were written", not "is clean right now"
    (which the run's own writes would otherwise self-pollute). Never
    mutates Git state (no add/commit/etc.).
    """
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, status.strip() == ""


def migrate_registry(old_path: Path, new_path: Path) -> None:
    """Migrate a registry CSV from ``OLD_HEADER`` to ``NEW_HEADER``.

    Existing values are preserved (``params`` -> ``params_json``, mapped
    directly); every new column is written blank; every migrated row gets
    ``record_status = "dev"`` (a legacy row predates the finalized-record
    contract and cannot retroactively prove Git/SHA lineage, so it can
    never be presented as finalized/reproducible).
    """
    with old_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None and tuple(reader.fieldnames) != OLD_HEADER:
            raise RegistryMigrationError(
                f"unexpected existing registry header: {reader.fieldnames!r}; "
                f"expected {OLD_HEADER!r}"
            )
        old_rows = list(reader)

    with new_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NEW_HEADER))
        writer.writeheader()
        for old_row in old_rows:
            new_row = dict.fromkeys(NEW_HEADER, "")
            for column in OLD_HEADER:
                if column == "params":
                    new_row["params_json"] = old_row.get("params", "")
                else:
                    new_row[column] = old_row.get(column, "")
            new_row["record_status"] = "dev"
            writer.writerow(new_row)
