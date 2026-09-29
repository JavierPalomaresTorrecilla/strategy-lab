"""Tests for the M3 experiment registry schema, finalized/dev contract, and
CSV migration."""

from __future__ import annotations

import csv

import pytest

from strategy_lab.validation.registry import (
    NEW_HEADER,
    OLD_HEADER,
    REQUIRED_DEPENDENCY_KEYS,
    RegistryMigrationError,
    RegistryRecord,
    RegistryRecordError,
    append_record,
    dependency_versions_json,
    migrate_registry,
)

_VALID_DEPENDENCY_VERSIONS_JSON = dependency_versions_json(
    {"python": "3.11.0", "pandas": "2.2.0", "pyarrow": "17.0.0", "vectorbt": "1.1.1"}
)


def _finalized_kwargs(**overrides) -> dict:
    base = dict(
        experiment_id="exp1",
        date="2026-01-01",
        git_commit="abc123",
        working_tree_clean=True,
        record_status="finalized",
        hypothesis="h",
        strategy_family_id="ema_crossover",
        strategy_id="ema_crossover_10_30",
        data_start="2011-01-01",
        data_end="2023-01-01",
        split="train+validation",
        processed_snapshot_sha256="deadbeef",
        source_raw_sha256="cafef00d",
        config_sha256="feedface",
        window_ids="2011..2022",
        params_json='{"fast": 10, "slow": 30}',
        cost_scenario="baseline",
        outcome="see report",
        metrics_report_path="reports/generated/exp1.json",
        dependency_versions_json=_VALID_DEPENDENCY_VERSIONS_JSON,
        test_evaluated=False,
        test_evaluated_date="",
        test_evaluation_experiment_id="",
        notes="",
    )
    base.update(overrides)
    return base


def test_finalized_record_requires_all_fields() -> None:
    RegistryRecord(**_finalized_kwargs())  # does not raise


def test_finalized_record_missing_field_raises() -> None:
    with pytest.raises(RegistryRecordError):
        RegistryRecord(**_finalized_kwargs(config_sha256=""))


def test_finalized_record_requires_working_tree_clean_true() -> None:
    with pytest.raises(RegistryRecordError):
        RegistryRecord(**_finalized_kwargs(working_tree_clean=False))
    with pytest.raises(RegistryRecordError):
        RegistryRecord(**_finalized_kwargs(working_tree_clean=None))


def test_dev_record_does_not_require_any_of_the_finalized_fields() -> None:
    record = RegistryRecord(
        **_finalized_kwargs(record_status="dev", working_tree_clean=None, config_sha256="")
    )
    assert record.record_status == "dev"


def test_invalid_record_status_rejected() -> None:
    with pytest.raises(RegistryRecordError):
        RegistryRecord(**_finalized_kwargs(record_status="bogus"))


def test_dependency_versions_json_helper_is_deterministic_and_compact() -> None:
    versions = {"vectorbt": "1.1.1", "python": "3.11.0", "pandas": "2.2.0", "pyarrow": "17.0.0"}
    encoded = dependency_versions_json(versions)
    assert encoded == (
        '{"pandas":"2.2.0","pyarrow":"17.0.0","python":"3.11.0","vectorbt":"1.1.1"}'
    )
    # Sorted keys regardless of input order -> same bytes every time.
    reordered = {"pandas": "2.2.0", "python": "3.11.0", "pyarrow": "17.0.0", "vectorbt": "1.1.1"}
    assert dependency_versions_json(reordered) == encoded


def test_finalized_record_requires_dependency_versions_json_present() -> None:
    with pytest.raises(RegistryRecordError):
        RegistryRecord(**_finalized_kwargs(dependency_versions_json=""))


def test_finalized_record_requires_dependency_versions_json_be_valid_json() -> None:
    with pytest.raises(RegistryRecordError, match="valid JSON"):
        RegistryRecord(**_finalized_kwargs(dependency_versions_json="not json"))


def test_finalized_record_requires_dependency_versions_json_be_an_object() -> None:
    with pytest.raises(RegistryRecordError, match="JSON object"):
        RegistryRecord(**_finalized_kwargs(dependency_versions_json="[1, 2, 3]"))


def test_finalized_record_requires_all_dependency_keys() -> None:
    incomplete = dependency_versions_json({"python": "3.11.0", "pandas": "2.2.0"})
    with pytest.raises(RegistryRecordError, match="missing"):
        RegistryRecord(**_finalized_kwargs(dependency_versions_json=incomplete))


def test_dev_record_allows_blank_dependency_versions_json() -> None:
    record = RegistryRecord(**_finalized_kwargs(record_status="dev", dependency_versions_json=""))
    assert record.record_status == "dev"


def test_required_dependency_keys_are_exactly_python_pandas_pyarrow_vectorbt() -> None:
    assert set(REQUIRED_DEPENDENCY_KEYS) == {"python", "pandas", "pyarrow", "vectorbt"}


def test_append_record_writes_header_once_and_round_trips(tmp_path) -> None:
    registry_path = tmp_path / "registry.csv"
    append_record(registry_path, RegistryRecord(**_finalized_kwargs(experiment_id="exp1")))
    append_record(registry_path, RegistryRecord(**_finalized_kwargs(experiment_id="exp2")))

    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == NEW_HEADER
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "exp1"
    assert rows[1]["experiment_id"] == "exp2"
    for row in rows:
        assert len(row) == len(NEW_HEADER)
        assert None not in row  # no extra unnamed fields (DictReader's restkey sentinel)


def test_append_record_params_json_with_commas_and_quotes_round_trips(tmp_path) -> None:
    registry_path = tmp_path / "registry.csv"
    tricky_json = '{"fast": 10, "slow": 30, "note": "a, \\"quoted\\" value"}'
    append_record(registry_path, RegistryRecord(**_finalized_kwargs(params_json=tricky_json)))

    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["params_json"] == tricky_json


def test_migrate_registry_preserves_values_and_marks_dev(tmp_path) -> None:
    old_path = tmp_path / "old_registry.csv"
    with old_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OLD_HEADER))
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "legacy1",
                "date": "2025-01-01",
                "git_commit": "old-commit",
                "hypothesis": "legacy hypothesis",
                "strategy_id": "ema_10_30",
                "data_start": "2010-01-01",
                "data_end": "2019-01-01",
                "split": "train",
                "params": "fast=10,slow=30",
                "outcome": "n/a",
                "notes": "pre-M3 row",
            }
        )

    new_path = tmp_path / "new_registry.csv"
    migrate_registry(old_path, new_path)

    with new_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == NEW_HEADER
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == "legacy1"
    assert row["hypothesis"] == "legacy hypothesis"
    assert row["params_json"] == "fast=10,slow=30"
    assert row["record_status"] == "dev"
    assert row["working_tree_clean"] == ""
    assert row["processed_snapshot_sha256"] == ""
    assert row["config_sha256"] == ""
    assert row["dependency_versions_json"] == ""
    for column in row:
        assert row[column] is not None


def test_migrate_registry_header_only_no_data_rows(tmp_path) -> None:
    """Matches the project's actual current registry.csv state: header
    only, zero data rows."""
    old_path = tmp_path / "old_registry.csv"
    with old_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OLD_HEADER))
        writer.writeheader()

    new_path = tmp_path / "new_registry.csv"
    migrate_registry(old_path, new_path)

    with new_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == NEW_HEADER
        rows = list(reader)
    assert rows == []


def test_migrate_registry_rejects_unexpected_header(tmp_path) -> None:
    old_path = tmp_path / "old_registry.csv"
    with old_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["unexpected", "columns"])
        writer.writeheader()

    with pytest.raises(RegistryMigrationError):
        migrate_registry(old_path, tmp_path / "new.csv")


def test_real_registry_csv_matches_expected_old_header() -> None:
    """Pins the actual repository-tracked experiments/registry.csv header
    to what `migrate_registry` expects -- catches accidental drift before
    a real migration run."""
    from pathlib import Path

    registry_path = Path(__file__).resolve().parents[1] / "experiments" / "registry.csv"
    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames)
    assert header in (OLD_HEADER, NEW_HEADER)


def test_real_registry_csv_report_paths_point_under_reports_generated() -> None:
    """The real registry's `metrics_report_path` values, if any, must point
    into the repository's existing `reports/generated/` location -- never
    a second `experiments/reports/` hierarchy."""
    from pathlib import Path

    registry_path = Path(__file__).resolve().parents[1] / "experiments" / "registry.csv"
    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames) != NEW_HEADER:
            pytest.skip("real registry.csv not yet migrated to NEW_HEADER")
        rows = list(reader)

    for row in rows:
        path = row.get("metrics_report_path", "")
        if path:
            assert path.startswith("reports/generated/"), path
            assert not path.startswith("experiments/reports/"), path
