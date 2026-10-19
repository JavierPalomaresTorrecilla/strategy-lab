"""Offline tests for the Coinbase raw-snapshot revision comparator."""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from strategy_lab.data.providers.coinbase_raw import RAW_SCHEMA_VERSION, save_raw_json_snapshot
from strategy_lab.data.providers.coinbase_revision import (
    ArtifactPairMismatchError,
    compare_raw_artifacts,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "compare_coinbase_raw.py"
REFERENCE_DAY = date(2026, 10, 20)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparator must remain entirely offline."""

    def forbidden_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("coinbase revision tests must not open sockets")

    monkeypatch.setattr(socket, "socket", forbidden_socket)


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_coinbase_raw_script_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candle(day: date, **changes: str) -> dict[str, str]:
    start = str(int(datetime.combine(day, time.min, tzinfo=UTC).timestamp()))
    candle = {
        "start": start,
        "low": "0.50",
        "high": "0.70",
        "open": "0.60",
        "close": "0.65",
        "volume": "100.0",
    }
    candle.update(changes)
    return candle


def _artifact(
    candles: list[dict[str, str]],
    *,
    product: str = "ADA-EUR",
    started_at: str = "2026-10-20T09:02:25Z",
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "provider": "coinbase",
        "product": product,
        "interval": "1d",
        "retrieval": {
            "python_version": "3.13.0",
            "sdk_version": "1.7.0",
            "started_at_utc": started_at,
            "completed_at_utc": started_at,
        },
        "pages": [
            {
                "request": {
                    "product_id": product,
                    "start": "0",
                    "end": "1",
                    "granularity": "ONE_DAY",
                },
                "rate_limit_headers": None,
                "candles": candles,
            }
        ],
    }


def _write_pair(
    tmp_path: Path,
    baseline_candles: list[dict[str, str]],
    audit_candles: list[dict[str, str]] | None = None,
    *,
    audit_product: str = "ADA-EUR",
) -> tuple[Path, Path]:
    baseline = save_raw_json_snapshot(_artifact(baseline_candles), tmp_path)
    audit = save_raw_json_snapshot(
        _artifact(
            audit_candles if audit_candles is not None else baseline_candles,
            product=audit_product,
            started_at="2026-10-27T09:02:25Z",
        ),
        tmp_path,
    )
    return baseline.json_path, audit.json_path


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, baseline: Path, audit: Path
) -> tuple[int, list[Path]]:
    module = _load_script_module()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    result = module.main(
        [
            "--baseline",
            str(baseline.relative_to(tmp_path)),
            "--audit",
            str(audit.relative_to(tmp_path)),
        ]
    )
    reports = sorted((tmp_path / "diagnostics" / "coinbase_revision").glob("*.json"))
    return result, reports


def _report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, baseline: Path, audit: Path
) -> dict[str, Any]:
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 0
    assert len(reports) == 1
    return json.loads(reports[0].read_text(encoding="utf-8"))


def _days(*ages: int) -> list[date]:
    return [REFERENCE_DAY - timedelta(days=age) for age in ages]


def test_identical_snapshots_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline, audit = _write_pair(tmp_path, [_candle(day) for day in _days(2, 1)])
    report = _report(monkeypatch, tmp_path, baseline, audit)

    assert report["verdict"] == "PASS"
    assert report["difference_records"] == []
    assert report["raw_representation_stable"] is True
    assert report["counts"]["total"]["differing_days"] == 0


def test_value_difference_records_exact_raw_strings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = _days(1)[0]
    baseline, audit = _write_pair(tmp_path, [_candle(day)], [_candle(day, close="0.66")])
    report = _report(monkeypatch, tmp_path, baseline, audit)

    assert report["difference_records"] == [
        {
            "day": day.isoformat(),
            "age_days": 1,
            "bucket": "1",
            "kind": "value",
            "field": "close",
            "old": "0.65",
            "new": "0.66",
        }
    ]


def test_format_only_old_candle_is_not_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_day, recent_day = _days(40, 1)
    baseline, audit = _write_pair(
        tmp_path,
        [_candle(old_day, close="0.650"), _candle(recent_day)],
        [_candle(old_day), _candle(recent_day)],
    )
    report = _report(monkeypatch, tmp_path, baseline, audit)

    assert report["verdict"] == "PASS_WITH_FORMAT_DRIFT"
    assert report["raw_representation_stable"] is False
    assert report["difference_records"][0]["kind"] == "format_only"
    assert report["difference_records"][0]["bucket"] == ">30"


def test_format_drift_does_not_override_one_day_embargo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_day, recent_day = _days(40, 1)
    baseline, audit = _write_pair(
        tmp_path,
        [_candle(old_day, close="0.650"), _candle(recent_day)],
        [_candle(old_day), _candle(recent_day, close="0.66")],
    )
    assert _report(monkeypatch, tmp_path, baseline, audit)["verdict"] == "PASS_WITH_1D_EMBARGO"


def test_parser_rejected_start_fails_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, audit = _write_pair(
        tmp_path,
        [_candle(_days(1)[0], start="1760832000.0")],
        [_candle(_days(1)[0])],
    )
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 1
    assert reports == []


def test_missing_days_are_full_candle_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_day, baseline_only_day, audit_only_day, recent_day = _days(4, 3, 2, 1)
    baseline_candles = [_candle(old_day), _candle(baseline_only_day), _candle(recent_day)]
    audit_candles = [_candle(old_day), _candle(audit_only_day), _candle(recent_day)]
    baseline, audit = _write_pair(tmp_path, baseline_candles, audit_candles)
    report = _report(monkeypatch, tmp_path, baseline, audit)

    records = {record["kind"]: record for record in report["difference_records"]}
    assert records["baseline_only"] == {
        "day": baseline_only_day.isoformat(),
        "age_days": 3,
        "bucket": "2-7",
        "kind": "baseline_only",
        "field": None,
        "old": _candle(baseline_only_day),
        "new": None,
    }
    assert records["audit_only"] == {
        "day": audit_only_day.isoformat(),
        "age_days": 2,
        "bucket": "2-7",
        "kind": "audit_only",
        "field": None,
        "old": None,
        "new": _candle(audit_only_day),
    }


def test_audit_days_after_baseline_are_only_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recent_day = _days(1)[0]
    baseline, audit = _write_pair(
        tmp_path, [_candle(recent_day)], [_candle(recent_day), _candle(REFERENCE_DAY)]
    )
    report = _report(monkeypatch, tmp_path, baseline, audit)
    assert report["audit_days_after_baseline"] == 1
    assert report["difference_records"] == []
    assert report["verdict"] == "PASS"


def test_input_candle_order_does_not_change_comparison_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    older_day, recent_day = _days(2, 1)
    first_baseline, first_audit = _write_pair(
        tmp_path,
        [_candle(older_day), _candle(recent_day)],
        [_candle(older_day), _candle(recent_day)],
    )
    first = _report(monkeypatch, tmp_path, first_baseline, first_audit)

    other_root = tmp_path / "other"
    other_root.mkdir()
    second_baseline, second_audit = _write_pair(
        other_root,
        [_candle(recent_day), _candle(older_day)],
        [_candle(recent_day), _candle(older_day)],
    )
    second = _report(monkeypatch, other_root, second_baseline, second_audit)
    assert second["difference_records"] == first["difference_records"]
    assert second["counts"] == first["counts"]
    assert second["verdict"] == first["verdict"]


def test_age_bucket_boundaries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ages = (1, 2, 7, 8, 30, 31)
    baseline_candles = [_candle(day) for day in _days(*ages)]
    audit_candles = [_candle(day, close="0.66") for day in _days(*ages)]
    baseline, audit = _write_pair(tmp_path, baseline_candles, audit_candles)
    report = _report(monkeypatch, tmp_path, baseline, audit)
    assert {record["age_days"]: record["bucket"] for record in report["difference_records"]} == {
        1: "1",
        2: "2-7",
        7: "2-7",
        8: "8-30",
        30: "8-30",
        31: ">30",
    }


@pytest.mark.parametrize(
    ("ages", "expected"),
    [
        ((), "PASS"),
        ((40,), "STOP"),
        ((1,), "PASS_WITH_1D_EMBARGO"),
        ((5,), "PASS_WITH_7D_EMBARGO"),
        ((1, 5), "PASS_WITH_7D_EMBARGO"),
        ((1, 40), "STOP"),
    ],
)
def test_semantic_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ages: tuple[int, ...], expected: str
) -> None:
    all_ages = tuple(sorted(set((*ages, 1))))
    baseline_candles = [_candle(day) for day in _days(*all_ages)]
    changed = set(ages)
    audit_candles = [
        _candle(day, **({"close": "0.66"} if age in changed else {}))
        for age, day in zip(all_ages, _days(*all_ages), strict=True)
    ]
    baseline, audit = _write_pair(tmp_path, baseline_candles, audit_candles)
    assert _report(monkeypatch, tmp_path, baseline, audit)["verdict"] == expected


def test_three_fields_on_one_day_are_one_differing_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = _days(1)[0]
    baseline, audit = _write_pair(
        tmp_path,
        [_candle(day)],
        [_candle(day, low="0.49", close="0.66", volume="101.0")],
    )
    report = _report(monkeypatch, tmp_path, baseline, audit)
    assert [record["field"] for record in report["difference_records"]] == [
        "low",
        "close",
        "volume",
    ]
    assert report["counts"]["total"]["differing_days"] == 1
    assert report["counts"]["total"]["difference_records_by_kind"]["value"] == 3


def test_tampered_raw_file_fails_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, audit = _write_pair(tmp_path, [_candle(_days(1)[0])])
    baseline.write_bytes(b"{}")
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 1
    assert reports == []


def test_product_mismatch_is_usage_failure_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, audit = _write_pair(tmp_path, [_candle(_days(1)[0])], audit_product="DOT-EUR")
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 2
    assert reports == []


def test_interval_mismatch_has_the_same_pair_mismatch_type() -> None:
    baseline = _artifact([_candle(_days(1)[0])])
    audit = _artifact([_candle(_days(1)[0])], started_at="2026-10-27T09:02:25Z")
    audit["interval"] = "other"
    with pytest.raises(ArtifactPairMismatchError):
        compare_raw_artifacts(
            baseline,
            audit,
            baseline_path="baseline.json",
            baseline_sha256="a",
            audit_path="audit.json",
            audit_sha256="b",
        )


def test_duplicate_normalized_start_fails_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = _days(1)[0]
    baseline, audit = _write_pair(tmp_path, [_candle(day), _candle(day)], [_candle(day)])
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 1
    assert reports == []


def test_baseline_last_bucket_validation_fails_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = _days(2)[0]
    baseline, audit = _write_pair(tmp_path, [_candle(day)])
    code, reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert code == 1
    assert reports == []


@pytest.mark.parametrize("which", ("baseline", "audit"))
def test_absolute_and_traversal_paths_are_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, which: str
) -> None:
    baseline, audit = _write_pair(tmp_path, [_candle(_days(1)[0])])
    module = _load_script_module()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    args = {
        "--baseline": str(baseline.relative_to(tmp_path)),
        "--audit": str(audit.relative_to(tmp_path)),
    }
    args[f"--{which}"] = str(baseline) if which == "baseline" else "../outside.json"
    assert module.main([item for pair in args.items() for item in pair]) == 2
    assert not (tmp_path / "diagnostics").exists()


def test_symlink_outside_data_root_is_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "outside-link.json"
    link.symlink_to(outside)
    _, audit = _write_pair(tmp_path, [_candle(_days(1)[0])])
    module = _load_script_module()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    assert (
        module.main(
            ["--baseline", "outside-link.json", "--audit", str(audit.relative_to(tmp_path))]
        )
        == 2
    )
    assert not (tmp_path / "diagnostics").exists()


def test_repository_data_root_is_path_safety_failure_without_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    data_root = REPOSITORY_ROOT / "data"
    reports_directory = data_root / "diagnostics" / "coinbase_revision"
    before = set(reports_directory.glob("*.json")) if reports_directory.exists() else set()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(data_root))

    assert module.main(["--baseline", "raw/baseline.json", "--audit", "raw/audit.json"]) == 2
    after = set(reports_directory.glob("*.json")) if reports_directory.exists() else set()
    assert after == before


def test_same_pair_produces_byte_identical_report_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, audit = _write_pair(tmp_path, [_candle(_days(1)[0])])
    first_code, first_reports = _run(monkeypatch, tmp_path, baseline, audit)
    second_code, second_reports = _run(monkeypatch, tmp_path, baseline, audit)
    assert first_code == second_code == 0
    assert len(first_reports) == 1
    assert len(second_reports) == 2
    assert first_reports[0].read_bytes() == second_reports[1].read_bytes()


def test_revision_module_does_not_import_client_or_coinbase_sdk() -> None:
    source_path = (
        REPOSITORY_ROOT / "src" / "strategy_lab" / "data" / "providers" / "coinbase_revision.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            for alias in node.names:
                imported.add(alias.name)
                if module:
                    imported.add(f"{module}.{alias.name}")

    assert "strategy_lab.data.providers.coinbase_client" not in imported
    assert not any(
        name in {"coinbase", "coinbase_client"} or name.startswith("coinbase.") for name in imported
    )
