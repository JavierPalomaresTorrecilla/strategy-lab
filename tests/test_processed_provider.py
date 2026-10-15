"""Regression coverage for legacy processed-snapshot metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow
import pytest

import strategy_lab.data.processed as processed
from strategy_lab.data.processed import processed_snapshot_dir, save_processed_snapshot
from strategy_lab.data.snapshot import RawSnapshotRef, compute_sha256


def _canonical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    )


def _raw_ref(data_root: Path) -> RawSnapshotRef:
    raw_path = (
        data_root
        / "raw"
        / "yfinance"
        / "SPY"
        / "1d"
        / "20240110T000000000000Z-raw.parquet"
    )
    return RawSnapshotRef(
        parquet_path=raw_path,
        metadata_path=raw_path.with_suffix(".meta.json"),
        sha256="0123456789abcdef" * 4,
    )


def _coinbase_kwargs(**overrides) -> dict:
    kwargs = {
        "volume_unit": None,
        "volume_unit_status": "reported_base_volume",
        "volume_unit_inference": "provider_documented",
        "volume_unit_evidence": "Coinbase candle documentation",
        "missing_days_utc": ["2024-01-04", "2024-01-05"],
        "missing_day_count": 2,
    }
    kwargs.update(overrides)
    return kwargs


def _save_coinbase(data_root: Path, **kwargs):
    return save_processed_snapshot(
        _canonical_df(),
        _raw_ref(data_root),
        data_root,
        symbol="SPY",
        interval="1d",
        provider="coinbase",
        **_coinbase_kwargs(**kwargs),
    )


@dataclass(frozen=True)
class _JsonRawRef:
    json_path: Path
    sha256: str

    @property
    def artifact_path(self) -> Path:
        return self.json_path


def test_provider_none_matches_pre_task4_golden(tmp_path, monkeypatch) -> None:
    """The no-provider call preserves the pre-provider metadata bytes."""
    golden_path = Path(__file__).parent / "fixtures" / "processed_spy_golden_metadata.json"
    golden_bytes = golden_path.read_bytes()
    golden_metadata = json.loads(golden_bytes)

    golden_pandas_version = golden_metadata["pandas_version"]
    golden_pyarrow_version = golden_metadata["pyarrow_version"]
    if (
        pd.__version__ != golden_pandas_version
        or pyarrow.__version__ != golden_pyarrow_version
    ):
        pytest.skip(
            "golden captured with pandas "
            f"{golden_pandas_version} and pyarrow {golden_pyarrow_version}; "
            f"currently running pandas {pd.__version__} and pyarrow "
            f"{pyarrow.__version__}"
        )

    fixed_now = datetime(2026, 4, 5, 6, 7, 8, 901234, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)

    raw_ref = _raw_ref(tmp_path)

    processed_ref = save_processed_snapshot(
        _canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d"
    )

    assert processed_ref.metadata_path.read_bytes() == golden_bytes
    assert compute_sha256(processed_ref.parquet_path) == golden_metadata["sha256"]


def test_coinbase_snapshot_is_provider_namespaced(tmp_path) -> None:
    processed_ref = _save_coinbase(tmp_path)

    assert processed_ref.parquet_path.parent == (
        tmp_path / "processed" / "coinbase" / "SPY" / "1d"
    )


def test_coinbase_metadata_has_exactly_the_expected_keys_and_json_null_volume_unit(
    tmp_path,
) -> None:
    processed_ref = _save_coinbase(tmp_path)
    metadata_bytes = processed_ref.metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    legacy_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "source_raw_file",
        "source_raw_sha256",
        "pandas_version",
        "pyarrow_version",
        "timestamp_semantics",
        "symbol",
        "interval",
        "columns",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "sha256",
        "file",
    }
    provider_keys = {
        "provider",
        "timezone",
        "volume_unit",
        "volume_unit_status",
        "volume_unit_inference",
        "volume_unit_evidence",
        "missing_days_utc",
        "missing_day_count",
    }

    assert set(metadata) == legacy_keys | provider_keys
    assert len(metadata) == 24
    assert b'"volume_unit": null' in metadata_bytes
    assert metadata["provider"] == "coinbase"
    assert metadata["timezone"] == "UTC"
    assert metadata["volume_unit"] is None
    assert isinstance(metadata["volume_unit_status"], str)
    assert isinstance(metadata["volume_unit_inference"], str)
    assert isinstance(metadata["volume_unit_evidence"], str)
    assert isinstance(metadata["missing_days_utc"], list)
    assert all(isinstance(day, str) for day in metadata["missing_days_utc"])
    assert type(metadata["missing_day_count"]) is int


def test_unknown_provider_is_rejected_by_save_processed_snapshot(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported provider 'binance'"):
        save_processed_snapshot(
            _canonical_df(),
            _raw_ref(tmp_path),
            tmp_path,
            symbol="SPY",
            interval="1d",
            provider="binance",
        )


def test_unknown_provider_is_rejected_by_processed_snapshot_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported provider 'binance'"):
        processed_snapshot_dir(tmp_path, "SPY", "1d", provider="binance")


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("volume_unit", "shares"),
        ("volume_unit_status", "reported_base_volume"),
        ("volume_unit_inference", "provider_documented"),
        ("volume_unit_evidence", "documentation"),
        ("missing_days_utc", ["2024-01-04"]),
        ("missing_day_count", 1),
    ],
)
def test_provider_none_rejects_each_coinbase_only_argument(
    tmp_path, argument, value
) -> None:
    with pytest.raises(ValueError, match=argument):
        save_processed_snapshot(
            _canonical_df(),
            _raw_ref(tmp_path),
            tmp_path,
            symbol="SPY",
            interval="1d",
            **{argument: value},
        )


@pytest.mark.parametrize("argument", [
    "volume_unit_status",
    "volume_unit_inference",
    "volume_unit_evidence",
    "missing_days_utc",
    "missing_day_count",
])
def test_coinbase_requires_each_required_argument(tmp_path, argument) -> None:
    with pytest.raises(ValueError, match=argument):
        _save_coinbase(tmp_path, **{argument: None})


def test_json_raw_artifact_reference_records_its_artifact_path_and_checksum(tmp_path) -> None:
    json_ref = _JsonRawRef(
        json_path=tmp_path / "raw" / "coinbase" / "SPY" / "snapshot.json",
        sha256="abcdef0123456789" * 4,
    )
    processed_ref = save_processed_snapshot(
        _canonical_df(),
        json_ref,
        tmp_path,
        symbol="SPY",
        interval="1d",
        provider="coinbase",
        **_coinbase_kwargs(),
    )
    metadata = json.loads(processed_ref.metadata_path.read_text())

    assert metadata["source_raw_file"] == "raw/coinbase/SPY/snapshot.json"
    assert metadata["source_raw_sha256"] == json_ref.sha256


def test_distinct_coinbase_snapshots_do_not_collide(tmp_path) -> None:
    first_ref = _save_coinbase(tmp_path)
    canonical = _canonical_df()
    canonical.loc[0, "open"] = 99.5
    second_ref = save_processed_snapshot(
        canonical,
        _raw_ref(tmp_path),
        tmp_path,
        symbol="SPY",
        interval="1d",
        provider="coinbase",
        **_coinbase_kwargs(),
    )

    assert first_ref.parquet_path != second_ref.parquet_path
    assert first_ref.parquet_path.exists()
    assert second_ref.parquet_path.exists()


def test_identical_coinbase_snapshots_refuse_to_overwrite(tmp_path, monkeypatch) -> None:
    fixed_now = datetime(2026, 4, 6, 7, 8, 9, 123456, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(processed, "datetime", _FixedDatetime)

    _save_coinbase(tmp_path)
    with pytest.raises(FileExistsError):
        _save_coinbase(tmp_path)


def test_coinbase_rejects_boolean_missing_day_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing_day_count"):
        _save_coinbase(tmp_path, missing_day_count=True)


def test_coinbase_rejects_negative_missing_day_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing_day_count"):
        _save_coinbase(tmp_path, missing_day_count=-1)


def test_coinbase_rejects_bad_missing_day_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid date format"):
        _save_coinbase(tmp_path, missing_days_utc=["2024-2-5"], missing_day_count=1)


def test_coinbase_rejects_non_calendar_missing_day(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid date format"):
        _save_coinbase(tmp_path, missing_days_utc=["2024-02-30"], missing_day_count=1)


def test_coinbase_rejects_missing_days_out_of_order(tmp_path) -> None:
    with pytest.raises(ValueError, match="strictly ascending order"):
        _save_coinbase(
            tmp_path,
            missing_days_utc=["2024-01-05", "2024-01-04"],
            missing_day_count=2,
        )


def test_coinbase_rejects_duplicate_missing_days(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate date found"):
        _save_coinbase(
            tmp_path,
            missing_days_utc=["2024-01-04", "2024-01-04"],
            missing_day_count=2,
        )


def test_coinbase_rejects_mismatched_missing_day_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="1 != 2"):
        _save_coinbase(tmp_path, missing_days_utc=["2024-01-04"], missing_day_count=2)


@pytest.mark.parametrize(
    "argument",
    ["volume_unit_status", "volume_unit_inference", "volume_unit_evidence"],
)
def test_coinbase_rejects_empty_required_volume_text(tmp_path, argument) -> None:
    with pytest.raises(ValueError, match=argument):
        _save_coinbase(tmp_path, **{argument: ""})


def test_coinbase_rejects_empty_volume_unit(tmp_path) -> None:
    with pytest.raises(ValueError, match="volume_unit"):
        _save_coinbase(tmp_path, volume_unit="")
