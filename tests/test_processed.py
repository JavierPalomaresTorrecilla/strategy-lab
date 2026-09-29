"""Tests for immutable processed (canonical) snapshot writing/reading."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd
import pytest

from strategy_lab.data.processed import load_processed_snapshot, save_processed_snapshot
from strategy_lab.data.providers.base import RawFetchResult
from strategy_lab.data.snapshot import compute_sha256, save_raw_snapshot


def _raw_ref(tmp_path):
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], name="Date"
        ),
    )
    result = RawFetchResult(
        dataframe=df,
        provider="yfinance",
        provider_library_version="1.7.0",
        symbol="SPY",
        interval="1d",
        requested_start="2024-01-01",
        requested_end="2024-01-10",
        retrieved_at_utc=datetime.now(UTC),
        options_used={"auto_adjust": False},
    )
    return save_raw_snapshot(result, tmp_path)


def _canonical_df(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "open": [100.0 + offset, 101.0 + offset],
            "high": [102.0 + offset, 103.0 + offset],
            "low": [99.0 + offset, 100.0 + offset],
            "close": [101.0 + offset, 102.0 + offset],
            "volume": [1_000_000, 1_100_000],
        }
    )


def test_processed_snapshot_records_lineage(tmp_path) -> None:
    raw_ref = _raw_ref(tmp_path)
    processed_ref = save_processed_snapshot(
        _canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d"
    )
    metadata = json.loads(processed_ref.metadata_path.read_text())
    assert metadata["source_raw_sha256"] == raw_ref.sha256
    assert metadata["source_raw_file"] == str(raw_ref.parquet_path.relative_to(tmp_path))
    assert metadata["timestamp_semantics"] == "exchange_session_date_naive"
    assert metadata["row_count"] == 2


def test_processed_metadata_has_required_fields(tmp_path) -> None:
    raw_ref = _raw_ref(tmp_path)
    processed_ref = save_processed_snapshot(
        _canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d"
    )
    metadata = json.loads(processed_ref.metadata_path.read_text())

    for field in (
        "created_at_utc",
        "sha256",
        "source_raw_file",
        "source_raw_sha256",
        "schema_version",
        "pandas_version",
        "pyarrow_version",
        "timestamp_semantics",
        "symbol",
        "interval",
        "columns",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "file",
    ):
        assert field in metadata, f"missing metadata field: {field}"

    # created_at_utc must be a genuine, parseable UTC timestamp.
    parsed = datetime.fromisoformat(metadata["created_at_utc"])
    assert parsed.tzinfo is not None


def test_processed_snapshot_refuses_to_overwrite(tmp_path, monkeypatch) -> None:
    """Two saves that resolve to the exact same identity (same creation
    instant and identical processed bytes, so the same hash) must not
    silently clobber one another."""
    raw_ref = _raw_ref(tmp_path)

    fixed_now = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)

    save_processed_snapshot(_canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d")
    with pytest.raises(FileExistsError):
        save_processed_snapshot(_canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d")


def test_processed_parquet_roundtrip_semantically_equal(tmp_path) -> None:
    raw_ref = _raw_ref(tmp_path)
    canonical = _canonical_df()
    processed_ref = save_processed_snapshot(
        canonical, raw_ref, tmp_path, symbol="SPY", interval="1d"
    )
    loaded = load_processed_snapshot(processed_ref)
    pd.testing.assert_frame_equal(loaded, canonical, check_freq=False)


def test_two_derivations_from_same_raw_snapshot_coexist(tmp_path) -> None:
    """Two independently-produced canonical derivations of the *same*
    immutable raw snapshot (e.g. before/after a canonicalization fix) must
    both persist immutably, at distinct paths, both tracing back to the
    same source raw SHA-256."""
    raw_ref = _raw_ref(tmp_path)

    first_ref = save_processed_snapshot(
        _canonical_df(offset=0.0), raw_ref, tmp_path, symbol="SPY", interval="1d"
    )
    second_ref = save_processed_snapshot(
        _canonical_df(offset=1.0), raw_ref, tmp_path, symbol="SPY", interval="1d"
    )

    assert first_ref.parquet_path != second_ref.parquet_path
    assert first_ref.metadata_path != second_ref.metadata_path
    assert first_ref.parquet_path.exists()
    assert second_ref.parquet_path.exists()

    first_metadata = json.loads(first_ref.metadata_path.read_text())
    second_metadata = json.loads(second_ref.metadata_path.read_text())

    assert first_metadata["source_raw_sha256"] == raw_ref.sha256
    assert second_metadata["source_raw_sha256"] == raw_ref.sha256

    assert first_metadata["sha256"] == first_ref.sha256
    assert second_metadata["sha256"] == second_ref.sha256
    assert first_ref.sha256 != second_ref.sha256

    assert compute_sha256(first_ref.parquet_path) == first_ref.sha256
    assert compute_sha256(second_ref.parquet_path) == second_ref.sha256


def test_no_leftover_temp_files_after_save(tmp_path) -> None:
    raw_ref = _raw_ref(tmp_path)
    save_processed_snapshot(_canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d")

    directory = tmp_path / "processed" / "SPY" / "1d"
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_publication_never_overwrites_concurrent_writer(tmp_path, monkeypatch) -> None:
    """Simulate a genuine race: another writer has already published an
    artifact at the exact identity our save is about to compute, by
    placing distinguishable sentinel content there *before*
    `save_processed_snapshot` is called. Production publication must
    refuse to touch it -- this exercises the real exclusive-create path,
    not a placeholder exception.
    """
    raw_ref = _raw_ref(tmp_path)
    canonical = _canonical_df()

    fixed_now = datetime(2026, 3, 1, 9, 30, 0, 654321, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)

    # Predict the exact identity save_processed_snapshot will compute,
    # using the same real Parquet serialization and hashing it relies on.
    probe_path = tmp_path / "probe.parquet"
    canonical.to_parquet(probe_path)
    expected_sha256 = compute_sha256(probe_path)
    stem = f"{fixed_now.strftime('%Y%m%dT%H%M%S%fZ')}-{expected_sha256[:16]}"

    directory = tmp_path / "processed" / "SPY" / "1d"
    directory.mkdir(parents=True)
    competing_parquet = directory / f"{stem}.canonical.parquet"
    competing_metadata = directory / f"{stem}.meta.json"

    parquet_sentinel = b"SENTINEL: written by a concurrent writer, must survive"
    metadata_sentinel = '{"sentinel": "written by a concurrent writer, must survive"}'
    competing_parquet.write_bytes(parquet_sentinel)
    competing_metadata.write_text(metadata_sentinel)

    with pytest.raises(FileExistsError):
        save_processed_snapshot(canonical, raw_ref, tmp_path, symbol="SPY", interval="1d")

    assert competing_parquet.read_bytes() == parquet_sentinel
    assert competing_metadata.read_text() == metadata_sentinel

    # No leftover temp file either, even on this collision path.
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_temp_file_cleaned_up_when_hash_computation_fails(tmp_path, monkeypatch) -> None:
    """A genuine failure after the temporary Parquet file is written (here,
    checksum computation raising) must still leave no `.tmp-*` file
    behind."""
    raw_ref = _raw_ref(tmp_path)

    def _boom(path):
        raise RuntimeError("simulated hash failure")

    monkeypatch.setattr("strategy_lab.data.processed.compute_sha256", _boom)

    with pytest.raises(RuntimeError, match="simulated hash failure"):
        save_processed_snapshot(_canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d")

    directory = tmp_path / "processed" / "SPY" / "1d"
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []
    # And no partial artifact was published either.
    assert list(directory.glob("*.canonical.parquet")) == []


def test_same_timestamp_concurrent_writers_with_different_content_both_succeed(
    tmp_path, monkeypatch
) -> None:
    """Two threads in the same process, forced to compute the exact same
    `created_at_utc` (via a fixed clock), but writing *different* canonical
    DataFrames so their processed SHA-256 values -- and therefore their
    final identities -- differ. Since the temporary file is now an
    OS-generated exclusive file (not derived from timestamp/PID), the two
    threads cannot collide at the temp-file layer; both publications must
    succeed independently.
    """
    raw_ref = _raw_ref(tmp_path)

    fixed_now = datetime(2026, 4, 1, 8, 0, 0, 111111, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)

    def _save(offset: float):
        return save_processed_snapshot(
            _canonical_df(offset=offset), raw_ref, tmp_path, symbol="SPY", interval="1d"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_save, 0.0), executor.submit(_save, 5.0)]
        refs = [future.result() for future in as_completed(futures)]

    assert len(refs) == 2
    ref_a, ref_b = refs

    assert ref_a.parquet_path != ref_b.parquet_path
    assert ref_a.metadata_path != ref_b.metadata_path
    assert ref_a.sha256 != ref_b.sha256

    for ref in (ref_a, ref_b):
        assert ref.parquet_path.exists()
        assert compute_sha256(ref.parquet_path) == ref.sha256
        metadata = json.loads(ref.metadata_path.read_text())
        assert metadata["sha256"] == ref.sha256
        assert metadata["created_at_utc"] == fixed_now.isoformat()

    directory = tmp_path / "processed" / "SPY" / "1d"
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_checksum_mismatch_after_publication_raises(tmp_path, monkeypatch) -> None:
    """The post-publication integrity check (`published_sha256 !=
    pre_publish_sha256`) must actually fire and raise `RuntimeError` with
    its corruption diagnostic, and must not leave a temp file behind.
    """
    raw_ref = _raw_ref(tmp_path)

    real_compute_sha256 = compute_sha256
    call_count = {"n": 0}

    def _flaky_compute_sha256(path):
        call_count["n"] += 1
        real_value = real_compute_sha256(path)
        if call_count["n"] == 1:
            # First call: hashing the temp artifact before publication --
            # return the real, correct value so publication proceeds.
            return real_value
        # Second call: post-publication verification -- return a
        # deliberately wrong value to simulate corruption during publish.
        return "0" * 64

    monkeypatch.setattr("strategy_lab.data.processed.compute_sha256", _flaky_compute_sha256)

    with pytest.raises(RuntimeError, match="does not match its computed checksum"):
        save_processed_snapshot(_canonical_df(), raw_ref, tmp_path, symbol="SPY", interval="1d")

    directory = tmp_path / "processed" / "SPY" / "1d"
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_metadata_only_collision_leaves_parquet_published_as_orphan(
    tmp_path, monkeypatch
) -> None:
    """The final Parquet destination does not exist, but the metadata
    destination already exists (a pre-existing sidecar with unrelated
    content). Production `save_processed_snapshot` must: publish the
    Parquet successfully (its identity wasn't occupied), then fail with
    `FileExistsError` when it tries to exclusively create the metadata
    sidecar, leaving the pre-existing metadata untouched and the freshly
    published Parquet in place as a documented orphan.
    """
    raw_ref = _raw_ref(tmp_path)
    canonical = _canonical_df()

    fixed_now = datetime(2026, 7, 4, 15, 45, 0, 222222, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)

    # Predict the exact identity save_processed_snapshot will compute.
    probe_path = tmp_path / "probe.parquet"
    canonical.to_parquet(probe_path)
    expected_sha256 = compute_sha256(probe_path)
    stem = f"{fixed_now.strftime('%Y%m%dT%H%M%S%fZ')}-{expected_sha256[:16]}"

    directory = tmp_path / "processed" / "SPY" / "1d"
    directory.mkdir(parents=True)
    expected_parquet_path = directory / f"{stem}.canonical.parquet"
    competing_metadata = directory / f"{stem}.meta.json"

    metadata_sentinel = '{"sentinel": "pre-existing metadata, must survive"}'
    competing_metadata.write_text(metadata_sentinel)
    assert not expected_parquet_path.exists()

    with pytest.raises(FileExistsError):
        save_processed_snapshot(canonical, raw_ref, tmp_path, symbol="SPY", interval="1d")

    # Parquet publication succeeded (its identity was free)...
    assert expected_parquet_path.exists()
    assert compute_sha256(expected_parquet_path) == expected_sha256
    # ...but metadata publication correctly refused to touch the existing
    # sidecar, leaving it byte-for-byte unchanged.
    assert competing_metadata.read_text() == metadata_sentinel

    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []
