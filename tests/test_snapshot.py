"""Tests for immutable raw snapshot writing/reading."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd
import pytest

from strategy_lab.data.providers.base import RawFetchResult
from strategy_lab.data.snapshot import (
    compute_sha256,
    load_raw_snapshot,
    raw_snapshot_dir,
    raw_snapshot_stem,
    save_raw_snapshot,
)


def _fake_result(retrieved_at=None) -> RawFetchResult:
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
    return RawFetchResult(
        dataframe=df,
        provider="yfinance",
        provider_library_version="1.7.0",
        symbol="SPY",
        interval="1d",
        requested_start="2024-01-01",
        requested_end="2024-01-10",
        retrieved_at_utc=retrieved_at or datetime.now(UTC),
        options_used={"auto_adjust": False, "ignore_tz": True},
    )


def test_snapshot_path_is_deterministic_and_namespaced(tmp_path) -> None:
    result = _fake_result()
    ref = save_raw_snapshot(result, tmp_path)
    assert "raw" in ref.parquet_path.parts
    assert "yfinance" in ref.parquet_path.parts
    assert "SPY" in ref.parquet_path.parts
    assert "1d" in ref.parquet_path.parts
    assert ref.parquet_path.suffix == ".parquet"


def test_two_saves_never_collide_or_overwrite(tmp_path) -> None:
    result_a = _fake_result(retrieved_at=datetime(2026, 1, 1, tzinfo=UTC))
    result_b = _fake_result(retrieved_at=datetime(2026, 1, 2, tzinfo=UTC))
    ref_a = save_raw_snapshot(result_a, tmp_path)
    ref_b = save_raw_snapshot(result_b, tmp_path)
    assert ref_a.parquet_path != ref_b.parquet_path
    assert ref_a.parquet_path.exists()
    assert ref_b.parquet_path.exists()


def test_save_raw_snapshot_refuses_to_overwrite(tmp_path) -> None:
    result = _fake_result(retrieved_at=datetime(2026, 1, 1, tzinfo=UTC))
    save_raw_snapshot(result, tmp_path)
    with pytest.raises(FileExistsError):
        save_raw_snapshot(result, tmp_path)


def test_metadata_schema(tmp_path) -> None:
    result = _fake_result()
    ref = save_raw_snapshot(result, tmp_path)
    metadata = json.loads(ref.metadata_path.read_text())

    for field in (
        "schema_version",
        "provider",
        "provider_library_version",
        "pandas_version",
        "pyarrow_version",
        "symbol",
        "interval",
        "requested_start",
        "requested_end",
        "retrieved_at_utc",
        "download_options",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "columns",
        "sha256",
        "file",
    ):
        assert field in metadata, f"missing metadata field: {field}"

    assert metadata["row_count"] == 2
    assert metadata["symbol"] == "SPY"
    assert metadata["sha256"] == ref.sha256


def test_sha256_stable_for_one_stored_artifact(tmp_path) -> None:
    result = _fake_result()
    ref = save_raw_snapshot(result, tmp_path)
    assert compute_sha256(ref.parquet_path) == compute_sha256(ref.parquet_path)
    assert compute_sha256(ref.parquet_path) == ref.sha256


def test_altered_bytes_change_sha256(tmp_path) -> None:
    result = _fake_result()
    ref = save_raw_snapshot(result, tmp_path)
    original_hash = compute_sha256(ref.parquet_path)

    with ref.parquet_path.open("ab") as f:
        f.write(b"\x00")

    assert compute_sha256(ref.parquet_path) != original_hash


def test_parquet_roundtrip_is_semantically_equal(tmp_path) -> None:
    result = _fake_result()
    ref = save_raw_snapshot(result, tmp_path)
    loaded = load_raw_snapshot(ref)
    pd.testing.assert_frame_equal(loaded, result.dataframe, check_freq=False)


def test_publication_never_overwrites_concurrent_writer(tmp_path) -> None:
    """Simulate a genuine race: another writer has already published a raw
    snapshot at the exact identity our save is about to compute (same
    `retrieved_at_utc`), by placing distinguishable sentinel content there
    *before* `save_raw_snapshot` is called. Production publication must
    refuse to touch it -- this exercises the real exclusive-create path,
    not a placeholder exception.
    """
    retrieved_at = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    result = _fake_result(retrieved_at=retrieved_at)

    directory = raw_snapshot_dir(tmp_path, result.provider, result.symbol, result.interval)
    directory.mkdir(parents=True)
    stem = raw_snapshot_stem(retrieved_at)
    competing_parquet = directory / f"{stem}.parquet"
    competing_metadata = directory / f"{stem}.meta.json"

    parquet_sentinel = b"SENTINEL: written by a concurrent writer, must survive"
    metadata_sentinel = '{"sentinel": "written by a concurrent writer, must survive"}'
    competing_parquet.write_bytes(parquet_sentinel)
    competing_metadata.write_text(metadata_sentinel)

    with pytest.raises(FileExistsError):
        save_raw_snapshot(result, tmp_path)

    assert competing_parquet.read_bytes() == parquet_sentinel
    assert competing_metadata.read_text() == metadata_sentinel

    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_temp_file_cleaned_up_when_hash_computation_fails(tmp_path, monkeypatch) -> None:
    """A genuine failure after the temporary Parquet file is written (here,
    checksum computation raising) must still leave no `.tmp-*` file
    behind, and no partial artifact published."""
    result = _fake_result()

    def _boom(path):
        raise RuntimeError("simulated hash failure")

    monkeypatch.setattr("strategy_lab.data.snapshot.compute_sha256", _boom)

    with pytest.raises(RuntimeError, match="simulated hash failure"):
        save_raw_snapshot(result, tmp_path)

    directory = raw_snapshot_dir(tmp_path, result.provider, result.symbol, result.interval)
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []
    assert list(directory.glob("*.parquet")) == []


def test_same_retrieved_at_utc_concurrent_writers_exactly_one_succeeds(tmp_path) -> None:
    """Raw identity is intentionally based only on `retrieved_at_utc`, not
    content. Two threads in the same process racing to publish a raw
    snapshot with the exact same `retrieved_at_utc` must not both succeed:
    exactly one wins the exclusive-create race for the final Parquet path,
    the other gets a clean `FileExistsError` -- never a confusing
    `FileNotFoundError` or a torn/partial hash from a temp-file race, since
    the temp file is now an OS-generated exclusive file rather than a
    timestamp/PID-derived name shared by both threads.
    """
    retrieved_at = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    result_a = _fake_result(retrieved_at=retrieved_at)
    result_b = _fake_result(retrieved_at=retrieved_at)

    def _save(result):
        try:
            return ("ok", save_raw_snapshot(result, tmp_path))
        except FileExistsError as exc:
            return ("collision", exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_save, result_a), executor.submit(_save, result_b)]
        outcomes = [future.result() for future in as_completed(futures)]

    kinds = sorted(kind for kind, _ in outcomes)
    assert kinds == ["collision", "ok"]

    winner_ref = next(value for kind, value in outcomes if kind == "ok")
    assert winner_ref.parquet_path.exists()
    assert winner_ref.metadata_path.exists()
    assert compute_sha256(winner_ref.parquet_path) == winner_ref.sha256

    directory = raw_snapshot_dir(tmp_path, result_a.provider, result_a.symbol, result_a.interval)
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_checksum_mismatch_after_publication_raises(tmp_path, monkeypatch) -> None:
    """The post-publication integrity check (`published_sha256 !=
    pre_publish_sha256`) must actually fire and raise `RuntimeError` with
    its corruption diagnostic, and must not leave a temp file behind.
    """
    result = _fake_result()

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

    monkeypatch.setattr("strategy_lab.data.snapshot.compute_sha256", _flaky_compute_sha256)

    with pytest.raises(RuntimeError, match="does not match its computed checksum"):
        save_raw_snapshot(result, tmp_path)

    directory = raw_snapshot_dir(tmp_path, result.provider, result.symbol, result.interval)
    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []


def test_metadata_only_collision_leaves_parquet_published_as_orphan(tmp_path) -> None:
    """The final Parquet destination does not exist, but the metadata
    destination already exists (a pre-existing sidecar with unrelated
    content). Production `save_raw_snapshot` must: publish the Parquet
    successfully (its identity wasn't occupied), then fail with
    `FileExistsError` when it tries to exclusively create the metadata
    sidecar, leaving the pre-existing metadata untouched and the freshly
    published Parquet in place as a documented orphan.
    """
    retrieved_at = datetime(2026, 8, 15, 6, 0, 0, tzinfo=UTC)
    result = _fake_result(retrieved_at=retrieved_at)

    directory = raw_snapshot_dir(tmp_path, result.provider, result.symbol, result.interval)
    directory.mkdir(parents=True)
    stem = raw_snapshot_stem(retrieved_at)
    expected_parquet_path = directory / f"{stem}.parquet"
    competing_metadata = directory / f"{stem}.meta.json"

    metadata_sentinel = '{"sentinel": "pre-existing metadata, must survive"}'
    competing_metadata.write_text(metadata_sentinel)
    assert not expected_parquet_path.exists()

    with pytest.raises(FileExistsError):
        save_raw_snapshot(result, tmp_path)

    # Parquet publication succeeded (its identity was free) and is a
    # genuine, readable artifact...
    assert expected_parquet_path.exists()
    loaded = pd.read_parquet(expected_parquet_path)
    assert len(loaded) == len(result.dataframe)
    # ...but metadata publication correctly refused to touch the existing
    # sidecar, leaving it byte-for-byte unchanged.
    assert competing_metadata.read_text() == metadata_sentinel

    leftovers = list(directory.glob(".tmp-*"))
    assert leftovers == []
