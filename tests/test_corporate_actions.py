"""Tests for the M3 corporate-action runtime guard."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from strategy_lab.data.processed import save_processed_snapshot
from strategy_lab.data.providers.base import RawFetchResult
from strategy_lab.data.snapshot import save_raw_snapshot
from strategy_lab.validation.corporate_actions import (
    RESEARCH_RANGE_END,
    RESEARCH_RANGE_START,
    CorporateActionCheckResult,
    UnsupportedCorporateActionError,
    check_corporate_actions,
    resolve_raw_ref_from_processed,
)


def _raw_frame(dates, splits, dividends) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.5] * n,
            "Volume": [1_000_000] * n,
            "Stock Splits": splits,
            "Dividends": dividends,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _save_raw(tmp_path, dates, splits, dividends):
    df = _raw_frame(dates, splits, dividends)
    result = RawFetchResult(
        dataframe=df,
        provider="yfinance",
        provider_library_version="1.7.0",
        symbol="SPY",
        interval="1d",
        requested_start=str(dates[0]),
        requested_end=str(dates[-1]),
        retrieved_at_utc=datetime.now(UTC),
        options_used={},
    )
    return save_raw_snapshot(result, tmp_path)


def test_all_zero_splits_in_range_passes_and_records_fact(tmp_path) -> None:
    dates = pd.date_range("2015-01-02", periods=5, freq="D")
    raw_ref = _save_raw(tmp_path, dates, splits=[0.0] * 5, dividends=[0.0, 0.0, 1.5, 0.0, 0.0])
    result = check_corporate_actions(raw_ref)
    assert isinstance(result, CorporateActionCheckResult)
    assert result.stock_splits_verified_zero_in_research_range is True
    assert result.dividend_event_count_in_research_range == 1


def test_non_zero_split_inside_research_range_raises(tmp_path) -> None:
    dates = pd.date_range("2015-01-02", periods=5, freq="D")
    splits = [0.0, 0.0, 4.0, 0.0, 0.0]  # a 4:1 split
    raw_ref = _save_raw(tmp_path, dates, splits=splits, dividends=[0.0] * 5)
    with pytest.raises(UnsupportedCorporateActionError, match="Stock Splits"):
        check_corporate_actions(raw_ref)


def test_non_zero_split_outside_research_range_is_never_read(tmp_path) -> None:
    # One split before RESEARCH_RANGE_START and one at/after
    # RESEARCH_RANGE_END -- neither may influence the guard.
    dates = [
        RESEARCH_RANGE_START - pd.Timedelta(days=2),
        RESEARCH_RANGE_START,
        RESEARCH_RANGE_END - pd.Timedelta(days=1),
        RESEARCH_RANGE_END,
        RESEARCH_RANGE_END + pd.Timedelta(days=2),
    ]
    splits = [4.0, 0.0, 0.0, 4.0, 4.0]
    raw_ref = _save_raw(tmp_path, dates, splits=splits, dividends=[0.0] * 5)
    result = check_corporate_actions(raw_ref)
    assert result.stock_splits_verified_zero_in_research_range is True


def test_range_boundaries_are_half_open(tmp_path) -> None:
    dates = [RESEARCH_RANGE_START, RESEARCH_RANGE_END - pd.Timedelta(days=1), RESEARCH_RANGE_END]
    # A split exactly on RESEARCH_RANGE_END (exclusive) must not be seen.
    splits = [0.0, 0.0, 4.0]
    raw_ref = _save_raw(tmp_path, dates, splits=splits, dividends=[0.0, 0.0, 0.0])
    result = check_corporate_actions(raw_ref)
    assert result.stock_splits_verified_zero_in_research_range is True


def test_resolve_raw_ref_from_processed_uses_recorded_lineage(tmp_path) -> None:
    dates = pd.date_range("2015-01-02", periods=3, freq="D")
    raw_ref = _save_raw(tmp_path, dates, splits=[0.0] * 3, dividends=[0.0] * 3)

    canonical_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.5] * 3,
            "volume": [1_000_000] * 3,
        }
    )
    processed_ref = save_processed_snapshot(
        canonical_df, raw_ref, tmp_path, symbol="SPY", interval="1d"
    )

    resolved = resolve_raw_ref_from_processed(processed_ref, tmp_path)
    assert resolved.sha256 == raw_ref.sha256
    assert resolved.parquet_path == raw_ref.parquet_path

    metadata = json.loads(processed_ref.metadata_path.read_text())
    assert metadata["source_raw_sha256"] == raw_ref.sha256
