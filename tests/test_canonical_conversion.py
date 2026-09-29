"""Tests for raw-provider-shape -> canonical-OHLCV conversion."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.data.canonical import CanonicalConversionError, to_canonical_ohlcv
from strategy_lab.data.ohlcv import OHLCVValidationError


def _yfinance_shaped_frame(rows: list[dict]) -> pd.DataFrame:
    index = pd.DatetimeIndex([r["date"] for r in rows], name="Date")
    return pd.DataFrame(
        {
            "Open": [r["open"] for r in rows],
            "High": [r["high"] for r in rows],
            "Low": [r["low"] for r in rows],
            "Close": [r["close"] for r in rows],
            "Adj Close": [r["close"] for r in rows],
            "Volume": [r["volume"] for r in rows],
            "Dividends": [0.0 for _ in rows],
            "Stock Splits": [0.0 for _ in rows],
        },
        index=index,
    )


def _valid_rows() -> list[dict]:
    return [
        {
            "date": pd.Timestamp("2024-01-02"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 1_100_000,
        },
    ]


def test_valid_yfinance_shape_converts_to_canonical() -> None:
    raw = _yfinance_shaped_frame(_valid_rows())
    canonical = to_canonical_ohlcv(raw)
    assert list(canonical.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(canonical) == 2
    assert canonical["open"].iloc[0] == 100.0
    assert canonical["timestamp"].iloc[0] == pd.Timestamp("2024-01-02")


def test_extra_provider_columns_are_dropped_not_kept() -> None:
    raw = _yfinance_shaped_frame(_valid_rows())
    canonical = to_canonical_ohlcv(raw)
    assert "Adj Close" not in canonical.columns
    assert "Dividends" not in canonical.columns
    assert "Stock Splits" not in canonical.columns


def test_already_naive_index_is_not_further_converted() -> None:
    # ignore_tz=True means the raw index from the provider is already
    # tz-naive; to_canonical_ohlcv must not call tz_localize/tz_convert on
    # it (doing so on an already-naive index would raise).
    raw = _yfinance_shaped_frame(_valid_rows())
    assert raw.index.tz is None
    canonical = to_canonical_ohlcv(raw)
    assert canonical["timestamp"].dt.tz is None


def test_missing_required_provider_column_rejected() -> None:
    raw = _yfinance_shaped_frame(_valid_rows()).drop(columns=["Volume"])
    with pytest.raises(CanonicalConversionError):
        to_canonical_ohlcv(raw)


def test_malformed_prices_fail_rather_than_repaired() -> None:
    rows = _valid_rows()
    rows[0]["high"] = 5.0  # high below close/open: impossible bar
    raw = _yfinance_shaped_frame(rows)
    with pytest.raises(OHLCVValidationError, match="high is below"):
        to_canonical_ohlcv(raw)


def test_negative_price_fails_rather_than_repaired() -> None:
    rows = _valid_rows()
    rows[0]["close"] = -1.0
    raw = _yfinance_shaped_frame(rows)
    with pytest.raises(OHLCVValidationError):
        to_canonical_ohlcv(raw)


def test_row_level_multiindex_rejected_not_silently_reshaped() -> None:
    raw = _yfinance_shaped_frame(_valid_rows())
    raw.index = pd.MultiIndex.from_arrays(
        [raw.index, ["SPY"] * len(raw)], names=["Date", "Symbol"]
    )
    with pytest.raises(CanonicalConversionError, match="DatetimeIndex"):
        to_canonical_ohlcv(raw)


def test_plain_rangeindex_rejected() -> None:
    raw = _yfinance_shaped_frame(_valid_rows()).reset_index(drop=True)
    with pytest.raises(CanonicalConversionError, match="DatetimeIndex"):
        to_canonical_ohlcv(raw)
