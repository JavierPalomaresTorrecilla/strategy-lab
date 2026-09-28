"""Shared deterministic synthetic OHLCV fixtures for tests."""

from __future__ import annotations

import pandas as pd
import pytest


def make_ohlcv(rows: list[dict]) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of row dicts (test helper)."""
    return pd.DataFrame(rows)


def bar(
    day: int, open_: float, high: float, low: float, close: float, volume: float = 1000.0
) -> dict:
    """Build a single synthetic OHLCV row, `day` days after 2024-01-01."""
    return {
        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


@pytest.fixture
def valid_ohlcv() -> pd.DataFrame:
    """Five bars of well-formed, flat synthetic OHLCV data."""
    return make_ohlcv(
        [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(1, 100.0, 101.0, 99.0, 100.5),
            bar(2, 100.5, 101.5, 99.5, 100.0),
            bar(3, 100.0, 101.0, 99.0, 100.2),
            bar(4, 100.2, 101.2, 99.2, 100.1),
        ]
    )


@pytest.fixture
def crossover_ohlcv() -> pd.DataFrame:
    """5 bars: a flat close is followed by a jump at bar 3 whose OPEN is
    the only correct execution price for a signal formed at bar 2's close.
    Bar 2 close and bar 3 open deliberately differ a lot so a look-ahead
    bug (executing at bar 2 close, or at bar 3 close) is easy to detect.
    """
    return make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 11.0, 9.0, 10.0),
            bar(3, 50.0, 60.0, 40.0, 55.0),
            bar(4, 60.0, 65.0, 55.0, 62.0),
        ]
    )
