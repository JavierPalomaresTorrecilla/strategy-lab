"""Tests for the EMA indicator."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.indicators.ema import ema


@pytest.mark.parametrize("bad_period", [0, -1, -5])
def test_non_positive_period_rejected(bad_period: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ema(pd.Series([1.0, 2.0, 3.0]), bad_period)


@pytest.mark.parametrize("bad_period", [1.5, "3", None, True, False])
def test_non_integer_period_rejected(bad_period: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        ema(pd.Series([1.0, 2.0, 3.0]), bad_period)  # type: ignore[arg-type]


def test_output_index_preserved() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    result = ema(series, 2)
    pd.testing.assert_index_equal(result.index, idx)


def test_known_values_period_2() -> None:
    # alpha = 2 / (2 + 1) = 2/3, adjust=False recursion:
    # y0 = 10
    # y1 = 2/3 * 11 + 1/3 * 10 = 10.666...7
    # y2 = 2/3 * 12 + 1/3 * y1 = 11.555...6
    close = pd.Series([10.0, 11.0, 12.0])
    result = ema(close, 2)

    assert pd.isna(result.iloc[0])  # min_periods=2: first value withheld
    assert result.iloc[1] == pytest.approx(10.0 + (2 / 3) * (11.0 - 10.0))
    assert result.iloc[2] == pytest.approx((2 / 3) * 12.0 + (1 / 3) * result.iloc[1])


def test_constant_series_is_constant() -> None:
    close = pd.Series([50.0] * 6)
    result = ema(close, 3)
    non_na = result.dropna()
    assert (non_na == 50.0).all()
