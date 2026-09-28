"""Tests for canonical OHLCV validation."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import bar, make_ohlcv
from strategy_lab.data.ohlcv import OHLCVValidationError, validate_ohlcv


def test_valid_data_accepted(valid_ohlcv: pd.DataFrame) -> None:
    result = validate_ohlcv(valid_ohlcv)
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(result) == len(valid_ohlcv)


def test_missing_columns_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.drop(columns=["volume"])
    with pytest.raises(OHLCVValidationError, match="missing required columns"):
        validate_ohlcv(broken)


def test_duplicate_timestamps_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[1, "timestamp"] = broken.loc[0, "timestamp"]
    with pytest.raises(OHLCVValidationError, match="duplicated timestamps"):
        validate_ohlcv(broken)


def test_unsorted_timestamps_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.iloc[::-1].reset_index(drop=True)
    with pytest.raises(OHLCVValidationError, match="not sorted"):
        validate_ohlcv(broken)


def test_non_positive_price_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "close"] = 0.0
    with pytest.raises(OHLCVValidationError, match="non-positive price"):
        validate_ohlcv(broken)


def test_negative_volume_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "volume"] = -1.0
    with pytest.raises(OHLCVValidationError, match="negative volume"):
        validate_ohlcv(broken)


def test_high_below_bar_rejected() -> None:
    # high (10) is below close (12): impossible bar.
    broken = make_ohlcv([bar(0, 10.0, 10.0, 9.0, 12.0)])
    with pytest.raises(OHLCVValidationError, match="high is below"):
        validate_ohlcv(broken)


def test_low_above_bar_rejected() -> None:
    # low (11) is above open (10): impossible bar.
    broken = make_ohlcv([bar(0, 10.0, 12.0, 11.0, 10.5)])
    with pytest.raises(OHLCVValidationError, match="low is above"):
        validate_ohlcv(broken)


def test_missing_values_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "close"] = float("nan")
    with pytest.raises(OHLCVValidationError, match="missing values"):
        validate_ohlcv(broken)


def test_does_not_mutate_input(valid_ohlcv: pd.DataFrame) -> None:
    original = valid_ohlcv.copy(deep=True)
    validate_ohlcv(valid_ohlcv)
    pd.testing.assert_frame_equal(valid_ohlcv, original)


def test_positive_infinity_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "close"] = float("inf")
    with pytest.raises(OHLCVValidationError, match="infinite"):
        validate_ohlcv(broken)


def test_negative_infinity_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "open"] = float("-inf")
    with pytest.raises(OHLCVValidationError, match="infinite"):
        validate_ohlcv(broken)


def test_nan_produces_single_diagnostic_not_also_infinite(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken.loc[0, "close"] = float("nan")
    with pytest.raises(OHLCVValidationError) as exc_info:
        validate_ohlcv(broken)
    assert "missing values" in str(exc_info.value)
    assert "infinite" not in str(exc_info.value)


def test_boolean_prices_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken["close"] = [True, False, True, False, True]
    with pytest.raises(OHLCVValidationError, match="boolean"):
        validate_ohlcv(broken)


def test_boolean_volume_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken["volume"] = [True, True, False, True, False]
    with pytest.raises(OHLCVValidationError, match="boolean"):
        validate_ohlcv(broken)


def test_numeric_looking_strings_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken["close"] = broken["close"].astype(str)
    with pytest.raises(OHLCVValidationError, match="non-numeric"):
        validate_ohlcv(broken)


def test_arbitrary_strings_rejected(valid_ohlcv: pd.DataFrame) -> None:
    broken = valid_ohlcv.copy()
    broken["open"] = broken["open"].astype(object)
    broken.loc[0, "open"] = "not a price"
    with pytest.raises(OHLCVValidationError, match="non-numeric"):
        validate_ohlcv(broken)


def test_empty_dataframe_rejected(valid_ohlcv: pd.DataFrame) -> None:
    empty = valid_ohlcv.iloc[0:0]
    with pytest.raises(OHLCVValidationError, match="empty"):
        validate_ohlcv(empty)
