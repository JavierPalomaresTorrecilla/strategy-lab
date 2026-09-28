"""Tests for the EMA crossover strategy's signal generation."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.strategies.ema_crossover import generate_signals


def test_fast_must_be_less_than_slow() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="fast_period"):
        generate_signals(close, fast_period=5, slow_period=5)
    with pytest.raises(ValueError, match="fast_period"):
        generate_signals(close, fast_period=6, slow_period=5)


def test_signal_only_depends_on_information_through_current_candle() -> None:
    """Two series sharing a common prefix must produce identical signals over
    that prefix, no matter how their (unknown-at-the-time) tails differ.
    """
    common_prefix = [10.0, 10.0, 10.0, 20.0, 20.0]
    tail_a = [21.0, 22.0, 5.0]
    tail_b = [1000.0, 1.0, 1000.0]

    series_a = pd.Series(common_prefix + tail_a)
    series_b = pd.Series(common_prefix + tail_b)

    signals_a = generate_signals(series_a, fast_period=2, slow_period=3)
    signals_b = generate_signals(series_b, fast_period=2, slow_period=3)

    prefix_len = len(common_prefix)
    pd.testing.assert_frame_equal(
        signals_a.iloc[:prefix_len], signals_b.iloc[:prefix_len]
    )


def test_crossover_behavior_on_controlled_series() -> None:
    """Hand-computed EMA(2)/EMA(3) crossover on a deliberately simple series.

    close = [10, 10, 10, 20, 20, 20, 20, 20], fast=2 (alpha=2/3), slow=3 (alpha=1/2).

    fast: 10, 10, 10, 16.6667, 18.8889, 19.6296, 19.8765, 19.9588
    slow: 10, 10, 10, 15,      17.5,    18.75,   19.375,  19.6875

    fast is only defined from index 1 (min_periods=2); slow only from index 2
    (min_periods=3), so signal is forced to 0 for index 0-1. At index 2 the
    two EMAs are equal (not a cross yet). fast > slow starting at index 3.
    """
    close = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    result = generate_signals(close, fast_period=2, slow_period=3)

    expected_signal = [0, 0, 0, 1, 1, 1, 1, 1]
    assert result["signal"].tolist() == expected_signal

    assert pd.isna(result["fast_ema"].iloc[0])
    assert pd.isna(result["slow_ema"].iloc[0])
    assert pd.isna(result["slow_ema"].iloc[1])  # slow needs 3 points

    assert result["fast_ema"].iloc[3] == pytest.approx(16.66666667, abs=1e-6)
    assert result["slow_ema"].iloc[3] == pytest.approx(15.0, abs=1e-6)
    assert result["fast_ema"].iloc[7] == pytest.approx(19.9588477, abs=1e-5)
    assert result["slow_ema"].iloc[7] == pytest.approx(19.6875, abs=1e-6)


def test_index_preserved() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    result = generate_signals(close, fast_period=2, slow_period=3)
    pd.testing.assert_index_equal(result.index, idx)
