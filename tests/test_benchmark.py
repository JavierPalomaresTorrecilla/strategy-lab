"""Tests for the M3 price-only buy-and-hold benchmark."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import random_walk_ohlcv
from strategy_lab.backtest.engine import BacktestConfig, run_backtest
from strategy_lab.validation.benchmark import (
    BENCHMARK_LABEL,
    STRATEGY_DIVIDEND_DISCLAIMER,
    BenchmarkConstructionError,
    run_price_only_benchmark,
)


def _ohlcv_2011_2012() -> pd.DataFrame:
    return random_walk_ohlcv("2011-01-01", "2012-12-31", seed=11)


def test_benchmark_entry_is_first_open_at_or_after_evaluation_start() -> None:
    ohlcv = _ohlcv_2011_2012()
    result = run_price_only_benchmark(
        ohlcv,
        pd.Timestamp("2011-01-01"),
        pd.Timestamp("2013-01-01"),
        initial_cash=100_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    first_row = ohlcv.iloc[0]
    assert result.entry_timestamp == first_row["timestamp"]
    assert result.entry_open == pytest.approx(float(first_row["open"]))


def test_benchmark_final_value_marks_to_last_close_before_evaluation_end() -> None:
    ohlcv = _ohlcv_2011_2012()
    result = run_price_only_benchmark(
        ohlcv,
        pd.Timestamp("2011-01-01"),
        pd.Timestamp("2013-01-01"),
        initial_cash=100_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    last_row_before_2013 = ohlcv.loc[ohlcv["timestamp"] < pd.Timestamp("2013-01-01")].iloc[-1]
    assert result.final_timestamp == last_row_before_2013["timestamp"]
    assert result.final_close == pytest.approx(float(last_row_before_2013["close"]))
    assert result.final_value == pytest.approx(result.quantity * result.final_close)


def test_benchmark_no_terminal_sale_no_exit_cost() -> None:
    ohlcv = _ohlcv_2011_2012()
    result = run_price_only_benchmark(
        ohlcv,
        pd.Timestamp("2011-01-01"),
        pd.Timestamp("2013-01-01"),
        initial_cash=100_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    # final_value is a pure mark-to-market valuation: quantity * close,
    # with no second commission/slippage applied anywhere.
    assert result.final_value == pytest.approx(result.quantity * result.final_close)


def test_benchmark_labeling_present() -> None:
    ohlcv = _ohlcv_2011_2012()
    result = run_price_only_benchmark(ohlcv, pd.Timestamp("2011-01-01"), pd.Timestamp("2013-01-01"))
    assert result.label == BENCHMARK_LABEL == "price-only; dividends excluded"
    assert result.dividend_disclaimer == STRATEGY_DIVIDEND_DISCLAIMER
    assert "burn-in" in result.initialization_asymmetry
    assert "same raw price series" in result.permitted_claim


def test_benchmark_no_rows_at_or_after_start_raises() -> None:
    ohlcv = _ohlcv_2011_2012()
    with pytest.raises(BenchmarkConstructionError):
        run_price_only_benchmark(ohlcv, pd.Timestamp("2050-01-01"), pd.Timestamp("2051-01-01"))


def test_benchmark_no_rows_before_end_raises() -> None:
    ohlcv = _ohlcv_2011_2012()
    with pytest.raises(BenchmarkConstructionError):
        run_price_only_benchmark(ohlcv, pd.Timestamp("2011-01-01"), pd.Timestamp("2000-01-01"))


def test_benchmark_sizing_matches_reference_engine_buy_formula() -> None:
    """Regression test against the M1 buy sizing formula: constructing a
    one-bar-buy through `run_backtest` (a signal that goes long
    immediately and never exits) must produce the exact same fill price,
    quantity, and commission as the benchmark's own independent
    computation of the same entry."""
    ohlcv = _ohlcv_2011_2012()
    config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)

    # Signal known at bar 0's close -> executes at bar 1's open (M1 shift
    # convention). Never exits, so exactly one buy Trade is produced.
    signal = pd.Series([1] * len(ohlcv))
    result = run_backtest(ohlcv, signal, config)
    buy_trade = result.trades[0]
    assert buy_trade.side == "buy"

    bench = run_price_only_benchmark(
        ohlcv,
        ohlcv["timestamp"].iloc[1],
        ohlcv["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        initial_cash=100_000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    assert bench.entry_exec_price == pytest.approx(buy_trade.price)
    assert bench.quantity == pytest.approx(buy_trade.quantity)
    assert bench.entry_commission == pytest.approx(buy_trade.commission)


def test_benchmark_defensively_filters_out_test_range_even_with_a_wide_end_boundary() -> None:
    """Even if a caller mistakenly passes an ``evaluation_end`` that reaches
    into TEST (a caller bug), the benchmark's own internal defensive
    filter (using the real, default `RESEARCH_ALLOWED_SPLITS` policy) must
    still exclude TEST rows from ever being read as the final valuation."""
    from strategy_lab.data.splits import load_splits

    splits = load_splits()  # real config/research.yaml
    test_start = next(s.start for s in splits if s.name == "test")

    ohlcv = random_walk_ohlcv(
        (test_start - pd.Timedelta(days=200)).date().isoformat(),
        (test_start + pd.Timedelta(days=200)).date().isoformat(),
        seed=5,
    )
    assert (ohlcv["timestamp"] >= test_start).any()  # fixture genuinely spans into TEST

    result = run_price_only_benchmark(
        ohlcv,
        evaluation_start=ohlcv["timestamp"].iloc[0],
        evaluation_end=test_start + pd.Timedelta(days=200),  # deliberately reaches into TEST
    )

    assert result.final_timestamp < test_start
