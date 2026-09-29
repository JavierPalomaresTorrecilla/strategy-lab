"""Cross-engine (reference vs. VectorBT) parity tests on deterministic
synthetic OHLCV data.

These tests actually execute VectorBT -- there is no `pytest.importorskip`
guard and VectorBT is not mocked. If the `parity` extra is not installed,
this file fails to collect, which is the intended, visible failure mode
(Milestone 2 acceptance on this machine requires the `parity` extra to be
installed and these tests to pass; see pyproject.toml and the README).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbt  # noqa: F401  (import here makes a missing install fail collection visibly)

from conftest import bar, make_ohlcv
from strategy_lab.backtest.engine import BacktestConfig
from strategy_lab.strategies.ema_crossover import generate_signals
from strategy_lab.validation.parity import (
    EQUITY_RELATIVE_TOLERANCE,
    PRICE_RELATIVE_TOLERANCE,
    compare_engines,
)
from strategy_lab.validation.vectorbt_adapter import build_entries_exits, run_vectorbt


@pytest.fixture
def crossover_ohlcv() -> pd.DataFrame:
    return make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 11.0, 9.0, 10.0),
            bar(3, 50.0, 60.0, 40.0, 55.0),
            bar(4, 60.0, 65.0, 55.0, 62.0),
        ]
    )


def _synthetic_random_walk_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    open_ = open_ * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000),
        }
    )


def test_build_entries_exits_matches_engine_shift_convention() -> None:
    signal = pd.Series([0, 0, 1, 0, 0])
    entries, exits = build_entries_exits(signal)
    assert entries.tolist() == [False, False, True, False, False]
    assert exits.tolist() == [False, False, False, True, False]


def test_build_entries_exits_treats_bar_before_start_as_flat() -> None:
    # signal starts already long -> that must be treated as an entry event
    # at bar 0 (transition from an implicit flat bar -1), matching the
    # reference engine's `signal.shift(1).fillna(0)`.
    signal = pd.Series([1, 1, 0, 0, 1])
    entries, exits = build_entries_exits(signal)
    assert entries.tolist() == [True, False, False, False, True]
    assert exits.tolist() == [False, False, True, False, False]


def test_parity_exact_entry_exit_dates_and_trade_count(crossover_ohlcv) -> None:
    signal = pd.Series([0, 0, 1, 0, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)
    report = compare_engines(crossover_ohlcv, signal, config)

    assert report.reference_entry_dates == report.vectorbt_entry_dates
    assert report.reference_exit_dates == report.vectorbt_exit_dates
    assert report.reference_num_trades == report.vectorbt_num_trades == 1
    assert report.divergences == []


def test_parity_fill_prices_within_tolerance(crossover_ohlcv) -> None:
    signal = pd.Series([0, 0, 1, 0, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)
    report = compare_engines(crossover_ohlcv, signal, config)

    for ref_price, vbt_price in zip(
        report.reference_prices, report.vectorbt_prices, strict=True
    ):
        assert ref_price == pytest.approx(vbt_price, rel=PRICE_RELATIVE_TOLERANCE)


def test_parity_final_equity_and_total_return_within_tolerance(crossover_ohlcv) -> None:
    signal = pd.Series([0, 0, 1, 0, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)
    report = compare_engines(crossover_ohlcv, signal, config)

    assert report.reference_final_equity == pytest.approx(
        report.vectorbt_final_equity, rel=EQUITY_RELATIVE_TOLERANCE
    )
    assert report.reference_total_return == pytest.approx(
        report.vectorbt_total_return, abs=1e-9
    )


def test_parity_zero_cost_no_trade_flat_signal(crossover_ohlcv) -> None:
    signal = pd.Series([0, 0, 0, 0, 0])
    config = BacktestConfig(initial_cash=1000.0)
    report = compare_engines(crossover_ohlcv, signal, config)
    assert report.reference_num_trades == report.vectorbt_num_trades == 0
    assert report.reference_final_equity == pytest.approx(1000.0)
    assert report.vectorbt_final_equity == pytest.approx(1000.0)
    assert report.divergences == []


def test_parity_holds_across_many_round_trips_on_random_walk() -> None:
    ohlcv = _synthetic_random_walk_ohlcv(n=300, seed=42)
    signal = generate_signals(ohlcv["close"], fast_period=10, slow_period=30)["signal"]
    config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)

    report = compare_engines(ohlcv, signal, config)

    assert report.reference_num_trades > 0, "fixture should exercise at least one round trip"
    assert report.reference_entry_dates == report.vectorbt_entry_dates
    assert report.reference_exit_dates == report.vectorbt_exit_dates
    assert report.reference_num_trades == report.vectorbt_num_trades
    assert report.divergences == []


def test_parity_open_position_at_end_no_forced_close() -> None:
    """The dataset ends while still long (no exit signal ever occurs).
    Both engines must agree on the entry, the absence of a completed exit,
    the completed-trade count, and the final mark-to-market equity/return
    -- without the test forcing the position closed to simplify things.
    """
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 11.0, 9.0, 10.0),
            bar(3, 50.0, 60.0, 40.0, 55.0),
            bar(4, 60.0, 65.0, 55.0, 62.0),
        ]
    )
    # Signal goes long at bar 2's close and never returns to flat.
    signal = pd.Series([0, 0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)

    report = compare_engines(ohlcv, signal, config)

    assert report.reference_entry_dates == report.vectorbt_entry_dates == [
        ohlcv["timestamp"].iloc[3]
    ]
    assert report.reference_exit_dates == report.vectorbt_exit_dates == []
    assert report.reference_num_trades == report.vectorbt_num_trades == 0
    assert report.reference_final_equity == pytest.approx(
        report.vectorbt_final_equity, rel=EQUITY_RELATIVE_TOLERANCE
    )
    assert report.reference_total_return == pytest.approx(
        report.vectorbt_total_return, abs=1e-9
    )
    assert report.divergences == []


def test_parity_transition_on_final_bar_is_never_executed() -> None:
    """The signal changes only at the very last bar's close -- there is no
    T+1 bar for either engine to execute that transition on, so neither
    engine may produce a trade for it.
    """
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 11.0, 9.0, 10.0),
            bar(3, 10.0, 11.0, 9.0, 10.0),
            bar(4, 50.0, 60.0, 40.0, 55.0),
        ]
    )
    # Flat through bar 3; the only transition happens at bar 4's close,
    # the dataset's last bar -- unexecutable by construction.
    signal = pd.Series([0, 0, 0, 0, 1])
    config = BacktestConfig(initial_cash=1000.0)

    report = compare_engines(ohlcv, signal, config)

    assert report.reference_entry_dates == []
    assert report.vectorbt_entry_dates == []
    assert report.reference_exit_dates == []
    assert report.vectorbt_exit_dates == []
    assert report.reference_num_trades == report.vectorbt_num_trades == 0
    assert report.reference_final_equity == pytest.approx(1000.0)
    assert report.vectorbt_final_equity == pytest.approx(1000.0)
    assert report.divergences == []


def test_run_vectorbt_uses_open_price_not_close_for_execution(crossover_ohlcv) -> None:
    # The critical timing property: VectorBT must fill at bar 3's OPEN
    # (50.0), never bar 2's close (10.0) or bar 3's close (55.0) -- either
    # of those would be look-ahead, mirroring the reference engine's own
    # look-ahead regression test.
    signal = pd.Series([0, 0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_vectorbt(crossover_ohlcv, signal, config)

    assert result.entry_timestamps[0] == crossover_ohlcv["timestamp"].iloc[3]
    assert result.entry_prices[0] == pytest.approx(50.0)
    assert result.entry_prices[0] != pytest.approx(10.0)
    assert result.entry_prices[0] != pytest.approx(55.0)
