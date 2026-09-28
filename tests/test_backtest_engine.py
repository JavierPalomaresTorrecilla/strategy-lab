"""Tests for the deterministic backtest engine.

The look-ahead tests here are the most important tests in this milestone:
they prove that a signal known only at a bar's close cannot execute using
information from that same bar.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import bar, make_ohlcv
from strategy_lab.backtest.engine import BacktestConfig, run_backtest
from strategy_lab.data.ohlcv import OHLCVValidationError


def test_signal_known_at_close_t_executes_at_open_t_plus_1(
    crossover_ohlcv: pd.DataFrame,
) -> None:
    # Signal becomes 1 starting at index 2 (i.e. known at bar 2's close).
    signal = pd.Series([0, 0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)

    result = run_backtest(crossover_ohlcv, signal, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "buy"

    # Must execute at bar 3 (T+1), not bar 2 (T).
    assert trade.timestamp == crossover_ohlcv["timestamp"].iloc[3]
    assert trade.timestamp != crossover_ohlcv["timestamp"].iloc[2]

    # Must use bar 3's OPEN (50.0), never bar 2's close (10.0) or bar 3's
    # close (55.0) — either of those would be look-ahead.
    assert trade.price == pytest.approx(50.0)
    assert trade.price != pytest.approx(10.0)
    assert trade.price != pytest.approx(55.0)


def test_no_trade_on_first_bar_with_no_prior_signal(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([1, 1, 1, 1, 1])  # signal is "long" from the very first bar
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)

    # There is no bar "-1" close to have generated a decision before bar 0,
    # so the position entering bar 0 must be flat; the earliest possible
    # trade is at bar 1's open (signal known at bar 0's close).
    assert len(result.trades) == 1
    assert result.trades[0].timestamp == crossover_ohlcv["timestamp"].iloc[1]


def test_flat_signal_produces_no_trades(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 0, 0, 0])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)
    assert result.trades == []
    assert result.final_equity == pytest.approx(1000.0)


def test_no_duplicate_entry_while_already_long(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([1, 1, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)
    buys = [t for t in result.trades if t.side == "buy"]
    assert len(buys) == 1


def test_no_exit_while_already_flat(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 0, 0, 0])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)
    sells = [t for t in result.trades if t.side == "sell"]
    assert len(sells) == 0


def test_round_trip_buy_then_sell(crossover_ohlcv: pd.DataFrame) -> None:
    # signal[2]=1: long decided at close of bar 2 -> executes at bar 3 open.
    # signal[3]=0: flat decided at close of bar 3 -> executes at bar 4 open.
    signal = pd.Series([0, 0, 1, 0, 0])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)

    assert [t.side for t in result.trades] == ["buy", "sell"]
    buy, sell = result.trades
    assert buy.timestamp == crossover_ohlcv["timestamp"].iloc[3]
    assert sell.timestamp == crossover_ohlcv["timestamp"].iloc[4]
    assert sell.price == pytest.approx(60.0)  # bar 4's open
    assert sell.units_after == 0.0
    assert result.metrics["num_completed_trades"] == 1


def test_commission_reduces_equity(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 1, 1, 0])
    no_commission = run_backtest(
        crossover_ohlcv, signal, BacktestConfig(initial_cash=1000.0, commission_rate=0.0)
    )
    with_commission = run_backtest(
        crossover_ohlcv, signal, BacktestConfig(initial_cash=1000.0, commission_rate=0.01)
    )
    assert with_commission.final_equity < no_commission.final_equity


def test_slippage_worsens_execution(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 1, 1, 1])
    no_slippage = run_backtest(
        crossover_ohlcv, signal, BacktestConfig(initial_cash=1000.0, slippage_rate=0.0)
    )
    with_slippage = run_backtest(
        crossover_ohlcv, signal, BacktestConfig(initial_cash=1000.0, slippage_rate=0.05)
    )
    # Slippage makes buying more expensive -> fewer units bought.
    assert with_slippage.trades[0].quantity < no_slippage.trades[0].quantity
    assert with_slippage.final_equity < no_slippage.final_equity


def test_no_negative_cash(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 1, 0, 1, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.01, slippage_rate=0.02)
    result = run_backtest(crossover_ohlcv, signal, config)
    assert (result.equity_curve["cash"] >= -1e-9).all()


def test_accounting_identity_holds(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 1, 1, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)
    result = run_backtest(crossover_ohlcv, signal, config)

    curve = result.equity_curve
    expected_position_value = curve["units"] * crossover_ohlcv["close"]
    pd.testing.assert_series_equal(
        curve["position_value"], expected_position_value, check_names=False
    )
    expected_equity = curve["cash"] + curve["position_value"]
    pd.testing.assert_series_equal(curve["equity"], expected_equity, check_names=False)


def test_deterministic_repeatability(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 1, 1, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.005, slippage_rate=0.01)

    result_a = run_backtest(crossover_ohlcv, signal, config)
    result_b = run_backtest(crossover_ohlcv, signal, config)

    pd.testing.assert_frame_equal(result_a.equity_curve, result_b.equity_curve)
    assert result_a.trades == result_b.trades
    assert result_a.final_equity == result_b.final_equity


def test_no_leverage_full_position_sizing(crossover_ohlcv: pd.DataFrame) -> None:
    signal = pd.Series([0, 0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(crossover_ohlcv, signal, config)
    buy = result.trades[0]
    # Full-position entry: cash after buying should be ~0 (no leverage, no
    # cash left idle beyond floating point noise).
    assert buy.cash_after == pytest.approx(0.0, abs=1e-9)


def test_signal_length_mismatch_rejected(crossover_ohlcv: pd.DataFrame) -> None:
    config = BacktestConfig(initial_cash=1000.0)
    with pytest.raises(ValueError, match="length"):
        run_backtest(crossover_ohlcv, pd.Series([0, 1, 0]), config)


def test_invalid_signal_values_rejected(crossover_ohlcv: pd.DataFrame) -> None:
    config = BacktestConfig(initial_cash=1000.0)
    with pytest.raises(ValueError, match="only 0"):
        run_backtest(crossover_ohlcv, pd.Series([0, 1, 2, 0, 1]), config)


@pytest.mark.parametrize(
    ("initial_cash", "commission_rate", "slippage_rate"),
    [(0.0, 0.0, 0.0), (-100.0, 0.0, 0.0), (1000.0, -0.01, 0.0), (1000.0, 0.0, -0.01)],
)
def test_invalid_config_rejected(
    initial_cash: float, commission_rate: float, slippage_rate: float
) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )


def test_malformed_ohlcv_rejected() -> None:
    broken = make_ohlcv([bar(0, 10.0, 5.0, 4.0, 6.0)])  # high below close
    config = BacktestConfig(initial_cash=1000.0)
    with pytest.raises(OHLCVValidationError, match="high is below"):
        run_backtest(broken, pd.Series([0]), config)


@pytest.mark.parametrize(
    ("initial_cash", "commission_rate", "slippage_rate"),
    [
        (float("nan"), 0.0, 0.0),
        (1000.0, float("nan"), 0.0),
        (1000.0, 0.0, float("nan")),
        (float("inf"), 0.0, 0.0),
        (float("-inf"), 0.0, 0.0),
        (1000.0, float("inf"), 0.0),
        (1000.0, 0.0, float("inf")),
        (True, 0.0, 0.0),
        (1000.0, True, 0.0),
        (1000.0, 0.0, True),
        (1000.0, 1.0, 0.0),
        (1000.0, 1.5, 0.0),
        (1000.0, 0.0, 1.0),
        (1000.0, 0.0, 1.1),
    ],
)
def test_non_finite_and_out_of_range_config_rejected(
    initial_cash: float, commission_rate: float, slippage_rate: float
) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )


def test_empty_ohlcv_rejected_before_engine_runs(crossover_ohlcv: pd.DataFrame) -> None:
    empty = crossover_ohlcv.iloc[0:0]
    config = BacktestConfig(initial_cash=1000.0)
    with pytest.raises(OHLCVValidationError, match="empty"):
        run_backtest(empty, pd.Series([], dtype=int), config)


def test_buy_sizing_accounting_identity_exact(crossover_ohlcv: pd.DataFrame) -> None:
    # quantity * exec_price * (1 + commission_rate) must equal the cash spent
    # exactly (within floating point tolerance) -- this is the full-position
    # sizing invariant that guarantees cash never goes negative from a
    # sizing error.
    signal = pd.Series([0, 0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.01, slippage_rate=0.02)
    result = run_backtest(crossover_ohlcv, signal, config)

    buy = result.trades[0]
    open_price = crossover_ohlcv["open"].iloc[3]
    expected_exec_price = open_price * (1.0 + config.slippage_rate)
    assert buy.price == pytest.approx(expected_exec_price)
    assert buy.quantity * buy.price * (1.0 + config.commission_rate) == pytest.approx(
        1000.0, rel=1e-12
    )
    assert buy.commission == pytest.approx(buy.quantity * buy.price * config.commission_rate)
    assert buy.cash_after == pytest.approx(0.0, abs=1e-9)
