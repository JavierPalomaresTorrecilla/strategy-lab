"""Synthetic execution-truth fixtures (M6 preregistration section 10).

Every expected outcome in this file is HAND-SPECIFIED from the documented
engine/strategy semantics (``strategy_lab.backtest.engine``,
``strategy_lab.strategies.ema_crossover``, ``strategy_lab.indicators.ema``)
-- never by running either implementation and copying its output. Each
case is then checked against:

1. the reference engine (``run_backtest``);
2. VectorBT, via ``compare_engines``, where a (ohlcv, signal, config)
   triple is well-formed enough to run through both.

This is deliberately independent of, and run before, any Coinbase
performance work. A genuine mismatch here must be reported, not papered
over by editing the engine/trades code to force agreement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import bar, make_ohlcv
from strategy_lab.backtest.engine import BacktestConfig, run_backtest
from strategy_lab.strategies.ema_crossover import generate_signals
from strategy_lab.validation.parity import compare_engines

# --- 1. signal known at close T -> fill at open T+1 -------------------------


def test_signal_close_t_fills_open_t_plus_1() -> None:
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 21.0, 9.0, 20.0),  # signal known at this close
            bar(3, 30.0, 35.0, 25.0, 33.0),  # execution bar: open=30 != close=33
        ]
    )
    signal = pd.Series([0, 0, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)

    # Hand-specified: target_position = signal.shift(1).fillna(0)
    #               = [0, 0, 0, 1] -> first buy decision materializes at bar 3.
    # exec_price = open[3] * (1 + 0 slippage) = 30.0
    # quantity = 1000 / (30 * (1 + 0 commission)) = 33.333333333333336
    # commission = 0; cash_after = 1000 - quantity*30 - 0 = 0.0 (exact)
    # equity[3] = cash_after + quantity * close[3] = 0 + (1000/30)*33 = 1100.0
    result = run_backtest(ohlcv, signal, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "buy"
    assert trade.price == pytest.approx(30.0)
    assert trade.price != pytest.approx(20.0)  # not bar 2's close (look-ahead)
    assert trade.price != pytest.approx(33.0)  # not bar 3's close
    assert trade.quantity == pytest.approx(1000.0 / 30.0)
    assert trade.commission == pytest.approx(0.0)
    assert trade.cash_after == pytest.approx(0.0, abs=1e-9)
    assert result.final_equity == pytest.approx(1100.0)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 2. EMA equality -> flat -------------------------------------------------


def test_ema_equality_gives_flat_signal() -> None:
    # A constant close series: ema[0] = x[0] = 100.0, and by the recursive
    # definition y[t] = alpha*x[t] + (1-alpha)*y[t-1], if y[t-1] == 100.0
    # and x[t] == 100.0 then y[t] == 100.0 for every t, for ANY period.
    # So fast_ema == slow_ema == 100.0 exactly at every bar once defined,
    # and generate_signals uses strict `>`, which is False on equality.
    close = pd.Series([100.0] * 10)
    signals = generate_signals(close, fast_period=2, slow_period=3)

    assert (signals["signal"] == 0).all()
    # Confirm the EMAs are genuinely equal (not just both NaN) past warm-up.
    defined = signals["fast_ema"].notna() & signals["slow_ema"].notna()
    assert defined.sum() > 0
    assert (signals.loc[defined, "fast_ema"] == signals.loc[defined, "slow_ema"]).all()
    assert (signals.loc[defined, "fast_ema"] == 100.0).all()


# --- 3. warm-up -> no premature trade ---------------------------------------


def test_warmup_bars_give_no_premature_trade() -> None:
    # slow_period=3 -> both_defined is False for the first 2 rows by
    # construction (ema()'s min_periods=period masks them to NaN), so
    # signal[0] == signal[1] == 0 regardless of the price path. Through the
    # engine's shift-by-one, target_position[0..2] are therefore all 0 too
    # -- no trade can occur before bar 3, no matter how sharply price rises.
    close = pd.Series([10.0, 50.0, 90.0, 95.0, 96.0])
    signals = generate_signals(close, fast_period=2, slow_period=3)
    assert signals["signal"].iloc[0] == 0
    assert signals["signal"].iloc[1] == 0

    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 50.0, 51.0, 49.0, 50.0),
            bar(2, 90.0, 91.0, 89.0, 90.0),
            bar(3, 95.0, 96.0, 94.0, 95.0),
            bar(4, 96.0, 97.0, 95.0, 96.0),
        ]
    )
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(ohlcv, signals["signal"], config)
    # No trade can land on bar 0, 1, or 2 (target_position there derives
    # only from signal[-1]/signal[0]/signal[1], all 0).
    assert all(t.timestamp != ohlcv["timestamp"].iloc[i] for t in result.trades for i in (0, 1, 2))

    report = compare_engines(ohlcv, signals["signal"], config)
    assert report.is_within_tolerance(), report.divergences


# --- 4. last-bar entry signal -> no impossible fill --------------------------


def test_last_bar_entry_signal_gives_no_impossible_fill() -> None:
    # signal flips to 1 only at the LAST bar's close. There is no bar
    # "n" to hold target_position=1 -- signal.shift(1) is length n and its
    # last element is signal[n-2] (still 0), so this intended entry is
    # simply never materialized. Hand-specified: zero trades, flat equity.
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),
            bar(2, 10.0, 11.0, 9.0, 10.0),
            bar(3, 10.0, 11.0, 9.0, 10.0),
        ]
    )
    signal = pd.Series([0, 0, 0, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(ohlcv, signal, config)

    assert result.trades == []
    assert result.final_equity == pytest.approx(1000.0)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 5. never enter ----------------------------------------------------------


def test_never_enter_produces_no_trades_and_flat_equity() -> None:
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 12.0, 13.0, 11.0, 12.0),
            bar(2, 9.0, 10.0, 8.0, 9.0),
            bar(3, 11.0, 12.0, 10.0, 11.0),
        ]
    )
    signal = pd.Series([0, 0, 0, 0])
    # Nonzero costs configured but never triggered -- proves they cannot
    # spuriously create a trade.
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.001, slippage_rate=0.0005)
    result = run_backtest(ohlcv, signal, config)

    assert result.trades == []
    assert result.final_equity == pytest.approx(1000.0)
    assert (result.equity_curve["equity"] == 1000.0).all()

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 6. enter and never exit -------------------------------------------------


def test_enter_and_never_exit_marks_to_market_every_bar() -> None:
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),  # signal known here: 1
            bar(2, 20.0, 21.0, 19.0, 20.0),  # buy here at open=20
            bar(3, 22.0, 23.0, 21.0, 22.0),  # still long, mark-to-market only
        ]
    )
    signal = pd.Series([0, 1, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(ohlcv, signal, config)

    # Hand-specified: quantity = 1000 / 20 = 50.0 exactly; cash_after = 0.
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "buy"
    assert trade.quantity == pytest.approx(50.0)
    assert trade.cash_after == pytest.approx(0.0, abs=1e-9)
    assert all(t.side != "sell" for t in result.trades)

    # equity[2] = 0 + 50*20 = 1000.0 ; equity[3] = 0 + 50*22 = 1100.0
    equity = result.equity_curve["equity"].tolist()
    assert equity[2] == pytest.approx(1000.0)
    assert equity[3] == pytest.approx(1100.0)
    assert result.final_equity == pytest.approx(1100.0)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 7. gap-open execution ---------------------------------------------------


def test_gap_open_execution_uses_the_gapped_open_not_prior_close() -> None:
    # Bar 2's open (200.0) gaps far above bar 1's close (10.0). The signal
    # known at bar 1's close must still execute at bar 2's own (gapped)
    # open, with slippage applied to THAT open -- never interpolated or
    # anchored to the pre-gap close.
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),  # signal known here: 1
            bar(2, 200.0, 210.0, 190.0, 205.0),  # gap-open bar
        ]
    )
    signal = pd.Series([0, 1, 1])
    config = BacktestConfig(initial_cash=1000.0, slippage_rate=0.01)
    result = run_backtest(ohlcv, signal, config)

    # exec_price = 200.0 * 1.01 = 202.0 ; quantity = 1000 / 202.0
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == pytest.approx(202.0)
    assert trade.price != pytest.approx(10.0 * 1.01)  # not anchored to prior close
    assert trade.quantity == pytest.approx(1000.0 / 202.0)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 8. missing-day sequence --------------------------------------------------


def test_missing_day_sequence_is_positional_not_calendar() -> None:
    # Calendar days 2, 3, 4 are absent (day jumps 1 -> 5) -- the engine
    # must operate purely on row order, with no implicit calendar
    # reindexing or gap-filling. The signal transition at row 1's close
    # must still execute at row 2's open (day 5), not be skipped or
    # reinterpreted because of the calendar gap.
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),  # signal known here: 1
            bar(5, 15.0, 16.0, 14.0, 15.0),  # next ROW, 4 calendar days later
        ]
    )
    signal = pd.Series([0, 1, 1])
    config = BacktestConfig(initial_cash=1000.0)
    result = run_backtest(ohlcv, signal, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.timestamp == ohlcv["timestamp"].iloc[2]
    assert trade.price == pytest.approx(15.0)
    assert trade.quantity == pytest.approx(1000.0 / 15.0)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences


# --- 9. commission and slippage on entry and exit ----------------------------


def test_commission_and_slippage_on_both_entry_and_exit() -> None:
    ohlcv = make_ohlcv(
        [
            bar(0, 10.0, 11.0, 9.0, 10.0),
            bar(1, 10.0, 11.0, 9.0, 10.0),  # entry known here: 1
            bar(2, 100.0, 105.0, 95.0, 100.0),  # buy executes here
            bar(3, 100.0, 105.0, 95.0, 100.0),  # exit known here: 0
            bar(4, 120.0, 125.0, 115.0, 120.0),  # sell executes here
        ]
    )
    signal = pd.Series([0, 1, 1, 0, 0])
    config = BacktestConfig(initial_cash=1000.0, commission_rate=0.01, slippage_rate=0.02)
    result = run_backtest(ohlcv, signal, config)

    assert len(result.trades) == 2
    buy, sell = result.trades

    # Hand-computed off-line by long division, independent of the engine's
    # own code path (literal constants, not expressions mirroring its
    # internal formula -- this is deliberate: a bug reproducing itself
    # identically in both the engine and an equivalent inline expression
    # would otherwise go undetected).
    #
    # Buy at bar 2: exec_price = 100 * 1.02 = 102.0
    # quantity = 1000 / (102.0 * 1.01) = 1000 / 103.02 = 50000/5151
    #          = 9.70685303824... (long division)
    # commission = quantity * exec_price * 0.01 = 990.099009876 * 0.01
    #            = 9.90099009876
    # cash_after = 1000 - 990.099009876 - 9.90099009876 = 0 (exact, by
    # construction of the sizing formula)
    assert buy.side == "buy"
    assert buy.price == pytest.approx(102.0)
    assert buy.quantity == pytest.approx(9.70685303824)
    assert buy.commission == pytest.approx(9.90099009876)
    assert buy.cash_after == pytest.approx(0.0, abs=1e-6)

    # Sell at bar 4: exec_price = 120 * (1 - 0.02) = 117.6
    # proceeds = 9.70685303824 * 117.6 = 1141.525917297024
    # commission = proceeds * 0.01 = 11.41525917297024
    # cash_after = 0 + 1141.525917297024 - 11.41525917297024
    #            = 1130.11065812405376
    assert sell.side == "sell"
    assert sell.price == pytest.approx(117.6)
    assert sell.commission == pytest.approx(11.41525917297024)
    assert sell.cash_after == pytest.approx(1130.11065812405376)
    assert result.final_equity == pytest.approx(1130.11065812405376)

    report = compare_engines(ohlcv, signal, config)
    assert report.is_within_tolerance(), report.divergences
