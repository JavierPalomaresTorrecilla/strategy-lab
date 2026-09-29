"""Tests for fixed-parameter expanding-history temporal validation."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import random_walk_ohlcv
from strategy_lab.backtest.engine import BacktestConfig, run_backtest
from strategy_lab.data.splits import DatasetSplit
from strategy_lab.validation.robustness_metrics import InsufficientSample
from strategy_lab.validation.temporal_validation import (
    WindowBoundaryError,
    _daily_returns,
    build_windows,
    equity_before,
    evaluate,
    filter_to_allowed_research_history,
    leave_one_window_out,
    window_return,
)

# --- build_windows --------------------------------------------------------


def test_build_windows_exactly_twelve_contiguous_years() -> None:
    burn_in, windows = build_windows(burn_in_year=2010, evaluation_years=list(range(2011, 2023)))
    assert len(windows) == 12
    assert [w.name for w in windows] == list(range(2011, 2023))
    assert burn_in.start == pd.Timestamp("2010-01-01")
    assert burn_in.end == pd.Timestamp("2011-01-01") == windows[0].start
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert earlier.end == later.start  # contiguous
    assert windows[-1].end == pd.Timestamp("2023-01-01")


def test_build_windows_rejects_non_consecutive_years() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        build_windows(burn_in_year=2010, evaluation_years=[2011, 2013])


def test_build_windows_rejects_unsorted_years() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_windows(burn_in_year=2010, evaluation_years=[2012, 2011])


def test_build_windows_rejects_mismatched_burn_in_year() -> None:
    with pytest.raises(ValueError, match="burn_in_year"):
        build_windows(burn_in_year=2009, evaluation_years=[2011, 2012])


def test_build_windows_rejects_empty_evaluation_years() -> None:
    with pytest.raises(ValueError):
        build_windows(burn_in_year=2010, evaluation_years=[])


# --- equity_before / window_return -----------------------------------------


def _equity_series(pairs: list[tuple[str, float]]) -> tuple[pd.Series, pd.Series]:
    ts = pd.Series([pd.Timestamp(d) for d, _ in pairs])
    eq = pd.Series([v for _, v in pairs])
    return ts, eq


def test_equity_before_resolves_last_observation_strictly_before_boundary() -> None:
    ts, eq = _equity_series([("2011-12-29", 100.0), ("2011-12-30", 101.0), ("2012-01-03", 105.0)])
    # 2012-01-01 falls on a weekend -- the last observation strictly before
    # it is the last trading day of 2011 (2011-12-30), not 2012-01-03.
    resolved_ts, resolved_eq = equity_before(ts, eq, pd.Timestamp("2012-01-01"))
    assert resolved_ts == pd.Timestamp("2011-12-30")
    assert resolved_eq == pytest.approx(101.0)


def test_equity_before_raises_when_no_prior_observation() -> None:
    ts, eq = _equity_series([("2011-06-01", 100.0)])
    with pytest.raises(WindowBoundaryError):
        equity_before(ts, eq, pd.Timestamp("2011-01-01"))


def test_window_return_uses_boundary_equity_not_capital_reset() -> None:
    from strategy_lab.validation.temporal_validation import EvaluationWindow

    ts, eq = _equity_series(
        [
            ("2010-12-31", 100.0),
            ("2011-06-01", 150.0),
            ("2011-12-30", 120.0),
            ("2012-06-01", 200.0),
        ]
    )
    window_2011 = EvaluationWindow(
        name=2011, start=pd.Timestamp("2011-01-01"), end=pd.Timestamp("2012-01-01")
    )
    r = window_return(ts, eq, window_2011)
    assert r == pytest.approx(120.0 / 100.0 - 1.0)


def test_daily_returns_uses_boundary_start_equity_for_first_observation() -> None:
    returns = _daily_returns(start_equity=100.0, equity_values=[110.0, 121.0])
    # Exactly 2 observations (not 1) -- the first day's return is computed
    # against start_equity, not dropped as NaN.
    assert len(returns) == 2
    assert returns[0] == pytest.approx(0.10)
    assert returns[1] == pytest.approx(0.10)


# --- leave_one_window_out ----------------------------------------------------


def test_leave_one_window_out_compounding_arithmetic() -> None:
    returns = {2011: 0.10, 2012: 0.10, 2013: -0.50}
    report = leave_one_window_out(returns)
    full_expected = (1.10 * 1.10 * 0.50) - 1.0
    assert report.full_aggregate == pytest.approx(full_expected)
    without_2013_expected = (1.10 * 1.10) - 1.0
    assert report.aggregate_without[2013] == pytest.approx(without_2013_expected)


def test_leave_one_window_out_detects_sign_flip() -> None:
    # Full aggregate is negative only because of one bad window; removing
    # it flips the sign.
    returns = {2011: 0.05, 2012: 0.05, 2013: -0.20}
    report = leave_one_window_out(returns)
    assert report.full_aggregate < 0
    assert 2013 in report.sign_flip_years
    assert report.aggregate_without[2013] > 0


def test_leave_one_window_out_rejects_empty() -> None:
    with pytest.raises(ValueError):
        leave_one_window_out({})


# --- filter_to_allowed_research_history -------------------------------------


def _small_splits() -> list[DatasetSplit]:
    return [
        DatasetSplit(
            name="train", start=pd.Timestamp("2020-01-01"), end_exclusive=pd.Timestamp("2020-04-01")
        ),
        DatasetSplit(
            name="validation",
            start=pd.Timestamp("2020-04-01"),
            end_exclusive=pd.Timestamp("2020-07-01"),
        ),
        DatasetSplit(
            name="test", start=pd.Timestamp("2020-07-01"), end_exclusive=pd.Timestamp("2020-10-01")
        ),
    ]


def test_filter_to_allowed_research_history_drops_test_rows() -> None:
    df = random_walk_ohlcv("2020-01-01", "2020-09-30", seed=1)
    splits = _small_splits()
    filtered = filter_to_allowed_research_history(df, splits)
    assert (filtered["timestamp"] < pd.Timestamp("2020-07-01")).all()
    assert filtered["timestamp"].is_monotonic_increasing


def test_filter_to_allowed_research_history_invariant_to_test_row_mutation() -> None:
    df = random_walk_ohlcv("2020-01-01", "2020-09-30", seed=1)
    splits = _small_splits()

    filtered_a = filter_to_allowed_research_history(df, splits)

    mutated = df.copy()
    test_mask = mutated["timestamp"] >= pd.Timestamp("2020-07-01")
    mutated.loc[test_mask, ["open", "high", "low", "close"]] *= 999.0

    filtered_b = filter_to_allowed_research_history(mutated, splits)

    pd.testing.assert_frame_equal(filtered_a, filtered_b)


# --- full evaluate() integration ---------------------------------------------


def test_evaluate_no_forced_liquidation_across_year_boundary() -> None:
    dates = pd.bdate_range("2010-12-01", "2012-02-15")
    n = len(dates)
    close = pd.Series([100.0] * n)
    open_ = pd.Series([100.0] * n)
    ohlcv = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": [1_000_000] * n,
        }
    )

    # Position opens shortly before the year boundary and closes shortly
    # after it, deliberately straddling 2011/2012.
    last_2011_idx = ohlcv.index[ohlcv["timestamp"] < pd.Timestamp("2012-01-01")][-1]
    first_2012_idx = ohlcv.index[ohlcv["timestamp"] >= pd.Timestamp("2012-01-01")][0]

    signal = pd.Series([0] * n)
    signal.iloc[last_2011_idx - 3 : first_2012_idx + 3] = 1

    config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)
    result = run_backtest(ohlcv, signal, config)

    burn_in_window, evaluation_windows = build_windows(
        burn_in_year=2010, evaluation_years=[2011, 2012]
    )
    tv_result = evaluate(result, burn_in_window, evaluation_windows)

    window_2011 = tv_result.window_metrics[0]
    window_2012 = tv_result.window_metrics[1]

    # The round trip crosses the boundary: not force-closed, and its exit
    # (2012) is where the completed-round-trip count lands, never 2011.
    assert window_2011.completed_round_trip_count == 0
    assert window_2012.completed_round_trip_count == 1
    # Both years see it flagged as cross-boundary.
    assert window_2011.cross_boundary_count == 1
    assert window_2012.cross_boundary_count == 1
    assert len(tv_result.round_trips) == 1
    assert tv_result.open_trade is None


def test_evaluate_causal_future_bar_perturbation_does_not_change_earlier_output() -> None:
    ohlcv = random_walk_ohlcv("2010-01-01", "2012-12-31", seed=7)
    from strategy_lab.strategies.ema_crossover import generate_signals

    config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)

    signal_a = generate_signals(ohlcv["close"], 10, 30)["signal"]
    result_a = run_backtest(ohlcv, signal_a, config)
    burn_in_window, evaluation_windows = build_windows(
        burn_in_year=2010, evaluation_years=[2011, 2012]
    )
    tv_a = evaluate(result_a, burn_in_window, evaluation_windows)

    mutated = ohlcv.copy()
    perturb_mask = mutated["timestamp"] >= pd.Timestamp("2012-06-01")
    mutated.loc[perturb_mask, ["open", "high", "low", "close"]] *= 5.0

    signal_b = generate_signals(mutated["close"], 10, 30)["signal"]
    result_b = run_backtest(mutated, signal_b, config)
    tv_b = evaluate(result_b, burn_in_window, evaluation_windows)

    # 2011's window return is entirely determined by data strictly before
    # the perturbation -- it must be bit-identical.
    assert tv_a.window_metrics[0].total_return == tv_b.window_metrics[0].total_return
    assert tv_a.window_metrics[0].start_equity == tv_b.window_metrics[0].start_equity
    assert tv_a.window_metrics[0].end_equity == tv_b.window_metrics[0].end_equity


def test_evaluate_smoke_test_produces_sensible_aggregate() -> None:
    ohlcv = random_walk_ohlcv("2010-01-01", "2012-12-31", seed=3)
    from strategy_lab.strategies.ema_crossover import generate_signals

    config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)
    signal = generate_signals(ohlcv["close"], 10, 30)["signal"]
    result = run_backtest(ohlcv, signal, config)

    burn_in_window, evaluation_windows = build_windows(
        burn_in_year=2010, evaluation_years=[2011, 2012]
    )
    tv_result = evaluate(result, burn_in_window, evaluation_windows)

    assert len(tv_result.window_metrics) == 2
    agg = tv_result.aggregate_metrics
    assert agg.start == pd.Timestamp("2011-01-01")
    assert agg.end == pd.Timestamp("2013-01-01")
    # Below the 20-round-trip gate on a 2-year synthetic sample -- must be
    # an explicit sentinel, not a float.
    assert isinstance(agg.hit_rate, InsufficientSample)
