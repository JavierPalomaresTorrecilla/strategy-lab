"""Tests for round-trip trade reconstruction from executed Trade records."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.backtest.trades import Trade
from strategy_lab.validation.round_trips import (
    RoundTripConstructionError,
    build_round_trips,
    burn_in_entered_round_trips,
    eligible_round_trips,
    is_cross_boundary,
    realized_pnl_concentration,
)


def _trade(side, ts, price, quantity, commission) -> Trade:
    return Trade(
        timestamp=pd.Timestamp(ts),
        side=side,
        price=price,
        quantity=quantity,
        commission=commission,
        cash_after=0.0,
        units_after=0.0 if side == "sell" else quantity,
    )


def test_build_round_trips_pairs_buy_sell_and_uses_executed_commissions() -> None:
    trades = [
        _trade("buy", "2011-01-01", 100.0, 10.0, 1.0),
        _trade("sell", "2011-02-01", 110.0, 10.0, 1.1),
    ]
    round_trips, open_trade = build_round_trips(trades)
    assert open_trade is None
    assert len(round_trips) == 1
    rt = round_trips[0]
    assert rt.entry_timestamp == pd.Timestamp("2011-01-01")
    assert rt.exit_timestamp == pd.Timestamp("2011-02-01")
    expected_pnl = (110.0 - 100.0) * 10.0 - 1.0 - 1.1
    assert rt.realized_pnl == pytest.approx(expected_pnl)


def test_trailing_buy_is_open_not_a_round_trip() -> None:
    trades = [
        _trade("buy", "2011-01-01", 100.0, 10.0, 1.0),
        _trade("sell", "2011-02-01", 110.0, 10.0, 1.1),
        _trade("buy", "2011-06-01", 120.0, 5.0, 0.5),
    ]
    round_trips, open_trade = build_round_trips(trades)
    assert len(round_trips) == 1
    assert open_trade is not None
    assert open_trade.timestamp == pd.Timestamp("2011-06-01")


def test_no_trades_produces_no_round_trips_and_no_open_position() -> None:
    round_trips, open_trade = build_round_trips([])
    assert round_trips == []
    assert open_trade is None


def test_consecutive_buys_rejected() -> None:
    trades = [
        _trade("buy", "2011-01-01", 100.0, 10.0, 1.0),
        _trade("buy", "2011-02-01", 100.0, 10.0, 1.0),
    ]
    with pytest.raises(RoundTripConstructionError):
        build_round_trips(trades)


def test_sell_without_prior_buy_rejected() -> None:
    trades = [_trade("sell", "2011-01-01", 100.0, 10.0, 1.0)]
    with pytest.raises(RoundTripConstructionError):
        build_round_trips(trades)


def test_quantity_mismatch_beyond_tolerance_rejected() -> None:
    trades = [
        _trade("buy", "2011-01-01", 100.0, 10.0, 1.0),
        _trade("sell", "2011-02-01", 110.0, 9.9, 1.0),
    ]
    with pytest.raises(RoundTripConstructionError, match="quantity mismatch"):
        build_round_trips(trades)


def test_quantity_mismatch_within_tolerance_accepted() -> None:
    trades = [
        _trade("buy", "2011-01-01", 100.0, 10.0, 1.0),
        _trade("sell", "2011-02-01", 110.0, 10.0 + 1e-12, 1.0),
    ]
    round_trips, open_trade = build_round_trips(trades)
    assert len(round_trips) == 1
    assert open_trade is None


def test_cross_boundary_entering_2011_exiting_2012_flags_both_windows() -> None:
    round_trips, _ = build_round_trips(
        [
            _trade("buy", "2011-06-01", 100.0, 10.0, 1.0),
            _trade("sell", "2012-03-01", 110.0, 10.0, 1.0),
        ]
    )
    rt = round_trips[0]

    window_2011_start, window_2011_end = pd.Timestamp("2011-01-01"), pd.Timestamp("2012-01-01")
    window_2012_start, window_2012_end = pd.Timestamp("2012-01-01"), pd.Timestamp("2013-01-01")

    assert is_cross_boundary(rt, window_2011_start, window_2011_end) is True
    assert is_cross_boundary(rt, window_2012_start, window_2012_end) is True


def test_fully_contained_round_trip_is_not_cross_boundary() -> None:
    round_trips, _ = build_round_trips(
        [
            _trade("buy", "2011-03-01", 100.0, 10.0, 1.0),
            _trade("sell", "2011-09-01", 110.0, 10.0, 1.0),
        ]
    )
    rt = round_trips[0]
    assert is_cross_boundary(rt, pd.Timestamp("2011-01-01"), pd.Timestamp("2012-01-01")) is False


def test_eligible_round_trips_excludes_burn_in_entry_and_still_open() -> None:
    round_trips, _ = build_round_trips(
        [
            _trade("buy", "2010-11-01", 100.0, 10.0, 1.0),  # burn-in entry
            _trade("sell", "2011-02-01", 110.0, 10.0, 1.0),  # exits during evaluation
            _trade("buy", "2011-06-01", 100.0, 10.0, 1.0),  # fully contained
            _trade("sell", "2011-09-01", 110.0, 10.0, 1.0),
        ]
    )
    eligible = eligible_round_trips(
        round_trips, pd.Timestamp("2011-01-01"), pd.Timestamp("2013-01-01")
    )
    assert len(eligible) == 1
    assert eligible[0].entry_timestamp == pd.Timestamp("2011-06-01")

    burn_in_entered = burn_in_entered_round_trips(round_trips, pd.Timestamp("2011-01-01"))
    assert len(burn_in_entered) == 1
    assert burn_in_entered[0].entry_timestamp == pd.Timestamp("2010-11-01")


def test_realized_pnl_concentration_identifies_dominant_winner() -> None:
    round_trips, _ = build_round_trips(
        [
            _trade("buy", "2011-01-01", 100.0, 1.0, 0.0),
            _trade("sell", "2011-02-01", 101.0, 1.0, 0.0),  # +1
            _trade("buy", "2011-03-01", 100.0, 1.0, 0.0),
            _trade("sell", "2011-04-01", 200.0, 1.0, 0.0),  # +100, dominant
        ]
    )
    stats = realized_pnl_concentration(round_trips)
    assert stats.largest_positive_fraction == pytest.approx(100.0 / 101.0)
    assert stats.round_trips_for_50pct_of_positive_pnl == 1


def test_realized_pnl_concentration_no_positive_trades() -> None:
    round_trips, _ = build_round_trips(
        [
            _trade("buy", "2011-01-01", 100.0, 1.0, 0.0),
            _trade("sell", "2011-02-01", 90.0, 1.0, 0.0),
        ]
    )
    stats = realized_pnl_concentration(round_trips)
    assert stats.largest_positive_fraction is None
    assert stats.round_trips_for_50pct_of_positive_pnl is None
