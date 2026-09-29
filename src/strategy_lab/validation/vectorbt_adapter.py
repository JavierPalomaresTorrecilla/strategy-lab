"""Run the EMA crossover strategy through VectorBT, aligned to the reference
engine's execution timing.

VectorBT's ``Portfolio.from_signals`` does **not** reproduce the reference
engine's "signal known at close T -> execute at open T+1" convention by
default: given ``entries``/``exits``, it executes at the *same bar's* price
(``close`` by default) unless told otherwise. Reproducing next-open timing
here requires two explicit, separate steps, both performed in this module:

1. **Timing**: ``signal`` (as returned by
   ``strategy_lab.strategies.ema_crossover.generate_signals`` — a
   position-state series, not discrete events) is converted into
   entry/exit *event* arrays, then those event arrays are shifted forward
   by one bar — exactly mirroring ``signal.shift(1)`` inside
   ``strategy_lab.backtest.engine.run_backtest``.
2. **Price**: the *unshifted* ``open`` series is passed as the execution
   ``price`` override, so a shifted-forward entry event on row ``i``
   executes using row ``i``'s own open — which, after the shift, is bar
   ``T+1``'s open for a signal that changed at bar ``T``'s close.

This module is the only place in the codebase allowed to import
``vectorbt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from strategy_lab.backtest.engine import BacktestConfig


@dataclass(frozen=True)
class VectorbtResult:
    """Enough of a VectorBT portfolio run to compare against the reference
    engine: per-trade entry/exit timestamps, sides, fill prices, and the
    resulting equity/return summary."""

    portfolio: vbt.Portfolio
    entry_timestamps: list
    exit_timestamps: list
    entry_prices: list[float]
    exit_prices: list[float]
    num_trades: int
    final_equity: float
    total_return: float


def build_entries_exits(signal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Derive boolean entry/exit crossover-event arrays from a raw,
    unshifted position-state ``signal`` (0=flat, 1=long).

    Treats the bar before the series starts as flat (``0``), matching the
    reference engine's ``signal.shift(1).fillna(0)``.
    """
    previous = signal.shift(1).fillna(0).astype(int)
    current = signal.astype(int)
    entries = (previous == 0) & (current == 1)
    exits = (previous == 1) & (current == 0)
    return entries, exits


def run_vectorbt(
    ohlcv: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig,
) -> VectorbtResult:
    """Run the EMA-crossover position-state ``signal`` through VectorBT,
    executing at next-bar open to match ``strategy_lab.backtest.engine``.

    ``ohlcv`` must already satisfy ``strategy_lab.data.ohlcv.validate_ohlcv``
    (canonical schema, indexed 0..n-1). ``signal`` must be the raw,
    unshifted position-state series (same input the reference engine takes).
    """
    entries, exits = build_entries_exits(signal.reset_index(drop=True))

    # Mirror the reference engine's internal shift-by-one: a crossover event
    # known at bar i's close executes no earlier than bar i+1.
    shifted_entries = entries.shift(1).fillna(False).astype(bool)
    shifted_exits = exits.shift(1).fillna(False).astype(bool)

    close = ohlcv["close"].reset_index(drop=True)
    open_ = ohlcv["open"].reset_index(drop=True)

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=shifted_entries,
        exits=shifted_exits,
        price=open_,
        size=np.inf,
        direction="longonly",
        fees=config.commission_rate,
        slippage=config.slippage_rate,
        init_cash=config.initial_cash,
        freq="1D",
    )

    timestamps = ohlcv["timestamp"].reset_index(drop=True)
    orders = portfolio.orders.records_readable

    entry_rows = orders[orders["Side"] == "Buy"]
    exit_rows = orders[orders["Side"] == "Sell"]

    # "Timestamp" in vectorbt's order records is the *bar position* in the
    # close/price series passed in (an integer row label here, since ohlcv
    # was reset to a plain RangeIndex) -- not the pandas row index of the
    # orders table itself.
    entry_timestamps = [timestamps.iloc[pos] for pos in entry_rows["Timestamp"]]
    exit_timestamps = [timestamps.iloc[pos] for pos in exit_rows["Timestamp"]]

    return VectorbtResult(
        portfolio=portfolio,
        entry_timestamps=entry_timestamps,
        exit_timestamps=exit_timestamps,
        entry_prices=entry_rows["Price"].tolist(),
        exit_prices=exit_rows["Price"].tolist(),
        num_trades=len(exit_rows),
        final_equity=float(portfolio.final_value()),
        total_return=float(portfolio.total_return()),
    )
