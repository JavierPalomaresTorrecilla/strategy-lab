"""Deterministic, long-only, single-instrument backtest engine.

Execution timing (look-ahead prevention)
-----------------------------------------
The engine receives a ``signal`` series where ``signal[i]`` is the desired
position *as of the close of bar i* (see ``strategy_lab.strategies``). It
must not be executable using any price available at or before that close.

The engine enforces this by shifting the signal forward by one bar before
acting on it: the position targeted for bar ``i`` is ``signal[i - 1]``. A
change in signal first known at the close of bar T therefore results in a
trade no earlier than bar T + 1, using bar T + 1's **open** price.

The open price (rather than that bar's close) is used because it is the
first price at which a market participant could plausibly act on
information known at the prior close — using T+1's close would silently
grant the strategy visibility into an entire extra bar of price action it
could not have had in practice.

Position sizing
----------------
Only two states are modeled: fully invested (all cash converted to units)
or fully in cash. There is no partial sizing and no leverage.

Costs
-----
Both commission and slippage are proportional rates applied to the
execution price:

- **Slippage** moves the effective execution price against the trader:
  worse (higher) than the quoted open when buying, worse (lower) than the
  quoted open when selling. ``effective_price = open * (1 + slippage_rate)``
  for buys, ``open * (1 - slippage_rate)`` for sells.
- **Commission** is charged as ``commission_rate`` of the notional value of
  the trade (``quantity * effective_price``), deducted from cash.

On a buy, the full available cash is spent such that
``quantity * effective_price * (1 + commission_rate) == cash`` exactly, so
cash can never go negative from a sizing error. On a sell, all held units
are sold; proceeds minus commission are added to cash.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import pandas as pd

from strategy_lab.backtest.metrics import compute_metrics
from strategy_lab.backtest.trades import Trade
from strategy_lab.data.ohlcv import validate_ohlcv


def _require_finite_number(name: str, value: object) -> float:
    """Reject non-numeric, boolean, NaN, and +/-inf configuration values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run.

    - ``initial_cash``: starting cash, must be finite and > 0.
    - ``commission_rate``: proportional commission per trade (e.g. 0.001 for
      10 bps), must be finite and satisfy ``0 <= commission_rate < 1``.
    - ``slippage_rate``: proportional slippage per trade, must be finite and
      satisfy ``0 <= slippage_rate < 1`` (a rate of 1 or more would make the
      effective sell price zero or negative).
    """

    initial_cash: float
    commission_rate: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_number("initial_cash", self.initial_cash)
        _require_finite_number("commission_rate", self.commission_rate)
        _require_finite_number("slippage_rate", self.slippage_rate)

        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {self.initial_cash}")
        if not (0 <= self.commission_rate < 1):
            raise ValueError(
                f"commission_rate must satisfy 0 <= commission_rate < 1, got {self.commission_rate}"
            )
        if not (0 <= self.slippage_rate < 1):
            raise ValueError(
                f"slippage_rate must satisfy 0 <= slippage_rate < 1, got {self.slippage_rate}"
            )


@dataclass
class BacktestResult:
    """Result of a backtest run: enough to inspect and reproduce it."""

    equity_curve: pd.DataFrame
    trades: list[Trade]
    final_equity: float
    metrics: dict[str, float]
    config: BacktestConfig = field(repr=False)


def run_backtest(
    ohlcv: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a deterministic long-only backtest.

    ``ohlcv`` must satisfy ``strategy_lab.data.ohlcv.validate_ohlcv`` (it is
    validated again here, defensively). ``signal`` must have the same length
    as ``ohlcv`` and contain only ``0`` or ``1``; ``signal.iloc[i]`` is the
    desired position as of the close of ``ohlcv.iloc[i]`` (see module
    docstring for how this is translated into executed trades).
    """
    data = validate_ohlcv(ohlcv)

    if len(signal) != len(data):
        raise ValueError(
            f"signal length ({len(signal)}) must match ohlcv length ({len(data)})"
        )
    if not signal.isin([0, 1]).all():
        raise ValueError("signal must contain only 0 (flat) or 1 (long)")

    # Signal known at close of bar i-1 determines the position targeted for
    # bar i's open. Bar 0 has no prior signal, so it starts flat.
    target_position = signal.shift(1).fillna(0).astype(int).reset_index(drop=True)

    cash = float(config.initial_cash)
    units = 0.0
    current_position = 0
    trades: list[Trade] = []
    equity_rows: list[dict[str, float]] = []

    for i in range(len(data)):
        timestamp = data["timestamp"].iloc[i]
        desired = int(target_position.iloc[i])

        if desired == 1 and current_position == 0:
            open_price = float(data["open"].iloc[i])
            exec_price = open_price * (1.0 + config.slippage_rate)
            quantity = cash / (exec_price * (1.0 + config.commission_rate))
            commission = quantity * exec_price * config.commission_rate
            cash = cash - quantity * exec_price - commission
            for name, value in (
                ("execution price", exec_price),
                ("quantity", quantity),
                ("commission", commission),
                ("cash", cash),
            ):
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} ({value}) computed at bar {timestamp}")
            units = quantity
            current_position = 1
            trades.append(
                Trade(
                    timestamp=timestamp,
                    side="buy",
                    price=exec_price,
                    quantity=quantity,
                    commission=commission,
                    cash_after=cash,
                    units_after=units,
                )
            )
        elif desired == 0 and current_position == 1:
            open_price = float(data["open"].iloc[i])
            exec_price = open_price * (1.0 - config.slippage_rate)
            proceeds = units * exec_price
            commission = proceeds * config.commission_rate
            cash = cash + proceeds - commission
            for name, value in (
                ("execution price", exec_price),
                ("proceeds", proceeds),
                ("commission", commission),
                ("cash", cash),
            ):
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} ({value}) computed at bar {timestamp}")
            sold_quantity = units
            units = 0.0
            current_position = 0
            trades.append(
                Trade(
                    timestamp=timestamp,
                    side="sell",
                    price=exec_price,
                    quantity=sold_quantity,
                    commission=commission,
                    cash_after=cash,
                    units_after=units,
                )
            )
        # else: desired position already matches current holdings; no trade.

        close_price = float(data["close"].iloc[i])
        position_value = units * close_price
        equity = cash + position_value
        for name, value in (("position value", position_value), ("equity", equity)):
            if not math.isfinite(value):
                raise ValueError(f"non-finite {name} ({value}) computed at bar {timestamp}")
        equity_rows.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "units": units,
                "position_value": position_value,
                "equity": equity,
            }
        )

        if cash < -1e-9:
            raise AssertionError("internal error: cash went negative")

    equity_curve = pd.DataFrame(equity_rows)
    final_equity = float(equity_curve["equity"].iloc[-1])
    metrics = compute_metrics(equity_curve["equity"], trades)

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_equity=final_equity,
        metrics=metrics,
        config=config,
    )
