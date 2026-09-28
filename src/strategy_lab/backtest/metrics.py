"""Minimal backtest performance metrics.

Deliberately small: total return, maximum drawdown, and completed trade
count. No annualized/risk-adjusted metrics (e.g. Sharpe ratio) are included
here — those require explicit, tested assumptions about bar frequency and
annualization that this milestone does not establish.
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.backtest.trades import Trade


def total_return(equity_curve: pd.Series) -> float:
    """Total return over the full equity curve: ``equity[-1] / equity[0] - 1``."""
    if len(equity_curve) == 0:
        raise ValueError("equity_curve must not be empty")
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, expressed as a negative fraction.

    E.g. ``-0.2`` means the largest observed decline from a prior equity
    peak was 20%. ``0.0`` if equity never declined from its running peak.
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve must not be empty")
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def num_completed_trades(trades: list[Trade]) -> int:
    """Number of completed round-trip trades (each closing sell counts once)."""
    return sum(1 for trade in trades if trade.side == "sell")


def compute_metrics(equity_curve: pd.Series, trades: list[Trade]) -> dict[str, float]:
    """Convenience wrapper computing all metrics in this module."""
    return {
        "total_return": total_return(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "num_completed_trades": num_completed_trades(trades),
    }
