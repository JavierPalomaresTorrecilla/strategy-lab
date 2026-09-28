"""Backtest layer: historical simulation of strategies against market data.

Must avoid look-ahead bias, survivorship bias, and unrealistic execution
assumptions. See ``strategy_lab.backtest.engine`` for execution timing
semantics.
"""

from strategy_lab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from strategy_lab.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    num_completed_trades,
    total_return,
)
from strategy_lab.backtest.trades import Trade

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "Trade",
    "compute_metrics",
    "total_return",
    "max_drawdown",
    "num_completed_trades",
]
