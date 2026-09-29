"""Robustness metrics with explicit sample-size gates.

Deliberately separate from ``strategy_lab.backtest.metrics`` (the M1
minimal metrics module, which stays untouched): these metrics establish
annualization/risk-free assumptions M1 explicitly declined to make, and
several of them are only meaningful above a minimum sample size. Below
that size, functions here return an explicit ``InsufficientSample`` or
``NotApplicable`` sentinel -- never a silently-misleading float, ``NaN``,
or ``inf``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252

MIN_DAILY_OBSERVATIONS_FOR_SHARPE = 60
MIN_ELIGIBLE_ROUND_TRIPS = 20


@dataclass(frozen=True)
class InsufficientSample:
    """A metric could not be computed because too few observations exist."""

    metric: str
    required: int
    actual: int


@dataclass(frozen=True)
class NotApplicable:
    """A metric is well-defined as a concept but undefined for this input
    (e.g. profit factor with zero gross losses, or zero return variance)."""

    metric: str
    reason: str


def cagr(start_equity: float, end_equity: float, years: float) -> float:
    """Compound annual growth rate over an exact ``years``-year span."""
    if years <= 0:
        raise ValueError(f"years must be > 0, got {years}")
    if start_equity <= 0:
        raise ValueError(f"start_equity must be > 0, got {start_equity}")
    return (end_equity / start_equity) ** (1.0 / years) - 1.0


def annualized_volatility(daily_returns: list[float]) -> float | InsufficientSample:
    """Sample standard deviation of ``daily_returns``, annualized by
    ``sqrt(TRADING_DAYS_PER_YEAR)``. Requires at least 2 observations."""
    if len(daily_returns) < 2:
        return InsufficientSample("annualized_volatility", required=2, actual=len(daily_returns))
    return statistics.stdev(daily_returns) * (TRADING_DAYS_PER_YEAR**0.5)


def local_max_drawdown(equity_values: list[float], start_equity: float) -> float:
    """Max drawdown over ``equity_values``, with the running peak
    initialized at ``start_equity`` -- never inherited from an earlier
    period's own running peak."""
    if not equity_values:
        raise ValueError("equity_values must not be empty")
    peak = start_equity
    max_dd = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def exposure(position_flags: list[bool]) -> float:
    """Fraction of observations with a nonzero position."""
    if not position_flags:
        raise ValueError("position_flags must not be empty")
    return sum(1 for flag in position_flags if flag) / len(position_flags)


def sharpe_like(
    daily_returns: list[float], risk_free_rate: float = 0.0
) -> float | InsufficientSample | NotApplicable:
    """Descriptive Sharpe-like ratio from daily equity returns -- not a
    formal significance result. Gated on a minimum number of *daily*
    observations, never on trade count.

    ``risk_free_rate`` is an ANNUAL rate (e.g. ``0.02`` for 2%/year), not a
    daily one -- it is converted to a daily rate via
    ``risk_free_rate / TRADING_DAYS_PER_YEAR`` before being subtracted from
    each daily return. This matches ``config/research.yaml``'s
    ``temporal_validation.risk_free_rate`` (currently ``0.0``, so this
    conversion is presently a no-op either way).
    """
    if len(daily_returns) < MIN_DAILY_OBSERVATIONS_FOR_SHARPE:
        return InsufficientSample(
            "sharpe_like", required=MIN_DAILY_OBSERVATIONS_FOR_SHARPE, actual=len(daily_returns)
        )
    daily_risk_free = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = [r - daily_risk_free for r in daily_returns]
    stdev = statistics.stdev(excess_returns)
    if stdev == 0:
        return NotApplicable("sharpe_like", "zero return volatility")
    mean_excess = statistics.fmean(excess_returns)
    return (mean_excess / stdev) * (TRADING_DAYS_PER_YEAR**0.5)


def hit_rate(round_trip_pnls: list[float]) -> float | InsufficientSample:
    """Fraction of round trips with positive realized P&L. Gated on a
    minimum number of *eligible round trips*, never a minimum window
    count."""
    n = len(round_trip_pnls)
    if n < MIN_ELIGIBLE_ROUND_TRIPS:
        return InsufficientSample("hit_rate", required=MIN_ELIGIBLE_ROUND_TRIPS, actual=n)
    wins = sum(1 for pnl in round_trip_pnls if pnl > 0)
    return wins / n


def profit_factor(round_trip_pnls: list[float]) -> float | InsufficientSample | NotApplicable:
    """Gross profit / gross loss across round trips. Gated on the same
    minimum eligible-round-trip count as ``hit_rate``; ``NotApplicable``
    (never ``inf``) when gross losses are exactly zero."""
    n = len(round_trip_pnls)
    if n < MIN_ELIGIBLE_ROUND_TRIPS:
        return InsufficientSample("profit_factor", required=MIN_ELIGIBLE_ROUND_TRIPS, actual=n)
    gross_profit = sum(pnl for pnl in round_trip_pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in round_trip_pnls if pnl < 0)
    if gross_loss == 0:
        return NotApplicable("profit_factor", "zero gross losses")
    return gross_profit / gross_loss
