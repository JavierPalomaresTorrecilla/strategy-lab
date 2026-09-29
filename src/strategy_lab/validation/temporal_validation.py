"""Fixed-parameter expanding-history temporal validation.

This is walk-forward *evaluation*, not walk-forward *optimization*: a
single, preregistered strategy configuration is run once, continuously,
across the whole allowed chronological history, and its behavior is then
examined per calendar-year window. No parameter is ever fit, re-fit, or
selected between windows -- that boundary is enforced by construction,
since this module never receives more than one ``(fast, slow)``
configuration's ``BacktestResult`` at a time.

2010 = burn-in / state-initialization period, not "warm-up": the
continuous simulation may trade normally during 2010 and cash/position
state carries forward from it, but 2010 contributes zero evaluation
statistics. Evaluation is exactly the 12 calendar years 2011-2022.

Continuous-state, single-simulation design
-------------------------------------------
There is exactly one causal signal computation and exactly one
``run_backtest`` call across the full ``[burn_in.start, evaluation.end)``
range -- never a reset at any calendar boundary. A position opened in one
window and closed in the next is not force-liquidated; its round trip is
simply cross-boundary (see ``strategy_lab.validation.round_trips``).

Window-return semantics
------------------------
For a half-open window ``[start, end)``: ``start_equity`` is the last
equity observation strictly before ``start``; ``end_equity`` is the last
equity observation strictly before ``end``; ``window_return =
end_equity / start_equity - 1``. This captures the window's first trading
day's full move (including any overnight gap from the prior period's last
close) rather than silently dropping it. See ``equity_before``.

Window performance (this module's per-window/aggregate metrics) is derived
*exclusively* from the continuous mark-to-market equity curve -- never from
attributing a round trip's P&L to whichever year it happened to exit in.
Round-trip *counts* are attributed by exit year for reporting, but that is
a count, not a P&L claim; see ``round_trips.is_cross_boundary``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategy_lab.backtest.trades import Trade
from strategy_lab.data.splits import RESEARCH_ALLOWED_SPLITS, DatasetSplit, load_splits, split_ohlcv
from strategy_lab.validation import robustness_metrics
from strategy_lab.validation.robustness_metrics import InsufficientSample, NotApplicable
from strategy_lab.validation.round_trips import (
    ConcentrationStats,
    RoundTrip,
    build_round_trips,
    burn_in_entered_round_trips,
    eligible_round_trips,
    is_cross_boundary,
    realized_pnl_concentration,
)


class WindowBoundaryError(ValueError):
    """Raised when a window boundary cannot be resolved against the
    available equity curve (no observation strictly before it)."""


@dataclass(frozen=True)
class EvaluationWindow:
    """A half-open ``[start, end)`` calendar-year window. ``name`` is the
    calendar year (also used for the burn-in period's own window)."""

    name: int
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class WindowMetrics:
    year: int
    start: pd.Timestamp
    end: pd.Timestamp
    start_equity: float
    end_equity: float
    total_return: float
    annualized_volatility: float | InsufficientSample
    local_max_drawdown: float
    exposure: float
    completed_round_trip_count: int
    cross_boundary_count: int


@dataclass(frozen=True)
class AggregateMetrics:
    start: pd.Timestamp
    end: pd.Timestamp
    start_equity: float
    end_equity: float
    cagr: float
    annualized_volatility: float | InsufficientSample
    max_drawdown: float
    exposure: float
    eligible_round_trip_count: int
    hit_rate: float | InsufficientSample
    profit_factor: float | InsufficientSample | NotApplicable
    sharpe_like: float | InsufficientSample | NotApplicable
    burn_in_entered_count: int
    still_open_at_end: bool


@dataclass(frozen=True)
class LeaveOneOutReport:
    full_aggregate: float
    aggregate_without: dict[int, float]
    sign_flip_years: list[int]
    largest_change: float
    label: str = "concentration diagnostic -- not an independent counterfactual backtest"


@dataclass(frozen=True)
class TemporalValidationResult:
    burn_in_window: EvaluationWindow
    evaluation_windows: list[EvaluationWindow] = field(default_factory=list)
    window_metrics: list[WindowMetrics] = field(default_factory=list)
    aggregate_metrics: AggregateMetrics = None
    leave_one_window_out: LeaveOneOutReport = None
    trade_concentration: ConcentrationStats = None
    round_trips: list[RoundTrip] = field(default_factory=list)
    open_trade: Trade | None = None


def filter_to_allowed_research_history(
    df: pd.DataFrame, splits: list[DatasetSplit] | None = None
) -> pd.DataFrame:
    """Defense-in-depth TEST filter: restrict ``df`` to
    ``RESEARCH_ALLOWED_SPLITS`` (train + validation) before any indicator,
    signal, or backtest is constructed from it. Reuses the existing
    ``strategy_lab.data.splits`` policy constant -- never a second copy."""
    if splits is None:
        splits = load_splits()
    partitions = split_ohlcv(df, splits)
    allowed = pd.concat([partitions[name] for name in RESEARCH_ALLOWED_SPLITS], ignore_index=True)
    return allowed.sort_values("timestamp").reset_index(drop=True)


def build_windows(
    burn_in_year: int, evaluation_years: list[int]
) -> tuple[EvaluationWindow, list[EvaluationWindow]]:
    """Build the burn-in window and the ordered list of evaluation windows.

    ``evaluation_years`` must be sorted, non-empty, and consecutive;
    ``burn_in_year`` must be exactly ``evaluation_years[0] - 1``.
    """
    if not evaluation_years:
        raise ValueError("evaluation_years must not be empty")
    sorted_years = sorted(evaluation_years)
    if sorted_years != list(evaluation_years):
        raise ValueError("evaluation_years must be sorted ascending")
    for earlier, later in zip(sorted_years, sorted_years[1:], strict=False):
        if later != earlier + 1:
            raise ValueError(
                f"evaluation_years must be consecutive; gap between {earlier} and {later}"
            )
    if burn_in_year != evaluation_years[0] - 1:
        raise ValueError(
            f"burn_in_year ({burn_in_year}) must equal evaluation_years[0] - 1 "
            f"({evaluation_years[0] - 1})"
        )

    burn_in_window = EvaluationWindow(
        name=burn_in_year,
        start=pd.Timestamp(f"{burn_in_year}-01-01"),
        end=pd.Timestamp(f"{evaluation_years[0]}-01-01"),
    )
    evaluation_windows = [
        EvaluationWindow(
            name=year,
            start=pd.Timestamp(f"{year}-01-01"),
            end=pd.Timestamp(f"{year + 1}-01-01"),
        )
        for year in evaluation_years
    ]
    return burn_in_window, evaluation_windows


def equity_before(
    timestamps: pd.Series, equity_values: pd.Series, boundary: pd.Timestamp
) -> tuple[pd.Timestamp, float]:
    """The ``(timestamp, equity)`` pair of the latest observation strictly
    before ``boundary``. Raises ``WindowBoundaryError`` if none exists.

    ``timestamps``/``equity_values`` are treated positionally (reset to a
    plain range index internally) so this works regardless of the
    original index of whichever equity source (reference engine or
    VectorBT) produced them.
    """
    ts = timestamps.reset_index(drop=True)
    eq = equity_values.reset_index(drop=True)
    mask = ts < boundary
    if not mask.any():
        raise WindowBoundaryError(f"no equity observation strictly before {boundary}")
    position = mask[mask].index[-1]
    return ts.iloc[position], float(eq.iloc[position])


def window_return(
    timestamps: pd.Series, equity_values: pd.Series, window: EvaluationWindow
) -> float:
    """``end_equity / start_equity - 1`` using ``equity_before`` at both
    boundaries of ``window``. No capital reset; positions may span the
    boundary freely."""
    _, start_equity = equity_before(timestamps, equity_values, window.start)
    _, end_equity = equity_before(timestamps, equity_values, window.end)
    return end_equity / start_equity - 1.0


def _daily_returns(start_equity: float, equity_values: list[float]) -> list[float]:
    """Daily returns for a range, with the *first* return computed against
    ``start_equity`` (the boundary value) rather than dropped -- this
    yields exactly ``len(equity_values)`` observations, never
    ``len(equity_values) - 1``, so annualization is never silently
    understated by one missing day."""
    previous = start_equity
    returns = []
    for equity in equity_values:
        returns.append(equity / previous - 1.0)
        previous = equity
    return returns


def leave_one_window_out(window_returns: dict[int, float]) -> LeaveOneOutReport:
    """Compounding-correct leave-one-window-out concentration diagnostic.
    Not an independent counterfactual backtest -- purely arithmetic on the
    already-realized per-window returns."""
    years = sorted(window_returns)
    if not years:
        raise ValueError("window_returns must not be empty")

    full_factor = 1.0
    for year in years:
        full_factor *= 1.0 + window_returns[year]
    full_aggregate = full_factor - 1.0

    without: dict[int, float] = {}
    for excluded_year in years:
        factor = 1.0
        for year in years:
            if year == excluded_year:
                continue
            factor *= 1.0 + window_returns[year]
        without[excluded_year] = factor - 1.0

    sign_flip_years = [year for year in years if (without[year] >= 0) != (full_aggregate >= 0)]
    largest_change = max(abs(without[year] - full_aggregate) for year in years)

    return LeaveOneOutReport(
        full_aggregate=full_aggregate,
        aggregate_without=without,
        sign_flip_years=sign_flip_years,
        largest_change=largest_change,
    )


def evaluate(
    backtest_result,
    burn_in_window: EvaluationWindow,
    evaluation_windows: list[EvaluationWindow],
    risk_free_rate: float = 0.0,
) -> TemporalValidationResult:
    """Compute the full M3 temporal-validation report from one continuous
    ``strategy_lab.backtest.engine.BacktestResult`` spanning
    ``[burn_in_window.start, evaluation_windows[-1].end)``.

    ``backtest_result`` must come from a single, unbroken ``run_backtest``
    call over that whole range -- this function does not itself run the
    backtest, so it never has the opportunity to reset capital or state at
    a boundary; it only slices the one continuous result it's given.
    """
    equity_curve = backtest_result.equity_curve
    timestamps = equity_curve["timestamp"]
    equity_values = equity_curve["equity"]
    units_values = equity_curve["units"]

    round_trips, open_trade = build_round_trips(backtest_result.trades)

    window_metrics: list[WindowMetrics] = []
    window_returns: dict[int, float] = {}

    for window in evaluation_windows:
        _, start_equity = equity_before(timestamps, equity_values, window.start)
        _, end_equity = equity_before(timestamps, equity_values, window.end)
        total_return = end_equity / start_equity - 1.0
        window_returns[window.name] = total_return

        mask = (timestamps >= window.start) & (timestamps < window.end)
        window_equity = equity_values.loc[mask].tolist()
        window_units = units_values.loc[mask].tolist()

        daily_returns = _daily_returns(start_equity, window_equity)
        volatility = robustness_metrics.annualized_volatility(daily_returns)
        local_dd = robustness_metrics.local_max_drawdown(window_equity, start_equity)
        expo = robustness_metrics.exposure([unit != 0 for unit in window_units])

        completed_count = sum(
            1 for rt in round_trips if window.start <= rt.exit_timestamp < window.end
        )
        cross_boundary_count = sum(
            1 for rt in round_trips if is_cross_boundary(rt, window.start, window.end)
        )

        window_metrics.append(
            WindowMetrics(
                year=window.name,
                start=window.start,
                end=window.end,
                start_equity=start_equity,
                end_equity=end_equity,
                total_return=total_return,
                annualized_volatility=volatility,
                local_max_drawdown=local_dd,
                exposure=expo,
                completed_round_trip_count=completed_count,
                cross_boundary_count=cross_boundary_count,
            )
        )

    evaluation_start = evaluation_windows[0].start
    evaluation_end = evaluation_windows[-1].end

    _, agg_start_equity = equity_before(timestamps, equity_values, evaluation_start)
    _, agg_end_equity = equity_before(timestamps, equity_values, evaluation_end)
    agg_cagr = robustness_metrics.cagr(agg_start_equity, agg_end_equity, len(evaluation_windows))

    agg_mask = (timestamps >= evaluation_start) & (timestamps < evaluation_end)
    agg_equity = equity_values.loc[agg_mask].tolist()
    agg_units = units_values.loc[agg_mask].tolist()
    agg_daily_returns = _daily_returns(agg_start_equity, agg_equity)
    agg_volatility = robustness_metrics.annualized_volatility(agg_daily_returns)
    agg_drawdown = robustness_metrics.local_max_drawdown(agg_equity, agg_start_equity)
    agg_exposure = robustness_metrics.exposure([unit != 0 for unit in agg_units])
    agg_sharpe = robustness_metrics.sharpe_like(agg_daily_returns, risk_free_rate)

    eligible = eligible_round_trips(round_trips, evaluation_start, evaluation_end)
    eligible_pnls = [rt.realized_pnl for rt in eligible]
    agg_hit_rate = robustness_metrics.hit_rate(eligible_pnls)
    agg_profit_factor = robustness_metrics.profit_factor(eligible_pnls)
    burn_in_entered = burn_in_entered_round_trips(round_trips, evaluation_start)

    aggregate_metrics = AggregateMetrics(
        start=evaluation_start,
        end=evaluation_end,
        start_equity=agg_start_equity,
        end_equity=agg_end_equity,
        cagr=agg_cagr,
        annualized_volatility=agg_volatility,
        max_drawdown=agg_drawdown,
        exposure=agg_exposure,
        eligible_round_trip_count=len(eligible),
        hit_rate=agg_hit_rate,
        profit_factor=agg_profit_factor,
        sharpe_like=agg_sharpe,
        burn_in_entered_count=len(burn_in_entered),
        still_open_at_end=open_trade is not None,
    )

    loo = leave_one_window_out(window_returns)
    concentration = realized_pnl_concentration(eligible)

    return TemporalValidationResult(
        burn_in_window=burn_in_window,
        evaluation_windows=evaluation_windows,
        window_metrics=window_metrics,
        aggregate_metrics=aggregate_metrics,
        leave_one_window_out=loo,
        trade_concentration=concentration,
        round_trips=round_trips,
        open_trade=open_trade,
    )
