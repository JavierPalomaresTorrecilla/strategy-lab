"""Round-trip trade reconstruction from executed ``Trade`` records.

Pure post-processing of ``strategy_lab.backtest.trades.Trade`` records
already produced by ``strategy_lab.backtest.engine.run_backtest``. This
module never recomputes commissions or prices from ``BacktestConfig`` --
the executed ``Trade`` records (timestamps, prices, quantities,
commissions) are the sole source of truth for realized P&L. It also never
modifies ``strategy_lab.backtest.trades.Trade`` itself.

The reference engine is single-instrument, always-flat-or-fully-invested:
trades strictly alternate ``buy``, ``sell``, ``buy``, ``sell``, .... Pairing
consecutive buy/sell trades into a closed "round trip" is therefore
unambiguous. A trailing, unpaired ``buy`` means the position was still open
when the simulation ended -- it is not a completed round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy_lab.backtest.trades import Trade

QUANTITY_TOLERANCE = 1e-9

MIN_ELIGIBLE_FOR_CONCENTRATION = 1


class RoundTripConstructionError(ValueError):
    """Raised when executed trades cannot be paired into round trips."""


@dataclass(frozen=True)
class RoundTrip:
    """A single closed round trip: one buy paired with the next sell."""

    entry_timestamp: object
    exit_timestamp: object
    entry_price: float
    exit_price: float
    quantity: float
    entry_commission: float
    exit_commission: float
    realized_pnl: float


def build_round_trips(trades: list[Trade]) -> tuple[list[RoundTrip], Trade | None]:
    """Pair executed buy/sell trades into closed round trips.

    Returns ``(round_trips, open_trade)`` where ``open_trade`` is the
    trailing unpaired buy (position still open at the end of the
    simulation), or ``None`` if the simulation ended flat.

    Raises ``RoundTripConstructionError`` if trades do not strictly
    alternate buy/sell, or if a paired buy/sell's quantities differ by more
    than ``QUANTITY_TOLERANCE``.
    """
    round_trips: list[RoundTrip] = []
    pending_buy: Trade | None = None

    for trade in trades:
        if trade.side == "buy":
            if pending_buy is not None:
                raise RoundTripConstructionError(
                    f"two consecutive buy trades with no intervening sell: "
                    f"{pending_buy.timestamp} and {trade.timestamp}"
                )
            pending_buy = trade
        elif trade.side == "sell":
            if pending_buy is None:
                raise RoundTripConstructionError(
                    f"sell trade at {trade.timestamp} has no matching prior buy"
                )
            if abs(pending_buy.quantity - trade.quantity) > QUANTITY_TOLERANCE:
                raise RoundTripConstructionError(
                    f"round trip quantity mismatch: buy {pending_buy.quantity} "
                    f"at {pending_buy.timestamp} vs sell {trade.quantity} "
                    f"at {trade.timestamp}"
                )
            realized_pnl = (
                (trade.price - pending_buy.price) * pending_buy.quantity
                - pending_buy.commission
                - trade.commission
            )
            round_trips.append(
                RoundTrip(
                    entry_timestamp=pending_buy.timestamp,
                    exit_timestamp=trade.timestamp,
                    entry_price=pending_buy.price,
                    exit_price=trade.price,
                    quantity=pending_buy.quantity,
                    entry_commission=pending_buy.commission,
                    exit_commission=trade.commission,
                    realized_pnl=realized_pnl,
                )
            )
            pending_buy = None
        else:
            raise RoundTripConstructionError(f"unknown trade side: {trade.side!r}")

    return round_trips, pending_buy


def is_cross_boundary(round_trip: RoundTrip, window_start, window_end) -> bool:
    """Whether ``round_trip``'s position lifetime crosses either boundary
    of the half-open window ``[window_start, window_end)``.

    A round trip crosses the *start* boundary if it was already open when
    the window began (entered before ``window_start``, still open at or
    after it). It crosses the *end* boundary if it was opened before the
    window ended but remained open at or after ``window_end``. Either
    condition marks it cross-boundary for this window -- a single round
    trip can be cross-boundary for two adjacent windows simultaneously
    (entered in the earlier one, exited in the later one).
    """
    crosses_start = round_trip.entry_timestamp < window_start <= round_trip.exit_timestamp
    crosses_end = round_trip.entry_timestamp < window_end <= round_trip.exit_timestamp
    return crosses_start or crosses_end


def eligible_round_trips(
    round_trips: list[RoundTrip], evaluation_start, evaluation_end
) -> list[RoundTrip]:
    """Round trips fully contained inside ``[evaluation_start, evaluation_end)``:
    entered at/after the evaluation start and exited before its end. These
    are the only round trips used for aggregate hit-rate/profit-factor/
    concentration statistics (see module docstring for why partially
    burn-in or still-open trades are excluded, not silently dropped)."""
    return [
        rt
        for rt in round_trips
        if rt.entry_timestamp >= evaluation_start and rt.exit_timestamp < evaluation_end
    ]


def burn_in_entered_round_trips(round_trips: list[RoundTrip], evaluation_start) -> list[RoundTrip]:
    """Round trips that entered before ``evaluation_start`` (during burn-in)
    but exited at/after it -- excluded from ``eligible_round_trips`` and
    reported separately so the exclusion is visible, not silent."""
    return [
        rt
        for rt in round_trips
        if rt.entry_timestamp < evaluation_start and rt.exit_timestamp >= evaluation_start
    ]


@dataclass(frozen=True)
class ConcentrationStats:
    """Descriptive realized-P&L concentration statistics -- not a
    trade-removal counterfactual re-simulation (removing a trade would
    change later position sizing under this engine's all-in sizing model,
    so no such re-simulation is attempted)."""

    largest_positive_fraction: float | None
    round_trips_for_50pct_of_positive_pnl: int | None
    total_positive_pnl: float
    note: str | None


def realized_pnl_concentration(round_trips: list[RoundTrip]) -> ConcentrationStats:
    """Concentration of realized P&L among ``round_trips``' positive
    outcomes: the largest single round trip's share of total positive P&L,
    and how many (ranked descending) are needed to reach 50% of it."""
    positive = sorted((rt.realized_pnl for rt in round_trips if rt.realized_pnl > 0), reverse=True)
    total_positive = sum(positive)

    if not positive or total_positive <= 0:
        return ConcentrationStats(
            largest_positive_fraction=None,
            round_trips_for_50pct_of_positive_pnl=None,
            total_positive_pnl=total_positive,
            note="no positive round trips",
        )

    largest_fraction = positive[0] / total_positive
    cumulative = 0.0
    count = 0
    for pnl in positive:
        cumulative += pnl
        count += 1
        if cumulative >= 0.5 * total_positive:
            break

    return ConcentrationStats(
        largest_positive_fraction=largest_fraction,
        round_trips_for_50pct_of_positive_pnl=count,
        total_positive_pnl=total_positive,
        note=None,
    )
