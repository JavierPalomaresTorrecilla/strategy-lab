"""Executed trade representation.

A ``Trade`` records a single full-position entry or exit executed by the
backtest engine. This milestone only ever fills orders completely — there
are no partial fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Trade:
    """A single executed trade.

    - ``timestamp``: the bar at which execution occurred (its open price).
    - ``side``: ``"buy"`` (entering the long position) or ``"sell"``
      (exiting it).
    - ``price``: the execution price actually used, i.e. the bar's open
      price *after* slippage has been applied.
    - ``quantity``: units of the instrument bought or sold.
    - ``commission``: commission charged on this trade, in cash terms.
    - ``cash_after``: cash balance immediately after this trade.
    - ``units_after``: units held immediately after this trade.
    """

    timestamp: Any
    side: Side
    price: float
    quantity: float
    commission: float
    cash_after: float
    units_after: float
