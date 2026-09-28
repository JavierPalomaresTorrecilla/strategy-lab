"""Exponential moving average (EMA).

Semantics
---------
Uses the standard recursive EMA definition with smoothing factor
``alpha = 2 / (period + 1)``:

    y[0] = x[0]
    y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]

This is computed via ``pandas.Series.ewm(span=period, adjust=False)``.
``adjust=False`` is used deliberately: it produces the standard recursive
(streaming-compatible) EMA definition above, rather than pandas' weighted
average of all prior observations (``adjust=True``), which is a different
and less commonly intended definition for trading indicators.

The first ``period - 1`` output values are ``NaN`` (via ``min_periods``)
rather than an EMA computed from a partial, shorter-than-requested window —
a partial-window EMA would silently understate the intended smoothing.

The output preserves the index of the input series exactly.
"""

from __future__ import annotations

import pandas as pd


def ema(prices: pd.Series, period: int) -> pd.Series:
    """Compute the EMA of ``prices`` over ``period`` bars.

    ``period`` must be a positive integer. Raises ``ValueError`` otherwise.
    """
    if isinstance(period, bool) or not isinstance(period, int):
        raise ValueError(f"period must be an integer, got {period!r}")
    if period < 1:
        raise ValueError(f"period must be a positive integer, got {period}")

    return prices.ewm(span=period, adjust=False, min_periods=period).mean()
