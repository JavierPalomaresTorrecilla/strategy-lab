"""Minimal market-data provider interface.

This is deliberately narrow: one method, one result type. It exists so the
research/backtest layers never need to know which provider produced a
dataset, not to support a general plugin framework. Adding a second provider
later means writing one more module that returns ``RawFetchResult`` from
``fetch`` — no registry or discovery mechanism is provided or needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RawFetchResult:
    """The raw, provider-shaped result of a single data fetch.

    - ``dataframe``: exactly what the provider returned, column names and
      all, unmodified. Provider-specific shape (e.g. ``Adj Close``,
      ``Dividends``) is preserved here; canonicalization happens later and
      separately (see ``strategy_lab.data.canonical``).
    - ``options_used``: the exact keyword arguments sent to the provider
      library for this request, so metadata never has to guess what was
      actually requested.
    """

    dataframe: pd.DataFrame
    provider: str
    provider_library_version: str
    symbol: str
    interval: str
    requested_start: str
    requested_end: str
    retrieved_at_utc: datetime
    options_used: dict[str, object]


class MarketDataProvider(Protocol):
    """Narrow interface a market-data provider must satisfy."""

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> RawFetchResult:
        """Fetch raw provider data for ``symbol`` at ``interval`` over
        ``[start, end)``. Must raise rather than silently coerce unexpected
        provider output into a valid-looking shape."""
        ...
