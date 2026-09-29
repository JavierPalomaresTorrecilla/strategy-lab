"""yfinance market-data provider.

This is the initial DEVELOPMENT data provider (not authoritative/production
grade). This module is the *only* place in the codebase allowed to import
``yfinance`` — everything downstream operates on the provider-agnostic
``RawFetchResult``.

Every option that materially affects data semantics is passed explicitly on
every call; none are left to yfinance's defaults (defaults have changed
across yfinance versions before, e.g. ``auto_adjust``).

``symbol``, ``interval``, ``start``, ``end`` are genuine parameters of
``fetch`` — this module does not hard-code SPY or any particular date range.
The fixed Milestone 2 dataset (SPY, 1d, 2010-01-01 to 2026-01-01 exclusive)
belongs in the human-run acquisition script (``scripts/fetch_dataset.py``),
not here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import yfinance

from strategy_lab.data.providers.base import RawFetchResult


class UnexpectedProviderShapeError(RuntimeError):
    """Raised when yfinance returns a DataFrame shape this provider does not
    know how to handle safely (e.g. a MultiIndex when a flat single-ticker
    frame was explicitly requested). Never silently coerced."""


def _fetch_kwargs(symbol: str, interval: str, start: str, end: str) -> dict[str, object]:
    """The exact, fully-explicit kwargs sent to ``yfinance.download``.

    Every value here materially affects data semantics and is therefore
    never left to a library default.
    """
    return {
        "tickers": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "auto_adjust": False,
        "repair": False,
        "actions": True,
        "keepna": True,
        "ignore_tz": True,
        "multi_level_index": False,
        "group_by": "column",
        "prepost": False,
        "progress": False,
        "threads": False,
    }


class YFinanceProvider:
    """``MarketDataProvider`` backed by the ``yfinance`` library."""

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> RawFetchResult:
        options_used = _fetch_kwargs(symbol, interval, start, end)
        dataframe = yfinance.download(**options_used)

        if dataframe is None:
            raise UnexpectedProviderShapeError(
                f"yfinance.download returned None for symbol={symbol!r} "
                f"interval={interval!r} start={start!r} end={end!r}"
            )
        if isinstance(dataframe.columns, pd.MultiIndex):
            raise UnexpectedProviderShapeError(
                "yfinance returned a MultiIndex-column DataFrame despite "
                "multi_level_index=False and group_by='column'; refusing to "
                "silently flatten or coerce it. Columns: "
                f"{dataframe.columns.tolist()!r}"
            )

        return RawFetchResult(
            dataframe=dataframe,
            provider="yfinance",
            provider_library_version=yfinance.__version__,
            symbol=symbol,
            interval=interval,
            requested_start=start,
            requested_end=end,
            retrieved_at_utc=datetime.now(UTC),
            options_used=options_used,
        )
