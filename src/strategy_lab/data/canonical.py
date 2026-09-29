"""Conversion from raw provider-shaped data to the canonical OHLCV schema.

This is the boundary between provider-specific representations and the
provider-agnostic rest of the codebase (indicators, strategies, backtest
engine). It is a pure function: no I/O, no filesystem access.

Daily timestamp semantics
--------------------------
For ``interval="1d"`` data, a canonical ``timestamp`` value means
``exchange_session_date_naive``: the calendar date of the exchange trading
session, timezone-naive, with no intraday time-of-day meaning implied (the
stored value carries a midnight time component only because that's how a
date is represented as a ``pandas.Timestamp``).

Because the raw fetch (``strategy_lab.data.providers.yfinance_provider``)
uses ``ignore_tz=True``, the raw DataFrame's index already arrives
timezone-naive. This module therefore does **not** perform any
``tz_localize``/``tz_convert`` call — doing so would be both unnecessary and
would silently assume a specific timezone offset that was never actually
present in the data.

This convention applies only to daily bars. Intraday interval timestamp
semantics (what a naive vs. tz-aware sub-daily timestamp should mean, which
exchange session boundary applies) are out of scope here and must be
designed separately before any intraday interval is supported.

Malformed data is never repaired here: if the raw frame can't satisfy
``validate_ohlcv``, this raises ``OHLCVValidationError`` and stops.
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.data.ohlcv import validate_ohlcv

_PROVIDER_TO_CANONICAL = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class CanonicalConversionError(ValueError):
    """Raised when raw provider data cannot be mapped onto the canonical
    schema at all (e.g. missing a required provider column)."""


def to_canonical_ohlcv(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw provider-shaped DataFrame (yfinance OHLCV shape, with a
    ``DatetimeIndex`` and capitalized ``Open/High/Low/Close/Volume`` columns)
    into the canonical OHLCV schema.

    Extra provider columns (``Adj Close``, ``Dividends``, ``Stock Splits``,
    ``Capital Gains``, ...) are dropped here — they remain permanently
    available in the raw snapshot this was derived from, so dropping them
    here is not destructive.

    Raises ``CanonicalConversionError`` if the raw frame's index is not a
    plain ``DatetimeIndex`` (e.g. a row-level ``MultiIndex``, which would
    otherwise make ``reset_index`` produce an ambiguous, unintended shape)
    or if a required provider column is missing, or ``OHLCVValidationError``
    (propagated from ``validate_ohlcv``) if the mapped data violates the
    canonical schema. None of these cases is silently repaired.
    """
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raise CanonicalConversionError(
            "raw data must be indexed by a plain pandas.DatetimeIndex, got "
            f"{type(raw_df.index).__name__}"
        )

    missing = [c for c in _PROVIDER_TO_CANONICAL if c not in raw_df.columns]
    if missing:
        raise CanonicalConversionError(
            f"raw data is missing required provider column(s): {missing}"
        )

    canonical = raw_df.loc[:, list(_PROVIDER_TO_CANONICAL)].rename(columns=_PROVIDER_TO_CANONICAL)
    canonical = canonical.reset_index(drop=False)
    canonical = canonical.rename(columns={canonical.columns[0]: "timestamp"})

    return validate_ohlcv(canonical)
