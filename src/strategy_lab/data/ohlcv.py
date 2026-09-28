"""Canonical OHLCV (open/high/low/close/volume) representation.

This module defines the single accepted shape for market data flowing into
the rest of the research pipeline (indicators, strategies, backtest engine).

Canonical assumptions
----------------------
- Data is a ``pandas.DataFrame`` with the columns ``timestamp``, ``open``,
  ``high``, ``low``, ``close``, ``volume`` (other columns are ignored).
- Exactly one row per bar/candle, for a single instrument.
- ``timestamp`` values are unique and sorted strictly ascending. This module
  does not know or assume a fixed bar size (e.g. daily vs hourly) — it only
  requires that timestamps be unique and monotonically increasing.
- Prices (``open``, ``high``, ``low``, ``close``) are strictly positive.
- ``volume`` is non-negative.
- ``high`` is the maximum and ``low`` is the minimum of the bar, so
  ``high >= max(open, close, low)`` and ``low <= min(open, close, high)``.
- ``open``, ``high``, ``low``, ``close``, ``volume`` are numeric, non-boolean,
  and finite (no ``NaN``, no ``+inf``/``-inf``).
- At least one row is present (an empty dataset is rejected).

Malformed data is rejected, not silently repaired: sorting, dropping
duplicates, coercing types, or clipping bad prices would hide problems in
upstream data sources that the researcher needs to know about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
_NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class OHLCVValidationError(ValueError):
    """Raised when input data violates the canonical OHLCV assumptions."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the canonical OHLCV assumptions.

    Returns a new DataFrame containing only ``REQUIRED_COLUMNS``, in that
    order, with a fresh ``RangeIndex``. The input ``df`` is not mutated.

    Type validation (numeric dtype, not boolean) runs first and fails fast:
    non-numeric columns can't be safely compared, so a type violation raises
    ``OHLCVValidationError`` immediately, before any later semantic check
    (NaN, infinity, sorting, sign, OHLC relationships, ...) runs. Once every
    column's type is confirmed valid, the remaining semantic checks all run
    and are reported together, describing every such violation found rather
    than raising only on the first one encountered.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise OHLCVValidationError(f"missing required columns: {missing}")

    data = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    # Type checks must happen first and raise immediately: the numeric
    # comparisons below (e.g. `<= 0`) raise a raw TypeError on non-numeric
    # dtypes such as strings, so those columns can't be safely compared
    # until we know they hold real numbers.
    type_errors: list[str] = []
    for col in _NUMERIC_COLUMNS:
        series = data[col]
        if pd.api.types.is_bool_dtype(series):
            type_errors.append(f"column '{col}' contains boolean values, not numeric prices/volume")
        elif not pd.api.types.is_numeric_dtype(series):
            type_errors.append(f"column '{col}' contains non-numeric values (dtype {series.dtype})")
    if type_errors:
        raise OHLCVValidationError("; ".join(type_errors))

    errors: list[str] = []

    if len(data) == 0:
        errors.append("dataset is empty (at least one row is required)")

    if data[list(REQUIRED_COLUMNS)].isna().to_numpy().any():
        errors.append("missing values (NaN) found in required columns")

    if np.isinf(data[list(_NUMERIC_COLUMNS)].to_numpy(dtype=float)).any():
        errors.append("infinite values (inf or -inf) found in required columns")

    if data["timestamp"].duplicated().any():
        errors.append("duplicated timestamps found")

    if not data["timestamp"].is_monotonic_increasing:
        errors.append("timestamps are not sorted in strictly ascending order")

    non_positive_prices = (data[list(_PRICE_COLUMNS)] <= 0).to_numpy().any()
    if non_positive_prices:
        errors.append("non-positive price(s) found in open/high/low/close")

    if (data["volume"] < 0).any():
        errors.append("negative volume found")

    bar_max = data[["open", "close", "low"]].max(axis=1)
    if (data["high"] < bar_max).any():
        errors.append("high is below max(open, close, low) for at least one bar")

    bar_min = data[["open", "close", "high"]].min(axis=1)
    if (data["low"] > bar_min).any():
        errors.append("low is above min(open, close, high) for at least one bar")

    if errors:
        raise OHLCVValidationError("; ".join(errors))

    return data.reset_index(drop=True)
