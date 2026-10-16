"""Pure decoding and canonicalization helpers for Coinbase daily candles.

This module accepts provider-decoded mapping data only.  Fetching, SDK imports,
and persistence deliberately belong elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

import pandas as pd

from strategy_lab.data.ohlcv import validate_ohlcv
from strategy_lab.data.providers.coinbase_candles import CandleWindow, is_complete

_RAW_CANDLE_KEYS = frozenset({"start", "low", "high", "open", "close", "volume"})
_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume")
_DAY = timedelta(days=1)


class CoinbaseDecodeError(ValueError):
    """Base class for malformed or unusable Coinbase candle history."""


class MalformedCandleError(CoinbaseDecodeError):
    """Raised when one raw Coinbase candle cannot be decoded exactly."""


class CandleOutsideWindowError(CoinbaseDecodeError):
    """Raised when a returned candle is outside its requested window."""


class DuplicateCandleError(CoinbaseDecodeError):
    """Raised when a candle bucket is returned more than once."""


class IncompleteCandleError(CoinbaseDecodeError):
    """Raised when a returned candle has not completed yet."""


class EmptyHistoryError(CoinbaseDecodeError):
    """Raised when every supplied page contains zero candles."""


@dataclass(frozen=True)
class CoinbaseCandle:
    """One exact Coinbase daily candle, decoded without numeric coercion."""

    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class CoinbaseDecodeResult:
    """Decoded candle history and non-fatal UTC daily-gap metadata."""

    product: str
    candles: tuple[CoinbaseCandle, ...]
    earliest_bucket_start: datetime
    latest_bucket_start: datetime
    candle_count: int
    missing_days_utc: tuple[str, ...]
    missing_day_count: int


def parse_candle(raw: Mapping[str, object]) -> CoinbaseCandle:
    """Decode one exact provider-shaped candle mapping.

    The Coinbase SDK object's attributes are expected to have been converted to
    this six-key mapping by the caller.  No numeric rounding or coercion occurs.
    """
    raw_keys = set(raw)
    missing = sorted(_RAW_CANDLE_KEYS - raw_keys)
    unexpected = sorted(raw_keys - _RAW_CANDLE_KEYS, key=repr)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing key(s) {missing}")
        if unexpected:
            details.append(f"unexpected key(s) {unexpected}")
        raise MalformedCandleError("malformed candle mapping: " + "; ".join(details))

    for field in _RAW_CANDLE_KEYS:
        value = raw[field]
        if not isinstance(value, str):
            raise MalformedCandleError(
                f"malformed candle field {field!r}: expected str, got {type(value).__name__}"
            )

    start = raw["start"]
    assert isinstance(start, str)  # Narrowed above; retained for static type checkers.
    if not start or not start.isascii() or not start.isdigit():
        raise MalformedCandleError(
            f"malformed candle field 'start': expected ASCII epoch-second digits, got {start!r}"
        )
    try:
        bucket_start = datetime.fromtimestamp(int(start), tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise MalformedCandleError(
            "malformed candle field 'start': "
            f"epoch-second value {start!r} cannot form a UTC datetime"
        ) from error
    if any(
        (
            bucket_start.hour,
            bucket_start.minute,
            bucket_start.second,
            bucket_start.microsecond,
        )
    ):
        raise MalformedCandleError(
            "malformed candle field 'start': "
            f"value {start!r} resolves to non-midnight UTC bucket {bucket_start.isoformat()}"
        )

    decimals: dict[str, Decimal] = {}
    for field in _NUMERIC_FIELDS:
        value = raw[field]
        assert isinstance(value, str)  # Narrowed above; retained for static type checkers.
        try:
            decimal_value = Decimal(value)
        except InvalidOperation as error:
            raise MalformedCandleError(
                f"malformed candle field {field!r}: invalid Decimal string {value!r}"
            ) from error
        if not decimal_value.is_finite():
            raise MalformedCandleError(
                f"malformed candle field {field!r}: non-finite Decimal value {value!r}"
            )
        decimals[field] = decimal_value

    return CoinbaseCandle(bucket_start=bucket_start, **decimals)


def decode_pages(
    product: str,
    pages: Sequence[tuple[CandleWindow, Sequence[Mapping[str, object]]]],
    now_utc: datetime,
) -> CoinbaseDecodeResult:
    """Decode daily pages, rejecting invalid coverage while retaining gap metadata."""
    parsed: list[CoinbaseCandle] = []
    seen_bucket_starts: set[datetime] = set()

    for window, raw_candles in pages:
        for raw in raw_candles:
            try:
                candle = parse_candle(raw)
            except MalformedCandleError as exc:
                raise MalformedCandleError(f"{product}: {exc}") from exc
            bucket = candle.bucket_start
            if not window.first_bucket_start <= bucket <= window.last_bucket_start:
                raise CandleOutsideWindowError(
                    f"product {product!r} returned bucket {bucket.isoformat()} outside requested "
                    f"window {window.first_bucket_start.isoformat()} through "
                    f"{window.last_bucket_start.isoformat()}"
                )
            if bucket in seen_bucket_starts:
                raise DuplicateCandleError(
                    f"product {product!r} returned duplicate bucket {bucket.isoformat()}"
                )
            if not is_complete(bucket, now_utc, "1d"):
                raise IncompleteCandleError(
                    f"product {product!r} returned incomplete bucket {bucket.isoformat()}"
                )
            seen_bucket_starts.add(bucket)
            parsed.append(candle)

    if not parsed:
        raise EmptyHistoryError(
            f"product {product!r} returned zero candles across all supplied pages"
        )

    candles = tuple(sorted(parsed, key=lambda candle: candle.bucket_start))
    earliest_bucket_start = candles[0].bucket_start
    latest_bucket_start = candles[-1].bucket_start
    last_planned_bucket_start = max(window.last_bucket_start for window, _ in pages)
    returned_buckets = {candle.bucket_start for candle in candles}
    missing_days: list[str] = []
    bucket = earliest_bucket_start
    while bucket <= last_planned_bucket_start:
        if bucket not in returned_buckets:
            missing_days.append(bucket.date().isoformat())
        bucket += _DAY

    return CoinbaseDecodeResult(
        product=product,
        candles=candles,
        earliest_bucket_start=earliest_bucket_start,
        latest_bucket_start=latest_bucket_start,
        candle_count=len(candles),
        missing_days_utc=tuple(missing_days),
        missing_day_count=len(missing_days),
    )


def to_canonical_frame(result: CoinbaseDecodeResult) -> pd.DataFrame:
    """Convert decoded candles to the project's naive-UTC canonical OHLCV frame."""
    timestamps = pd.DatetimeIndex([candle.bucket_start for candle in result.candles])
    naive_utc_timestamps = timestamps.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]")
    frame = pd.DataFrame(
        {
            "timestamp": naive_utc_timestamps,
            "open": [float(candle.open) for candle in result.candles],
            "high": [float(candle.high) for candle in result.candles],
            "low": [float(candle.low) for candle in result.candles],
            "close": [float(candle.close) for candle in result.candles],
            "volume": [float(candle.volume) for candle in result.candles],
        }
    )
    return validate_ohlcv(frame)
