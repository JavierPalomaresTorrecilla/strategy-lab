"""Offline planning helpers for Coinbase daily-candle request windows.

This module deliberately contains no fetching, SDK integration, or candle parsing.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

INTERVAL_1D: Final = "1d"
DAY: Final = timedelta(days=1)
HISTORY_SEARCH_FLOOR: Final = datetime(2015, 1, 1, tzinfo=UTC)
# V1 deliberately leaves margin below the confirmed 350-per-request maximum: exact
# candle-count behavior for midnight-aligned start and end remains to be confirmed.
WINDOW_DAYS: Final = 300


def _validate_utc(value: datetime, name: str) -> None:
    """Raise ValueError unless *value* has an explicit zero UTC offset."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware datetime with UTC offset +00:00")


def _validate_utc_midnight(value: datetime, name: str) -> None:
    """Raise ValueError unless *value* is an exact UTC midnight."""
    _validate_utc(value, name)
    if any((value.hour, value.minute, value.second, value.microsecond)):
        raise ValueError(f"{name} must be exactly at UTC midnight")


def is_complete(bucket_start: datetime, now_utc: datetime, interval: str) -> bool:
    """Return whether a complete one-day bucket has elapsed by ``now_utc``."""
    if interval != INTERVAL_1D:
        raise ValueError(f"unsupported interval: {interval!r}")
    _validate_utc_midnight(bucket_start, "bucket_start")
    _validate_utc(now_utc, "now_utc")
    return bucket_start + DAY <= now_utc


def last_completed_bucket_start(now_utc: datetime) -> datetime:
    """Return the previous UTC midnight, the start of the latest completed day."""
    _validate_utc(now_utc, "now_utc")
    return datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=UTC) - DAY


@dataclass(frozen=True)
class CandleWindow:
    """An inclusive UTC-midnight daily-candle range.

    Confirmed empirical behavior indicates that candles with ``start >=
    request_start`` are returned and that the bucket containing ``request_end`` is
    also returned. Therefore, the midnight bounds below are expected to cover the
    intended inclusive day range exactly. A later task must verify that expectation
    with a per-window membership check; it is not proven here.
    """

    first_bucket_start: datetime
    last_bucket_start: datetime

    def __post_init__(self) -> None:
        _validate_utc_midnight(self.first_bucket_start, "first_bucket_start")
        _validate_utc_midnight(self.last_bucket_start, "last_bucket_start")
        if self.last_bucket_start < self.first_bucket_start:
            raise ValueError("last_bucket_start must not precede first_bucket_start")

    @property
    def day_count(self) -> int:
        """Number of inclusive daily buckets in this window."""
        return (self.last_bucket_start - self.first_bucket_start).days + 1

    @property
    def request_start(self) -> str:
        """Unix timestamp for the first intended bucket."""
        return str(int(self.first_bucket_start.timestamp()))

    @property
    def request_end(self) -> str:
        """Unix timestamp for the last intended bucket."""
        return str(int(self.last_bucket_start.timestamp()))


def plan_daily_windows(
    now_utc: datetime,
    floor: datetime = HISTORY_SEARCH_FLOOR,
    window_days: int = WINDOW_DAYS,
) -> tuple[CandleWindow, ...]:
    """Plan contiguous oldest-first daily windows from ``floor`` through yesterday."""
    _validate_utc_midnight(floor, "floor")
    if not 1 <= window_days <= 350:
        raise ValueError("window_days must be between 1 and 350 inclusive")

    last = last_completed_bucket_start(now_utc)
    if last < floor:
        raise ValueError("now_utc is too early to include a completed bucket at floor")

    windows: list[CandleWindow] = []
    first = floor
    while first <= last:
        window_last = min(first + DAY * (window_days - 1), last)
        windows.append(CandleWindow(first, window_last))
        first = window_last + DAY
    return tuple(windows)
