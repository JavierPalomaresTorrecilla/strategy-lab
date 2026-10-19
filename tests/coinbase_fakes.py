"""Offline Coinbase SDK-shaped fakes shared by pipeline tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace


def candle(start: datetime, **overrides: str) -> SimpleNamespace:
    """Return one SDK-shaped daily candle object."""
    values = {
        "start": str(int(start.timestamp())),
        "low": "1.0",
        "high": "3.0",
        "open": "1.5",
        "close": "2.5",
        "volume": "4.0",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient:
    """Serve fixed per-product candles while recording SDK-style requests."""

    def __init__(
        self,
        now_utc: datetime,
        candles_by_product: Mapping[str, list[SimpleNamespace]],
        *,
        rate_limit_headers_by_call: Mapping[int, Mapping[str, str | None]] | None = None,
    ):
        assert now_utc.tzinfo is UTC
        self.now_utc = now_utc
        self.candles_by_product = candles_by_product
        self.rate_limit_headers_by_call = rate_limit_headers_by_call or {}
        self.calls: list[tuple[str, ...]] = []

    def get_unix_time(self) -> SimpleNamespace:
        self.calls.append(("get_unix_time",))
        epoch_seconds = str(int(self.now_utc.timestamp()))
        return SimpleNamespace(
            epoch_seconds=epoch_seconds,
            epoch_millis=str(int(self.now_utc.timestamp() * 1000)),
            iso=self.now_utc.isoformat(),
        )

    def get_public_candles(self, **request: str) -> SimpleNamespace:
        self.calls.append(
            (
                "get_public_candles",
                request["product_id"],
                request["start"],
                request["end"],
                request["granularity"],
            )
        )
        first = int(request["start"])
        last = int(request["end"])
        candles = [
            raw
            for raw in self.candles_by_product.get(request["product_id"], [])
            if first <= int(raw.start) <= last
        ]
        headers = self.rate_limit_headers_by_call.get(len(self.calls), {})
        return SimpleNamespace(candles=candles, **headers)


def utc_day(year: int, month: int, day: int) -> datetime:
    """Return a UTC midnight suitable for a Coinbase daily bucket."""
    return datetime(year, month, day, tzinfo=UTC)
