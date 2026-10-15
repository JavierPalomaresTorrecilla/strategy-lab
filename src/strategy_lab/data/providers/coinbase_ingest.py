"""Throttled, capped, in-memory Coinbase daily-candle ingestion.

``retrieval.started_at_utc`` is Coinbase server time.  A future replay consumer
uses that frozen value as ``now_utc`` for ``is_complete`` and to re-derive the
same planned windows, so no later live time request is needed.  In contrast,
``retrieval.completed_at_utc`` is a local wall-clock bookkeeping timestamp only.
The local throttle below is deliberately conservative and self-imposed; it makes
no claim about Coinbase's actual public-endpoint rate limits, which are unknown.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from requests.exceptions import HTTPError

from strategy_lab.data.providers.coinbase_candles import CandleWindow, plan_daily_windows
from strategy_lab.data.providers.coinbase_raw import (
    ALLOWED_PRODUCTS,
    CANDLE_KEYS,
    INTERVAL,
    PROVIDER,
    RATE_LIMIT_HEADER_KEYS,
    RAW_SCHEMA_VERSION,
    validate_raw_artifact,
)

REQUEST_CAP: Final = 40
MIN_REQUEST_INTERVAL_S: Final = 1.0
SDK_HEADER_ATTRS: dict[str, str] = {
    "x_ratelimit_limit": "x-ratelimit-limit",
    "x_ratelimit_remaining": "x-ratelimit-remaining",
    "x_ratelimit_reset": "x-ratelimit-reset",
}

assert set(SDK_HEADER_ATTRS.values()) == RATE_LIMIT_HEADER_KEYS


class IngestAbort(RuntimeError):
    """Base class for runtime failures that abort an ingestion run."""


class RequestCapExceeded(IngestAbort):
    """Raised before a request would exceed this run's fixed request cap."""


class ProviderHTTPError(IngestAbort):
    """A Coinbase HTTP failure, retaining bounded diagnostic response data."""

    status_code: int
    headers: dict[str, str]
    body: str

    def __init__(self, status_code: int, headers: dict[str, str], body: str) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body[:2000]
        super().__init__(f"Coinbase returned HTTP {status_code}")


class UnexpectedResponseShape(IngestAbort):
    """Raised when an SDK response cannot be represented by the closed raw schema."""


def _response_vars(response: object, required: set[str], context: str) -> dict[str, Any]:
    """Return response attributes after rejecting unknown or missing shape fields."""
    try:
        response_vars = vars(response)
    except TypeError as exc:
        raise UnexpectedResponseShape(f"{context}: response has no attribute dictionary") from exc

    actual_keys = set(response_vars)
    allowed_keys = required | set(SDK_HEADER_ATTRS)
    missing = required - actual_keys
    unexpected = actual_keys - allowed_keys
    if missing:
        raise UnexpectedResponseShape(f"{context}: missing response key {sorted(missing)[0]!r}")
    if unexpected:
        raise UnexpectedResponseShape(
            f"{context}: unexpected response key {sorted(unexpected)[0]!r}"
        )
    return response_vars


def server_now_utc(client: Any) -> datetime:
    """Read and validate Coinbase's server time as an aware UTC datetime."""
    response_vars = _response_vars(
        client.get_unix_time(), {"epoch_seconds", "epoch_millis", "iso"}, "unix-time response"
    )
    epoch_seconds = response_vars["epoch_seconds"]
    if not (
        isinstance(epoch_seconds, str)
        and epoch_seconds.isascii()
        and epoch_seconds.isdigit()
    ):
        raise UnexpectedResponseShape(
            "unix-time response: epoch_seconds must be an ASCII digit string"
        )
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise UnexpectedResponseShape(
            "unix-time response: epoch_seconds cannot form a UTC datetime"
        ) from exc


def page_record_from_response(product: str, window: CandleWindow, response: object) -> dict:
    """Convert one validated SDK candle response into the Task-5 raw page shape."""
    response_vars = _response_vars(response, {"candles"}, "candle response")

    rate_limit_headers: dict[str, str] = {}
    for sdk_attr, raw_key in SDK_HEADER_ATTRS.items():
        value = response_vars.get(sdk_attr)
        if value is None:
            continue
        if not isinstance(value, str):
            raise UnexpectedResponseShape(
                f"candle response: header {sdk_attr!r} must be str or None"
            )
        rate_limit_headers[raw_key] = value

    response_candles = response_vars["candles"]
    if not isinstance(response_candles, list):
        raise UnexpectedResponseShape("candle response: candles must be a list")
    candles: list[dict[str, str]] = []
    for candle_index, candle in enumerate(response_candles):
        try:
            candle_dict = dict(vars(candle))
        except TypeError as exc:
            raise UnexpectedResponseShape(
                f"candle {candle_index}: has no attribute dictionary"
            ) from exc
        candle_keys = set(candle_dict)
        if candle_keys != CANDLE_KEYS:
            missing = sorted(CANDLE_KEYS - candle_keys)
            unexpected = sorted(candle_keys - CANDLE_KEYS)
            if missing:
                raise UnexpectedResponseShape(f"candle {candle_index}: missing key {missing[0]!r}")
            raise UnexpectedResponseShape(
                f"candle {candle_index}: unexpected key {unexpected[0]!r}"
            )
        for key, value in candle_dict.items():
            if not isinstance(value, str):
                raise UnexpectedResponseShape(
                    f"candle {candle_index}: key {key!r} must be str, got {type(value).__name__}"
                )
        candles.append(candle_dict)

    return {
        "request": {
            "product_id": product,
            "start": window.request_start,
            "end": window.request_end,
            "granularity": "ONE_DAY",
        },
        "rate_limit_headers": rate_limit_headers or None,
        "candles": candles,
    }


class _SequentialThrottledCaller:
    """Count and serialize requests with the project's local minimum spacing."""

    def __init__(
        self,
        monotonic_fn: Callable[[], float],
        sleep_fn: Callable[[float], None],
    ) -> None:
        self._monotonic_fn = monotonic_fn
        self._sleep_fn = sleep_fn
        self._previous_request_at: float | None = None
        self.request_count = 0

    def call(self, request_fn: Callable[[], Any]) -> Any:
        """Make exactly one request after cap and local-throttle checks."""
        if self.request_count >= REQUEST_CAP:
            raise RequestCapExceeded(f"request cap of {REQUEST_CAP} would be exceeded")

        request_at = self._monotonic_fn()
        if self._previous_request_at is not None:
            elapsed = request_at - self._previous_request_at
            if elapsed < MIN_REQUEST_INTERVAL_S:
                self._sleep_fn(MIN_REQUEST_INTERVAL_S - elapsed)
                request_at = self._monotonic_fn()

        self._previous_request_at = request_at
        self.request_count += 1
        return request_fn()


def _validate_products(products: Sequence[str]) -> None:
    if not products:
        raise ValueError("products must be non-empty")
    seen: set[str] = set()
    for product in products:
        if product not in ALLOWED_PRODUCTS:
            raise ValueError(f"unsupported Coinbase product: {product!r}")
        if product in seen:
            raise ValueError(f"duplicate Coinbase product: {product!r}")
        seen.add(product)


def _provider_http_error(error: HTTPError) -> ProviderHTTPError:
    response = error.response
    return ProviderHTTPError(
        status_code=response.status_code,
        headers=dict(response.headers),
        body=response.text[:2000],
    )


def fetch_run(
    client: Any,
    products: Sequence[str],
    *,
    local_clock_fn: Callable[[], datetime],
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    python_version: str,
    sdk_version: str,
) -> dict[str, dict]:
    """Fetch, validate, and return all product artifacts without writing to disk.

    The server-derived start time is intentionally frozen for reproducible future
    replay.  The single final local-clock value is human-readable duration
    bookkeeping and must not be used by replay logic.
    """
    _validate_products(products)
    caller = _SequentialThrottledCaller(monotonic_fn, sleep_fn)
    now_utc = caller.call(lambda: server_now_utc(client))
    windows = plan_daily_windows(now_utc)

    required_requests = 1 + len(products) * len(windows)
    if required_requests > REQUEST_CAP:
        raise RequestCapExceeded(
            f"request cap of {REQUEST_CAP} is insufficient for {required_requests} requests"
        )

    pages_by_product: dict[str, list[dict]] = {product: [] for product in products}
    for product in products:
        for window in windows:
            try:
                response = caller.call(
                    lambda product=product, window=window: client.get_public_candles(
                        product_id=product,
                        start=window.request_start,
                        end=window.request_end,
                        granularity="ONE_DAY",
                    )
                )
            except HTTPError as exc:
                raise _provider_http_error(exc) from exc
            pages_by_product[product].append(page_record_from_response(product, window, response))

    completed_at_utc = local_clock_fn().isoformat()
    started_at_utc = now_utc.isoformat()
    artifacts: dict[str, dict] = {}
    for product in products:
        artifact = {
            "schema_version": RAW_SCHEMA_VERSION,
            "provider": PROVIDER,
            "product": product,
            "interval": INTERVAL,
            "retrieval": {
                "python_version": python_version,
                "sdk_version": sdk_version,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "pages": pages_by_product[product],
        }
        validate_raw_artifact(artifact)
        artifacts[product] = artifact
    return artifacts
