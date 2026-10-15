"""Fully offline tests for Coinbase's throttled raw-ingestion loop."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from requests.exceptions import HTTPError

from strategy_lab.data.providers import coinbase_ingest
from strategy_lab.data.providers.coinbase_candles import CandleWindow, plan_daily_windows
from strategy_lab.data.providers.coinbase_canonical import decode_pages
from strategy_lab.data.providers.coinbase_ingest import (
    ProviderHTTPError,
    RequestCapExceeded,
    UnexpectedResponseShape,
    fetch_run,
    page_record_from_response,
    server_now_utc,
)
from strategy_lab.data.providers.coinbase_raw import (
    CANDLE_KEYS,
    RATE_LIMIT_HEADER_KEYS,
    validate_raw_artifact,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if this entirely fake-response suite opens a socket."""

    def forbidden_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("coinbase ingestion tests must not open sockets")

    monkeypatch.setattr(socket, "socket", forbidden_socket)


def _candle(start: str = "1420070400", **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "start": start,
        "low": "1.0",
        "high": "3.0",
        "open": "1.5",
        "close": "2.5",
        "volume": "4.0",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _candle_response(
    candles: list[object] | None = None, **headers: object
) -> SimpleNamespace:
    return SimpleNamespace(candles=[] if candles is None else candles, **headers)


class FakeClient:
    def __init__(
        self,
        now_utc: datetime,
        *,
        time_response: object | None = None,
        candle_response_fn: Callable[[int, dict[str, str]], object] | None = None,
        error_at_candle_call: int | None = None,
        error_status_code: int = 429,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._now_utc = now_utc
        self._time_response = time_response
        self._candle_response_fn = candle_response_fn or (
            lambda _index, _request: _candle_response()
        )
        self._error_at_candle_call = error_at_candle_call
        self._error_status_code = error_status_code
        self._candle_call_count = 0

    def get_unix_time(self) -> object:
        self.calls.append(("get_unix_time",))
        if self._time_response is not None:
            return self._time_response
        return SimpleNamespace(
            epoch_seconds=str(int(self._now_utc.timestamp())),
            epoch_millis=str(int(self._now_utc.timestamp() * 1000)),
            iso=self._now_utc.isoformat(),
        )

    def get_public_candles(self, **request: str) -> object:
        self._candle_call_count += 1
        self.calls.append(
            (
                "get_public_candles",
                request["product_id"],
                request["start"],
                request["end"],
                request["granularity"],
            )
        )
        if self._candle_call_count == self._error_at_candle_call:
            response = SimpleNamespace(
                status_code=self._error_status_code,
                headers={"Retry-After": "3", "X-Trace": "fake"},
                text="x" * 2500,
            )
            raise HTTPError(response=response)
        return self._candle_response_fn(self._candle_call_count, request)


class SteppingClock:
    def __init__(self, step: float) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


NOW_UTC = datetime(2026, 10, 15, 14, 13, 27, tzinfo=UTC)
LOCAL_COMPLETED = datetime(2026, 10, 15, 14, 14, tzinfo=UTC)


def _fetch(
    client: FakeClient,
    products: list[str],
    *,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, dict]:
    return fetch_run(
        client,
        products,
        local_clock_fn=lambda: LOCAL_COMPLETED,
        monotonic_fn=monotonic_fn or SteppingClock(2.0),
        sleep_fn=sleep_fn or (lambda _seconds: None),
        python_version="3.11.9",
        sdk_version="1.7.0",
    )


def test_happy_path_uses_server_time_shared_oldest_first_windows_and_raw_shape() -> None:
    def candle_response(index: int, _request: dict[str, str]) -> object:
        if index == 1:
            return _candle_response(
                x_ratelimit_limit="10",
                x_ratelimit_remaining=None,
                x_ratelimit_reset=None,
            )
        return _candle_response()

    client = FakeClient(NOW_UTC, candle_response_fn=candle_response)
    artifacts = _fetch(client, ["ADA-EUR", "DOT-EUR"])

    windows = plan_daily_windows(NOW_UTC)
    assert len(windows) == 15
    assert len(client.calls) == 31 <= coinbase_ingest.REQUEST_CAP
    assert client.calls[0] == ("get_unix_time",)
    candle_calls = client.calls[1:]
    assert [call[1] for call in candle_calls[:15]] == ["ADA-EUR"] * 15
    assert [call[1] for call in candle_calls[15:]] == ["DOT-EUR"] * 15
    for call, window in zip(candle_calls[:15], windows, strict=True):
        assert call[2:5] == (window.request_start, window.request_end, "ONE_DAY")
    for call in candle_calls:
        assert call[2].isascii() and call[2].isdigit()
        assert call[3].isascii() and call[3].isdigit()
        assert call[4] == "ONE_DAY"

    for artifact in artifacts.values():
        validate_raw_artifact(artifact)
        assert artifact["retrieval"]["started_at_utc"] == NOW_UTC.isoformat()
        assert artifact["retrieval"]["completed_at_utc"] == LOCAL_COMPLETED.isoformat()
    assert artifacts["ADA-EUR"]["pages"][0]["rate_limit_headers"] == {
        "x-ratelimit-limit": "10"
    }
    assert artifacts["ADA-EUR"]["pages"][1]["rate_limit_headers"] is None


def test_throttle_sleeps_for_short_gaps_and_not_for_a_long_gap() -> None:
    class SimulatedClock:
        def __init__(self) -> None:
            self.base = -0.2
            self.slept = 0.0

        def __call__(self) -> float:
            self.base += 0.2
            return self.base + self.slept

        def sleep(self, seconds: float) -> None:
            self.slept += seconds
            sleeps.append(seconds)

    sleeps: list[float] = []
    clock = SimulatedClock()
    caller = coinbase_ingest._SequentialThrottledCaller(clock, clock.sleep)
    request_times: list[float] = []
    for _ in range(3):
        caller.call(lambda: request_times.append(clock.base + clock.slept))
    assert sleeps == pytest.approx([0.8, 0.8])
    assert all(
        later - earlier >= coinbase_ingest.MIN_REQUEST_INTERVAL_S
        for earlier, later in zip(request_times, request_times[1:], strict=False)
    )

    long_gap_values = iter((0.0, 1.2))
    no_sleep: list[float] = []
    long_gap_caller = coinbase_ingest._SequentialThrottledCaller(
        lambda: next(long_gap_values), no_sleep.append
    )
    long_gap_caller.call(lambda: None)
    long_gap_caller.call(lambda: None)
    assert no_sleep == []


def test_cap_precheck_stops_after_the_one_server_time_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coinbase_ingest, "REQUEST_CAP", 5)
    client = FakeClient(NOW_UTC)

    with pytest.raises(RequestCapExceeded, match="insufficient"):
        _fetch(client, ["ADA-EUR"])

    assert client.calls == [("get_unix_time",)]


@pytest.mark.parametrize("status_code", [429, 500])
def test_http_errors_abort_without_retry_and_retain_response_details(status_code: int) -> None:
    client = FakeClient(NOW_UTC, error_at_candle_call=7, error_status_code=status_code)

    with pytest.raises(ProviderHTTPError) as captured:
        _fetch(client, ["ADA-EUR", "DOT-EUR"])

    error = captured.value
    assert error.status_code == status_code
    assert error.headers == {"Retry-After": "3", "X-Trace": "fake"}
    assert error.body == "x" * 2000
    assert len(client.calls) == 8
    assert client.calls[-1][0] == "get_public_candles"


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (_candle_response(extra="unexpected"), "unexpected response key"),
        (
            _candle_response([SimpleNamespace(**vars(_candle()), extra="unexpected")]),
            "unexpected key",
        ),
        (
            _candle_response(
                [SimpleNamespace(start="1", low="1", high="1", open="1", close="1")]
            ),
            "missing key",
        ),
        (_candle_response([_candle(volume=1)]), "must be str"),
        (_candle_response(x_ratelimit_limit=1), "must be str or None"),
    ],
)
def test_candle_response_shape_errors(response: object, match: str) -> None:
    with pytest.raises(UnexpectedResponseShape, match=match):
        page_record_from_response("ADA-EUR", _window(), response)


@pytest.mark.parametrize(
    "epoch_seconds",
    ["12a", "-12", 12],
)
def test_server_time_rejects_non_ascii_digit_epoch_seconds(epoch_seconds: object) -> None:
    client = FakeClient(
        NOW_UTC,
        time_response=SimpleNamespace(epoch_seconds=epoch_seconds, epoch_millis="1", iso="ignored"),
    )
    with pytest.raises(UnexpectedResponseShape, match="epoch_seconds"):
        server_now_utc(client)


@pytest.mark.parametrize(
    "products",
    [[], ["BTC-USD"], ["ADA-EUR", "ADA-EUR"]],
)
def test_product_validation_happens_before_any_request(products: list[str]) -> None:
    client = FakeClient(NOW_UTC)
    with pytest.raises(ValueError):
        _fetch(client, products)
    assert client.calls == []


def _window() -> CandleWindow:
    return CandleWindow(
        datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)
    )


def test_header_mapping_handles_none_partial_absence_and_unknown_attributes() -> None:
    window = _window()
    all_none = page_record_from_response(
        "ADA-EUR",
        window,
        _candle_response(
            x_ratelimit_limit=None,
            x_ratelimit_remaining=None,
            x_ratelimit_reset=None,
        ),
    )
    partial = page_record_from_response(
        "ADA-EUR",
        window,
        _candle_response(x_ratelimit_limit="10", x_ratelimit_remaining=None),
    )
    absent = page_record_from_response("ADA-EUR", window, _candle_response())

    assert all_none["rate_limit_headers"] is None
    assert partial["rate_limit_headers"] == {"x-ratelimit-limit": "10"}
    assert absent["rate_limit_headers"] is None
    with pytest.raises(UnexpectedResponseShape, match="unexpected response key"):
        page_record_from_response(
            "ADA-EUR", window, _candle_response(x_ratelimit_unknown="10")
        )


def test_header_mapping_invariant_matches_raw_schema_keys() -> None:
    assert set(coinbase_ingest.SDK_HEADER_ATTRS.values()) == RATE_LIMIT_HEADER_KEYS


def test_page_shape_decodes_with_the_existing_canonical_decoder() -> None:
    windows = plan_daily_windows(NOW_UTC)
    first_window = windows[0]

    def candle_response(index: int, _request: dict[str, str]) -> object:
        if index == 1:
            return _candle_response([_candle(start=first_window.request_start)])
        return _candle_response()

    artifact = _fetch(
        FakeClient(NOW_UTC, candle_response_fn=candle_response), ["ADA-EUR"]
    )["ADA-EUR"]
    result = decode_pages(
        "ADA-EUR",
        [(first_window, artifact["pages"][0]["candles"])],
        NOW_UTC,
    )
    assert result.candle_count == 1
    assert result.earliest_bucket_start == first_window.first_bucket_start


def test_ingest_import_does_not_import_coinbase_sdk() -> None:
    child_script = """
import sys
import strategy_lab.data.providers.coinbase_ingest
raise SystemExit(1 if "coinbase" in sys.modules else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True,
        check=False,
        env=os.environ.copy(),
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_fake_candle_shape_matches_the_confirmed_six_provider_fields() -> None:
    assert set(vars(_candle())) == CANDLE_KEYS
