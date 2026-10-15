"""Offline safety tests for the narrow Coinbase public-data client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip(
    "coinbase",
    reason=(
        "requires the optional `crypto` extra (coinbase-advanced-py) — "
        "install with `pip install .[crypto]`"
    ),
)

import coinbase.jwt_generator

from strategy_lab.data.providers import coinbase_client
from strategy_lab.data.providers.coinbase_client import make_public_client


class _OfflineResponse:
    status_code = 200
    reason = "OK"
    text = ""
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_allowed_methods_are_unauthenticated_and_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    jwt_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def network_forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a Coinbase public-client test attempted network access")

    def jwt_forbidden(*args: Any, **kwargs: Any) -> None:
        jwt_calls.append((args, kwargs))
        raise AssertionError("an unauthenticated client attempted to create a JWT")

    def fake_request(
        method: str,
        url: str,
        *,
        params: dict[str, Any],
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _OfflineResponse:
        requests.append({"method": method, "url": url, "headers": headers})
        if url.endswith("/time"):
            return _OfflineResponse({"iso": "2024-01-01T00:00:00Z", "epochSeconds": 1})
        if url.endswith("/candles"):
            return _OfflineResponse({"candles": []})
        if url.endswith("/market/products/BTC-USD"):
            return _OfflineResponse({"product_id": "BTC-USD"})
        if url.endswith("/market/products"):
            return _OfflineResponse({"products": [], "num_products": 0})
        raise AssertionError(f"unexpected Coinbase endpoint: {url}")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(coinbase.jwt_generator, "build_rest_jwt", jwt_forbidden)

    public_client = make_public_client()
    public_client._client.session.request = fake_request

    calls = (
        public_client.get_unix_time,
        lambda: public_client.get_public_product("BTC-USD"),
        lambda: public_client.get_public_products(limit=1),
        lambda: public_client.get_public_candles(
            "BTC-USD", "1704067200", "1704070800", "ONE_MINUTE", limit=1
        ),
    )
    for call in calls:
        call()
        assert jwt_calls == []
        assert "Authorization" not in requests[-1]["headers"]

    assert len(requests) == 4


def test_public_client_exposes_only_the_allowlisted_methods() -> None:
    public_client = make_public_client()

    for name in (
        "get_accounts",
        "create_order",
        "market_order_buy",
        "cancel_orders",
        "get_portfolios",
        "get_fills",
    ):
        assert not hasattr(public_client, name)

    assert {name for name in dir(public_client) if not name.startswith("_")} == {
        "get_unix_time",
        "get_public_product",
        "get_public_products",
        "get_public_candles",
    }


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "COINBASE_API_KEY": "organizations/fake/apiKeys/fake",
            "COINBASE_API_SECRET": "fake-secret",
        }
    )
    return env


def test_factory_ignores_credentials_bound_before_sdk_import() -> None:
    child_script = """
import json
import socket

def network_forbidden(*args, **kwargs):
    raise AssertionError("network access is forbidden")

socket.socket.connect = network_forbidden

from strategy_lab.data.providers.coinbase_client import make_public_client

client = make_public_client()._client
print(json.dumps({
    "is_authenticated": client.is_authenticated,
    "api_key": getattr(client, "api_key", None),
    "api_secret": getattr(client, "api_secret", None),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "is_authenticated": False,
        "api_key": None,
        "api_secret": None,
    }


def test_sdk_default_construction_binds_import_time_credentials() -> None:
    child_script = """
import json

from coinbase.rest import RESTClient

print(json.dumps({"is_authenticated": RESTClient().is_authenticated}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"is_authenticated": True}


def test_factory_rejects_an_unexpectedly_authenticated_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAuthenticatedRESTClient:
        def __init__(self, **kwargs: Any) -> None:
            self.is_authenticated = True

    monkeypatch.setattr(coinbase_client, "_RESTClient", FakeAuthenticatedRESTClient)

    with pytest.raises(RuntimeError, match="unexpectedly authenticated"):
        make_public_client()
