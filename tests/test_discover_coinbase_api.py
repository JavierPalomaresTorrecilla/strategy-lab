"""Offline tests for the human-run Coinbase discovery utility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_discovery_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "discover_coinbase_api.py"
    spec = importlib.util.spec_from_file_location("discover_coinbase_api_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, **attributes: Any) -> None:
        self.__dict__.update(attributes)


class FakePublicClient:
    """In-process double; it never imports or invokes Coinbase networking code."""

    def __init__(self, fail_product: str | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail_product = fail_product

    def get_unix_time(self) -> FakeResponse:
        self.calls.append(("get_unix_time", (), {}))
        return FakeResponse(epoch_seconds="1704067200")

    def get_public_product(self, product_id: str) -> FakeResponse:
        self.calls.append(("get_public_product", (product_id,), {}))
        if product_id == self.fail_product:
            raise RuntimeError("configured fake product failure")
        return FakeResponse(product_id=product_id, nested=FakeResponse(currency="EUR"))

    def get_public_products(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(("get_public_products", (), kwargs))
        return FakeResponse(products=[])

    def get_public_candles(
        self, product_id: str, start: str, end: str, granularity: str, limit: int | None = None
    ) -> FakeResponse:
        self.calls.append(
            ("get_public_candles", (product_id, start, end, granularity), {"limit": limit})
        )
        start_timestamp = int(start)
        return FakeResponse(
            candles=[
                FakeResponse(start=str(start_timestamp + 86400), close="2"),
                FakeResponse(start=str(start_timestamp), close="1"),
            ]
        )


class MidnightAlignedFakePublicClient(FakePublicClient):
    """Fake candles that follow Coinbase's UTC-midnight daily alignment."""

    def __init__(self, omitted_day: int | None = None) -> None:
        super().__init__()
        self.omitted_day = omitted_day

    def get_public_candles(
        self, product_id: str, start: str, end: str, granularity: str, limit: int | None = None
    ) -> FakeResponse:
        self.calls.append(
            ("get_public_candles", (product_id, start, end, granularity), {"limit": limit})
        )
        aligned_start = int(start) - int(start) % 86400
        aligned_end = int(end) - int(end) % 86400
        return FakeResponse(
            candles=[
                FakeResponse(start=str(timestamp), close="1")
                for day, timestamp in enumerate(range(aligned_start, aligned_end, 86400))
                if day != self.omitted_day
            ]
        )


@pytest.fixture
def discovery_module() -> ModuleType:
    return _load_discovery_module()


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 10, 15, 12, 30, 45, tzinfo=UTC)


def _configure_offline_run(
    monkeypatch: pytest.MonkeyPatch,
    discovery_module: ModuleType,
    tmp_path: Path,
    client: FakePublicClient,
) -> None:
    # Existing data-root tests configure the resolver through this environment variable.
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(discovery_module, "make_public_client", lambda: client)


def test_dry_run_makes_no_client_calls(
    monkeypatch: pytest.MonkeyPatch, discovery_module: ModuleType
) -> None:
    client = FakePublicClient()
    monkeypatch.setattr(discovery_module, "make_public_client", lambda: client)

    assert discovery_module.main([]) == 0
    assert client.calls == []


def test_confirmed_run_writes_evidence_and_matching_hashes(
    monkeypatch: pytest.MonkeyPatch,
    discovery_module: ModuleType,
    tmp_path: Path,
    fixed_now: datetime,
) -> None:
    client = FakePublicClient()
    _configure_offline_run(monkeypatch, discovery_module, tmp_path, client)
    sleeps: list[float] = []

    directory = discovery_module.run_discovery(sleep_fn=sleeps.append, now_fn=lambda: fixed_now)

    summary = json.loads((directory / "summary.json").read_text())
    assert len(summary["evidence_files"]) == 14
    for item in summary["evidence_files"]:
        raw_bytes = (directory / item["file"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    product_evidence = json.loads((directory / "03_product_ADA-EUR.json").read_text())
    assert product_evidence["response"]["nested"]["__type__"] == "FakeResponse"
    assert len(client.calls) <= 40
    assert len(sleeps) > 0
    assert len(sleeps) == len(client.calls) - 1
    monkeypatch.setattr(discovery_module, "run_discovery", lambda: directory)
    assert discovery_module.main(["--confirm-network"]) == 0


@pytest.mark.parametrize(
    ("omitted_day", "expected_missing"),
    [
        (None, []),
        (1, ["2026-10-06T00:00:00+00:00"]),
    ],
)
def test_candle_observation_uses_utc_midnight_grid(
    discovery_module: ModuleType, omitted_day: int | None, expected_missing: list[str]
) -> None:
    client = MidnightAlignedFakePublicClient(omitted_day=omitted_day)
    start = datetime(2026, 10, 5, 12, 30, 45, tzinfo=UTC)
    end = datetime(2026, 10, 8, 12, 30, 45, tzinfo=UTC)
    request = discovery_module._candle_request("ADA-EUR", start, end)
    response = client.get_public_candles(**request["params"])
    evidence = {
        "request": request,
        "outcome": "ok",
        "response": discovery_module.serialize_sdk_value(response),
    }

    observation = discovery_module._candle_observation("fake.json", evidence, end)

    assert observation["missing_daily_buckets_utc"] == expected_missing


def test_request_cap_stops_before_an_extra_client_call(
    monkeypatch: pytest.MonkeyPatch,
    discovery_module: ModuleType,
    tmp_path: Path,
    fixed_now: datetime,
) -> None:
    client = FakePublicClient()
    _configure_offline_run(monkeypatch, discovery_module, tmp_path, client)
    monkeypatch.setattr(discovery_module, "MAX_REQUESTS", 2)

    with pytest.raises(discovery_module.RequestCapExceeded):
        discovery_module.run_discovery(sleep_fn=lambda _: None, now_fn=lambda: fixed_now)

    assert len(client.calls) == 2


@pytest.mark.parametrize("case", ["existing", "unset", "inside_repository"])
def test_confirmed_run_refuses_unsafe_data_roots(
    monkeypatch: pytest.MonkeyPatch,
    discovery_module: ModuleType,
    tmp_path: Path,
    fixed_now: datetime,
    case: str,
) -> None:
    client = FakePublicClient()
    if case == "unset":
        monkeypatch.delenv("STRATEGY_LAB_DATA_ROOT", raising=False)
    elif case == "inside_repository":
        monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(Path(__file__).parents[1]))
    else:
        _configure_offline_run(monkeypatch, discovery_module, tmp_path, client)
        directory = (
            tmp_path
            / "downloads"
            / "coinbase"
            / "discovery"
            / fixed_now.strftime("%Y%m%dT%H%M%SZ")
        )
        directory.mkdir(parents=True)
    monkeypatch.setattr(discovery_module, "make_public_client", lambda: client)

    with pytest.raises(discovery_module.DiscoverySetupError):
        discovery_module.run_discovery(sleep_fn=lambda _: None, now_fn=lambda: fixed_now)
    assert client.calls == []


def test_probe_error_is_recorded_and_other_probes_continue(
    monkeypatch: pytest.MonkeyPatch,
    discovery_module: ModuleType,
    tmp_path: Path,
    fixed_now: datetime,
) -> None:
    client = FakePublicClient(fail_product="DOT-EUR")
    _configure_offline_run(monkeypatch, discovery_module, tmp_path, client)

    directory = discovery_module.run_discovery(sleep_fn=lambda _: None, now_fn=lambda: fixed_now)

    failed_probe = json.loads((directory / "04_product_DOT-EUR.json").read_text())
    assert failed_probe["outcome"] == "error"
    assert failed_probe["exception_class"] == "RuntimeError"
    assert "configured fake product failure" in failed_probe["message"]
    assert (directory / "14_server_time_end.json").exists()
    assert len(client.calls) <= 40
    monkeypatch.setattr(discovery_module, "run_discovery", lambda: directory)
    assert discovery_module.main(["--confirm-network"]) == 2
