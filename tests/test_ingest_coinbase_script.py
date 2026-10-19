"""Offline tests for the human-confirmed Coinbase ingestion CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from requests.exceptions import HTTPError

from coinbase_fakes import FakeClient, candle
from strategy_lab.data.providers.coinbase_candles import plan_daily_windows

REPOSITORY_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "ingest_coinbase.py"
NOW_UTC = datetime(2026, 10, 15, 14, 13, 27, tzinfo=UTC)


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ingest_coinbase_script_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if a fake-client test accidentally opens a socket."""

    def forbidden_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ingest script tests must not open sockets")

    monkeypatch.setattr(socket, "socket", forbidden_socket)


@pytest.fixture(scope="module", autouse=True)
def repository_stays_clean() -> None:
    before = _git_status()
    yield
    assert _git_status() == before


@pytest.fixture
def script_module() -> ModuleType:
    return _load_script_module()


def _fake_client() -> FakeClient:
    windows = plan_daily_windows(NOW_UTC)
    candles_by_product = {
        product: [candle(window.first_bucket_start) for window in windows]
        for product in ("ADA-EUR", "DOT-EUR")
    }
    return FakeClient(
        NOW_UTC,
        candles_by_product,
        rate_limit_headers_by_call={
            2: {"x_ratelimit_limit": "10", "x_ratelimit_remaining": "9"},
            31: {"x_ratelimit_limit": "10", "x_ratelimit_remaining": "0"},
        },
    )


def _configure_confirmed_run(
    monkeypatch: pytest.MonkeyPatch,
    script_module: ModuleType,
    tmp_path: Path,
    client: FakeClient,
) -> None:
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(script_module, "make_public_client", lambda **_kwargs: client)
    monkeypatch.setattr(script_module.importlib.metadata, "version", lambda _name: "1.7.0")
    monkeypatch.setattr(script_module.time, "sleep", lambda _seconds: None)


def test_dry_run_creates_no_files_and_never_imports_coinbase_package(tmp_path: Path) -> None:
    before = set(tmp_path.rglob("*"))
    environment = os.environ.copy()
    environment["STRATEGY_LAB_DATA_ROOT"] = str(tmp_path)
    environment.pop("COINBASE_API_KEY", None)
    environment.pop("COINBASE_API_SECRET", None)
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    command = (
        "import runpy, sys; "
        f"module = runpy.run_path({str(SCRIPT_PATH)!r}); "
        "code = module['main']([]); "
        "print('COINBASE_IMPORTED=' + str(any(name == 'coinbase' or name.startswith('coinbase.') "
        "for name in sys.modules))); "
        "raise SystemExit(code)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert set(tmp_path.rglob("*")) == before
    assert "estimate; the real run plans from Coinbase server time" in completed.stdout
    assert "Total request count:" in completed.stdout
    assert "COINBASE_IMPORTED=False" in completed.stdout


def test_confirmed_run_publishes_both_products_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    script_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _fake_client()
    _configure_confirmed_run(monkeypatch, script_module, tmp_path, client)

    assert script_module.main(["--confirm-network"]) == 0

    summaries = list((tmp_path / "diagnostics" / "coinbase_ingest").glob("*.summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert set(summary) >= {
        "products",
        "results",
        "run_start_utc",
        "run_end_utc",
        "total_request_count",
    }
    assert summary["products"] == ["ADA-EUR", "DOT-EUR"]
    assert summary["total_request_count"] == 31
    assert len(summary["results"]["ADA-EUR"]["pages"]) == 15
    output = capsys.readouterr().out
    for product in summary["products"]:
        result = summary["results"][product]
        assert Path(result["raw"]["json_path"]).is_file()
        assert Path(result["processed"]["parquet_path"]).is_file()
        assert result["raw"]["sha256"] in output
        assert result["processed"]["sha256"] in output
    assert client.calls[0] == ("get_unix_time",)


def test_product_selection_publishes_only_dot_eur(
    monkeypatch: pytest.MonkeyPatch, script_module: ModuleType, tmp_path: Path
) -> None:
    client = _fake_client()
    _configure_confirmed_run(monkeypatch, script_module, tmp_path, client)

    assert script_module.main(["--confirm-network", "--product", "DOT-EUR"]) == 0

    assert list((tmp_path / "raw" / "coinbase" / "DOT-EUR").rglob("*.json"))
    assert list((tmp_path / "processed" / "coinbase" / "DOT-EUR").rglob("*.parquet"))
    assert not (tmp_path / "raw" / "coinbase" / "ADA-EUR").exists()
    assert not (tmp_path / "processed" / "coinbase" / "ADA-EUR").exists()


def test_unknown_product_is_an_argparse_error_without_client_call(
    monkeypatch: pytest.MonkeyPatch, script_module: ModuleType, tmp_path: Path
) -> None:
    factory = Mock()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(script_module, "make_public_client", factory)

    with pytest.raises(SystemExit) as captured:
        script_module.main(["--product", "BTC-EUR"])

    assert captured.value.code == 2
    factory.assert_not_called()


def test_duplicate_product_is_rejected_without_silent_deduplication(
    monkeypatch: pytest.MonkeyPatch,
    script_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as captured:
        script_module.main(["--product", "ADA-EUR", "--product", "ADA-EUR"])

    assert captured.value.code == 2
    assert "duplicate --product value: ADA-EUR" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--confirm-network"]])
def test_credentials_preflight_blocks_both_modes_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    script_module: ModuleType,
    tmp_path: Path,
    argv: list[str],
) -> None:
    factory = Mock()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("COINBASE_API_KEY", "")
    monkeypatch.setattr(script_module, "make_public_client", factory)

    assert script_module.main(argv) == 2
    factory.assert_not_called()


def test_http_failure_writes_error_diagnostics_without_publishing_outputs(
    monkeypatch: pytest.MonkeyPatch, script_module: ModuleType, tmp_path: Path
) -> None:
    class HttpFailureClient(FakeClient):
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
            response = SimpleNamespace(
                status_code=429,
                headers={"Retry-After": "3", "X-Trace": "fake"},
                text="too many requests",
            )
            raise HTTPError(response=response)

    base = _fake_client()
    client = HttpFailureClient(base.now_utc, base.candles_by_product)
    _configure_confirmed_run(monkeypatch, script_module, tmp_path, client)

    assert script_module.main(["--confirm-network"]) == 1

    errors = list((tmp_path / "diagnostics" / "coinbase_ingest").glob("*.error.json"))
    assert len(errors) == 1
    payload = json.loads(errors[0].read_text(encoding="utf-8"))
    assert payload["status_code"] == 429
    assert payload["headers"] == {"Retry-After": "3", "X-Trace": "fake"}
    assert payload["request_count"] == 2
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "processed").exists()


def test_reporting_failure_is_diagnosed_and_prints_failed_line(
    monkeypatch: pytest.MonkeyPatch,
    script_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _fake_client()
    _configure_confirmed_run(monkeypatch, script_module, tmp_path, client)

    def fail_reporting(*_args: object) -> None:
        raise RuntimeError("configured reporting failure")

    monkeypatch.setattr(script_module, "_print_published_results", fail_reporting)

    assert script_module.main(["--confirm-network"]) == 1

    stderr = capsys.readouterr().err
    assert "FAILED: configured reporting failure" in stderr
    errors = list((tmp_path / "diagnostics" / "coinbase_ingest").glob("*.error.json"))
    assert len(errors) == 1
    assert json.loads(errors[0].read_text(encoding="utf-8"))["exception_class"] == "RuntimeError"


def test_unset_data_root_has_one_clear_message_without_traceback(
    monkeypatch: pytest.MonkeyPatch, script_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("STRATEGY_LAB_DATA_ROOT", raising=False)

    assert script_module.main([]) == 1

    stderr = capsys.readouterr().err
    assert "STRATEGY_LAB_DATA_ROOT is not set" in stderr
    assert stderr.count("FAILED:") == 1
    assert "Traceback" not in stderr
