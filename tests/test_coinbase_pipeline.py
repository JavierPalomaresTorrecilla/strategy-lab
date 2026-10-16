"""Offline publication and replay coverage for Coinbase raw snapshots."""

from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from coinbase_fakes import FakeClient, candle, utc_day
from strategy_lab.data.providers.coinbase_candles import CandleWindow, plan_daily_windows
from strategy_lab.data.providers.coinbase_canonical import (
    CandleOutsideWindowError,
    DuplicateCandleError,
    IncompleteCandleError,
    MalformedCandleError,
    decode_pages,
)
from strategy_lab.data.providers.coinbase_ingest import fetch_run
from strategy_lab.data.providers.coinbase_pipeline import (
    VOLUME_UNIT_FIELDS,
    build_processed_from_raw,
    publish_run,
    windows_from_artifact,
)
from strategy_lab.data.providers.coinbase_raw import RawJsonSnapshotRef
from strategy_lab.data.snapshot import compute_sha256

NOW_UTC = datetime(2026, 10, 15, 14, 13, 27, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 10, 15, 14, 14, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if this fake-response suite opens a socket."""

    def forbidden_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("coinbase pipeline tests must not open sockets")

    monkeypatch.setattr(socket, "socket", forbidden_socket)


def _fetch(candles_by_product: dict[str, list]) -> dict[str, dict]:
    return fetch_run(
        FakeClient(NOW_UTC, candles_by_product),
        list(candles_by_product),
        local_clock_fn=lambda: COMPLETED_AT,
        monotonic_fn=_SteppingClock(),
        sleep_fn=lambda _seconds: None,
        python_version="3.11.9",
        sdk_version="1.7.0",
    )


class _SteppingClock:
    def __init__(self) -> None:
        self.value = -2.0

    def __call__(self) -> float:
        self.value += 2.0
        return self.value


def _freeze_processed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2026, 10, 15, 15, 0, 0, 123456, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("strategy_lab.data.processed.datetime", _FixedDatetime)


def _single_artifact() -> dict:
    return _fetch({"ADA-EUR": [candle(utc_day(2026, 10, 12)), candle(utc_day(2026, 10, 14))]})[
        "ADA-EUR"
    ]


def _raw_candle(day: datetime, *, volume: str = "3") -> dict[str, str]:
    return {
        "start": str(int(day.timestamp())),
        "low": "1",
        "high": "2",
        "open": "1",
        "close": "2",
        "volume": volume,
    }


def test_end_to_end_publishes_namespaced_lineage_and_volume_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_processed_clock(monkeypatch)
    artifacts = _fetch(
        {
            "ADA-EUR": [candle(utc_day(2026, 10, 12)), candle(utc_day(2026, 10, 13))],
            "DOT-EUR": [candle(utc_day(2026, 10, 13)), candle(utc_day(2026, 10, 14))],
        }
    )

    published = publish_run(artifacts, tmp_path)

    for product, (raw_ref, processed_ref) in published.items():
        assert raw_ref.json_path.parent == tmp_path / "raw" / "coinbase" / product / "1d"
        assert processed_ref.parquet_path.parent == (
            tmp_path / "processed" / "coinbase" / product / "1d"
        )
        metadata_bytes = processed_ref.metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        assert metadata["source_raw_file"] == str(raw_ref.json_path.relative_to(tmp_path))
        assert metadata["source_raw_sha256"] == raw_ref.sha256
        assert metadata["provider"] == "coinbase"
        assert metadata["timezone"] == "UTC"
        assert metadata["timestamp_semantics"] == "utc_day_bucket_start"
        assert metadata["volume_unit"] is VOLUME_UNIT_FIELDS["volume_unit"]
        for field in ("volume_unit_status", "volume_unit_inference", "volume_unit_evidence"):
            assert metadata[field] == VOLUME_UNIT_FIELDS[field]
        assert b'"volume_unit": null' in metadata_bytes

    ada_metadata = json.loads(published["ADA-EUR"][1].metadata_path.read_text())
    assert ada_metadata["first_timestamp"] == "2026-10-12T00:00:00"
    assert ada_metadata["missing_days_utc"] == ["2026-10-14"]


def test_gaps_are_published_as_metadata_not_errors(tmp_path: Path) -> None:
    artifacts = _fetch(
        {"ADA-EUR": [candle(utc_day(2026, 10, 10)), candle(utc_day(2026, 10, 12))]}
    )

    processed_ref = publish_run(artifacts, tmp_path)["ADA-EUR"][1]

    metadata = json.loads(processed_ref.metadata_path.read_text())
    assert metadata["missing_days_utc"] == ["2026-10-11", "2026-10-13", "2026-10-14"]
    assert metadata["missing_day_count"] == 3


def test_replay_from_only_raw_is_byte_deterministic(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_processed_clock(monkeypatch)
    artifacts = _fetch({"ADA-EUR": [candle(utc_day(2026, 10, 12)), candle(utc_day(2026, 10, 14))]})
    original_raw, original_processed = publish_run(artifacts, tmp_path)["ADA-EUR"]
    replay_root = tmp_path_factory.mktemp("replay")
    shutil.copytree(tmp_path / "raw", replay_root / "raw")
    replay_raw = RawJsonSnapshotRef(
        json_path=replay_root / original_raw.json_path.relative_to(tmp_path),
        metadata_path=replay_root / original_raw.metadata_path.relative_to(tmp_path),
        sha256=original_raw.sha256,
    )

    replay_processed = build_processed_from_raw(replay_raw, replay_root)

    assert compute_sha256(original_processed.parquet_path) == compute_sha256(
        replay_processed.parquet_path
    )
    assert (
        original_processed.metadata_path.read_bytes()
        == replay_processed.metadata_path.read_bytes()
    )


@pytest.mark.parametrize(
    ("failure", "exception_type"),
    [
        ("duplicate", DuplicateCandleError),
        ("outside", CandleOutsideWindowError),
        ("incomplete", IncompleteCandleError),
        ("malformed", MalformedCandleError),
    ],
)
def test_validation_failure_writes_nothing_for_any_product(
    failure: str, exception_type: type[ValueError], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _fetch(
        {
            "ADA-EUR": [candle(utc_day(2026, 10, 12))],
            "DOT-EUR": [candle(utc_day(2026, 10, 12))],
        }
    )
    broken = artifacts["DOT-EUR"]
    page = next(page for page in broken["pages"] if page["candles"])
    if failure == "duplicate":
        page["candles"].append(copy.deepcopy(page["candles"][0]))
    elif failure == "outside":
        page["candles"][0]["start"] = str(int(page["request"]["end"]) + 86_400)
    elif failure == "incomplete":
        # The real planner excludes today, so widen only the replanned final
        # window in this test to exercise decoder protection for an API that
        # returns today's still-open bucket.
        today = datetime(2026, 10, 15, tzinfo=UTC)
        for artifact in artifacts.values():
            artifact["pages"][-1]["request"]["end"] = str(int(today.timestamp()))
        final_page = broken["pages"][-1]
        final_page["candles"].append(
            {
                "start": str(int(today.timestamp())),
                "low": "1.0",
                "high": "3.0",
                "open": "1.5",
                "close": "2.5",
                "volume": "4.0",
            }
        )
        original_planner = plan_daily_windows

        def planner_with_today(now_utc: datetime):
            windows = list(original_planner(now_utc))
            last = windows[-1]
            windows[-1] = CandleWindow(last.first_bucket_start, today)
            return tuple(windows)

        monkeypatch.setattr(
            "strategy_lab.data.providers.coinbase_pipeline.plan_daily_windows", planner_with_today
        )
    else:
        page["candles"][0]["volume"] = "not-a-decimal"

    with pytest.raises(exception_type):
        publish_run(artifacts, tmp_path)

    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").rglob("*"))
    assert not (tmp_path / "processed").exists() or not list((tmp_path / "processed").rglob("*"))


@pytest.mark.parametrize("mutation", ["end", "remove", "swap"])
def test_windows_from_artifact_rejects_replanned_request_mismatches(mutation: str) -> None:
    artifact = _single_artifact()
    if mutation == "end":
        artifact["pages"][0]["request"]["end"] = str(
            int(artifact["pages"][0]["request"]["end"]) - 86_400
        )
    elif mutation == "remove":
        artifact["pages"].pop()
    else:
        artifact["pages"][0], artifact["pages"][1] = (
            artifact["pages"][1],
            artifact["pages"][0],
        )

    with pytest.raises(ValueError, match="(page|count).*(mismatch|match)"):
        windows_from_artifact(artifact)


def test_decoder_gap_carry_forward_is_page_order_independent() -> None:
    first = CandleWindow(utc_day(2026, 10, 1), utc_day(2026, 10, 2))
    second = CandleWindow(utc_day(2026, 10, 3), utc_day(2026, 10, 4))
    first_page = [_raw_candle(utc_day(2026, 10, 1))]
    second_page = [_raw_candle(utc_day(2026, 10, 4))]
    now = utc_day(2026, 10, 5)

    ascending = decode_pages("ADA-EUR", [(first, first_page), (second, second_page)], now)
    reversed_pages = decode_pages("ADA-EUR", [(second, second_page), (first, first_page)], now)

    assert ascending.missing_days_utc == reversed_pages.missing_days_utc == (
        "2026-10-02",
        "2026-10-03",
    )


def test_malformed_candle_error_includes_product_and_chains_parse_error() -> None:
    window = CandleWindow(utc_day(2026, 10, 1), utc_day(2026, 10, 1))
    malformed = [_raw_candle(utc_day(2026, 10, 1), volume="bad")]

    with pytest.raises(MalformedCandleError) as captured:
        decode_pages("ADA-EUR", [(window, malformed)], utc_day(2026, 10, 2))

    assert "ADA-EUR" in str(captured.value)
    assert isinstance(captured.value.__cause__, MalformedCandleError)


def test_pipeline_import_does_not_import_coinbase_sdk() -> None:
    child_script = """
import sys
import strategy_lab.data.providers.coinbase_pipeline
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
