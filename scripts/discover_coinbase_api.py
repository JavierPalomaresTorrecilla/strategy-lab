#!/usr/bin/env python3
"""Human-run evidence collection for Coinbase public REST endpoint behavior.

This utility makes no request unless ``--confirm-network`` is supplied.  Its
JSON captures are SDK-decoded evidence, rather than raw HTTP wire captures:
the recursive serializer records the plain attributes exposed by the SDK's
response objects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from strategy_lab.data.env import DataRootError, resolve_data_root
from strategy_lab.data.providers.coinbase_client import make_public_client

MAX_REQUESTS = 40
GRANULARITY = "ONE_DAY"
PRODUCT_IDS = ("ADA-EUR", "DOT-EUR")
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class DiscoverySetupError(RuntimeError):
    """A precondition prevents safe discovery evidence collection."""


class RequestCapExceeded(RuntimeError):
    """The fixed safety cap for outbound requests would be exceeded."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp, isolated for offline tests."""
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _timestamp_name(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is equal to or contained by ``parent``."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _discovery_directory(run_start: datetime) -> Path:
    """Resolve and validate the sole permitted output directory."""
    try:
        data_root = resolve_data_root().resolve()
    except DataRootError as exc:
        raise DiscoverySetupError(f"unable to resolve data root: {exc}") from exc

    run_directory = (
        data_root / "downloads" / "coinbase" / "discovery" / _timestamp_name(run_start)
    ).resolve()
    repository_root = REPOSITORY_ROOT.resolve()
    if _is_within(run_directory, repository_root):
        raise DiscoverySetupError(
            "refusing to write discovery evidence inside the repository working tree: "
            f"{run_directory}"
        )
    if run_directory.exists():
        raise DiscoverySetupError(
            f"refusing to overwrite existing discovery directory: {run_directory}"
        )
    return run_directory


def serialize_sdk_value(value: Any) -> Any:
    """Recursively serialize SDK-decoded objects, preserving their type names."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [serialize_sdk_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_sdk_value(item) for key, item in value.items()}
    try:
        attributes = vars(value)
    except TypeError:
        return str(value)
    serialized = {str(key): serialize_sdk_value(item) for key, item in attributes.items()}
    serialized["__type__"] = type(value).__name__
    return serialized


def _candle_request(product_id: str, start: datetime, end: datetime) -> dict[str, Any]:
    """Build an evidence request block without changing the SDK arguments."""
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    return {
        "method": "get_public_candles",
        "params": {
            "product_id": product_id,
            "start": str(int(start.timestamp())),
            "end": str(int(end.timestamp())),
            "granularity": GRANULARITY,
        },
        "datetime_equivalents_utc": {
            "start": _isoformat(start),
            "end": _isoformat(end),
        },
    }


def _environment_request() -> dict[str, Any]:
    return {"method": "environment", "params": {}}


def _environment_response() -> dict[str, Any]:
    try:
        coinbase_version = importlib.metadata.version("coinbase-advanced-py")
    except importlib.metadata.PackageNotFoundError:
        coinbase_version = None
    return {
        "python_version": sys.version,
        "coinbase_advanced_py_version": coinbase_version,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
    }


def planned_probes() -> list[str]:
    """Describe the stable probe list used by the dry-run command."""
    return [
        "environment metadata (no network request)",
        "server time at run start",
        "public product metadata for ADA-EUR",
        "public product metadata for DOT-EUR",
        "recent approximately 10-day daily candles for ADA-EUR",
        "recent approximately 10-day daily candles for DOT-EUR",
        "2015 pre-listing daily candles for ADA-EUR",
        "2015 pre-listing daily candles for DOT-EUR",
        "ADA-EUR 300-day request-size candle probe",
        "ADA-EUR 350-day request-size candle probe",
        "ADA-EUR 351-day request-size candle probe",
        "ADA-EUR long-window daily candles selected from request-size results",
        "DOT-EUR long-window daily candles selected from request-size results",
        "server time at run end",
    ]


class EvidenceRun:
    """Write one JSON evidence record per probe and enforce request pacing."""

    def __init__(self, directory: Path, sleep_fn: Callable[[float], None]) -> None:
        self.directory = directory
        self.sleep_fn = sleep_fn
        self.request_count = 0
        self._evidence_files: list[Path] = []

    @property
    def evidence_files(self) -> list[Path]:
        return list(self._evidence_files)

    def _write(self, filename: str, evidence: dict[str, Any]) -> None:
        path = self.directory / filename
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._evidence_files.append(path)

    def environment(self) -> None:
        request_time = utc_now()
        self._write(
            "01_environment.json",
            {
                "request": _environment_request(),
                "local_utc_request_time": _isoformat(request_time),
                "elapsed_milliseconds": 0.0,
                "outcome": "ok",
                "response": _environment_response(),
            },
        )

    def probe(self, filename: str, request: dict[str, Any], call: Callable[[], Any]) -> bool:
        """Execute a single client call, recording every result or exception."""
        if self.request_count >= MAX_REQUESTS:
            raise RequestCapExceeded(
                f"request safety cap of {MAX_REQUESTS} would be exceeded before {filename}"
            )
        if self.request_count > 0:
            self.sleep_fn(1.0)
        self.request_count += 1
        request_time = utc_now()
        started = time.perf_counter()
        try:
            response = call()
        except Exception as exc:  # Network failures and SDK response-decoding failures.
            evidence = _error_evidence(request, request_time, started, exc)
        else:
            evidence = {
                "request": request,
                "local_utc_request_time": _isoformat(request_time),
                "elapsed_milliseconds": (time.perf_counter() - started) * 1000,
                "outcome": "ok",
                "response": serialize_sdk_value(response),
            }
        self._write(filename, evidence)
        return evidence["outcome"] == "ok"


def _error_evidence(
    request: dict[str, Any], request_time: datetime, started: float, exc: Exception
) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    return {
        "request": request,
        "local_utc_request_time": _isoformat(request_time),
        "elapsed_milliseconds": (time.perf_counter() - started) * 1000,
        "outcome": "error",
        "exception_class": type(exc).__name__,
        "message": str(exc),
        "status_code": getattr(response, "status_code", None),
        "body": getattr(response, "text", None),
    }


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _candle_observation(
    filename: str, evidence: dict[str, Any], run_start: datetime
) -> dict[str, Any]:
    """Compute only directly observable candle properties from saved evidence."""
    request = evidence["request"]
    params = request["params"]
    observation: dict[str, Any] = {
        "evidence_file": filename,
        "product_id": params["product_id"],
        "outcome": evidence["outcome"],
    }
    if evidence["outcome"] != "ok":
        return observation
    response = evidence.get("response")
    candles = response.get("candles", []) if isinstance(response, dict) else []
    candle_starts = [
        _as_int(candle.get("start")) if isinstance(candle, dict) else None for candle in candles
    ]
    timestamps = [timestamp for timestamp in candle_starts if timestamp is not None]
    requested_start = _as_int(params["start"])
    requested_end = _as_int(params["end"])
    expected_timestamps = (
        list(
            range(
                requested_start - requested_start % 86400,
                requested_end - requested_end % 86400,
                86400,
            )
        )
        if requested_start is not None and requested_end is not None
        else []
    )
    if len(timestamps) < 2:
        ordering = "single_or_empty"
    elif all(left <= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
        ordering = "ascending"
    elif all(left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
        ordering = "descending"
    else:
        ordering = "mixed"
    newest = max(timestamps) if timestamps else None
    observation.update(
        {
            "ordering_by_start_timestamp": ordering,
            "candle_start_timestamp_modulo_86400": [
                timestamp % 86400 if timestamp is not None else None for timestamp in candle_starts
            ],
            "returned_candle_count": len(candles),
            "returned_candles_with_parseable_start_count": len(timestamps),
            "requested_calendar_day_count": len(expected_timestamps),
            "missing_daily_buckets_utc": [
                _isoformat(datetime.fromtimestamp(timestamp, UTC))
                for timestamp in expected_timestamps
                if timestamp not in set(timestamps)
            ],
            "newest_candle_start_is_on_run_start_utc_date": (
                datetime.fromtimestamp(newest, UTC).date() == run_start.date()
                if newest is not None
                else None
            ),
        }
    )
    return observation


def _write_summary(evidence_run: EvidenceRun, run_start: datetime, run_end: datetime) -> None:
    """Hash files and derive mechanical observations from their stored JSON."""
    files = []
    candle_observations = []
    for path in evidence_run.evidence_files:
        raw_bytes = path.read_bytes()
        files.append({"file": path.name, "sha256": hashlib.sha256(raw_bytes).hexdigest()})
        evidence = json.loads(raw_bytes)
        if evidence.get("request", {}).get("method") == "get_public_candles":
            candle_observations.append(_candle_observation(path.name, evidence, run_start))
    summary = {
        "run_start_utc": _isoformat(run_start),
        "run_end_utc": _isoformat(run_end),
        "total_request_count": evidence_run.request_count,
        "evidence_files": files,
        "mechanical_observations": {"candle_probes": candle_observations},
    }
    (evidence_run.directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _has_probe_errors(directory: Path) -> bool:
    """Return whether any evidence file in the completed run recorded an error."""
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return any(
        json.loads((directory / item["file"]).read_text(encoding="utf-8")).get("outcome")
        == "error"
        for item in summary["evidence_files"]
    )


def run_discovery(
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = utc_now,
) -> Path:
    """Run the confirmed discovery probes and return their evidence directory."""
    run_start = now_fn().astimezone(UTC)
    directory = _discovery_directory(run_start)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DiscoverySetupError(
            f"refusing to overwrite existing discovery directory: {directory}"
        ) from exc

    evidence_run = EvidenceRun(directory, sleep_fn)
    evidence_run.environment()
    try:
        client = make_public_client()
    except Exception as exc:
        raise DiscoverySetupError(f"unable to construct Coinbase public client: {exc}") from exc

    evidence_run.probe(
        "02_server_time_start.json",
        {"method": "get_unix_time", "params": {}},
        client.get_unix_time,
    )
    for index, product_id in enumerate(PRODUCT_IDS, start=3):
        evidence_run.probe(
            f"{index:02d}_product_{product_id}.json",
            {"method": "get_public_product", "params": {"product_id": product_id}},
            lambda product_id=product_id: client.get_public_product(product_id),
        )

    recent_start = run_start - timedelta(days=10)
    for index, product_id in enumerate(PRODUCT_IDS, start=5):
        request = _candle_request(product_id, recent_start, run_start)
        evidence_run.probe(
            f"{index:02d}_recent_candles_{product_id}.json",
            request,
            lambda request=request: client.get_public_candles(**request["params"]),
        )

    pre_listing_start = datetime(2015, 1, 1, tzinfo=UTC)
    pre_listing_end = datetime(2015, 1, 10, tzinfo=UTC)
    for index, product_id in enumerate(PRODUCT_IDS, start=7):
        request = _candle_request(product_id, pre_listing_start, pre_listing_end)
        evidence_run.probe(
            f"{index:02d}_pre_listing_candles_{product_id}.json",
            request,
            lambda request=request: client.get_public_candles(**request["params"]),
        )

    size_probe_end = run_start - timedelta(days=400)
    size_probe_successes: list[int] = []
    for index, days in enumerate((300, 350, 351), start=9):
        request = _candle_request("ADA-EUR", size_probe_end - timedelta(days=days), size_probe_end)
        if evidence_run.probe(
            f"{index:02d}_size_{days}_days_ADA-EUR.json",
            request,
            lambda request=request: client.get_public_candles(**request["params"]),
        ):
            size_probe_successes.append(days)

    long_days = max(size_probe_successes, default=300)
    for index, product_id in enumerate(PRODUCT_IDS, start=12):
        request = _candle_request(
            product_id, size_probe_end - timedelta(days=long_days), size_probe_end
        )
        evidence_run.probe(
            f"{index:02d}_long_{long_days}_days_{product_id}.json",
            request,
            lambda request=request: client.get_public_candles(**request["params"]),
        )
    evidence_run.probe(
        "14_server_time_end.json",
        {"method": "get_unix_time", "params": {}},
        client.get_unix_time,
    )

    run_end = now_fn().astimezone(UTC)
    _write_summary(evidence_run, run_start, run_end)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-network",
        action="store_true",
        help="permit the documented public-data client calls",
    )
    args = parser.parse_args(argv)
    if not args.confirm_network:
        probes = planned_probes()
        print(f"Dry run: {len(probes)} probes would be collected; no client calls were made.")
        for number, description in enumerate(probes, start=1):
            print(f"{number}. {description}")
        return 0
    try:
        directory = run_discovery()
    except (DiscoverySetupError, RequestCapExceeded) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Discovery evidence written to: {directory}")
    if _has_probe_errors(directory):
        print(
            "Some probes recorded errors; evidence was written for the completed run.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
