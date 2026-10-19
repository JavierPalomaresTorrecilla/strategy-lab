#!/usr/bin/env python3
"""Human-confirmed ingestion of Coinbase public daily-candle snapshots.

The default command is an entirely local dry run.  Supplying
``--confirm-network`` permits the public Coinbase calls and publishes immutable
raw and processed snapshots beneath the configured external data root.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strategy_lab.data.env import resolve_data_root
from strategy_lab.data.processed import processed_snapshot_dir
from strategy_lab.data.providers.coinbase_candles import plan_daily_windows
from strategy_lab.data.providers.coinbase_client import make_public_client
from strategy_lab.data.providers.coinbase_ingest import REQUEST_CAP, ProviderHTTPError, fetch_run
from strategy_lab.data.providers.coinbase_pipeline import publish_run
from strategy_lab.data.providers.coinbase_raw import (
    ALLOWED_PRODUCTS,
    INTERVAL,
    PROVIDER,
    RAW_SCHEMA_VERSION,
)
from strategy_lab.data.snapshot import raw_snapshot_dir

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRODUCTS = tuple(sorted(ALLOWED_PRODUCTS))
_CREDENTIAL_ENV_VARS = ("COINBASE_API_KEY", "COINBASE_API_SECRET")


class _AlreadyReported(RuntimeError):
    """Stop after a confirmed run has already printed its failure message."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp, isolated for offline tests."""
    return datetime.now(UTC)


def _timestamp_name(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is equal to or contained by ``parent``."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_safe_data_root(data_root: Path) -> Path:
    """Reject a configured root that could place output in this repository."""
    resolved = data_root.resolve()
    if _is_within(resolved, REPOSITORY_ROOT.resolve()):
        raise RuntimeError(
            "refusing to write Coinbase ingestion output inside the repository working tree: "
            f"{resolved}"
        )
    return resolved


def _diagnostics_directory(data_root: Path) -> Path:
    directory = (data_root / "diagnostics" / "coinbase_ingest").resolve()
    if _is_within(directory, REPOSITORY_ROOT.resolve()):
        raise RuntimeError(
            "refusing to write Coinbase ingestion diagnostics inside the repository working tree: "
            f"{directory}"
        )
    return directory


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _output_entries(data_root: Path) -> set[str]:
    """Return existing raw/processed entries, relative to the data root."""
    entries: set[str] = set()
    for tree_name in ("raw", "processed"):
        tree = data_root / tree_name
        if tree.exists():
            entries.add(str(tree.relative_to(data_root)))
            entries.update(str(path.relative_to(data_root)) for path in tree.rglob("*"))
    return entries


def _selected_products(
    values: list[str] | None, parser: argparse.ArgumentParser
) -> tuple[str, ...]:
    products = tuple(values) if values else DEFAULT_PRODUCTS
    duplicates = sorted({product for product in products if products.count(product) > 1})
    if duplicates:
        parser.error(f"duplicate --product value: {duplicates[0]}")
    return products


def _credential_preflight() -> str | None:
    configured = [name for name in _CREDENTIAL_ENV_VARS if name in os.environ]
    if not configured:
        return None
    return (
        "refusing Coinbase public ingestion while credential environment variable(s) are set: "
        + ", ".join(configured)
    )


def _format_window(window: Any) -> str:
    return (
        f"{window.first_bucket_start.date().isoformat()} through "
        f"{window.last_bucket_start.date().isoformat()}"
    )


def _dry_run(data_root: Path, products: tuple[str, ...]) -> None:
    local_now = utc_now()
    windows = plan_daily_windows(local_now)
    request_count = 1 + len(products) * len(windows)
    print("Dry run: no client was constructed and no network request was made.")
    print(f"Selected products: {', '.join(products)}")
    print(f"Local UTC planning time: {local_now.isoformat()}")
    print(f"Planned windows: {len(windows)}")
    print(f"First window: {_format_window(windows[0])}")
    print(f"Last window: {_format_window(windows[-1])}")
    print(
        f"Total request count: {request_count} of cap {REQUEST_CAP} "
        "(estimate; the real run plans from Coinbase server time)"
    )
    for product in products:
        raw_directory = raw_snapshot_dir(data_root, PROVIDER, product, INTERVAL)
        print(f"{product} raw directory: {raw_directory}")
        print(
            f"{product} processed directory: "
            f"{processed_snapshot_dir(data_root, product, INTERVAL, provider=PROVIDER)}"
        )


def _result_summary(
    artifacts: dict[str, dict], results: dict[str, tuple[Any, Any]], products: tuple[str, ...]
) -> dict[str, Any]:
    product_results: dict[str, dict[str, Any]] = {}
    for product in products:
        raw_ref, processed_ref = results[product]
        product_results[product] = {
            "raw": {
                "json_path": str(raw_ref.json_path),
                "metadata_path": str(raw_ref.metadata_path),
                "sha256": raw_ref.sha256,
            },
            "processed": {
                "parquet_path": str(processed_ref.parquet_path),
                "metadata_path": str(processed_ref.metadata_path),
                "sha256": processed_ref.sha256,
            },
            "pages": [
                {
                    "request": page["request"],
                    "rate_limit_headers": page["rate_limit_headers"],
                }
                for page in artifacts[product]["pages"]
            ],
        }
    return product_results


def _print_published_results(
    artifacts: dict[str, dict], results: dict[str, tuple[Any, Any]], products: tuple[str, ...]
) -> None:
    for product in products:
        raw_ref, processed_ref = results[product]
        metadata = json.loads(processed_ref.metadata_path.read_text(encoding="utf-8"))
        missing_days = metadata["missing_days_utc"]
        missing_count = metadata["missing_day_count"]
        if missing_count <= 20:
            missing_display = ", ".join(missing_days) or "none"
        else:
            missing_display = ", ".join(missing_days[:10]) + f", ... and {missing_count - 10} more"
        pages = artifacts[product]["pages"]
        first_headers = pages[0]["rate_limit_headers"]
        last_headers = pages[-1]["rate_limit_headers"]
        first_timestamp = metadata["first_timestamp"]
        last_timestamp = metadata["last_timestamp"]
        first_date = (
            datetime.fromisoformat(first_timestamp).date().isoformat()
            if first_timestamp
            else "none"
        )
        last_date = (
            datetime.fromisoformat(last_timestamp).date().isoformat()
            if last_timestamp
            else "none"
        )
        print(f"{product} raw: {raw_ref.json_path} sha256={raw_ref.sha256}")
        print(f"{product} processed: {processed_ref.parquet_path} sha256={processed_ref.sha256}")
        print(
            f"{product} candles: {metadata['row_count']}; earliest UTC bucket: {first_date}; "
            f"latest UTC bucket: {last_date}"
        )
        print(f"{product} missing days ({missing_count}): {missing_display}")
        print(f"{product} first-page rate-limit headers: {first_headers}")
        print(f"{product} last-page rate-limit headers: {last_headers}")


class _RequestCountingClient:
    """Count the two public calls used by ``fetch_run`` without altering them."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.request_count = 0

    def get_unix_time(self) -> Any:
        self.request_count += 1
        return self._client.get_unix_time()

    def get_public_candles(self, **request: str) -> Any:
        self.request_count += 1
        return self._client.get_public_candles(**request)


def _write_error(
    data_root: Path,
    run_start: datetime,
    error: Exception,
    request_count: int | None,
    new_output_entries: set[str],
) -> Path:
    payload: dict[str, Any] = {
        "exception_class": type(error).__name__,
        "message": str(error),
        "new_raw_or_processed_entries": sorted(new_output_entries),
        "request_count": request_count,
        "run_end_utc": utc_now().isoformat(),
        "run_start_utc": run_start.isoformat(),
    }
    if isinstance(error, ProviderHTTPError):
        payload["body"] = error.body
        payload["headers"] = error.headers
        payload["status_code"] = error.status_code
    path = _diagnostics_directory(data_root) / f"{_timestamp_name(run_start)}.error.json"
    _write_json(path, payload)
    return path


def _confirmed_run(data_root: Path, products: tuple[str, ...]) -> None:
    run_start = utc_now()
    output_before = _output_entries(data_root)
    counting_client: _RequestCountingClient | None = None
    try:
        counting_client = _RequestCountingClient(make_public_client(rate_limit_headers=True))
        artifacts = fetch_run(
            counting_client,
            products,
            local_clock_fn=utc_now,
            monotonic_fn=time.monotonic,
            sleep_fn=time.sleep,
            python_version=platform.python_version(),
            sdk_version=importlib.metadata.version("coinbase-advanced-py"),
        )
        results = publish_run(artifacts, data_root)
        run_end = utc_now()
        request_count = counting_client.request_count
        summary = {
            "products": list(products),
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "results": _result_summary(artifacts, results, products),
            "run_end_utc": run_end.isoformat(),
            "run_start_utc": run_start.isoformat(),
            "total_request_count": request_count,
        }
        summary_path = (
            _diagnostics_directory(data_root) / f"{_timestamp_name(run_start)}.summary.json"
        )
        _write_json(summary_path, summary)
        _print_published_results(artifacts, results, products)
        print(f"Run summary: {summary_path}")
    except Exception as error:
        output_after = _output_entries(data_root)
        new_entries = output_after - output_before
        request_count = counting_client.request_count if counting_client is not None else None
        try:
            _write_error(data_root, run_start, error, request_count, new_entries)
        except Exception as diagnostic_error:
            print(
                f"FAILED: {error} (also could not write error diagnostics: "
                f"{diagnostic_error})",
                file=sys.stderr,
            )
        else:
            if new_entries:
                print(
                    f"FAILED: {error} ({len(new_entries)} new raw/processed entries were created "
                    "before failure; see error diagnostics.)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"FAILED: {error} (verified no new raw/processed entries; "
                    "see error diagnostics.)",
                    file=sys.stderr,
                )
        raise _AlreadyReported from error


def main(argv: list[str] | None = None) -> int:
    credential_error = _credential_preflight()
    if credential_error is not None:
        print(f"FAILED: {credential_error}", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-network",
        action="store_true",
        help="permit public Coinbase requests and immutable snapshot publication",
    )
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(ALLOWED_PRODUCTS),
        help="product to ingest; repeat to select more than one (defaults to both)",
    )
    args = parser.parse_args(argv)
    products = _selected_products(args.product, parser)

    try:
        data_root = _ensure_safe_data_root(resolve_data_root())
        if not args.confirm_network:
            _dry_run(data_root, products)
            return 0
        _confirmed_run(data_root, products)
    except _AlreadyReported:
        return 1
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
