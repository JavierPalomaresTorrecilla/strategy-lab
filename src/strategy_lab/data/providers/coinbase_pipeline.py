"""Offline raw-to-processed publication for Coinbase daily candles.

``publish_run`` validates every supplied artifact through the exact serialized
replay path before writing anything.  This gives validation failures an
all-or-nothing guarantee: no raw or processed artifact is published for any
product.  Actual Phase 2/3 filesystem failures can still leave files published
for earlier products, because the underlying immutable exclusive-create
primitives are per-artifact rather than a transaction spanning products.  This
module deliberately does not attempt rollback of those independently published
snapshots.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from strategy_lab.data.processed import ProcessedSnapshotRef, save_processed_snapshot
from strategy_lab.data.providers.coinbase_candles import CandleWindow, plan_daily_windows
from strategy_lab.data.providers.coinbase_canonical import (
    CoinbaseDecodeResult,
    decode_pages,
    to_canonical_frame,
)
from strategy_lab.data.providers.coinbase_raw import (
    RawJsonSnapshotRef,
    load_raw_json_snapshot,
    save_raw_json_snapshot,
    serialize_raw_artifact,
)

VOLUME_UNIT_FIELDS: dict[str, str | None] = {
    "volume_unit": None,
    "volume_unit_status": "unconfirmed",
    "volume_unit_inference": "base_asset",
    "volume_unit_evidence": (
        "experiments/manifests/m5a_coinbase_discovery.json#"
        "inferred.candle_volume_is_in_base_units"
    ),
}


def _started_at_utc(artifact: dict) -> datetime:
    """Parse the frozen server time, rejecting naive or non-UTC values."""
    value = artifact["retrieval"]["started_at_utc"]
    if not isinstance(value, str):
        raise ValueError("artifact retrieval.started_at_utc must be an ISO-8601 UTC string")
    parse_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise ValueError(
            f"artifact retrieval.started_at_utc must be an ISO-8601 UTC string, got {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(
            f"artifact retrieval.started_at_utc must be an aware UTC value, got {value!r}"
        )
    return parsed.astimezone(UTC)


def windows_from_artifact(artifact: dict) -> tuple[CandleWindow, ...]:
    """Replan and verify the frozen request windows recorded in *artifact*.

    ``ValueError`` identifies the mismatched page index when the stored request
    plan differs from the plan deterministically derived from server time.
    """
    now_utc = _started_at_utc(artifact)
    windows = plan_daily_windows(now_utc)
    pages = artifact["pages"]
    if len(pages) != len(windows):
        raise ValueError(
            "artifact page count does not match replanned Coinbase windows: "
            f"expected {len(windows)}, found {len(pages)}"
        )

    for index, (window, page) in enumerate(zip(windows, pages, strict=True)):
        request = page["request"]
        found_start = request["start"]
        found_end = request["end"]
        if found_start != window.request_start or found_end != window.request_end:
            raise ValueError(
                f"artifact page {index} request window mismatch: expected "
                f"start={window.request_start!r}, end={window.request_end!r}; found "
                f"start={found_start!r}, end={found_end!r}"
            )
    return windows


def decode_artifact(artifact: dict) -> CoinbaseDecodeResult:
    """Decode a raw artifact after independently re-deriving its page plan."""
    windows = windows_from_artifact(artifact)
    started_at_utc = _started_at_utc(artifact)
    pages = list(zip(windows, (page["candles"] for page in artifact["pages"]), strict=True))
    return decode_pages(artifact["product"], pages, now_utc=started_at_utc)


def build_processed_from_raw(raw_ref: RawJsonSnapshotRef, data_root: Path) -> ProcessedSnapshotRef:
    """Rebuild one processed snapshot solely from a hash-verified raw file."""
    artifact = load_raw_json_snapshot(raw_ref)
    result = decode_artifact(artifact)
    frame = to_canonical_frame(result)
    return save_processed_snapshot(
        frame,
        raw_ref,
        data_root,
        symbol=artifact["product"],
        interval="1d",
        provider="coinbase",
        missing_days_utc=list(result.missing_days_utc),
        missing_day_count=result.missing_day_count,
        **VOLUME_UNIT_FIELDS,
    )


def publish_run(
    artifacts: Mapping[str, dict], data_root: Path
) -> dict[str, tuple[RawJsonSnapshotRef, ProcessedSnapshotRef]]:
    """Validate then publish a set of Coinbase artifacts and their derivatives.

    Phase 1 performs no writes and runs the exact serialized-artifact decoding
    and canonicalization path for every product.  Thus any validation failure
    leaves every product absent from disk.  Phase 2/3 filesystem failures may
    leave earlier per-artifact publications in place; the snapshot primitives
    are immutable exclusive-create operations, not a global transaction.
    """
    for key, artifact in artifacts.items():
        if artifact.get("product") != key:
            raise ValueError(
                f"artifact mapping key {key!r} does not match artifact product "
                f"{artifact.get('product')!r}"
            )
        serialized = serialize_raw_artifact(artifact)
        round_tripped = json.loads(serialized)
        result = decode_artifact(round_tripped)
        to_canonical_frame(result)

    raw_refs: dict[str, RawJsonSnapshotRef] = {}
    for product, artifact in artifacts.items():
        raw_refs[product] = save_raw_json_snapshot(artifact, data_root)

    published: dict[str, tuple[RawJsonSnapshotRef, ProcessedSnapshotRef]] = {}
    for product, raw_ref in raw_refs.items():
        published[product] = (raw_ref, build_processed_from_raw(raw_ref, data_root))
    return published
