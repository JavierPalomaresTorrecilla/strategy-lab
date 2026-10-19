"""Offline comparison of two immutable Coinbase daily-candle snapshots.

This module deliberately operates only on already-loaded raw artifacts.  It
does not know about the Coinbase client or any network transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from strategy_lab.data.providers.coinbase_canonical import CoinbaseCandle, parse_candle

_CANDLE_FIELDS = ("start", "low", "high", "open", "close", "volume")
_NUMERIC_FIELDS = frozenset({"low", "high", "open", "close", "volume"})
_DIFFERENCE_KINDS = ("value", "format_only", "baseline_only", "audit_only")
_DAY = timedelta(days=1)


class CoinbaseRevisionError(ValueError):
    """Raised when two raw artifacts cannot be compared safely."""


class ArtifactPairMismatchError(CoinbaseRevisionError):
    """Raised when the two otherwise-valid artifacts describe different series."""


@dataclass(frozen=True)
class _RawCandle:
    """A raw candle paired with the canonical parser's normalized bucket."""

    epoch_seconds: int
    day: date
    raw: dict[str, str]
    parsed: CoinbaseCandle


def _bucket_for_age(age_days: int) -> str:
    if age_days == 1:
        return "1"
    if 2 <= age_days <= 7:
        return "2-7"
    if 8 <= age_days <= 30:
        return "8-30"
    if age_days > 30:
        return ">30"
    raise CoinbaseRevisionError(f"candle day has invalid non-positive age: {age_days}")


def _decimal_value(candle: _RawCandle, field: str) -> Decimal:
    """Return the Decimal already validated by :func:`parse_candle`."""
    return getattr(candle.parsed, field)


def _collect_candles(artifact: dict[str, Any], label: str) -> dict[int, _RawCandle]:
    """Flatten pages, reusing the canonical parser and rejecting duplicates."""
    candles: dict[int, _RawCandle] = {}
    for page in artifact["pages"]:
        for raw_candle in page["candles"]:
            # Raw-artifact validation guarantees the dict/string shape.  The
            # canonical parser is intentionally still the sole authority for
            # start normalization and Decimal validation.
            parsed = parse_candle(raw_candle)
            epoch_seconds = int(raw_candle["start"])
            if epoch_seconds in candles:
                raise CoinbaseRevisionError(
                    f"{label} contains duplicate normalized candle start {epoch_seconds}"
                )
            raw = dict(raw_candle)
            candles[epoch_seconds] = _RawCandle(
                epoch_seconds=epoch_seconds,
                day=parsed.bucket_start.date(),
                raw=raw,
                parsed=parsed,
            )
    return candles


def _parse_retrieval_day(artifact: dict[str, Any]) -> date:
    """Read the server-time provenance field frozen in the raw artifact."""
    value = artifact["retrieval"]["started_at_utc"]
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    return parsed.astimezone(UTC).date()


def _empty_count() -> dict[str, Any]:
    return {
        "days_evaluated": 0,
        "matched_days": 0,
        "identical_days": 0,
        "differing_days": 0,
        "semantic_differing_days": 0,
        "difference_records_by_kind": {kind: 0 for kind in _DIFFERENCE_KINDS},
    }


def _record(
    day: date,
    reference_day: date,
    kind: str,
    field: str | None,
    old: object,
    new: object,
) -> dict[str, Any]:
    age_days = (reference_day - day).days
    return {
        "day": day.isoformat(),
        "age_days": age_days,
        "bucket": _bucket_for_age(age_days),
        "kind": kind,
        "field": field,
        "old": old,
        "new": new,
    }


def _verdict(records: list[dict[str, Any]]) -> str:
    semantic_days = {
        (record["day"], record["age_days"]) for record in records if record["kind"] != "format_only"
    }
    if not semantic_days:
        return "PASS_WITH_FORMAT_DRIFT" if records else "PASS"
    ages = {age_days for _, age_days in semantic_days}
    if any(age_days >= 8 for age_days in ages):
        return "STOP"
    if any(2 <= age_days <= 7 for age_days in ages):
        return "PASS_WITH_7D_EMBARGO"
    return "PASS_WITH_1D_EMBARGO"


def compare_raw_artifacts(
    baseline: dict[str, Any],
    audit: dict[str, Any],
    *,
    baseline_path: str,
    baseline_sha256: str,
    audit_path: str,
    audit_sha256: str,
) -> dict[str, Any]:
    """Compare two verified raw artifacts and return a deterministic report body.

    ``baseline_path`` and ``audit_path`` are intentionally supplied by the
    caller as data-root-relative paths, keeping the report portable and free
    of environment-specific absolute paths.
    """
    if baseline["product"] != audit["product"]:
        raise ArtifactPairMismatchError("baseline and audit products do not match")
    if baseline["interval"] != audit["interval"]:
        raise ArtifactPairMismatchError("baseline and audit intervals do not match")

    baseline_candles = _collect_candles(baseline, "baseline")
    audit_candles = _collect_candles(audit, "audit")
    if not baseline_candles:
        raise CoinbaseRevisionError("baseline contains no candles")

    reference_day = _parse_retrieval_day(baseline)
    expected_last_day = reference_day - _DAY
    baseline_last_day = max(candle.day for candle in baseline_candles.values())
    if baseline_last_day != expected_last_day:
        raise CoinbaseRevisionError(
            "baseline last candle bucket must equal reference day minus one day "
            f"({baseline_last_day.isoformat()} != {expected_last_day.isoformat()})"
        )

    audit_days_after_baseline = sum(
        candle.day > expected_last_day for candle in audit_candles.values()
    )
    in_range_days = [
        candle.day
        for candle in (*baseline_candles.values(), *audit_candles.values())
        if candle.day <= expected_last_day
    ]
    first_day = min(in_range_days)

    baseline_by_day = {candle.day: candle for candle in baseline_candles.values()}
    audit_by_day = {
        candle.day: candle for candle in audit_candles.values() if candle.day <= expected_last_day
    }
    counts = {bucket: _empty_count() for bucket in ("1", "2-7", "8-30", ">30")}
    counts["total"] = _empty_count()
    records: list[dict[str, Any]] = []

    current_day = first_day
    while current_day <= expected_last_day:
        age_days = (reference_day - current_day).days
        bucket = _bucket_for_age(age_days)
        bucket_counts = counts[bucket]
        for count in (bucket_counts, counts["total"]):
            count["days_evaluated"] += 1

        old_candle = baseline_by_day.get(current_day)
        new_candle = audit_by_day.get(current_day)
        day_records: list[dict[str, Any]] = []
        if old_candle is None and new_candle is not None:
            day_records.append(
                _record(current_day, reference_day, "audit_only", None, None, new_candle.raw)
            )
        elif old_candle is not None and new_candle is None:
            day_records.append(
                _record(current_day, reference_day, "baseline_only", None, old_candle.raw, None)
            )
        elif old_candle is not None and new_candle is not None:
            for count in (bucket_counts, counts["total"]):
                count["matched_days"] += 1
            for field in _CANDLE_FIELDS:
                old_value = old_candle.raw[field]
                new_value = new_candle.raw[field]
                if old_value == new_value:
                    continue
                kind = "format_only"
                if field in _NUMERIC_FIELDS and _decimal_value(old_candle, field) != _decimal_value(
                    new_candle, field
                ):
                    kind = "value"
                day_records.append(
                    _record(current_day, reference_day, kind, field, old_value, new_value)
                )

        if day_records:
            has_semantic_difference = any(record["kind"] != "format_only" for record in day_records)
            for count in (bucket_counts, counts["total"]):
                count["differing_days"] += 1
                if has_semantic_difference:
                    count["semantic_differing_days"] += 1
                for record in day_records:
                    count["difference_records_by_kind"][record["kind"]] += 1
            records.extend(day_records)
        elif old_candle is not None and new_candle is not None:
            for count in (bucket_counts, counts["total"]):
                count["identical_days"] += 1

        current_day += _DAY

    return {
        "schema_version": 1,
        "product": baseline["product"],
        "interval": baseline["interval"],
        "baseline_path": baseline_path,
        "baseline_sha256": baseline_sha256,
        "audit_path": audit_path,
        "audit_sha256": audit_sha256,
        "reference_day": reference_day.isoformat(),
        "comparison_range": {
            "start_day": first_day.isoformat(),
            "end_day": expected_last_day.isoformat(),
        },
        "counts": counts,
        "difference_records": records,
        "audit_days_after_baseline": audit_days_after_baseline,
        "verdict": _verdict(records),
        "raw_representation_stable": not any(record["kind"] == "format_only" for record in records),
    }
