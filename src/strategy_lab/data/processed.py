"""Immutable processed (canonical) snapshots, derived from a raw snapshot.

Mirrors ``strategy_lab.data.snapshot`` for the canonical side: Parquet +
JSON metadata + SHA-256, never overwritten. Processed metadata records the
exact raw snapshot (by path and checksum) it was derived from, so lineage
from canonical data back to the original provider download is always
traceable.

Processed artifact identity
----------------------------
A processed artifact's on-disk identity (filename) is independent of the
source raw snapshot's identity: it is derived from *this artifact's own*
creation time (UTC, microsecond precision) and *this artifact's own*
Parquet SHA-256. This means two different processed derivations of the
same raw snapshot (e.g. before and after a canonicalization bugfix) can
coexist immutably rather than colliding on a shared filename. The source
raw SHA-256 remains recorded in metadata as lineage, never as the
processed artifact's own identity.

The Parquet file is first written to an OS-generated, exclusively-created
temporary file in the same directory (``tempfile.mkstemp``, not a
timestamp/PID-derived name -- unique regardless of clock resolution, PID
reuse, or thread scheduling, so two same-process concurrent writers cannot
collide at the temp-file layer), then published to its final, hash-derived
name once its checksum is known. Publication is atomically no-clobber: the
final destination is claimed via exclusive creation (``open(..., "xb")``,
i.e. ``O_CREAT|O_EXCL``), never a plain write that could silently
overwrite a concurrently-created file at the same path. The metadata
sidecar is published the same way (``Path.open("x", ...)``). The temporary
Parquet file is always removed afterward, whether publication succeeds or
fails.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow

from strategy_lab.data.raw_artifact import RawArtifactRef
from strategy_lab.data.snapshot import compute_sha256

SCHEMA_VERSION = 1

_ALLOWED_PROVIDERS: frozenset[str] = frozenset({"coinbase"})
_PROVIDER_TIMESTAMP_SEMANTICS: dict[str, str] = {"coinbase": "utc_day_bucket_start"}
_LEGACY_TIMESTAMP_SEMANTICS = "exchange_session_date_naive"
_COINBASE_REQUIRED_KWARGS = (
    "volume_unit_status",
    "volume_unit_inference",
    "volume_unit_evidence",
    "missing_days_utc",
    "missing_day_count",
)


@dataclass(frozen=True)
class ProcessedSnapshotRef:
    parquet_path: Path
    metadata_path: Path
    sha256: str


def processed_snapshot_dir(
    data_root: Path, symbol: str, interval: str, provider: str | None = None
) -> Path:
    if provider is not None and provider not in _ALLOWED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; allowed: {sorted(_ALLOWED_PROVIDERS)}"
        )
    if provider is None:
        return data_root / "processed" / symbol / interval
    return data_root / "processed" / provider / symbol / interval


def _processed_stem(created_at_utc: datetime, processed_sha256: str) -> str:
    timestamp_part = created_at_utc.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp_part}-{processed_sha256[:16]}"


def _publish_file_exclusive(tmp_path: Path, final_path: Path) -> None:
    """Copy ``tmp_path``'s exact bytes to ``final_path``, claiming
    ``final_path`` exclusively so an existing file there is never
    overwritten.

    Uses ``open(..., "xb")`` (``O_CREAT|O_EXCL``) rather than a hard link
    so this works regardless of whether the destination filesystem
    supports hard links (some external/network filesystems do not).
    Raises ``FileExistsError`` if ``final_path`` already exists.
    """
    with open(final_path, "xb") as dst, tmp_path.open("rb") as src:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def save_processed_snapshot(
    canonical_df: pd.DataFrame,
    source_raw_ref: RawArtifactRef,
    data_root: Path,
    symbol: str,
    interval: str,
    *,
    provider: str | None = None,
    volume_unit: str | None = None,
    volume_unit_status: str | None = None,
    volume_unit_inference: str | None = None,
    volume_unit_evidence: str | None = None,
    missing_days_utc: Sequence[str] | None = None,
    missing_day_count: int | None = None,
) -> ProcessedSnapshotRef:
    """Write ``canonical_df`` as a new, immutable processed snapshot.

    Named after this artifact's own creation timestamp and its own Parquet
    SHA-256 -- not the source raw snapshot's checksum -- so a second,
    independently-produced derivation from the same raw snapshot never
    collides with an earlier one. Lineage back to the raw snapshot is
    recorded in metadata (``source_raw_file``, ``source_raw_sha256``), not
    encoded in the filename.

    Publication of both the Parquet file and its metadata sidecar is
    atomically no-clobber (see module docstring): a destination that
    already exists -- from an earlier run or a genuine concurrent writer
    -- is never overwritten; this raises ``FileExistsError`` instead.
    With ``provider=None`` (the default), behavior and output remain
    byte-identical to the legacy path. With ``provider="coinbase"``, output
    is namespaced under ``processed/coinbase/<symbol>/<interval>/`` and
    metadata gains provider, timezone, volume-unit, and gap fields, all
    required together (``ValueError`` if any is missing). Coinbase-only
    arguments with the default provider also raise ``ValueError``.
    """
    _validate_provider_arguments(
        provider=provider,
        volume_unit=volume_unit,
        volume_unit_status=volume_unit_status,
        volume_unit_inference=volume_unit_inference,
        volume_unit_evidence=volume_unit_evidence,
        missing_days_utc=missing_days_utc,
        missing_day_count=missing_day_count,
    )
    directory = processed_snapshot_dir(data_root, symbol, interval, provider=provider)
    directory.mkdir(parents=True, exist_ok=True)

    created_at_utc = datetime.now(UTC)

    tmp_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".parquet")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        canonical_df.to_parquet(tmp_path)
        sha256 = compute_sha256(tmp_path)

        stem = _processed_stem(created_at_utc, sha256)
        parquet_path = directory / f"{stem}.canonical.parquet"
        metadata_path = directory / f"{stem}.meta.json"

        try:
            _publish_file_exclusive(tmp_path, parquet_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing immutable processed snapshot at "
                f"{parquet_path}"
            ) from exc

        published_sha256 = compute_sha256(parquet_path)
        if published_sha256 != sha256:
            raise RuntimeError(
                f"published processed artifact {parquet_path} does not match its "
                f"computed checksum ({published_sha256} != {sha256}); possible "
                "corruption during publication"
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "processed",
        "created_at_utc": created_at_utc.isoformat(),
        "source_raw_file": str(source_raw_ref.artifact_path.relative_to(data_root)),
        "source_raw_sha256": source_raw_ref.sha256,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "timestamp_semantics": (
            _LEGACY_TIMESTAMP_SEMANTICS
            if provider is None
            else _PROVIDER_TIMESTAMP_SEMANTICS[provider]
        ),
        "symbol": symbol,
        "interval": interval,
        "columns": [str(c) for c in canonical_df.columns],
        "row_count": int(len(canonical_df)),
        "first_timestamp": _timestamp_or_none(canonical_df["timestamp"].min()),
        "last_timestamp": _timestamp_or_none(canonical_df["timestamp"].max()),
        "sha256": sha256,
        "file": str(parquet_path.relative_to(data_root)),
    }
    if provider is not None:
        metadata["provider"] = provider
        metadata["timezone"] = "UTC"
        metadata["volume_unit"] = volume_unit
        metadata["volume_unit_status"] = volume_unit_status
        metadata["volume_unit_inference"] = volume_unit_inference
        metadata["volume_unit_evidence"] = volume_unit_evidence
        metadata["missing_days_utc"] = list(missing_days_utc)
        metadata["missing_day_count"] = missing_day_count
    # Exclusive metadata publication: never overwrite an existing sidecar.
    # If this raises, the Parquet artifact published just above is left in
    # place untouched, as an orphan with no metadata sidecar (e.g. after a
    # crash, or a metadata-identity collision, between the two writes).
    # Nothing here is known to have existed before this call, so there is
    # nothing to delete or repair -- but this is NOT a silently-tolerated
    # state: `_find_latest_processed_snapshot` and any other selector that
    # relies on metadata fails loudly the moment it encounters such an
    # orphan, rather than skipping it or accepting it as a valid, complete
    # snapshot. Resolving an orphan (deleting it, or manually reconciling
    # it) is an operational step, not something this function attempts on
    # its own.
    try:
        with metadata_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=2, sort_keys=True))
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing processed snapshot metadata at "
            f"{metadata_path}"
        ) from exc

    return ProcessedSnapshotRef(
        parquet_path=parquet_path, metadata_path=metadata_path, sha256=sha256
    )


def load_processed_snapshot(ref: ProcessedSnapshotRef) -> pd.DataFrame:
    return pd.read_parquet(ref.parquet_path)


def _timestamp_or_none(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _validate_provider_arguments(
    *,
    provider: str | None,
    volume_unit: str | None,
    volume_unit_status: str | None,
    volume_unit_inference: str | None,
    volume_unit_evidence: str | None,
    missing_days_utc: Sequence[str] | None,
    missing_day_count: int | None,
) -> None:
    """Validate provider-specific processed snapshot metadata arguments."""
    provider_arguments = {
        "volume_unit": volume_unit,
        "volume_unit_status": volume_unit_status,
        "volume_unit_inference": volume_unit_inference,
        "volume_unit_evidence": volume_unit_evidence,
        "missing_days_utc": missing_days_utc,
        "missing_day_count": missing_day_count,
    }
    if provider is None:
        for name, value in provider_arguments.items():
            if value is not None:
                raise ValueError(f"{name} is only valid with provider='coinbase'")
        return

    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; allowed: {sorted(_ALLOWED_PROVIDERS)}"
        )

    for name in _COINBASE_REQUIRED_KWARGS:
        if provider_arguments[name] is None:
            raise ValueError(f"{name} is required with provider='coinbase'")

    if type(missing_day_count) is not int or missing_day_count < 0:
        raise ValueError(
            "missing_day_count must be a non-negative int; "
            f"got {missing_day_count!r}"
        )

    if isinstance(missing_days_utc, str) or not isinstance(missing_days_utc, Sequence):
        raise ValueError(
            "missing_days_utc must be a sequence of YYYY-MM-DD strings; "
            f"got {missing_days_utc!r}"
        )
    for missing_day in missing_days_utc:
        if not isinstance(missing_day, str):
            raise ValueError(
                "missing_days_utc contains an invalid date format; "
                f"got {missing_day!r}"
            )
        try:
            parsed_missing_day = datetime.strptime(missing_day, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                "missing_days_utc contains an invalid date format; "
                f"got {missing_day!r}"
            ) from exc
        if parsed_missing_day.strftime("%Y-%m-%d") != missing_day:
            raise ValueError(
                "missing_days_utc contains an invalid date format; "
                f"got {missing_day!r}"
            )
    if len(missing_days_utc) != len(set(missing_days_utc)):
        raise ValueError("missing_days_utc contains a duplicate date found")
    if list(missing_days_utc) != sorted(missing_days_utc):
        raise ValueError("missing_days_utc is not in strictly ascending order")
    if len(missing_days_utc) != missing_day_count:
        raise ValueError(
            "missing_days_utc length does not match missing_day_count: "
            f"{len(missing_days_utc)} != {missing_day_count}"
        )

    for name in (
        "volume_unit_status",
        "volume_unit_inference",
        "volume_unit_evidence",
    ):
        value = provider_arguments[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    if volume_unit is not None and (not isinstance(volume_unit, str) or not volume_unit):
        raise ValueError(f"volume_unit must be None or a non-empty string; got {volume_unit!r}")
