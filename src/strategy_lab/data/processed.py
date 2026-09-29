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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow

from strategy_lab.data.snapshot import RawSnapshotRef, compute_sha256

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProcessedSnapshotRef:
    parquet_path: Path
    metadata_path: Path
    sha256: str


def processed_snapshot_dir(data_root: Path, symbol: str, interval: str) -> Path:
    return data_root / "processed" / symbol / interval


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
    source_raw_ref: RawSnapshotRef,
    data_root: Path,
    symbol: str,
    interval: str,
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
    """
    directory = processed_snapshot_dir(data_root, symbol, interval)
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
        "source_raw_file": str(source_raw_ref.parquet_path.relative_to(data_root)),
        "source_raw_sha256": source_raw_ref.sha256,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "timestamp_semantics": "exchange_session_date_naive",
        "symbol": symbol,
        "interval": interval,
        "columns": [str(c) for c in canonical_df.columns],
        "row_count": int(len(canonical_df)),
        "first_timestamp": _timestamp_or_none(canonical_df["timestamp"].min()),
        "last_timestamp": _timestamp_or_none(canonical_df["timestamp"].max()),
        "sha256": sha256,
        "file": str(parquet_path.relative_to(data_root)),
    }
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
