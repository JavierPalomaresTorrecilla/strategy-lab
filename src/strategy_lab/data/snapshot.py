"""Immutable raw provider snapshots.

Every successful download is frozen to disk under the external data root
(see ``strategy_lab.data.env``) as a Parquet file plus a JSON metadata
sidecar, identified by a SHA-256 checksum of the Parquet file's own bytes.

Raw snapshots are never overwritten or mutated in place: each save produces
a new, timestamp-named file. Re-downloading the same symbol/interval/range
later creates an additional snapshot rather than replacing the old one,
because provider data for recent bars can be revised between downloads.

The SHA-256 checksum's meaning is artifact integrity/identity ("this exact
file has not changed"), not a claim that two independent Parquet
serializations of the same logical DataFrame must be byte-identical.

Publication is atomically no-clobber: the Parquet file is first written to
an OS-generated, exclusively-created temporary file in the same directory
(``tempfile.mkstemp``, not a timestamp/PID-derived name -- unique
regardless of clock resolution, PID reuse, or thread scheduling, so two
same-process concurrent writers cannot collide at the temp-file layer),
then claimed at its final path via exclusive creation (``open(..., "xb")``,
i.e. ``O_CREAT|O_EXCL``) rather than a plain write that would silently
overwrite a concurrently-created file at the same path. The metadata
sidecar is published the same way (``Path.open("x", ...)``). A destination
that already exists -- whether from an earlier run or a genuine concurrent
writer -- is never overwritten; publication raises ``FileExistsError``
instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow

from strategy_lab.data.providers.base import RawFetchResult

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RawSnapshotRef:
    """Reference to one immutable raw snapshot on disk."""

    parquet_path: Path
    metadata_path: Path
    sha256: str


def compute_sha256(path: Path) -> str:
    """SHA-256 of a file's exact bytes on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_snapshot_dir(data_root: Path, provider: str, symbol: str, interval: str) -> Path:
    return data_root / "raw" / provider / symbol / interval


def raw_snapshot_stem(retrieved_at_utc) -> str:
    return retrieved_at_utc.strftime("%Y%m%dT%H%M%S%fZ")


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


def save_raw_snapshot(result: RawFetchResult, data_root: Path) -> RawSnapshotRef:
    """Write ``result`` as a new, immutable raw snapshot under ``data_root``.

    Publication is atomically no-clobber for both the Parquet file and its
    metadata sidecar (see module docstring): if the computed identity
    already exists at either path -- from an earlier run or a genuine
    concurrent writer -- this raises ``FileExistsError`` rather than
    silently overwriting it. The dataframe is first written to a
    same-directory temporary file so the final destination is only ever
    touched by the exclusive-create step; the temporary file is always
    removed afterward, whether publication succeeds or fails.
    """
    directory = raw_snapshot_dir(data_root, result.provider, result.symbol, result.interval)
    directory.mkdir(parents=True, exist_ok=True)

    stem = raw_snapshot_stem(result.retrieved_at_utc)
    parquet_path = directory / f"{stem}.parquet"
    metadata_path = directory / f"{stem}.meta.json"

    # An OS-generated exclusive temp file, not a timestamp/PID-derived name:
    # unique regardless of clock resolution, PID reuse, or thread
    # scheduling, so two same-process concurrent writers can never collide
    # at the temporary-file layer (only at the immutable final identity,
    # which is the intended collision point).
    tmp_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".parquet")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        result.dataframe.to_parquet(tmp_path)
        sha256 = compute_sha256(tmp_path)

        try:
            _publish_file_exclusive(tmp_path, parquet_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing immutable raw snapshot at {parquet_path}"
            ) from exc

        published_sha256 = compute_sha256(parquet_path)
        if published_sha256 != sha256:
            raise RuntimeError(
                f"published raw snapshot {parquet_path} does not match its computed "
                f"checksum ({published_sha256} != {sha256}); possible corruption "
                "during publication"
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "raw",
        "provider": result.provider,
        "provider_library_version": result.provider_library_version,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "symbol": result.symbol,
        "interval": result.interval,
        "requested_start": result.requested_start,
        "requested_end": result.requested_end,
        "retrieved_at_utc": result.retrieved_at_utc.isoformat(),
        "download_options": _json_safe(result.options_used),
        "row_count": int(len(result.dataframe)),
        "first_timestamp": _timestamp_or_none(result.dataframe.index.min()),
        "last_timestamp": _timestamp_or_none(result.dataframe.index.max()),
        "columns": [str(c) for c in result.dataframe.columns],
        "sha256": sha256,
        "file": str(parquet_path.relative_to(data_root)),
    }
    # Exclusive metadata publication: never overwrite an existing sidecar.
    # If this raises, the Parquet artifact published just above is left in
    # place untouched, as an orphan with no metadata sidecar (e.g. after a
    # crash, or a metadata-identity collision, between the two writes).
    # Nothing here is known to have existed before this call, so there is
    # nothing to delete or repair -- but this is NOT a silently-tolerated
    # state: any workflow that selects/loads snapshots via their metadata
    # (e.g. a "find the latest snapshot" selector) fails loudly the moment
    # it encounters such an orphan, rather than skipping it or accepting
    # it as a valid, complete snapshot. Resolving an orphan (deleting it,
    # or manually reconciling it) is an operational step, not something
    # this function attempts on its own.
    try:
        with metadata_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=2, sort_keys=True))
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing raw snapshot metadata at {metadata_path}"
        ) from exc

    return RawSnapshotRef(parquet_path=parquet_path, metadata_path=metadata_path, sha256=sha256)


def load_raw_snapshot(ref: RawSnapshotRef) -> pd.DataFrame:
    return pd.read_parquet(ref.parquet_path)


def _timestamp_or_none(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _json_safe(options: dict[str, object]) -> dict[str, object]:
    return dict(options)
