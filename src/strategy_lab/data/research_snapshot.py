"""Immutable, physically TEST-free research snapshots for M6 Coinbase data.

This module deliberately has no TEST-opening loader.  Its sole row-level
parent read is a PyArrow predicate read before the frozen M6 cutoff.  Parent
SHA-256 verification reads every parent-file byte, including physical TEST
bytes, only as an opaque byte stream: it never decodes Parquet rows or
creates a Python-addressable TEST record.  The native PyArrow predicate read
may likewise read/decompress TEST-containing storage internally for a
one-row-group parent, but excludes those rows from its Python-visible result.

Real ADA/DOT TEST rows must never become Python-addressable objects during \
research-artifact derivation or M6 research execution.

Deterministic byte-identical output is guaranteed only for identical source
bytes, identical derivation predicate, identical code/schema version,
identical serialization options, and identical relevant pandas/pyarrow
versions. It is NOT guaranteed across arbitrary future library version
changes. Semantic content and provenance remain independently verifiable even
if future library versions produce different parquet bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow

from strategy_lab.data.env import ensure_safe_data_root
from strategy_lab.data.snapshot import compute_sha256

SCHEMA_VERSION = 1
RESEARCH_CUTOFF = pd.Timestamp("2025-01-01")
TEST_ROW_GUARANTEE = (
    "Real ADA/DOT TEST rows must never become Python-addressable objects during "
    "research-artifact derivation or M6 research execution."
)


class ResearchSnapshotError(ValueError):
    """Raised when research-snapshot provenance or row invariants fail."""


@dataclass(frozen=True)
class ResearchSnapshotRef:
    """Reference to an immutable, physically TEST-free research artifact."""

    parquet_path: Path
    metadata_path: Path
    sha256: str
    research_half_open_end: str


@dataclass(frozen=True)
class _VerifiedProcessedParent:
    parquet_path: Path
    relative_path: str
    sha256: str
    provider: str
    product: str
    interval: str


def require_research_snapshot_ref(ref: object) -> ResearchSnapshotRef:
    """Require the distinct runtime research-artifact reference type."""
    if not isinstance(ref, ResearchSnapshotRef):
        raise TypeError("expected ResearchSnapshotRef")
    return ref


def read_research_rows(
    parent_parquet_path: Path,
    cutoff: pd.Timestamp,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Read only Python-visible rows strictly before ``cutoff`` via PyArrow."""
    result = pd.read_parquet(
        parent_parquet_path,
        engine="pyarrow",
        filters=[(timestamp_column, "<", cutoff)],
    )
    if timestamp_column not in result.columns:
        raise ResearchSnapshotError(f"missing timestamp column {timestamp_column!r}")
    timestamps = result[timestamp_column]
    if timestamps.dtype != "datetime64[ns]" or timestamps.dt.tz is not None:
        raise ResearchSnapshotError(
            f"{timestamp_column!r} must be tz-naive datetime64[ns], got {timestamps.dtype}"
        )
    if result.empty:
        raise ResearchSnapshotError("predicate read returned no research rows")
    if not (timestamps < cutoff).all() or not timestamps.max() < cutoff:
        raise ResearchSnapshotError("predicate read exposed a row at or after the cutoff")
    return result


def derive_research_snapshot(
    parent_parquet_path: Path,
    expected_parent_sha256: str,
    data_root: Path,
    *,
    cutoff: pd.Timestamp = RESEARCH_CUTOFF,
) -> ResearchSnapshotRef:
    """Verify a processed parent, predicate-read it, and publish research data.

    Verification precedes the only row-level read.  In particular, the
    parent checksum is intentionally calculated over opaque source bytes;
    that integrity check is not a Parquet decode and does not expose TEST
    records to Python.
    """
    safe_root = ensure_safe_data_root(data_root)
    verified_parent = _verify_processed_parent(
        parent_parquet_path, expected_parent_sha256, safe_root
    )
    rows = read_research_rows(verified_parent.parquet_path, cutoff, "timestamp")
    return _write_research_snapshot(rows, verified_parent, safe_root, cutoff)


def _verify_processed_parent(
    parent_parquet_path: Path,
    expected_parent_sha256: str,
    data_root: Path,
) -> _VerifiedProcessedParent:
    path = Path(parent_parquet_path)
    if path.is_absolute() or ".." in path.parts:
        raise ResearchSnapshotError("parent artifact path must be data-root-relative")
    resolved_parent = (data_root / path).resolve()
    try:
        relative_path = resolved_parent.relative_to(data_root)
    except ValueError as exc:
        raise ResearchSnapshotError("parent artifact path escapes the data root") from exc
    if not resolved_parent.is_file():
        raise ResearchSnapshotError(f"parent artifact does not exist: {relative_path}")
    metadata_path = _processed_metadata_path(resolved_parent)
    if not metadata_path.is_file():
        raise ResearchSnapshotError(f"parent metadata does not exist: {metadata_path}")

    actual_sha256 = compute_sha256(resolved_parent)
    if actual_sha256 != expected_parent_sha256:
        raise ResearchSnapshotError("parent SHA-256 does not match the expected SHA-256")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchSnapshotError("parent metadata is unreadable") from exc
    if not isinstance(metadata, dict):
        raise ResearchSnapshotError("parent metadata must be a JSON object")
    if metadata.get("kind") != "processed":
        raise ResearchSnapshotError("parent metadata is not a processed artifact")
    if metadata.get("sha256") != actual_sha256:
        raise ResearchSnapshotError("parent metadata SHA-256 disagrees with parent bytes")

    provider, product, interval = _validated_parent_identity(metadata, relative_path)
    return _VerifiedProcessedParent(
        parquet_path=resolved_parent,
        relative_path=str(relative_path),
        sha256=actual_sha256,
        provider=provider,
        product=product,
        interval=interval,
    )


def _processed_metadata_path(parent_path: Path) -> Path:
    suffix = ".canonical.parquet"
    if not parent_path.name.endswith(suffix):
        raise ResearchSnapshotError("parent artifact must end in .canonical.parquet")
    return parent_path.with_name(parent_path.name.removesuffix(suffix) + ".meta.json")


def _validated_parent_identity(
    metadata: dict[str, Any], relative_path: Path
) -> tuple[str, str, str]:
    expected_prefix = ("processed",)
    if relative_path.parts[:1] != expected_prefix or len(relative_path.parts) != 5:
        raise ResearchSnapshotError("parent path is not a canonical provider-processed location")
    _, path_provider, path_product, path_interval, _ = relative_path.parts
    provider = metadata.get("provider")
    product = metadata.get("symbol")
    interval = metadata.get("interval")
    if not all(isinstance(value, str) and value for value in (provider, product, interval)):
        raise ResearchSnapshotError("parent metadata lacks provider, symbol, or interval identity")
    if provider != "coinbase":
        raise ResearchSnapshotError("M6 research snapshots require a Coinbase processed parent")
    if (provider, product, interval) != (path_provider, path_product, path_interval):
        raise ResearchSnapshotError("parent metadata identity disagrees with canonical parent path")
    if metadata.get("file") != str(relative_path):
        raise ResearchSnapshotError("parent metadata file identity disagrees with parent path")
    return provider, product, interval


def _write_research_snapshot(
    rows: pd.DataFrame,
    parent: _VerifiedProcessedParent,
    data_root: Path,
    cutoff: pd.Timestamp,
) -> ResearchSnapshotRef:
    directory = ensure_safe_data_root(
        data_root / "research" / parent.provider / parent.product / parent.interval
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{parent.parquet_path.stem}__research-before-{cutoff.date().isoformat()}"
    parquet_path = directory / f"{stem}.parquet"
    metadata_path = directory / f"{stem}.parquet.meta.json"

    temporary_parquet = _temporary_path(directory, ".parquet")
    try:
        rows.reset_index(drop=True).to_parquet(
            temporary_parquet, engine="pyarrow", compression="snappy"
        )
        artifact_sha256 = compute_sha256(temporary_parquet)
        ref = require_research_snapshot_ref(
            ResearchSnapshotRef(
                parquet_path,
                metadata_path,
                artifact_sha256,
                cutoff.date().isoformat(),
            )
        )
        _publish_exclusive(temporary_parquet, ref.parquet_path)
        if compute_sha256(ref.parquet_path) != ref.sha256:
            raise RuntimeError("published research artifact checksum changed during publication")
    finally:
        temporary_parquet.unlink(missing_ok=True)

    metadata = _metadata(rows, parent, ref, cutoff)
    temporary_metadata = _temporary_path(directory, ".json")
    try:
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_exclusive(temporary_metadata, ref.metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    return require_research_snapshot_ref(ref)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _publish_exclusive(temporary_path: Path, final_path: Path) -> None:
    try:
        with final_path.open("xb") as destination, temporary_path.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable research artifact {final_path}"
        ) from exc


def _metadata(
    rows: pd.DataFrame,
    parent: _VerifiedProcessedParent,
    ref: ResearchSnapshotRef,
    cutoff: pd.Timestamp,
) -> dict[str, object]:
    timestamps = rows["timestamp"]
    return {
        "kind": "research_snapshot",
        "schema_version": SCHEMA_VERSION,
        "provider": parent.provider,
        "product": parent.product,
        "interval": parent.interval,
        "source_processed_file": parent.relative_path,
        "source_processed_sha256": parent.sha256,
        "derivation_predicate": {
            "column": "timestamp",
            "op": "<",
            "cutoff_utc_naive": cutoff.isoformat(),
        },
        "research_half_open_end": ref.research_half_open_end,
        "artifact_sha256": ref.sha256,
        "row_count": int(len(rows)),
        "first_timestamp": pd.Timestamp(timestamps.min()).isoformat(),
        "last_timestamp": pd.Timestamp(timestamps.max()).isoformat(),
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "parquet_engine": "pyarrow",
        "parquet_compression": "snappy",
        "test_row_guarantee": TEST_ROW_GUARANTEE,
    }
