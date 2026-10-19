#!/usr/bin/env python3
"""Offline comparison of two verified Coinbase raw JSON snapshots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strategy_lab.data.env import resolve_data_root
from strategy_lab.data.providers.coinbase_raw import (
    RawArtifactError,
    RawJsonSnapshotRef,
    load_raw_json_snapshot,
)
from strategy_lab.data.providers.coinbase_revision import (
    ArtifactPairMismatchError,
    compare_raw_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class PathSafetyError(ValueError):
    """Raised for an artifact path that escapes the configured data root."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_safe_data_root(data_root: Path) -> Path:
    resolved = data_root.resolve()
    if _is_within(resolved, REPOSITORY_ROOT.resolve()):
        raise PathSafetyError(
            "refusing to write Coinbase revision output inside the repository working tree: "
            f"{resolved}"
        )
    return resolved


def _diagnostics_directory(data_root: Path) -> Path:
    directory = (data_root / "diagnostics" / "coinbase_revision").resolve()
    if _is_within(directory, REPOSITORY_ROOT.resolve()):
        raise PathSafetyError(
            "refusing to write Coinbase revision diagnostics inside the repository working tree: "
            f"{directory}"
        )
    return directory


def _safe_artifact_path(data_root: Path, value: str) -> tuple[Path, str]:
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise PathSafetyError(
            f"artifact path must be a traversal-free data-root-relative path: {value}"
        )
    resolved = (data_root / supplied).resolve()
    if resolved == data_root or not _is_within(resolved, data_root):
        raise PathSafetyError(f"artifact path escapes configured data root: {value}")
    return resolved, str(supplied)


def _snapshot_ref(json_path: Path) -> RawJsonSnapshotRef:
    metadata_path = json_path.with_suffix(".meta.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawArtifactError(f"cannot read raw JSON metadata at {metadata_path}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
        raise RawArtifactError(f"raw JSON metadata has no valid sha256 at {metadata_path}")
    return RawJsonSnapshotRef(
        json_path=json_path, metadata_path=metadata_path, sha256=metadata["sha256"]
    )


def _write_report(directory: Path, product: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"{timestamp}.{product}.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        try:
            with destination.open("xb") as published, temporary_path.open("rb") as source:
                shutil.copyfileobj(source, published)
                published.flush()
                os.fsync(published.fileno())
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing Coinbase revision report at {destination}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="raw JSON path relative to data root")
    parser.add_argument("--audit", required=True, help="raw JSON path relative to data root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data_root = _ensure_safe_data_root(resolve_data_root())
        baseline_path, baseline_relative = _safe_artifact_path(data_root, args.baseline)
        audit_path, audit_relative = _safe_artifact_path(data_root, args.audit)

        baseline_ref = _snapshot_ref(baseline_path)
        audit_ref = _snapshot_ref(audit_path)
        baseline = load_raw_json_snapshot(baseline_ref)
        audit = load_raw_json_snapshot(audit_ref)
        report = compare_raw_artifacts(
            baseline,
            audit,
            baseline_path=baseline_relative,
            baseline_sha256=baseline_ref.sha256,
            audit_path=audit_relative,
            audit_sha256=audit_ref.sha256,
        )
        report_path = _write_report(_diagnostics_directory(data_root), report["product"], report)
    except (PathSafetyError, ArtifactPairMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = report["counts"]["total"]
    print(
        f"{report['verdict']}: {total['differing_days']} differing days, "
        f"{len(report['difference_records'])} records ({report_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
