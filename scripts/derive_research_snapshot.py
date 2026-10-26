#!/usr/bin/env python3
"""Offline derivation of a verified, physically TEST-free research snapshot."""

from __future__ import annotations

import argparse
import json
import sys

from strategy_lab.data.env import DataRootError, resolve_data_root
from strategy_lab.data.research_snapshot import (
    RESEARCH_CUTOFF,
    TEST_ROW_GUARANTEE,
    ResearchSnapshotError,
    derive_research_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        required=True,
        help="processed parent path relative to STRATEGY_LAB_DATA_ROOT",
    )
    parser.add_argument("--expected-sha256", required=True, help="verified parent SHA-256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        ref = derive_research_snapshot(
            args.parent,
            args.expected_sha256,
            resolve_data_root(),
            cutoff=RESEARCH_CUTOFF,
        )
        metadata = ref.metadata_path.read_text(encoding="utf-8")
    except (DataRootError, ResearchSnapshotError, FileExistsError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    details = json.loads(metadata)
    print(f"artifact path: {ref.parquet_path}")
    print(f"artifact SHA-256: {ref.sha256}")
    print(f"row_count: {details['row_count']}")
    print(f"first_timestamp: {details['first_timestamp']}")
    print(f"last_timestamp: {details['last_timestamp']}")
    print(f"source parent SHA-256: {details['source_processed_sha256']}")
    print(f"cutoff used: {RESEARCH_CUTOFF.isoformat()}")
    print(f"pandas version: {details['pandas_version']}")
    print(f"pyarrow version: {details['pyarrow_version']}")
    print(TEST_ROW_GUARANTEE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
