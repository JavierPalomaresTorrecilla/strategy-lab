"""Corporate-action runtime guard for M3 real-data reports.

Raw yfinance snapshots preserve ``Stock Splits``/``Dividends`` alongside
raw OHLC (see ``strategy_lab.data.canonical``'s docstring); canonical
execution prices intentionally drop them. This module checks, at report
run time against the *actual* fetched history, whether that's safe: a
non-zero stock split inside the allowed research range means the current
raw-OHLC-based execution path has not been verified to handle it, and the
report must fail rather than silently assume correctness.

Inspection is strictly bounded to ``[RESEARCH_RANGE_START,
RESEARCH_RANGE_END)`` -- rows at or after ``RESEARCH_RANGE_END`` (i.e. the
TEST partition) are never examined by this guard, even though the
underlying Parquet read may load the whole file into memory before this
module filters it (that's an acceptable storage-layer detail; the
requirement is that no *value* at or after 2023-01-01 ever influences the
guard's decision or the report metadata it produces).

Dividends are read here only as data-quality/context metadata -- never
credited to the strategy, never used to construct an adjusted execution
price.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from strategy_lab.data.processed import ProcessedSnapshotRef
from strategy_lab.data.snapshot import RawSnapshotRef, load_raw_snapshot

RESEARCH_RANGE_START = pd.Timestamp("2010-01-01")
RESEARCH_RANGE_END = pd.Timestamp("2023-01-01")  # exclusive; TEST starts here


class UnsupportedCorporateActionError(RuntimeError):
    """Raised when a non-zero stock-split event is found inside the
    allowed research range -- current raw-OHLC execution semantics have
    not been verified to handle this, so the report must not proceed."""


@dataclass(frozen=True)
class CorporateActionCheckResult:
    stock_splits_verified_zero_in_research_range: bool
    dividend_event_count_in_research_range: int
    dividend_date_range: tuple[pd.Timestamp, pd.Timestamp] | None


def resolve_raw_ref_from_processed(
    processed_ref: ProcessedSnapshotRef, data_root: Path
) -> RawSnapshotRef:
    """Reconstruct a ``RawSnapshotRef`` for the raw snapshot a processed
    snapshot was derived from, using the lineage already recorded in its
    metadata (``source_raw_file``/``source_raw_sha256``)."""
    metadata = json.loads(processed_ref.metadata_path.read_text())
    raw_parquet_path = data_root / metadata["source_raw_file"]
    raw_metadata_path = raw_parquet_path.with_suffix("").with_suffix(".meta.json")
    return RawSnapshotRef(
        parquet_path=raw_parquet_path,
        metadata_path=raw_metadata_path,
        sha256=metadata["source_raw_sha256"],
    )


def check_corporate_actions(raw_ref: RawSnapshotRef) -> CorporateActionCheckResult:
    """Load ``raw_ref``, restrict immediately to the allowed research
    range, and verify no non-zero ``Stock Splits`` event exists in it.

    Raises ``UnsupportedCorporateActionError`` on any non-zero split found
    inside the research range. Records a dividend-event count/date-range
    as informational metadata only.
    """
    raw_df = load_raw_snapshot(raw_ref)
    in_range = raw_df.loc[
        (raw_df.index >= RESEARCH_RANGE_START) & (raw_df.index < RESEARCH_RANGE_END)
    ]

    if "Stock Splits" not in in_range.columns:
        raise UnsupportedCorporateActionError(
            "raw snapshot is missing the 'Stock Splits' column required for this guard"
        )
    non_zero_splits = in_range.loc[in_range["Stock Splits"] != 0.0]
    if not non_zero_splits.empty:
        raise UnsupportedCorporateActionError(
            "non-zero Stock Splits event(s) found inside the allowed research range "
            f"[{RESEARCH_RANGE_START}, {RESEARCH_RANGE_END}): "
            f"{non_zero_splits['Stock Splits'].to_dict()}"
        )

    dividend_count = 0
    dividend_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if "Dividends" in in_range.columns:
        dividend_rows = in_range.loc[in_range["Dividends"] != 0.0]
        dividend_count = int(len(dividend_rows))
        if dividend_count:
            dividend_range = (dividend_rows.index.min(), dividend_rows.index.max())

    return CorporateActionCheckResult(
        stock_splits_verified_zero_in_research_range=True,
        dividend_event_count_in_research_range=dividend_count,
        dividend_date_range=dividend_range,
    )
