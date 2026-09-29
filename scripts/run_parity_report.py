#!/usr/bin/env python3
"""Human-run script: cross-engine (reference vs. VectorBT) parity report for
the fixed EMA 10/30 SPY strategy, on TRAIN and VALIDATION data only.

TEST is methodologically sealed for Milestone 2: this script only ever reads
the "train" and "validation" partitions out of ``split_ohlcv`` and never
constructs a signal, backtest, or parity comparison from "test" rows. No
real-data strategy result is computed on TEST here.

Requires the ``parity`` extra (VectorBT) installed and a processed SPY
snapshot already produced by ``scripts/fetch_dataset.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

from strategy_lab.backtest.engine import BacktestConfig
from strategy_lab.data.env import DataRootError, resolve_data_root
from strategy_lab.data.processed import ProcessedSnapshotRef, load_processed_snapshot
from strategy_lab.data.splits import RESEARCH_ALLOWED_SPLITS, load_splits, split_ohlcv
from strategy_lab.strategies.ema_crossover import generate_signals
from strategy_lab.validation.parity import compare_engines

SYMBOL = "SPY"
INTERVAL = "1d"
FAST_PERIOD = 10
SLOW_PERIOD = 30


def _find_latest_processed_snapshot(data_root) -> ProcessedSnapshotRef:
    """Return the processed snapshot with the latest ``created_at_utc``.

    Selection is by metadata timestamp, never by filename/hash lexical
    order (unrelated to recency) or filesystem mtime (not semantic
    provenance -- a copy, restore, or re-mounted external drive can change
    mtime without changing when the artifact was actually produced).
    """
    directory = data_root / "processed" / SYMBOL / INTERVAL
    candidates = sorted(directory.glob("*.canonical.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no processed snapshot found under {directory}; run scripts/fetch_dataset.py first"
        )

    best_ref: ProcessedSnapshotRef | None = None
    best_created_at = None
    for parquet_path in candidates:
        metadata_path = parquet_path.with_suffix("").with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"processed artifact {parquet_path} has no metadata sidecar at {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())

        created_at_raw = metadata.get("created_at_utc")
        if not created_at_raw:
            raise ValueError(
                f"processed artifact metadata {metadata_path} is missing `created_at_utc`"
            )
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError as exc:
            raise ValueError(
                f"processed artifact metadata {metadata_path} has an invalid "
                f"`created_at_utc` ({created_at_raw!r})"
            ) from exc

        if best_created_at is None or created_at > best_created_at:
            best_created_at = created_at
            best_ref = ProcessedSnapshotRef(
                parquet_path=parquet_path, metadata_path=metadata_path, sha256=metadata["sha256"]
            )

    assert best_ref is not None  # candidates is non-empty, loop always assigns
    return best_ref


def main() -> int:
    try:
        data_root = resolve_data_root()
    except DataRootError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    ref = _find_latest_processed_snapshot(data_root)
    canonical_df = load_processed_snapshot(ref)
    print(f"loaded processed snapshot: {ref.parquet_path} ({len(canonical_df)} rows)")

    splits = load_splits()
    partitions = split_ohlcv(canonical_df, splits)

    for name in RESEARCH_ALLOWED_SPLITS:
        partition_df = partitions[name]
        print(f"\n=== {name.upper()} ({len(partition_df)} rows) ===")
        if partition_df.empty:
            print("no rows in this partition; skipping")
            continue

        signal = generate_signals(partition_df["close"], FAST_PERIOD, SLOW_PERIOD)["signal"]
        config = BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)

        report = compare_engines(partition_df, signal, config)
        print(f"reference: trades={report.reference_num_trades} "
              f"final_equity={report.reference_final_equity:.2f} "
              f"total_return={report.reference_total_return:.4%}")
        print(f"vectorbt:  trades={report.vectorbt_num_trades} "
              f"final_equity={report.vectorbt_final_equity:.2f} "
              f"total_return={report.vectorbt_total_return:.4%}")
        print(f"divergences: {len(report.divergences)}")
        for d in report.divergences:
            print(f"  - {d}")

    # TEST is intentionally never loaded from `partitions` above.
    assert "test" not in RESEARCH_ALLOWED_SPLITS

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
