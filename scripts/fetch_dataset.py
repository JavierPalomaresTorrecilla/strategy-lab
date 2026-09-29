#!/usr/bin/env python3
"""Human-run script: download, snapshot, and canonicalize the fixed
Milestone 2 dataset (SPY, 1d, 2010-01-01 to 2026-01-01 exclusive).

This script owns the fixed dataset choice; the reusable provider
(``strategy_lab.data.providers.yfinance_provider``) does not hard-code any
symbol, interval, or date range.

Not run automatically by tests or by any library code. Requires network
access and a valid ``STRATEGY_LAB_DATA_ROOT``.
"""

from __future__ import annotations

import sys

from strategy_lab.data.canonical import to_canonical_ohlcv
from strategy_lab.data.env import DataRootError, resolve_data_root
from strategy_lab.data.processed import save_processed_snapshot
from strategy_lab.data.providers.yfinance_provider import YFinanceProvider
from strategy_lab.data.snapshot import save_raw_snapshot

SYMBOL = "SPY"
INTERVAL = "1d"
START = "2010-01-01"
END = "2026-01-01"  # exclusive


def main() -> int:
    try:
        data_root = resolve_data_root()
    except DataRootError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"data root: {data_root}")
    print(f"fetching {SYMBOL} {INTERVAL} [{START}, {END}) from yfinance ...")

    provider = YFinanceProvider()
    raw_result = provider.fetch(symbol=SYMBOL, interval=INTERVAL, start=START, end=END)
    print(f"received {len(raw_result.dataframe)} rows")
    print(f"columns: {list(raw_result.dataframe.columns)}")

    raw_ref = save_raw_snapshot(raw_result, data_root)
    print(f"raw snapshot:       {raw_ref.parquet_path}")
    print(f"raw metadata:       {raw_ref.metadata_path}")
    print(f"raw sha256:         {raw_ref.sha256}")

    canonical_df = to_canonical_ohlcv(raw_result.dataframe)
    processed_ref = save_processed_snapshot(
        canonical_df, raw_ref, data_root, symbol=SYMBOL, interval=INTERVAL
    )
    print(f"processed snapshot: {processed_ref.parquet_path}")
    print(f"processed metadata: {processed_ref.metadata_path}")
    print(f"processed sha256:   {processed_ref.sha256}")

    print(f"canonical rows: {len(canonical_df)}")
    print(f"first timestamp: {canonical_df['timestamp'].min()}")
    print(f"last timestamp:  {canonical_df['timestamp'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
