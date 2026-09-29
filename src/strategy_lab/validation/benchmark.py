"""Price-only buy-and-hold benchmark -- the single M3 benchmark.

Deliberately independent of ``strategy_lab.backtest.engine.run_backtest``:
a buy-and-hold position is a single entry with no subsequent decisions, so
manufacturing it by driving the full backtest engine through a
never-changing signal would add indirection without adding correctness.
Instead this module applies the *same* buy-sizing formula the reference
engine uses (see ``strategy_lab.backtest.engine.run_backtest``'s buy
branch) directly, once, and is regression-tested against it.

No Adj Close, no total-return proxy: this benchmark uses the exact same
canonical raw ``open``/``close`` series the strategy trades. See the
required-labeling constants below -- every report showing this benchmark
must carry them verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.validation.temporal_validation import filter_to_allowed_research_history

BENCHMARK_LABEL = "price-only; dividends excluded"
STRATEGY_DIVIDEND_DISCLAIMER = (
    "dividend cash flows are not credited by the current reference engine"
)
INITIALIZATION_ASYMMETRY_NOTE = (
    "benchmark entry begins fresh at the first 2011 open; strategy may already hold a "
    "burn-in-initiated position -- entry timing is not identical between the two legs"
)
PERMITTED_CLAIM = (
    "comparison on the same raw price series, same baseline execution-cost assumptions"
)

# Explicitly prohibited claims (documented, not enforced in code beyond the
# labeling above): economic total-return outperformance, dividend-inclusive
# performance, total-return-benchmark outperformance, or any framing that
# calls this comparison economically complete or fully symmetric.


class BenchmarkConstructionError(ValueError):
    """Raised when the benchmark cannot be constructed from the given
    canonical OHLCV (e.g. no rows at/after the evaluation start, or none
    before the evaluation end)."""


@dataclass(frozen=True)
class BenchmarkResult:
    entry_timestamp: pd.Timestamp
    entry_open: float
    entry_exec_price: float
    quantity: float
    entry_commission: float
    final_timestamp: pd.Timestamp
    final_close: float
    final_value: float
    total_return: float
    label: str = BENCHMARK_LABEL
    dividend_disclaimer: str = STRATEGY_DIVIDEND_DISCLAIMER
    initialization_asymmetry: str = INITIALIZATION_ASYMMETRY_NOTE
    permitted_claim: str = PERMITTED_CLAIM


def run_price_only_benchmark(
    ohlcv: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> BenchmarkResult:
    """Buy at the first canonical ``open`` at/after ``evaluation_start``
    (2011), using the identical buy-sizing formula as
    ``run_backtest``'s buy branch; hold continuously; mark-to-market at the
    last canonical ``close`` strictly before ``evaluation_end`` (2023).

    No terminal sale is ever performed: the benchmark's final value is a
    valuation, not a realized exit trade, so no exit commission or
    slippage is ever applied to it.

    Defensively re-filters ``ohlcv`` to ``RESEARCH_ALLOWED_SPLITS`` itself
    before touching any row, rather than trusting the caller to have
    already done so.
    """
    ohlcv = filter_to_allowed_research_history(ohlcv)
    entry_candidates = ohlcv.loc[ohlcv["timestamp"] >= evaluation_start]
    if entry_candidates.empty:
        raise BenchmarkConstructionError(f"no canonical rows at/after {evaluation_start}")
    entry_row = entry_candidates.iloc[0]

    exit_candidates = ohlcv.loc[ohlcv["timestamp"] < evaluation_end]
    if exit_candidates.empty:
        raise BenchmarkConstructionError(f"no canonical rows before {evaluation_end}")
    exit_row = exit_candidates.iloc[-1]

    if exit_row["timestamp"] < entry_row["timestamp"]:
        raise BenchmarkConstructionError(
            f"resolved exit boundary ({exit_row['timestamp']}) precedes entry "
            f"boundary ({entry_row['timestamp']})"
        )

    open_price = float(entry_row["open"])
    exec_price = open_price * (1.0 + slippage_rate)
    quantity = initial_cash / (exec_price * (1.0 + commission_rate))
    commission = quantity * exec_price * commission_rate

    close_price = float(exit_row["close"])
    final_value = quantity * close_price
    total_return = final_value / initial_cash - 1.0

    return BenchmarkResult(
        entry_timestamp=entry_row["timestamp"],
        entry_open=open_price,
        entry_exec_price=exec_price,
        quantity=quantity,
        entry_commission=commission,
        final_timestamp=exit_row["timestamp"],
        final_close=close_price,
        final_value=final_value,
        total_return=total_return,
    )
