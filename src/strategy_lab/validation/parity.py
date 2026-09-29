"""Deterministic cross-engine parity comparison: reference engine vs. VectorBT.

The purpose here is parity validation, not optimization or a performance
claim. This module runs the same EMA crossover concept through both
``strategy_lab.backtest.engine`` (the reference engine, unmodified) and
``strategy_lab.validation.vectorbt_adapter`` (VectorBT, explicitly
configured to match the reference engine's next-open execution timing), and
reports where they agree and where they diverge — it never adjusts either
implementation to force numbers to match.

Tolerances
----------
Discrete facts (which bars trades occur on, how many trades, which side)
must match exactly between the two engines — any difference there indicates
a real construction bug, not floating-point noise.

Continuous values (fill prices, final equity, total return) are compared
with a relative tolerance of ``1e-9``. This was chosen empirically: on
synthetic deterministic fixtures, the reference engine's
``quantity * price * (1 + commission_rate) == cash`` sizing formula and
VectorBT's internal sizing produced final equity and fill prices identical
to at least 12 significant digits, differing only in the last one or two
representable bits of a 64-bit float — the level of noise expected from two
independent implementations of the same arithmetic in a different order.
A tolerance this tight will still fail loudly on any genuine accounting
divergence, which would be many orders of magnitude larger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from strategy_lab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from strategy_lab.validation.vectorbt_adapter import VectorbtResult, run_vectorbt

PRICE_RELATIVE_TOLERANCE = 1e-9
EQUITY_RELATIVE_TOLERANCE = 1e-9
RETURN_ABSOLUTE_TOLERANCE = 1e-9


@dataclass
class ParityReport:
    """A deterministic, serializable comparison of the two engines' output
    on one (ohlcv, signal, config) run."""

    num_signal_transitions: int
    reference_entry_dates: list
    reference_exit_dates: list
    vectorbt_entry_dates: list
    vectorbt_exit_dates: list
    reference_prices: list[float]
    vectorbt_prices: list[float]
    reference_num_trades: int
    vectorbt_num_trades: int
    reference_final_equity: float
    vectorbt_final_equity: float
    reference_total_return: float
    vectorbt_total_return: float
    divergences: list[dict] = field(default_factory=list)

    def is_within_tolerance(self) -> bool:
        return len(self.divergences) == 0

    def to_dict(self) -> dict:
        return {
            "num_signal_transitions": self.num_signal_transitions,
            "reference_entry_dates": [str(d) for d in self.reference_entry_dates],
            "reference_exit_dates": [str(d) for d in self.reference_exit_dates],
            "vectorbt_entry_dates": [str(d) for d in self.vectorbt_entry_dates],
            "vectorbt_exit_dates": [str(d) for d in self.vectorbt_exit_dates],
            "reference_prices": self.reference_prices,
            "vectorbt_prices": self.vectorbt_prices,
            "reference_num_trades": self.reference_num_trades,
            "vectorbt_num_trades": self.vectorbt_num_trades,
            "reference_final_equity": self.reference_final_equity,
            "vectorbt_final_equity": self.vectorbt_final_equity,
            "reference_total_return": self.reference_total_return,
            "vectorbt_total_return": self.vectorbt_total_return,
            "divergences": self.divergences,
        }


def _relative_diff(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator


def compare_engines(
    ohlcv: pd.DataFrame, signal: pd.Series, config: BacktestConfig
) -> ParityReport:
    """Run both engines on the same inputs and build a ``ParityReport``.

    Does not modify either engine or its inputs to force agreement; any
    divergence beyond the documented tolerances is recorded, not hidden.
    """
    reference: BacktestResult = run_backtest(ohlcv, signal, config)
    vectorbt_result: VectorbtResult = run_vectorbt(ohlcv, signal, config)

    ref_entry_dates = [t.timestamp for t in reference.trades if t.side == "buy"]
    ref_exit_dates = [t.timestamp for t in reference.trades if t.side == "sell"]
    ref_entry_prices = [t.price for t in reference.trades if t.side == "buy"]
    ref_exit_prices = [t.price for t in reference.trades if t.side == "sell"]
    # Compare entry-to-entry and exit-to-exit, each in chronological order --
    # never concatenate across sides, since that silently depends on
    # matching interleaving conventions between the two engines' outputs.
    ref_prices = ref_entry_prices + ref_exit_prices
    vbt_prices = vectorbt_result.entry_prices + vectorbt_result.exit_prices

    num_transitions = int(
        (signal.astype(int).diff().fillna(0) != 0).sum()
    )

    divergences: list[dict] = []

    if ref_entry_dates != vectorbt_result.entry_timestamps:
        divergences.append(
            {
                "field": "entry_dates",
                "reference": [str(d) for d in ref_entry_dates],
                "vectorbt": [str(d) for d in vectorbt_result.entry_timestamps],
                "explanation": "entry dates differ exactly; likely a shift/adapter bug",
            }
        )
    if ref_exit_dates != vectorbt_result.exit_timestamps:
        divergences.append(
            {
                "field": "exit_dates",
                "reference": [str(d) for d in ref_exit_dates],
                "vectorbt": [str(d) for d in vectorbt_result.exit_timestamps],
                "explanation": "exit dates differ exactly; likely a shift/adapter bug",
            }
        )

    ref_num_completed_trades = reference.metrics["num_completed_trades"]
    if ref_num_completed_trades != vectorbt_result.num_trades:
        divergences.append(
            {
                "field": "num_trades",
                "reference": ref_num_completed_trades,
                "vectorbt": vectorbt_result.num_trades,
                "explanation": "completed trade counts differ exactly",
            }
        )

    if len(ref_prices) == len(vbt_prices):
        for i, (rp, vp) in enumerate(zip(ref_prices, vbt_prices, strict=True)):
            if _relative_diff(rp, vp) > PRICE_RELATIVE_TOLERANCE:
                divergences.append(
                    {
                        "field": f"execution_price[{i}]",
                        "reference": rp,
                        "vectorbt": vp,
                        "explanation": (
                            f"relative difference exceeds tolerance "
                            f"({PRICE_RELATIVE_TOLERANCE:g})"
                        ),
                    }
                )
    else:
        divergences.append(
            {
                "field": "execution_price_count",
                "reference": len(ref_prices),
                "vectorbt": len(vbt_prices),
                "explanation": "different number of executed trades; prices not comparable",
            }
        )

    if _relative_diff(reference.final_equity, vectorbt_result.final_equity) > (
        EQUITY_RELATIVE_TOLERANCE
    ):
        divergences.append(
            {
                "field": "final_equity",
                "reference": reference.final_equity,
                "vectorbt": vectorbt_result.final_equity,
                "explanation": (
                    f"relative difference exceeds tolerance ({EQUITY_RELATIVE_TOLERANCE:g})"
                ),
            }
        )

    ref_total_return = reference.metrics["total_return"]
    if not math.isclose(
        ref_total_return, vectorbt_result.total_return, abs_tol=RETURN_ABSOLUTE_TOLERANCE
    ):
        divergences.append(
            {
                "field": "total_return",
                "reference": ref_total_return,
                "vectorbt": vectorbt_result.total_return,
                "explanation": (
                    f"absolute difference exceeds tolerance ({RETURN_ABSOLUTE_TOLERANCE:g})"
                ),
            }
        )

    return ParityReport(
        num_signal_transitions=num_transitions,
        reference_entry_dates=ref_entry_dates,
        reference_exit_dates=ref_exit_dates,
        vectorbt_entry_dates=vectorbt_result.entry_timestamps,
        vectorbt_exit_dates=vectorbt_result.exit_timestamps,
        reference_prices=ref_prices,
        vectorbt_prices=vbt_prices,
        reference_num_trades=ref_num_completed_trades,
        vectorbt_num_trades=vectorbt_result.num_trades,
        reference_final_equity=reference.final_equity,
        vectorbt_final_equity=vectorbt_result.final_equity,
        reference_total_return=ref_total_return,
        vectorbt_total_return=vectorbt_result.total_return,
        divergences=divergences,
    )
