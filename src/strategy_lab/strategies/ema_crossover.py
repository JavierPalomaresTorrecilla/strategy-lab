"""Long-only EMA crossover strategy.

This is a deliberately simple strategy used to exercise the research
pipeline (data validation, indicators, backtest engine). It is not a claim
that EMA crossovers have predictive value.

Signal convention
------------------
- ``1`` = desired position is long (fully invested).
- ``0`` = desired position is cash (flat).

No short selling: the signal is never negative.

Signal calculation vs. execution timing
----------------------------------------
``generate_signals`` computes, for each bar ``T``, the desired position
*as of the close of bar T*, using only ``close`` prices up to and including
``T``. It says nothing about when that position is actually entered or
exited in a portfolio.

That separation is deliberate and is enforced by the backtest engine
(``strategy_lab.backtest``), not here: a signal computed from bar T's close
must not be executable at a price that was available at or before that
close. The backtest engine is responsible for shifting this signal forward
so that a change in signal known at close T executes no earlier than the
open of bar T + 1. Mixing that timing logic into signal generation would
make it easy to accidentally introduce look-ahead bias.
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.indicators.ema import ema


def generate_signals(close: pd.Series, fast_period: int, slow_period: int) -> pd.DataFrame:
    """Generate EMA crossover signals from a series of close prices.

    Requires ``fast_period < slow_period``; raises ``ValueError`` otherwise
    (a fast EMA that is not faster than the slow EMA is not a crossover
    strategy). Period validity itself is enforced by ``ema``.

    Returns a DataFrame, indexed like ``close``, with columns:

    - ``fast_ema``, ``slow_ema``: the two EMAs.
    - ``signal``: ``1`` (long) while ``fast_ema > slow_ema``, else ``0``.
      While either EMA is not yet defined (the initial warm-up bars), the
      signal is ``0`` — there is not enough information yet to take a
      position.
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be strictly less than "
            f"slow_period ({slow_period})"
        )

    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)

    both_defined = fast_ema.notna() & slow_ema.notna()
    signal = (fast_ema > slow_ema).astype(int).where(both_defined, other=0)

    return pd.DataFrame(
        {"fast_ema": fast_ema, "slow_ema": slow_ema, "signal": signal},
        index=close.index,
    )
