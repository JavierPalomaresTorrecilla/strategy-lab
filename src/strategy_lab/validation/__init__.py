"""Validation layer: out-of-sample testing, robustness checks, and statistical
significance testing to guard against overfitting and parameter mining.

Also home to cross-engine parity checks (``strategy_lab.validation.parity``,
``strategy_lab.validation.vectorbt_adapter``): whether the reference
backtest engine's results can be trusted is itself a validation concern.
Importing this subpackage's ``__init__`` does not require VectorBT to be
installed; only ``vectorbt_adapter``/``parity`` do.
"""
