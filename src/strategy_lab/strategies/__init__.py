"""Strategy layer: hypothesis definitions and signal generation logic.

Strategies here generate a signal only; they never decide execution timing
or prices. This project does not assume that a historically profitable
strategy has predictive value.
"""

from strategy_lab.strategies.ema_crossover import generate_signals

__all__ = ["generate_signals"]

