"""Data layer: ingestion, cleaning, and storage of market data.

No exchange or broker connectivity exists yet.
"""

from strategy_lab.data.ohlcv import REQUIRED_COLUMNS, OHLCVValidationError, validate_ohlcv

__all__ = ["OHLCVValidationError", "REQUIRED_COLUMNS", "validate_ohlcv"]

