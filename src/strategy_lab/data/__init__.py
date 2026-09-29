"""Data layer: ingestion, cleaning, and storage of market data.

No exchange or broker connectivity exists yet. Market-data ingestion (via
``strategy_lab.data.providers``) exists solely for offline historical
research; nothing here places orders or executes trades.
"""

from strategy_lab.data.canonical import CanonicalConversionError, to_canonical_ohlcv
from strategy_lab.data.env import (
    DATA_ROOT_ENV_VAR,
    DataRootError,
    DataRootNotSetError,
    DataRootUnavailableError,
    resolve_data_root,
)
from strategy_lab.data.ohlcv import REQUIRED_COLUMNS, OHLCVValidationError, validate_ohlcv
from strategy_lab.data.splits import (
    RESEARCH_ALLOWED_SPLITS,
    DatasetSplit,
    SplitConfigError,
    assign_split,
    load_splits,
    split_ohlcv,
)

__all__ = [
    "DATA_ROOT_ENV_VAR",
    "CanonicalConversionError",
    "DataRootError",
    "DataRootNotSetError",
    "DataRootUnavailableError",
    "DatasetSplit",
    "OHLCVValidationError",
    "REQUIRED_COLUMNS",
    "RESEARCH_ALLOWED_SPLITS",
    "SplitConfigError",
    "assign_split",
    "load_splits",
    "resolve_data_root",
    "split_ohlcv",
    "to_canonical_ohlcv",
    "validate_ohlcv",
]

