"""Market-data provider layer: a narrow interface plus one concrete provider.

No plugin framework: adding a second provider means writing one more module
implementing ``MarketDataProvider``, not registering into a discovery system.
"""

from strategy_lab.data.providers.base import MarketDataProvider, RawFetchResult
from strategy_lab.data.providers.yfinance_provider import (
    UnexpectedProviderShapeError,
    YFinanceProvider,
)

__all__ = [
    "MarketDataProvider",
    "RawFetchResult",
    "UnexpectedProviderShapeError",
    "YFinanceProvider",
]
