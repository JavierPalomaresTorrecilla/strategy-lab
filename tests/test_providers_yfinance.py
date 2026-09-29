"""Tests for the yfinance provider. No real network access: `yfinance.download`
is monkeypatched in every test."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.data.providers.yfinance_provider import (
    UnexpectedProviderShapeError,
    YFinanceProvider,
)

_DOWNLOAD_TARGET = "strategy_lab.data.providers.yfinance_provider.yfinance.download"
_VERSION_TARGET = "strategy_lab.data.providers.yfinance_provider.yfinance.__version__"


def _synthetic_yfinance_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_100_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )


def test_provider_passes_all_explicit_semantics_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return _synthetic_yfinance_frame()

    monkeypatch.setattr(_DOWNLOAD_TARGET, fake_download)

    YFinanceProvider().fetch(symbol="SPY", interval="1d", start="2010-01-01", end="2026-01-01")

    assert captured == {
        "tickers": "SPY",
        "start": "2010-01-01",
        "end": "2026-01-01",
        "interval": "1d",
        "auto_adjust": False,
        "repair": False,
        "actions": True,
        "keepna": True,
        "ignore_tz": True,
        "multi_level_index": False,
        "group_by": "column",
        "prepost": False,
        "progress": False,
        "threads": False,
    }


def test_provider_is_parameterized_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return _synthetic_yfinance_frame()

    monkeypatch.setattr(_DOWNLOAD_TARGET, fake_download)

    YFinanceProvider().fetch(symbol="AAPL", interval="1wk", start="2020-05-01", end="2021-01-01")

    assert captured["tickers"] == "AAPL"
    assert captured["interval"] == "1wk"
    assert captured["start"] == "2020-05-01"
    assert captured["end"] == "2021-01-01"


def test_result_metadata_records_actual_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DOWNLOAD_TARGET, lambda **kwargs: _synthetic_yfinance_frame())
    monkeypatch.setattr(_VERSION_TARGET, "9.9.9")

    result = YFinanceProvider().fetch(
        symbol="SPY", interval="1d", start="2010-01-01", end="2026-01-01"
    )

    assert result.provider == "yfinance"
    assert result.provider_library_version == "9.9.9"
    assert result.symbol == "SPY"
    assert result.interval == "1d"
    assert result.requested_start == "2010-01-01"
    assert result.requested_end == "2026-01-01"
    assert result.options_used["auto_adjust"] is False
    assert result.options_used["ignore_tz"] is True
    assert len(result.dataframe) == 2


def test_none_result_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DOWNLOAD_TARGET, lambda **kwargs: None)
    with pytest.raises(UnexpectedProviderShapeError):
        YFinanceProvider().fetch(symbol="SPY", interval="1d", start="2010-01-01", end="2026-01-01")


def test_unexpected_multiindex_columns_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _synthetic_yfinance_frame()
    frame.columns = pd.MultiIndex.from_product([["SPY"], frame.columns])

    monkeypatch.setattr(_DOWNLOAD_TARGET, lambda **kwargs: frame)
    with pytest.raises(UnexpectedProviderShapeError):
        YFinanceProvider().fetch(symbol="SPY", interval="1d", start="2010-01-01", end="2026-01-01")
