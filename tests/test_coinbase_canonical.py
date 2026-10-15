"""Offline tests for Coinbase daily-candle decoding and canonicalization."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from strategy_lab.data.canonical import to_canonical_ohlcv
from strategy_lab.data.ohlcv import OHLCVValidationError, validate_ohlcv
from strategy_lab.data.providers.coinbase_candles import CandleWindow
from strategy_lab.data.providers.coinbase_canonical import (
    CandleOutsideWindowError,
    CoinbaseCandle,
    CoinbaseDecodeResult,
    DuplicateCandleError,
    EmptyHistoryError,
    IncompleteCandleError,
    MalformedCandleError,
    decode_pages,
    parse_candle,
    to_canonical_frame,
)

PRODUCT = "BTC-USD"
NOW_UTC = datetime(2026, 10, 15, 12, tzinfo=UTC)


def _bucket(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=UTC)


def _raw_candle(bucket_start: datetime, **overrides: object) -> dict[str, object]:
    candle: dict[str, object] = {
        "start": str(int(bucket_start.timestamp())),
        "low": "9.5",
        "high": "11.0",
        "open": "10.0",
        "close": "10.5",
        "volume": "42.0",
    }
    candle.update(overrides)
    return candle


def _window(first_day: int, last_day: int) -> CandleWindow:
    return CandleWindow(_bucket(first_day), _bucket(last_day))


def _result_for(*candles: CoinbaseCandle) -> CoinbaseDecodeResult:
    ordered = tuple(sorted(candles, key=lambda candle: candle.bucket_start))
    return CoinbaseDecodeResult(
        product=PRODUCT,
        candles=ordered,
        earliest_bucket_start=ordered[0].bucket_start,
        latest_bucket_start=ordered[-1].bucket_start,
        candle_count=len(ordered),
        missing_days_utc=(),
        missing_day_count=0,
    )


def test_parse_candle_valid() -> None:
    candle = parse_candle(_raw_candle(_bucket(10)))

    assert candle.bucket_start == _bucket(10)
    assert candle.open == Decimal("10.0")
    assert candle.high == Decimal("11.0")
    assert candle.low == Decimal("9.5")
    assert candle.close == Decimal("10.5")
    assert candle.volume == Decimal("42.0")


@pytest.mark.parametrize("missing_key", ["start", "low", "high", "open", "close", "volume"])
def test_parse_candle_rejects_each_missing_key(missing_key: str) -> None:
    raw = _raw_candle(_bucket(10))
    raw.pop(missing_key)

    with pytest.raises(MalformedCandleError, match=missing_key):
        parse_candle(raw)


def test_parse_candle_rejects_unexpected_key() -> None:
    raw = _raw_candle(_bucket(10), unexpected="value")

    with pytest.raises(MalformedCandleError, match="unexpected"):
        parse_candle(raw)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("start", 1),
        ("low", 1.0),
        ("high", True),
        ("open", None),
        ("close", 2),
        ("volume", False),
    ],
)
def test_parse_candle_rejects_non_string_values(field: str, bad_value: object) -> None:
    raw = _raw_candle(_bucket(10), **{field: bad_value})

    with pytest.raises(MalformedCandleError, match=field):
        parse_candle(raw)


@pytest.mark.parametrize("start", ["not-digits", "-1791590400", "١٧٩١٥٩٠٤٠٠"])
def test_parse_candle_rejects_non_digit_start(start: str) -> None:
    with pytest.raises(MalformedCandleError, match="start"):
        parse_candle(_raw_candle(_bucket(10), start=start))


def test_parse_candle_rejects_non_midnight_start() -> None:
    raw = _raw_candle(_bucket(10), start=str(int(_bucket(10, hour=1).timestamp())))

    with pytest.raises(MalformedCandleError, match="non-midnight"):
        parse_candle(raw)


@pytest.mark.parametrize("bad_value", ["NaN", "Infinity", "abc", ""])
def test_parse_candle_rejects_nonfinite_or_invalid_decimal(bad_value: str) -> None:
    with pytest.raises(MalformedCandleError, match="open"):
        parse_candle(_raw_candle(_bucket(10), open=bad_value))


def test_parse_candle_preserves_decimal_precision() -> None:
    candle = parse_candle(_raw_candle(_bucket(10), open="0.00001"))

    assert candle.open == Decimal("0.00001")


def test_decode_pages_sorts_newest_first_input_to_ascending() -> None:
    result = decode_pages(
        PRODUCT,
        [(_window(10, 12), [_raw_candle(_bucket(12)), _raw_candle(_bucket(10))])],
        NOW_UTC,
    )

    assert [candle.bucket_start for candle in result.candles] == [_bucket(10), _bucket(12)]


@pytest.mark.parametrize("outside_day", [9, 12])
def test_decode_pages_rejects_bucket_outside_window(outside_day: int) -> None:
    with pytest.raises(CandleOutsideWindowError, match=PRODUCT):
        decode_pages(PRODUCT, [(_window(10, 11), [_raw_candle(_bucket(outside_day))])], NOW_UTC)


def test_decode_pages_rejects_duplicate_within_page() -> None:
    raw = _raw_candle(_bucket(10))

    with pytest.raises(DuplicateCandleError, match=PRODUCT):
        decode_pages(PRODUCT, [(_window(10, 11), [raw, raw.copy()])], NOW_UTC)


def test_decode_pages_rejects_duplicate_across_pages() -> None:
    with pytest.raises(DuplicateCandleError, match=PRODUCT):
        decode_pages(
            PRODUCT,
            [
                (_window(10, 11), [_raw_candle(_bucket(11))]),
                (_window(11, 12), [_raw_candle(_bucket(11))]),
            ],
            NOW_UTC,
        )


def test_decode_pages_rejects_incomplete_today_bucket() -> None:
    with pytest.raises(IncompleteCandleError, match=PRODUCT):
        decode_pages(PRODUCT, [(_window(15, 15), [_raw_candle(_bucket(15))])], NOW_UTC)


def test_decode_pages_rejects_empty_history() -> None:
    with pytest.raises(EmptyHistoryError, match=PRODUCT):
        decode_pages(PRODUCT, [(_window(10, 11), []), (_window(12, 13), [])], NOW_UTC)


def test_decode_pages_does_not_report_pre_history_for_empty_early_pages() -> None:
    result = decode_pages(
        PRODUCT,
        [
            (_window(1, 3), []),
            (_window(4, 6), [_raw_candle(_bucket(5))]),
        ],
        NOW_UTC,
    )

    assert result.earliest_bucket_start == _bucket(5)
    assert result.missing_days_utc == ("2026-10-06",)
    assert all(day >= "2026-10-05" for day in result.missing_days_utc)


def test_decode_pages_records_internal_and_trailing_gaps() -> None:
    result = decode_pages(
        PRODUCT,
        [(_window(1, 5), [_raw_candle(_bucket(1)), _raw_candle(_bucket(3))])],
        NOW_UTC,
    )

    assert result.missing_days_utc == ("2026-10-02", "2026-10-04", "2026-10-05")
    assert result.missing_day_count == 3


def test_decode_pages_contiguous_history_has_no_gaps() -> None:
    result = decode_pages(
        PRODUCT,
        [
            (
                _window(10, 12),
                [_raw_candle(_bucket(10)), _raw_candle(_bucket(11)), _raw_candle(_bucket(12))],
            )
        ],
        NOW_UTC,
    )

    assert result.missing_days_utc == ()
    assert result.missing_day_count == 0


def test_to_canonical_frame_matches_spy_canonical_schema_and_timestamps() -> None:
    raw_spy = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [42.0, 43.0],
        },
        index=pd.DatetimeIndex(
            [datetime(2026, 10, 10), datetime(2026, 10, 11)], dtype="datetime64[ns]"
        ),
    )
    spy_canonical = to_canonical_ohlcv(raw_spy)
    result = decode_pages(
        PRODUCT,
        [(_window(10, 11), [_raw_candle(_bucket(10)), _raw_candle(_bucket(11))])],
        NOW_UTC,
    )

    frame = to_canonical_frame(result)

    assert list(frame.columns) == list(spy_canonical.columns)
    assert frame.dtypes.tolist() == spy_canonical.dtypes.tolist()
    assert frame["timestamp"].dt.tz is None
    assert all(timestamp.time() == datetime.min.time() for timestamp in frame["timestamp"])
    validate_ohlcv(frame)


def test_to_canonical_frame_delegates_ohlc_validation() -> None:
    invalid = CoinbaseCandle(
        bucket_start=_bucket(10),
        open=Decimal("10"),
        high=Decimal("8"),
        low=Decimal("9"),
        close=Decimal("10"),
        volume=Decimal("1"),
    )

    with pytest.raises(OHLCVValidationError) as error:
        to_canonical_frame(_result_for(invalid))

    assert error.type is OHLCVValidationError


def test_to_canonical_frame_is_independent_of_process_timezone() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import json
from datetime import UTC, datetime
from decimal import Decimal
from strategy_lab.data.providers.coinbase_canonical import (
    CoinbaseCandle,
    CoinbaseDecodeResult,
    to_canonical_frame,
)
candles = (
    CoinbaseCandle(
        datetime(2026, 10, 10, tzinfo=UTC),
        Decimal('10'), Decimal('11'), Decimal('9'), Decimal('10.5'), Decimal('42'),
    ),
    CoinbaseCandle(
        datetime(2026, 10, 11, tzinfo=UTC),
        Decimal('11'), Decimal('12'), Decimal('10'), Decimal('11.5'), Decimal('43'),
    ),
)
result = CoinbaseDecodeResult(
    'BTC-USD', candles, candles[0].bucket_start, candles[-1].bucket_start, 2, (), 0,
)
frame = to_canonical_frame(result)
print(json.dumps({
    'dtype': str(frame['timestamp'].dtype),
    'timestamps': [value.value for value in frame['timestamp']],
}))
"""
    environment = os.environ | {
        "PYTHONPATH": str(project_root / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    }
    baseline = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    new_york = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=project_root,
        env=environment | {"TZ": "America/New_York"},
        capture_output=True,
        text=True,
    )

    assert new_york.stdout == baseline.stdout


def test_import_does_not_load_coinbase_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ | {
        "PYTHONPATH": str(project_root / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import strategy_lab.data.providers.coinbase_canonical; "
            "assert 'coinbase' not in sys.modules",
        ],
        check=True,
        cwd=project_root,
        env=environment,
    )
