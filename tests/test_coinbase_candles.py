"""Offline tests for daily Coinbase candle-window planning."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from strategy_lab.data.providers.coinbase_candles import (
    DAY,
    HISTORY_SEARCH_FLOOR,
    WINDOW_DAYS,
    CandleWindow,
    is_complete,
    last_completed_bucket_start,
    plan_daily_windows,
)


def test_is_complete_boundaries_and_validation() -> None:
    bucket_start = datetime(2026, 10, 14, tzinfo=UTC)

    assert is_complete(bucket_start, bucket_start + DAY, "1d")
    assert not is_complete(bucket_start, bucket_start + DAY - timedelta(microseconds=1), "1d")

    with pytest.raises(ValueError):
        is_complete(bucket_start.replace(tzinfo=None), bucket_start + DAY, "1d")
    with pytest.raises(ValueError):
        is_complete(bucket_start, (bucket_start + DAY).replace(tzinfo=None), "1d")
    with pytest.raises(ValueError):
        is_complete(
            bucket_start.replace(tzinfo=timezone(timedelta(hours=2))), bucket_start + DAY, "1d"
        )
    with pytest.raises(ValueError):
        is_complete(
            bucket_start,
            (bucket_start + DAY).replace(tzinfo=timezone(timedelta(hours=2))),
            "1d",
        )
    with pytest.raises(ValueError):
        is_complete(bucket_start + timedelta(hours=1), bucket_start + DAY, "1d")
    with pytest.raises(ValueError):
        is_complete(bucket_start, bucket_start + DAY, "1h")


@pytest.mark.parametrize(
    "now_utc",
    [
        datetime(2026, 10, 15, tzinfo=UTC),
        datetime(2026, 10, 15, 23, 59, 59, 999999, tzinfo=UTC),
    ],
)
def test_last_completed_bucket_start(now_utc: datetime) -> None:
    assert last_completed_bucket_start(now_utc) == datetime(2026, 10, 14, tzinfo=UTC)


def test_plan_daily_windows_coverage_and_boundaries() -> None:
    now_utc = datetime(2026, 10, 15, 14, 13, 27, tzinfo=UTC)
    windows = plan_daily_windows(now_utc)
    last = last_completed_bucket_start(now_utc)

    assert len(windows) == 15
    assert sum(window.day_count for window in windows) == (
        last_completed_bucket_start(now_utc) - HISTORY_SEARCH_FLOOR
    ).days + 1
    assert windows[0].first_bucket_start == HISTORY_SEARCH_FLOOR
    assert windows[-1].last_bucket_start == last
    assert all(window.day_count <= WINDOW_DAYS for window in windows)
    assert all(
        window.request_start.isdigit() and window.request_end.isdigit() for window in windows
    )
    assert all(
        window.first_bucket_start.tzinfo is not None
        and window.first_bucket_start.utcoffset() == timedelta(0)
        and window.last_bucket_start.tzinfo is not None
        and window.last_bucket_start.utcoffset() == timedelta(0)
        and window.first_bucket_start.time() == datetime.min.time()
        and window.last_bucket_start.time() == datetime.min.time()
        for window in windows
    )
    assert all(
        current.first_bucket_start == previous.last_bucket_start + DAY
        for previous, current in zip(windows, windows[1:], strict=False)
    )

    covered_dates = {
        (window.first_bucket_start + DAY * offset).date()
        for window in windows
        for offset in range(window.day_count)
    }
    expected_dates = {
        (HISTORY_SEARCH_FLOOR + DAY * offset).date()
        for offset in range((last - HISTORY_SEARCH_FLOOR).days + 1)
    }
    assert covered_dates == expected_dates


def test_plan_daily_windows_validation_and_determinism() -> None:
    floor = datetime(2026, 10, 14, tzinfo=UTC)
    now_utc = datetime(2026, 10, 16, 14, 13, 27, tzinfo=UTC)

    with pytest.raises(ValueError):
        plan_daily_windows(floor, floor=floor)
    with pytest.raises(ValueError):
        plan_daily_windows(now_utc, floor=floor, window_days=0)
    with pytest.raises(ValueError):
        plan_daily_windows(now_utc, floor=floor, window_days=351)
    with pytest.raises(ValueError):
        plan_daily_windows(now_utc, floor=datetime(2015, 1, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        plan_daily_windows(now_utc, floor=datetime(2015, 1, 1))

    single_day_windows = plan_daily_windows(now_utc, floor=floor, window_days=1)
    assert all(window.day_count == 1 for window in single_day_windows)
    plan_daily_windows(now_utc, floor=floor, window_days=350)
    assert plan_daily_windows(now_utc, floor=floor) == plan_daily_windows(now_utc, floor=floor)


def test_candle_window_rejects_reverse_range() -> None:
    with pytest.raises(ValueError):
        CandleWindow(
            first_bucket_start=datetime(2026, 10, 15, tzinfo=UTC),
            last_bucket_start=datetime(2026, 10, 14, tzinfo=UTC),
        )


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
            "import sys; import strategy_lab.data.providers.coinbase_candles; "
            "assert 'coinbase' not in sys.modules",
        ],
        check=True,
        cwd=project_root,
        env=environment,
    )
