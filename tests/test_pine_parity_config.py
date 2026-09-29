"""Pins the `pine_parity` block in the real, tracked `config/research.yaml`
to exactly the Milestone 4a facts frozen before any real TradingView export
exists -- catches accidental drift in these dates/settings.

Milestone 4a deliberately adds no Python config-loader module for this
block (there is nothing to load it *into* yet -- see Milestone 4b), so this
test reads the real file directly, the same way `test_splits.py` pins the
tracked `splits` block.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "research.yaml"


def _pine_parity_config() -> dict:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    return config["pine_parity"]


def test_pine_parity_block_present() -> None:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    assert "pine_parity" in config


def test_symbol_timeframe_and_ui_language() -> None:
    pp = _pine_parity_config()
    assert pp["symbol"] == "AMEX:SPY"
    assert pp["timeframe"] == "1D"
    assert pp["ui_language"] == "English"


def test_fixed_ema_periods_match_pine_source() -> None:
    pp = _pine_parity_config()
    assert pp["fast_period"] == 10
    assert pp["slow_period"] == 30


def test_research_and_execution_boundaries() -> None:
    pp = _pine_parity_config()
    assert pp["first_research_session"] == "2010-01-04"
    assert pp["first_test_session"] == "2023-01-03"
    assert pp["research_start"] == "2010-01-01"
    assert pp["research_end"] == "2022-12-30"
    assert pp["last_order_submission_date"] == "2022-12-29"
    assert pp["final_possible_fill_date"] == "2022-12-30"
    assert pp["research_end_exclusive"] == "2022-12-31"
    assert pp["submission_end_exclusive"] == "2022-12-30"


def test_requested_plot_precision_is_a_request_not_an_observed_fact() -> None:
    pp = _pine_parity_config()
    assert pp["requested_plot_precision"] == 8
    # Milestone 4a deliberately freezes no observed-export-precision or
    # EMA-tolerance value -- those require a real export (Milestone 4b).
    assert "observed_plot_precision" not in pp
    assert "ema_tolerance" not in pp


def test_required_settings_all_off_except_execution_delay() -> None:
    pp = _pine_parity_config()
    required = pp["required_settings"]
    for key in (
        "dividend_adjustment",
        "deep_backtesting",
        "bar_magnifier",
        "calc_on_every_tick",
        "calc_on_order_fills",
        "calc_on_every_history_tick",
        "process_orders_on_close",
    ):
        assert required[key] is False, key
    assert required["order_execution_delay"] == "one_tick"


def test_no_csv_schema_frozen_yet() -> None:
    """Milestone 4a must not freeze an assumed CSV header schema --
    that belongs to Milestone 4b, after a real export is observed."""
    pp = _pine_parity_config()
    assert "chart_data_schema" not in pp
    assert "list_of_trades_schema" not in pp
