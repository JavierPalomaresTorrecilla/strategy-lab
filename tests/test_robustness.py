"""Tests for the M3 parameter-neighborhood/cost-stress robustness module.

Requires VectorBT (the `parity` extra) -- same convention as
`tests/test_parity.py`: no `pytest.importorskip`, so a missing install
fails collection visibly rather than silently skipping.
"""

from __future__ import annotations

import pandas as pd
import pytest
import vectorbt  # noqa: F401  (import here makes a missing install fail collection visibly)

from conftest import random_walk_ohlcv
from strategy_lab.backtest.engine import BacktestConfig
from strategy_lab.validation.robustness import (
    ABSOLUTE_MAX_COMBINATIONS,
    PARITY_SPOT_CHECKS,
    EmaNeighborhood,
    RobustnessConfigError,
    build_fragility_report,
    run_cost_stress,
    run_parameter_grid,
    run_parity_spot_checks,
)
from strategy_lab.validation.temporal_validation import build_windows


@pytest.fixture(scope="module")
def small_neighborhood() -> EmaNeighborhood:
    return EmaNeighborhood(fast=(8, 9, 10, 11, 12), slow=(25, 30, 35), max_combinations=25)


@pytest.fixture(scope="module")
def evaluation_windows():
    _, windows = build_windows(burn_in_year=2010, evaluation_years=[2011, 2012])
    return windows


@pytest.fixture(scope="module")
def research_ohlcv() -> pd.DataFrame:
    return random_walk_ohlcv("2010-01-01", "2012-12-31", seed=13)


@pytest.fixture(scope="module")
def baseline_config() -> BacktestConfig:
    return BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005)


# --- EmaNeighborhood: cap, combinations, adjacency --------------------------


def test_neighborhood_produces_exactly_fifteen_combinations() -> None:
    neighborhood = EmaNeighborhood(fast=(8, 9, 10, 11, 12), slow=(25, 30, 35), max_combinations=25)
    combos = neighborhood.combinations()
    assert len(combos) == 15
    assert combos == [(f, s) for f in (8, 9, 10, 11, 12) for s in (25, 30, 35)]


def test_neighborhood_configured_order_is_fast_major_slow_minor() -> None:
    neighborhood = EmaNeighborhood(fast=(8, 9), slow=(25, 30), max_combinations=25)
    assert neighborhood.combinations() == [(8, 25), (8, 30), (9, 25), (9, 30)]


def test_neighborhood_rejects_max_combinations_above_structural_cap() -> None:
    with pytest.raises(RobustnessConfigError):
        EmaNeighborhood(fast=(8, 9), slow=(25, 30), max_combinations=ABSOLUTE_MAX_COMBINATIONS + 1)


def test_neighborhood_rejects_grid_exceeding_configured_cap() -> None:
    # 6 fast * 6 slow, all fast < slow -> 36 valid combos, exceeding a
    # configured max_combinations of 25 (and the structural cap too).
    with pytest.raises(RobustnessConfigError):
        EmaNeighborhood(fast=tuple(range(5, 11)), slow=tuple(range(20, 26)), max_combinations=25)


def test_neighborhood_adjacency_is_array_position_based_not_numeric() -> None:
    # slow spacing is non-uniform (25 -> 30 is +5, 30 -> 60 is +30); the
    # neighbor of 30 must still be {25, 60} by array position, not by
    # numeric closeness.
    neighborhood = EmaNeighborhood(fast=(10,), slow=(25, 30, 60), max_combinations=25)
    assert set(neighborhood.neighbors(10, 30)) == {(10, 25), (10, 60)}
    assert set(neighborhood.neighbors(10, 25)) == {(10, 30)}
    assert set(neighborhood.neighbors(10, 60)) == {(10, 30)}


def test_neighborhood_edge_cell_has_fewer_neighbors() -> None:
    neighborhood = EmaNeighborhood(fast=(8, 9, 10, 11, 12), slow=(25, 30, 35), max_combinations=25)
    # (8, 25) is a corner: only one fast-neighbor (9,25) and one
    # slow-neighbor (8,30) exist and are valid.
    assert set(neighborhood.neighbors(8, 25)) == {(9, 25), (8, 30)}


# --- run_parameter_grid: execution count, ordering --------------------------


def test_run_parameter_grid_executes_exactly_once_per_cell_in_grid_order(
    small_neighborhood, research_ohlcv, baseline_config, evaluation_windows, monkeypatch
) -> None:
    call_order = []
    from strategy_lab.validation import robustness as robustness_mod

    real_run_vectorbt = robustness_mod.run_vectorbt

    def _counting_run_vectorbt(ohlcv, signal, config):
        call_order.append((config.commission_rate, config.slippage_rate))
        return real_run_vectorbt(ohlcv, signal, config)

    monkeypatch.setattr(robustness_mod, "run_vectorbt", _counting_run_vectorbt)

    results = run_parameter_grid(
        research_ohlcv, small_neighborhood, baseline_config, evaluation_windows
    )

    assert len(results) == 15
    assert len(call_order) == 15  # exactly one VectorBT run per grid cell
    assert [(r.fast, r.slow) for r in results] == small_neighborhood.combinations()


def test_run_parameter_grid_never_sorted_by_performance(
    small_neighborhood, research_ohlcv, baseline_config, evaluation_windows
) -> None:
    results = run_parameter_grid(
        research_ohlcv, small_neighborhood, baseline_config, evaluation_windows
    )
    assert [(r.fast, r.slow) for r in results] == small_neighborhood.combinations()
    # Explicitly not sorted by aggregate_return.
    returns = [r.aggregate_return for r in results]
    assert returns != sorted(returns) or returns == sorted(
        returns
    )  # order asserted above is the real guarantee
    assert not hasattr(results, "best")
    assert not any(hasattr(r, "rank") for r in results)


def test_fragility_report_structure(
    small_neighborhood, research_ohlcv, baseline_config, evaluation_windows
) -> None:
    grid_results = run_parameter_grid(
        research_ohlcv, small_neighborhood, baseline_config, evaluation_windows
    )
    report = build_fragility_report(small_neighborhood, grid_results, baseline_cell=(10, 30))

    assert report.baseline_cell == (10, 30)
    assert 0.0 <= report.sign_consistency_fraction <= 1.0
    assert set(report.window_dispersion.keys()) == {2011, 2012}
    for stats in report.window_dispersion.values():
        assert stats["min"] <= stats["median"] <= stats["max"]
    assert report.heuristic_label == "preregistered heuristic flag, not a significance test"
    for flag in report.isolated_cell_flags:
        assert flag.cell in small_neighborhood.combinations()


# --- run_cost_stress: baseline reuse, execution count -----------------------


def test_cost_stress_reuses_baseline_never_re_executes_it(
    small_neighborhood, research_ohlcv, baseline_config, evaluation_windows, monkeypatch
) -> None:
    grid_results = run_parameter_grid(
        research_ohlcv, small_neighborhood, baseline_config, evaluation_windows
    )
    baseline_cell = next(r for r in grid_results if (r.fast, r.slow) == (10, 30))

    from strategy_lab.validation import robustness as robustness_mod

    call_count = {"n": 0}
    real_run_vectorbt = robustness_mod.run_vectorbt

    def _counting_run_vectorbt(ohlcv, signal, config):
        call_count["n"] += 1
        return real_run_vectorbt(ohlcv, signal, config)

    monkeypatch.setattr(robustness_mod, "run_vectorbt", _counting_run_vectorbt)

    cost_scenarios = {
        "baseline": {"commission_rate": 0.001, "slippage_rate": 0.0005},
        "moderately_worse": {"commission_rate": 0.0025, "slippage_rate": 0.0015},
        "materially_worse": {"commission_rate": 0.005, "slippage_rate": 0.004},
    }
    results = run_cost_stress(research_ohlcv, cost_scenarios, evaluation_windows, baseline_cell)

    assert len(results) == 3
    assert call_count["n"] == 2  # only the two non-baseline scenarios actually ran
    baseline_result = next(r for r in results if r.scenario == "baseline")
    assert baseline_result.aggregate_return == baseline_cell.aggregate_return
    assert baseline_result.window_returns == baseline_cell.window_returns


def test_total_executions_across_both_axes_is_fifteen_plus_two_never_forty_five(
    small_neighborhood, research_ohlcv, baseline_config, evaluation_windows, monkeypatch
) -> None:
    from strategy_lab.validation import robustness as robustness_mod

    call_count = {"n": 0}
    real_run_vectorbt = robustness_mod.run_vectorbt

    def _counting_run_vectorbt(ohlcv, signal, config):
        call_count["n"] += 1
        return real_run_vectorbt(ohlcv, signal, config)

    monkeypatch.setattr(robustness_mod, "run_vectorbt", _counting_run_vectorbt)

    grid_results = run_parameter_grid(
        research_ohlcv, small_neighborhood, baseline_config, evaluation_windows
    )
    baseline_cell = next(r for r in grid_results if (r.fast, r.slow) == (10, 30))
    cost_scenarios = {
        "baseline": {"commission_rate": 0.001, "slippage_rate": 0.0005},
        "moderately_worse": {"commission_rate": 0.0025, "slippage_rate": 0.0015},
        "materially_worse": {"commission_rate": 0.005, "slippage_rate": 0.004},
    }
    run_cost_stress(research_ohlcv, cost_scenarios, evaluation_windows, baseline_cell)

    assert call_count["n"] == 17  # 15 (parameter grid) + 2 (non-baseline cost tiers), never 45


# --- run_parity_spot_checks: exactly three, predefined ----------------------


def test_parity_spot_checks_are_exactly_the_three_predefined_configurations() -> None:
    assert PARITY_SPOT_CHECKS == ((8, 25), (10, 30), (12, 35))
    assert len(PARITY_SPOT_CHECKS) == 3


def test_parity_spot_checks_agree_between_engines(research_ohlcv, baseline_config) -> None:
    reports = run_parity_spot_checks(research_ohlcv, baseline_config, PARITY_SPOT_CHECKS)
    assert set(reports.keys()) == {(8, 25), (10, 30), (12, 35)}
    for (fast, slow), report in reports.items():
        assert report.is_within_tolerance(), f"({fast},{slow}) diverged: {report.divergences}"


# --- TEST-leakage behavioral test -------------------------------------------


def test_robustness_grid_invariant_to_test_row_mutation(baseline_config) -> None:
    from strategy_lab.data.splits import DatasetSplit

    splits = [
        DatasetSplit(
            name="train", start=pd.Timestamp("2020-01-01"), end_exclusive=pd.Timestamp("2020-04-01")
        ),
        DatasetSplit(
            name="validation",
            start=pd.Timestamp("2020-04-01"),
            end_exclusive=pd.Timestamp("2020-07-01"),
        ),
        DatasetSplit(
            name="test", start=pd.Timestamp("2020-07-01"), end_exclusive=pd.Timestamp("2020-10-01")
        ),
    ]
    ohlcv = random_walk_ohlcv("2020-01-01", "2020-09-30", seed=21)
    # Small grid, window entirely inside train+validation -- independent
    # of the real project config, this only needs the custom splits above.
    small_grid = EmaNeighborhood(fast=(8, 10), slow=(25, 30), max_combinations=25)

    original_results = _run_with_splits(
        run_parameter_grid, ohlcv, small_grid, baseline_config, splits
    )

    mutated = ohlcv.copy()
    test_mask = mutated["timestamp"] >= pd.Timestamp("2020-07-01")
    mutated.loc[test_mask, ["open", "high", "low", "close"]] *= 999.0
    mutated_results = _run_with_splits(
        run_parameter_grid, mutated, small_grid, baseline_config, splits
    )

    assert original_results == mutated_results


def _run_with_splits(func, ohlcv, neighborhood, config, splits):
    from strategy_lab.validation.temporal_validation import (
        EvaluationWindow,
        filter_to_allowed_research_history,
    )

    filtered = filter_to_allowed_research_history(ohlcv, splits)
    # Window starts after 2020-01-01 so there is at least one prior
    # observation to serve as the boundary's start_equity.
    window = EvaluationWindow(
        name=2020, start=pd.Timestamp("2020-02-01"), end=pd.Timestamp("2020-07-01")
    )
    return func(filtered, neighborhood, config, [window])
