"""Tests for robustness metrics and their sample-size gates."""

from __future__ import annotations

import pytest

from strategy_lab.validation.robustness_metrics import (
    MIN_DAILY_OBSERVATIONS_FOR_SHARPE,
    MIN_ELIGIBLE_ROUND_TRIPS,
    InsufficientSample,
    NotApplicable,
    annualized_volatility,
    cagr,
    exposure,
    hit_rate,
    local_max_drawdown,
    profit_factor,
    sharpe_like,
)


def test_cagr_basic() -> None:
    # doubling over 1 year -> 100% CAGR
    assert cagr(100.0, 200.0, years=1) == pytest.approx(1.0)
    # doubling over 2 years -> ~41.4% CAGR
    assert cagr(100.0, 200.0, years=2) == pytest.approx(2**0.5 - 1)


def test_annualized_volatility_insufficient_sample() -> None:
    result = annualized_volatility([0.01])
    assert isinstance(result, InsufficientSample)
    assert result.required == 2


def test_annualized_volatility_computes_with_enough_observations() -> None:
    result = annualized_volatility([0.01, -0.01, 0.02, -0.02])
    assert isinstance(result, float)
    assert result > 0


def test_local_max_drawdown_peak_initialized_at_start_equity() -> None:
    # peak starts at 100 (start_equity), even though equity never exceeds
    # 100 within the window itself -- so drawdown is measured against the
    # inherited starting point, not a fresh in-window high.
    dd = local_max_drawdown([100.0, 90.0, 95.0], start_equity=100.0)
    assert dd == pytest.approx(-0.10)


def test_local_max_drawdown_does_not_inherit_a_higher_prior_peak() -> None:
    # A prior period's peak (say 500) must NOT be used here -- only
    # start_equity (100) seeds the peak.
    dd = local_max_drawdown([100.0, 105.0, 95.0], start_equity=100.0)
    assert dd == pytest.approx(95.0 / 105.0 - 1.0)


def test_exposure() -> None:
    assert exposure([True, True, False, False]) == pytest.approx(0.5)
    assert exposure([False, False]) == pytest.approx(0.0)


def test_exposure_empty_rejected() -> None:
    with pytest.raises(ValueError):
        exposure([])


def test_sharpe_like_insufficient_sample() -> None:
    daily_returns = [0.001] * (MIN_DAILY_OBSERVATIONS_FOR_SHARPE - 1)
    result = sharpe_like(daily_returns)
    assert isinstance(result, InsufficientSample)


def test_sharpe_like_computes_with_enough_observations() -> None:
    import random

    rng = random.Random(42)
    daily_returns = [rng.gauss(0.0005, 0.01) for _ in range(MIN_DAILY_OBSERVATIONS_FOR_SHARPE)]
    result = sharpe_like(daily_returns)
    assert isinstance(result, float)


def test_sharpe_like_zero_volatility_not_applicable() -> None:
    daily_returns = [0.001] * MIN_DAILY_OBSERVATIONS_FOR_SHARPE
    result = sharpe_like(daily_returns)
    assert isinstance(result, NotApplicable)


def test_hit_rate_insufficient_sample() -> None:
    pnls = [1.0] * (MIN_ELIGIBLE_ROUND_TRIPS - 1)
    result = hit_rate(pnls)
    assert isinstance(result, InsufficientSample)
    assert result.required == MIN_ELIGIBLE_ROUND_TRIPS


def test_hit_rate_computes_with_enough_round_trips() -> None:
    pnls = [1.0] * 15 + [-1.0] * 5
    result = hit_rate(pnls)
    assert result == pytest.approx(0.75)


def test_profit_factor_insufficient_sample() -> None:
    pnls = [1.0] * (MIN_ELIGIBLE_ROUND_TRIPS - 1)
    result = profit_factor(pnls)
    assert isinstance(result, InsufficientSample)


def test_profit_factor_zero_gross_losses_not_applicable_not_inf() -> None:
    pnls = [1.0] * MIN_ELIGIBLE_ROUND_TRIPS
    result = profit_factor(pnls)
    assert isinstance(result, NotApplicable)
    assert result.reason == "zero gross losses"


def test_profit_factor_computes_with_enough_round_trips() -> None:
    pnls = [2.0] * 15 + [-1.0] * 5
    result = profit_factor(pnls)
    assert result == pytest.approx(30.0 / 5.0)
