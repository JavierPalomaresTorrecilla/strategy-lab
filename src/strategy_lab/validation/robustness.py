"""Parameter-neighborhood and cost-stress sensitivity, run via VectorBT.

Descriptive sensitivity only -- nothing here selects, ranks, or optimizes.
The reference engine (``strategy_lab.backtest.engine``) remains the
authoritative single-configuration report (see
``strategy_lab.validation.temporal_validation``); this module runs the
small, preregistered parameter grid and cost-stress matrix efficiently via
``strategy_lab.validation.vectorbt_adapter``, and cross-checks a handful of
predefined configurations against the reference engine for parity, never
the whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategy_lab.backtest.engine import BacktestConfig
from strategy_lab.strategies.ema_crossover import generate_signals
from strategy_lab.validation.parity import ParityReport, compare_engines
from strategy_lab.validation.temporal_validation import (
    EvaluationWindow,
    filter_to_allowed_research_history,
    window_return,
)
from strategy_lab.validation.vectorbt_adapter import run_vectorbt

ABSOLUTE_MAX_COMBINATIONS = 25

PARITY_SPOT_CHECKS: tuple[tuple[int, int], ...] = ((8, 25), (10, 30), (12, 35))


class RobustnessConfigError(ValueError):
    """Raised when the configured parameter neighborhood is invalid or
    exceeds the structural combination cap."""


@dataclass(frozen=True)
class EmaNeighborhood:
    """A small, preregistered EMA (fast, slow) neighborhood.

    ``combinations()`` and ``neighbors()`` both operate purely on
    *configured array position*, never on numeric spacing -- so a
    non-uniformly-spaced grid (e.g. ``slow=[25, 30, 35]``) still has
    correct adjacency.
    """

    fast: tuple[int, ...]
    slow: tuple[int, ...]
    max_combinations: int = ABSOLUTE_MAX_COMBINATIONS

    def __post_init__(self) -> None:
        if self.max_combinations > ABSOLUTE_MAX_COMBINATIONS:
            raise RobustnessConfigError(
                f"configured max_combinations ({self.max_combinations}) exceeds the "
                f"structural cap ({ABSOLUTE_MAX_COMBINATIONS})"
            )
        combos = self.combinations()
        if len(combos) > self.max_combinations:
            raise RobustnessConfigError(
                f"neighborhood has {len(combos)} valid (fast<slow) combinations, "
                f"exceeding configured max_combinations ({self.max_combinations})"
            )

    def combinations(self) -> list[tuple[int, int]]:
        """All valid (fast < slow) combinations, in configured-array order
        (fast-major, slow-minor) -- this is also the only order any report
        may ever present them in."""
        return [(f, s) for f in self.fast for s in self.slow if f < s]

    def neighbors(self, fast: int, slow: int) -> list[tuple[int, int]]:
        """4-connected neighbors of ``(fast, slow)``: the adjacent
        configured value in ``fast`` (holding ``slow`` fixed) and the
        adjacent configured value in ``slow`` (holding ``fast`` fixed),
        restricted to combinations that are actually valid (fast < slow)
        and present in this neighborhood."""
        fi = self.fast.index(fast)
        si = self.slow.index(slow)
        candidates = []
        if fi > 0:
            candidates.append((self.fast[fi - 1], slow))
        if fi < len(self.fast) - 1:
            candidates.append((self.fast[fi + 1], slow))
        if si > 0:
            candidates.append((fast, self.slow[si - 1]))
        if si < len(self.slow) - 1:
            candidates.append((fast, self.slow[si + 1]))
        valid = set(self.combinations())
        return [c for c in candidates if c in valid]


@dataclass(frozen=True)
class CellResult:
    fast: int
    slow: int
    window_returns: dict[int, float]
    aggregate_return: float


def _run_cell(
    ohlcv: pd.DataFrame,
    fast: int,
    slow: int,
    config: BacktestConfig,
    windows: list[EvaluationWindow],
) -> CellResult:
    signal = generate_signals(ohlcv["close"], fast, slow)["signal"]
    result = run_vectorbt(ohlcv, signal, config)
    timestamps = ohlcv["timestamp"].reset_index(drop=True)
    equity = result.portfolio.value().reset_index(drop=True)

    window_returns: dict[int, float] = {}
    compounded = 1.0
    for window in windows:
        r = window_return(timestamps, equity, window)
        window_returns[window.name] = r
        compounded *= 1.0 + r

    return CellResult(
        fast=fast, slow=slow, window_returns=window_returns, aggregate_return=compounded - 1.0
    )


def run_parameter_grid(
    ohlcv: pd.DataFrame,
    neighborhood: EmaNeighborhood,
    baseline_cost: BacktestConfig,
    windows: list[EvaluationWindow],
) -> list[CellResult]:
    """Run every configured (fast, slow) combination at ``baseline_cost``
    only, via VectorBT. Returned in configured-grid order -- never sorted
    by any performance value.

    Defensively re-filters ``ohlcv`` to ``RESEARCH_ALLOWED_SPLITS`` itself
    (see ``strategy_lab.validation.temporal_validation.filter_to_allowed_research_history``)
    before any signal/backtest construction, rather than trusting the
    caller to have already done so.
    """
    ohlcv = filter_to_allowed_research_history(ohlcv)
    return [
        _run_cell(ohlcv, fast, slow, baseline_cost, windows)
        for fast, slow in neighborhood.combinations()
    ]


@dataclass(frozen=True)
class FragilityFlag:
    cell: tuple[int, int]
    detail: str


HEURISTIC_LABEL = "preregistered heuristic flag, not a significance test"
ISOLATED_CELL_THRESHOLD_MULTIPLIER = 2.0


@dataclass(frozen=True)
class FragilityReport:
    sign_consistency_fraction: float
    sign_flip_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    baseline_cell: tuple[int, int]
    baseline_return: float
    baseline_neighbor_returns: dict[tuple[int, int], float]
    window_dispersion: dict[int, dict[str, float]]
    isolated_cell_flags: list[FragilityFlag] = field(default_factory=list)
    heuristic_label: str = HEURISTIC_LABEL


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_fragility_report(
    neighborhood: EmaNeighborhood,
    grid_results: list[CellResult],
    baseline_cell: tuple[int, int] = (10, 30),
) -> FragilityReport:
    """Descriptive, mechanical fragility diagnostics over the parameter
    grid -- no statistical inference, no ranking, no winner selection."""
    by_cell = {(r.fast, r.slow): r for r in grid_results}

    # 1. Sign consistency across neighboring cells (each adjacent pair
    #    counted once).
    seen_pairs: set[frozenset[tuple[int, int]]] = set()
    matching = 0
    total_pairs = 0
    sign_flip_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell, result in by_cell.items():
        for neighbor in neighborhood.neighbors(*cell):
            pair_key = frozenset((cell, neighbor))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            total_pairs += 1
            neighbor_result = by_cell[neighbor]
            same_sign = (result.aggregate_return >= 0) == (neighbor_result.aggregate_return >= 0)
            if same_sign:
                matching += 1
            else:
                sign_flip_pairs.append((cell, neighbor))
    sign_consistency_fraction = matching / total_pairs if total_pairs else 1.0

    # 2. Baseline position within its local neighborhood.
    baseline_result = by_cell[baseline_cell]
    baseline_neighbor_returns = {
        neighbor: by_cell[neighbor].aggregate_return
        for neighbor in neighborhood.neighbors(*baseline_cell)
    }

    # 3. Dispersion of window returns across parameter cells.
    window_dispersion: dict[int, dict[str, float]] = {}
    if grid_results:
        years = sorted(grid_results[0].window_returns)
        for year in years:
            values = [r.window_returns[year] for r in grid_results]
            window_dispersion[year] = {
                "min": min(values),
                "median": _median(values),
                "max": max(values),
            }

    # 4. Isolated-cell heuristic (preregistered, mechanical, not a test).
    neighbor_diffs: list[float] = []
    for cell, result in by_cell.items():
        for neighbor in neighborhood.neighbors(*cell):
            neighbor_diffs.append(abs(result.aggregate_return - by_cell[neighbor].aggregate_return))
    global_mean_abs_diff = sum(neighbor_diffs) / len(neighbor_diffs) if neighbor_diffs else 0.0
    threshold = ISOLATED_CELL_THRESHOLD_MULTIPLIER * global_mean_abs_diff

    isolated_flags: list[FragilityFlag] = []
    for cell, result in by_cell.items():
        neighbors = neighborhood.neighbors(*cell)
        if not neighbors:
            continue
        neighbor_mean = sum(by_cell[n].aggregate_return for n in neighbors) / len(neighbors)
        diff = abs(result.aggregate_return - neighbor_mean)
        if diff > threshold:
            isolated_flags.append(
                FragilityFlag(
                    cell=cell,
                    detail=(
                        f"differs from neighbor mean ({neighbor_mean:.4%}) by {diff:.4%}, "
                        f"exceeding the preregistered threshold ({threshold:.4%} = "
                        f"{ISOLATED_CELL_THRESHOLD_MULTIPLIER}x the grid's mean "
                        "neighbor-to-neighbor difference)"
                    ),
                )
            )

    return FragilityReport(
        sign_consistency_fraction=sign_consistency_fraction,
        sign_flip_pairs=sign_flip_pairs,
        baseline_cell=baseline_cell,
        baseline_return=baseline_result.aggregate_return,
        baseline_neighbor_returns=baseline_neighbor_returns,
        window_dispersion=window_dispersion,
        isolated_cell_flags=isolated_flags,
    )


@dataclass(frozen=True)
class CostStressResult:
    scenario: str
    commission_rate: float
    slippage_rate: float
    window_returns: dict[int, float]
    aggregate_return: float


def run_cost_stress(
    ohlcv: pd.DataFrame,
    cost_scenarios: dict[str, dict[str, float]],
    windows: list[EvaluationWindow],
    baseline_cell: CellResult,
    baseline_key: str = "baseline",
    fast: int = 10,
    slow: int = 30,
    initial_cash: float = 100_000.0,
) -> list[CostStressResult]:
    """Run EMA(``fast``, ``slow``) under each of ``cost_scenarios``.

    The ``baseline_key`` scenario is *not* re-executed -- it reuses
    ``baseline_cell`` (already computed by the parameter grid at the same
    cost), so this function only performs new work for the non-baseline
    (worse) scenarios. This is what keeps total strategy executions at
    ``len(grid) + (len(cost_scenarios) - 1)``, never
    ``len(grid) * len(cost_scenarios)``.
    """
    ohlcv = filter_to_allowed_research_history(ohlcv)
    results = []
    for name, cost in cost_scenarios.items():
        if name == baseline_key:
            results.append(
                CostStressResult(
                    scenario=name,
                    commission_rate=cost["commission_rate"],
                    slippage_rate=cost["slippage_rate"],
                    window_returns=baseline_cell.window_returns,
                    aggregate_return=baseline_cell.aggregate_return,
                )
            )
            continue
        config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=cost["commission_rate"],
            slippage_rate=cost["slippage_rate"],
        )
        cell = _run_cell(ohlcv, fast, slow, config, windows)
        results.append(
            CostStressResult(
                scenario=name,
                commission_rate=cost["commission_rate"],
                slippage_rate=cost["slippage_rate"],
                window_returns=cell.window_returns,
                aggregate_return=cell.aggregate_return,
            )
        )
    return results


def run_parity_spot_checks(
    ohlcv: pd.DataFrame,
    config: BacktestConfig,
    spot_checks: tuple[tuple[int, int], ...] = PARITY_SPOT_CHECKS,
) -> dict[tuple[int, int], ParityReport]:
    """Reference-engine-vs-VectorBT parity for exactly the predefined
    ``spot_checks`` configurations -- fixed before any result is observed,
    never selected based on observed grid results, never the full grid."""
    ohlcv = filter_to_allowed_research_history(ohlcv)
    results = {}
    for fast, slow in spot_checks:
        signal = generate_signals(ohlcv["close"], fast, slow)["signal"]
        results[(fast, slow)] = compare_engines(ohlcv, signal, config)
    return results
