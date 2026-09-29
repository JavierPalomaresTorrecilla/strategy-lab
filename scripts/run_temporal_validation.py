#!/usr/bin/env python3
"""Human-run script: Milestone 3 fixed-parameter expanding-history temporal
validation report for the fixed EMA 10/30 SPY strategy.

OFFLINE: this script never performs network I/O. It consumes an
already-existing immutable processed SPY snapshot under
``STRATEGY_LAB_DATA_ROOT``, produced earlier by ``scripts/fetch_dataset.py``
(the only network-dependent script in this project). If no suitable
snapshot exists, this script fails with a clear message naming that
script -- it never fetches data itself.

Reuses ``scripts/run_parity_report.py``'s existing
``_find_latest_processed_snapshot`` snapshot-selection logic rather than
inventing a second selector.

TEST is methodologically sealed: this script only ever computes over the
"train" + "validation" (2010-2022) history and never constructs a signal,
backtest, or metric from "test"-partition rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from strategy_lab.backtest.engine import BacktestConfig, run_backtest
from strategy_lab.data.env import DataRootError, resolve_data_root
from strategy_lab.data.processed import load_processed_snapshot
from strategy_lab.data.snapshot import compute_sha256
from strategy_lab.data.splits import load_splits
from strategy_lab.strategies.ema_crossover import generate_signals
from strategy_lab.validation import benchmark as benchmark_mod
from strategy_lab.validation import corporate_actions, robustness
from strategy_lab.validation import registry as registry_mod
from strategy_lab.validation import robustness_metrics as rm
from strategy_lab.validation import temporal_validation as tv
from strategy_lab.validation.parity import ParityReport

SYMBOL = "SPY"
INTERVAL = "1d"
BASELINE_FAST = 10
BASELINE_SLOW = 30
INITIAL_CASH = 100_000.0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_PARITY_REPORT_PATH = _REPO_ROOT / "scripts" / "run_parity_report.py"
_CONFIG_PATH = _REPO_ROOT / "config" / "research.yaml"
# The repository's existing, gitignored generated-artifact location (see
# reports/README.md and .gitignore's `reports/generated/` entry) -- not a
# second, M3-only hierarchy under experiments/.
_REPORTS_DIR = _REPO_ROOT / "reports" / "generated"
_REGISTRY_PATH = _REPO_ROOT / "experiments" / "registry.csv"


def _load_run_parity_report():
    spec = importlib.util.spec_from_file_location("run_parity_report", _RUN_PARITY_REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _json_safe(obj):
    if isinstance(obj, (rm.InsufficientSample, rm.NotApplicable)):
        return {
            "type": type(obj).__name__,
            **{f.name: getattr(obj, f.name) for f in dc_fields(obj)},
        }
    if isinstance(obj, ParityReport):
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _json_safe(getattr(obj, f.name)) for f in dc_fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def main() -> int:
    try:
        data_root = resolve_data_root()
    except DataRootError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    run_parity_report = _load_run_parity_report()
    try:
        processed_ref = run_parity_report._find_latest_processed_snapshot(data_root)
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        print(
            "Run `python scripts/fetch_dataset.py` first to produce a processed "
            "snapshot -- this script never fetches data itself.",
            file=sys.stderr,
        )
        return 1

    canonical_df = load_processed_snapshot(processed_ref)
    print(f"loaded processed snapshot: {processed_ref.parquet_path} ({len(canonical_df)} rows)")

    raw_ref = corporate_actions.resolve_raw_ref_from_processed(processed_ref, data_root)
    try:
        ca_result = corporate_actions.check_corporate_actions(raw_ref)
    except corporate_actions.UnsupportedCorporateActionError as exc:
        print(f"FAILED (unsupported corporate action): {exc}", file=sys.stderr)
        return 1
    print(
        f"corporate-action check: splits verified zero = "
        f"{ca_result.stock_splits_verified_zero_in_research_range}, "
        f"dividend events in research range = {ca_result.dividend_event_count_in_research_range}"
    )

    with _CONFIG_PATH.open() as f:
        config_yaml = yaml.safe_load(f)
    tv_config = config_yaml["temporal_validation"]
    robustness_config = config_yaml["robustness"]

    # Capture Git/config/dependency provenance now -- after the artifacts
    # this run is about to use are fully resolved and validated, but
    # strictly before any output (report file, registry row, or any other
    # repository-side write) is produced. This is what makes
    # `working_tree_clean` mean "clean immediately before this run's own
    # output artifacts were written" rather than being self-polluted by
    # those very writes. See `registry.git_commit_and_clean`'s docstring.
    git_commit, working_tree_clean = registry_mod.git_commit_and_clean(_REPO_ROOT)
    config_sha256 = compute_sha256(_CONFIG_PATH)
    dependency_versions = _dependency_versions()

    splits = load_splits()
    research_df = tv.filter_to_allowed_research_history(canonical_df, splits)

    burn_in_window, evaluation_windows = tv.build_windows(
        burn_in_year=tv_config["burn_in_year"], evaluation_years=list(tv_config["evaluation_years"])
    )
    risk_free_rate = tv_config["risk_free_rate"]

    baseline_cost = robustness_config["cost_stress"]["baseline"]
    baseline_config = BacktestConfig(initial_cash=INITIAL_CASH, **baseline_cost)

    signal = generate_signals(research_df["close"], BASELINE_FAST, BASELINE_SLOW)["signal"]
    backtest_result = run_backtest(research_df, signal, baseline_config)

    tv_result = tv.evaluate(
        backtest_result, burn_in_window, evaluation_windows, risk_free_rate=risk_free_rate
    )

    print(
        f"\n=== Baseline EMA({BASELINE_FAST},{BASELINE_SLOW}), evaluation "
        f"{evaluation_windows[0].name}-{evaluation_windows[-1].name} ==="
    )
    for w in tv_result.window_metrics:
        print(
            f"  {w.year}: return={w.total_return:.4%} local_dd={w.local_max_drawdown:.4%} "
            f"exposure={w.exposure:.2%} round_trips={w.completed_round_trip_count} "
            f"cross_boundary={w.cross_boundary_count}"
        )
    agg = tv_result.aggregate_metrics
    print(
        f"  aggregate: CAGR={agg.cagr:.4%} max_dd={agg.max_drawdown:.4%} "
        f"exposure={agg.exposure:.2%} eligible_round_trips={agg.eligible_round_trip_count} "
        f"hit_rate={agg.hit_rate} profit_factor={agg.profit_factor} sharpe_like={agg.sharpe_like}"
    )
    print(
        f"  leave-one-window-out: sign_flip_years={tv_result.leave_one_window_out.sign_flip_years}"
    )
    print(f"  trade concentration: {tv_result.trade_concentration}")

    neighborhood = robustness.EmaNeighborhood(
        fast=tuple(robustness_config["ema_neighborhood"]["fast"]),
        slow=tuple(robustness_config["ema_neighborhood"]["slow"]),
        max_combinations=robustness_config["ema_neighborhood"]["max_combinations"],
    )
    grid_results = robustness.run_parameter_grid(
        research_df, neighborhood, baseline_config, evaluation_windows
    )
    fragility = robustness.build_fragility_report(
        neighborhood, grid_results, baseline_cell=(BASELINE_FAST, BASELINE_SLOW)
    )
    print(f"\n=== Parameter neighborhood ({len(grid_results)} cells) ===")
    print(f"  sign consistency: {fragility.sign_consistency_fraction:.2%}")
    print(f"  sign-flip pairs: {fragility.sign_flip_pairs}")
    print(f"  isolated-cell flags ({fragility.heuristic_label}): {fragility.isolated_cell_flags}")

    baseline_cell = next(
        c for c in grid_results if (c.fast, c.slow) == (BASELINE_FAST, BASELINE_SLOW)
    )
    cost_results = robustness.run_cost_stress(
        research_df, robustness_config["cost_stress"], evaluation_windows, baseline_cell
    )
    print("\n=== Cost stress (EMA 10,30) ===")
    for c in cost_results:
        print(f"  {c.scenario}: aggregate_return={c.aggregate_return:.4%}")

    bench = benchmark_mod.run_price_only_benchmark(
        research_df,
        evaluation_windows[0].start,
        evaluation_windows[-1].end,
        initial_cash=INITIAL_CASH,
        **baseline_cost,
    )
    print(f"\n=== Benchmark ({bench.label}) ===")
    print(f"  entry {bench.entry_timestamp} @ {bench.entry_exec_price:.4f}")
    print(
        f"  final {bench.final_timestamp} @ {bench.final_close:.4f}, "
        f"total_return={bench.total_return:.4%}"
    )
    print(f"  {bench.dividend_disclaimer}")
    print(f"  {bench.initialization_asymmetry}")
    print(f"  permitted claim: {bench.permitted_claim}")

    spot_checks_config = [tuple(pair) for pair in robustness_config["parity_spot_checks"]]
    spot_checks = robustness.run_parity_spot_checks(
        research_df, baseline_config, tuple(spot_checks_config)
    )
    print("\n=== Reference/VectorBT parity spot checks ===")
    for (f, s), report in spot_checks.items():
        status = "OK" if report.is_within_tolerance() else "DIVERGED"
        print(f"  ({f},{s}): {status} (divergences={len(report.divergences)})")

    # -- Report artifact + registry record --
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    experiment_id = f"m3_temporal_validation_{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}"

    report_payload = {
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at.isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "processed_snapshot_sha256": processed_ref.sha256,
        "source_raw_sha256": raw_ref.sha256,
        "corporate_action_check": _json_safe(ca_result),
        "baseline_params": {"fast": BASELINE_FAST, "slow": BASELINE_SLOW},
        "baseline_cost": baseline_cost,
        "temporal_validation": _json_safe(tv_result),
        "parameter_neighborhood": {
            "combinations": neighborhood.combinations(),
            "grid_results": _json_safe(grid_results),
            "fragility": _json_safe(fragility),
        },
        "cost_stress": _json_safe(cost_results),
        "benchmark": _json_safe(bench),
        "parity_spot_checks": {
            f"{f}_{s}": report.to_dict() for (f, s), report in spot_checks.items()
        },
        "dependency_versions": dependency_versions,
    }
    report_path = _REPORTS_DIR / f"{experiment_id}.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True, default=str))
    print(f"\nreport artifact: {report_path}")

    record_status = "finalized" if (git_commit and working_tree_clean) else "dev"

    record = registry_mod.RegistryRecord(
        experiment_id=experiment_id,
        date=generated_at.date().isoformat(),
        git_commit=git_commit,
        working_tree_clean=working_tree_clean,
        record_status=record_status,
        hypothesis="EMA crossover exhibits temporally consistent, cost-robust behavior on SPY",
        strategy_family_id="ema_crossover",
        strategy_id=f"ema_crossover_{BASELINE_FAST}_{BASELINE_SLOW}",
        data_start=str(evaluation_windows[0].start.date()),
        data_end=str(evaluation_windows[-1].end.date()),
        split="train+validation",
        processed_snapshot_sha256=processed_ref.sha256,
        source_raw_sha256=raw_ref.sha256,
        config_sha256=config_sha256,
        window_ids=f"{evaluation_windows[0].name}..{evaluation_windows[-1].name}",
        params_json=json.dumps({"fast": BASELINE_FAST, "slow": BASELINE_SLOW}),
        cost_scenario="baseline",
        outcome="see report artifact",
        metrics_report_path=str(report_path.relative_to(_REPO_ROOT)),
        dependency_versions_json=registry_mod.dependency_versions_json(dependency_versions),
        test_evaluated=False,
        test_evaluated_date="",
        test_evaluation_experiment_id="",
        notes="Milestone 3 fixed-parameter expanding-history temporal validation",
    )
    registry_mod.append_record(_REGISTRY_PATH, record)
    print(f"registry record: {experiment_id} (record_status={record_status})")

    # TEST is intentionally never loaded above.
    return 0


def _dependency_versions() -> dict[str, str | None]:
    import platform

    import pandas
    import pyarrow

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
    }
    try:
        import vectorbt

        versions["vectorbt"] = vectorbt.__version__
    except ImportError:
        versions["vectorbt"] = None
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
