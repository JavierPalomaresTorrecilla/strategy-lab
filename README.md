# strategy-lab

A systematic market research and strategy-validation laboratory.

## Purpose

`strategy-lab` exists to test investment and trading hypotheses rigorously,
using historical data and disciplined statistical practice. It is a research
platform, not a trading system.

A central working assumption of this project is that **historical
profitability does not imply future profitability**. The project is
deliberately structured to reduce common sources of false confidence:

- look-ahead bias
- survivorship bias
- data leakage
- overfitting and parameter mining
- unrealistic execution assumptions
- incorrect transaction-cost assumptions
- accidental use of future information

A backtest that looks good is treated as a hypothesis that survived one test,
not as proof that a strategy has edge.

## Architecture

The codebase is organized into layers with distinct responsibilities:

| Layer | Location | Responsibility |
|---|---|---|
| Data | `src/strategy_lab/data/` | Ingesting, cleaning, and storing market data |
| Indicators | `src/strategy_lab/indicators/` | Reusable, side-effect-free transforms of data |
| Strategies | `src/strategy_lab/strategies/` | Hypotheses and signal-generation logic |
| Backtest | `src/strategy_lab/backtest/` | Historical simulation of strategies |
| **Validation** | `src/strategy_lab/validation/` | Out-of-sample testing, robustness, significance checks |
| Portfolio | `src/strategy_lab/portfolio/` | Position sizing and multi-strategy allocation |
| **Risk** | `src/strategy_lab/risk/` | Exposure limits and drawdown controls |
| **Execution** | `src/strategy_lab/execution/` | Order translation and broker/exchange interaction |

**Research vs. validation vs. risk vs. execution** are kept as separate
concerns on purpose:

- **Research** (data → indicators → strategies → backtest) is where
  hypotheses are formed and tested. It is optimistic by construction and
  must not be trusted on its own.
- **Validation** exists specifically to challenge research results: held-out
  test periods, robustness across parameter neighborhoods, and statistical
  scrutiny. A strategy that hasn't passed validation isn't a strategy yet.
- **Risk** defines limits independent of any single strategy's backtest
  (position sizing, leverage, drawdown thresholds) so that a good-looking
  backtest can never bypass account-level safety controls.
- **Execution** is the (currently unimplemented) boundary where a validated,
  risk-checked decision would eventually become a real order. Keeping it
  separate means research and validation code never needs to know about
  broker APIs, and no research code can accidentally place a trade.

## Research principles

These principles apply to any research or code contributed to this project:

- Test hypotheses; do not search for attractive backtests.
- Never use information that would not have been available at decision time
  (no look-ahead bias, no survivorship bias, no leaked future data).
- Include realistic transaction costs (commissions, slippage, spread)
  whenever a result depends on them.
- Once strategy research begins on a dataset, explicitly separate train,
  validation, and test periods. The test period stays untouched until a
  strategy is finalized on train/validation data.
- Prefer parameter regions that are robust across neighboring values over a
  single isolated "optimal" value.
- Record assumptions and experiment provenance (see `experiments/registry.csv`).
- Treat a good backtest as a hypothesis that survived one test, not as proof
  of future profitability. High historical returns alone are not evidence of
  edge.

## Current development phase

**Milestone 1: minimum deterministic research pipeline.** This milestone
exists to validate the plumbing of the research pipeline — not to find a
profitable strategy. It adds:

- a canonical OHLCV validation layer (`strategy_lab.data.ohlcv`);
- an EMA indicator (`strategy_lab.indicators.ema`);
- a deliberately simple long-only EMA crossover strategy
  (`strategy_lab.strategies.ema_crossover`), used only to exercise the
  pipeline;
- a deterministic, long-only, single-instrument backtest engine
  (`strategy_lab.backtest.engine`) with configurable commission and
  slippage;
- a minimal metrics layer (`strategy_lab.backtest.metrics`): total return,
  maximum drawdown, and completed trade count.

All data used in tests is deterministic synthetic OHLCV data. No external
market data is downloaded.

**Live trading, paper trading, autonomous trading agents, and order execution
are intentionally not implemented.** No exchange or broker execution
connectivity exists. Such integrations may only be added in future work,
deliberately and incrementally, once the research and validation layers are
mature enough to justify them. Nothing in this milestone should be read as a
claim that the EMA crossover strategy is profitable or production-ready — it
is test scaffolding.

### Execution timing semantics

The backtest engine treats `signal[i]` as the desired position *as of the
close of bar i*. That signal cannot execute using bar i's own close, or any
earlier price — it is shifted forward and executes no earlier than **bar
i + 1's open**. Bar i+1's open is used (rather than its close) because it is
the first price at which a decision made at bar i's close could plausibly be
acted on; using bar i+1's close would grant the strategy an extra bar of
information it could not have had in practice. See
`strategy_lab.backtest.engine` for the full rationale and
`tests/test_backtest_engine.py` for a fixture-based proof that a crossover
known only at close T does not execute at close T.

**Milestone 2: real historical data, reproducibility, and cross-engine
parity.** This milestone is about trustworthy data plumbing and validating
the reference engine against a second, independent implementation — it is
**not** about finding profitable parameters. It adds:

- an external, immutable snapshot store for market data
  (`strategy_lab.data.env`, `strategy_lab.data.snapshot`,
  `strategy_lab.data.processed`): large datasets live outside this
  repository, under the directory named by `STRATEGY_LAB_DATA_ROOT`, never
  as a silent fallback inside the repo or on the internal disk;
- a narrow, provider-agnostic market-data interface
  (`strategy_lab.data.providers`) with one concrete, DEVELOPMENT-only
  provider (`strategy_lab.data.providers.yfinance_provider`) — not a plugin
  framework, and not an authoritative production data source;
- raw-vs-canonical separation (`strategy_lab.data.canonical`): the raw
  provider snapshot is preserved unmodified; conversion to the canonical
  OHLCV schema is a separate, pure step that fails loudly rather than
  repairing malformed data;
- YAML-driven chronological TRAIN/VALIDATION/TEST partitions
  (`strategy_lab.data.splits`, boundaries defined once in
  `config/research.yaml`). **TEST is methodologically sealed** in this
  milestone: it is loaded and validated, but no real-data strategy result is
  computed on it — see `strategy_lab.data.splits.RESEARCH_ALLOWED_SPLITS`
  and `scripts/run_parity_report.py`;
- a second, independent quantitative engine — VectorBT
  (`strategy_lab.validation.vectorbt_adapter`) — explicitly configured to
  match the reference engine's next-open execution timing, and a
  deterministic cross-engine parity report
  (`strategy_lab.validation.parity`) that surfaces and explains any
  divergence rather than forcing agreement by adjusting either
  implementation.

The reference engine (`strategy_lab.backtest.engine`) is unmodified by
Milestone 2.

### Optional `parity` dependency group

VectorBT (and its `numba`/`llvmlite` dependency chain) is kept in an
optional `parity` extra so the core package install stays light:

```bash
pip install -e ".[dev,parity]"
```

`tests/test_parity.py` imports VectorBT directly (no `pytest.importorskip`)
— if the `parity` extra isn't installed, that one test file fails to
collect visibly. This is intentional: Milestone 2 acceptance on a properly
set up machine requires the `parity` extra installed and
`tests/test_parity.py` passing, not silently skipped. The rest of the suite
requires neither VectorBT nor network access.

### Fetching and validating the real dataset (human-run only)

Two scripts, never run automatically by tests or library code:

```bash
# requires STRATEGY_LAB_DATA_ROOT set to a writable external directory,
# and network access to Yahoo Finance
python scripts/fetch_dataset.py

# requires a processed snapshot from the script above, and the `parity` extra
python scripts/run_parity_report.py
```

`run_parity_report.py` only ever reads the `train` and `validation`
partitions — it never constructs a signal, backtest, or parity comparison
from `test`-partition rows.

**Milestone 3: fixed-parameter expanding-history temporal validation.**
This milestone is about defining a validation methodology that makes
overfitting and self-deception harder — it is **not** about finding
profitable parameters, and it introduces no parameter optimization or
performance ranking anywhere. It adds:

- **temporal validation** (`strategy_lab.validation.temporal_validation`):
  one continuous, causal simulation over the full TRAIN ∪ VALIDATION range
  (2010–2022). 2010 is a burn-in/state-initialization period — EMA and
  position state evolve causally through it, but it contributes zero
  evaluation statistics. Evaluation is exactly the 12 contiguous
  calendar-year windows 2011–2022, with no capital reset at any boundary
  and no forced liquidation of a position that spans a year end;
- **round-trip accounting** (`strategy_lab.validation.round_trips`): closed
  trades reconstructed from the unmodified Milestone 1 `Trade` records —
  window *performance* always comes from the continuous mark-to-market
  equity curve, never from attributing a whole trade's P&L to whichever
  calendar year it happened to exit in;
- **parameter/cost sensitivity** (`strategy_lab.validation.robustness`): a
  small, preregistered, hard-capped (≤25) 15-cell EMA neighborhood around
  the (10, 30) baseline, and a 3-tier cost-stress matrix whose baseline
  exactly reproduces `scripts/run_parity_report.py`'s cost assumptions.
  The two axes are never crossed (15 + 2 runs, never 45), and no report
  ever ranks, sorts by performance, or selects a "best" configuration —
  only descriptive, mechanically-labeled fragility diagnostics;
  three fixed reference-engine/VectorBT parity spot checks
  (`(8,25)`, `(10,30)`, `(12,35)`) are predefined before any result is
  observed, never chosen after seeing the grid;
- **one benchmark** (`strategy_lab.validation.benchmark`): price-only
  buy-and-hold on the same canonical raw price series, at the same
  baseline cost assumptions, with no dividend adjustment and no synthetic
  terminal sale — explicitly labeled `"price-only; dividends excluded"`
  and never presented as an economically complete comparison;
- a **corporate-action runtime guard**
  (`strategy_lab.validation.corporate_actions`) that fails the real-data
  report loudly if a non-zero stock split is found in the allowed research
  range, rather than silently assuming raw OHLC execution handles it;
- an **experiment registry** (`strategy_lab.validation.registry`) extending
  `experiments/registry.csv` with a `finalized`/`dev` reproducibility
  contract and Git-revision/snapshot/config lineage — see
  `experiments/README.md`.

The reference engine (`strategy_lab.backtest.engine`,
`strategy_lab.backtest.metrics`, `strategy_lab.backtest.trades`) is
unmodified by Milestone 3.

### Running the offline temporal-validation report (human-run only)

```bash
# OFFLINE -- consumes an already-existing processed snapshot; never fetches
# data itself. Requires STRATEGY_LAB_DATA_ROOT and the `parity` extra.
# If no processed snapshot exists yet, run scripts/fetch_dataset.py first.
python scripts/run_temporal_validation.py
```

Unlike `scripts/fetch_dataset.py`, this script performs no network access
at all — `scripts/fetch_dataset.py` remains the only network-dependent
script in this project.

## Setting up the Python environment

Requires Python 3.11+ (developed against 3.14 locally; check with `python3 --version`).

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install one of:

- Core development dependencies only (no network, no VectorBT):

  ```bash
  pip install -e ".[dev]"
  ```

- Full Milestone 2 acceptance, including VectorBT parity:

  ```bash
  pip install -e ".[dev,parity]"
  ```

## Running tests

```bash
pytest
```

Running the entire suite requires the `parity` extra: `tests/test_parity.py`
intentionally imports VectorBT directly, with no `pytest.importorskip`, so
that a missing `parity` extra fails collection visibly instead of silently
skipping. `pip install -e ".[dev]"` alone will not produce a fully green
`pytest` run.

## Minimal example

```python
import pandas as pd

from strategy_lab.backtest import BacktestConfig, run_backtest
from strategy_lab.data import validate_ohlcv
from strategy_lab.strategies import generate_signals

ohlcv = validate_ohlcv(my_raw_dataframe)  # columns: timestamp, open, high, low, close, volume
signals = generate_signals(ohlcv["close"], fast_period=10, slow_period=30)

result = run_backtest(
    ohlcv,
    signals["signal"],
    BacktestConfig(initial_cash=100_000.0, commission_rate=0.001, slippage_rate=0.0005),
)

print(result.final_equity, result.metrics)
```

## Running linting

```bash
ruff check .
```

## Configuration

- `config/research.yaml` — research/simulation defaults (data paths, train/
  validation/test split placeholders, cost assumptions, backtest defaults,
  experiment registry).
- `config/risk.yaml` — risk limits for simulation only; documents (but does
  not enable) future real-money risk controls.
- `.env.example` — template for environment variables; copy to `.env` (never
  committed) when credentials are actually needed. No integrations exist yet,
  so no real values are required today.

## Experiment tracking

Every research experiment should be logged in `experiments/registry.csv` with
its hypothesis, configuration, data range, and outcome, so that results remain
auditable and parameter mining is visible rather than hidden.

## Contributing

Before editing a file, read its current contents rather than assuming what
state it's in — changes may already be in progress. Preserve unrelated
work when making a change, and avoid destructive operations (discarding
uncommitted changes, resetting history) unless that is specifically the task
at hand.

Never commit or expose credentials, API keys, tokens, or other secrets.
`.env` is gitignored; only `.env.example` (a template with no real values)
is tracked.
