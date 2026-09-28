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
are intentionally not implemented.** No connection exists to Coinbase,
TradingView, or any broker/exchange. Such integrations may only be added in
future work, deliberately and incrementally, once the research and validation
layers are mature enough to justify them. Nothing in this milestone should be
read as a claim that the EMA crossover strategy is profitable or
production-ready — it is test scaffolding.

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

## Setting up the Python environment

Requires Python 3.11+ (developed against 3.14 locally; check with `python3 --version`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

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
