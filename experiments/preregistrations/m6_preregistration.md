# M6: preregistration (FROZEN)

Status: FROZEN on 2026-10-26.
- The methodology and costs were agreed on 2026-10-19 (draft v3). The only value filled in afterwards is the dataset end date, taken from the M5c verdict under its preregistered rule; see `experiments/manifests/m5c_revision_audit.json`.
- It is built from a read-only repo inventory at HEAD 2c49a4e.
- **No ADA or DOT strategy performance had been computed or inspected when this was frozen.**

Item status markers: **[existing]** = already in the code, with evidence; **[decided]** = approved by Javi.

## 1. Question

Characterize how the existing long-only EMA crossover, run unchanged, behaves on ADA-EUR and DOT-EUR daily data compared with buy-and-hold, after realistic costs. Each product is assessed on its own. Nothing is pooled, and no product is chosen after results are seen.

## 2. Data

- The baseline is the M5b processed snapshots fetched 2026-10-19 (M5b manifest, 332a527).
- **[decided by M5c rule] M5c verdict: PASS for both ADA-EUR and DOT-EUR**, so the overall verdict is PASS. There were 0 semantic and 0 format-only differences, comparing the 2026-10-19 baseline with the 2026-10-26 audit. Embargo = 0 days.
- **Frozen dataset:** the M5b 2026-10-19 baseline snapshots, unchanged.
  - The last included bucket is **2026-10-18**, and the half-open end is **2026-10-19**.
  - Audit-only candles from 2026-10-19 to 2026-10-25 are **not** admitted.
- This supports only "no revisions or representation drift were observed over this 7-day audit". It is not a general claim about Coinbase revision behaviour.
- **[decided]** Any gap in the frozen dataset stops M6. There are currently 0 gaps, and none are ever filled.
- Prices are quoted in EUR, so returns are EUR returns, with no FX conversion.
- Timestamps are UTC day buckets, through the Coinbase path, not canonical.py.

## 3. Strategy and parameters [existing; decided: unchanged]

- The only strategy is `strategies/ema_crossover.py:37`: long while fast EMA > slow EMA, otherwise flat. During warm-up the signal is 0.
- The baseline is (10,30) (`config/research.yaml:127-128`).
- The grid is fast [8..12] x slow [25,30,35] = 15 cells. It is for sensitivity only, never for picking a winner (`research.yaml:86-89`, cap at `robustness.py:29`).
- No new strategy, no new parameters, and no grid change.

## 4. Splits [decided]

| Segment | Range (half-open) |
|---|---|
| Burn-in | listing → 2022-01-01 (excluded from evaluation) |
| TRAIN | [2022-01-01, 2024-01-01) |
| VALIDATION | [2024-01-01, 2025-01-01) |
| TEST (sealed) | [2025-01-01, 2026-10-19) |

- There is one continuous backtest from listing with no state reset, and evaluation is by window. This is the existing M3 rule (`temporal_validation.py:144-181`).
- The alternative of a TEST from 2025-07-01 was rejected.
- **Prior-exposure note:** the two largest daily moves recorded in M5b (ADA 2025-03-02 and DOT 2026-02-25) fall inside TEST. Only these raw moves are known; no strategy result is. The boundaries are not moved because of this.
- **InsufficientSample is a valid, reported result.** It never triggers window merging, metric substitution, threshold changes or boundary changes after execution.

## 5. TEST hard seal [decided]

- The research runner cannot read TEST at all, not merely by convention. Only TRAIN and VALIDATION can be reached in research mode (builds on `splits.py:33`).
- Opening TEST is a separate, explicit action, run exactly once. It happens only after the VALIDATION conclusions are committed.
- The M6 manifest records commit X (methodology plus VALIDATION conclusions frozen) and run Y (TEST opened).
- The TEST result is reported whatever it shows. No re-tuning follows.

## 6. Execution and costs

- [existing] A signal at close T fills at open T+1 (`engine.py:131-133`). Sizing is binary, with no leverage. Commission and slippage are proportional.
- **[decided] Verified fee observation (Javi, 2026-10-19):** the source is the Coinbase Advanced fee-tier page, logged in. The current tier is **Intro**. Spot maker fee is 0.25% (25 bps) and spot taker fee is 0.50% (50 bps). For ADA-EUR and DOT-EUR, the result is "none observed on the fee-tier page". That is not a claim that pair exceptions never exist. This value is frozen for M6 even if the tier changes later.
- **[decided] Commission = 50 bps taker per execution in every scenario.** Maker is not used, because the engine assumes a guaranteed next-open fill, which is a taker execution. The old SPY 10 bps is not reused.
- **[decided] Cost stress varies slippage only.** Commission stays at the verified 50 bps; the old 2.5x and 5x commission multipliers were rejected as ungrounded.

| Scenario | Commission | Slippage |
|---|---|---|
| baseline | 50 bps | 5 bps |
| moderately_worse | 50 bps | 15 bps |
| materially_worse | 50 bps | 40 bps |

- The slippage values are **preregistered scenario assumptions, not measured Coinbase slippage**.
- Model arithmetic only: the approximate round-trip cost under the current proportional model is 2 x (50 bps commission + 5 bps slippage), about 110 bps of notional. It is not an observed Coinbase execution cost, not a profitability threshold and not a strategy result.
- Buy-and-hold uses the same cost definition. It trades once, so its total cost is naturally lower than the strategy's. That difference is part of the comparison and is not normalized away.

## 7. Robustness design: grid and costs are NOT crossed [decided; preserves M3]

For each product:
- **Parameter sensitivity:** all 15 EMA cells at baseline cost only.
- **Cost sensitivity:** EMA(10,30) only, at baseline, moderately worse and materially worse cost.
- (10,30) at baseline is already a grid cell, so that is **17 unique configurations per product, 34 in total.** This multiplicity is stated in every report.
- Every preregistered result is reported, including negative results and InsufficientSample. Nothing is selected for display.

## 8. Metrics and annualization

- [existing] The M1 metrics (`metrics.py`) and the M3 metrics (`robustness_metrics.py:41-129`) keep their InsufficientSample guards. The risk-free rate is 0.
- **[decided]** Periods-per-year becomes an explicit parameter:
  - legacy/SPY default is 252, and every existing SPY output stays byte-identical;
  - Coinbase daily passes 365 explicitly;
  - leap days stay as observations, but the convention stays 365.
- Size of the current error: on crypto, the 252 convention gives volatility and Sharpe-like values about 16.9% lower than the 365 convention. Put the other way, the correct value is about 20.4% higher.
- Check first whether `robustness_metrics.py` is on the protected list.

## 9. Benchmark and parity [existing]

- The benchmark is price-only buy-and-hold on the same OHLCV (`benchmark.py:66-123`). Its dividend wording needs a documentation note for crypto.
- Parity: the reference engine vs VectorBT on 3 cells, (8,25), (10,30) and (12,35), at 1e-9 tolerances (`parity.py:39-41`), for both products. Discrete facts must match exactly.

## 10. Synthetic execution truth [decided; new]

Synthetic fixtures with **hand-specified expected outcomes**, never derived from either engine. They run through both the reference engine and VectorBT before any Coinbase performance run. Cases:
- a signal at close T fills at open T+1;
- EMA equality gives flat;
- warm-up gives no premature trade;
- a last-bar entry signal gives no impossible fill;
- never enter;
- enter and never exit;
- gap-open execution;
- a missing-day sequence;
- commission and slippage on entry and exit.

This adds a third independent layer: known outcome ↔ reference ↔ VectorBT.

## 11. Must NOT touch the crypto path [existing]

`corporate_actions.py`, `data/canonical.py` and `pine_parity` (AMEX:SPY) must stay off the crypto path. A test asserts this.

## 12. Result policy [decided: option A]

- M6 is **descriptive**, with no numeric profitability gate.
- **The M6 completion gate is** methodological correctness, reproducibility, parity, synthetic execution truth and sealed-TEST discipline. It is **not** profitability.

## 13. M6 tasks (implementation starts only after the preregistration is frozen)

1. Periods-per-year parameter, with the SPY golden output unchanged.
2. Synthetic execution-truth fixtures (§10).
3. Crypto split config, plus the hard TEST seal (§5) with tests.
4. Crypto run path: the processed Coinbase snapshot through the existing engine, temporal validation, robustness (§7 design) and parity, plus the §11 guard.
5. Write the §6 cost scenarios into the M6 config, with a test that the values are exactly 50/5, 50/15 and 50/40 bps.
6. This preregistration, committed on its own before any performance run (done when committed).
7. VALIDATION run and report, then the conclusions committed (commit X).
8. A single TEST opening (run Y), then the final M6 manifest.

## 14. Authentication state and scope boundaries [decided]

- **Recorded state, 2026-10-19:** Claude has no active Coinbase for Agents connection, and Claude Code and strategy-lab have no locally configured authenticated Coinbase access.
- Evidence, from Javi's checks:
  - `claude mcp list` shows no Coinbase MCP (only Claude Docs).
  - `~/.claude.json` has no Coinbase or agents.coinbase.com reference.
  - There are no COINBASE, CDP or CB variables in the shell or in the strategy-lab `.env`.
  - Coinbase for Agents, while logged in, shows "Connect Coinbase to Claude" on the Claude tab, which means there is no existing connection.
- Still being checked by Javi on the Coinbase side: Manage agent connections, Connected Applications, and API keys or the Developer Platform. The Coinbase-side state is **not** claimed as fully verified until he confirms it. This does not block M6.
- During M6, nothing connects or configures any of: "Connect Coinbase to Claude", "Connect Coinbase to Claude Code", OAuth, API keys, MCP, or any private endpoint.
- M6 uses public market data only, plus the fee observation Javi supplied manually.
- Scope is ADA-EUR and DOT-EUR only. No BTC or ETH is fetched, ingested, inspected or analyzed now.

## 15. Future M6b candidate (not implemented)

- **M6b:** cross-asset replication on BTC-EUR and ETH-EUR. It reuses the methodology frozen by M6 unchanged. There is no tuning of strategy, grid, metrics, cost model or result policy after the ADA and DOT results are seen.
- **[decided] Same evaluation boundaries:**
  - TRAIN [2022-01-01, 2024-01-01);
  - VALIDATION [2024-01-01, 2025-01-01);
  - TEST [2025-01-01, future frozen replication end).
- Older BTC and ETH history may serve only as burn-in or an unused pre-period. Using it any other way needs a separately preregistered experiment.
- The M6b end date is a future data-integrity decision.
- Prerequisites: BTC and ETH first go through equivalent public-data discovery, immutable ingestion, provenance and revision-audit steps before any performance run.

## 16. Future architecture candidate: Coinbase for Agents (M7 or later; not now)

- Coinbase offers account connections for AI agents, with user-approved actions. This is recorded as a candidate only. Seeing the page is not evidence of any connection.
- If it is evaluated later:
  - start with least privilege, and read-only or observation access where possible;
  - keep research data acquisition independent and reproducible;
  - never let the agent connection bypass the deterministic strategy, risk or execution controls;
  - no autonomous order execution; any trading action requires explicit human approval.
- M6 and M6b need no authenticated access.
