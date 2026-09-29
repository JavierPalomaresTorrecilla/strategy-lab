"""Offline structural regression tests for the tracked Pine parity source.

Normal pytest cannot invoke the TradingView compiler, so these are static
guardrails on the source text -- they pin down guardrails that must never
silently regress (e.g. an accidental switch to `ta.ema()`, a boundary date
drifting), not a claim that the script compiles or runs correctly in Pine.
Actual compilation remains a human Milestone 4a acceptance step.
"""

from __future__ import annotations

import re
from pathlib import Path

_PINE_PATH = (
    Path(__file__).resolve().parents[1] / "pine" / "ema_crossover_parity.pine"
)


def _source() -> str:
    return _PINE_PATH.read_text()


def test_pine_source_file_exists() -> None:
    assert _PINE_PATH.is_file()


def test_declares_version_6() -> None:
    assert "//@version=6" in _source()


def test_fixed_ema_periods() -> None:
    source = _source()
    assert "fastPeriod = 10" in source
    assert "slowPeriod = 30" in source


def test_research_and_submission_boundaries() -> None:
    source = _source()
    assert '"America/New_York", 2010, 1, 1, 0, 0' in source
    assert '"America/New_York", 2022, 12, 31, 0, 0' in source
    assert '"America/New_York", 2022, 12, 30, 0, 0' in source
    assert "isResearchBar = time >= researchStart and time < researchEndExclusive" in source
    assert "canSubmitOrder = isResearchBar and time < submissionEndExclusive" in source


def test_execution_control_flags_present() -> None:
    source = _source()
    for flag in (
        "process_orders_on_close=false",
        "calc_on_every_tick=false",
        "calc_on_order_fills=false",
        "calc_on_every_history_tick=false",
        "use_bar_magnifier=false",
    ):
        assert flag in source


def test_zero_commission_and_slippage_fixed_qty() -> None:
    source = _source()
    assert "commission_type=strategy.commission.percent" in source
    assert "commission_value=0" in source
    assert "slippage=0" in source
    assert "default_qty_type=strategy.fixed" in source
    assert "default_qty_value=1" in source
    assert 'strategy.entry("Long", strategy.long, qty=1)' in source
    assert "pyramiding=0" in source
    assert "precision=8" in source


def test_all_required_m4_plot_titles_present() -> None:
    source = _source()
    for title in (
        'title="M4_Open"',
        'title="M4_High"',
        'title="M4_Low"',
        'title="M4_Close"',
        'title="M4_Volume"',
        'title="M4_FastEMA"',
        'title="M4_SlowEMA"',
        'title="M4_DesiredPosition"',
        'title="M4_EntryTransition"',
        'title="M4_ExitTransition"',
        'title="M4_PositionState"',
    ):
        assert source.count(title) == 1


def _code_lines() -> list[str]:
    """Source lines with full-line `//` comments dropped, so guardrails
    below check what the script actually does, not its prose comments
    (which legitimately name the forbidden functions to explain why they
    are avoided)."""
    return [line for line in _source().splitlines() if not line.strip().startswith("//")]


def test_does_not_use_builtin_ema_or_crossover_functions_as_the_oracle() -> None:
    code = "\n".join(_code_lines())
    assert "ta.ema(" not in code
    assert "ta.crossover(" not in code
    assert "ta.crossunder(" not in code


def test_explicit_recursive_ema_and_flat_warmup_are_research_gated() -> None:
    code = "\n".join(_code_lines())
    assert re.search(
        r"if isResearchBar\n"
        r"    barsSinceSeed \+= 1\n"
        r"    fastEma := na\(fastEma\[1\]\) \? close : "
        r"alphaFast \* close \+ \(1\.0 - alphaFast\) \* fastEma\[1\]\n"
        r"    slowEma := na\(slowEma\[1\]\) \? close : "
        r"alphaSlow \* close \+ \(1\.0 - alphaSlow\) \* slowEma\[1\]",
        code,
    )
    assert "bothDefined = isResearchBar and barsSinceSeed >= slowPeriod" in code
    assert (
        "desiredPosition = isResearchBar ? (bothDefined and fastEma > slowEma ? 1 : 0) : na"
        in code
    )
    assert "if isResearchBar\n    prevDesired := desiredPosition" in code


def test_transitions_and_orders_keep_the_final_fill_inside_research() -> None:
    code = "\n".join(_code_lines())
    assert "entryTransition = isResearchBar and prevDesired == 0 and desiredPosition == 1" in code
    assert "exitTransition = isResearchBar and prevDesired == 1 and desiredPosition == 0" in code
    assert "if canSubmitOrder and entryTransition" in code
    assert "if canSubmitOrder and exitTransition\n    strategy.close(\"Long\")" in code
    assert "strategy.close_all(" not in code
    assert "strategy.exit(" not in code


def test_does_not_use_security_or_immediately_or_optimization() -> None:
    code = "\n".join(_code_lines())
    assert "request.security(" not in code
    assert "immediately" not in code
    assert "input." not in code  # no optimization/UI-exposed inputs
