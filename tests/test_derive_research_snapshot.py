"""Offline CLI coverage for synthetic research-snapshot derivation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.test_research_snapshot import _PARENT_RELATIVE, _make_parent

from strategy_lab.data.snapshot import compute_sha256

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_research_snapshot.py"
_SPEC = importlib.util.spec_from_file_location("derive_research_snapshot_script", _SCRIPT)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_cli_derives_synthetic_snapshot_and_reports_required_provenance(
    monkeypatch, tmp_path, capsys
) -> None:
    parent, sha256 = _make_parent(tmp_path)
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    assert script.main(["--parent", str(_PARENT_RELATIVE), "--expected-sha256", sha256]) == 0
    output = capsys.readouterr().out
    assert "artifact path:" in output
    assert "artifact SHA-256:" in output
    assert f"source parent SHA-256: {compute_sha256(parent)}" in output
    assert "cutoff used: 2025-01-01T00:00:00" in output
    assert "Real ADA/DOT TEST rows must never become Python-addressable objects" in output
