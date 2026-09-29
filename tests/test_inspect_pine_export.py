"""Tests for `scripts/inspect_pine_export.py`'s offline Milestone 4a
schema-discovery helper.

No real TradingView export is used or committed -- all fixtures are
synthetic, deterministic, temp-file CSVs. This script must never guess a
schema, map columns, or interpret trade P&L; these tests only check its
identity/header-reporting behavior.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_pine_export.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("inspect_pine_export", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sha256_matches_known_fixture_bytes(tmp_path: Path) -> None:
    module = _load_module()
    content = b"M4_Open,M4_Close\n1.0,2.0\n"
    path = tmp_path / "chart_data.csv"
    path.write_bytes(content)

    result = module.inspect_export_file(path)

    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["byte_size"] == len(content)


def test_header_raw_text_and_parsed_fields(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "chart_data.csv"
    path.write_text("time,M4_Open,M4_High,M4_Low,M4_Close,M4_Volume\n2010-01-04,1,2,3,4,5\n")

    result = module.inspect_export_file(path)

    assert result["header_line_raw"] == "time,M4_Open,M4_High,M4_Low,M4_Close,M4_Volume"
    assert result["header_fields"] == [
        "time",
        "M4_Open",
        "M4_High",
        "M4_Low",
        "M4_Close",
        "M4_Volume",
    ]
    assert result["header_field_count"] == 6


def test_header_fields_with_quoted_comma_value(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "chart_data.csv"
    path.write_text('a,"b, with comma",c\n1,2,3\n')

    result = module.inspect_export_file(path)

    assert result["header_fields"] == ["a", "b, with comma", "c"]


def test_utf8_bom_is_preserved_in_literal_header_facts(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "chart_data.csv"
    path.write_bytes(b"\xef\xbb\xbftime,M4_Close\n2010-01-04,1.0\n")

    result = module.inspect_export_file(path)

    assert result["header_line_raw"] == "\ufefftime,M4_Close"
    assert result["header_fields"] == ["\ufefftime", "M4_Close"]


def test_multiline_quoted_header_is_reported_as_one_literal_csv_record(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "chart_data.csv"
    path.write_text('time,"M4_Close\n(raw)",M4_Volume\r\n2010-01-04,1,2\r\n')

    result = module.inspect_export_file(path)

    assert result["header_line_raw"] == 'time,"M4_Close\n(raw)",M4_Volume'
    assert result["header_fields"] == ["time", "M4_Close\n(raw)", "M4_Volume"]


def test_missing_file_raises_actionable_error(tmp_path: Path) -> None:
    module = _load_module()
    missing = tmp_path / "does_not_exist.csv"

    try:
        module.inspect_export_file(missing)
        raise AssertionError("expected PineExportInspectionError")
    except module.PineExportInspectionError as exc:
        assert str(missing) in str(exc)


def test_undecodable_bytes_raise_actionable_error(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff\xfe\x00\x01not utf-8")

    try:
        module.inspect_export_file(path)
        raise AssertionError("expected PineExportInspectionError")
    except module.PineExportInspectionError as exc:
        assert str(path) in str(exc)


def test_empty_file_raises_actionable_error(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "empty.csv"
    path.write_text("")

    try:
        module.inspect_export_file(path)
        raise AssertionError("expected PineExportInspectionError")
    except module.PineExportInspectionError as exc:
        assert str(path) in str(exc)


def test_malformed_header_raises_instead_of_guessing(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "malformed.csv"
    path.write_text('time,"M4_Close\n')

    try:
        module.inspect_export_file(path)
        raise AssertionError("expected PineExportInspectionError")
    except module.PineExportInspectionError as exc:
        assert str(path) in str(exc)


def test_cli_multiple_files_prints_deterministic_json_list(tmp_path: Path, capsys) -> None:
    module = _load_module()
    chart_path = tmp_path / "chart_data.csv"
    chart_path.write_text("time,M4_Close\n2010-01-04,1.0\n")
    trades_path = tmp_path / "list_of_trades.csv"
    trades_path.write_text("Trade #,Type,Date\n1,Entry,2010-01-05\n")

    exit_code = module.main([str(chart_path), str(trades_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["filename"] in {"chart_data.csv", "list_of_trades.csv"}
    assert payload[1]["filename"] in {"chart_data.csv", "list_of_trades.csv"}


def test_cli_missing_file_fails_clearly_and_identifies_file(tmp_path: Path, capsys) -> None:
    module = _load_module()
    ok_path = tmp_path / "chart_data.csv"
    ok_path.write_text("time,M4_Close\n2010-01-04,1.0\n")
    missing_path = tmp_path / "missing.csv"

    exit_code = module.main([str(ok_path), str(missing_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert str(missing_path) in captured.err


def test_cli_writes_nothing_besides_stdout(tmp_path: Path, capsys) -> None:
    module = _load_module()
    chart_path = tmp_path / "chart_data.csv"
    chart_path.write_text("time,M4_Close\n2010-01-04,1.0\n")
    before = set(tmp_path.iterdir())

    exit_code = module.main([str(chart_path)])

    assert exit_code == 0
    capsys.readouterr()
    after = set(tmp_path.iterdir())
    assert before == after


def test_output_is_limited_to_identity_and_header_facts(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "chart_data.csv"
    path.write_text("time,M4_Close\n2010-01-04,1.0\n")

    assert set(module.inspect_export_file(path)) == {
        "supplied_path",
        "filename",
        "sha256",
        "byte_size",
        "header_line_raw",
        "header_fields",
        "header_field_count",
    }


def test_imports_are_limited_to_offline_stdlib_and_project_sha256_helper() -> None:
    tree = ast.parse(_SCRIPT_PATH.read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imports == {"argparse", "csv", "io", "json", "sys"}
    assert from_imports == {"__future__", "pathlib", "strategy_lab.data.snapshot"}


def test_reuses_existing_sha256_helper_not_a_duplicate() -> None:
    source = _SCRIPT_PATH.read_text()
    assert "from strategy_lab.data.snapshot import compute_sha256" in source
    assert "def compute_sha256" not in source
