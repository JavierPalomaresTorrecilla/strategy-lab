"""Tests for `scripts/run_parity_report.py`'s processed-snapshot selection.

`scripts/` is not an installed package, so the script is loaded directly
from its file path rather than imported normally.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_parity_report.py"


def _load_run_parity_report():
    spec = importlib.util.spec_from_file_location("run_parity_report", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_parity_report = _load_run_parity_report()


def _write_processed_artifact(
    directory: Path, *, created_at_utc: str, sha256: str, stem: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / f"{stem}.canonical.parquet"
    metadata_path = directory / f"{stem}.meta.json"
    parquet_path.write_bytes(b"not real parquet bytes, unused by this test")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "processed",
                "created_at_utc": created_at_utc,
                "sha256": sha256,
            }
        )
    )


def test_selects_by_created_at_not_filename_lexical_order(tmp_path) -> None:
    directory = tmp_path / "processed" / run_parity_report.SYMBOL / run_parity_report.INTERVAL

    # Stem "a-..." sorts before stem "z-..." lexicographically, but the
    # "a" artifact was created *later* -- selection must follow
    # `created_at_utc`, not filename/hash order.
    _write_processed_artifact(
        directory,
        created_at_utc="2026-06-01T00:00:00+00:00",
        sha256="aaaa000000000000",
        stem="a-aaaa000000000000",
    )
    _write_processed_artifact(
        directory,
        created_at_utc="2020-01-01T00:00:00+00:00",
        sha256="zzzz111111111111",
        stem="z-zzzz111111111111",
    )

    ref = run_parity_report._find_latest_processed_snapshot(tmp_path)

    assert ref.sha256 == "aaaa000000000000"
    assert ref.parquet_path.name == "a-aaaa000000000000.canonical.parquet"


def test_no_processed_snapshots_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        run_parity_report._find_latest_processed_snapshot(tmp_path)


def test_missing_metadata_sidecar_raises(tmp_path) -> None:
    directory = tmp_path / "processed" / run_parity_report.SYMBOL / run_parity_report.INTERVAL
    directory.mkdir(parents=True)
    (directory / "orphan.canonical.parquet").write_bytes(b"data")

    with pytest.raises(FileNotFoundError):
        run_parity_report._find_latest_processed_snapshot(tmp_path)


def test_missing_created_at_utc_raises(tmp_path) -> None:
    directory = tmp_path / "processed" / run_parity_report.SYMBOL / run_parity_report.INTERVAL
    directory.mkdir(parents=True)
    (directory / "x.canonical.parquet").write_bytes(b"data")
    (directory / "x.meta.json").write_text(json.dumps({"sha256": "deadbeef"}))

    with pytest.raises(ValueError, match="created_at_utc"):
        run_parity_report._find_latest_processed_snapshot(tmp_path)


def test_invalid_created_at_utc_raises(tmp_path) -> None:
    directory = tmp_path / "processed" / run_parity_report.SYMBOL / run_parity_report.INTERVAL
    directory.mkdir(parents=True)
    (directory / "x.canonical.parquet").write_bytes(b"data")
    (directory / "x.meta.json").write_text(
        json.dumps({"sha256": "deadbeef", "created_at_utc": "not-a-timestamp"})
    )

    with pytest.raises(ValueError, match="created_at_utc"):
        run_parity_report._find_latest_processed_snapshot(tmp_path)
