#!/usr/bin/env python3
"""Milestone 4a schema-discovery helper.

Inspects real, manually-downloaded TradingView chart-data / list-of-trades
CSV exports without interpreting, normalizing, or mapping their schema in
any way. Its only job is to record, for each file: its identity (path,
filename, SHA-256, byte size) and its literal, unmodified header row --
both as raw decoded text and as parsed by Python's stdlib `csv` reader.

Deliberately offline: no network access, no TradingView API call. Reuses
`strategy_lab.data.snapshot.compute_sha256` rather than reimplementing
hashing.

This tool does not parse trade P&L, compute any metric, or infer
entry/exit semantics -- that belongs to Milestone 4b, once the real
schema this script reveals has been reviewed and frozen.

Usage:
    python scripts/inspect_pine_export.py FILE [FILE ...]

Prints deterministic JSON to stdout: a list of one object per input file,
in the order given. It does not write files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategy_lab.data.snapshot import compute_sha256  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


class PineExportInspectionError(Exception):
    """A supplied file could not be inspected (missing, undecodable, or
    has no header line to parse)."""


def inspect_export_file(path: Path) -> dict:
    """Return raw identity/header facts about one CSV export file.

    Never guesses a schema, never inspects rows past the header, never
    interprets trade P&L or entry/exit semantics.
    """
    if not path.is_file():
        raise PineExportInspectionError(f"file not found: {path}")

    sha256 = compute_sha256(path)
    byte_size = path.stat().st_size

    raw_bytes = path.read_bytes()
    try:
        # Do not use ``utf-8-sig`` here. A BOM, if present, is part of the
        # literal export bytes and must be exposed rather than normalized.
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PineExportInspectionError(f"file is not valid UTF-8 text: {path}") from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        parsed_header = next(reader)
    except (csv.Error, StopIteration) as exc:
        raise PineExportInspectionError(f"could not parse a header row: {path}") from exc

    if not parsed_header:
        raise PineExportInspectionError(f"file has no header row: {path}")

    # ``line_num`` is the number of physical CSV lines consumed for the
    # first record. Keep the record's content verbatim, removing only its
    # record terminator. This also handles a legal quoted newline in a
    # header field without inspecting any data row.
    header_line = "".join(text.splitlines(keepends=True)[: reader.line_num])
    if header_line.endswith("\r\n"):
        header_line = header_line[:-2]
    elif header_line.endswith(("\r", "\n")):
        header_line = header_line[:-1]

    return {
        "supplied_path": str(path),
        "filename": path.name,
        "sha256": sha256,
        "byte_size": byte_size,
        "header_line_raw": header_line,
        "header_fields": parsed_header,
        "header_field_count": len(parsed_header),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="one or more CSV export files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    results = []
    for path in args.paths:
        try:
            results.append(inspect_export_file(path))
        except PineExportInspectionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
