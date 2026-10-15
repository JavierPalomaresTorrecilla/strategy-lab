#!/usr/bin/env python3
"""Local safety checks intended for use before creating a commit."""

from __future__ import annotations

import fnmatch
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

PROTECTED_PATHS = frozenset(
    {
        "src/strategy_lab/backtest/engine.py",
        "src/strategy_lab/backtest/metrics.py",
        "src/strategy_lab/backtest/trades.py",
        "src/strategy_lab/data/canonical.py",
        "src/strategy_lab/data/ohlcv.py",
        "src/strategy_lab/data/snapshot.py",
        "src/strategy_lab/data/processed.py",
    }
)
DATA_PATH_PREFIXES = (
    "raw/",
    "processed/",
    "legacy/",
    "data/raw/",
    "data/processed/",
)
SENSITIVE_BASENAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*secret*",
    "*key*.json",
)
MAX_STAGED_BLOB_SIZE = 5 * 1024 * 1024


def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True)
    except FileNotFoundError:
        message = f"{command[0]} not found (did you create the venv?)"
        return subprocess.CompletedProcess(command, 127, b"", message.encode())


def _git_output(command: Sequence[str], repo: Path) -> bytes:
    result = _run(["git", *command], repo)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(detail or f"git {' '.join(command)} failed")
    return result.stdout


def _staged_paths(repo: Path) -> list[str]:
    output = _git_output(["diff", "--cached", "--name-only", "-z"], repo)
    return [
        entry.decode("utf-8", "surrogateescape")
        for entry in output.split(b"\0")
        if entry
    ]


def _staged_blobs(repo: Path) -> dict[str, list[str]]:
    """Return index blob object IDs for each path, including unmerged entries."""
    output = _git_output(["ls-files", "-s", "-z"], repo)
    blobs: dict[str, list[str]] = {}
    for entry in output.split(b"\0"):
        if not entry:
            continue
        metadata, separator, raw_path = entry.partition(b"\t")
        if not separator:
            continue
        fields = metadata.split()
        if len(fields) < 2:
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        blobs.setdefault(path, []).append(fields[1].decode("ascii"))
    return blobs


def _is_sensitive_basename(path: str) -> bool:
    basename = path.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatchcase(basename, pattern)
        for pattern in SENSITIVE_BASENAME_PATTERNS
    )


def staged_file_failures(
    repo: Path | None = None,
    *,
    allow_protected: bool | None = None,
) -> list[str]:
    """Return all safety violations found in the current repository index."""
    repository = (repo or Path.cwd()).resolve()
    if allow_protected is None:
        allow_protected = os.environ.get("ALLOW_PROTECTED") == "1"

    paths = _staged_paths(repository)
    blobs = _staged_blobs(repository)
    failures: list[str] = []

    for path in paths:
        if not allow_protected and path in PROTECTED_PATHS:
            failures.append(f"protected staged file: {path}")
        if path.startswith(DATA_PATH_PREFIXES):
            failures.append(f"staged data path is forbidden: {path}")
        if _is_sensitive_basename(path):
            failures.append(f"sensitive staged filename: {path}")

        for blob in blobs.get(path, []):
            size_output = _git_output(["cat-file", "-s", blob], repository)
            size = int(size_output.strip())
            if size > MAX_STAGED_BLOB_SIZE:
                failures.append(f"staged blob exceeds 5 MB: {path} ({size} bytes)")
                break

    return failures


def _tool_failure(label: str, result: subprocess.CompletedProcess[bytes]) -> str | None:
    if result.returncode == 0:
        return None
    detail = (
        (result.stderr or result.stdout).decode("utf-8", "replace").strip().splitlines()
    )
    suffix = f": {detail[-1]}" if detail else ""
    return f"{label} failed (exit {result.returncode}){suffix}"


def main() -> int:
    repo = Path.cwd()
    failures: list[str] = []

    ruff = _run([".venv/bin/ruff", "check", "."], repo)
    ruff_failure = _tool_failure("ruff check", ruff)
    if ruff_failure:
        failures.append(ruff_failure)

    pytest = _run([".venv/bin/python", "-m", "pytest", "-q"], repo)
    if pytest.returncode == 5:
        failures.append("pytest collected 0 tests")
    else:
        pytest_failure = _tool_failure("pytest", pytest)
        if pytest_failure:
            failures.append(pytest_failure)

    try:
        failures.extend(staged_file_failures(repo))
    except RuntimeError as error:
        failures.append(f"could not inspect staged files: {error}")

    if failures:
        for failure in failures:
            print(f"pre-commit: FAIL: {failure}")
        return 1

    print("pre-commit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
