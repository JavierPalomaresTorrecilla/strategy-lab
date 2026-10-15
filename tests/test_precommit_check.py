"""Unit tests for the staged-file rules in scripts/precommit_check.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).parents[1] / "scripts" / "precommit_check.py"
SPEC = importlib.util.spec_from_file_location("precommit_check", CHECKER_PATH)
assert SPEC and SPEC.loader
precommit_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precommit_check)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    return repo


def _stage(repo: Path, name: str, content: bytes = b"content") -> Path:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _git(repo, "add", name)
    return path


def test_protected_staged_path_requires_explicit_allowance(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    protected = "src/strategy_lab/backtest/engine.py"
    _stage(repo, protected)

    failures = precommit_check.staged_file_failures(repo, allow_protected=False)

    assert f"protected staged file: {protected}" in failures
    assert precommit_check.staged_file_failures(repo, allow_protected=True) == []


@pytest.mark.parametrize(
    "path",
    [
        "raw/input.csv",
        "processed/output.csv",
        "legacy/archive.csv",
        "data/raw/input.csv",
        "data/processed/output.csv",
    ],
)
def test_staged_data_paths_are_rejected(tmp_path: Path, path: str) -> None:
    repo = _repo(tmp_path)
    _stage(repo, path)

    assert f"staged data path is forbidden: {path}" in precommit_check.staged_file_failures(repo)


def test_ordinary_staged_path_is_not_a_data_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = "src/strategy_lab/foo.py"
    _stage(repo, path)

    assert not any(
        failure.startswith("staged data path is forbidden:")
        for failure in precommit_check.staged_file_failures(repo)
    )


def test_staged_blob_size_is_checked_not_worktree_size(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = _stage(repo, "large.bin", b"x" * (5 * 1024 * 1024 + 1))
    path.write_bytes(b"small working tree file")

    failures = precommit_check.staged_file_failures(repo)

    assert any(failure.startswith("staged blob exceeds 5 MB: large.bin") for failure in failures)


def test_small_staged_blob_is_not_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _stage(repo, "small.bin", b"x" * 1024)

    assert not any(
        failure.startswith("staged blob exceeds 5 MB:")
        for failure in precommit_check.staged_file_failures(repo)
    )


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "certificate.pem", "mysecret.txt", "service-key.json"],
)
def test_sensitive_staged_basenames_are_rejected(tmp_path: Path, path: str) -> None:
    repo = _repo(tmp_path)
    _stage(repo, path)

    assert f"sensitive staged filename: {path}" in precommit_check.staged_file_failures(repo)


def test_secret_directory_does_not_make_a_safe_basename_sensitive(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = "secrets/config.json"
    _stage(repo, path)

    assert not any(
        failure.startswith("sensitive staged filename:")
        for failure in precommit_check.staged_file_failures(repo)
    )
