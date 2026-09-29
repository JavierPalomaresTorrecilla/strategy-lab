"""Tests for external data-root resolution."""

from __future__ import annotations

import os
import stat

import pytest

from strategy_lab.data.env import (
    DataRootNotSetError,
    DataRootUnavailableError,
    resolve_data_root,
)


def test_unset_env_var_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRATEGY_LAB_DATA_ROOT", raising=False)
    with pytest.raises(DataRootNotSetError):
        resolve_data_root()


def test_empty_env_var_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", "")
    with pytest.raises(DataRootNotSetError):
        resolve_data_root()


def test_missing_path_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(missing))
    with pytest.raises(DataRootUnavailableError):
        resolve_data_root()


def test_path_is_a_file_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    a_file = tmp_path / "not_a_directory.txt"
    a_file.write_text("hello")
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(a_file))
    with pytest.raises(DataRootUnavailableError):
        resolve_data_root()


def test_unwritable_directory_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    if os.geteuid() == 0:
        pytest.skip("cannot test unwritable directories while running as root")
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(read_only))
        with pytest.raises(DataRootUnavailableError):
            resolve_data_root()
    finally:
        read_only.chmod(stat.S_IRWXU)


def test_valid_directory_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))
    assert resolve_data_root() == tmp_path
