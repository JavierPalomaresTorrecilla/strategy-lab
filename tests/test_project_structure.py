"""Sanity checks that the project skeleton is in place."""

from pathlib import Path

import strategy_lab

ROOT = Path(__file__).resolve().parents[1]


def test_package_importable() -> None:
    assert strategy_lab.__version__


def test_expected_top_level_files_exist() -> None:
    for name in [
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".env.example",
    ]:
        assert (ROOT / name).is_file(), f"missing {name}"


def test_expected_directories_exist() -> None:
    for rel in [
        "config",
        "data/raw",
        "data/processed",
        "src/strategy_lab",
        "experiments",
        "reports",
        "notebooks",
        "scripts",
        "tests",
    ]:
        assert (ROOT / rel).is_dir(), f"missing directory {rel}"


def test_package_submodules_exist() -> None:
    for sub in [
        "data",
        "indicators",
        "strategies",
        "backtest",
        "validation",
        "portfolio",
        "risk",
        "execution",
    ]:
        init_file = ROOT / "src" / "strategy_lab" / sub / "__init__.py"
        assert init_file.is_file(), f"missing {init_file}"


def test_config_files_exist() -> None:
    assert (ROOT / "config" / "research.yaml").is_file()
    assert (ROOT / "config" / "risk.yaml").is_file()


def test_no_env_file_committed() -> None:
    # .env itself should never exist as a tracked template; only .env.example.
    assert (ROOT / ".env.example").is_file()
