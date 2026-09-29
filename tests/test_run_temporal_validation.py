"""Tests for `scripts/run_temporal_validation.py`'s offline behavior.

Requires VectorBT (the script imports `strategy_lab.validation.robustness`
at module load time) -- same no-`importorskip` convention as
`tests/test_parity.py` and `tests/test_robustness.py`.
"""

from __future__ import annotations

import csv
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import vectorbt  # noqa: F401  (import here makes a missing install fail collection visibly)
import yaml

from conftest import random_walk_ohlcv
from strategy_lab.validation.corporate_actions import CorporateActionCheckResult

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_temporal_validation.py"
_REPO_ROOT = _SCRIPT_PATH.resolve().parents[1]


def _load_run_temporal_validation():
    spec = importlib.util.spec_from_file_location("run_temporal_validation", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_never_imports_yfinance() -> None:
    source = _SCRIPT_PATH.read_text()
    assert "import yfinance" not in source
    assert "from yfinance" not in source
    assert "yfinance" not in source.lower().replace("scripts/fetch_dataset.py", "")


def test_script_reuses_run_parity_report_selector_not_a_duplicate() -> None:
    source = _SCRIPT_PATH.read_text()
    assert "_find_latest_processed_snapshot" in source
    assert "def _find_latest_processed_snapshot" not in source  # reused, not redefined


def test_missing_snapshot_fails_actionably_without_network(tmp_path, monkeypatch, capsys) -> None:
    def _blocked_socket(*args, **kwargs):
        raise AssertionError("network access attempted during offline temporal validation script")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(tmp_path))

    module = _load_run_temporal_validation()
    exit_code = module.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "fetch_dataset.py" in captured.err


def test_missing_data_root_env_var_fails_without_network(monkeypatch, capsys) -> None:
    def _blocked_socket(*args, **kwargs):
        raise AssertionError("network access attempted during offline temporal validation script")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.delenv("STRATEGY_LAB_DATA_ROOT", raising=False)

    module = _load_run_temporal_validation()
    exit_code = module.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.err


def test_dependency_versions_helper_records_python_pandas_pyarrow_vectorbt() -> None:
    module = _load_run_temporal_validation()
    versions = module._dependency_versions()
    assert "python" in versions
    assert "pandas" in versions
    assert "pyarrow" in versions
    assert "vectorbt" in versions
    assert versions["vectorbt"] is not None
    assert versions["python"] is not None


def test_reports_dir_resolves_under_repository_reports_generated() -> None:
    """Path-semantics check (not source-string grep): the resolved
    `_REPORTS_DIR` must literally be `reports/generated/` under the repo
    root -- the existing, gitignored generated-artifact location -- never
    a second `experiments/reports/` hierarchy."""
    module = _load_run_temporal_validation()
    assert module._REPORTS_DIR == module._REPO_ROOT / "reports" / "generated"
    assert module._REPORTS_DIR.is_relative_to(module._REPO_ROOT / "reports")
    assert not module._REPORTS_DIR.is_relative_to(module._REPO_ROOT / "experiments")


def _init_fake_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    (repo_dir / "placeholder.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_dir, check=True)


def _small_research_config() -> dict:
    return {
        "temporal_validation": {
            "burn_in_year": 2020,
            "evaluation_years": [2021, 2022],
            "risk_free_rate": 0.0,
        },
        "robustness": {
            "ema_neighborhood": {
                "fast": [8, 9, 10, 11, 12],
                "slow": [25, 30, 35],
                "max_combinations": 25,
            },
            "cost_stress": {
                "baseline": {"commission_rate": 0.001, "slippage_rate": 0.0005},
                "moderately_worse": {"commission_rate": 0.0025, "slippage_rate": 0.0015},
                "materially_worse": {"commission_rate": 0.005, "slippage_rate": 0.004},
            },
            "parity_spot_checks": [[8, 25], [10, 30], [12, 35]],
        },
    }


def test_git_state_captured_before_any_output_write(tmp_path, monkeypatch) -> None:
    """Production-path proof, run through the real `main()`: Git/config
    state is captured strictly before the report JSON (and registry row)
    are written, so a repo that was genuinely clean at experiment start is
    recorded as clean even though the run's own output writes make it
    dirty by the time the function returns.
    """
    fake_repo = tmp_path / "fake_repo"
    _init_fake_repo(fake_repo)

    data_root = tmp_path / "data_root"
    data_root.mkdir()
    monkeypatch.setenv("STRATEGY_LAB_DATA_ROOT", str(data_root))

    module = _load_run_temporal_validation()
    monkeypatch.setattr(module, "_REPO_ROOT", fake_repo)
    monkeypatch.setattr(module, "_REPORTS_DIR", fake_repo / "reports" / "generated")
    monkeypatch.setattr(module, "_REGISTRY_PATH", fake_repo / "registry.csv")

    # Deliberately NOT under fake_repo: writing it there before `main()`
    # runs would itself dirty the repo and invalidate the "clean at start"
    # assertion below for reasons unrelated to what this test checks.
    config_path = tmp_path / "research.yaml"
    config_path.write_text(yaml.safe_dump(_small_research_config()))
    monkeypatch.setattr(module, "_CONFIG_PATH", config_path)

    ohlcv = random_walk_ohlcv("2020-01-01", "2022-12-31", seed=99)

    fake_processed_ref = type(
        "FakeProcessedRef", (), {"sha256": "deadbeef", "parquet_path": tmp_path / "p.parquet"}
    )()
    fake_raw_ref = type("FakeRawRef", (), {"sha256": "cafef00d"})()

    fake_run_parity_report = type(
        "FakeModule",
        (),
        {"_find_latest_processed_snapshot": staticmethod(lambda root: fake_processed_ref)},
    )()
    monkeypatch.setattr(module, "_load_run_parity_report", lambda: fake_run_parity_report)
    monkeypatch.setattr(module, "load_processed_snapshot", lambda ref: ohlcv)
    monkeypatch.setattr(
        module.corporate_actions, "resolve_raw_ref_from_processed", lambda *a, **k: fake_raw_ref
    )
    monkeypatch.setattr(
        module.corporate_actions,
        "check_corporate_actions",
        lambda ref: CorporateActionCheckResult(True, 0, None),
    )

    call_order: list[str] = []

    real_git_commit_and_clean = module.registry_mod.git_commit_and_clean

    def _spy_git_commit_and_clean(repo_root):
        call_order.append("git_check")
        return real_git_commit_and_clean(repo_root)

    monkeypatch.setattr(module.registry_mod, "git_commit_and_clean", _spy_git_commit_and_clean)

    real_write_text = Path.write_text

    def _spy_write_text(self, *args, **kwargs):
        if self.suffix == ".json" and "reports" in self.parts:
            call_order.append("report_write")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _spy_write_text)

    exit_code = module.main()
    assert exit_code == 0

    # Ordering: the Git-state check happened before the report was written.
    assert call_order == ["git_check", "report_write"]

    # The repo was clean when captured...
    with (fake_repo / "registry.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["working_tree_clean"] == "true"
    assert rows[0]["record_status"] == "finalized"
    assert rows[0]["dependency_versions_json"]
    assert rows[0]["metrics_report_path"].startswith("reports/generated/")

    # ...even though the repo is now genuinely dirty because of this run's
    # own output writes -- proving early capture was necessary, not
    # incidental.
    status = subprocess.run(
        ["git", "-C", str(fake_repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert status.strip() != ""

    report_path = fake_repo / rows[0]["metrics_report_path"]
    assert report_path.exists()
    assert report_path.is_relative_to(fake_repo / "reports" / "generated")
