"""Synthetic-only tests for the M6 physically TEST-free research artifact."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pyarrow.lib
import pytest

import strategy_lab.data.research_snapshot as research_snapshot
from strategy_lab.data.processed import ProcessedSnapshotRef
from strategy_lab.data.research_snapshot import (
    RESEARCH_CUTOFF,
    TEST_ROW_GUARANTEE,
    ResearchSnapshotError,
    ResearchSnapshotRef,
    derive_research_snapshot,
    read_research_rows,
    require_research_snapshot_ref,
)
from strategy_lab.data.snapshot import compute_sha256

_PARENT_RELATIVE = Path("processed/coinbase/ADA-EUR/1d/synthetic-parent.canonical.parquet")
_SENTINEL = -999_999.0


def _parent_metadata_path(parent_path: Path) -> Path:
    return parent_path.with_name(parent_path.name.removesuffix(".canonical.parquet") + ".meta.json")


def _make_parent(
    data_root: Path,
    *,
    relative_path: Path = _PARENT_RELATIVE,
    metadata_product: str = "ADA-EUR",
) -> tuple[Path, str]:
    parent = data_root / relative_path
    parent.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.Series(
                ["2021-12-31", "2022-01-01", "2024-12-31", "2025-01-01", "2025-01-02"],
                dtype="datetime64[ns]",
            ),
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.1, 2.1, 3.1, _SENTINEL, _SENTINEL],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).to_parquet(parent, engine="pyarrow", compression="snappy", index=False)
    sha256 = compute_sha256(parent)
    _, provider, _, interval, _ = relative_path.parts
    metadata = {
        "kind": "processed",
        "provider": provider,
        "symbol": metadata_product,
        "interval": interval,
        "file": str(relative_path),
        "sha256": sha256,
    }
    _parent_metadata_path(parent).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return parent, sha256


def _derive(data_root: Path):
    parent, sha256 = _make_parent(data_root)
    return derive_research_snapshot(_PARENT_RELATIVE, sha256, data_root)


def test_deterministic_replay_is_byte_identical_across_data_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    parent_a, parent_sha_a = _make_parent(root_a)
    parent_b, parent_sha_b = _make_parent(root_b)
    assert parent_a.read_bytes() == parent_b.read_bytes()
    assert parent_sha_a == parent_sha_b

    ref_a = derive_research_snapshot(_PARENT_RELATIVE, parent_sha_a, root_a)
    ref_b = derive_research_snapshot(_PARENT_RELATIVE, parent_sha_b, root_b)

    assert ref_a.parquet_path.relative_to(root_a) == ref_b.parquet_path.relative_to(root_b)
    assert ref_a.parquet_path.read_bytes() == ref_b.parquet_path.read_bytes()
    assert ref_a.sha256 == ref_b.sha256
    assert ref_a.metadata_path.read_bytes() == ref_b.metadata_path.read_bytes()


def test_derived_parquet_physically_excludes_test_rows_and_sentinel(tmp_path: Path) -> None:
    ref = _derive(tmp_path)
    direct = pd.read_parquet(ref.parquet_path, engine="pyarrow")
    assert not (direct["timestamp"] >= RESEARCH_CUTOFF).any()
    assert direct["timestamp"].max() < RESEARCH_CUTOFF
    assert _SENTINEL not in direct["close"].tolist()


def test_predicate_reader_and_artifact_never_expose_synthetic_test_sentinel(tmp_path: Path) -> None:
    parent, sha256 = _make_parent(tmp_path)
    visible = read_research_rows(parent, RESEARCH_CUTOFF)
    ref = derive_research_snapshot(_PARENT_RELATIVE, sha256, tmp_path)
    final = pd.read_parquet(ref.parquet_path, engine="pyarrow")
    for frame in (visible, final):
        assert _SENTINEL not in frame["close"].tolist()
        assert not (frame["timestamp"] >= RESEARCH_CUTOFF).any()


def test_tz_aware_cutoff_fails_loudly_against_naive_parquet(tmp_path: Path) -> None:
    parent, _ = _make_parent(tmp_path)
    with pytest.raises(pyarrow.ArrowException):
        read_research_rows(parent, pd.Timestamp("2025-01-01", tz="UTC"))


def test_missing_predicate_column_fails_loudly(tmp_path: Path) -> None:
    parent, _ = _make_parent(tmp_path)
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        read_research_rows(parent, RESEARCH_CUTOFF, "not_timestamp")


def test_parent_lineage_is_recorded_and_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    normal_root = tmp_path / "normal"
    normal_root.mkdir()
    _, sha256 = _make_parent(normal_root)
    ref = derive_research_snapshot(_PARENT_RELATIVE, sha256, normal_root)
    assert json.loads(ref.metadata_path.read_text())["source_processed_sha256"] == sha256

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    bad_relative = Path("processed/coinbase/DOT-EUR/1d/synthetic-parent.canonical.parquet")
    _, bad_sha256 = _make_parent(
        mismatch_root, relative_path=bad_relative, metadata_product="ADA-EUR"
    )
    with pytest.raises(ResearchSnapshotError, match="identity disagrees"):
        derive_research_snapshot(bad_relative, bad_sha256, mismatch_root)
    assert not (mismatch_root / "research").exists()


def test_metadata_schema_is_exact_and_correct(tmp_path: Path) -> None:
    ref = _derive(tmp_path)
    metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
    expected_keys = {
        "kind",
        "schema_version",
        "provider",
        "product",
        "interval",
        "source_processed_file",
        "source_processed_sha256",
        "derivation_predicate",
        "research_half_open_end",
        "artifact_sha256",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "pandas_version",
        "pyarrow_version",
        "parquet_engine",
        "parquet_compression",
        "test_row_guarantee",
    }
    assert set(metadata) == expected_keys
    assert metadata == {
        "kind": "research_snapshot",
        "schema_version": 1,
        "provider": "coinbase",
        "product": "ADA-EUR",
        "interval": "1d",
        "source_processed_file": str(_PARENT_RELATIVE),
        "source_processed_sha256": compute_sha256(tmp_path / _PARENT_RELATIVE),
        "derivation_predicate": {
            "column": "timestamp",
            "op": "<",
            "cutoff_utc_naive": "2025-01-01T00:00:00",
        },
        "research_half_open_end": "2025-01-01",
        "artifact_sha256": ref.sha256,
        "row_count": 3,
        "first_timestamp": "2021-12-31T00:00:00",
        "last_timestamp": "2024-12-31T00:00:00",
        "pandas_version": pd.__version__,
        "pyarrow_version": __import__("pyarrow").__version__,
        "parquet_engine": "pyarrow",
        "parquet_compression": "snappy",
        "test_row_guarantee": TEST_ROW_GUARANTEE,
    }


def test_second_publication_refuses_to_clobber_original_artifact(tmp_path: Path) -> None:
    parent, sha256 = _make_parent(tmp_path)
    first = derive_research_snapshot(_PARENT_RELATIVE, sha256, tmp_path)
    original = first.parquet_path.read_bytes()
    with pytest.raises(FileExistsError):
        derive_research_snapshot(_PARENT_RELATIVE, sha256, tmp_path)
    assert first.parquet_path.read_bytes() == original
    assert parent.exists()


def test_bad_expected_parent_sha_fails_before_predicate_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_parent(tmp_path)
    calls = 0

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("predicate read must not occur")

    monkeypatch.setattr(research_snapshot.pd, "read_parquet", spy)
    with pytest.raises(ResearchSnapshotError, match="expected SHA"):
        derive_research_snapshot(_PARENT_RELATIVE, "0" * 64, tmp_path)
    assert calls == 0


def test_runtime_ref_boundary_rejects_processed_and_arbitrary_objects(tmp_path: Path) -> None:
    processed_ref = ProcessedSnapshotRef(tmp_path / "x.parquet", tmp_path / "x.json", "a" * 64)
    with pytest.raises(TypeError):
        require_research_snapshot_ref(processed_ref)
    with pytest.raises(TypeError):
        require_research_snapshot_ref(object())
    research_ref = ResearchSnapshotRef(
        tmp_path / "research.parquet", tmp_path / "research.json", "b" * 64, "2025-01-01"
    )
    assert research_ref.research_half_open_end == "2025-01-01"
    assert not hasattr(processed_ref, "research_half_open_end")


def test_repository_contained_data_root_is_rejected_before_any_write(tmp_path: Path) -> None:
    parent, sha256 = _make_parent(tmp_path)
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(Exception, match="inside the repository"):
        derive_research_snapshot(parent.relative_to(tmp_path), sha256, repository_root)
    assert not (repository_root / "research").exists()


def test_safe_data_root_rejects_a_symlink_resolving_into_the_repository(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    link = tmp_path / "repository-link"
    link.symlink_to(repository_root, target_is_directory=True)
    with pytest.raises(Exception, match="inside the repository"):
        research_snapshot.ensure_safe_data_root(link)


def test_research_destination_symlink_into_repository_is_rejected_without_evidence(
    tmp_path: Path,
) -> None:
    _, sha256 = _make_parent(tmp_path)
    repository_root = Path(__file__).resolve().parents[1]
    (tmp_path / "research").symlink_to(repository_root, target_is_directory=True)
    with pytest.raises(Exception, match="inside the repository"):
        derive_research_snapshot(_PARENT_RELATIVE, sha256, tmp_path)


def test_task_c_modules_expose_no_test_opening_api() -> None:
    public_names = {
        name
        for name, value in inspect.getmembers(research_snapshot)
        if not name.startswith("_") and callable(value)
    }
    assert not {name for name in public_names if "test" in name.lower()}
    assert not {name for name in public_names if name.startswith("load_")}
    source = inspect.getsource(research_snapshot)
    assert "load_processed_snapshot" not in source
    assert "df[df[" not in source
    assert TEST_ROW_GUARANTEE in research_snapshot.__doc__
    for task_c_path in (
        Path(__file__).resolve().parents[1] / "src/strategy_lab/data/m6_splits.py",
        Path(__file__).resolve().parents[1] / "scripts/derive_research_snapshot.py",
    ):
        source = task_c_path.read_text(encoding="utf-8")
        assert "open_test" not in source
        assert "load_test" not in source
