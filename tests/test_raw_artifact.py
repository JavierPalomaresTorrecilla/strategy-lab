"""Runtime Protocol checks verify member presence, not declared return types."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from strategy_lab.data.raw_artifact import RawArtifactRef
from strategy_lab.data.snapshot import RawSnapshotRef


@dataclass(frozen=True)
class JsonArtifactRef:
    json_path: Path
    sha256: str

    @property
    def artifact_path(self) -> Path:
        return self.json_path


@dataclass
class ShaOnlyRef:
    sha256: str


def test_raw_snapshot_ref_artifact_path_is_parquet_path() -> None:
    ref = RawSnapshotRef(Path("artifact.parquet"), Path("artifact.meta.json"), "abc123")
    assert ref.artifact_path == ref.parquet_path


def test_raw_snapshot_ref_satisfies_raw_artifact_protocol() -> None:
    ref = RawSnapshotRef(Path("artifact.parquet"), Path("artifact.meta.json"), "abc123")
    assert isinstance(ref, RawArtifactRef)


def test_raw_snapshot_ref_dataclass_fields_are_unchanged() -> None:
    assert [field.name for field in fields(RawSnapshotRef)] == [
        "parquet_path",
        "metadata_path",
        "sha256",
    ]


def test_equal_raw_snapshot_refs_have_equal_hashes() -> None:
    ref_a = RawSnapshotRef(Path("artifact.parquet"), Path("artifact.meta.json"), "abc123")
    ref_b = RawSnapshotRef(Path("artifact.parquet"), Path("artifact.meta.json"), "abc123")
    assert ref_a == ref_b
    assert hash(ref_a) == hash(ref_b)


def test_raw_snapshot_ref_artifact_path_is_read_only() -> None:
    ref = RawSnapshotRef(Path("artifact.parquet"), Path("artifact.meta.json"), "abc123")
    with pytest.raises(AttributeError):
        ref.artifact_path = Path("other.parquet")


def test_json_artifact_ref_satisfies_raw_artifact_protocol() -> None:
    ref = JsonArtifactRef(Path("artifact.json"), "abc123")
    assert isinstance(ref, RawArtifactRef)


def test_ref_without_artifact_path_does_not_satisfy_raw_artifact_protocol() -> None:
    assert not isinstance(ShaOnlyRef("abc123"), RawArtifactRef)
