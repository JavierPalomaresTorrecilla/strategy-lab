from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from strategy_lab.data.providers.coinbase_candles import CandleWindow
from strategy_lab.data.providers.coinbase_canonical import decode_pages
from strategy_lab.data.providers.coinbase_raw import (
    RAW_SCHEMA_VERSION,
    RawArtifactError,
    load_raw_json_snapshot,
    save_raw_json_snapshot,
    serialize_raw_artifact,
    validate_raw_artifact,
)
from strategy_lab.data.raw_artifact import RawArtifactRef


def _artifact(*, product: str = "ADA-EUR", candles: list[dict[str, str]] | None = None) -> dict:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "provider": "coinbase",
        "product": product,
        "interval": "1d",
        "retrieval": {
            "python_version": "3.13.0",
            "sdk_version": "1.7.0",
            "started_at_utc": "2026-01-05T12:34:56.123456Z",
            "completed_at_utc": "2026-01-05T12:35:00Z",
        },
        "pages": [
            {
                "request": {
                    "product_id": product,
                    "start": "1767484800",
                    "end": "1767657600",
                    "granularity": "ONE_DAY",
                },
                "rate_limit_headers": {"x-ratelimit-limit": "10"},
                "candles": [] if candles is None else candles,
            }
        ],
    }


def _candle(start: str = "1767484800") -> dict[str, str]:
    return {
        "start": start,
        "low": "0.90",
        "high": "1.10",
        "open": "1.00",
        "close": "1.05",
        "volume": "123.45",
    }


def test_serialization_is_deterministic_utf8_and_newline() -> None:
    artifact = _artifact()
    reordered = {
        "pages": artifact["pages"],
        "retrieval": {
            "completed_at_utc": "2026-01-05T12:35:00Z",
            "sdk_version": "é-sdk",
            "started_at_utc": "2026-01-05T12:34:56.123456Z",
            "python_version": "3.13.0",
        },
        "interval": "1d",
        "product": "ADA-EUR",
        "provider": "coinbase",
        "schema_version": RAW_SCHEMA_VERSION,
    }
    artifact["retrieval"]["sdk_version"] = "é-sdk"

    serialized = serialize_raw_artifact(artifact)
    assert serialized == serialize_raw_artifact(reordered)
    assert serialized.endswith(b"\n")
    assert b"\xc3\xa9-sdk" in serialized
    assert b"\\u00e9" not in serialized


def test_serialization_rejects_nan() -> None:
    artifact = _artifact()
    artifact["schema_version"] = float("nan")
    with pytest.raises((RawArtifactError, ValueError)):
        serialize_raw_artifact(artifact)


@pytest.mark.parametrize(
    ("mutate"),
    [
        lambda artifact: artifact.pop("pages"),
        lambda artifact: artifact.update(unexpected=True),
        lambda artifact: artifact.update(product="BTC-USD"),
        lambda artifact: artifact.update(interval="1h"),
        lambda artifact: artifact.update(provider="other"),
        lambda artifact: artifact.update(pages=[]),
        lambda artifact: artifact["pages"][0]["candles"].append(
            {**_candle(), "__type__": "Candle"}
        ),
        lambda artifact: artifact["pages"][0]["candles"].append(
            {key: value for key, value in _candle().items() if key != "volume"}
        ),
        lambda artifact: artifact["pages"][0]["candles"].append(
            {**_candle(), "close": 1.05}
        ),
        lambda artifact: artifact["pages"][0].update(
            rate_limit_headers={"made-up-header": "x"}
        ),
        lambda artifact: artifact["pages"][0].update(
            rate_limit_headers={"x-ratelimit-limit": 10}
        ),
        lambda artifact: artifact["pages"][0]["request"].update(product_id="DOT-EUR"),
        lambda artifact: artifact["pages"][0]["request"].update(granularity="ONE_HOUR"),
    ],
)
def test_validation_rejects_schema_violations(mutate) -> None:
    artifact = _artifact()
    mutate(artifact)
    with pytest.raises(RawArtifactError):
        validate_raw_artifact(artifact)


def test_save_writes_expected_paths_and_metadata(tmp_path: Path) -> None:
    artifact = _artifact(candles=[_candle(), _candle("1767571200")])
    ref = save_raw_json_snapshot(artifact, tmp_path)

    expected_directory = tmp_path / "raw" / "coinbase" / "ADA-EUR" / "1d"
    assert ref.json_path == expected_directory / "20260105T123456123456Z.json"
    assert ref.metadata_path == expected_directory / "20260105T123456123456Z.meta.json"
    metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
    assert set(metadata) == {
        "schema_version", "kind", "provider", "product", "interval", "sha256", "file",
        "byte_size", "page_count", "candle_count", "python_version", "sdk_version",
        "started_at_utc", "completed_at_utc",
    }
    assert metadata == {
        "schema_version": 1,
        "kind": "raw_json",
        "provider": "coinbase",
        "product": "ADA-EUR",
        "interval": "1d",
        "sha256": hashlib.sha256(ref.json_path.read_bytes()).hexdigest(),
        "file": str(ref.json_path.relative_to(tmp_path)),
        "byte_size": len(ref.json_path.read_bytes()),
        "page_count": 1,
        "candle_count": 2,
        **artifact["retrieval"],
    }
    assert not list(expected_directory.glob(".tmp-*"))


def test_save_refuses_overwrite_without_changing_first_file(tmp_path: Path) -> None:
    artifact = _artifact()
    ref = save_raw_json_snapshot(artifact, tmp_path)
    original_bytes = ref.json_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_raw_json_snapshot(artifact, tmp_path)
    assert ref.json_path.read_bytes() == original_bytes


def test_sidecar_failure_removes_published_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact()

    def fail_sidecar(metadata_path: Path, metadata: dict) -> None:
        raise OSError("forced sidecar failure")

    monkeypatch.setattr(
        "strategy_lab.data.providers.coinbase_raw._publish_metadata_exclusive", fail_sidecar
    )
    with pytest.raises(OSError, match="forced sidecar failure"):
        save_raw_json_snapshot(artifact, tmp_path)

    directory = tmp_path / "raw" / "coinbase" / "ADA-EUR" / "1d"
    assert not list(directory.glob("*.json"))
    assert not list(directory.glob(".tmp-*"))


def test_load_round_trip_and_integrity_checks(tmp_path: Path) -> None:
    artifact = _artifact(candles=[_candle()])
    ref = save_raw_json_snapshot(artifact, tmp_path)
    loaded = load_raw_json_snapshot(ref)
    assert loaded == artifact
    assert serialize_raw_artifact(loaded) == ref.json_path.read_bytes()

    corrupted = bytearray(ref.json_path.read_bytes())
    corrupted[0] ^= 1
    ref.json_path.write_bytes(corrupted)
    with pytest.raises(RawArtifactError, match="checksum mismatch"):
        load_raw_json_snapshot(ref)


def test_load_rejects_tampered_sidecar_checksum(tmp_path: Path) -> None:
    ref = save_raw_json_snapshot(_artifact(), tmp_path)
    metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
    metadata["sha256"] = "not-the-ref-checksum"
    ref.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RawArtifactError, match="metadata checksum mismatch"):
        load_raw_json_snapshot(ref)


def test_reference_satisfies_protocol(tmp_path: Path) -> None:
    ref = save_raw_json_snapshot(_artifact(), tmp_path)
    assert isinstance(ref, RawArtifactRef)
    assert ref.artifact_path == ref.json_path


def test_loaded_candles_feed_canonical_decoder(tmp_path: Path) -> None:
    candles = [_candle("1767484800"), _candle("1767571200"), _candle("1767657600")]
    artifact = _artifact(candles=candles)
    ref = save_raw_json_snapshot(artifact, tmp_path)
    loaded = load_raw_json_snapshot(ref)
    window = CandleWindow(
        first_bucket_start=datetime(2026, 1, 4, tzinfo=UTC),
        last_bucket_start=datetime(2026, 1, 6, tzinfo=UTC),
    )
    decoded = decode_pages(
        "ADA-EUR",
        [(window, loaded["pages"][0]["candles"])],
        datetime(2026, 1, 8, tzinfo=UTC),
    )
    assert decoded.candle_count == 3


def test_import_does_not_import_coinbase_sdk() -> None:
    environment = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(project_root / "src")
    code = (
        "import sys; import strategy_lab.data.providers.coinbase_raw; "
        "raise SystemExit('coinbase' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
