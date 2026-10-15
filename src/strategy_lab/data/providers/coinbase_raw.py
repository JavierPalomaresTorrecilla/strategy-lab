"""Immutable, offline JSON snapshots for Coinbase daily-candle responses.

This module deliberately knows nothing about the Coinbase SDK or networking.
It validates the small provider-shaped payload assembled by a caller, freezes
it as deterministic JSON, and verifies the payload again when it is loaded.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from strategy_lab.data.raw_artifact import RawArtifactRef
from strategy_lab.data.snapshot import compute_sha256, raw_snapshot_dir, raw_snapshot_stem

RAW_SCHEMA_VERSION = 1
PROVIDER = "coinbase"
ALLOWED_PRODUCTS: frozenset[str] = frozenset({"ADA-EUR", "DOT-EUR"})
INTERVAL = "1d"
CANDLE_KEYS: frozenset[str] = frozenset({"start", "low", "high", "open", "close", "volume"})
RATE_LIMIT_HEADER_KEYS: frozenset[str] = frozenset(
    {"x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)

_ARTIFACT_KEYS = frozenset(
    {"schema_version", "provider", "product", "interval", "retrieval", "pages"}
)
_RETRIEVAL_KEYS = frozenset(
    {"python_version", "sdk_version", "started_at_utc", "completed_at_utc"}
)
_PAGE_KEYS = frozenset({"request", "rate_limit_headers", "candles"})
_REQUEST_KEYS = frozenset({"product_id", "start", "end", "granularity"})


class RawArtifactError(ValueError):
    """Raised when a raw JSON artifact does not meet its closed schema."""


@dataclass(frozen=True)
class RawJsonSnapshotRef:
    """Reference to an immutable JSON raw artifact and its metadata sidecar."""

    json_path: Path
    metadata_path: Path
    sha256: str

    @property
    def artifact_path(self) -> Path:
        """Path of the raw payload, satisfying :class:`RawArtifactRef`."""
        return self.json_path


def _require_exact_keys(value: object, expected: frozenset[str], path: str) -> dict:
    if not isinstance(value, dict):
        raise RawArtifactError(f"{path}: expected dict, got {type(value).__name__}")

    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected, key=repr)
    if missing:
        raise RawArtifactError(f"{path}: missing key {missing[0]!r}")
    if unexpected:
        raise RawArtifactError(f"{path}: unexpected key {unexpected[0]!r}")
    return value


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise RawArtifactError(f"{path}: expected str, got {type(value).__name__}")
    return value


def _parse_utc_iso(value: str, path: str) -> datetime:
    """Parse an ISO-8601 string and require an explicit zero UTC offset."""
    parse_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise RawArtifactError(f"{path}: expected ISO-8601 UTC string, got {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RawArtifactError(f"{path}: expected ISO-8601 UTC string, got {value!r}")
    return parsed


def validate_raw_artifact(artifact: object) -> None:
    """Fail closed unless *artifact* has exactly the documented raw schema."""
    payload = _require_exact_keys(artifact, _ARTIFACT_KEYS, "artifact")

    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RAW_SCHEMA_VERSION
    ):
        raise RawArtifactError(
            "artifact.schema_version: "
            f"expected {RAW_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    if payload["provider"] != PROVIDER:
        raise RawArtifactError(
            f"artifact.provider: expected {PROVIDER!r}, got {payload['provider']!r}"
        )
    if not isinstance(payload["product"], str) or payload["product"] not in ALLOWED_PRODUCTS:
        raise RawArtifactError(
            "artifact.product: expected one of "
            f"{sorted(ALLOWED_PRODUCTS)!r}, got {payload['product']!r}"
        )
    if payload["interval"] != INTERVAL:
        raise RawArtifactError(
            f"artifact.interval: expected {INTERVAL!r}, got {payload['interval']!r}"
        )

    retrieval = _require_exact_keys(payload["retrieval"], _RETRIEVAL_KEYS, "artifact.retrieval")
    for field in ("python_version", "sdk_version", "started_at_utc", "completed_at_utc"):
        _require_string(retrieval[field], f"artifact.retrieval.{field}")
    _parse_utc_iso(retrieval["started_at_utc"], "artifact.retrieval.started_at_utc")
    _parse_utc_iso(retrieval["completed_at_utc"], "artifact.retrieval.completed_at_utc")

    pages = payload["pages"]
    if not isinstance(pages, list):
        raise RawArtifactError(f"artifact.pages: expected list, got {type(pages).__name__}")
    if not pages:
        raise RawArtifactError("artifact.pages: expected a non-empty list")

    for page_index, page in enumerate(pages):
        page_path = f"artifact.pages[{page_index}]"
        page_dict = _require_exact_keys(page, _PAGE_KEYS, page_path)

        # A page only exists after the SDK successfully returned it; it raises
        # before constructing a page for HTTP errors, so no HTTP-status field is
        # meaningful or retained in this immutable provider payload.
        request = _require_exact_keys(page_dict["request"], _REQUEST_KEYS, f"{page_path}.request")
        for field in ("product_id", "start", "end", "granularity"):
            _require_string(request[field], f"{page_path}.request.{field}")
        if request["granularity"] != "ONE_DAY":
            raise RawArtifactError(
                f"{page_path}.request.granularity: expected 'ONE_DAY', "
                f"got {request['granularity']!r}"
            )
        if request["product_id"] != payload["product"]:
            raise RawArtifactError(
                f"{page_path}.request.product_id: expected artifact.product "
                f"{payload['product']!r}, got {request['product_id']!r}"
            )

        rate_limit_headers = page_dict["rate_limit_headers"]
        headers_path = f"{page_path}.rate_limit_headers"
        if rate_limit_headers is not None:
            if not isinstance(rate_limit_headers, dict):
                raise RawArtifactError(
                    f"{headers_path}: expected dict or None, got "
                    f"{type(rate_limit_headers).__name__}"
                )
            for key, value in rate_limit_headers.items():
                if key not in RATE_LIMIT_HEADER_KEYS:
                    raise RawArtifactError(f"{headers_path}: unexpected key {key!r}")
                _require_string(value, f"{headers_path}.{key}")

        candles = page_dict["candles"]
        candles_path = f"{page_path}.candles"
        if not isinstance(candles, list):
            raise RawArtifactError(f"{candles_path}: expected list, got {type(candles).__name__}")
        for candle_index, candle in enumerate(candles):
            candle_path = f"{candles_path}[{candle_index}]"
            candle_dict = _require_exact_keys(candle, CANDLE_KEYS, candle_path)
            for key in sorted(CANDLE_KEYS):
                _require_string(candle_dict[key], f"{candle_path}.{key}")


def serialize_raw_artifact(artifact: dict) -> bytes:
    """Validate and serialize an artifact deterministically as UTF-8 JSON."""
    validate_raw_artifact(artifact)
    return (
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_file_exclusive(tmp_path: Path, final_path: Path) -> None:
    """Exclusively claim *final_path* by copying exact bytes from *tmp_path*."""
    with open(final_path, "xb") as destination, tmp_path.open("rb") as source:
        shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())


def _publish_metadata_exclusive(metadata_path: Path, metadata: dict) -> None:
    """Create a sidecar once, never replacing a pre-existing one."""
    with metadata_path.open("x", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=2, sort_keys=True))


def save_raw_json_snapshot(artifact: dict, data_root: Path) -> RawJsonSnapshotRef:
    """Save *artifact* with exclusive publication of JSON and its sidecar."""
    serialized = serialize_raw_artifact(artifact)
    product = artifact["product"]
    retrieval = artifact["retrieval"]
    assert isinstance(product, str)
    assert isinstance(retrieval, dict)
    started_at_utc = _parse_utc_iso(
        retrieval["started_at_utc"], "artifact.retrieval.started_at_utc"
    )
    directory = raw_snapshot_dir(data_root, PROVIDER, product, INTERVAL)
    directory.mkdir(parents=True, exist_ok=True)

    stem = raw_snapshot_stem(started_at_utc)
    json_path = directory / f"{stem}.json"
    metadata_path = directory / f"{stem}.meta.json"
    tmp_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_bytes(serialized)
        sha256 = compute_sha256(tmp_path)
        try:
            _publish_file_exclusive(tmp_path, json_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing immutable raw snapshot at {json_path}"
            ) from exc

        published_sha256 = compute_sha256(json_path)
        if published_sha256 != sha256:
            raise RuntimeError(
                f"published raw snapshot {json_path} does not match its computed checksum "
                f"({published_sha256} != {sha256}); possible corruption during publication"
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata = {
        "schema_version": RAW_SCHEMA_VERSION,
        "kind": "raw_json",
        "provider": PROVIDER,
        "product": product,
        "interval": INTERVAL,
        "sha256": sha256,
        "file": str(json_path.relative_to(data_root)),
        "byte_size": len(serialized),
        "page_count": len(artifact["pages"]),
        "candle_count": sum(len(page["candles"]) for page in artifact["pages"]),
        "python_version": retrieval["python_version"],
        "sdk_version": retrieval["sdk_version"],
        "started_at_utc": retrieval["started_at_utc"],
        "completed_at_utc": retrieval["completed_at_utc"],
    }
    try:
        _publish_metadata_exclusive(metadata_path, metadata)
    except Exception:
        # Preserve an independently-created replacement: only remove the JSON
        # if it still is the exact artifact this invocation published.
        try:
            if json_path.exists() and compute_sha256(json_path) == sha256:
                json_path.unlink()
        except OSError:
            pass
        raise

    return RawJsonSnapshotRef(json_path=json_path, metadata_path=metadata_path, sha256=sha256)


def _reject_non_finite_json_token(token: str) -> object:
    raise RawArtifactError(f"artifact: non-finite JSON token {token!r} is not allowed")


def load_raw_json_snapshot(ref: RawJsonSnapshotRef) -> dict:
    """Load and validate a snapshot, raising :class:`RawArtifactError` on bad data."""
    raw_bytes = ref.json_path.read_bytes()
    actual_sha256 = compute_sha256(ref.json_path)
    if actual_sha256 != ref.sha256:
        raise RawArtifactError(
            f"raw JSON checksum mismatch at {ref.json_path}: "
            f"{actual_sha256} != {ref.sha256}"
        )

    try:
        metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawArtifactError(f"cannot read raw JSON metadata at {ref.metadata_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("sha256") != ref.sha256:
        raise RawArtifactError(
            f"raw JSON metadata checksum mismatch at {ref.metadata_path}: "
            f"expected {ref.sha256}"
        )

    try:
        artifact = json.loads(raw_bytes, parse_constant=_reject_non_finite_json_token)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawArtifactError(f"cannot parse raw JSON artifact at {ref.json_path}") from exc
    validate_raw_artifact(artifact)
    assert isinstance(artifact, dict)
    return artifact


_raw_artifact_ref_typecheck: RawArtifactRef | None = None
_raw_artifact_ref_typecheck = RawJsonSnapshotRef(Path(), Path(), "")
