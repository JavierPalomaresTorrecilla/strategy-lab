"""Protocol for any immutable raw-artifact reference.

A ``RawArtifactRef`` identifies one immutable raw artifact file on disk,
whatever its format (Parquet, JSON, ...), by its path and a SHA-256
checksum. ``sha256`` is the SHA-256 of the exact artifact bytes on disk,
not a claim about the equality of two independent serializations of the
same logical data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class RawArtifactRef(Protocol):
    """Structural interface satisfied by any raw-snapshot reference type."""

    @property
    def artifact_path(self) -> Path: ...

    @property
    def sha256(self) -> str: ...
