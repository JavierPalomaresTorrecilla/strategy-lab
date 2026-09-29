#!/usr/bin/env python3
"""One-off, human-run script: migrate ``experiments/registry.csv`` from its
pre-Milestone-3 header to the Milestone 3 schema.

Offline; touches only the registry CSV. Existing values are preserved
(``params`` -> ``params_json`` directly); every new column is written
blank; every migrated row is marked ``record_status=dev`` (a legacy row
predates the finalized-record contract and cannot retroactively prove
Git/SHA lineage).

Writes to a temporary file first and only replaces the original after a
successful migration, so a failed migration never leaves the registry in a
partially-rewritten state.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from strategy_lab.validation.registry import RegistryMigrationError, migrate_registry

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _REPO_ROOT / "experiments" / "registry.csv"


def main() -> int:
    if not _REGISTRY_PATH.exists():
        print(f"FAILED: {_REGISTRY_PATH} does not exist", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "registry.migrated.csv"
        try:
            migrate_registry(_REGISTRY_PATH, tmp_path)
        except RegistryMigrationError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        shutil.copyfile(tmp_path, _REGISTRY_PATH)

    print(f"migrated: {_REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
