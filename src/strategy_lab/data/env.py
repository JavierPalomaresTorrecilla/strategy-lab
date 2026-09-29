"""Resolution of the external large-data storage root.

Large datasets (raw provider snapshots, processed/canonical snapshots) must
never live inside this Git repository or on the machine's internal disk as a
silent fallback. Their location is controlled exclusively by the
``STRATEGY_LAB_DATA_ROOT`` environment variable, which on a properly set up
machine points at an external drive.

This module fails closed: if the variable is unset, empty, points at
something that isn't a directory, or isn't writable, callers get a clear,
typed exception. There is no fallback location.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV_VAR = "STRATEGY_LAB_DATA_ROOT"


class DataRootError(RuntimeError):
    """Base class for problems resolving the external data root."""


class DataRootNotSetError(DataRootError):
    """``STRATEGY_LAB_DATA_ROOT`` is unset or empty."""


class DataRootUnavailableError(DataRootError):
    """``STRATEGY_LAB_DATA_ROOT`` is set but unusable (missing/not a
    directory/not writable)."""


def resolve_data_root() -> Path:
    """Return the configured external data root, or raise a clear error.

    Never falls back to a repository-local or internal-disk path.
    """
    raw_value = os.environ.get(DATA_ROOT_ENV_VAR)
    if raw_value is None or raw_value.strip() == "":
        raise DataRootNotSetError(
            f"{DATA_ROOT_ENV_VAR} is not set. Refusing to guess a location "
            "for large datasets; set it to an external, writable directory."
        )

    path = Path(raw_value)
    if not path.exists():
        raise DataRootUnavailableError(
            f"{DATA_ROOT_ENV_VAR}={raw_value!r} does not exist. "
            "Is the external drive connected and mounted?"
        )
    if not path.is_dir():
        raise DataRootUnavailableError(
            f"{DATA_ROOT_ENV_VAR}={raw_value!r} exists but is not a directory."
        )
    if not os.access(path, os.W_OK):
        raise DataRootUnavailableError(
            f"{DATA_ROOT_ENV_VAR}={raw_value!r} exists but is not writable."
        )

    return path
