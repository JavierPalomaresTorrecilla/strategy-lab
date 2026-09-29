"""Chronological TRAIN/VALIDATION/TEST partitions, loaded from config.

``config/research.yaml`` (key ``splits.partitions``) is the single source of
truth for partition boundaries. This module only loads and validates that
definition — it does not hard-code a second copy of the dates.

Boundaries are half-open: ``start <= timestamp < end``. This resolves the
otherwise-ambiguous "through 2018-12-31" phrasing into an unambiguous rule
that doesn't depend on which days are actually trading days.

TEST is methodologically sealed, not physically inaccessible: it is loaded
and validated exactly like the other partitions, and code may inspect
timestamps while partitioning a full canonical snapshot. What must never
happen during Milestone 2 is passing TEST rows into signal generation, the
backtest engine, VectorBT, metric computation, or parity comparison. See
``RESEARCH_ALLOWED_SPLITS`` and ``scripts/run_parity_report.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_RESEARCH_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "research.yaml"
)

# The only real-data partitions that may be used for strategy/backtest work
# in Milestone 2. "test" is intentionally excluded.
RESEARCH_ALLOWED_SPLITS: tuple[str, ...] = ("train", "validation")


class SplitConfigError(ValueError):
    """Raised when the splits configuration is missing, malformed, or
    violates the required ordering/overlap/contiguity invariants."""


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def load_splits(config_path: Path = DEFAULT_RESEARCH_CONFIG_PATH) -> list[DatasetSplit]:
    """Load and validate the partition list from ``config_path``."""
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    try:
        raw_partitions = config["splits"]["partitions"]
    except (KeyError, TypeError) as exc:
        raise SplitConfigError(
            f"{config_path} is missing a `splits.partitions` list"
        ) from exc

    if not raw_partitions:
        raise SplitConfigError(f"{config_path} `splits.partitions` is empty")

    splits: list[DatasetSplit] = []
    for entry in raw_partitions:
        try:
            name = entry["name"]
            start = pd.Timestamp(entry["start"])
            end = pd.Timestamp(entry["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitConfigError(f"malformed split entry: {entry!r}") from exc
        splits.append(DatasetSplit(name=name, start=start, end_exclusive=end))

    validate_splits(splits)
    return splits


def validate_splits(splits: list[DatasetSplit]) -> None:
    """Validate start<end, chronological order, no overlap, and contiguity.

    Raises ``SplitConfigError`` naming the offending partitions on the first
    violation found.
    """
    if not splits:
        raise SplitConfigError("no splits to validate")

    for split in splits:
        if not split.start < split.end_exclusive:
            raise SplitConfigError(
                f"split {split.name!r} has start ({split.start}) >= "
                f"end ({split.end_exclusive})"
            )

    for earlier, later in zip(splits, splits[1:], strict=False):
        if not earlier.end_exclusive <= later.start:
            raise SplitConfigError(
                f"splits {earlier.name!r} and {later.name!r} are out of "
                f"chronological order or overlap: {earlier.name!r} ends at "
                f"{earlier.end_exclusive}, {later.name!r} starts at {later.start}"
            )
        if earlier.end_exclusive != later.start:
            raise SplitConfigError(
                f"gap between splits {earlier.name!r} (ends {earlier.end_exclusive}) "
                f"and {later.name!r} (starts {later.start}); expected them to be "
                "contiguous"
            )


def assign_split(timestamp, splits: list[DatasetSplit] | None = None) -> str | None:
    """Return the name of the split ``timestamp`` falls in, or ``None`` if
    it falls outside every configured partition."""
    if splits is None:
        splits = load_splits()
    ts = pd.Timestamp(timestamp)
    for split in splits:
        if split.start <= ts < split.end_exclusive:
            return split.name
    return None


def split_ohlcv(
    df: pd.DataFrame, splits: list[DatasetSplit] | None = None
) -> dict[str, pd.DataFrame]:
    """Partition a canonical OHLCV DataFrame (must have a ``timestamp``
    column) into one DataFrame per configured split, keyed by split name.

    Rows outside every configured partition are dropped (not silently kept
    unlabeled) — this should not happen for data fetched within the
    documented dataset range, but is not treated as an error here since it
    is a valid state for out-of-range data, not malformed data.
    """
    if splits is None:
        splits = load_splits()

    result: dict[str, pd.DataFrame] = {}
    for split in splits:
        mask = (df["timestamp"] >= split.start) & (df["timestamp"] < split.end_exclusive)
        result[split.name] = df.loc[mask].reset_index(drop=True)
    return result
