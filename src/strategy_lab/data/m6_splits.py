"""Explicit M6 Coinbase partition loader, independent of legacy SPY splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from strategy_lab.data.splits import (
    DEFAULT_RESEARCH_CONFIG_PATH,
    DatasetSplit,
    SplitConfigError,
    validate_splits,
)


def load_m6_crypto_splits(
    config_path: Path = DEFAULT_RESEARCH_CONFIG_PATH,
) -> list[DatasetSplit]:
    """Load only ``m6_crypto_splits.partitions`` and validate it."""
    with Path(config_path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    try:
        entries = config["m6_crypto_splits"]["partitions"]
    except (KeyError, TypeError) as exc:
        raise SplitConfigError(
            f"{config_path} is missing a `m6_crypto_splits.partitions` list"
        ) from exc
    if not entries:
        raise SplitConfigError(f"{config_path} `m6_crypto_splits.partitions` is empty")

    splits: list[DatasetSplit] = []
    for entry in entries:
        try:
            splits.append(
                DatasetSplit(
                    name=entry["name"],
                    start=pd.Timestamp(entry["start"]),
                    end_exclusive=pd.Timestamp(entry["end"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitConfigError(f"malformed M6 crypto split entry: {entry!r}") from exc
    validate_splits(splits)
    return splits
