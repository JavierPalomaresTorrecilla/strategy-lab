"""M6 Coinbase configuration stays separate from legacy SPY split loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from strategy_lab.data.m6_splits import load_m6_crypto_splits
from strategy_lab.data.splits import RESEARCH_ALLOWED_SPLITS, SplitConfigError, load_splits


def test_m6_crypto_splits_are_exact_validated_partitions() -> None:
    splits = load_m6_crypto_splits()
    assert [(split.name, split.start, split.end_exclusive) for split in splits] == [
        ("train", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")),
        ("validation", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
        ("test", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-19")),
    ]
    assert all(
        left.end_exclusive == right.start for left, right in zip(splits, splits[1:], strict=False)
    )


def test_legacy_spy_splits_and_research_allowlist_are_unchanged() -> None:
    with (Path(__file__).resolve().parents[1] / "config" / "research.yaml").open() as source:
        config = yaml.safe_load(source)
    assert config["splits"]["partitions"] == [
        {"name": "train", "start": "2010-01-01", "end": "2019-01-01"},
        {"name": "validation", "start": "2019-01-01", "end": "2023-01-01"},
        {"name": "test", "start": "2023-01-01", "end": "2026-01-01"},
    ]
    assert [(split.name, split.start, split.end_exclusive) for split in load_splits()] == [
        ("train", pd.Timestamp("2010-01-01"), pd.Timestamp("2019-01-01")),
        ("validation", pd.Timestamp("2019-01-01"), pd.Timestamp("2023-01-01")),
        ("test", pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-01")),
    ]
    assert RESEARCH_ALLOWED_SPLITS == ("train", "validation")


def test_loaders_do_not_fall_through_to_each_others_namespace(tmp_path: Path) -> None:
    m6_only = tmp_path / "m6-only.yaml"
    m6_only.write_text(
        "m6_crypto_splits:\n  partitions:\n    - name: train\n"
        "      start: '2022-01-01'\n      end: '2023-01-01'\n"
    )
    with pytest.raises(SplitConfigError, match="splits.partitions"):
        load_splits(m6_only)

    spy_only = tmp_path / "spy-only.yaml"
    spy_only.write_text(
        "splits:\n  partitions:\n    - name: train\n"
        "      start: '2010-01-01'\n      end: '2011-01-01'\n"
    )
    with pytest.raises(SplitConfigError, match="m6_crypto_splits.partitions"):
        load_m6_crypto_splits(spy_only)
