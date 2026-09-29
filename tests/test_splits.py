"""Tests for chronological split loading/validation.

Uses temporary YAML fixtures rather than the real `config/research.yaml`,
so these tests are isolated from (and don't need to track) the real
project configuration's exact dates.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from strategy_lab.data.splits import (
    RESEARCH_ALLOWED_SPLITS,
    SplitConfigError,
    assign_split,
    load_splits,
    split_ohlcv,
)

VALID_YAML = textwrap.dedent(
    """
    splits:
      schema_version: 1
      partitions:
        - name: train
          start: "2010-01-01"
          end: "2019-01-01"
        - name: validation
          start: "2019-01-01"
          end: "2023-01-01"
        - name: test
          start: "2023-01-01"
          end: "2026-01-01"
    """
)


def _write(tmp_path, contents: str):
    config_path = tmp_path / "research.yaml"
    config_path.write_text(contents)
    return config_path


def test_loads_valid_yaml(tmp_path) -> None:
    path = _write(tmp_path, VALID_YAML)
    splits = load_splits(path)
    assert [s.name for s in splits] == ["train", "validation", "test"]
    assert splits[0].start == pd.Timestamp("2010-01-01")
    assert splits[0].end_exclusive == pd.Timestamp("2019-01-01")


@pytest.mark.parametrize(
    ("start", "end", "at"),
    [
        ("2018-12-31", "train", "2018-12-31"),
        ("2019-01-01", "validation", "2019-01-01"),
        ("2022-12-31", "validation", "2022-12-31"),
        ("2023-01-01", "test", "2023-01-01"),
        ("2025-12-31", "test", "2025-12-31"),
    ],
)
def test_exact_boundary_dates(tmp_path, start, end, at) -> None:
    path = _write(tmp_path, VALID_YAML)
    splits = load_splits(path)
    assert assign_split(at, splits) == end


def test_end_exclusive_boundary_belongs_to_next_split(tmp_path) -> None:
    path = _write(tmp_path, VALID_YAML)
    splits = load_splits(path)
    assert assign_split("2019-01-01", splits) == "validation"
    assert assign_split("2026-01-01", splits) is None


def test_out_of_range_timestamp_is_unassigned(tmp_path) -> None:
    path = _write(tmp_path, VALID_YAML)
    splits = load_splits(path)
    assert assign_split("2005-01-01", splits) is None
    assert assign_split("2030-01-01", splits) is None


def test_start_greater_than_or_equal_end_rejected(tmp_path) -> None:
    bad_yaml = textwrap.dedent(
        """
        splits:
          partitions:
            - name: train
              start: "2019-01-01"
              end: "2010-01-01"
        """
    )
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(SplitConfigError):
        load_splits(path)


def test_overlap_rejected(tmp_path) -> None:
    bad_yaml = textwrap.dedent(
        """
        splits:
          partitions:
            - name: train
              start: "2010-01-01"
              end: "2019-06-01"
            - name: validation
              start: "2019-01-01"
              end: "2023-01-01"
        """
    )
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(SplitConfigError):
        load_splits(path)


def test_gap_rejected(tmp_path) -> None:
    bad_yaml = textwrap.dedent(
        """
        splits:
          partitions:
            - name: train
              start: "2010-01-01"
              end: "2018-06-01"
            - name: validation
              start: "2019-01-01"
              end: "2023-01-01"
        """
    )
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(SplitConfigError):
        load_splits(path)


def test_out_of_order_partitions_rejected(tmp_path) -> None:
    bad_yaml = textwrap.dedent(
        """
        splits:
          partitions:
            - name: validation
              start: "2019-01-01"
              end: "2023-01-01"
            - name: train
              start: "2010-01-01"
              end: "2019-01-01"
        """
    )
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(SplitConfigError):
        load_splits(path)


def test_no_overlap_between_all_three_real_partitions() -> None:
    splits = load_splits()  # real config/research.yaml
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            assert a.end_exclusive <= b.start, f"{a.name} and {b.name} overlap"


def test_full_coverage_of_real_dataset_range() -> None:
    splits = load_splits()  # real config/research.yaml
    all_days = pd.date_range("2010-01-01", "2025-12-31", freq="D")
    assignments = [assign_split(day, splits) for day in all_days]
    assert all(a is not None for a in assignments), "every day must be assigned to a partition"


def test_split_ohlcv_partitions_a_dataframe() -> None:
    splits = load_splits()
    dates = pd.date_range("2015-01-01", "2024-01-01", freq="YS")
    df = pd.DataFrame({"timestamp": dates, "close": range(len(dates))})
    result = split_ohlcv(df, splits)
    assert set(result.keys()) == {"train", "validation", "test"}
    assert (result["train"]["timestamp"] < pd.Timestamp("2019-01-01")).all()
    assert (result["validation"]["timestamp"] >= pd.Timestamp("2019-01-01")).all()
    assert (result["validation"]["timestamp"] < pd.Timestamp("2023-01-01")).all()


def test_test_partition_excluded_from_research_allowed_splits() -> None:
    assert "test" not in RESEARCH_ALLOWED_SPLITS
    assert set(RESEARCH_ALLOWED_SPLITS) == {"train", "validation"}


def test_real_config_has_exact_milestone_2_partitions() -> None:
    """Regression test pinning the tracked `config/research.yaml` to the
    intended Milestone 2 partition names and boundaries.

    This intentionally duplicates the dates as a literal assertion, not as
    a second runtime source of truth: `load_splits()` still reads only
    from `config/research.yaml` at runtime. This test exists purely to
    catch an accidental drift in the checked-in file (e.g. a renamed
    partition or a shifted boundary) that the structural invariant tests
    above (no overlap, full coverage) would not otherwise catch.
    """
    splits = load_splits()  # real config/research.yaml

    assert [s.name for s in splits] == ["train", "validation", "test"]

    train, validation, test = splits

    assert train.start == pd.Timestamp("2010-01-01")
    assert train.end_exclusive == pd.Timestamp("2019-01-01")

    assert validation.start == pd.Timestamp("2019-01-01")
    assert validation.end_exclusive == pd.Timestamp("2023-01-01")

    assert test.start == pd.Timestamp("2023-01-01")
    assert test.end_exclusive == pd.Timestamp("2026-01-01")
