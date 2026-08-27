"""Tests for taters.helpers.feature_gather.

This is the last step of most pipelines — it decides what your modelling table
actually contains — so the interesting cases are the ones where data could go
missing without anyone noticing: column-name collisions, non-numeric columns,
grouping keys that exist in both the filename and the file.
"""

import csv
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="feature_gather requires pandas")

from taters.helpers.feature_gather import (  # noqa: E402
    AggregationPlan,
    aggregate_features,
    feature_gather,
    gather_csvs_to_one,
    make_plan,
)


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def features_dir(tmp_path) -> Path:
    """Two per-file feature CSVs, each with two speakers and two features."""
    root = tmp_path / "features"
    write_csv(root / "session_a.csv", [
        {"speaker": "alice", "f1": "1", "f2": "10"},
        {"speaker": "alice", "f1": "3", "f2": "30"},
        {"speaker": "bob", "f1": "5", "f2": "50"},
    ])
    write_csv(root / "session_b.csv", [
        {"speaker": "alice", "f1": "7", "f2": "70"},
        {"speaker": "bob", "f1": "9", "f2": "90"},
    ])
    return root


def read_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# gather_csvs_to_one — plain concatenation
# ---------------------------------------------------------------------------

def test_gather_stacks_every_row_and_labels_its_origin(features_dir, tmp_path):
    out = gather_csvs_to_one(root_dir=features_dir, out_csv=tmp_path / "all.csv")
    rows = read_rows(out)

    assert len(rows) == 5
    assert list(rows[0])[0] == "source"            # origin column leads
    assert {r["source"] for r in rows} == {"session_a", "session_b"}
    assert sum(1 for r in rows if r["source"] == "session_a") == 3


def test_gather_can_add_the_absolute_source_path(features_dir, tmp_path):
    out = gather_csvs_to_one(
        root_dir=features_dir, out_csv=tmp_path / "all.csv", add_source_path=True
    )
    rows = read_rows(out)
    assert list(rows[0])[:2] == ["source", "source_path"]
    assert Path(rows[0]["source_path"]).is_absolute()


def test_gather_does_not_clobber_an_existing_source_column(tmp_path):
    """
    Transcripts already carry a 'source' column. The gatherer must keep the
    original data instead of overwriting it with the filename.
    """
    root = tmp_path / "in"
    write_csv(root / "file_one.csv", [{"source": "original_value", "f1": "1"}])

    out = gather_csvs_to_one(root_dir=root, out_csv=tmp_path / "all.csv")
    row = read_rows(out)[0]

    assert row["source"] == "file_one"            # injected file stem
    assert row["source.1"] == "original_value"    # original data preserved


def test_gather_accepts_a_single_file_as_root(features_dir, tmp_path):
    out = gather_csvs_to_one(
        root_dir=features_dir / "session_a.csv", out_csv=tmp_path / "one.csv"
    )
    assert len(read_rows(out)) == 3


def test_gather_pattern_selects_a_subset(features_dir, tmp_path):
    out = gather_csvs_to_one(
        root_dir=features_dir, pattern="session_a.csv", out_csv=tmp_path / "sub.csv"
    )
    assert {r["source"] for r in read_rows(out)} == {"session_a"}


def test_gather_non_recursive_ignores_subfolders(features_dir, tmp_path):
    write_csv(features_dir / "nested" / "session_c.csv", [{"speaker": "carol", "f1": "1"}])
    out = gather_csvs_to_one(
        root_dir=features_dir, recursive=False, out_csv=tmp_path / "flat.csv"
    )
    assert "session_c" not in {r["source"] for r in read_rows(out)}


def test_gather_recursive_includes_subfolders(features_dir, tmp_path):
    write_csv(features_dir / "nested" / "session_c.csv", [{"speaker": "carol", "f1": "1"}])
    out = gather_csvs_to_one(root_dir=features_dir, out_csv=tmp_path / "deep.csv")
    assert "session_c" in {r["source"] for r in read_rows(out)}


def test_gather_raises_when_nothing_matches(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No files matched"):
        gather_csvs_to_one(root_dir=empty, out_csv=tmp_path / "x.csv")


def test_gather_default_output_sits_beside_the_folder(features_dir):
    out = Path(gather_csvs_to_one(root_dir=features_dir))
    assert out == features_dir.parent / "features.csv"


def test_gather_respects_the_overwrite_rule(features_dir, tmp_path):
    out_path = tmp_path / "all.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")

    gather_csvs_to_one(root_dir=features_dir, out_csv=out_path)
    assert out_path.read_text(encoding="utf-8") == "sentinel\n"

    gather_csvs_to_one(root_dir=features_dir, out_csv=out_path, overwrite_existing=True)
    assert len(read_rows(out_path)) == 5


# ---------------------------------------------------------------------------
# aggregate_features
# ---------------------------------------------------------------------------

def test_aggregate_per_file_groups_within_each_source(features_dir, tmp_path):
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"])
    out = aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")
    rows = read_rows(out)

    # 2 files x 2 speakers
    assert len(rows) == 4
    assert list(rows[0])[:2] == ["source", "speaker"]

    by_key = {(r["source"], r["speaker"]): r for r in rows}
    assert float(by_key[("session_a", "alice")]["f1__mean"]) == 2.0   # (1+3)/2
    assert float(by_key[("session_a", "bob")]["f1__mean"]) == 5.0
    assert float(by_key[("session_b", "alice")]["f1__mean"]) == 7.0


def test_aggregate_across_files_collapses_sources(features_dir, tmp_path):
    plan = make_plan(group_by=["speaker"], per_file=False, stats=["mean"])
    out = aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")
    rows = read_rows(out)

    assert len(rows) == 2                                    # one row per speaker
    by_speaker = {r["speaker"]: r for r in rows}
    assert float(by_speaker["alice"]["f1__mean"]) == pytest.approx((1 + 3 + 7) / 3)
    assert float(by_speaker["bob"]["f1__mean"]) == pytest.approx((5 + 9) / 2)


def test_aggregate_computes_several_statistics(features_dir, tmp_path):
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean", "std", "max"])
    out = aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")
    header = list(read_rows(out)[0])
    for stat in ("mean", "std", "max"):
        assert f"f1__{stat}" in header
        assert f"f2__{stat}" in header


def test_aggregate_drops_non_numeric_feature_columns(tmp_path):
    root = tmp_path / "in"
    write_csv(root / "a.csv", [
        {"speaker": "alice", "text": "some words here", "f1": "2"},
        {"speaker": "alice", "text": "more words", "f1": "4"},
    ])
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"])
    out = aggregate_features(root_dir=root, plan=plan, out_csv=tmp_path / "agg.csv")

    header = list(read_rows(out)[0])
    assert "f1__mean" in header
    assert not any(h.startswith("text") for h in header)


def test_aggregate_exclude_cols_removes_a_numeric_column(features_dir, tmp_path):
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"], exclude_cols=["f2"])
    out = aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")
    header = list(read_rows(out)[0])
    assert "f1__mean" in header and "f2__mean" not in header


def test_aggregate_include_regex_keeps_only_matching_features(tmp_path):
    root = tmp_path / "in"
    write_csv(root / "a.csv", [
        {"speaker": "alice", "e0": "1", "e1": "2", "wordcount": "100"},
        {"speaker": "alice", "e0": "3", "e1": "4", "wordcount": "200"},
    ])
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"], include_regex=r"^e\d+$")
    out = aggregate_features(root_dir=root, plan=plan, out_csv=tmp_path / "agg.csv")

    header = list(read_rows(out)[0])
    assert {"e0__mean", "e1__mean"} <= set(header)
    assert "wordcount__mean" not in header


def test_aggregate_exclude_regex_removes_matching_features(features_dir, tmp_path):
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"], exclude_regex=r"2$")
    out = aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")
    header = list(read_rows(out)[0])
    assert "f1__mean" in header and "f2__mean" not in header


def test_aggregate_requires_the_group_column_to_exist(features_dir, tmp_path):
    plan = make_plan(group_by=["nonexistent"], per_file=True, stats=["mean"])
    with pytest.raises(ValueError, match="Missing group-by columns"):
        aggregate_features(root_dir=features_dir, plan=plan, out_csv=tmp_path / "agg.csv")


def test_aggregate_errors_when_no_numeric_columns_survive(tmp_path):
    root = tmp_path / "in"
    write_csv(root / "a.csv", [{"speaker": "alice", "text": "only words"}])
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"])
    with pytest.raises(ValueError, match="No numeric columns"):
        aggregate_features(root_dir=root, plan=plan, out_csv=tmp_path / "agg.csv")


def test_aggregate_keeps_rows_whose_group_key_is_missing_by_default(tmp_path):
    """
    dropna defaults to False so a blank speaker becomes its own visible group
    rather than silently disappearing from the results.
    """
    root = tmp_path / "in"
    write_csv(root / "a.csv", [
        {"speaker": "alice", "f1": "2"},
        {"speaker": "", "f1": "8"},
    ])
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"])
    out = aggregate_features(root_dir=root, plan=plan, out_csv=tmp_path / "agg.csv")
    assert len(read_rows(out)) == 2


def test_aggregate_dropna_true_discards_them(tmp_path):
    root = tmp_path / "in"
    write_csv(root / "a.csv", [
        {"speaker": "alice", "f1": "2"},
        {"speaker": "", "f1": "8"},
    ])
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"], dropna=True)
    out = aggregate_features(root_dir=root, plan=plan, out_csv=tmp_path / "agg.csv")
    rows = read_rows(out)
    assert len(rows) == 1 and rows[0]["speaker"] == "alice"


def test_aggregate_respects_the_overwrite_rule(features_dir, tmp_path):
    out_path = tmp_path / "agg.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")
    plan = make_plan(group_by=["speaker"], per_file=True, stats=["mean"])

    aggregate_features(root_dir=features_dir, plan=plan, out_csv=out_path)
    assert out_path.read_text(encoding="utf-8") == "sentinel\n"

    aggregate_features(
        root_dir=features_dir, plan=plan, out_csv=out_path, overwrite_existing=True
    )
    assert len(read_rows(out_path)) == 4


# ---------------------------------------------------------------------------
# feature_gather — the single entry point
# ---------------------------------------------------------------------------

def test_feature_gather_concatenates_when_aggregate_is_false(features_dir, tmp_path):
    out = feature_gather(root_dir=features_dir, out_csv=tmp_path / "out.csv")
    assert len(read_rows(out)) == 5


def test_feature_gather_aggregates_from_quick_arguments(features_dir, tmp_path):
    out = feature_gather(
        root_dir=features_dir,
        aggregate=True,
        group_by=["speaker"],
        per_file=True,
        stats=["mean"],
        out_csv=tmp_path / "out.csv",
    )
    assert len(read_rows(out)) == 4


def test_feature_gather_accepts_an_explicit_plan(features_dir, tmp_path):
    plan = AggregationPlan(group_by=("speaker",), per_file=False, stats=("mean",))
    out = feature_gather(
        root_dir=features_dir, aggregate=True, plan=plan, out_csv=tmp_path / "out.csv"
    )
    assert len(read_rows(out)) == 2


def test_feature_gather_needs_keys_when_aggregating(features_dir, tmp_path):
    with pytest.raises(ValueError, match="plan.*group_by|group_by"):
        feature_gather(root_dir=features_dir, aggregate=True, out_csv=tmp_path / "out.csv")


def test_make_plan_defaults():
    plan = make_plan(group_by=["speaker"])
    assert plan.group_by == ("speaker",)
    assert plan.per_file is True
    assert plan.stats == ("mean", "std")
    assert plan.dropna is False
