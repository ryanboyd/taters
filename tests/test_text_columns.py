"""
Tests for how CSV text columns reach the output.

Two bugs live here, both found in a real run.

The first: `text_id` is written explicitly as the first output column *and*
composed from whatever id columns you name. Name an id column `text_id` -- the
obvious thing to call one -- and it came out twice, with identical contents.

The second: analyzing several text columns separately produces one row per
column per input row, which is meaningless unless something says which column
each row came from. `source_col` is that something, and it was being dropped
exactly when it mattered.
"""

from __future__ import annotations

import csv

import pytest

from taters.helpers.text_gather import _emit_names, csv_to_analysis_ready_csv


@pytest.fixture()
def survey(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text(
        "text_id,pid,answer,followup\n"
        'r1,p1,"A short thought.","A considerably longer reflection on the matter."\n'
        'r2,p2,"Brief.","Also brief."\n',
        encoding="utf-8",
    )
    return path


def header_of(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh))


def rows_of(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# the duplicate column
# ---------------------------------------------------------------------------

def test_an_id_column_called_text_id_is_not_written_twice(survey, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer"], id_cols=["text_id"], mode="concat",
    )
    header = header_of(out)
    assert header.count("text_id") == 1, header


def test_grouping_by_text_id_does_not_duplicate_it_either(survey, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer"], group_by=["text_id"], mode="concat",
    )
    assert header_of(out).count("text_id") == 1


def test_other_id_columns_are_still_carried_through(survey, tmp_path):
    """The fix must drop only the collision, not identifiers in general."""
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer"], id_cols=["pid"], mode="concat",
    )
    assert "pid" in header_of(out)


def test_a_mixed_list_keeps_the_others_and_drops_the_collision(survey, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer"], id_cols=["text_id", "pid"], mode="concat",
    )
    header = header_of(out)
    assert header.count("text_id") == 1
    assert "pid" in header


def test_every_row_matches_the_header_width(survey, tmp_path):
    """
    The real danger in dropping a column: header and rows are written in
    different places, so filtering one and not the other shifts every field.
    """
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer"], id_cols=["text_id", "pid"], mode="concat",
    )
    with open(out, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    widths = {len(r) for r in rows}
    assert len(widths) == 1, rows


def test_emit_names_is_the_single_rule():
    assert _emit_names(["text_id"]) == []
    assert _emit_names(["pid", "text_id", "cond"]) == ["pid", "cond"]
    assert _emit_names(None) == []


# ---------------------------------------------------------------------------
# several text columns
# ---------------------------------------------------------------------------

def test_concatenating_gives_one_row_per_input_row(survey, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer", "followup"], id_cols=["pid"], mode="concat",
    )
    rows = rows_of(out)
    assert len(rows) == 2
    assert "A short thought." in rows[0]["text"]
    assert "considerably longer" in rows[0]["text"]


def test_separating_gives_one_row_per_column_and_says_which(survey, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=survey, out_csv=tmp_path / "g.csv",
        text_cols=["answer", "followup"], id_cols=["pid"], mode="separate",
    )
    rows = rows_of(out)
    assert len(rows) == 4
    assert {r["source_col"] for r in rows} == {"answer", "followup"}


@pytest.mark.parametrize("analyzer,out_name", [
    ("taters.text.analyze_readability:analyze_readability", "readability.csv"),
    ("taters.text.analyze_lexical_richness:analyze_lexical_richness", "lexrich.csv"),
])
def test_the_analyzer_keeps_source_col_and_one_text_id(survey, tmp_path, analyzer, out_name):
    """
    End to end through the analyzer, which is where the duplicate actually
    surfaced: the gather was fine and the analyzer added `text_id` a second
    time when id columns were named.
    """
    pytest.importorskip("textstat")
    from taters.ui.introspect import load_target

    fn = load_target(analyzer)
    features = tmp_path / out_name
    fn(
        csv_path=survey,
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=features,
        text_cols=["answer", "followup"],
        id_cols=["text_id"],
        mode="separate",
        overwrite_existing=True,
    )

    header = header_of(features)
    assert header.count("text_id") == 1, header
    assert "source_col" in header, header

    rows = rows_of(features)
    assert len(rows) == 4
    assert {r["source_col"] for r in rows} == {"answer", "followup"}
