"""
Tests for `taters.helpers.feature_average`: averaging a feature table up.

The module is small arithmetic over a table written by hand -- four turns by
two speakers in two conversations, so every mean can be checked by eye -- and
most of these tests are about the two contracts it has to keep rather than
about the averaging. First, the row key it writes has to be composed exactly
the way the text gather composes a grouped one, or the analysis table's join
silently finds nothing. Second, nothing may be renamed, because a model is
fitted on column names and has to stay applicable to text that arrives at any
shape.

The test that says why the module exists at all is
`test_averaging_twice_is_not_the_same_as_averaging_once`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from taters.helpers.feature_average import (COUNT_COLUMN, GROUP_FALLBACK_ID,
                                            average_feature_table)
from taters.helpers.text_gather import csv_to_analysis_ready_csv

#: Two conversations, two speakers, four turns. Alice has two turns in c1, so
#: she is the group that averaging actually changes.
TURNS = ("conv,speaker,text\n"
         "c1,alice,one two three\n"
         "c1,alice,four five\n"
         "c1,bob,six\n"
         "c2,alice,seven eight nine ten\n")

#: A per-row feature table keyed the way the text gather keys one when no id
#: column was named: `row_<n>`, numbered over the source rows.
FEATURES = ("text_id,words,grade\n"
            "row_1,3,1\n"
            "row_2,2,3\n"
            "row_3,1,5\n"
            "row_4,4,9\n")


def _rows(path: Path) -> list:
    import csv

    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def study(tmp_path) -> dict:
    """A turn-level spreadsheet, its key map, and a feature table."""
    turns = tmp_path / "turns.csv"
    turns.write_text(TURNS, encoding="utf-8")
    features = tmp_path / "words.csv"
    features.write_text(FEATURES, encoding="utf-8")
    keys = csv_to_analysis_ready_csv(
        csv_path=turns, out_csv=tmp_path / "keys.csv", text_cols=[],
        carry_cols=["conv", "speaker"], verbose=False)
    return {"dir": tmp_path, "turns": turns, "features": features,
            "keys": Path(keys)}


# ---------------------------------------------------------------------------
# Why the module exists
# ---------------------------------------------------------------------------

def test_averaging_twice_is_not_the_same_as_averaging_once(study):
    """
    The whole reason this takes a table rather than a corpus.

    Averaging turns to speakers and then speakers to conversations weights
    the two speakers equally. Averaging turns straight to conversations
    weights whoever talked most. Both are defensible and they are different
    numbers, so the chain has to be something the user states and the
    pipeline records -- not a detail of how one step was called.

    c1 has alice on 3 and 2 words and bob on 1. By speaker: mean(2.5, 1) =
    1.75. Straight to conversation: mean(3, 2, 1) = 2.
    """
    by_speaker = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)
    chained = average_feature_table(
        in_csv=by_speaker, out_csv=study["dir"] / "b.csv",
        group_by=["conv"], verbose=False)
    direct = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "c.csv",
        group_by=["conv"], keys_csv=study["keys"], verbose=False)

    c1_chained = next(r for r in _rows(chained) if r["conv"] == "c1")
    c1_direct = next(r for r in _rows(direct) if r["conv"] == "c1")
    assert float(c1_chained["words"]) == pytest.approx(1.75)
    assert float(c1_direct["words"]) == pytest.approx(2.0)
    assert c1_chained["words"] != c1_direct["words"], \
        "the chain collapsed to a single average and lost the distinction"


def test_an_average_is_not_a_copy_of_its_input(study):
    """
    The failure this shape of step has shipped before, elsewhere in the tree:
    group by something already unique per row and you get a file of the right
    shape, with the right columns and real numbers in them, that averaged
    nothing. Here alice's two turns have to become one row.
    """
    out = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)
    rows = _rows(out)

    assert len(rows) == 3, "four turns by three speaker-conversations"
    alice = next(r for r in rows if r["speaker"] == "alice" and r["conv"] == "c1")
    assert int(alice[COUNT_COLUMN]) == 2
    assert float(alice["words"]) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# The join contract -- the row key
# ---------------------------------------------------------------------------

def test_the_row_key_is_composed_exactly_as_the_text_gather_composes_it(study):
    """
    The one contract that cannot be off by a character. The averaged table
    joins the analysis table against a metadata table the *text gather*
    wrote, so both sides have to spell a group's id the same way. Rather
    than assert on the format, this asserts the two agree -- so if the
    gather's spelling ever changes, this fails rather than the join going
    quietly empty.
    """
    grouped = csv_to_analysis_ready_csv(
        csv_path=study["turns"], out_csv=study["dir"] / "grouped.csv",
        text_cols=["text"], group_by=["conv", "speaker"], verbose=False)
    averaged = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)

    assert {r["text_id"] for r in _rows(averaged)} == \
        {r["text_id"] for r in _rows(Path(grouped))}


def test_a_group_whose_key_is_entirely_blank_gets_the_same_fallback_id(tmp_path):
    """
    `_compose_id` drops blank values, so a group keyed on nothing at all
    would get an empty id. The text gather falls back to the literal
    "group"; anything else here and that one group fails to join.
    """
    table = tmp_path / "f.csv"
    table.write_text("text_id,g,x\nr1,,1\nr2,,3\n", encoding="utf-8")
    out = average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                                group_by=["g"], verbose=False)

    assert [r["text_id"] for r in _rows(out)] == [GROUP_FALLBACK_ID]


def test_the_key_map_stays_aligned_with_the_feature_table_across_blank_rows(tmp_path):
    """
    A subtle, load-bearing and undocumented property of the text gather: a
    row with no text is skipped *after* its row number is taken, so a table
    gathered with a text column and a key map gathered without one number
    their rows the same way. If they did not, every feature table would join
    the wrong rows to the wrong groups and the numbers would all be wrong
    while looking perfectly reasonable.
    """
    source = tmp_path / "t.csv"
    source.write_text("g,text\na,hello there\na,\nb,goodbye now\n", encoding="utf-8")
    measured = csv_to_analysis_ready_csv(
        csv_path=source, out_csv=tmp_path / "m.csv", text_cols=["text"],
        verbose=False)
    keys = csv_to_analysis_ready_csv(
        csv_path=source, out_csv=tmp_path / "k.csv", text_cols=[],
        carry_cols=["g"], verbose=False)

    measured_ids = [r["text_id"] for r in _rows(Path(measured))]
    key_ids = [r["text_id"] for r in _rows(Path(keys))]
    assert measured_ids == ["row_1", "row_3"], "the blank row took a number"
    assert set(measured_ids) <= set(key_ids), "the key map cannot place a row"


def test_a_row_the_key_map_has_never_heard_of_is_left_out(study):
    """
    Better than inventing a group for it. The count column is what shows
    rows went missing, so this is visible rather than silent.
    """
    extra = study["dir"] / "extra.csv"
    extra.write_text(FEATURES + "row_99,7,7\n", encoding="utf-8")
    out = average_feature_table(
        in_csv=extra, out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)

    assert sum(int(r[COUNT_COLUMN]) for r in _rows(out)) == 4, \
        "the unplaceable row was averaged into something"


# ---------------------------------------------------------------------------
# Nothing is renamed -- the portability contract
# ---------------------------------------------------------------------------

def test_the_averaged_columns_keep_their_own_names(study):
    """
    A model is fitted on column names and has to be applicable to text that
    arrives at any shape, so a ridge fitted on averages must find its
    predictors in an un-averaged table. Suffixing `__mean` onto everything
    would make every such model permanently unusable outside the exact
    pipeline that produced it.
    """
    out = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)
    names = set(_rows(out)[0])

    assert {"words", "grade"} <= names
    assert not [n for n in names if "__" in n or n.endswith("_mean")], names


def test_the_step_declares_itself_a_thing_that_shapes_the_dataset(study):
    """
    `defines_text` is what keeps a model portable: it holds this step's
    settings out of the chain digest the apply-time gate compares, so a
    table averaged differently is not mistaken for one *measured*
    differently. And the grain is recorded rather than compared, for the
    same reason.
    """
    declared = average_feature_table.__provenance__

    assert declared["defines_text"] is True
    assert "group_by" in declared["grain"]
    assert COUNT_COLUMN in declared["bookkeeping"], \
        "the row count could be picked up as a predictor"


def test_the_count_column_is_written_once_however_often_we_average(study):
    """
    Averaging an already-averaged table used to write `rows_averaged` twice
    -- once as the mean of the previous counts, once as this level's own --
    which is a duplicated header and a meaningless number.
    """
    first = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)
    second = average_feature_table(
        in_csv=first, out_csv=study["dir"] / "b.csv", group_by=["conv"],
        verbose=False)

    header = Path(second).read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.count(COUNT_COLUMN) == 1, header
    c1 = next(r for r in _rows(second) if r["conv"] == "c1")
    assert int(c1[COUNT_COLUMN]) == 2, "two speaker rows went into c1"


# ---------------------------------------------------------------------------
# What gets averaged, and what does not
# ---------------------------------------------------------------------------

def test_a_blank_cell_is_skipped_rather_than_counted_as_zero(tmp_path):
    """
    A column measured on only some of a speaker's turns is the mean of the
    turns it was measured on. Treating a blank as zero would drag the mean
    toward zero and call it a measurement.
    """
    table = tmp_path / "f.csv"
    table.write_text("text_id,g,x\nr1,a,2\nr2,a,\n", encoding="utf-8")
    out = average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                                group_by=["g"], verbose=False)
    row = _rows(out)[0]

    assert float(row["x"]) == pytest.approx(2.0)
    assert int(row[COUNT_COLUMN]) == 2, "the count is rows in the group"


def test_a_column_of_words_is_dropped_because_there_is_no_mean_of_a_word(tmp_path):
    table = tmp_path / "f.csv"
    table.write_text("text_id,g,x,note\nr1,a,2,hello\nr2,a,4,there\n",
                     encoding="utf-8")
    out = average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                                group_by=["g"], verbose=False)

    assert "note" not in _rows(out)[0]
    assert float(_rows(out)[0]["x"]) == pytest.approx(3.0)


def test_a_table_with_nothing_numeric_is_refused_by_name(tmp_path):
    table = tmp_path / "f.csv"
    table.write_text("text_id,g,note\nr1,a,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no numeric column"):
        average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                              group_by=["g"], verbose=False)


def test_averaging_by_nothing_is_refused_rather_than_copying_the_table(tmp_path):
    table = tmp_path / "f.csv"
    table.write_text("text_id,x\nr1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nothing to average by"):
        average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                              group_by=[], verbose=False)


def test_a_column_that_splits_a_row_legitimately_survives_the_average(tmp_path):
    """
    When each text column was measured separately, one row per text column
    is the point, not an accident -- and the analysis table joins on
    `text_id` *and* that column. Grouping it away would leave duplicate keys
    and the assemble step refuses those.
    """
    table = tmp_path / "f.csv"
    table.write_text("text_id,g,source_col,x\n"
                     "r1,a,q1,2\nr2,a,q1,4\nr3,a,q2,10\n", encoding="utf-8")
    out = average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                                group_by=["g"], split_col="source_col",
                                verbose=False)
    rows = _rows(out)

    assert {r["source_col"] for r in rows} == {"q1", "q2"}
    assert len(rows) == 2, "the two questions were averaged together"
    assert float(next(r for r in rows if r["source_col"] == "q1")["x"]) == \
        pytest.approx(3.0)


def test_groups_assembled_from_too_few_rows_can_be_left_out(study):
    """One turn is not a person's language, and a mean of one is the value
    itself wearing a mean's name."""
    first = average_feature_table(
        in_csv=study["features"], out_csv=study["dir"] / "a.csv",
        group_by=["conv", "speaker"], keys_csv=study["keys"], verbose=False)
    out = average_feature_table(
        in_csv=first, out_csv=study["dir"] / "b.csv", group_by=["conv"],
        min_rows=2, verbose=False)

    assert [r["conv"] for r in _rows(out)] == ["c1"], "c2 had one speaker"


def test_leaving_everything_out_says_so_rather_than_writing_an_empty_table(study):
    with pytest.raises(ValueError, match="nothing is left to write"):
        average_feature_table(
            in_csv=study["features"], out_csv=study["dir"] / "a.csv",
            group_by=["conv", "speaker"], keys_csv=study["keys"],
            min_rows=50, verbose=False)


# ---------------------------------------------------------------------------
# Saying what went wrong
# ---------------------------------------------------------------------------

def test_grouping_by_a_column_the_table_does_not_carry_names_the_way_out(tmp_path):
    """A feature table carries `text_id` and numbers. Grouping it by
    `speaker` needs the key map, and the message has to say so."""
    table = tmp_path / "f.csv"
    table.write_text("text_id,x\nr1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keys_csv"):
        average_feature_table(in_csv=table, out_csv=tmp_path / "a.csv",
                              group_by=["speaker"], verbose=False)


def test_a_key_map_missing_its_grouping_column_is_refused_by_name(study):
    bad = study["dir"] / "bad.csv"
    bad.write_text("text_id,conv\nrow_1,c1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="speaker"):
        average_feature_table(
            in_csv=study["features"], out_csv=study["dir"] / "a.csv",
            group_by=["conv", "speaker"], keys_csv=bad, verbose=False)


def test_a_table_this_step_already_wrote_is_not_written_again(study):
    """The overwrite contract: an existing output with a record of how it was
    made is handed back, and only `overwrite_existing` rebuilds it."""
    out = study["dir"] / "a.csv"
    average_feature_table(
        in_csv=study["features"], out_csv=out, group_by=["conv"],
        keys_csv=study["keys"], verbose=False)
    written = out.read_bytes()

    # proof it short-circuits rather than quietly redoing the same work:
    # take the input away. a second call that reads anything cannot succeed.
    study["features"].unlink()
    average_feature_table(
        in_csv=study["features"], out_csv=out, group_by=["conv"],
        keys_csv=study["keys"], verbose=False)
    assert out.read_bytes() == written

    with pytest.raises(FileNotFoundError):
        average_feature_table(
            in_csv=study["features"], out_csv=out, group_by=["conv"],
            keys_csv=study["keys"], overwrite_existing=True, verbose=False)


def test_a_table_with_no_record_of_how_it_was_made_is_redone(study):
    """
    Declared `redo_without_record`, same as the two text gathers: this takes
    no time worth saving, and a leftover table from an earlier pipeline in
    the same folder is not something to build an analysis on.
    """
    out = study["dir"] / "a.csv"
    out.write_text("already,here\n1,2\n", encoding="utf-8")
    average_feature_table(
        in_csv=study["features"], out_csv=out, group_by=["conv"],
        keys_csv=study["keys"], verbose=False)

    assert "conv" in _rows(out)[0], "a table of unknown origin was trusted"
