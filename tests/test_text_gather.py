"""Tests for taters.helpers.text_gather.

This module is the template for what "thorough" looks like for one function.
`csv_to_analysis_ready_csv` is a good candidate: everything downstream in the
text stack depends on the CSV it produces, it has several interacting options
(mode, grouping, id columns, delimiters), and it has an on-disk caching rule
that has already caused one real bug.

For each function we check four kinds of thing:

  * the happy path      — right columns, right rows, right values
  * the option matrix   — each switch actually changes the output
  * the edge cases      — empty text, missing columns, weird delimiters
  * the contracts       — "don't overwrite unless asked", predictable paths
"""

import csv
from pathlib import Path

import pytest

from taters.helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)


# --- small helpers used by several tests ------------------------------------

def read_csv(path) -> list[dict]:
    """Read a written CSV back into dicts. utf-8-sig strips the BOM Excel likes."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def header_of(path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


@pytest.fixture
def utterances_csv(tmp_path) -> Path:
    """Three utterances from two speakers, in a plain comma-delimited CSV."""
    p = tmp_path / "utterances.csv"
    p.write_text(
        "speaker,session,text,note\n"
        "alice,s1,first thing alice said,n1\n"
        "bob,s1,the only thing bob said,n2\n"
        "alice,s1,second thing alice said,n3\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — happy path
# ---------------------------------------------------------------------------

def test_concat_without_ids_emits_text_id_and_text(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv", text_cols=["text"]
    )
    assert header_of(out) == ["text_id", "text"]

    rows = read_csv(out)
    assert len(rows) == 3
    # Without id_cols, ids are synthetic and 1-based in file order.
    assert [r["text_id"] for r in rows] == ["row_1", "row_2", "row_3"]
    assert rows[0]["text"] == "first thing alice said"


def test_id_cols_become_both_the_id_and_their_own_columns(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv,
        out_csv=tmp_path / "out.csv",
        text_cols=["text"],
        id_cols=["speaker", "session"],
    )
    assert header_of(out) == ["text_id", "speaker", "session", "text"]

    rows = read_csv(out)
    assert rows[0]["text_id"] == "alice | s1"     # composed with " | "
    assert rows[0]["speaker"] == "alice"          # and carried through as data
    assert rows[1]["text_id"] == "bob | s1"


def test_multiple_text_cols_are_joined_by_the_joiner(tmp_path):
    src = tmp_path / "qa.csv"
    src.write_text("prompt,response\nwhat is up,not much\n", encoding="utf-8")

    out = csv_to_analysis_ready_csv(
        csv_path=src,
        out_csv=tmp_path / "out.csv",
        text_cols=["prompt", "response"],
        joiner=" || ",
    )
    assert read_csv(out)[0]["text"] == "what is up || not much"


def test_separate_mode_emits_one_row_per_text_column(tmp_path):
    src = tmp_path / "qa.csv"
    src.write_text("prompt,response\nwhat is up,not much\n", encoding="utf-8")

    out = csv_to_analysis_ready_csv(
        csv_path=src,
        out_csv=tmp_path / "out.csv",
        text_cols=["prompt", "response"],
        mode="separate",
    )
    assert header_of(out) == ["text_id", "text", "source_col"]

    rows = read_csv(out)
    assert len(rows) == 2
    assert [r["source_col"] for r in rows] == ["prompt", "response"]
    assert [r["text"] for r in rows] == ["what is up", "not much"]


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — grouping
# ---------------------------------------------------------------------------

def test_group_by_concatenates_within_group_and_counts_pieces(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv,
        out_csv=tmp_path / "grouped.csv",
        text_cols=["text"],
        group_by=["speaker"],
    )
    assert header_of(out) == ["text_id", "speaker", "text", "group_count"]

    # Bucket order is hash-based, so index by speaker rather than by position.
    by_speaker = {r["speaker"]: r for r in read_csv(out)}
    assert set(by_speaker) == {"alice", "bob"}

    assert by_speaker["alice"]["group_count"] == "2"
    assert by_speaker["alice"]["text"] == "first thing alice said second thing alice said"
    assert by_speaker["bob"]["group_count"] == "1"


def test_group_by_multiple_keys_composes_the_id(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv,
        out_csv=tmp_path / "grouped.csv",
        text_cols=["text"],
        group_by=["session", "speaker"],
    )
    assert header_of(out) == ["text_id", "session", "speaker", "text", "group_count"]
    ids = {r["text_id"] for r in read_csv(out)}
    assert ids == {"s1 | alice", "s1 | bob"}


def test_grouping_is_correct_when_rows_spill_across_many_buckets(tmp_path):
    """
    Grouping does not require a sorted input: rows are hash-partitioned to
    on-disk buckets and aggregated per bucket. This forces that path hard —
    many groups, few buckets, and a writer cache too small to hold them —
    to prove nothing is dropped or truncated when files are evicted and
    reopened.
    """
    src = tmp_path / "big.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "text"])
        for i in range(300):                       # 30 speakers x 10 rows, interleaved
            w.writerow([f"spk{i % 30}", f"word{i}"])

    out = csv_to_analysis_ready_csv(
        csv_path=src,
        out_csv=tmp_path / "out.csv",
        text_cols=["text"],
        group_by=["speaker"],
        num_buckets=4,              # far fewer buckets than groups
        max_open_bucket_files=2,    # forces the LRU to evict and reopen files
    )

    rows = read_csv(out)
    assert len(rows) == 30
    assert all(r["group_count"] == "10" for r in rows)
    # every original token survives exactly once
    tokens = " ".join(r["text"] for r in rows).split()
    assert sorted(tokens) == sorted(f"word{i}" for i in range(300))


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — edge cases
# ---------------------------------------------------------------------------

def test_rows_with_no_text_are_skipped(tmp_path):
    src = tmp_path / "gaps.csv"
    src.write_text("text\nsomething\n\nsomething else\n", encoding="utf-8")

    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "out.csv", text_cols=["text"]
    )
    assert [r["text"] for r in read_csv(out)] == ["something", "something else"]


def test_delimiter_is_sniffed_when_not_given(tmp_path):
    src = tmp_path / "semis.csv"
    src.write_text("speaker;text\nalice;hello there\n", encoding="utf-8")

    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "out.csv", text_cols=["text"], delimiter=None
    )
    assert read_csv(out)[0]["text"] == "hello there"


def test_missing_text_column_raises_with_a_useful_message(utterances_csv, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv, out_csv=tmp_path / "out.csv", text_cols=["nope"]
        )


def test_missing_group_column_raises(utterances_csv, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv,
            out_csv=tmp_path / "out.csv",
            text_cols=["text"],
            group_by=["nope"],
        )


@pytest.mark.parametrize("bad_mode", ["", "CONCAT_ALL", "merge"])
def test_invalid_mode_raises(utterances_csv, tmp_path, bad_mode):
    with pytest.raises(ValueError, match="mode"):
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv,
            out_csv=tmp_path / "out.csv",
            text_cols=["text"],
            mode=bad_mode,
        )


def test_empty_text_cols_raises(utterances_csv, tmp_path):
    with pytest.raises(ValueError, match="text_cols"):
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv, out_csv=tmp_path / "out.csv", text_cols=[]
        )


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — contracts
# ---------------------------------------------------------------------------

def test_default_output_path_is_derived_from_the_input(utterances_csv):
    out = Path(csv_to_analysis_ready_csv(csv_path=utterances_csv, text_cols=["text"]))
    assert out == utterances_csv.parent / "utterances_concat_text.csv"


def test_default_output_path_notes_the_grouping(utterances_csv):
    out = Path(
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv, text_cols=["text"], group_by=["speaker"]
        )
    )
    assert out == utterances_csv.parent / "utterances_grouped_speaker.csv"


def test_existing_output_is_left_alone_by_default(utterances_csv, tmp_path):
    out_path = tmp_path / "out.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")

    returned = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=out_path, text_cols=["text"]
    )

    assert Path(returned) == out_path
    assert out_path.read_text(encoding="utf-8") == "sentinel\n"


def test_overwrite_existing_rebuilds_from_the_current_source(utterances_csv, tmp_path):
    """The regression that let stale intermediates leak into feature files."""
    out_path = tmp_path / "out.csv"
    csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=out_path, text_cols=["text"]
    )
    assert len(read_csv(out_path)) == 3

    with utterances_csv.open("a", encoding="utf-8") as f:
        f.write("carol,s1,a brand new utterance,n4\n")

    csv_to_analysis_ready_csv(
        csv_path=utterances_csv,
        out_csv=out_path,
        text_cols=["text"],
        overwrite_existing=True,
    )
    assert len(read_csv(out_path)) == 4


# ---------------------------------------------------------------------------
# txt_folder_to_analysis_ready_csv
# ---------------------------------------------------------------------------

@pytest.fixture
def txt_tree(tmp_path) -> Path:
    root = tmp_path / "corpus"
    (root / "nested").mkdir(parents=True)
    (root / "one.txt").write_text("the first document", encoding="utf-8")
    (root / "two.txt").write_text("the second document", encoding="utf-8")
    (root / "nested" / "three.txt").write_text("a nested document", encoding="utf-8")
    (root / "ignore.md").write_text("not a text file", encoding="utf-8")
    return root


def test_txt_folder_reads_one_row_per_file(txt_tree, tmp_path):
    out = txt_folder_to_analysis_ready_csv(
        root_dir=txt_tree, out_csv=tmp_path / "out.csv"
    )
    rows = read_csv(out)
    assert {r["text_id"] for r in rows} == {"one", "two"}   # non-recursive by default
    assert {r["text"] for r in rows} == {"the first document", "the second document"}


def test_txt_folder_recursive_picks_up_subfolders(txt_tree, tmp_path):
    out = txt_folder_to_analysis_ready_csv(
        root_dir=txt_tree, out_csv=tmp_path / "out.csv", recursive=True
    )
    assert len(read_csv(out)) == 3


def test_txt_folder_pattern_filters_by_extension(txt_tree, tmp_path):
    out = txt_folder_to_analysis_ready_csv(
        root_dir=txt_tree, out_csv=tmp_path / "out.csv", pattern="*.md"
    )
    rows = read_csv(out)
    assert [r["text_id"] for r in rows] == ["ignore"]


@pytest.mark.parametrize(
    "id_from,expected",
    [("stem", "one"), ("name", "one.txt")],
)
def test_txt_folder_id_from_controls_the_id(txt_tree, tmp_path, id_from, expected):
    out = txt_folder_to_analysis_ready_csv(
        root_dir=txt_tree, out_csv=tmp_path / "out.csv", id_from=id_from
    )
    assert expected in {r["text_id"] for r in read_csv(out)}


def test_txt_folder_existing_output_is_left_alone_by_default(txt_tree, tmp_path):
    out_path = tmp_path / "out.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")
    txt_folder_to_analysis_ready_csv(root_dir=txt_tree, out_csv=out_path)
    assert out_path.read_text(encoding="utf-8") == "sentinel\n"
