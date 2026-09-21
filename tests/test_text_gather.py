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
    # without id_cols, the ids are made up for us: 1-based, in file order
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

    # bucket order is hash-based, so we index by speaker rather than by position
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


def test_no_text_cols_wrangles_without_a_text_column(utterances_csv, tmp_path):
    """A spreadsheet without text (or whose text the user does not want) can
    still be combined and summarized; the output simply has no text column."""
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv", text_cols=[],
        group_by=["speaker"], verbose=False)
    rows = {r["text_id"]: r for r in read_csv(out)}
    assert "text" not in header_of(out)
    assert rows["alice"]["group_count"] == "2"
    assert rows["bob"]["group_count"] == "1"


def test_no_text_cols_without_grouping_copies_every_row(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv", text_cols=None,
        id_cols=["speaker"], verbose=False)
    rows = read_csv(out)
    assert "text" not in header_of(out)
    assert len(rows) == 3 and rows[0]["speaker"] == "alice"


def test_no_text_cols_with_separate_mode_is_refused(utterances_csv, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        csv_to_analysis_ready_csv(
            csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
            text_cols=[], mode="separate", verbose=False)


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
    """The resume contract, for a gathered table: an existing output that
    was made *this way* is handed back untouched. A file with no record of
    how it was made is not trusted -- a gather takes seconds, and a stale
    one from another pipeline in the same folder sank a real run -- so the
    sentinel here is a real gathered table with its record beside it."""
    out_path = tmp_path / "out.csv"
    csv_to_analysis_ready_csv(csv_path=utterances_csv, out_csv=out_path,
                              text_cols=["text"])
    stamp = out_path.read_bytes()

    returned = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=out_path, text_cols=["text"]
    )

    assert Path(returned) == out_path
    assert out_path.read_bytes() == stamp


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
    """Same contract as the spreadsheet gather: reused when its record says
    it was made this way, rebuilt when there is no record to say so."""
    out_path = tmp_path / "out.csv"
    txt_folder_to_analysis_ready_csv(root_dir=txt_tree, out_csv=out_path)
    stamp = out_path.read_bytes()
    txt_folder_to_analysis_ready_csv(root_dir=txt_tree, out_csv=out_path)
    assert out_path.read_bytes() == stamp

    out_path.write_text("sentinel\n", encoding="utf-8")   # no record: redone
    txt_folder_to_analysis_ready_csv(root_dir=txt_tree, out_csv=out_path)
    assert out_path.read_text(encoding="utf-8") != "sentinel\n"


# ---------------------------------------------------------------------------
# carry_cols — columns that survive the gather
#
# every text analyzer offers `pass_through_cols`, but it gets applied when the
# analysis-ready CSV is read back, not when it's written. since the writer only
# ever emitted `text_id` and `text`, every column we asked for showed up
# present-but-empty, and an aggregation grouping on one of them collapsed into
# a single meaningless bucket without ever failing
# ---------------------------------------------------------------------------

def test_a_carried_column_reaches_the_output(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text"], carry_cols=["speaker"],
    )
    assert header_of(out) == ["text_id", "speaker", "text"]
    assert [r["speaker"] for r in read_csv(out)] == ["alice", "bob", "alice"]


def test_a_carried_column_is_not_an_id_column(utterances_csv, tmp_path):
    """
    The distinction the old code had no way to express. `id_cols` *composes*
    `text_id`, so carrying `speaker` that way would give both of alice's
    utterances the same id -- which is the one thing text_id must not do.
    """
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text"], carry_cols=["speaker"],
    )
    rows = read_csv(out)
    ids = [r["text_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    # and the column really is carried along, not just left out of the id
    assert [r["speaker"] for r in rows] == ["alice", "bob", "alice"]


def test_carried_columns_survive_separate_mode(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text", "note"], mode="separate", carry_cols=["speaker"],
    )
    rows = read_csv(out)
    assert len(rows) == 6                       # three rows x two text columns
    assert all(r["speaker"] in {"alice", "bob"} for r in rows)


def test_a_carried_column_survives_grouping_when_the_group_agrees(utterances_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text"], group_by=["speaker"], carry_cols=["session"],
    )
    rows = read_csv(out)
    assert {r["session"] for r in rows} == {"s1"}


def test_a_carried_column_is_blank_where_the_group_disagrees(tmp_path):
    """
    Two sessions collapsed into one speaker row have no single session. Writing
    the first one seen would be a guess indistinguishable from a fact once it is
    in a published CSV, so the cell is left empty instead.
    """
    src = tmp_path / "in.csv"
    src.write_text(
        "speaker,session,text\n"
        "alice,s1,one\n"
        "alice,s2,two\n",
        encoding="utf-8",
    )
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "out.csv",
        text_cols=["text"], group_by=["speaker"], carry_cols=["session"],
    )
    assert [r["session"] for r in read_csv(out)] == [""]


def test_a_column_asked_for_twice_is_written_once(utterances_csv, tmp_path):
    """`speaker` is a reasonable thing to group by *and* to carry."""
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text"], group_by=["speaker"], carry_cols=["speaker"],
    )
    header = header_of(out)
    assert header.count("speaker") == 1
    assert "speaker" in header
    assert {r["speaker"] for r in read_csv(out)} == {"alice", "bob"}


def test_a_carried_column_the_input_lacks_is_dropped_with_a_warning(
    utterances_csv, tmp_path, capsys
):
    """
    Not an error, unlike a missing `text_cols`: half the presets asking for
    `speaker` run over essays that never had one. But it must be said out loud,
    because the alternative -- a column of blanks -- is what this whole feature
    exists to stop.
    """
    out = csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "out.csv",
        text_cols=["text"], carry_cols=["speaker", "nonesuch"],
    )
    assert "nonesuch" in capsys.readouterr().out
    assert header_of(out) == ["text_id", "speaker", "text"]


# ---------------------------------------------------------------------------
# malformed dictionaries fail with their name on (a run of ours died like this)
# ---------------------------------------------------------------------------


def test_a_headerless_dic_is_named_and_explained(tmp_path):
    """
    A bare word list saved as `.dic` -- no %...% category block -- made
    contentcoder die with a bare "list index out of range": no file name, no
    hint, an empty features folder. Real case: an invective word list from a
    shared dictionary library.
    """
    import pytest

    from taters.text.dictionary_analyzers.multi_dict_analyzer import _load_coders

    bad = tmp_path / "invective words.dic"
    bad.write_text("abnormal\nabusive\nafraid\n", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        _load_coders([bad])
    msg = str(err.value)
    assert "invective words.dic" in msg
    assert "category header" in msg


def test_an_empty_dic_is_named_too(tmp_path):
    import pytest

    from taters.text.dictionary_analyzers.multi_dict_analyzer import _load_coders

    empty = tmp_path / "hollow.dic"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        _load_coders([empty])
    assert "hollow.dic" in str(err.value) and "empty" in str(err.value)


def test_a_valid_dic_still_loads(tmp_path):
    from taters.text.dictionary_analyzers.multi_dict_analyzer import _load_coders

    good = tmp_path / "moods.dic"
    good.write_text("%\n1\tAnimals\n2\tCats\n%\ncat\t1\t2\ndog\t1\n",
                    encoding="utf-8")

    coders = _load_coders([good])
    assert len(coders) == 1 and coders[0][0] == "moods"


def test_one_bad_dictionary_costs_only_itself(tmp_path):
    """
    From the user's second failed run: a single headerless .dic zeroed the
    whole content-coding step -- no features from the seven good dictionaries
    either. A broken file is now skipped with a warning naming it; only when
    NOTHING loads does the step fail.
    """
    import warnings

    from taters.text.dictionary_analyzers.multi_dict_analyzer import _load_coders

    good = tmp_path / "moods.dic"
    good.write_text("%\n1\tCats\n%\ncat\t1\n", encoding="utf-8")
    bad = tmp_path / "invectives.dic"
    bad.write_text("abnormal\nabusive\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coders = _load_coders([good, bad])

    assert [prefix for prefix, _ in coders] == ["moods"]
    messages = [str(w.message) for w in caught]
    assert any("invectives.dic" in m and "Skipped" in m for m in messages)


def test_when_no_dictionary_loads_the_step_still_fails(tmp_path):
    """With nothing loaded there is no result to stand behind."""
    import pytest

    from taters.text.dictionary_analyzers.multi_dict_analyzer import _load_coders

    bad = tmp_path / "invectives.dic"
    bad.write_text("abnormal\n", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        _load_coders([bad])
    assert "none of the dictionaries" in str(err.value)
    assert "invectives.dic" in str(err.value)


def test_an_id_column_named_text_id_is_not_duplicated(tmp_path):
    """
    From a real run: the analyzer writes `text_id` explicitly as the first
    column AND injected id_cols right after it -- so an id column actually
    called `text_id` (the obvious name for one) came out twice, identical.
    Readability and lexical richness already filtered it; dictionaries had not.
    """
    import csv as _csv

    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    dic = tmp_path / "cats.dic"
    dic.write_text("%\n1\tCats\n%\ncat\t1\ncats\t1\n", encoding="utf-8")
    src = tmp_path / "texts.csv"
    src.write_text("text_id,text\na.txt,cats are angry\nb.txt,a calm cat\n",
                   encoding="utf-8")

    out = analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], id_cols=["text_id"], mode="concat",
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "dictionary.csv",
        dict_paths=[dic], overwrite_existing=True)

    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.reader(f))
    assert rows[0].count("text_id") == 1
    assert rows[1][0] == "a.txt"


def test_a_differently_named_id_column_still_passes_through(tmp_path):
    """Two id columns compose text_id together, so each is a real part of the
    identity and rides along right after text_id -- once. A lone id column
    IS text_id, and is not repeated beside it (resolve_passthrough_columns)."""
    import csv as _csv

    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    dic = tmp_path / "cats.dic"
    dic.write_text("%\n1\tCats\n%\ncat\t1\n", encoding="utf-8")
    src = tmp_path / "texts.csv"
    src.write_text("speaker,turn,text\nA,1,one cat\nB,2,no animals\n",
                   encoding="utf-8")

    out = analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], id_cols=["speaker", "turn"],
        mode="concat", gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "dictionary.csv",
        dict_paths=[dic], overwrite_existing=True)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.reader(f))
    assert rows[0][:3] == ["text_id", "speaker", "turn"]
    assert rows[0].count("speaker") == 1

    alone = analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], id_cols=["speaker"], mode="concat",
        gathered_csv=tmp_path / "g1.csv",
        out_features_csv=tmp_path / "d1.csv",
        dict_paths=[dic], overwrite_existing=True)
    with open(alone, encoding="utf-8-sig", newline="") as f:
        header = next(_csv.reader(f))
    assert "speaker" not in header, "the lone identity column was carried twice"


def test_archetypes_do_not_duplicate_a_text_id_id_column(tmp_path, monkeypatch):
    """
    Same defect, same fix, in the archetype analyzer -- proven without loading
    a sentence-transformer by capturing what the middle layer is handed.
    """
    from taters.text import analyze_with_archetypes as awa
    from taters.text.dictionary_analyzers import multi_archetype_analyzer as maa

    seen = {}

    def capture(*, items, pass_through_cols, **kwargs):
        seen["pass_through_cols"] = list(pass_through_cols)
        seen["metas"] = [meta for _tid, _txt, meta in items]

    monkeypatch.setattr(maa, "analyze_texts_to_csv", capture)

    src = tmp_path / "texts.csv"
    src.write_text("text_id,speaker,text\na.txt,A,hello there\n",
                   encoding="utf-8")
    # only checked for existence up-front, never read: the stub stands in for
    # the model-loading middle layer
    arche = tmp_path / "themes.csv"
    arche.write_text("archetype,prompt\ncalm,a calm scene\n", encoding="utf-8")
    awa.analyze_with_archetypes(
        csv_path=src, text_cols=["text"], id_cols=["text_id", "speaker"],
        mode="concat", gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "archetypes.csv",
        archetype_csvs=[arche])

    assert seen["pass_through_cols"] == ["speaker"]
    assert seen["metas"] == [{"speaker": "A"}]


def test_the_dictionary_analyzer_names_the_row_it_is_scoring(tmp_path):
    """The tick carries the text_id, so a document that takes minutes shows
    *which* document is taking minutes."""
    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    dic = tmp_path / "cats.dic"
    dic.write_text("%\n1\tCats\n%\ncat\t1\n", encoding="utf-8")
    src = tmp_path / "texts.csv"
    src.write_text("text_id,text\nslowpoke,the cat sat\n", encoding="utf-8")

    calls = []
    analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "g.csv", out_features_csv=tmp_path / "f.csv",
        dict_paths=[dic], overwrite_existing=True,
        on_progress=lambda done, total, message=None, unit=None, **extra:
            calls.append((message, extra.get("inflight"))))
    assert any(m == "scoring documents" for m, _f in calls)
    assert any(f and "slowpoke" in f for _m, f in calls)


def test_parallel_dictionary_scoring_is_byte_identical_to_serial(tmp_path):
    """One worker or three, the feature CSV is the same file: rows are scored
    by one shared function and written in input order."""
    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    dic = tmp_path / "cats.dic"
    dic.write_text("%\n1\tCats\n%\ncat\t1\ncats\t1\n", encoding="utf-8")
    src = tmp_path / "texts.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["text_id", "text"])
        for i in range(12):
            w.writerow([f"d{i}", f"cats appear {i} times in text {i} " * (i + 1)])

    outs = {}
    for n in (1, 3):
        out = analyze_with_dictionaries(
            csv_path=src, text_cols=["text"], id_cols=["text_id"],
            gathered_csv=tmp_path / f"g{n}.csv",
            out_features_csv=tmp_path / f"f{n}.csv",
            dict_paths=[dic], overwrite_existing=True, workers=n)
        outs[n] = Path(out).read_bytes()
    assert outs[1] == outs[3]


def test_a_quiet_dictionary_run_prints_nothing_over_the_display(tmp_path, capfd):
    """
    contentcoder prints "Dictionary loaded." per construction -- and worker
    processes print to the inherited file descriptor, straight past the live
    display's stdout redirection. A quiet run (one driven by a progress sink)
    must reach stdout not at all, parent or child. capfd, not capsys: only
    descriptor capture sees what children write.
    """
    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    dic = tmp_path / "cats.dic"
    dic.write_text("%\n1\tCats\n%\ncat\t1\n", encoding="utf-8")
    src = tmp_path / "texts.csv"
    import csv as _csv
    with src.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["text_id", "text"])
        for i in range(6):
            w.writerow([f"d{i}", f"the cat sat {i}"])

    analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "g.csv", out_features_csv=tmp_path / "f.csv",
        dict_paths=[dic], overwrite_existing=True, workers=2,
        on_progress=lambda *a, **k: None)

    out = capfd.readouterr().out
    assert "Dictionary loaded" not in out


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — per-group summaries (agg_cols)
# ---------------------------------------------------------------------------

@pytest.fixture
def scored_csv(tmp_path) -> Path:
    """Posts with a numeric score column -- one value unparseable, one blank."""
    p = tmp_path / "scored.csv"
    p.write_text(
        "user,score,age,text\n"
        "alice,3,30,meow one\n"
        "alice,4,,meow two\n"
        "bob,oops,41,purr\n",
        encoding="utf-8",
    )
    return p


def test_agg_cols_average_numbers_per_group(scored_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=scored_csv, out_csv=tmp_path / "o.csv", text_cols=["text"],
        group_by=["user"], agg_cols={"score": "mean", "age": "mean"},
        verbose=False)
    rows = {r["text_id"]: r for r in read_csv(out)}
    assert rows["alice"]["score_mean"] == "3.5"
    # blank cells get skipped, not counted as zero: alice's mean age is 30
    # (one value, one blank), not 15, and the n column says so
    assert rows["alice"]["age_mean"] == "30"
    assert rows["alice"]["age_n"] == "1"
    assert rows["alice"]["score_n"] == "2"
    # an unparseable value isn't a number, so bob's only score is no score
    assert rows["bob"]["score_mean"] == ""
    assert rows["bob"]["score_n"] == "0"
    assert rows["bob"]["age_mean"] == "41"
    # the summaries live in their own columns after group_count, each statistic
    # followed by its n
    hdr = header_of(out)
    assert hdr.index("score_mean") > hdr.index("group_count")
    assert hdr.index("score_n") == hdr.index("score_mean") + 1


def test_agg_cols_as_a_bare_list_means_the_mean(scored_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=scored_csv, out_csv=tmp_path / "o.csv", text_cols=["text"],
        group_by=["user"], agg_cols=["score"], verbose=False)
    assert {r["text_id"]: r["score_mean"] for r in read_csv(out)} == {
        "alice": "3.5", "bob": ""}


def test_agg_cols_sum_min_and_max(scored_csv, tmp_path):
    out = csv_to_analysis_ready_csv(
        csv_path=scored_csv, out_csv=tmp_path / "o.csv", text_cols=["text"],
        group_by=["user"], agg_cols={"score": "sum", "age": "min"},
        verbose=False)
    rows = {r["text_id"]: r for r in read_csv(out)}
    assert rows["alice"]["score_sum"] == "7"
    assert rows["alice"]["age_min"] == "30"

    out = csv_to_analysis_ready_csv(
        csv_path=scored_csv, out_csv=tmp_path / "o2.csv", text_cols=["text"],
        group_by=["user"], agg_cols={"score": "max"}, verbose=False)
    assert {r["text_id"]: r["score_max"] for r in read_csv(out)} == {
        "alice": "4", "bob": ""}


def test_agg_cols_without_groups_is_refused(scored_csv, tmp_path):
    """No groups means nothing to summarize over -- silently ignoring the ask
    would hide a real mistake."""
    with pytest.raises(ValueError, match="group_by"):
        csv_to_analysis_ready_csv(
            csv_path=scored_csv, out_csv=tmp_path / "o.csv",
            text_cols=["text"], agg_cols=["score"], verbose=False)


def test_agg_cols_with_an_unknown_statistic_is_refused(scored_csv, tmp_path):
    with pytest.raises(ValueError, match="median"):
        csv_to_analysis_ready_csv(
            csv_path=scored_csv, out_csv=tmp_path / "o.csv",
            text_cols=["text"], group_by=["user"],
            agg_cols={"score": "median"}, verbose=False)


def test_agg_cols_missing_from_the_source_raise(scored_csv, tmp_path):
    """A summary is an explicit computation, not a convenience like carry_cols:
    a misspelled column must not vanish into a silently absent output column."""
    with pytest.raises(ValueError, match="karma"):
        csv_to_analysis_ready_csv(
            csv_path=scored_csv, out_csv=tmp_path / "o.csv",
            text_cols=["text"], group_by=["user"], agg_cols=["karma"],
            verbose=False)


# ---------------------------------------------------------------------------
# csv_to_analysis_ready_csv — progress reporting
# ---------------------------------------------------------------------------

def test_the_csv_gather_names_its_phases(utterances_csv, tmp_path):
    """A silent spinner over a big file reads as a hang: the gather must say
    what it is doing -- counting, sorting into groups, combining them."""
    heard = []

    def sink(done, total, message=None, **_):
        heard.append((done, total, message))

    csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "o.csv",
        text_cols=["text"], group_by=["speaker"], verbose=False,
        on_progress=sink)

    messages = {m for _, _, m in heard if m}
    assert "counting rows" in messages
    assert "sorting rows into groups (pass 1 of 2)" in messages
    assert "combining the groups (pass 2 of 2)" in messages
    # the sort pass has a real denominator: three utterances to sort
    assert (3, 3, "sorting rows into groups (pass 1 of 2)") in heard


def test_the_ungrouped_csv_gather_reports_rows_too(utterances_csv, tmp_path):
    heard = []

    def sink(done, total, message=None, **_):
        heard.append((done, total, message))

    csv_to_analysis_ready_csv(
        csv_path=utterances_csv, out_csv=tmp_path / "o.csv",
        text_cols=["text"], verbose=False, on_progress=sink)

    assert (3, 3, "copying rows") in heard


def test_summaries_run_over_textless_rows_too(tmp_path):
    """
    The audit question that prompted this: a user with rows whose text cell
    is empty. group_count follows the text (those rows are not joined), but
    the numeric summary follows the numbers -- every row of the group counts,
    and the <col>_n column shows the resulting N so the two denominators are
    never confused.
    """
    src = tmp_path / "posts.csv"
    src.write_text(
        "user,score,text\n"
        "alice,3,meow one\n"
        "alice,4,meow two\n"
        "alice,5,\n"          # no text: not joined, not counted, but still scored
        "bob,10,purr\n",
        encoding="utf-8",
    )
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "o.csv", text_cols=["text"],
        group_by=["user"], agg_cols=["score"], verbose=False)

    rows = {r["text_id"]: r for r in read_csv(out)}
    assert rows["alice"]["group_count"] == "2", "two rows had text to join"
    assert rows["alice"]["text"] == "meow one meow two"
    assert rows["alice"]["score_mean"] == "4", "the mean of 3, 4 and 5"
    assert rows["alice"]["score_n"] == "3", "all three scores counted"


# ---------------------------------------------------------------------------
# Leaving rows out
# ---------------------------------------------------------------------------

def _gathered(path):
    import csv

    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_a_row_is_left_out_before_it_joins_anybody_elses_text(tmp_path):
    """
    The reason this lives in the gather rather than in a filter afterwards:
    once a row is inside a joined text there is no way to take it back out,
    so the only time it can be dropped is before.
    """
    src = tmp_path / "turns.csv"
    src.write_text("who,arm,text\n"
                   "a,keep,one two three\n"
                   "a,drop,rubbish\n"
                   "b,keep,four five six\n", encoding="utf-8")
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "g.csv", text_cols=["text"],
        group_by=["who"], row_filters=[["arm", "in", ["keep"]]],
        verbose=False)
    rows = {r["who"]: r for r in _gathered(out)}

    assert "rubbish" not in rows["a"]["text"]
    assert int(rows["a"]["group_count"]) == 1, "the dropped row was counted"


def test_the_filter_judges_the_spreadsheets_own_columns(tmp_path):
    """Nothing has been measured yet, so what it can read is what came in
    the file."""
    src = tmp_path / "t.csv"
    src.write_text("age,text\n17,too young\n40,about right\n",
                   encoding="utf-8")
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "g.csv", text_cols=["text"],
        row_filters=[["age", ">=", 18]], verbose=False)

    assert [r["text"] for r in _gathered(out)] == ["about right"]


def test_every_filter_has_to_be_cleared_not_just_one(tmp_path):
    src = tmp_path / "t.csv"
    src.write_text("age,arm,text\n40,A,yes\n40,B,no\n17,A,no\n",
                   encoding="utf-8")
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "g.csv", text_cols=["text"],
        row_filters=[["age", ">=", 18], ["arm", "in", ["A"]]], verbose=False)

    assert [r["text"] for r in _gathered(out)] == ["yes"]


def test_a_row_dropped_by_a_filter_does_not_shift_the_others_ids(tmp_path):
    """
    `text_id` is `row_<n>` over the *source* rows, and a key map gathered
    without any text keeps every row. If dropping a row renumbered the rest,
    every feature would join the wrong group and the numbers would all be
    wrong while looking perfectly reasonable.
    """
    src = tmp_path / "t.csv"
    src.write_text("arm,text\nkeep,one two\ndrop,no\nkeep,three four\n",
                   encoding="utf-8")
    measured = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "m.csv", text_cols=["text"],
        row_filters=[["arm", "in", ["keep"]]], verbose=False)
    keys = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "k.csv", text_cols=[],
        carry_cols=["arm"], verbose=False)

    assert [r["text_id"] for r in _gathered(measured)] == ["row_1", "row_3"]
    assert set(r["text_id"] for r in _gathered(measured)) <= \
        set(r["text_id"] for r in _gathered(keys))


def test_no_filter_leaves_every_row_alone(tmp_path):
    src = tmp_path / "t.csv"
    src.write_text("arm,text\ndrop,x\nkeep,one two three\n",
                   encoding="utf-8")
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "g.csv", text_cols=["text"],
        verbose=False)

    assert len(_gathered(out)) == 2


def test_a_blank_cell_in_the_filtered_column_does_not_survive(tmp_path):
    """An unknown value is not evidence of anything, and keeping it means a
    half-empty column quietly stops filtering."""
    src = tmp_path / "t.csv"
    src.write_text("age,text\n,unknown age\n40,known\n", encoding="utf-8")
    out = csv_to_analysis_ready_csv(
        csv_path=src, out_csv=tmp_path / "g.csv", text_cols=["text"],
        row_filters=[["age", ">=", 18]], verbose=False)

    assert [r["text"] for r in _gathered(out)] == ["known"]
