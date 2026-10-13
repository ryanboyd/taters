"""
Merging, stacking and reshaping spreadsheets, on every messy input we could
think of.

The helpers are built around a plan that writes nothing and a writer that
runs the same plan again, so a lot of what is tested here is that the two
agree: the row count the plan promised is the row count written, every time.
The rest is the edge cases the design doc decided one by one -- keys that
differ by case or leading zeros, a column present on both sides, a cp1252
export, an ID Excel turned into 1.23457E+11 -- each with a known answer.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from csvhelpers import read_rows, write_rows
from taters.helpers import reshape
from taters.helpers.reshape import (detect_format, long_to_wide,
                                    merge_spreadsheets, normalize_key,
                                    plan_long_to_wide, plan_merge, plan_stack,
                                    plan_wide_to_long, stack_spreadsheets,
                                    wide_to_long)


def _csv(tmp_path, name, header, rows, **kw):
    return write_rows(tmp_path / name, header, rows)


def _text(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


def _header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh))


@pytest.fixture
def trials(tmp_path):
    """Three trials for p1, two for p2, one for an unmatched p9; p2's age blank."""
    return _csv(tmp_path, "trials.csv", ["pid", "trial", "score", "age"], [
        ["p1", "1", "10", "30"], ["p1", "2", "12", "30"], ["p1", "3", "11", "30"],
        ["p2", "1", "7", ""], ["p2", "2", "8", ""],
        ["p9", "1", "1", "22"],
    ])


@pytest.fixture
def participants(tmp_path):
    """One row each; p3 has no trials."""
    return _csv(tmp_path, "participants.csv", ["pid", "age", "group"], [
        ["p1", "30", "A"], ["p2", "25", "B"], ["p3", "50", "C"],
    ])


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------

def test_a_utf8_bom_never_reaches_a_column_name(tmp_path):
    p = _text(tmp_path, "bom.csv", "﻿pid,x\n1,2\n")
    assert detect_format(p).columns == ("pid", "x")


def test_a_cp1252_file_is_detected_even_when_the_accent_is_deep_in(tmp_path):
    """A 64 KB sample would call this UTF-8; the one accented name is past it."""
    body = "pid,name\n" + "".join(f"p{i},plain\n" for i in range(8000))
    body += "p9,José\n"
    p = _text(tmp_path, "win.csv", body, "cp1252")

    fmt = detect_format(p)

    assert fmt.encoding == "cp1252"
    assert "Windows (cp1252)" in " ".join(fmt.notes())
    rows = list(reshape._Reader(fmt))
    assert rows[-1] == ["p9", "José"]


def test_a_byte_cp1252_does_not_define_falls_back_to_latin1(tmp_path):
    p = tmp_path / "odd.csv"
    p.write_bytes(b"pid,x\n1,\x81\n")       # 0x81 is undefined in cp1252
    assert detect_format(p).encoding == "latin-1"


def test_utf16_unicode_text_export_reads_as_tabs(tmp_path):
    """Excel's "Unicode Text" is UTF-16 with a BOM and tabs; sniffing the raw
    bytes used to call it a one-column comma file."""
    p = tmp_path / "unicode.txt"
    p.write_bytes(codecs_bom_utf16() + "pid\tx\n1\t2\n".encode("utf-16-le"))

    fmt = detect_format(p)

    assert fmt.encoding == "utf-16" and fmt.delimiter == "\t"
    assert fmt.columns == ("pid", "x")
    assert list(reshape._Reader(fmt)) == [["1", "2"]]


def codecs_bom_utf16():
    import codecs
    return codecs.BOM_UTF16_LE


def test_an_explicit_wrong_encoding_says_how_to_fix_it(tmp_path):
    p = _text(tmp_path, "win.csv", "pid,name\n1,José\n", "cp1252")
    with pytest.raises(ValueError, match="leave the encoding unset"):
        detect_format(p, encoding="utf-8")


def test_a_semicolon_file_and_a_comma_file_merge_together(tmp_path):
    left = _text(tmp_path, "l.csv", "pid;x\n1;a\n2;b\n")
    right = _text(tmp_path, "r.csv", "pid,y\n1,A\n2,B\n")
    out = merge_spreadsheets(left_csv=left, right_csv=right, on="pid", verbose=False)
    assert read_rows(out) == [{"pid": "1", "x": "a", "y": "A"},
                              {"pid": "2", "x": "b", "y": "B"}]


def test_an_empty_file_is_refused_and_a_header_only_file_is_fine(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="is empty"):
        detect_format(empty)
    header_only = _text(tmp_path, "h.csv", "pid,x\n")
    fmt = detect_format(header_only)
    assert fmt.columns == ("pid", "x") and list(reshape._Reader(fmt)) == []


def test_an_excel_file_is_refused_with_the_save_as_hint(tmp_path):
    p = tmp_path / "sheet.csv"
    p.write_bytes(b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00")
    with pytest.raises(ValueError, match="Save As"):
        detect_format(p)


def test_duplicate_and_blank_headers_are_renamed_and_announced(tmp_path):
    p = _text(tmp_path, "dup.csv", "pid,score,,score\n1,2,3,4\n")

    fmt = detect_format(p)

    assert fmt.columns == ("pid", "score", "column_3", "score_2")
    assert ("", "column_3") in fmt.renamed and ("score", "score_2") in fmt.renamed
    notes = " ".join(fmt.notes())
    assert "more than one column called score" in notes and "column_3" in notes


def test_short_rows_are_padded_long_rows_trimmed_and_both_counted(tmp_path):
    p = _text(tmp_path, "ragged.csv", "pid,x,y\n1,a\n2,b,c,d\n3,e,f\n")
    reader = reshape._Reader(detect_format(p))

    rows = list(reader)

    assert rows == [["1", "a", ""], ["2", "b", "c"], ["3", "e", "f"]]
    assert (reader.short, reader.long, reader.rows) == (1, 1, 3)


def test_writing_over_an_input_is_refused_even_with_overwrite(trials, participants):
    with pytest.raises(ValueError, match="one of the files being read"):
        merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                           out_csv=trials, overwrite_existing=True, verbose=False)
    assert read_rows(trials)[0]["pid"] == "p1", "and the original is untouched"


def test_an_existing_output_is_returned_untouched_by_default(trials, participants,
                                                            tmp_path, monkeypatch):
    """
    Same settings, same file: the second call returns what is there and does
    no work. (A leftover with no record of how it was made is redone instead,
    the way the text gathers do -- a stale table from an earlier run was once
    trusted and the analyses then could not find their columns.)
    """
    out = tmp_path / "out.csv"
    merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                       out_csv=out, verbose=False)

    def boom(*a, **k):
        raise AssertionError("the merge ran again")
    monkeypatch.setattr(reshape._MergeEngine, "load_right", boom)
    assert merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                              out_csv=out, verbose=False) == out
    assert len(read_rows(out)) == 6


def test_an_interrupted_write_leaves_no_file_and_no_temp_folder(trials, participants,
                                                                tmp_path, monkeypatch):
    out = tmp_path / "out.csv"
    made = []
    real = reshape._sqlite

    def spy(tmp_root):
        con, folder = real(tmp_root)
        made.append(folder)
        return con, folder
    monkeypatch.setattr(reshape, "_sqlite", spy)

    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(reshape._MergeEngine, "write", boom)
    with pytest.raises(KeyboardInterrupt):
        merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                           out_csv=out, verbose=False)

    assert not out.exists() and not list(tmp_path.glob("out.csv*"))
    assert made and not made[0].exists(), "the working database is gone"


def test_the_temp_folder_is_removed_after_success(trials, participants, tmp_path):
    merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                       tmp_root=tmp_path, verbose=False)
    assert not list(tmp_path.glob("taters_reshape_*"))


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,exact,trim,loose", [
    (" 7 ", " 7 ", "7", "7"),
    ("007", "007", "007", "7"),
    ("7.0", "7.0", "7.0", "7"),
    ("+7", "+7", "+7", "7"),
    ("-0", "-0", "-0", "0"),
    (".50", ".50", ".50", "0.5"),
    ("0.50", "0.50", "0.50", "0.5"),
    ("-2.50", "-2.50", "-2.50", "-2.5"),
    ("1e3", "1e3", "1e3", "1e3"),            # not a number here
    ("nan", "nan", "nan", "nan"),
    ("1_000", "1_000", "1_000", "1_000"),
    ("1,234", "1,234", "1,234", "1,234"),
    ("P01", "P01", "P01", "p01"),
    (" x ", " x ", "x", "x"),      # NBSP is whitespace
    ("a​b", "a​b", "a​b", "ab"),         # zero-width space
    ("Mary  Smith", "Mary  Smith", "Mary  Smith", "mary smith"),
    ("José", "José", "José", "josé"),   # NFD -> NFC
])
def test_normalize_key_by_mode(value, exact, trim, loose):
    assert normalize_key(value, "exact") == exact
    assert normalize_key(value, "trim") == trim
    assert normalize_key(value, "loose") == loose


def test_seventeen_digit_ids_never_collide_even_loosely():
    """float() would call these equal. A key is never a float."""
    a, b = "12345678901234567", "12345678901234568"
    assert float(a) == float(b), "the trap is real"
    assert normalize_key(a, "loose") != normalize_key(b, "loose")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="key_match"):
        normalize_key("x", "fuzzy")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def test_left_join_keeps_every_left_row_in_its_own_order(trials, participants):
    out = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                             verbose=False)

    rows = read_rows(out)
    assert [r["pid"] for r in rows] == ["p1", "p1", "p1", "p2", "p2", "p9"]
    assert _header(out) == ["pid", "trial", "score", "age", "group"]
    assert rows[0]["group"] == "A" and rows[-1]["group"] == ""
    assert out.name == "trials_with_participants.csv"


def test_inner_and_outer_counts_and_outer_appends_the_right_only_rows(
        trials, participants, tmp_path):
    report = plan_merge(left_csv=trials, right_csv=participants, on="pid")
    assert report.rows_out == {"inner": 5, "left": 6, "outer": 7}

    inner = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                               how="inner", out_csv=tmp_path / "i.csv", verbose=False)
    outer = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                               how="outer", out_csv=tmp_path / "o.csv", verbose=False)

    assert [r["pid"] for r in read_rows(inner)] == ["p1", "p1", "p1", "p2", "p2"]
    last = read_rows(outer)[-1]
    assert last == {"pid": "p3", "trial": "", "score": "", "age": "50", "group": "C"}, \
        "the right-only row carries its key and its columns, blanks elsewhere"


def test_the_relationship_is_named_and_one_to_many_repeats_left_rows(
        trials, participants, tmp_path):
    many_to_one = plan_merge(left_csv=trials, right_csv=participants, on="pid")
    assert many_to_one.relationship == "many-to-one"

    one_to_many = plan_merge(left_csv=participants, right_csv=trials, on="pid")
    assert one_to_many.relationship == "one-to-many"
    assert one_to_many.rows_out["left"] == 6, "p1 x3, p2 x2, p3 once"
    assert any("repeat once per match" in s for s in one_to_many.sentences())

    out = merge_spreadsheets(left_csv=participants, right_csv=trials, on="pid",
                             verbose=False)
    assert [r["pid"] for r in read_rows(out)] == ["p1", "p1", "p1", "p2", "p2", "p3"]


def test_many_to_many_is_refused_with_the_count_and_examples(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["1", "a"], ["1", "b"], ["2", "c"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["1", "A"], ["1", "B"], ["1", "C"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")

    assert report.relationship == "many-to-many"
    assert report.many_to_many_examples == (("1", 2, 3),)
    assert "2 rows in a.csv × 3 in b.csv = 6" in report.refusal()
    with pytest.raises(ValueError, match="one more column"):
        merge_spreadsheets(left_csv=left, right_csv=right, on="k", verbose=False)

    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k",
                             allow_many_to_many=True, verbose=False)
    assert len(read_rows(out)) == 7, "2x3 for key 1, plus the lone 2"


def test_blank_keys_never_match_and_are_counted_apart(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["", "a"], [" ", "b"], ["1", "c"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["", "A"], ["1", "C"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")

    assert report.left.blank_keys == 2 and report.right.blank_keys == 1
    assert report.left.rows_unmatched == 0, "a blank is not 'unmatched'"
    assert report.rows_out == {"inner": 1, "left": 3, "outer": 4}
    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k", how="outer",
                             verbose=False)
    rows = read_rows(out)
    assert rows[0]["y"] == "" and rows[-1] == {"k": "", "x": "", "y": "A"}


def test_a_composite_key_with_one_blank_part_is_blank(tmp_path):
    left = _csv(tmp_path, "a.csv", ["p", "s", "x"], [["1", "", "a"], ["1", "2", "b"]])
    right = _csv(tmp_path, "b.csv", ["p", "s", "y"], [["1", "", "A"], ["1", "2", "B"]])
    report = plan_merge(left_csv=left, right_csv=right, on=["p", "s"])
    assert report.left.blank_keys == 1 and report.rows_out["inner"] == 1


def test_right_on_maps_a_differently_named_key(trials, tmp_path):
    right = _csv(tmp_path, "people.csv", ["participant", "group"],
                 [["p1", "A"], ["p2", "B"]])
    out = merge_spreadsheets(left_csv=trials, right_csv=right, on="pid",
                             right_on="participant", verbose=False)
    assert _header(out) == ["pid", "trial", "score", "age", "group"], \
        "the right key column is matched, not repeated"
    assert read_rows(out)[0]["group"] == "A"


def test_a_missing_key_column_names_the_near_miss(trials, participants):
    with pytest.raises(ValueError, match="did you mean 'pid'"):
        plan_merge(left_csv=trials, right_csv=participants, on="PID")
    with pytest.raises(ValueError, match="right_on names 2"):
        plan_merge(left_csv=trials, right_csv=participants, on="pid",
                   right_on=["pid", "age"])


def test_trim_is_the_default_so_padded_ids_match(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [[" p1 ", "a"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["p1", "A"]])
    assert plan_merge(left_csv=left, right_csv=right, on="k").rows_out["inner"] == 1
    assert plan_merge(left_csv=left, right_csv=right, on="k",
                      key_match="exact").rows_out["inner"] == 0


def test_loose_is_offered_in_the_plan_but_not_applied_until_asked(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["007", "a"], ["P02", "b"], ["3", "c"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["7", "A"], ["p02", "B"], ["3", "C"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")

    assert report.left.rows_matched == 1 and report.loose_extra_matches == 2
    assert report.loose_examples[0] == ("007", "7")
    assert any("would add 2 rows" in s for s in report.sentences())
    loose = plan_merge(left_csv=left, right_csv=right, on="k", key_match="loose")
    assert loose.left.rows_matched == 3 and loose.loose_extra_matches == 0

    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k",
                             key_match="loose", verbose=False)
    rows = read_rows(out)
    assert [r["k"] for r in rows] == ["007", "P02", "3"], "the left spelling is kept"
    assert [r["y"] for r in rows] == ["A", "B", "C"]


def test_loose_collisions_within_one_file_are_reported(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["aBc", "1"], ["ABC", "2"], ["z", "3"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["abc", "A"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")

    assert report.left.loose_collisions == 1
    assert report.left.loose_collision_examples == (("aBc", "ABC"),)
    assert any("differ only in case" in s for s in report.sentences())


def test_excel_mangled_ids_are_flagged_and_never_matched_as_numbers(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["1.23457E+11", "a"], ["5", "b"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["123457000000", "A"], ["5", "B"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k", key_match="loose")

    assert report.left.mangled_ids == 1 and report.left.rows_matched == 1
    assert any("Re-export that column as text" in s for s in report.sentences())


def test_a_shared_column_that_agrees_is_kept_once_and_filled_in(trials, participants):
    report = plan_merge(left_csv=trials, right_csv=participants, on="pid")

    clash = report.clashes[0]
    assert clash.name == "age" and clash.agrees and clash.output_names == ("age",)
    out = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                             verbose=False)
    rows = read_rows(out)
    assert rows[3]["age"] == "25", "p2's blank age came from participants.csv"
    assert rows[-1]["age"] == "22", "and an unmatched row keeps its own"


def test_a_shared_column_that_disagrees_keeps_both_with_a_suffix(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "group"], [["1", "A"], ["2", "B"]])
    right = _csv(tmp_path, "people.csv", ["k", "group"], [["1", "A"], ["2", "C"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")

    clash = report.clashes[0]
    assert not clash.agrees and clash.disagreements == 1
    assert clash.example == ("2", "B", "C")
    assert clash.output_names == ("group", "people__group")
    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k", verbose=False)
    assert _header(out) == ["k", "group", "people__group"]
    assert read_rows(out)[1] == {"k": "2", "group": "B", "people__group": "C"}


def test_a_shared_column_agrees_when_the_numbers_are_written_differently(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "age"], [["1", "25"]])
    right = _csv(tmp_path, "b.csv", ["k", "age"], [["1", "25.0"]])
    assert plan_merge(left_csv=left, right_csv=right, on="k").clashes[0].agrees


def test_a_taken_suffix_name_gets_a_number(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "g", "b__g"], [["1", "x", "taken"]])
    right = _csv(tmp_path, "b.csv", ["k", "g"], [["1", "y"]])
    report = plan_merge(left_csv=left, right_csv=right, on="k")
    assert report.columns_out == ("k", "g", "b__g", "b__g_2")


def test_right_cols_brings_over_only_what_is_asked(trials, participants):
    out = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                             right_cols=["group", "pid"], verbose=False)
    assert _header(out) == ["pid", "trial", "score", "age", "group"]
    with pytest.raises(ValueError, match="no column 'nope'"):
        plan_merge(left_csv=trials, right_csv=participants, on="pid",
                   right_cols=["nope"])


def test_zero_matches_is_written_with_a_warning_not_refused(tmp_path, capsys):
    left = _csv(tmp_path, "a.csv", ["k", "x"], [["P001", "a"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["1", "A"]])

    report = plan_merge(left_csv=left, right_csv=right, on="k")
    assert not report.matched
    assert "a.csv keys look like: P001" in report.sentences()[0]
    assert "b.csv keys look like: 1" in report.sentences()[0]

    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k")
    assert "WARNING: nothing matched" in capsys.readouterr().out
    assert read_rows(out) == [{"k": "P001", "x": "a", "y": ""}]


@pytest.mark.parametrize("how", ["left", "inner", "outer"])
@pytest.mark.parametrize("shape", ["one-to-one", "many-to-one", "one-to-many"])
def test_the_plan_promises_exactly_the_rows_the_writer_writes(tmp_path, how, shape):
    a = [["1", "a"], ["2", "b"], ["3", "c"], ["", "blank"], ["9", "lonely"]]
    b = [["1", "A"], ["2", "B"], ["3", "C"], ["", "BLANK"], ["8", "LONELY"]]
    if shape == "many-to-one":
        a += [["1", "a2"], ["2", "b2"]]
    if shape == "one-to-many":
        b += [["1", "A2"], ["3", "C2"]]
    left = _csv(tmp_path, "a.csv", ["k", "x"], a)
    right = _csv(tmp_path, "b.csv", ["k", "y"], b)

    report = plan_merge(left_csv=left, right_csv=right, on="k")
    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k", how=how,
                             out_csv=tmp_path / f"{how}.csv", verbose=False)

    assert len(read_rows(out)) == report.rows_out[how]


def test_unmatched_examples_are_the_first_few_keys_in_file_order(tmp_path):
    left = _csv(tmp_path, "a.csv", ["k", "x"],
                [[f"u{i}", "x"] for i in range(9, 0, -1)] + [["m", "x"], ["u9", "again"]])
    right = _csv(tmp_path, "b.csv", ["k", "y"], [["m", "Y"]])
    report = plan_merge(left_csv=left, right_csv=right, on="k")
    assert report.left.unmatched_examples == ("u9", "u8", "u7", "u6", "u5")
    assert report.left.rows_unmatched == 10 and report.left.unmatched_keys == 9


def test_the_report_sentences_read_the_way_the_wizard_shows_them(trials, participants):
    lines = plan_merge(left_csv=trials, right_csv=participants, on="pid").sentences()
    assert lines[0] == ("5 of 6 rows in trials.csv match a row in participants.csv; "
                        "1 do not (e.g. p9).")
    assert lines[1] == ("1 of 3 rows in participants.csv have no match in "
                        "trials.csv (e.g. p3).")
    assert lines[2] == ("age is in both files and agrees on every matched row, "
                        "so it's kept once.")


def test_merge_memory_stays_flat_on_a_big_right_side(tmp_path):
    """200k right rows must not become 200k Python rows: the store is on disk."""
    import tracemalloc
    n = 200_000
    left = _csv(tmp_path, "a.csv", ["k", "x"], [[str(i), "x"] for i in range(0, n, 50)])
    right = _csv(tmp_path, "b.csv", ["k", "y", "z"],
                 [[str(i), f"value {i}", "more text here"] for i in range(n)])
    tracemalloc.start()
    out = merge_spreadsheets(left_csv=left, right_csv=right, on="k", verbose=False)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(read_rows(out)) == n // 50
    assert peak < 40 * 1024 * 1024, f"peak {peak / 1e6:.0f} MB"


def test_merge_records_both_inputs_in_its_provenance(trials, participants, tmp_path):
    """A record is written beside a *named* output, the way every other
    writer does it; the wizard and the pipeline always name theirs."""
    from taters.helpers.provenance import read
    out = merge_spreadsheets(left_csv=trials, right_csv=participants, on="pid",
                             out_csv=tmp_path / "merged.csv", verbose=False)
    rec = read(out)
    assert rec and rec["binding"]["left_csv"].endswith("trials.csv")
    assert rec["binding"]["right_csv"].endswith("participants.csv")
    assert rec["grain"] == {"how": "left"}


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

def test_stack_unions_columns_in_first_seen_order_with_a_source_column(tmp_path):
    a = _csv(tmp_path, "wave1.csv", ["pid", "x"], [["1", "a"]])
    b = _csv(tmp_path, "wave2.csv", ["x", "pid", "y"], [["b", "2", "B"]])

    out = stack_spreadsheets(csv_paths=[a, b], verbose=False)

    assert _header(out) == ["source", "pid", "x", "y"]
    assert read_rows(out) == [{"source": "wave1", "pid": "1", "x": "a", "y": ""},
                              {"source": "wave2", "pid": "2", "x": "b", "y": "B"}]
    assert out.name == "wave1_stacked.csv"


def test_stack_source_collision_and_no_source_at_all(tmp_path):
    a = _csv(tmp_path, "a.csv", ["source", "x"], [["s", "1"]])
    b = _csv(tmp_path, "b.csv", ["source", "x"], [["t", "2"]])
    report = plan_stack(csv_paths=[a, b])
    assert report.source_col == "source_file"
    out = stack_spreadsheets(csv_paths=[a, b], source_col="", verbose=False)
    assert _header(out) == ["source", "x"]


def test_same_stems_in_different_folders_use_the_path(tmp_path):
    a = _csv(tmp_path / "s1", "data.csv", ["x"], [["1"]])
    b = _csv(tmp_path / "s2", "data.csv", ["x"], [["2"]])
    out = stack_spreadsheets(csv_paths=[a, b], out_csv=tmp_path / "all.csv",
                             verbose=False)
    assert [r["source"] for r in read_rows(out)] == ["s1/data", "s2/data"]


def test_header_whitespace_always_merges_but_case_only_when_loose(tmp_path):
    a = _text(tmp_path, "a.csv", "Age ,pid\n1,p\n")
    b = _text(tmp_path, "b.csv", "age,pid\n2,q\n")

    exact = plan_stack(csv_paths=[a, b])
    assert exact.columns_out == ("Age", "pid", "age")
    assert exact.near_duplicates == (("Age", "age"),)
    assert any("differ only in capitals" in s for s in exact.sentences())

    loose = plan_stack(csv_paths=[a, b], match_columns="loose")
    assert loose.columns_out == ("Age", "pid"), "the first spelling wins"
    out = stack_spreadsheets(csv_paths=[a, b], match_columns="loose", verbose=False)
    assert [r["Age"] for r in read_rows(out)] == ["1", "2"]


def test_a_file_sharing_no_columns_is_flagged_as_a_checkerboard(tmp_path):
    a = _csv(tmp_path, "a.csv", ["pid", "x"], [["1", "a"]])
    b = _csv(tmp_path, "b.csv", ["pid", "y"], [["1", "b"]])
    c = _csv(tmp_path, "c.csv", ["q", "r"], [["1", "2"]])
    report = plan_stack(csv_paths=[a, b, c])
    assert report.disjoint_files == ("c.csv",)
    assert any("merge" in s for s in report.sentences())


def test_a_file_listed_twice_is_stacked_once_and_one_file_is_refused(tmp_path):
    a = _csv(tmp_path, "a.csv", ["x"], [["1"]])
    b = _csv(tmp_path, "b.csv", ["x"], [["2"]])
    report = plan_stack(csv_paths=[a, b, a])
    assert report.duplicates_dropped == 1 and report.rows_out == 2
    with pytest.raises(ValueError, match="at least two"):
        plan_stack(csv_paths=[a, a])


def test_stack_reads_each_file_in_its_own_encoding_and_delimiter(tmp_path):
    a = _text(tmp_path, "a.csv", "pid;name\n1;José\n", "cp1252")
    b = _text(tmp_path, "b.csv", "pid,name\n2,Zoë\n", "utf-8")
    out = stack_spreadsheets(csv_paths=[a, b], verbose=False)
    assert [r["name"] for r in read_rows(out)] == ["José", "Zoë"]


# ---------------------------------------------------------------------------
# Wide to long
# ---------------------------------------------------------------------------

@pytest.fixture
def wide(tmp_path):
    return _csv(tmp_path, "wide.csv", ["pid", "group", "score_t1", "score_t2", "score_t3"],
                [["p1", "A", "1", "2", "3"], ["p2", "B", "4", "", "6"]])


def test_unpivot_is_row_major_and_keeps_the_other_columns(wide):
    out = wide_to_long(csv_path=wide, value_cols=["score_t1", "score_t2", "score_t3"],
                       verbose=False)

    rows = read_rows(out)
    assert _header(out) == ["pid", "group", "variable", "value"]
    assert len(rows) == 6
    assert rows[0] == {"pid": "p1", "group": "A", "variable": "score_t1", "value": "1"}
    assert rows[4] == {"pid": "p2", "group": "B", "variable": "score_t2", "value": ""}
    assert out.name == "wide_long.csv"


def test_unpivot_with_no_id_columns_numbers_the_rows(tmp_path):
    p = _csv(tmp_path, "w.csv", ["a", "b"], [["1", "2"], ["3", "4"]])
    out = wide_to_long(csv_path=p, value_cols=["a", "b"], verbose=False)
    assert _header(out) == ["row", "variable", "value"]
    assert [r["row"] for r in read_rows(out)] == ["1", "1", "2", "2"]


def test_unpivot_names_prefix_and_blank_handling(wide):
    report = plan_wide_to_long(csv_path=wide, value_cols=["score_t1", "score_t2", "score_t3"],
                               id_cols=["pid"], names_to="time", values_to="score",
                               strip_prefix="score_t", drop_blank=True)

    assert report.names == ("1", "2", "3")
    assert report.columns_out == ("pid", "time", "score")
    assert report.blanks == 1 and report.dropped == 1 and report.rows_out == 5
    out = wide_to_long(csv_path=wide, value_cols=["score_t1", "score_t2", "score_t3"],
                       id_cols=["pid"], names_to="time", values_to="score",
                       strip_prefix="score_t", drop_blank=True, verbose=False)
    assert len(read_rows(out)) == 5
    with pytest.raises(ValueError, match="do not start with it"):
        plan_wide_to_long(csv_path=wide, value_cols=["score_t1", "group"],
                          strip_prefix="score_t")


def test_unpivot_refuses_names_that_collide(wide):
    with pytest.raises(ValueError, match="already a column called 'group'"):
        plan_wide_to_long(csv_path=wide, value_cols=["score_t1", "score_t2"],
                          names_to="group")
    with pytest.raises(ValueError, match="have to differ"):
        plan_wide_to_long(csv_path=wide, value_cols=["score_t1", "score_t2"],
                          names_to="v", values_to="v")
    with pytest.raises(ValueError, match="at least two"):
        plan_wide_to_long(csv_path=wide, value_cols=["score_t1"])


def test_unpivot_flags_mixed_kinds_and_copies_values_verbatim(tmp_path):
    p = _csv(tmp_path, "w.csv", ["pid", "n", "note"], [["1", "2.50", "fine"]])
    report = plan_wide_to_long(csv_path=p, value_cols=["n", "note"])
    assert report.mixed_kinds == ("note",)
    assert any("read as text" in s for s in report.sentences())
    out = wide_to_long(csv_path=p, value_cols=["n", "note"], verbose=False)
    assert [r["value"] for r in read_rows(out)] == ["2.50", "fine"]


# ---------------------------------------------------------------------------
# Long to wide
# ---------------------------------------------------------------------------

@pytest.fixture
def long(tmp_path):
    return _csv(tmp_path, "long.csv", ["pid", "group", "cond", "score"], [
        ["p1", "A", "B", "2"], ["p1", "A", "A", "1"],
        ["p2", "B", "A", "3"], ["p2", "B", "C", "4"],
    ])


def test_pivot_one_row_per_key_in_first_appearance_order(long):
    out = long_to_wide(csv_path=long, id_cols=["pid"], names_from="cond",
                       values_from="score", carry_cols=["group"], verbose=False)

    assert _header(out) == ["pid", "group", "B", "A", "C"], \
        "names in order of first appearance; carry beside the key"
    assert read_rows(out) == [
        {"pid": "p1", "group": "A", "B": "2", "A": "1", "C": ""},
        {"pid": "p2", "group": "B", "B": "", "A": "3", "C": "4"}]
    assert out.name == "long_wide.csv"


def test_numeric_names_come_out_in_numeric_order(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"],
             [["p", "10", "a"], ["p", "2", "b"], ["p", "1", "c"]])
    out = long_to_wide(csv_path=p, id_cols=["pid"], names_from="t", values_from="v",
                       verbose=False)
    assert _header(out) == ["pid", "1", "2", "10"]


def test_duplicates_are_refused_by_default_then_first_last_or_mean(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"],
             [["p", "a", "1"], ["p", "a", "3"], ["p", "b", "5"]])

    report = plan_long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                               values_from="v")
    assert report.collisions == 1 and report.collisions_numeric
    assert report.collision_examples == (("p", "a", ("1", "3")),)
    with pytest.raises(ValueError, match="2 rows for t = a"):
        long_to_wide(csv_path=p, id_cols=["pid"], names_from="t", values_from="v",
                     verbose=False)

    got = {}
    for policy in ("first", "last", "mean"):
        out = long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                           values_from="v", on_duplicate=policy,
                           out_csv=tmp_path / f"{policy}.csv", verbose=False)
        got[policy] = read_rows(out)[0]["a"]
    assert got == {"first": "1", "last": "3", "mean": "2"}


def test_mean_is_refused_when_a_repeated_value_is_not_a_number(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"], [["p", "a", "1"], ["p", "a", "x"]])
    report = plan_long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                               values_from="v", on_duplicate="mean")
    assert not report.collisions_numeric
    assert "needs numbers" in report.refusal()


def test_blank_names_and_blank_keys_are_left_out_and_counted(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"],
             [["p", "", "1"], ["", "a", "2"], ["p", "a", "3"]])
    report = plan_long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                               values_from="v")
    assert (report.blank_names, report.blank_keys, report.rows_out) == (1, 1, 1)


def test_too_many_generated_columns_is_refused(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"],
             [["p", f"t{i}", "1"] for i in range(12)])
    report = plan_long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                               values_from="v", max_columns=10)
    assert report.too_many_columns
    assert "picked by mistake" in report.refusal()


def test_a_generated_name_that_collides_prefixes_every_generated_column(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v"],
             [["p", "pid", "1"], ["p", "b", "2"]])
    out = long_to_wide(csv_path=p, id_cols=["pid"], names_from="t", values_from="v",
                       verbose=False)
    assert _header(out) == ["pid", "v_pid", "v_b"]


def test_several_value_columns_are_always_prefixed(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "t", "v", "w"], [["p", "a", "1", "x"]])
    out = long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                       values_from=["v", "w"], verbose=False)
    assert _header(out) == ["pid", "v_a", "w_a"]
    assert read_rows(out) == [{"pid": "p", "v_a": "1", "w_a": "x"}]


def test_a_carry_column_that_varies_within_a_key_is_refused_not_blanked(tmp_path):
    p = _csv(tmp_path, "l.csv", ["pid", "age", "t", "v"],
             [["p", "30", "a", "1"], ["p", "", "b", "2"], ["p", "31", "c", "3"]])
    report = plan_long_to_wide(csv_path=p, id_cols=["pid"], names_from="t",
                               values_from="v", carry_cols=["age"])
    assert report.varying_carry == (("age", "p", "30", "31"),)
    assert "cannot be brought along" in report.refusal()
    steady = _csv(tmp_path, "s.csv", ["pid", "age", "t", "v"],
                  [["p", "", "a", "1"], ["p", "30", "b", "2"]])
    out = long_to_wide(csv_path=steady, id_cols=["pid"], names_from="t",
                       values_from="v", carry_cols=["age"], verbose=False)
    assert read_rows(out)[0]["age"] == "30", "a blank defers to the other rows"


def test_long_to_wide_to_long_is_an_identity(long, tmp_path):
    wide_out = long_to_wide(csv_path=long, id_cols=["pid"], names_from="cond",
                            values_from="score", carry_cols=["group"], verbose=False)
    back = wide_to_long(csv_path=wide_out, value_cols=["B", "A", "C"],
                        names_to="cond", values_to="score", drop_blank=True,
                        out_csv=tmp_path / "back.csv", verbose=False)

    want = sorted((r["pid"], r["group"], r["cond"], r["score"]) for r in read_rows(long))
    got = sorted((r["pid"], r["group"], r["cond"], r["score"]) for r in read_rows(back))
    assert got == want


def test_pivot_argument_mistakes_are_named(long):
    with pytest.raises(ValueError, match="cannot be both"):
        plan_long_to_wide(csv_path=long, id_cols=["pid"], names_from="pid",
                          values_from="score")
    with pytest.raises(ValueError, match="on_duplicate"):
        plan_long_to_wide(csv_path=long, id_cols=["pid"], names_from="cond",
                          values_from="score", on_duplicate="whatever")
    with pytest.raises(ValueError, match="at least one column"):
        plan_long_to_wide(csv_path=long, id_cols=[], names_from="cond",
                          values_from="score")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [merge_spreadsheets, stack_spreadsheets,
                                wide_to_long, long_to_wide])
def test_every_declared_provenance_key_is_a_real_parameter(fn):
    import inspect
    params = set(inspect.signature(fn).parameters)
    declared = fn.__provenance__
    for key in (*declared["binding"], *declared["grain"], *declared["outputs"]):
        assert key in params, (fn.__name__, key)


def test_the_module_imports_without_pandas():
    import subprocess
    import sys
    code = ("import sys; import taters.helpers.reshape; "
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "False"


def test_the_command_line_runs_a_merge(trials, participants, tmp_path):
    from taters.helpers.reshape import main
    out = tmp_path / "cli.csv"
    rc = main(["merge", "--left-csv", str(trials), "--right-csv", str(participants),
               "--on", "pid", "--out-csv", str(out), "--verbose", "false"])
    assert rc in (0, None) and out.is_file()
    assert os.path.getsize(out) > 0
