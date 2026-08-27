"""Tests for taters.text.subtitle_parser.

Parsers are the classic place for tests to earn their keep: the formats are
full of small optional variations (numbered or not, comma or dot, cue ids,
NOTE blocks, multi-line cues) and each one is a chance to silently drop a cue.
"""

import csv
from pathlib import Path

import pytest

from taters.text.subtitle_parser import (
    SubtitleSegment,
    _fmt_ms_srt,
    _fmt_ms_vtt,
    _parse_timestamp,
    convert_subtitles,
    parse_srt,
    parse_subtitles,
    parse_vtt,
    render_to_csv,
    render_to_srt,
    render_to_vtt,
)

SRT_SAMPLE = """\
1
00:00:01,000 --> 00:00:03,500
Hello there.

2
00:00:04,000 --> 00:00:06,250
General Kenobi.
You are a bold one.
"""

VTT_SAMPLE = """\
WEBVTT
Kind: captions

NOTE this is a comment
that spans two lines

cue-1
00:00:01.000 --> 00:00:03.500
Hello there.

00:00:04.000 --> 00:00:06.250 align:start position:10%
General Kenobi.
"""


# --- timestamps -------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_ms",
    [
        ("00:00:00,000", 0),
        ("00:00:01,000", 1_000),
        ("00:00:01.500", 1_500),          # dot form accepted too
        ("00:01:00,000", 60_000),
        ("01:00:00,000", 3_600_000),
        ("01:02:03,004", 3_723_004),
        ("  00:00:02,000  ", 2_000),      # surrounding whitespace tolerated
    ],
)
def test_parse_timestamp(text, expected_ms):
    assert _parse_timestamp(text) == expected_ms


@pytest.mark.parametrize(
    "bad",
    ["00:00:01", "1:2:3,4", "00:60:00,000", "00:00:60,000", "abc", "", "00:00:01,1000"],
)
def test_parse_timestamp_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        _parse_timestamp(bad)


@pytest.mark.parametrize("ms", [0, 1, 999, 1_000, 61_000, 3_723_004, 359_999_999])
def test_timestamp_formatting_round_trips(ms):
    """Format then re-parse must land on the same millisecond."""
    assert _parse_timestamp(_fmt_ms_srt(ms)) == ms
    assert _parse_timestamp(_fmt_ms_vtt(ms)) == ms


def test_srt_and_vtt_formats_differ_only_in_the_separator():
    assert _fmt_ms_srt(3_723_004) == "01:02:03,004"
    assert _fmt_ms_vtt(3_723_004) == "01:02:03.004"


# --- SRT parsing ------------------------------------------------------------

def test_parse_srt_reads_every_cue():
    segs = parse_srt(SRT_SAMPLE)
    assert len(segs) == 2
    assert segs[0] == SubtitleSegment(number=1, start_ms=1_000, end_ms=3_500,
                                      text="Hello there.", name=None)


def test_parse_srt_keeps_multi_line_text_intact():
    segs = parse_srt(SRT_SAMPLE)
    assert segs[1].text == "General Kenobi.\nYou are a bold one."


def test_parse_srt_without_index_numbers():
    segs = parse_srt("00:00:01,000 --> 00:00:02,000\nno index here\n")
    assert len(segs) == 1
    assert segs[0].number is None
    assert segs[0].text == "no index here"


def test_parse_srt_tolerates_crlf_and_extra_blank_lines():
    text = "1\r\n00:00:01,000 --> 00:00:02,000\r\nline\r\n\r\n\r\n"
    segs = parse_srt(text)
    assert len(segs) == 1
    assert segs[0].text == "line"


def test_parse_srt_accepts_dot_separated_milliseconds():
    segs = parse_srt("1\n00:00:01.250 --> 00:00:02.500\nmixed form\n")
    assert (segs[0].start_ms, segs[0].end_ms) == (1_250, 2_500)


def test_parse_srt_on_empty_input_returns_nothing():
    assert parse_srt("") == []
    assert parse_srt("\n\n\n") == []


def test_parse_srt_raises_on_a_missing_timestamp():
    with pytest.raises(ValueError, match="SRT parse error"):
        parse_srt("1\nthis is not a timestamp\nsome text\n")


# --- VTT parsing ------------------------------------------------------------

def test_parse_vtt_skips_header_notes_and_cue_ids():
    segs = parse_vtt(VTT_SAMPLE)
    assert len(segs) == 2
    assert segs[0].start_ms == 1_000
    assert segs[0].text == "Hello there."
    assert segs[0].number is None          # VTT has no block numbers


def test_parse_vtt_drops_cue_settings_after_the_end_time():
    segs = parse_vtt(VTT_SAMPLE)
    assert segs[1].end_ms == 6_250        # "align:start position:10%" ignored


def test_parse_vtt_without_a_header():
    segs = parse_vtt("00:00:01.000 --> 00:00:02.000\nbare cue\n")
    assert len(segs) == 1 and segs[0].text == "bare cue"


def test_parse_vtt_skips_style_blocks():
    text = (
        "WEBVTT\n\n"
        "STYLE\n::cue { color: peachpuff; }\n\n"
        "00:00:01.000 --> 00:00:02.000\nafter style\n"
    )
    segs = parse_vtt(text)
    assert len(segs) == 1 and segs[0].text == "after style"


def test_parse_vtt_raises_on_a_missing_timestamp():
    with pytest.raises(ValueError, match="VTT parse error"):
        parse_vtt("WEBVTT\n\nnot a timestamp\nnor this\n")


# --- file dispatch ----------------------------------------------------------

def test_parse_subtitles_picks_the_parser_from_the_extension(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_SAMPLE, encoding="utf-8")
    vtt = tmp_path / "a.vtt"
    vtt.write_text(VTT_SAMPLE, encoding="utf-8")

    assert len(parse_subtitles(srt)) == 2
    assert len(parse_subtitles(vtt)) == 2
    # SRT is the fallback for unknown extensions, which is common in the wild.
    other = tmp_path / "a.sub"
    other.write_text(SRT_SAMPLE, encoding="utf-8")
    assert len(parse_subtitles(other)) == 2


def test_parse_subtitles_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_subtitles(tmp_path / "nope.srt")


def test_parse_subtitles_honors_an_explicit_encoding(tmp_path):
    p = tmp_path / "latin.srt"
    p.write_bytes("1\n00:00:01,000 --> 00:00:02,000\ncafé\n".encode("latin-1"))
    segs = parse_subtitles(p, encoding="latin-1")
    assert segs[0].text == "café"


# --- rendering --------------------------------------------------------------

def test_render_to_csv_uses_millisecond_columns(tmp_path):
    segs = parse_srt(SRT_SAMPLE)
    out = render_to_csv(segs, tmp_path / "out.csv")
    with Path(out).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["start_time", "end_time", "text"]
    assert rows[0]["start_time"] == "1000"
    assert rows[1]["text"] == "General Kenobi.\nYou are a bold one."


def test_render_to_csv_can_include_a_name_column(tmp_path):
    segs = [SubtitleSegment(number=1, start_ms=0, end_ms=1, text="hi", name="Alice")]
    out = render_to_csv(segs, tmp_path / "out.csv", include_name=True)
    with Path(out).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["name"] == "Alice"


def test_render_to_srt_renumbers_from_one(tmp_path):
    segs = [
        SubtitleSegment(number=99, start_ms=0, end_ms=1_000, text="a"),
        SubtitleSegment(number=None, start_ms=1_000, end_ms=2_000, text="b"),
    ]
    body = Path(render_to_srt(segs, tmp_path / "out.srt")).read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n")


def test_render_to_vtt_writes_the_header(tmp_path):
    segs = parse_srt(SRT_SAMPLE)
    body = Path(render_to_vtt(segs, tmp_path / "out.vtt")).read_text(encoding="utf-8")
    assert body.startswith("WEBVTT\n\n")
    assert "00:00:01.000 --> 00:00:03.500" in body


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_parse_render_parse_is_lossless_for_times_and_text(tmp_path, fmt):
    """
    A round-trip test: whatever we write must read back identically. This is
    the highest-value shape of test for a converter, because it checks the
    reader and the writer against each other rather than against a fixture.
    """
    original = parse_srt(SRT_SAMPLE)
    out = tmp_path / f"round.{fmt}"
    (render_to_srt if fmt == "srt" else render_to_vtt)(original, out)

    reparsed = parse_subtitles(out)
    assert [(s.start_ms, s.end_ms, s.text) for s in reparsed] == [
        (s.start_ms, s.end_ms, s.text) for s in original
    ]


# --- convert_subtitles (the public entry point) -----------------------------

@pytest.mark.parametrize("to,expected_suffix", [("csv", ".csv"), ("srt", ".srt"), ("vtt", ".vtt")])
def test_convert_subtitles_writes_the_requested_format(tmp_path, to, expected_suffix):
    src = tmp_path / "in.srt"
    src.write_text(SRT_SAMPLE, encoding="utf-8")
    out = convert_subtitles(input=src, to=to, output=tmp_path / f"out{expected_suffix}")
    assert Path(out).suffix == expected_suffix
    assert Path(out).stat().st_size > 0


def test_convert_subtitles_default_output_location(tmp_path, sandbox):
    """With no `output`, files land under ./features/subtitles/<stem>.<ext>."""
    src = tmp_path / "session.srt"
    src.write_text(SRT_SAMPLE, encoding="utf-8")
    out = Path(convert_subtitles(input=src, to="csv"))
    assert out == sandbox / "features" / "subtitles" / "session.csv"
    assert out.is_file()


@pytest.mark.parametrize("explicit_output", [False, True])
def test_convert_subtitles_rejects_an_unknown_format(tmp_path, explicit_output):
    """
    Regression: with an explicit `output` path, an unrecognized format used to
    fall through the if/elif chain and silently render VTT into it.
    """
    src = tmp_path / "in.srt"
    src.write_text(SRT_SAMPLE, encoding="utf-8")
    kwargs = {"output": tmp_path / "out.json"} if explicit_output else {}
    with pytest.raises(ValueError, match="Unsupported output format"):
        convert_subtitles(input=src, to="json", **kwargs)


def test_convert_subtitles_keeps_an_existing_output(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(SRT_SAMPLE, encoding="utf-8")
    out_path = tmp_path / "out.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")

    returned = convert_subtitles(input=src, to="csv", output=out_path)

    assert Path(returned) == out_path
    assert out_path.read_text(encoding="utf-8") == "sentinel\n"


def test_convert_subtitles_overwrite_existing_re_renders(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(SRT_SAMPLE, encoding="utf-8")
    out_path = tmp_path / "out.csv"
    out_path.write_text("sentinel\n", encoding="utf-8")

    convert_subtitles(input=src, to="csv", output=out_path, overwrite_existing=True)

    assert out_path.read_text(encoding="utf-8") != "sentinel\n"
    with out_path.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_convert_subtitles_checks_the_input_before_anything_else(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_subtitles(input=tmp_path / "missing.srt", to="csv")
