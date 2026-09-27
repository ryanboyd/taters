"""
Tests for `taters.video.analyze_visual_dynamics`.

Two halves, and the first one matters more. The **parsers** read loose text that
ffmpeg's filters print on stderr in formats nobody versioned, so they are tested
against captured real output rather than by shelling out -- a future ffmpeg that
changes its wording should fail here with an obvious message instead of quietly
reporting zero cuts in every video anyone ever runs.

The second half builds a video whose structure is known by construction --
five two-second segments -- so the shot count, average shot length and black
span are arithmetic rather than eyeballed.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

from taters.video._ffmpeg_probe import (parse_black_spans, parse_freeze_spans,
                                        parse_scene_scores)
from taters.video.analyze_visual_dynamics import (BOOKKEEPING, MEASURES,
                                                  _shot_lengths,
                                                  analyze_visual_dynamics)

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe are not on PATH")


# ---------------------------------------------------------------------------
# The parsers, against real captured output
# ---------------------------------------------------------------------------

SCDET_REAL = """\
frame:0    pts:0       pts_time:0
lavfi.scd.score=0.000
frame:1    pts:1024    pts_time:0.1
lavfi.scd.score=0.000
frame:20   pts:20480   pts_time:2
lavfi.scd.score=85.547
frame:40   pts:40960   pts_time:4
lavfi.scd.score=43.000
"""

BLACK_REAL = ("frame=    0 fps=0.0 q=-0.0 size=       0kB "
              "[blackdetect @ 0x76dd0c0016c0] "
              "black_start:2 black_end:4 black_duration:2\n")

FREEZE_REAL = """\
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_start: 0
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_duration: 2
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_end: 2
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_start: 6
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_duration: 2
[freezedetect @ 0x59285dcfc0c0] lavfi.freezedetect.freeze_end: 8
"""


def test_scene_scores_carry_their_frame_and_time():
    """
    `metadata=print` puts the frame header and the value on separate lines, so
    a parser that reads them independently loses the pairing and reports every
    score against the wrong moment.
    """
    got = parse_scene_scores(SCDET_REAL)

    assert got == [(0, 0.0, 0.0), (1, 0.1, 0.0),
                   (20, 2.0, 85.547), (40, 4.0, 43.0)]


def test_a_score_without_a_frame_header_is_not_invented():
    """A stray value line belongs to no frame, and guessing which one would put
    a real number at a made-up time."""
    assert parse_scene_scores("lavfi.scd.score=99.0\n") == []


def test_black_spans_come_back_as_start_end_duration():
    assert parse_black_spans(BLACK_REAL) == [(2.0, 4.0, 2.0)]


def test_freeze_spans_pair_their_starts_with_their_ends():
    """freezedetect spreads one interval over three lines, unlike blackdetect."""
    assert parse_freeze_spans(FREEZE_REAL) == [(0.0, 2.0, 2.0), (6.0, 8.0, 2.0)]


def test_a_freeze_running_to_the_end_is_closed_at_the_duration():
    """
    freezedetect announces a start before it knows the end, so a video that
    freezes and stays frozen has a start and no end at all.
    """
    text = "lavfi.freezedetect.freeze_start: 7\n"

    assert parse_freeze_spans(text, duration=10.0) == [(7.0, 10.0, 3.0)]
    assert parse_freeze_spans(text) == [], "no duration means no invented end"


def test_nothing_detected_is_an_empty_list_not_an_error():
    for fn in (parse_scene_scores, parse_black_spans, parse_freeze_spans):
        assert fn("") == []


# ---------------------------------------------------------------------------
# Shot lengths
# ---------------------------------------------------------------------------

def test_shot_lengths_are_the_gaps_between_cuts():
    assert _shot_lengths([2.0, 4.0], duration=6.0) == [2.0, 2.0, 2.0]


def test_a_video_with_no_cuts_is_one_shot_the_length_of_the_file():
    """Not zero shots, and not a missing value. An uncut video has an average
    shot length, and it is the whole thing."""
    assert _shot_lengths([], duration=12.0) == [12.0]


def test_cuts_outside_the_file_are_ignored():
    assert _shot_lengths([-1.0, 3.0, 99.0], duration=6.0) == [3.0, 3.0]


def test_without_a_duration_we_decline_to_guess():
    assert _shot_lengths([2.0], duration=None) == []


# ---------------------------------------------------------------------------
# End to end, against a video whose structure is known by construction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def known_video(tmp_path_factory) -> Path:
    """
    Five two-second segments, chosen to differ in *luminance* rather than only
    in hue: `scdet` scores brightness, so a red-to-green cut at matched
    brightness scores zero and the fixture would be testing nothing.

    white | black | testsrc | gray | smptebars
    -> 4 cuts at 2/4/6/8s, five 2.0s shots, one 2s black span, and the
       animated testsrc segment is the only one that is not frozen.
    """
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg is not on PATH")
    d = tmp_path_factory.mktemp("known_video")
    sources = [("white", "color=c=white:s=320x240:d=2:r=10"),
               ("black", "color=c=black:s=320x240:d=2:r=10"),
               ("test", "testsrc=s=320x240:d=2:r=10"),
               ("gray", "color=c=gray:s=320x240:d=2:r=10"),
               ("bars", "smptebars=s=320x240:d=2:r=10")]
    for name, src in sources:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src,
                        "-pix_fmt", "yuv420p", str(d / f"{name}.mp4")], check=True)
    listing = d / "list.txt"
    listing.write_text("".join(f"file '{d / (n + '.mp4')}'\n" for n, _ in sources),
                       encoding="utf-8")
    out = d / "known.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(out)], check=True)
    return out


def _row(video: Path, out: Path, **kw) -> dict:
    p = analyze_visual_dynamics(video_path=video, out_csv=out,
                                overwrite_existing=True, verbose=False, **kw)
    with Path(p).open(encoding="utf-8-sig", newline="") as fh:
        return next(iter(csv.DictReader(fh)))


@needs_ffmpeg
def test_the_known_video_measures_exactly_what_was_built(known_video, tmp_path):
    row = _row(known_video, tmp_path / "out.csv")

    assert int(row["vid_cut_count"]) == 4
    assert float(row["vid_shot_len_mean"]) == pytest.approx(2.0, abs=0.01)
    assert float(row["vid_shot_len_sd"]) == pytest.approx(0.0, abs=0.01)
    assert float(row["vid_duration_s"]) == pytest.approx(10.0, abs=0.05)
    assert float(row["vid_cuts_per_min"]) == pytest.approx(24.0, abs=0.5)


@needs_ffmpeg
def test_the_black_segment_is_found_and_measured(known_video, tmp_path):
    """One two-second black segment in a ten-second video."""
    row = _row(known_video, tmp_path / "out.csv")

    assert int(row["vid_black_count"]) == 1
    assert float(row["vid_black_total_s"]) == pytest.approx(2.0, abs=0.1)
    assert float(row["vid_black_prop"]) == pytest.approx(0.2, abs=0.02)


@needs_ffmpeg
def test_the_animated_segment_is_the_one_that_is_not_frozen(known_video, tmp_path):
    """
    Four of the five segments are a static color or pattern; `testsrc` moves.
    A freeze detector that flagged all five would be measuring nothing.
    """
    row = _row(known_video, tmp_path / "out.csv")

    assert float(row["vid_freeze_prop"]) == pytest.approx(0.8, abs=0.05)


@needs_ffmpeg
def test_raising_the_cut_threshold_finds_fewer_cuts(known_video, tmp_path):
    """The threshold is the one knob that matters, so it has to do something."""
    loose = _row(known_video, tmp_path / "a.csv", cut_threshold=10.0)
    strict = _row(known_video, tmp_path / "b.csv", cut_threshold=80.0)

    assert int(loose["vid_cut_count"]) > int(strict["vid_cut_count"])


@needs_ffmpeg
def test_every_declared_column_is_actually_written(known_video, tmp_path):
    """The registry and the file cannot be allowed to drift."""
    row = _row(known_video, tmp_path / "out.csv")

    for col in list(MEASURES) + list(BOOKKEEPING):
        assert col in row, col


@needs_ffmpeg
def test_the_format_facts_are_bookkeeping_not_measures(known_video, tmp_path):
    """
    Frame width is a fact about how you filmed, not about what you filmed. A
    model that found your outcome correlates with it has found your collection
    procedure, so these are declared bookkeeping and the statistics stage keeps
    them out of the feature sets.
    """
    assert "vid_width" in BOOKKEEPING and "vid_fps" in BOOKKEEPING
    assert not set(MEASURES) & set(BOOKKEEPING)
    declared = analyze_visual_dynamics.__provenance__["bookkeeping"]
    assert set(BOOKKEEPING) == set(declared)


@needs_ffmpeg
def test_an_existing_output_is_not_recomputed(known_video, tmp_path):
    out = tmp_path / "out.csv"
    analyze_visual_dynamics(video_path=known_video, out_csv=out, verbose=False)
    out.write_text("sentinel", encoding="utf-8")
    analyze_visual_dynamics(video_path=known_video, out_csv=out, verbose=False)

    assert out.read_text(encoding="utf-8") == "sentinel"


def test_a_missing_video_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_visual_dynamics(video_path=tmp_path / "nope.mp4",
                                out_csv=tmp_path / "out.csv")
