"""Tests for taters.audio.split_wav_by_speaker.

The unit of interest here is *duration*: given a transcript and a WAV, does
each speaker's file contain exactly the audio it should? Durations are cheap to
compute and catch the mistakes that matter (dropped segments, double-counted
padding, merged turns that should not have been).
"""

import csv
from pathlib import Path

import pytest

pydub = pytest.importorskip("pydub", reason="split_wav_by_speaker requires pydub")

from pydub import AudioSegment  # noqa: E402

from taters.audio.split_wav_by_speaker import (  # noqa: E402
    _sanitize_speaker,
    make_speaker_wavs_from_csv,
)

pytestmark = pytest.mark.needs_ffmpeg


def write_transcript(path: Path, rows: list[tuple], unit: str = "ms") -> Path:
    """rows are (start, end, speaker) in `unit`."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time", "speaker", "text"])
        for start, end, speaker in rows:
            w.writerow([start, end, speaker, "some words"])
    return path


def seconds(path) -> float:
    return len(AudioSegment.from_file(path)) / 1000.0


# --- speaker labels ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Speaker 0", "Speaker_0"),
        ("SPEAKER_1", "SPEAKER_1"),
        ("a/b", "a_b"),
        ("wei?rd*chars", "wei_rd_chars"),
        ("", "SPEAKER_0"),
    ],
)
def test_sanitize_speaker(raw, expected):
    assert _sanitize_speaker(raw) == expected


# --- basic splitting --------------------------------------------------------

def test_one_file_per_speaker(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "Speaker 0"),
        (1000, 2000, "Speaker 1"),
        (2000, 3000, "Speaker 0"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert set(out) == {"Speaker 0", "Speaker 1"}
    assert all(Path(p).is_file() for p in out.values())


def test_output_filenames_include_source_stem_and_speaker(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "Speaker 0")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert Path(out["Speaker 0"]).name == "tiny_Speaker 0.wav"


def test_durations_match_the_transcript_spans(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (1000, 2000, "B"),
        (2000, 3000, "A"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert seconds(out["A"]) == pytest.approx(2.0, abs=0.05)   # two 1s turns
    assert seconds(out["B"]) == pytest.approx(1.0, abs=0.05)


def test_speaker_with_no_usable_segments_produces_no_file(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (500, 500, "B"),        # zero-length: skipped entirely
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert set(out) == {"A"}


# --- merging consecutive turns ---------------------------------------------

def test_consecutive_turns_are_merged_by_default(tiny_wav, tmp_path):
    """
    Two back-to-back rows from the same speaker with a gap between them become
    one span *including* the gap — 0–1000 and 2000–3000 becomes 0–3000.
    """
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (2000, 3000, "A"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert seconds(out["A"]) == pytest.approx(3.0, abs=0.05)


def test_merging_can_be_turned_off(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (2000, 3000, "A"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0, merge_consecutive=False,
    )
    # Only the two spoken seconds, with the silent gap dropped.
    assert seconds(out["A"]) == pytest.approx(2.0, abs=0.05)


def test_turns_interrupted_by_another_speaker_are_not_merged(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (1000, 2000, "B"),
        (2000, 3000, "A"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert seconds(out["A"]) == pytest.approx(2.0, abs=0.05)   # not 3.0


# --- padding ----------------------------------------------------------------

def test_silence_ms_pads_both_sides_of_every_clip(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 1000, "A"),
        (1000, 2000, "B"),
        (2000, 3000, "A"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=500,
    )
    # A has two clips, each padded with 0.5 s before and after: 2 + 4*0.5
    assert seconds(out["A"]) == pytest.approx(4.0, abs=0.05)


def test_pre_and_post_silence_can_be_set_separately(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split",
        silence_ms=999, pre_silence_ms=200, post_silence_ms=800,
    )
    assert seconds(out["A"]) == pytest.approx(1.0 + 0.2 + 0.8, abs=0.05)


# --- time units, bounds, filters -------------------------------------------

def test_seconds_time_unit(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1.5, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", time_unit="s", silence_ms=0,
    )
    assert seconds(out["A"]) == pytest.approx(1.5, abs=0.05)


def test_invalid_time_unit_raises(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    with pytest.raises(ValueError, match="time_unit"):
        make_speaker_wavs_from_csv(
            source_wav=tiny_wav, transcript_csv_path=transcript,
            output_dir=tmp_path / "split", time_unit="minutes",
        )


def test_times_past_the_end_of_the_audio_are_clamped(tiny_wav, tmp_path):
    """A transcript that runs long must not produce a longer-than-source clip."""
    transcript = write_transcript(tmp_path / "t.csv", [(0, 99_000, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert seconds(out["A"]) == pytest.approx(3.0, abs=0.05)


def test_backwards_intervals_are_skipped(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (2000, 1000, "A"),      # end before start
        (0, 1000, "B"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert set(out) == {"B"}


def test_unparseable_rows_are_skipped_not_fatal(tiny_wav, tmp_path):
    path = tmp_path / "t.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time", "speaker", "text"])
        w.writerow(["not_a_number", "1000", "A", "x"])
        w.writerow([0, 1000, "B", "y"])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=path,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert set(out) == {"B"}


def test_min_dur_ms_drops_very_short_segments(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [
        (0, 30, "A"),           # 30 ms, below the threshold
        (1000, 2000, "B"),
    ])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0, min_dur_ms=50,
    )
    assert set(out) == {"B"}


# --- output format ----------------------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000, 44100])
def test_output_sample_rate(tiny_wav, tmp_path, sr):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", sr=sr, silence_ms=0,
    )
    assert AudioSegment.from_file(out["A"]).frame_rate == sr


def test_output_is_mono_by_default(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=tmp_path / "split", silence_ms=0,
    )
    assert AudioSegment.from_file(out["A"]).channels == 1


def test_default_output_directory(tiny_wav, tmp_path, sandbox):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript, silence_ms=0
    )
    assert Path(out["A"]).parent == sandbox / "audio_split" / "tiny"


# --- overwrite contract -----------------------------------------------------

def test_existing_speaker_file_is_left_alone_by_default(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out_dir = tmp_path / "split"
    out_dir.mkdir()
    existing = out_dir / "tiny_A.wav"
    existing.write_bytes(b"sentinel")

    out = make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=out_dir, silence_ms=0,
    )
    assert Path(out["A"]) == existing
    assert existing.read_bytes() == b"sentinel"


def test_overwrite_existing_regenerates(tiny_wav, tmp_path):
    transcript = write_transcript(tmp_path / "t.csv", [(0, 1000, "A")])
    out_dir = tmp_path / "split"
    out_dir.mkdir()
    existing = out_dir / "tiny_A.wav"
    existing.write_bytes(b"sentinel")

    make_speaker_wavs_from_csv(
        source_wav=tiny_wav, transcript_csv_path=transcript,
        output_dir=out_dir, silence_ms=0, overwrite_existing=True,
    )
    assert existing.read_bytes() != b"sentinel"
    assert seconds(existing) == pytest.approx(1.0, abs=0.05)
