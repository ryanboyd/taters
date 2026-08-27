"""Tests for taters.audio.extract_wav_from_video.

This is the multi-track path: Zoom/OBS recordings often carry one audio stream
per participant, and the whole point of this function is to keep them apart and
name them recognizably.
"""

from pathlib import Path

import pytest

from taters.audio.extract_wav_from_video import (
    _build_wav_name,
    _safe_slug,
    split_audio_streams_to_wav,
)
from conftest import audio_stream, probe

pytestmark = pytest.mark.needs_ffmpeg


# --- name construction (pure functions, no ffmpeg needed) -------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("English", "english"),
        ("Speaker One", "speaker-one"),
        ("  spaced  ", "spaced"),
        ("weird///chars", "weird-chars"),
        ("a--b", "a-b"),
        (None, ""),
        ("", ""),
    ],
)
def test_safe_slug(raw, expected):
    assert _safe_slug(raw) == expected


def test_wav_name_includes_index_language_and_title():
    assert _build_wav_name("session", 1, "eng", "Host Mic") == "session_a1_eng_host-mic.wav"


def test_wav_name_omits_absent_tags():
    assert _build_wav_name("session", 0, None, None) == "session_a0.wav"
    assert _build_wav_name("session", 2, "spa", None) == "session_a2_spa.wav"


# --- extraction -------------------------------------------------------------

def test_extracts_one_wav_per_audio_stream(tiny_video_two_audio_streams, tmp_path):
    outs = split_audio_streams_to_wav(tiny_video_two_audio_streams, output_dir=tmp_path / "out")
    assert len(outs) == 2
    assert all(Path(p).is_file() for p in outs)


def test_each_output_contains_only_audio(tiny_video_two_audio_streams, tmp_path):
    outs = split_audio_streams_to_wav(tiny_video_two_audio_streams, output_dir=tmp_path / "out")
    for p in outs:
        assert {s["codec_type"] for s in probe(Path(p))["streams"]} == {"audio"}


def test_output_names_follow_the_documented_pattern(tiny_video_two_audio_streams, tmp_path):
    outs = split_audio_streams_to_wav(tiny_video_two_audio_streams, output_dir=tmp_path / "out")
    names = sorted(Path(p).name for p in outs)
    stem = tiny_video_two_audio_streams.stem
    # Language tags were set on the fixture, so they should appear in the names.
    assert names[0].startswith(f"{stem}_a") and names[0].endswith("_eng.wav")
    assert names[1].endswith("_spa.wav")


def test_the_two_tracks_are_actually_different(tiny_video_two_audio_streams, tmp_path):
    """
    Guards against a real hazard in stream mapping: emitting the same track
    twice under two names. The fixture puts a 440 Hz tone on one track and
    880 Hz on the other, so identical bytes would mean a mapping bug.
    """
    outs = sorted(split_audio_streams_to_wav(
        tiny_video_two_audio_streams, output_dir=tmp_path / "out"
    ))
    assert Path(outs[0]).read_bytes() != Path(outs[1]).read_bytes()


def test_single_track_video_yields_one_wav(tiny_video, tmp_path):
    outs = split_audio_streams_to_wav(tiny_video, output_dir=tmp_path / "out")
    assert len(outs) == 1


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_sample_rate_is_honored(tiny_video, tmp_path, sample_rate):
    (out,) = split_audio_streams_to_wav(
        tiny_video, output_dir=tmp_path / "out", sample_rate=sample_rate
    )
    assert int(audio_stream(Path(out))["sample_rate"]) == sample_rate


@pytest.mark.parametrize(
    "bit_depth,codec", [(16, "pcm_s16le"), (24, "pcm_s24le"), (32, "pcm_s32le")]
)
def test_bit_depth_is_honored(tiny_video, tmp_path, bit_depth, codec):
    (out,) = split_audio_streams_to_wav(
        tiny_video, output_dir=tmp_path / "out", bit_depth=bit_depth
    )
    assert audio_stream(Path(out))["codec_name"] == codec


def test_default_output_is_the_audio_folder(tiny_video, sandbox):
    (out,) = split_audio_streams_to_wav(tiny_video)
    assert Path(out).parent == sandbox / "audio"


def test_returns_absolute_paths(tiny_video, tmp_path):
    outs = split_audio_streams_to_wav(tiny_video, output_dir=tmp_path / "out")
    assert all(Path(p).is_absolute() for p in outs)


# --- error paths ------------------------------------------------------------

def test_video_without_audio_raises(tmp_path):
    """
    The exact situation yt-dlp produced: a video-only container. The error
    should say so plainly rather than yielding an empty list.
    """
    import subprocess
    silent = tmp_path / "video_only.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        check=True, capture_output=True,
    )
    with pytest.raises(ValueError, match="No audio streams"):
        split_audio_streams_to_wav(silent, output_dir=tmp_path / "out")


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_audio_streams_to_wav(tmp_path / "nope.mp4", output_dir=tmp_path)


@pytest.mark.parametrize("bad_depth", [8, 20, 64])
def test_invalid_bit_depth_raises(tiny_video, tmp_path, bad_depth):
    with pytest.raises(ValueError, match="bit_depth"):
        split_audio_streams_to_wav(
            tiny_video, output_dir=tmp_path / "out", bit_depth=bad_depth
        )


def test_existing_output_is_kept_and_returned(tiny_video, tmp_path):
    """Same contract as every other writer: keep the file, hand back the path."""
    out_dir = tmp_path / "out"
    (first,) = split_audio_streams_to_wav(tiny_video, output_dir=out_dir)
    Path(first).write_bytes(b"sentinel")

    (again,) = split_audio_streams_to_wav(tiny_video, output_dir=out_dir)

    assert again == first
    assert Path(first).read_bytes() == b"sentinel"


def test_overwrite_existing_re_extracts(tiny_video, tmp_path):
    out_dir = tmp_path / "out"
    (first,) = split_audio_streams_to_wav(tiny_video, output_dir=out_dir)
    Path(first).write_bytes(b"sentinel")

    (again,) = split_audio_streams_to_wav(
        tiny_video, output_dir=out_dir, overwrite_existing=True
    )
    assert Path(again).read_bytes() != b"sentinel"
    assert int(audio_stream(Path(again))["sample_rate"]) == 48000


def test_a_partially_extracted_folder_is_completed(tiny_video_two_audio_streams, tmp_path):
    """
    Skipping an existing stream must not skip the *other* streams: a run that
    died halfway should be finishable by re-running it.
    """
    out_dir = tmp_path / "out"
    outs = sorted(split_audio_streams_to_wav(
        tiny_video_two_audio_streams, output_dir=out_dir
    ))
    Path(outs[1]).unlink()          # pretend the second stream never finished

    again = sorted(split_audio_streams_to_wav(
        tiny_video_two_audio_streams, output_dir=out_dir
    ))
    assert again == outs
    assert all(Path(p).is_file() for p in again)


def test_the_deprecated_overwrite_alias_still_works(tiny_video, tmp_path):
    out_dir = tmp_path / "out"
    (first,) = split_audio_streams_to_wav(tiny_video, output_dir=out_dir)
    Path(first).write_bytes(b"sentinel")

    with pytest.warns(DeprecationWarning, match="overwrite_existing"):
        split_audio_streams_to_wav(tiny_video, output_dir=out_dir, overwrite=True)

    assert Path(first).read_bytes() != b"sentinel"
