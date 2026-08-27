"""Tests for taters.audio.convert_to_wav.

Everything here needs ffmpeg, so the whole module carries the `needs_ffmpeg`
marker: `pytestmark` applies a marker to every test in the file. On a machine
without ffmpeg these skip rather than fail.

Note what is being asserted: not "the function returned a path" but "the file
at that path really is 8 kHz stereo 24-bit". Checking the artifact instead of
the return value is what makes a media test worth running.
"""

from pathlib import Path

import pytest

from taters.audio.convert_to_wav import FFmpegNotFoundError, convert_audio_to_wav
from conftest import audio_stream, probe

pytestmark = pytest.mark.needs_ffmpeg


def duration_of(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


# --- happy path -------------------------------------------------------------

def test_converts_a_wav_to_the_requested_format(tiny_wav, tmp_path):
    out = convert_audio_to_wav(tiny_wav, output_path=tmp_path / "out.wav")
    stream = audio_stream(out)
    assert Path(out).is_file()
    assert stream["codec_name"] == "pcm_s16le"      # 16-bit by default
    assert int(stream["sample_rate"]) == 16000      # ASR-friendly default
    assert int(stream["channels"]) == 1             # mono default


@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000])
def test_sample_rate_is_honored(tiny_wav, tmp_path, sample_rate):
    out = convert_audio_to_wav(
        tiny_wav, output_path=tmp_path / "out.wav", sample_rate=sample_rate
    )
    assert int(audio_stream(out)["sample_rate"]) == sample_rate


@pytest.mark.parametrize(
    "bit_depth,codec", [(16, "pcm_s16le"), (24, "pcm_s24le"), (32, "pcm_s32le")]
)
def test_bit_depth_maps_to_the_right_codec(tiny_wav, tmp_path, bit_depth, codec):
    out = convert_audio_to_wav(
        tiny_wav, output_path=tmp_path / "out.wav", bit_depth=bit_depth
    )
    assert audio_stream(out)["codec_name"] == codec


@pytest.mark.parametrize("channels", [1, 2])
def test_channel_count_is_honored(tiny_wav, tmp_path, channels):
    out = convert_audio_to_wav(
        tiny_wav, output_path=tmp_path / "out.wav", channels=channels
    )
    assert int(audio_stream(out)["channels"]) == channels


def test_audio_length_is_preserved(tiny_wav, tmp_path):
    out = convert_audio_to_wav(tiny_wav, output_path=tmp_path / "out.wav")
    assert duration_of(out) == pytest.approx(duration_of(tiny_wav), abs=0.05)


def test_extracts_the_audio_track_from_a_video(tiny_video, tmp_path):
    """A video container in, a plain WAV out — no video stream survives."""
    out = convert_audio_to_wav(tiny_video, output_path=tmp_path / "out.wav")
    kinds = {s["codec_type"] for s in probe(out)["streams"]}
    assert kinds == {"audio"}


# --- output location --------------------------------------------------------

def test_default_output_goes_to_the_audio_folder(tiny_wav, sandbox):
    out = Path(convert_audio_to_wav(tiny_wav))
    assert out == sandbox / "audio" / "tiny.wav"
    assert out.is_file()


def test_output_dir_keeps_the_input_stem(tiny_wav, tmp_path):
    out = Path(convert_audio_to_wav(tiny_wav, output_dir=tmp_path / "elsewhere"))
    assert out == tmp_path / "elsewhere" / "tiny.wav"


def test_output_dir_is_created_if_absent(tiny_wav, tmp_path):
    target = tmp_path / "deep" / "nested" / "dir"
    out = Path(convert_audio_to_wav(tiny_wav, output_dir=target))
    assert out.parent == target and out.is_file()


def test_output_path_and_output_dir_are_mutually_exclusive(tiny_wav, tmp_path):
    with pytest.raises(ValueError, match="at most one"):
        convert_audio_to_wav(
            tiny_wav, output_path=tmp_path / "a.wav", output_dir=tmp_path
        )


# --- validation and error paths ---------------------------------------------

def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_audio_to_wav(tmp_path / "nope.wav", output_path=tmp_path / "o.wav")


@pytest.mark.parametrize("bad_depth", [8, 12, 64, 0, -16])
def test_invalid_bit_depth_raises(tiny_wav, tmp_path, bad_depth):
    with pytest.raises(ValueError, match="bit_depth"):
        convert_audio_to_wav(
            tiny_wav, output_path=tmp_path / "o.wav", bit_depth=bad_depth
        )


@pytest.mark.parametrize("bad_channels", [0, 3, -1])
def test_invalid_channel_count_raises(tiny_wav, tmp_path, bad_channels):
    with pytest.raises(ValueError, match="channels"):
        convert_audio_to_wav(
            tiny_wav, output_path=tmp_path / "o.wav", channels=bad_channels
        )


def test_undecodable_input_raises_runtime_error(tmp_path):
    junk = tmp_path / "not_audio.wav"
    junk.write_bytes(b"this is definitely not a wav file")
    with pytest.raises(RuntimeError):
        convert_audio_to_wav(junk, output_path=tmp_path / "o.wav")


# --- the overwrite contract -------------------------------------------------

def test_existing_output_is_returned_untouched(tiny_wav, tmp_path):
    out_path = tmp_path / "out.wav"
    out_path.write_bytes(b"sentinel")

    returned = convert_audio_to_wav(tiny_wav, output_path=out_path)

    assert Path(returned) == out_path
    assert out_path.read_bytes() == b"sentinel"


def test_overwrite_existing_replaces_the_file(tiny_wav, tmp_path):
    out_path = tmp_path / "out.wav"
    out_path.write_bytes(b"sentinel")

    convert_audio_to_wav(tiny_wav, output_path=out_path, overwrite_existing=True)

    assert out_path.read_bytes() != b"sentinel"
    assert int(audio_stream(out_path)["sample_rate"]) == 16000


def test_ffmpeg_missing_is_its_own_error(monkeypatch, tiny_wav, tmp_path):
    """
    Pretend ffmpeg vanished from PATH: users should get a clear message about
    the missing dependency, not an obscure failure deep in a pipeline.
    """
    import taters.audio.convert_to_wav as mod
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg"):
        convert_audio_to_wav(tiny_wav, output_path=tmp_path / "o.wav")
