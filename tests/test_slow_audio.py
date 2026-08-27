"""End-to-end audio tests against real media and real models.

Everything here is marked `slow`, so `pytest` skips it and `pytest -m slow`
runs it. These need:

  * files in ./test_vids (merged audio+video, or at least audio)
  * the diarization extras installed
  * ideally a CUDA GPU — on CPU these take a while

They answer a question the fast tests cannot: does the whole chain still work
on real speech? Assertions stay structural (columns, ordering, alignment,
plausible ranges) rather than exact values, because ASR output legitimately
changes between model versions and a test that pins transcript text would fail
for no good reason.
"""

import csv
from pathlib import Path

import pytest

from taters import Taters

pytestmark = [pytest.mark.slow, pytest.mark.needs_ffmpeg, pytest.mark.needs_media]


def read_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def diarization(real_audio_clip, tmp_path_factory):
    """
    Diarize the shared clip once and reuse the result across this module.

    `scope="module"` matters here: diarization is the expensive step, and
    several tests want to look at its output.
    """
    # Check the heavy dependencies up front so a machine without them reports a
    # clean skip instead of a subprocess failure buried in a stack trace.
    pytest.importorskip("faster_whisper", reason="diarization needs the whisper stack")
    pytest.importorskip("nemo", reason="install the extras: pip install 'taters[diarization]'")

    out_dir = tmp_path_factory.mktemp("diarized")
    return Taters().audio.diarize_with_thirdparty(
        audio_path=real_audio_clip,
        out_dir=out_dir,
        whisper_model="tiny.en",     # smallest model that still produces sane turns
        language="en",
        device="auto",
        overwrite_existing=True,
    )


# --- diarization ------------------------------------------------------------

def test_diarization_produces_the_three_artifacts(diarization):
    assert set(diarization.raw_files) >= {"csv", "txt", "srt"}
    for kind, path in diarization.raw_files.items():
        assert Path(path).is_file(), f"{kind} missing"
        assert Path(path).stat().st_size > 0, f"{kind} is empty"


def test_transcript_csv_has_the_expected_schema(diarization):
    rows = read_rows(diarization.raw_files["csv"])
    assert rows, "diarization produced no utterances"
    assert set(rows[0]) >= {"start_time", "end_time", "speaker", "text"}


def test_transcript_times_are_ordered_and_within_the_clip(diarization):
    rows = read_rows(diarization.raw_files["csv"])
    starts = [float(r["start_time"]) for r in rows]
    ends = [float(r["end_time"]) for r in rows]

    assert starts == sorted(starts), "utterances are not in chronological order"
    assert all(e > s for s, e in zip(starts, ends)), "an utterance ends before it starts"
    # Times are milliseconds; the clip is 30 s, allow a little slack at the tail.
    assert max(ends) <= 31_000


def test_transcript_has_text_and_speaker_labels(diarization):
    rows = read_rows(diarization.raw_files["csv"])
    assert all(r["speaker"].strip() for r in rows)
    assert any(r["text"].strip() for r in rows)


def test_temporary_folders_are_cleaned_up(diarization):
    leftovers = list(Path(diarization.work_dir).glob("temp_outputs*"))
    assert leftovers == [], f"temp folders left behind: {leftovers}"


def test_the_copied_input_wav_is_removed(diarization):
    """The wrapper copies the input next to its outputs; it should tidy up."""
    strays = list(Path(diarization.work_dir).glob("*.wav"))
    assert strays == [], f"left a working copy behind: {strays}"


# --- per-speaker splitting on real turns ------------------------------------

def test_split_by_speaker_on_a_real_transcript(real_audio_clip, diarization, tmp_path):
    out = Taters().audio.split_wav_by_speaker(
        source_wav=real_audio_clip,
        transcript_csv_path=diarization.raw_files["csv"],
        output_dir=tmp_path / "split",
        time_unit="ms",
        silence_ms=0,
        overwrite_existing=True,
    )
    assert out, "no speaker files were produced"
    for label, path in out.items():
        assert Path(path).is_file() and Path(path).stat().st_size > 0, label


# --- Whisper embeddings -----------------------------------------------------

def test_whisper_embeddings_align_with_the_transcript(real_audio_clip, diarization, tmp_path):
    pytest.importorskip("faster_whisper", reason="needs the whisper stack")

    out = Taters().audio.extract_whisper_embeddings(
        source_wav=real_audio_clip,
        transcript_csv=diarization.raw_files["csv"],
        output_dir=tmp_path / "emb",
        model_name="tiny.en",
        device="auto",
        time_unit="ms",
        overwrite_existing=True,
    )
    rows = read_rows(out)
    assert rows, "no embedding rows written"

    header = list(rows[0])
    assert header[:3] == ["start_time", "end_time", "speaker"]

    dims = [c for c in header if c.startswith("e") and c[1:].isdigit()]
    assert len(dims) >= 128, f"suspiciously small embedding: {len(dims)} dims"

    # One row per transcript segment, and every vector fully populated.
    assert len(rows) == len(read_rows(diarization.raw_files["csv"]))
    assert all(rows[0][d] not in ("", None) for d in dims)


def test_whisper_embeddings_without_a_transcript(real_audio_clip, tmp_path):
    """General-audio mode: fixed windows, mean-pooled to a single row."""
    pytest.importorskip("faster_whisper", reason="needs the whisper stack")

    out = Taters().audio.extract_whisper_embeddings(
        source_wav=real_audio_clip,
        output_dir=tmp_path / "emb2",
        strategy="nonsilent",
        aggregate="mean",
        model_name="tiny.en",
        device="auto",
        overwrite_existing=True,
    )
    rows = read_rows(out)
    assert len(rows) == 1, "aggregate='mean' should collapse to one row"


# --- vocal acoustics --------------------------------------------------------

def test_acoustics_summary_on_real_audio(real_audio_clip, tmp_path):
    pytest.importorskip("parselmouth", reason="needs praat-parselmouth")

    res = Taters().audio.analyze_vocal_acoustics(
        wav_path=real_audio_clip,
        out_dir=tmp_path / "acoustics",
        mode="simple",
        include_framewise=True,
        overwrite_existing=True,
    )
    summary = read_rows(res["summary_csv"])
    assert len(summary) == 1

    columns = set(summary[0])
    assert {"f0_mean", "f0_std", "f1_mean", "loudness_mean", "hnr_mean"} <= columns

    f0 = float(summary[0]["f0_mean"])
    assert 50.0 < f0 < 500.0, f"implausible mean f0 for speech: {f0}"

    framewise = read_rows(res["framewise_csv"])
    assert len(framewise) > 100, "framewise track is suspiciously short"


def test_acoustics_per_speaker_from_a_transcript(real_audio_clip, diarization, tmp_path):
    pytest.importorskip("parselmouth", reason="needs praat-parselmouth")

    res = Taters().audio.analyze_vocal_acoustics(
        wav_path=real_audio_clip,
        transcript_csv=diarization.raw_files["csv"],
        time_unit="ms",
        group_by=["speaker"],
        out_dir=tmp_path / "acoustics_by_speaker",
        include_framewise=False,
        overwrite_existing=True,
    )
    rows = read_rows(res["summary_csv"])
    speakers = {r["speaker"] for r in read_rows(diarization.raw_files["csv"])}
    assert len(rows) == len(speakers), "expected one summary row per speaker"


# --- audio extraction from a real container ---------------------------------

def test_extract_wavs_from_a_real_video(real_media_with_both_streams, tmp_path):
    outs = Taters().audio.extract_wavs_from_video(
        input_path=real_media_with_both_streams,
        output_dir=tmp_path / "audio",
        sample_rate=16000,
    )
    assert outs
    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in outs)
