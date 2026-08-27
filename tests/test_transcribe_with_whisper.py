"""Tests for the single-speaker faster-whisper transcriber.

Almost everything here runs without loading a model. That is deliberate: the
interesting logic in this module is *not* the transcription — Whisper's own
accuracy is not ours to test — but the plumbing around it. Does it write the
same CSV schema the diarizer writes? Does it convert seconds to milliseconds?
Does it refuse to clobber an existing transcript? Does it pick a compute type
that actually works on CPU?

The technique used throughout is a **fake model**: `monkeypatch` replaces the
model loader with a stand-in object whose `.transcribe()` returns canned
segments. The real function then runs end to end, exercising every line of our
code, in milliseconds. One `slow` test at the bottom runs a genuine `tiny.en`
model against real speech to confirm the fake is not lying about the shape of
what faster-whisper returns.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from taters import Taters
from taters.audio import transcribe_with_whisper as twh
from taters.audio.transcribe_with_whisper import (
    TranscriptionOutputFiles,
    _resolve_compute_type,
    _resolve_device,
    transcribe_with_whisper,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeSegment(SimpleNamespace):
    """Stands in for a faster-whisper segment: `.start`, `.end` (seconds), `.text`."""


class FakeModel:
    """
    A stand-in for `faster_whisper.WhisperModel`.

    Records the kwargs it was called with so tests can assert that options
    (language, beam_size, vad_filter, ...) are actually forwarded rather than
    silently dropped.
    """

    def __init__(self, segments, duration=10.0):
        self._segments = segments
        self._duration = duration
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        info = SimpleNamespace(duration=self._duration, language="en")
        return iter(self._segments), info


@pytest.fixture
def fake_model(monkeypatch):
    """
    Install a FakeModel in place of the real loader, and hand it to the test.

    Returns a factory so each test can choose its own segments; calling it
    swaps in the model and returns it for later assertions.
    """
    def install(segments, duration=10.0) -> FakeModel:
        model = FakeModel(segments, duration=duration)
        monkeypatch.setattr(twh, "_get_model", lambda *a, **k: model)
        return model
    return install


@pytest.fixture
def wav(tmp_path) -> Path:
    """
    A file that merely has to exist.

    The transcriber never opens it itself — it hands the path to faster-whisper,
    which the fake model replaces — so real audio would only slow these down.
    """
    p = tmp_path / "session.wav"
    p.write_bytes(b"RIFF....WAVEfmt ")
    return p


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given", ["cuda", "cpu"])
def test_explicit_device_is_returned_unchanged(given):
    assert _resolve_device(given) == given


@pytest.mark.parametrize("given", ["CUDA", "  cpu  "])
def test_device_is_normalised(given):
    assert _resolve_device(given) == given.strip().lower()


@pytest.mark.parametrize("given", ["auto", "", None])
def test_auto_device_resolves_to_something_concrete(given):
    """Whatever this machine has, "auto" must never leak through to CTranslate2."""
    assert _resolve_device(given) in {"cuda", "cpu"}


def test_auto_device_falls_back_to_cpu_when_ctranslate2_is_unavailable(monkeypatch):
    """A machine without a working CTranslate2 CUDA build must still transcribe."""
    import builtins
    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "ctranslate2":
            raise ImportError("no ctranslate2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert _resolve_device("auto") == "cpu"


@pytest.mark.parametrize("compute_type,device,expected", [
    (None, "cuda", "float16"),      # the only sensible GPU default
    (None, "cpu", "int8"),          # float16 is unsupported on CPU
    ("float32", "cpu", "float32"),  # explicit always wins
    ("int8_float16", "cuda", "int8_float16"),
])
def test_compute_type_resolution(compute_type, device, expected):
    assert _resolve_compute_type(compute_type, device) == expected


def test_cpu_default_is_not_float16():
    """
    Regression guard.

    `extract_whisper_embeddings` defaults to float16 unconditionally, which
    CTranslate2 silently downgrades to float32 on CPU. This module is meant to
    be the one that works on a plain CPU-only install, so it must not inherit
    that default.
    """
    assert _resolve_compute_type(None, "cpu") != "float16"


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------

def read_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_csv_has_the_diarizer_schema(wav, fake_model):
    """
    The whole point of this module: its CSV must be interchangeable with the
    diarizer's, or none of the downstream steps can consume it.
    """
    fake_model([FakeSegment(start=0.0, end=1.5, text="hello there")])
    outs = transcribe_with_whisper(wav, verbose=False)

    with Path(outs.raw_files["csv"]).open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert header == ["start_time", "end_time", "speaker", "text"]


def test_times_are_written_in_milliseconds(wav, fake_model):
    """faster-whisper reports seconds; every consumer here defaults to ms."""
    fake_model([FakeSegment(start=1.25, end=2.5, text="a phrase")])
    outs = transcribe_with_whisper(wav, verbose=False)

    row = read_rows(outs.raw_files["csv"])[0]
    assert float(row["start_time"]) == 1250.0
    assert float(row["end_time"]) == 2500.0


def test_every_row_carries_the_speaker_label(wav, fake_model):
    fake_model([
        FakeSegment(start=0.0, end=1.0, text="one"),
        FakeSegment(start=1.0, end=2.0, text="two"),
    ])
    outs = transcribe_with_whisper(wav, speaker_label="Narrator", verbose=False)

    rows = read_rows(outs.raw_files["csv"])
    assert [r["speaker"] for r in rows] == ["Narrator", "Narrator"]


def test_default_speaker_label_matches_the_diarizer(wav, fake_model):
    """
    The vendored diarizer emits "Speaker 0", "Speaker 1", ... Using the same
    spelling means features grouped by speaker line up across the two
    pipelines instead of splitting into two near-identical categories.
    """
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert read_rows(outs.raw_files["csv"])[0]["speaker"] == "Speaker 0"


def test_blank_segments_are_dropped(wav, fake_model):
    """Whisper emits empty and whitespace-only segments; they are not utterances."""
    fake_model([
        FakeSegment(start=0.0, end=1.0, text="real text"),
        FakeSegment(start=1.0, end=2.0, text="   "),
        FakeSegment(start=2.0, end=3.0, text=""),
    ])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert len(read_rows(outs.raw_files["csv"])) == 1


def test_segment_text_is_stripped(wav, fake_model):
    """faster-whisper pads segment text with a leading space."""
    fake_model([FakeSegment(start=0.0, end=1.0, text=" padded on both sides ")])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert read_rows(outs.raw_files["csv"])[0]["text"] == "padded on both sides"


# ---------------------------------------------------------------------------
# Timestamp clamping
# ---------------------------------------------------------------------------

def test_end_times_are_clamped_to_the_audio_duration(wav, fake_model):
    """
    Regression test for observed behaviour: on a 15.0 s clip, Whisper reported
    a final segment ending at 16.92 s. `split_wav_by_speaker` and
    `analyze_vocal_acoustics` slice the WAV using these numbers, so an overrun
    silently produces a truncated segment rather than an error.
    """
    fake_model([FakeSegment(start=13.0, end=16.92, text="overrunning")], duration=15.0)
    outs = transcribe_with_whisper(wav, verbose=False)

    row = read_rows(outs.raw_files["csv"])[0]
    assert float(row["end_time"]) == 15000.0


def test_segments_starting_past_the_end_are_dropped(wav, fake_model):
    fake_model([
        FakeSegment(start=0.0, end=5.0, text="inside"),
        FakeSegment(start=11.0, end=12.0, text="entirely past the end"),
    ], duration=10.0)
    outs = transcribe_with_whisper(wav, verbose=False)

    rows = read_rows(outs.raw_files["csv"])
    assert [r["text"] for r in rows] == ["inside"]


def test_zero_length_segments_are_dropped(wav, fake_model):
    fake_model([FakeSegment(start=2.0, end=2.0, text="instantaneous")], duration=10.0)
    outs = transcribe_with_whisper(wav, verbose=False)
    assert read_rows(outs.raw_files["csv"]) == []


def test_unknown_duration_disables_clamping(wav, fake_model):
    """
    If Whisper does not report a duration we have nothing to clamp against.
    Passing the times through unchanged beats inventing a limit.
    """
    fake_model([FakeSegment(start=0.0, end=99.0, text="long")], duration=0.0)
    outs = transcribe_with_whisper(wav, verbose=False)
    assert float(read_rows(outs.raw_files["csv"])[0]["end_time"]) == 99000.0


# ---------------------------------------------------------------------------
# Sidecar files
# ---------------------------------------------------------------------------

def test_srt_and_txt_are_written_by_default(wav, fake_model):
    fake_model([FakeSegment(start=0.0, end=1.0, text="hello")])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert set(outs.raw_files) == {"csv", "srt", "txt"}
    assert all(p.is_file() for p in outs.raw_files.values())


def test_sidecars_can_be_turned_off(wav, fake_model):
    fake_model([FakeSegment(start=0.0, end=1.0, text="hello")])
    outs = transcribe_with_whisper(wav, write_srt=False, write_txt=False, verbose=False)
    assert set(outs.raw_files) == {"csv"}


def test_srt_is_well_formed(wav, fake_model):
    fake_model([
        FakeSegment(start=0.0, end=1.5, text="first cue"),
        FakeSegment(start=1.5, end=3.0, text="second cue"),
    ])
    outs = transcribe_with_whisper(wav, verbose=False)

    body = Path(outs.raw_files["srt"]).read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,500\nfirst cue")
    assert "2\n00:00:01,500 --> 00:00:03,000\nsecond cue" in body


def test_txt_is_one_labelled_block(wav, fake_model):
    """Matches how the diarizer renders a stretch of single-speaker audio."""
    fake_model([
        FakeSegment(start=0.0, end=1.0, text="one"),
        FakeSegment(start=1.0, end=2.0, text="two"),
    ])
    outs = transcribe_with_whisper(wav, speaker_label="Speaker 0", verbose=False)

    body = Path(outs.raw_files["txt"]).read_text(encoding="utf-8-sig")
    assert body.strip() == "Speaker 0: one two"


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

def test_outputs_land_in_a_per_file_subdirectory(wav, fake_model, tmp_path):
    """Mirrors the diarizer's `<out_dir>/<stem>/<stem>.csv` layout."""
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = transcribe_with_whisper(wav, out_dir=tmp_path / "out", verbose=False)

    assert outs.work_dir == (tmp_path / "out" / "session").resolve()
    assert outs.raw_files["csv"].name == "session.csv"


def test_default_out_dir_is_transcripts_under_cwd(wav, fake_model, sandbox):
    """`sandbox` has already chdir'd us into a tmp dir, so this is safe."""
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert outs.work_dir == (sandbox / "transcripts" / "session").resolve()


def test_metadata_is_reported(wav, fake_model):
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")], duration=12.5)
    outs = transcribe_with_whisper(wav, verbose=False)
    assert isinstance(outs, TranscriptionOutputFiles)
    assert outs.language == "en"
    assert outs.duration == 12.5


# ---------------------------------------------------------------------------
# Option forwarding
# ---------------------------------------------------------------------------

def test_decoding_options_reach_the_model(wav, fake_model):
    """A dropped option is invisible at runtime — the transcript just gets worse."""
    model = fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    transcribe_with_whisper(
        wav, language="fr", beam_size=1, vad_filter=False,
        initial_prompt="Taters, LIWC", verbose=False,
    )

    call = model.calls[0]
    assert call["language"] == "fr"
    assert call["beam_size"] == 1
    assert call["vad_filter"] is False
    assert call["initial_prompt"] == "Taters, LIWC"


def test_model_configuration_reaches_the_loader(wav, monkeypatch):
    seen = {}

    def spy(model_name, device, compute_type):
        seen.update(model_name=model_name, device=device, compute_type=compute_type)
        return FakeModel([FakeSegment(start=0.0, end=1.0, text="hi")])

    monkeypatch.setattr(twh, "_get_model", spy)
    transcribe_with_whisper(wav, whisper_model="small.en", device="cpu", verbose=False)

    assert seen == {"model_name": "small.en", "device": "cpu", "compute_type": "int8"}


# ---------------------------------------------------------------------------
# Overwrite contract
# ---------------------------------------------------------------------------

def test_existing_transcript_is_not_recomputed(wav, monkeypatch, tmp_path):
    """
    The short-circuit must happen *before* the model loads.

    Loading a model is the expensive part, so a short-circuit that still pays
    for it has missed the point. Making the loader raise proves it is never
    reached.
    """
    work_dir = tmp_path / "out" / "session"
    work_dir.mkdir(parents=True)
    (work_dir / "session.csv").write_text("start_time,end_time,speaker,text\n", encoding="utf-8")

    def explode(*a, **k):
        raise AssertionError("model was loaded despite an existing transcript")

    monkeypatch.setattr(twh, "_get_model", explode)
    outs = transcribe_with_whisper(wav, out_dir=tmp_path / "out", verbose=False)
    assert outs.raw_files["csv"] == work_dir / "session.csv"


def test_existing_sidecars_are_reported_on_short_circuit(wav, monkeypatch, tmp_path):
    work_dir = tmp_path / "out" / "session"
    work_dir.mkdir(parents=True)
    for ext in ("csv", "srt", "txt"):
        (work_dir / f"session.{ext}").write_text("x", encoding="utf-8")

    monkeypatch.setattr(twh, "_get_model", lambda *a, **k: pytest.fail("should not load"))
    outs = transcribe_with_whisper(wav, out_dir=tmp_path / "out", verbose=False)
    assert set(outs.raw_files) == {"csv", "srt", "txt"}


def test_overwrite_existing_forces_a_rerun(wav, fake_model, tmp_path):
    work_dir = tmp_path / "out" / "session"
    work_dir.mkdir(parents=True)
    (work_dir / "session.csv").write_text("stale\n", encoding="utf-8")

    fake_model([FakeSegment(start=0.0, end=1.0, text="fresh text")])
    outs = transcribe_with_whisper(
        wav, out_dir=tmp_path / "out", overwrite_existing=True, verbose=False,
    )
    assert read_rows(outs.raw_files["csv"])[0]["text"] == "fresh text"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_missing_audio_raises_before_loading_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(twh, "_get_model", lambda *a, **k: pytest.fail("should not load"))
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_with_whisper(tmp_path / "nope.wav", verbose=False)


def test_a_transcript_with_no_speech_still_writes_a_valid_csv(wav, fake_model):
    """Silence is a legitimate result, not an error. Downstream must not crash."""
    fake_model([])
    outs = transcribe_with_whisper(wav, verbose=False)
    assert outs.raw_files["csv"].is_file()
    assert read_rows(outs.raw_files["csv"]) == []


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

def test_model_is_cached_across_calls(monkeypatch):
    """
    The pipeline runner fans item steps across threads. Without a cache, an
    8-worker run loads the same model eight times.
    """
    loads = []

    class Counting:
        def __init__(self, name, device=None, compute_type=None):
            loads.append(name)

    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=Counting))

    first = twh._get_model("tiny", "cpu", "int8")
    second = twh._get_model("tiny", "cpu", "int8")
    assert first is second
    assert loads == ["tiny"]


def test_different_configurations_get_different_models(monkeypatch):
    class Counting:
        def __init__(self, name, device=None, compute_type=None):
            self.name = name

    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=Counting))

    assert twh._get_model("tiny", "cpu", "int8") is not twh._get_model("base", "cpu", "int8")


# ---------------------------------------------------------------------------
# Facade and CLI
# ---------------------------------------------------------------------------

def test_reachable_through_the_facade(wav, fake_model):
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = Taters().audio.transcribe_with_whisper(audio_path=wav, verbose=False)
    assert outs.raw_files["csv"].is_file()


def test_facade_rejects_unknown_parameters(wav):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Taters().audio.transcribe_with_whisper(audio_path=wav, speaker="oops")


def test_cli_defaults_match_the_function_defaults():
    """
    A CLI that drifts from its function is worse than no CLI: the same
    invocation means two different things depending on how you spell it.
    """
    import inspect
    args = twh._build_arg_parser().parse_args(["--audio_path", "x.wav"])
    params = inspect.signature(transcribe_with_whisper).parameters

    for name in ("whisper_model", "language", "compute_type", "beam_size",
                 "vad_filter", "initial_prompt", "speaker_label",
                 "write_srt", "write_txt", "overwrite_existing"):
        assert getattr(args, name) == params[name].default, f"--{name} drifted"


@pytest.mark.parametrize("flag,expected", [
    ("true", True), ("false", False), ("1", True), ("0", False),
])
def test_cli_boolean_flags_accept_explicit_values(flag, expected):
    """Guards the `type=bool` trap, where `--flag false` evaluated to True."""
    args = twh._build_arg_parser().parse_args(
        ["--audio_path", "x.wav", "--overwrite_existing", flag]
    )
    assert args.overwrite_existing is expected


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.needs_ffmpeg
@pytest.mark.needs_media
def test_real_transcription_of_real_speech(real_audio_clip, tmp_path):
    """
    Runs an actual `tiny.en` model over 30 seconds of real speech.

    This is the test that keeps the fakes honest: if faster-whisper changes the
    shape of what `transcribe()` returns, every fast test above would keep
    passing while the module was broken. Assertions stay structural, because
    ASR output legitimately changes between model versions.
    """
    pytest.importorskip("faster_whisper")

    outs = transcribe_with_whisper(
        real_audio_clip,
        out_dir=tmp_path / "transcripts",
        whisper_model="tiny.en",
        device="cpu",
        verbose=False,
    )

    rows = read_rows(outs.raw_files["csv"])
    assert rows, "real speech produced no transcript rows"

    duration_ms = outs.duration * 1000.0
    for r in rows:
        start, end = float(r["start_time"]), float(r["end_time"])
        assert 0 <= start < end <= duration_ms + 1, f"implausible times: {r}"
        assert r["text"].strip()
        assert r["speaker"] == "Speaker 0"

    # Times must be non-overlapping and ordered, or downstream slicing is wrong.
    ends = [float(r["end_time"]) for r in rows]
    starts = [float(r["start_time"]) for r in rows]
    assert starts == sorted(starts)
    assert all(s >= e - 1 for s, e in zip(starts[1:], ends[:-1]))


@pytest.mark.slow
@pytest.mark.needs_ffmpeg
@pytest.mark.needs_media
def test_real_transcript_feeds_a_downstream_consumer(real_audio_clip, tmp_path):
    """
    The drop-in claim, end to end: a transcript produced here must be readable
    by a tool written for the diarizer's output, with no translation layer.
    """
    pytest.importorskip("faster_whisper")

    outs = transcribe_with_whisper(
        real_audio_clip, out_dir=tmp_path / "transcripts",
        whisper_model="tiny.en", device="cpu", verbose=False,
    )
    made = Taters().audio.split_wav_by_speaker(
        source_wav=real_audio_clip,
        transcript_csv_path=str(outs.raw_files["csv"]),
        time_unit="ms",
        output_dir=str(tmp_path / "speakers"),
    )
    assert made
    assert all(Path(p).is_file() for p in made.values())
