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
from csvhelpers import read_rows


# ---------------------------------------------------------------------------
# test doubles
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
        # `_get_model` reports what it actually delivered, not what we asked for:
        # it proves the GPU works before handing it over and falls back to the CPU
        # when it doesn't
        monkeypatch.setattr(twh, "_get_model",
                            lambda name, device, compute, *a, **k: (model, device, compute))
        return model
    return install


def _counting_model(loads: list):
    """A `WhisperModel` stand-in that records every load."""
    class Counting:
        def __init__(self, name, device=None, compute_type=None):
            loads.append(name)
            self.name = name
    return Counting


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
# runtime resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given", ["cuda", "cpu"])
def test_explicit_device_is_returned_unchanged(given):
    assert _resolve_device(given) == given


@pytest.mark.parametrize("given", ["CUDA", "  cpu  "])
def test_device_is_normalized(given):
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
# the output contract
# ---------------------------------------------------------------------------


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
# timestamp clamping
# ---------------------------------------------------------------------------

def test_end_times_are_clamped_to_the_audio_duration(wav, fake_model):
    """
    Regression test for observed behavior: on a 15.0 s clip, Whisper reported
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
# sidecar files
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


def test_txt_is_one_labeled_block(wav, fake_model):
    """Matches how the diarizer renders a stretch of single-speaker audio."""
    fake_model([
        FakeSegment(start=0.0, end=1.0, text="one"),
        FakeSegment(start=1.0, end=2.0, text="two"),
    ])
    outs = transcribe_with_whisper(wav, speaker_label="Speaker 0", verbose=False)

    body = Path(outs.raw_files["txt"]).read_text(encoding="utf-8-sig")
    assert body.strip() == "Speaker 0: one two"


# ---------------------------------------------------------------------------
# output layout
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
# option forwarding
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
        return FakeModel([FakeSegment(start=0.0, end=1.0, text="hi")]), device, compute_type

    monkeypatch.setattr(twh, "_get_model", spy)
    transcribe_with_whisper(wav, whisper_model="small.en", device="cpu", verbose=False)

    assert seen == {"model_name": "small.en", "device": "cpu", "compute_type": "int8"}


# ---------------------------------------------------------------------------
# overwrite contract
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
# failure modes
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
# model cache
# ---------------------------------------------------------------------------

def test_model_is_cached_across_calls(monkeypatch):
    """
    The pipeline runner fans item steps across threads. Without a cache, an
    8-worker run loads the same model eight times.
    """
    loads = []

    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setattr(twh, "_prove_it_works", lambda model: None)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_counting_model(loads)))

    first = twh._get_model("tiny", "cpu", "int8")
    second = twh._get_model("tiny", "cpu", "int8")
    assert first is second
    assert loads == ["tiny"]


def test_different_configurations_get_different_models(monkeypatch):
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setattr(twh, "_prove_it_works", lambda model: None)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_counting_model([])))

    assert twh._get_model("tiny", "cpu", "int8") is not twh._get_model("base", "cpu", "int8")


# ---------------------------------------------------------------------------
# facade and CLI
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
    The parser is derived from the signature now, and a parameter that is
    not on the line is not passed at all -- so the function's own default
    rules and there is nothing left to drift.
    """
    import inspect

    fn, kwargs = twh.CLI.kwargs(["--audio_path", "x.wav"])
    assert fn is transcribe_with_whisper
    params = inspect.signature(transcribe_with_whisper).parameters
    for name in ("whisper_model", "language", "compute_type", "beam_size",
                 "initial_prompt", "speaker_label"):
        assert name not in kwargs, f"--{name} was passed unasked"
    # booleans always get passed (a bare flag means true), so when they're absent
    # the value has to be the function's own default
    for name in ("vad_filter", "write_srt", "write_txt", "overwrite_existing"):
        assert kwargs[name] == params[name].default, f"--{name} drifted"
    assert kwargs["audio_path"] == "x.wav"


@pytest.mark.parametrize("flag,expected", [
    ("true", True), ("false", False), ("1", True), ("0", False),
])
def test_cli_boolean_flags_accept_explicit_values(flag, expected):
    """Guards the `type=bool` trap, where `--flag false` evaluated to True."""
    _fn, kwargs = twh.CLI.kwargs(
        ["--audio_path", "x.wav", "--overwrite_existing", flag]
    )
    assert kwargs["overwrite_existing"] is expected
    # the dashed spelling and the bare flag work too
    _fn, kwargs = twh.CLI.kwargs(["--audio-path", "x.wav", "--overwrite-existing"])
    assert kwargs["overwrite_existing"] is True


# ---------------------------------------------------------------------------
# the real thing
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

    # times have to be non-overlapping and in order, or downstream slicing goes wrong
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


# ---------------------------------------------------------------------------
# saying which device actually ran
#
# somebody reported this: a pipeline that takes 90 seconds here ran for two
# hours on their machine and produced nothing. `device: auto` quietly resolving
# to the CPU was the difference, and there was no way (during the run, or
# afterwards from the manifest) to tell which had happened
# ---------------------------------------------------------------------------

def test_a_cpu_fallback_explains_itself(monkeypatch):
    """
    The reason matters more than the fact. CTranslate2 does not depend on torch
    and does not borrow torch's CUDA libraries, so "but torch sees my GPU" is
    the natural and wrong conclusion to draw unaided.
    """
    from taters.helpers.gpu import resolve_device as _resolve_device_explained

    import sys
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 0))

    device, reason = _resolve_device_explained("auto", backend="ctranslate2")
    assert device == "cpu"
    assert reason and "torch" in reason


def test_choosing_the_cpu_deliberately_needs_no_explanation(monkeypatch):
    from taters.helpers.gpu import resolve_device as _resolve_device_explained

    assert _resolve_device_explained("cpu") == ("cpu", None)


def test_cuda_needs_no_explanation(monkeypatch):
    import sys
    from taters.helpers.gpu import resolve_device as _resolve_device_explained

    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert _resolve_device_explained("auto", backend="ctranslate2") == ("cuda", None)


def test_the_result_records_what_actually_ran(wav, fake_model):
    """
    Recorded on the returned dataclass, which is what the pipeline runner writes
    into the manifest -- so "was this run on the GPU?" is answerable after the
    fact, from the run's own output, months later.
    """
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = transcribe_with_whisper(wav, out_dir=wav.parent / "t",
                                   device="cpu", verbose=False)
    assert outs.device == "cpu"
    assert outs.compute_type == "int8"


def test_progress_is_reported_as_a_position_in_the_recording(wav, fake_model):
    """
    Seconds decoded out of seconds of audio. The denominator is known from the
    first moment -- Whisper reports the duration up front -- whereas the number
    of segments is only known once decoding has finished, which is too late to
    be a progress bar.
    """
    fake_model(
        [FakeSegment(start=0.0, end=30.0, text="one"),
         FakeSegment(start=30.0, end=90.0, text="two")],
        duration=120.0,
    )
    seen = []
    transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                            verbose=False, on_progress=lambda *a: seen.append(a))

    # `announce` sends three arguments and the decoding ticks send four, so the
    # unit is what tells a phase name apart from a position
    counted = [a for a in seen if len(a) == 4 and a[3] == "seconds"]
    assert [(a[0], a[1]) for a in counted] == [(30, 120), (90, 120)]


def test_the_phases_before_any_audio_is_decoded_are_named(wav, fake_model):
    """
    Loading a model and scanning for speech are both silent from the outside and
    both take real time -- the first can be a multi-gigabyte download, and a
    broken CUDA install stalls there. A row naming a file and nothing else does
    not distinguish that from a working run.
    """
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    seen = []
    transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                            whisper_model="tiny", verbose=False,
                            on_progress=lambda *a: seen.append(a))

    messages = [a[2] for a in seen if a[2]]
    assert any("tiny" in m and "cpu" in m for m in messages)
    assert "finding speech" in messages


def test_transcription_still_works_with_nobody_watching(wav, fake_model):
    fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    outs = transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                                   verbose=False, on_progress=None)
    assert outs.raw_files["csv"].is_file()


# ---------------------------------------------------------------------------
# a GPU that builds a model but can't run one
#
# this happened to us, and it's why four files went in and nothing came out:
#
#     RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
#       ... transcribe.py:1400 in encode
#
# `get_cuda_device_count()` only asks the CUDA driver, which is always there.
# CTranslate2 loads cuBLAS lazily at the first *encode*, so the model builds
# fine, gets cached, gets handed to every worker, and then the first one to
# reach an encode dies, while the rest sit on a replica that's never going to
# answer them. one file failed in two seconds; three sat at zero CPU forever
# ---------------------------------------------------------------------------

class _LazilyBrokenCuda:
    """Builds happily on CUDA and raises at the first encode, like CTranslate2."""

    def __init__(self, name, device=None, compute_type=None):
        self.name, self.device, self.compute_type = name, device, compute_type

    def transcribe(self, audio, **kwargs):
        if self.device == "cuda":
            def boom():
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield  # pragma: no cover - unreachable, makes this a generator
            return boom(), SimpleNamespace(duration=1.0, language="en")
        return iter([]), SimpleNamespace(duration=1.0, language="en")


def test_a_gpu_that_cannot_encode_is_caught_before_any_file_is_handed_it(monkeypatch):
    """
    The whole point of the probe. Failing here costs one encode of silence;
    failing at the first real file costs every other worker as well.
    """
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    _model, device, compute_type = twh._get_model("tiny", "cuda", "float16")
    assert device == "cpu", "a GPU that cannot encode is not a GPU we can use"
    assert compute_type == "int8", "and the compute type has to follow it down"


def test_the_fallback_says_why_it_happened(monkeypatch, capsys):
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    twh._get_model("tiny", "cuda", "float16")

    said = capsys.readouterr().out
    assert "cublas64_12.dll" in said, "the actual error is the useful part"
    assert "torch" in said, "'but torch sees my GPU' is the wrong conclusion to draw"


def test_every_worker_gets_the_same_answer_about_the_gpu(monkeypatch):
    """
    Decided once, under the cache lock. If each worker probed for itself they
    could disagree, and a worker that still believed in the GPU would queue work
    on a model that cannot serve it -- which is the hang, restored.
    """
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    asked_for_cuda = twh._get_model("tiny", "cuda", "float16")
    asked_again = twh._get_model("tiny", "cuda", "float16")
    asked_for_cpu = twh._get_model("tiny", "cpu", "int8")

    assert asked_for_cuda is asked_again, "the failed probe must not be repeated"
    assert asked_for_cuda is asked_for_cpu, "and it is the same working model"


def test_a_deliberate_cuda_request_is_an_error_not_a_slow_run(monkeypatch):
    """
    `device="auto"` is a preference and gets the fallback. Naming `cuda` is an
    instruction, and silently doing something else to it would hide a broken
    install from the one person who has said they care.
    """
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    resolved, _reason = __import__("taters.helpers.gpu", fromlist=["x"]).resolve_device("cuda")
    assert resolved == "cuda"


def test_a_cpu_failure_is_not_swallowed(monkeypatch):
    """There is nowhere left to fall back to, so it has to be reported."""
    class AlwaysBroken(_LazilyBrokenCuda):
        def transcribe(self, audio, **kwargs):
            def boom():
                raise RuntimeError("no")
                yield  # pragma: no cover
            return boom(), SimpleNamespace(duration=1.0, language="en")

    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=AlwaysBroken))

    with pytest.raises(RuntimeError):
        twh._get_model("tiny", "cpu", "int8")


def test_the_probe_drains_the_generator_where_the_work_actually_is(monkeypatch):
    """
    `transcribe()` returns a lazy generator and does no encoding by itself.
    A probe that called it and walked away would prove nothing at all -- which
    is precisely the mistake that let this reach a user.
    """
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    # _LazilyBrokenCuda only raises once we iterate its generator
    _model, device, _compute = twh._get_model("tiny", "cuda", "float16")
    assert device == "cpu"


def test_the_run_records_the_device_it_fell_back_to(wav, monkeypatch):
    """
    Not the one that was asked for. "Was this run on the GPU?" is the first
    question anyone asks about a transcription that took an hour, and the
    manifest is where they will look months later.
    """
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=_LazilyBrokenCuda))

    outs = transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cuda",
                                   verbose=False)
    assert outs.device == "cpu"
    assert outs.compute_type == "int8"


# ---------------------------------------------------------------------------
# segment boundaries that mean something
#
# somebody noticed that "the timestamps are comprehensive -- there are no
# periods of silence that can be inferred from the transcript file".
#
# they're right, and the VAD isn't the cause. Whisper's decoder ends each
# segment where the next one begins, so the boundaries are an artifact of the
# segmentation rather than a claim about the audio. we measured this on a
# 771-second lecture: 126 of 133 gaps were exactly zero, and the transcript
# claimed 750s of speech where the words underneath only account for 681s.
# aligning the text to the audio fixes the boundaries, since faster-whisper
# then reports segment times taken from the words themselves
# ---------------------------------------------------------------------------

def test_text_is_aligned_to_the_audio_by_default(wav, fake_model):
    """
    On by default, and deliberately so: the alternative is a transcript whose
    times cannot be used to find a pause. `split_wav_by_speaker` cuts on these
    times and `analyze_vocal_acoustics` measures the pieces, pause features
    included, so the default has to be the honest one.
    """
    model = fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu", verbose=False)

    assert model.calls[0]["word_timestamps"] is True


def test_alignment_can_be_turned_off_for_speed(wav, fake_model):
    """~20% of the runtime, for anyone who would rather have the throughput."""
    model = fake_model([FakeSegment(start=0.0, end=1.0, text="hi")])
    transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                            verbose=False, word_timestamps=False)

    assert model.calls[0]["word_timestamps"] is False


def test_a_silent_stretch_survives_into_the_csv(wav, fake_model):
    """
    The point of the whole change: a gap between two segments has to still be a
    gap once it has been written out and read back. Nothing in the writer should
    round it away or stretch a segment to meet its neighbor.
    """
    fake_model(
        [FakeSegment(start=0.0, end=2.0, text="first"),
         FakeSegment(start=9.5, end=12.0, text="after a long pause")],
        duration=15.0,
    )
    outs = transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                                   verbose=False)

    rows = read_rows(outs.raw_files["csv"])
    assert len(rows) == 2
    gap = float(rows[1]["start_time"]) - float(rows[0]["end_time"])
    assert gap == pytest.approx(7500), f"a 7.5s silence became {gap}ms"


def test_the_transcript_does_not_claim_more_speech_than_the_audio_holds(wav, fake_model):
    fake_model(
        [FakeSegment(start=0.0, end=2.0, text="one"),
         FakeSegment(start=9.5, end=12.0, text="two")],
        duration=15.0,
    )
    outs = transcribe_with_whisper(wav, out_dir=wav.parent / "t", device="cpu",
                                   verbose=False)

    rows = read_rows(outs.raw_files["csv"])
    speech = sum(float(r["end_time"]) - float(r["start_time"]) for r in rows)
    assert speech < 15_000
    assert speech == pytest.approx(4500)


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------

def test_translate_asks_the_model_for_its_translate_task(wav, fake_model):
    """
    Whisper's multilingual models translate to English as a second decoding
    task; `translate=True` has to reach the model as `task="translate"`, and
    the default has to stay a plain transcription.
    """
    model = fake_model([FakeSegment(start=0.0, end=1.0, text="hello")])
    transcribe_with_whisper(wav, whisper_model="base", translate=True,
                            verbose=False)
    assert model.calls[-1]["task"] == "translate"
    transcribe_with_whisper(wav, whisper_model="base", verbose=False,
                            overwrite_existing=True)
    assert model.calls[-1]["task"] == "transcribe"


def test_translating_with_an_english_only_model_is_refused_before_loading(
        wav, monkeypatch):
    """An English-only model has no other language to translate from; what it
    returns is nonsense that looks like output, so the ask is refused in
    words, naming the model to use instead, before any weights load."""
    monkeypatch.setattr(twh, "_get_model",
                        lambda *a, **k: pytest.fail("should not load"))
    with pytest.raises(ValueError, match="English-only.*'small'"):
        transcribe_with_whisper(wav, whisper_model="small.en", translate=True,
                                verbose=False)


@pytest.mark.parametrize("name,expected", [
    ("base.en", True), ("small.en", True), ("base", False),
    ("large-v3", False), ("models/my-ct2", False), ("models/mine.en/", True),
])
def test_english_only_is_read_off_the_model_name(name, expected):
    assert twh.english_only(name) is expected
