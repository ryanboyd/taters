"""Shared test setup for the Taters test suite.

`conftest.py` is a magic filename: pytest imports it automatically before it
runs any test in this folder, and every test can use what it defines without
importing anything. It is where "fixtures" live.

A **fixture** is a named piece of setup. You write a function, decorate it with
`@pytest.fixture`, and then any test that takes an argument with that name gets
the value back. pytest builds it fresh for each test and tears it down after:

    @pytest.fixture
    def greeting():
        return "hello"

    def test_it(greeting):        # <- pytest sees the name and injects it
        assert greeting == "hello"

That is the whole idea. Everything below is just bigger versions of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Resolved once, at import time — before any test chdir's somewhere else.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Working directory isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """
    Run every test inside its own empty temporary directory.

    This matters a lot for Taters specifically: when you do not pass an output
    path, these functions write to `./features/...`, `./audio/`, `./transcripts/`
    relative to the *current working directory*. Without this fixture, running
    the tests would litter your repo and — worse — the "don't overwrite unless
    asked" logic would start returning files left over from a previous test run,
    so tests would pass or fail depending on what ran before them.

    `autouse=True` means every test gets this whether it asks for it or not.
    `tmp_path` and `monkeypatch` are fixtures pytest provides out of the box:
    a fresh temp dir, and a tool that undoes its changes automatically.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the project root (valid even though we chdir'd away)."""
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Capability checks — skip instead of fail when something isn't installed
# ---------------------------------------------------------------------------

def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


HAVE_FFMPEG = _have("ffmpeg") and _have("ffprobe")

# A ready-made decorator: put @requires_ffmpeg above a test and it is skipped
# (not failed) on machines without ffmpeg, with that reason printed.
requires_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason="ffmpeg/ffprobe not found on PATH"
)


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip tests marked `needs_ffmpeg` when ffmpeg is missing.

    This is a pytest *hook* — a function with a reserved name that pytest calls
    at a known moment. This one runs after tests are collected but before any
    of them execute, so we can attach a skip to whole groups at once instead of
    repeating a decorator on every test.
    """
    if HAVE_FFMPEG:
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not found on PATH")
    for item in items:
        if "needs_ffmpeg" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Synthetic media — small, deterministic, generated on the fly
# ---------------------------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    """Run ffmpeg quietly, raising with its stderr if it fails."""
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()}")


def probe(path: Path) -> dict:
    """
    Return ffprobe's view of a media file: {"streams": [...], "format": {...}}.

    Tests assert on this rather than on ffmpeg's exit code, because "the command
    succeeded" and "the file has the sample rate you asked for" are different
    claims and only the second one is useful.
    """
    import json
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    return json.loads(res.stdout or "{}")


def audio_stream(path: Path) -> dict:
    """The first audio stream of a file, as an ffprobe dict."""
    for s in probe(path).get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    raise AssertionError(f"no audio stream in {path}")


@pytest.fixture
def ffprobe_json():
    """Expose `probe` to tests as a fixture, for readability."""
    return probe


@pytest.fixture
def tiny_wav(tmp_path) -> Path:
    """
    A 3-second 16 kHz mono WAV containing two tones separated by silence.

    Real interview footage is the wrong tool for testing plumbing: it is slow,
    it is not in the repo, and its content can change. A generated clip is a
    fraction of a second to build and identical on every machine, which is what
    you want for tests that only care about "did this function do the right
    thing with the bytes it was given".
    """
    out = tmp_path / "tiny.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        str(out),
    )
    return out


@pytest.fixture
def tiny_video(tmp_path) -> Path:
    """A 2-second video with one video stream and one audio stream."""
    out = tmp_path / "tiny.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out),
    )
    return out


@pytest.fixture
def tiny_video_two_audio_streams(tmp_path) -> Path:
    """A 2-second video with *two* audio streams, tagged with languages."""
    out = tmp_path / "two_tracks.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=spa",
        str(out),
    )
    return out


# ---------------------------------------------------------------------------
# Synthetic tabular inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def transcript_csv(tmp_path) -> Path:
    """
    A diarization-shaped transcript: start_time,end_time,speaker,text in ms.

    Matches what `diarize_with_thirdparty` emits, so anything that consumes a
    transcript can be tested without running a diarizer.
    """
    p = tmp_path / "transcript.csv"
    p.write_text(
        "start_time,end_time,speaker,text\n"
        "0,1000,Speaker 0,hello there how are you doing today\n"
        "1000,2000,Speaker 1,i am doing quite well thank you for asking\n"
        "2000,3000,Speaker 0,that is very good to hear my friend\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def analysis_ready_csv(tmp_path) -> Path:
    """A prebuilt `text_id,text` CSV — the format every text analyzer accepts."""
    p = tmp_path / "analysis_ready.csv"
    p.write_text(
        "text_id,text\n"
        "a,The quick brown fox jumps over the lazy dog. It was remarkably quick.\n"
        "b,Colorless green ideas sleep furiously. Furiously, they sleep.\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Real media, for the opt-in slow tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_vids_dir() -> Path:
    """
    The ./test_vids folder, or a skip if it is empty/absent.

    `scope="session"` means this is computed once for the whole run instead of
    once per test — appropriate for something read-only that never changes.
    """
    d = REPO_ROOT / "test_vids"
    if not d.is_dir() or not any(d.iterdir()):
        pytest.skip("no media in ./test_vids")
    return d


def _has_stream(path: Path, kind: str) -> bool:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", {"audio": "a", "video": "v"}[kind],
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    return res.returncode == 0 and bool(res.stdout.strip())


@pytest.fixture(scope="session")
def real_media_with_audio(test_vids_dir) -> Path:
    """
    The smallest file in ./test_vids that actually carries an audio stream.

    Smallest, because these tests are already slow enough; and audio-verified,
    because a video-only download (yt-dlp's `.f137.mp4`) would fail for a reason
    that has nothing to do with Taters.
    """
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    candidates = sorted(
        (p for p in test_vids_dir.rglob("*") if p.is_file() and p.suffix.lower() != ".part"),
        key=lambda p: p.stat().st_size,
    )
    for p in candidates:
        if _has_stream(p, "audio"):
            return p
    pytest.skip("no file in ./test_vids has an audio stream (are these video-only downloads?)")


@pytest.fixture(scope="session")
def real_media_with_both_streams(test_vids_dir) -> Path:
    """The smallest file in ./test_vids carrying both video and audio."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    candidates = sorted(
        (p for p in test_vids_dir.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    for p in candidates:
        if _has_stream(p, "audio") and _has_stream(p, "video"):
            return p
    pytest.skip("no muxed audio+video file in ./test_vids")


@pytest.fixture(scope="session")
def real_audio_clip(real_media_with_audio, tmp_path_factory) -> Path:
    """
    A 30-second 16 kHz mono WAV cut from real media, built once per session.

    Slow tests should still be *real* — real speech, real noise, real overlap —
    but a full interview would make the suite unrunnable. Thirty seconds is
    enough for diarization to find turns while keeping a run to minutes.
    """
    out = tmp_path_factory.mktemp("real_media") / "clip.wav"
    _ffmpeg(
        "-t", "30", "-i", str(real_media_with_audio),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out),
    )
    return out
