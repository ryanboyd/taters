"""
Getting a model's files, kept apart from putting the model on the GPU.

A real run: the first use of Whisper large-v3 on Windows downloaded 3 GB,
one file of which failed with WinError 1314 (the hub library had told one of
its download threads it could make a link, and Windows said no). The failure
landed inside the GPU test, was reported as "the GPU is visible but
CTranslate2 cannot use it", and all six videos were transcribed on the CPU.

Three things are tested here: the hub library's race itself, and that
settling its question first ends it; that a download failure is said as a
download failure and never sends a run to the CPU; and that only the GPU's
own failures do.
"""
from __future__ import annotations

import os
import threading
import types

import pytest

from taters.audio import transcribe_with_whisper as twh
from taters.helpers import hub_models as hm
from taters.helpers.gpu import is_gpu_failure


# ---------------------------------------------------------------------------
# what counts as the GPU's failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [
    RuntimeError("CUDA failed with error unknown error"),
    RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"),
    RuntimeError("cuDNN failed with status CUDNN_STATUS_NOT_INITIALIZED"),
    ValueError("Requested float16 compute type, but the target device or backend do not "
               "support efficient float16 computation."),
    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    RuntimeError("no kernel image is available for execution on the device"),
    OSError("libcudnn_ops.so.9: cannot open shared object file"),
])
def test_the_gpus_own_failures_are_recognized(error):
    assert is_gpu_failure(error)


@pytest.mark.parametrize("error", [
    OSError("[WinError 1314] A required privilege is not held by the client: "
            "'..\\\\..\\\\blobs\\\\931c' -> 'C:\\\\Users\\\\x\\\\snapshots\\\\preprocessor_config.json'"),
    OSError(28, "No space left on device"),
    ValueError("Invalid model size 'large-v9', expected one of: tiny, base"),
    ConnectionError("HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded"),
    PermissionError("[Errno 13] Permission denied: '/models/model.bin'"),
])
def test_other_failures_are_not_the_gpus(error):
    assert not is_gpu_failure(error)


# ---------------------------------------------------------------------------
# the hub library's race, and settling it
# ---------------------------------------------------------------------------

def test_the_hub_library_tells_a_second_thread_yes_before_it_has_tried(tmp_path, monkeypatch):
    """The race as the hub library has it: the answer is recorded as yes, then
    tested. A thread that asks while the test is under way gets the yes."""
    from huggingface_hub import file_download as fd

    started, release = threading.Event(), threading.Event()

    def slow_refusal(src, dst, *a, **k):
        started.set()
        release.wait(5)
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", slow_refusal)
    folder = tmp_path / "cache"
    # the library's own check, as it ships (without our guard, if a test
    # before this one installed it)
    check = getattr(fd.are_symlinks_supported, "__wrapped__", fd.are_symlinks_supported)
    first = threading.Thread(target=check, args=(folder,))
    first.start()
    assert started.wait(5)
    told = check(folder)      # asked while the test runs
    release.set()
    first.join(5)
    assert told is True, "the race: told yes"
    assert check(folder) is False, "the true answer, afterwards"


def test_settling_first_means_every_later_thread_is_told_the_truth(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub import file_download as fd

    def refuse(src, dst, *a, **k):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", refuse)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    folder = tmp_path / "cache"
    monkeypatch.setattr(hm, "_SETTLED", set())
    hm.settle_links(folder)
    answers = []
    threads = [threading.Thread(target=lambda: answers.append(fd.are_symlinks_supported(folder)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert answers == [False] * 8


def test_a_models_own_folder_is_told_the_truth_too(tmp_path, monkeypatch):
    """The real failure in a chat (2026-11-06): the cache root was settled,
    but the library asks again about each model's own folder
    (models--org--name) mid-download, and a second thread got the early yes.
    With the check guarded, the second thread waits and is told no."""
    from huggingface_hub import constants
    from huggingface_hub import file_download as fd

    started, release = threading.Event(), threading.Event()

    def slow_refusal(src, dst, *a, **k):
        started.set()
        release.wait(5)
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", slow_refusal)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(fd, "are_symlinks_supported",
                        getattr(fd.are_symlinks_supported, "__wrapped__",
                                fd.are_symlinks_supported))
    monkeypatch.setattr(hm, "_SETTLED", set())
    release.set()                       # settling the root: no waiting
    hm.settle_links(tmp_path / "hub")
    release.clear()
    started.clear()
    repo = tmp_path / "hub" / "models--org--name"
    first = threading.Thread(target=fd.are_symlinks_supported, args=(repo,))
    first.start()
    assert started.wait(5)
    answers = []
    second = threading.Thread(target=lambda: answers.append(fd.are_symlinks_supported(repo)))
    second.start()
    second.join(0.3)
    assert second.is_alive(), "the second thread waits for the first one's test"
    release.set()
    first.join(5)
    second.join(5)
    assert answers == [False]


def test_the_guard_is_installed_once(tmp_path, monkeypatch):
    from huggingface_hub import file_download as fd

    monkeypatch.setattr(fd, "are_symlinks_supported", lambda folder=None: True)
    monkeypatch.setattr(hm, "_SETTLED", set())
    hm.settle_links(tmp_path / "a")
    guarded = fd.are_symlinks_supported
    hm.settle_links(tmp_path / "b")
    assert fd.are_symlinks_supported is guarded and guarded._taters_guard


def test_settling_happens_once_per_folder_however_many_threads_ask(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub import file_download as fd

    calls = []
    monkeypatch.setattr(fd, "are_symlinks_supported", lambda folder: calls.append(folder))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hm, "_SETTLED", set())
    threads = [threading.Thread(target=hm.settle_links, args=(tmp_path / "cache",))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(calls) == sorted({str(tmp_path / "hub"), str(tmp_path / "cache")})


def test_settling_never_raises(tmp_path, monkeypatch):
    from huggingface_hub import file_download as fd

    def broken(folder):
        raise PermissionError("read-only")

    monkeypatch.setattr(fd, "are_symlinks_supported", broken)
    monkeypatch.setattr(hm, "_SETTLED", set())
    hm.settle_links(tmp_path / "x")


# ---------------------------------------------------------------------------
# fetching, and what a failed fetch is not
# ---------------------------------------------------------------------------

@pytest.fixture
def downloads(monkeypatch):
    """faster-whisper's download, scripted: each entry is raised or returned."""
    import faster_whisper.utils as fwu

    script = []
    seen = []

    def download_model(name, **kw):
        seen.append(name)
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(fwu, "download_model", download_model)
    settled = []
    monkeypatch.setattr(hm, "settle_links", lambda *a, **k: settled.append(True))
    return types.SimpleNamespace(script=script, seen=seen, settled=settled)


def test_a_folder_is_used_as_it_is(tmp_path, downloads):
    assert twh.fetch_whisper_model(str(tmp_path)) == str(tmp_path)
    assert downloads.seen == []


def test_a_name_is_fetched_after_the_links_question_is_settled(downloads):
    downloads.script.append("/cache/large-v3")
    assert twh.fetch_whisper_model("large-v3") == "/cache/large-v3"
    assert downloads.settled == [True] and downloads.seen == ["large-v3"]


def test_a_download_that_fails_once_is_tried_again(downloads):
    downloads.script += [OSError(1314, "A required privilege is not held by the client"),
                         "/cache/large-v3"]
    assert twh.fetch_whisper_model("large-v3") == "/cache/large-v3"


def test_a_download_that_keeps_failing_is_said_as_a_download(downloads):
    downloads.script += [OSError(28, "No space left on device")] * 2
    with pytest.raises(RuntimeError, match="could not be downloaded.*not the GPU"):
        twh.fetch_whisper_model("large-v3")


class _Model:
    """A WhisperModel that fails on the GPU in the way it is told to."""

    built: list = []
    gpu_error: BaseException | None = None

    def __init__(self, path, device, compute_type):
        _Model.built.append((path, device))
        if device == "cuda" and _Model.gpu_error is not None:
            raise _Model.gpu_error

    def transcribe(self, audio, **kw):
        return iter([]), types.SimpleNamespace(duration=1.0, language="en")


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=_Model))
    monkeypatch.setattr(twh, "ensure_cuda_libraries", lambda: True)
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    _Model.built = []
    return monkeypatch


def test_a_failed_download_never_sends_a_run_to_the_cpu(loader):
    """The real run, end to end: the download fails, and the run stops with
    the download's own words instead of quietly transcribing on the CPU."""
    def fail(name):
        raise RuntimeError("the Whisper model 'large-v3' could not be downloaded (OSError: "
                           "[WinError 1314] ...). This is the download, not the GPU")

    loader.setattr(twh, "fetch_whisper_model", fail)
    with pytest.raises(RuntimeError, match="not the GPU"):
        twh._get_model("large-v3", "cuda", "float16")
    assert _Model.built == [], "no model was built on any device"


def test_a_gpu_failure_still_falls_back_to_the_cpu(loader, capsys):
    loader.setattr(twh, "fetch_whisper_model", lambda name: "/cache/large-v3")
    _Model.gpu_error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    try:
        _model, device, _ct = twh._get_model("large-v3", "cuda", "float16")
    finally:
        _Model.gpu_error = None
    assert device == "cpu"
    assert _Model.built == [("/cache/large-v3", "cuda"), ("/cache/large-v3", "cpu")], \
        "both tries use the fetched folder, so the CPU never downloads again"
    assert "Falling back to the CPU" in capsys.readouterr().out


def test_an_error_that_is_not_the_gpus_is_raised_not_hidden_on_the_cpu(loader):
    loader.setattr(twh, "fetch_whisper_model", lambda name: "/cache/x")
    _Model.gpu_error = PermissionError("[Errno 13] Permission denied: '/cache/x/model.bin'")
    try:
        with pytest.raises(PermissionError):
            twh._get_model("x", "cuda", "float16")
    finally:
        _Model.gpu_error = None
    assert _Model.built == [("/cache/x", "cuda")], "no CPU attempt"
