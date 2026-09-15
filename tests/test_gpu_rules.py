"""
The universal rules for anything that touches a GPU.

Two questions, answered in one place -- `taters.helpers.gpu` -- so that every
entry point gives the same answer: the pipeline, the facade, and each module's
own CLI.

  1. Which device? "auto" has to become "cuda" or "cpu", and the answer has to
     be *true* rather than merely plausible.
  2. How many files at once? More workers on a GPU step is not the free win it
     is on a CPU step.

Both used to be answered module by module, with the predictable result:
`transcribe_with_whisper` asked CTranslate2 how many CUDA devices it could see
while `whisper_diar_wrapper` asked torch, three modules had no device argument
at all, and the concurrency limit lived in the wizard's recipe catalog where the
pipeline runner could not read it.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest import mock

import pytest

from taters.helpers import gpu


# ---------------------------------------------------------------------------
# Every module that can touch a GPU
# ---------------------------------------------------------------------------

# (import path, the public function users and the pipeline call)
GPU_MODULES = [
    ("taters.audio.transcribe_with_whisper", "transcribe_with_whisper"),
    ("taters.audio.extract_whisper_embeddings", "extract_whisper_embeddings"),
    ("taters.audio.diarizer.whisper_diar_wrapper", "run_whisper_diarization_repo"),
    ("taters.text.extract_sentence_embeddings", "extract_sentence_embeddings"),
    ("taters.text.analyze_with_archetypes", "analyze_with_archetypes"),
]


@pytest.fixture(params=GPU_MODULES, ids=lambda p: p[0].rsplit(".", 1)[-1])
def gpu_function(request):
    module_name, func_name = request.param
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"{module_name} needs an optional dependency: {exc}")
    return getattr(module, func_name)


def test_every_gpu_module_lets_you_choose_the_device(gpu_function):
    """
    `extract_sentence_embeddings` and the archetype analyzer had no `device`
    argument at all, so sentence-transformers took the GPU whenever torch
    reported one -- fine until it is the third model in a pipeline to do so, and
    impossible to opt out of.
    """
    params = inspect.signature(gpu_function).parameters
    assert "device" in params, (
        f"{gpu_function.__name__} can run a model on the GPU but offers no way "
        f"to say where"
    )


def test_the_default_is_auto_everywhere(gpu_function):
    """
    Consistency is the point: the same word means the same thing in every
    module, so someone who has learned one `device` argument has learned all of
    them.
    """
    default = inspect.signature(gpu_function).parameters["device"].default
    assert default in (gpu.DEVICE_DEFAULT, None), (
        f"{gpu_function.__name__} defaults to {default!r}, not 'auto'"
    )


def test_no_module_resolves_the_device_for_itself():
    """
    The rule that keeps the rest true. Two modules used to carry their own
    resolver -- one asking torch, one asking CTranslate2 -- and a third asked
    nobody. A local copy is how they drift.
    """
    offenders = []
    for module_name, _func in GPU_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        source = inspect.getsource(module)
        if "torch.cuda.is_available()" in source:
            offenders.append(f"{module_name} asks torch directly")
        if "get_cuda_device_count()" in source:
            offenders.append(f"{module_name} asks CTranslate2 directly")
    assert not offenders, "; ".join(offenders)


def test_every_module_cli_offers_the_same_device_flag():
    """
    A module CLI is the override hatch -- `--device cuda` has to mean what it
    says. It also has to look identical everywhere, which it will not if each
    parser spells out its own choices and help text.
    """
    checked = 0
    for module_name, _func in GPU_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cli = getattr(module, "CLI", None)
        if cli is None:
            continue
        for parser in cli.parsers_for_test().values():
            action = next(a for a in parser._actions if "--device" in a.option_strings)
            assert tuple(action.choices) == gpu.DEVICE_CHOICES, module_name
            assert action.default == gpu.DEVICE_DEFAULT, module_name
            assert action.help == gpu.DEVICE_HELP, module_name
            checked += 1
    assert checked, "no module CLI was actually inspected"


# ---------------------------------------------------------------------------
# Rule 1: which device
# ---------------------------------------------------------------------------

def test_naming_a_device_is_an_instruction_not_a_preference():
    """
    Silently doing something slower to the one person who has said they care
    about the GPU is worse than failing in front of them.
    """
    assert gpu.resolve_device("cuda") == ("cuda", None)
    assert gpu.resolve_device("cpu") == ("cpu", None)


def test_auto_falls_back_to_the_cpu_with_a_reason(monkeypatch):
    monkeypatch.setattr(gpu, "_cuda_visible", lambda backend: (False, "no driver"))

    device, reason = gpu.resolve_device("auto")
    assert device == "cpu"
    assert "no driver" in reason
    assert "torch" in reason, "'but torch sees my GPU' is the wrong conclusion to draw"


def test_the_device_note_says_where_the_work_runs_and_why_when_it_fell_back():
    """
    A run that quietly took the CPU looked exactly like one on the card, so
    somebody sat through an hour of transformer embeddings before noticing.
    The note goes to the progress display, where it stays up for the length
    of the work, and it carries the reason when `auto` had to fall back --
    "on the CPU" without a reason is an invitation to ask why.
    """
    assert gpu.device_note("encoding", "cuda") == "encoding on the GPU"
    assert gpu.device_note("encoding", "cuda:1") == "encoding on the GPU"
    assert gpu.device_note("encoding", "cpu") == "encoding on the CPU"
    note = gpu.device_note("scoring 12 text(s)", "cpu", "torch reports no usable CUDA device")
    assert note.startswith("scoring 12 text(s) on the CPU")
    assert "no usable CUDA device" in note


#: The steps that pick a device and then work for a long time on it. Each has
#: to say which device it got, through the progress display rather than a
#: print, or a run that fell back to the CPU is indistinguishable from a fast
#: one on the card until somebody checks a GPU meter.
DEVICE_ANNOUNCERS = ("taters.text.transformer_embeddings",
                     "taters.text.extract_sentence_embeddings",
                     "taters.text.hf_classifier")


@pytest.mark.parametrize("module_name", DEVICE_ANNOUNCERS)
def test_a_long_running_device_step_says_which_device_it_got(module_name):
    """
    The transformer-embeddings step is the one that prompted this -- it ran an
    hour on the CPU without a word -- but a rule that only one module follows
    is a rule that the next module forgets.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} needs an extra that is not installed")
    source = inspect.getsource(module)
    assert "device_note(" in source, (
        f"{module_name} picks a device and never says which one it got")
    assert "announce(" in source, (
        f"{module_name} has the note but never sends it to the progress display")


def test_a_visible_gpu_that_cannot_run_anything_is_not_used(monkeypatch):
    """
    The failure that cost a user two hours. Both backends load their CUDA
    libraries at the first inference, so a broken install builds a model without
    complaint and raises later -- by which time the other workers are queued
    behind it.
    """
    monkeypatch.setattr(gpu, "_cuda_visible", lambda backend: (True, None))

    def broken():
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    device, reason = gpu.resolve_device("auto", probe=broken)
    assert device == "cpu"
    assert "cublas64_12.dll" in reason


def test_a_probe_failure_on_an_explicit_cuda_is_raised(monkeypatch):
    monkeypatch.setattr(gpu, "_cuda_visible", lambda backend: (True, None))

    def broken():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        gpu.resolve_device("cuda", probe=broken)


def test_asking_for_the_cpu_never_runs_the_probe():
    """
    This used to assert the opposite -- "there is nowhere left to fall back
    to, so it has to be reported" -- and the intent was right but the probe is
    not what that intent assumed. Every caller's probe is a *CUDA* probe
    (`torch.zeros(1, device="cuda")`); there is no such thing as a CPU probe
    in this codebase. So running it after resolving to the CPU asks a question
    about hardware nobody is about to use, and on a CPU-only build of torch it
    does not return an answer at all -- it raises "Torch not compiled with
    CUDA enabled" and takes the run down.

    It survived because the old test passed a probe that raised whatever it
    was asked, and because a machine whose torch has CUDA never notices. CI on
    a CPU-only runner failed about thirty tests on it.
    """
    called = []

    def probe():
        called.append(True)
        raise RuntimeError("a CUDA probe, on a machine with no CUDA")

    assert gpu.resolve_device("cpu", probe=probe) == ("cpu", None)
    assert not called, "the CPU answer does not depend on the GPU working"


def test_the_two_backends_are_asked_separately(monkeypatch):
    """
    Not an inconsistency to be tidied away: torch and CTranslate2 ship their own
    CUDA libraries and genuinely disagree about the same card -- which is exactly
    the bug that started this. A faster-whisper module must ask CTranslate2; a
    torch module must ask torch.
    """
    import sys
    from types import SimpleNamespace

    # two fake libraries that disagree: torch sees a card, CTranslate2 doesn't.
    # asking the wrong one is how a run ends up believing in a GPU it can't use,
    # so what we're really testing here is the dispatch itself -- not just that
    # the argument gets passed along.
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 0))

    assert gpu.resolve_device("auto", backend="torch")[0] == "cuda"
    assert gpu.resolve_device("auto", backend="ctranslate2")[0] == "cpu"


def test_a_backend_that_will_not_import_means_the_cpu(monkeypatch):
    """A missing library is not a reason to fail; it is a reason to use the CPU."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in ("torch", "ctranslate2"):
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    device, reason = gpu.resolve_device("auto", backend="ctranslate2")
    assert device == "cpu"
    assert "would not import" in reason


def test_an_unknown_device_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        gpu.resolve_device("gpu")          # a plausible typo for "cuda"


@pytest.mark.parametrize("given", [None, "", "  ", "AUTO", "Cuda", " cpu "])
def test_device_strings_are_read_forgivingly(given):
    device, _reason = gpu.resolve_device(given)
    assert device in ("cuda", "cpu")


def test_the_gpu_can_be_hidden_from_a_library_that_will_not_be_told(monkeypatch):
    """
    `ArchetypeQuantifier` builds a `SentenceTransformer` itself and takes no
    device, so making the GPU invisible is the only way to honor device="cpu".
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    import os
    with gpu.hidden_gpu():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0", "the setting has to come back"


def test_hiding_nothing_leaves_the_environment_alone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    import os
    with gpu.hidden_gpu(False):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_an_unset_variable_is_left_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    import os
    with gpu.hidden_gpu():
        pass
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


# ---------------------------------------------------------------------------
# Rule 2: how many files at once
# ---------------------------------------------------------------------------

def test_the_caps_are_the_measured_ones():
    """
    Measured over four files with `tiny`. `gpu_one_model` keeps VRAM flat at
    272 MiB from 1 worker to 4 and stops getting faster after 2; `gpu_model_each`
    grows 240 -> 479 -> 893 MiB and is *slower* at every step up.
    """
    assert gpu.WORKER_CAP == {"cpu": None, "gpu_one_model": 2, "gpu_model_each": 1}


def test_an_unknown_answer_is_treated_as_the_expensive_one():
    """
    The safety net. Wrong-but-slow is recoverable; four copies of `large-v3` on
    an 8 GB card is an out-of-memory error halfway through a batch.
    """
    assert gpu.worker_cap(None) == 1
    assert gpu.worker_cap("gpu_somethingelse") == 1


def test_a_cpu_step_is_never_capped():
    """`--workers 8` has to keep meaning 8 for ffmpeg and the merge steps."""
    assert gpu.worker_cap("cpu") is None


def test_the_rules_do_not_depend_on_the_wizard():
    """
    They live in `helpers`, not `ui`. The pipeline runner needs them and must not
    import the wizard to get them -- which is where they started, and why the
    runner used to carry a second copy of the cautious default.
    """
    source = inspect.getsource(gpu)
    assert "taters.ui" not in source and "from ..ui" not in source


# ---------------------------------------------------------------------------
# The CUDA math libraries, which are a separate question from the card
#
# this one bit us for real: "Check my setup" reported "Transcription on GPU:
# yes -- 1 CUDA device(s) available" and "Everything checks out", and then
# every file failed with `Library cublas64_12.dll is not found or cannot be
# loaded`.
#
# both the checker and the transcriber were asking `get_cuda_device_count()`,
# which the *driver* answers. the driver is always there on a machine with a
# card. the math libraries ship separately, get loaded lazily at the first
# inference, and are what actually does the work.
# ---------------------------------------------------------------------------

def test_the_library_check_is_fair_on_a_machine_that_works():
    """
    The trap in writing this check. On Linux the libraries come from the
    `nvidia-*` wheels and are found through the RPATH in CTranslate2's own
    extension module -- not through any path the loader searches by default. A
    bare `CDLL("libcublas.so.12")` therefore fails on a machine where
    transcription runs on the GPU perfectly well, and a diagnostic that cries
    wolf is worse than no diagnostic at all.
    """
    import ctranslate2

    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device here, so there is nothing to be fair about")

    ok, detail = gpu.cuda_libraries_ok()
    assert ok is not False, f"false alarm on a working machine: {detail}"


def test_the_check_imports_ctranslate2_before_asking():
    """That fairness is not incidental -- it is the mechanism."""
    source = inspect.getsource(gpu.cuda_libraries_ok)
    assert "_import_ctranslate2()" in source


def test_no_opinion_is_offered_where_the_question_makes_no_sense(monkeypatch):
    """
    macOS has no cuBLAS to find, and `None` means "no opinion" rather than
    "broken" -- a red row on a Mac would be a lie.
    """
    monkeypatch.setattr(gpu.platform, "system", lambda: "Darwin")

    ok, detail = gpu.cuda_libraries_ok()
    assert ok is None
    assert "not applicable" in detail


def test_a_missing_library_is_reported_with_somewhere_to_look(monkeypatch):
    """
    The detail is the whole value. "cuBLAS is missing" sends someone to a search
    engine; "a copy is in <torch>/lib and nothing adds that folder to the search
    path" tells them what is actually wrong.
    """
    from pathlib import Path

    monkeypatch.setattr(gpu, "_can_load", lambda name: False)
    monkeypatch.setattr(gpu, "_import_ctranslate2", lambda: True)
    monkeypatch.setattr(gpu, "_library_dirs",
                        lambda names=(): [Path("/x/torch/lib")])

    ok, detail = gpu.cuda_libraries_ok()
    assert ok is False
    assert "/x/torch/lib" in detail
    assert "search path" in detail


def test_a_library_nothing_ships_says_so_instead(monkeypatch):
    monkeypatch.setattr(gpu, "_can_load", lambda name: False)
    monkeypatch.setattr(gpu, "_import_ctranslate2", lambda: True)
    monkeypatch.setattr(gpu, "_library_dirs", lambda names=(): [])
    monkeypatch.setattr(gpu, "_candidate_dirs", lambda: [])

    ok, detail = gpu.cuda_libraries_ok()
    assert ok is False
    assert "no installed package" in detail


def test_a_directory_only_counts_if_the_library_is_actually_in_it(tmp_path):
    """
    The bug this replaced, reported from a real machine. `_library_dirs` used to
    return any directory that merely existed, so the check announced "a copy
    appears to be in <torch>/lib" about a folder holding no such file -- and the
    repair then pointed the loader at it and called that an attempt.
    """
    empty = tmp_path / "torch" / "lib"
    empty.mkdir(parents=True)

    with mock.patch.object(gpu, "_candidate_dirs", lambda: [empty]):
        assert gpu._library_dirs(("cublas64_12.dll",)) == []
        (empty / "cublas64_12.dll").write_bytes(b"")
        assert gpu._library_dirs(("cublas64_12.dll",)) == [empty]


def test_the_wrong_cuda_version_is_named_as_such(tmp_path):
    """
    From the same machine: torch 2.13.0+cu132 ships `cublas64_13.dll`, and
    ctranslate2 4.8.1 wants `cublas64_12.dll`. That is a version mismatch, not a
    search-path problem, and the two need opposite advice -- telling someone to
    fix their path when their torch is built against the wrong CUDA sends them
    in a circle.
    """
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "cublas64_13.dll").write_bytes(b"")

    with mock.patch.dict(gpu._CUDA_LIBRARIES, {"Linux": ("cublas64_12.dll",)}), \
         mock.patch.object(gpu, "_candidate_dirs", lambda: [lib]), \
         mock.patch.object(gpu, "_import_ctranslate2", lambda: True), \
         mock.patch.object(gpu.platform, "system", lambda: "Linux"):
        ok, detail = gpu.cuda_libraries_ok()

    assert ok is False
    assert "cublas64_13.dll" in detail
    assert "version mismatch" in detail
    assert "would not help" in detail


def test_a_repair_that_cannot_work_is_not_attempted(tmp_path):
    """
    A no-op reported as a fix is the worst outcome available here: it sends
    someone away believing the problem is solved.
    """
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "cublas64_13.dll").write_bytes(b"")

    with mock.patch.dict(gpu._CUDA_LIBRARIES, {"Linux": ("cublas64_12.dll",)}), \
         mock.patch.object(gpu, "_candidate_dirs", lambda: [lib]), \
         mock.patch.object(gpu, "_import_ctranslate2", lambda: True), \
         mock.patch.object(gpu.platform, "system", lambda: "Linux"):
        assert gpu.ensure_cuda_libraries() is False


def test_repair_is_not_attempted_when_nothing_is_wrong(monkeypatch):
    """Adding directories to a loader that is already finding things is noise."""
    monkeypatch.setattr(gpu, "cuda_libraries_ok", lambda: (True, "fine"))
    monkeypatch.setattr(gpu, "_library_dirs",
                        lambda names=(): pytest.fail("looked for libraries it did not need"))

    assert gpu.ensure_cuda_libraries() is True


def test_repair_reports_honestly_when_it_cannot_help(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(gpu, "cuda_libraries_ok", lambda: (False, "nope"))
    monkeypatch.setattr(gpu, "_library_dirs", lambda names=(): [Path("/nowhere")])

    assert gpu.ensure_cuda_libraries() is False


def test_the_transcriber_tries_the_repair_before_loading_a_cuda_model(monkeypatch):
    """
    Before, not after. The libraries are loaded by `WhisperModel`'s first
    inference, so a repair that runs afterwards has missed its only chance --
    which is why this watches the order of real calls rather than reading the
    source. A source check cannot tell a call that runs from one sitting behind
    a condition that is never true.
    """
    import sys
    from types import SimpleNamespace

    from taters.audio import transcribe_with_whisper as twh

    order = []
    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setattr(twh, "ensure_cuda_libraries",
                        lambda: order.append("repair") or True)
    monkeypatch.setattr(twh, "_prove_it_works", lambda model: None)

    class Model:
        def __init__(self, name, device=None, compute_type=None):
            order.append("load")

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=Model))

    twh._get_model("tiny", "cuda", "float16")
    assert order == ["repair", "load"]


def test_no_repair_is_attempted_for_a_cpu_model(monkeypatch):
    """Nothing on the CPU path needs cuBLAS, so looking for it is pure cost."""
    import sys
    from types import SimpleNamespace

    from taters.audio import transcribe_with_whisper as twh

    monkeypatch.setattr(twh, "_MODEL_CACHE", {})
    monkeypatch.setattr(twh, "ensure_cuda_libraries",
                        lambda: pytest.fail("looked for CUDA libraries on the CPU path"))
    monkeypatch.setattr(twh, "_prove_it_works", lambda model: None)
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=lambda *a, **k: object()))

    twh._get_model("tiny", "cpu", "int8")


def test_the_repair_actually_makes_a_library_loadable():
    """
    The claim, tested rather than reasoned about.

    Every other test here mocks the loader, which proves the plumbing and not
    the premise. This one finds a real shared library that ships inside an
    installed wheel and is genuinely *not* loadable by bare name -- the exact
    state a broken CUDA install is in -- points the real machinery at it, and
    checks that it becomes loadable.

    Linux only, because the mechanisms differ: here a preload by absolute path
    puts the library in the process, where Windows instead extends the DLL
    search path. The Windows half cannot be exercised from a POSIX runner.
    """
    import ctypes
    import platform
    from unittest import mock

    if platform.system() != "Linux":
        pytest.skip("this exercises the preload mechanism, which is POSIX-only")

    def loadable(name: str) -> bool:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            return False

    # we want anything in a candidate directory that the loader can't find on
    # its own.
    target = None
    for directory in gpu._library_dirs():
        for so in sorted(directory.glob("*.so.*")):
            if not loadable(so.name):
                target = so.name
                break
        if target:
            break
    if target is None:
        pytest.skip("nothing installed here is unloadable, so there is nothing to fix")

    with mock.patch.dict(gpu._CUDA_LIBRARIES, {"Linux": (target,)}):
        assert gpu.cuda_libraries_ok()[0] is False, "the premise did not hold"
        assert gpu.ensure_cuda_libraries() is True, f"could not make {target} loadable"

    assert loadable(target), f"{target} is still not loadable after the repair"


def test_no_useless_directory_is_added_to_the_dll_search_path(tmp_path):
    """
    Windows-only behavior, exercised by pretending. `os.add_dll_directory`
    changes how *every* later library load resolves in this process, so adding
    folders that cannot possibly hold what is wanted is not merely wasted work:
    it widens the search path of an unrelated program for the rest of its life.
    """
    has_it = tmp_path / "cublas" / "lib"
    has_it.mkdir(parents=True)
    (has_it / "cublas64_12.dll").write_bytes(b"")
    hasnt = tmp_path / "torch" / "lib"
    hasnt.mkdir(parents=True)
    (hasnt / "cublas64_13.dll").write_bytes(b"")

    added = []
    with mock.patch.dict(gpu._CUDA_LIBRARIES, {"Windows": ("cublas64_12.dll",)}), \
         mock.patch.object(gpu.platform, "system", lambda: "Windows"), \
         mock.patch.object(gpu, "_candidate_dirs", lambda: [hasnt, has_it]), \
         mock.patch.object(gpu, "_import_ctranslate2", lambda: True), \
         mock.patch.object(gpu, "_can_load", lambda name: False), \
         mock.patch.object(gpu.os, "add_dll_directory",
                           lambda d: added.append(d) or object(),
                           create=True):     # POSIX `os` has no such attribute
        gpu.ensure_cuda_libraries()

    assert added == [str(has_it)], f"pointed the loader at folders that cannot help: {added}"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "docs" / "install-guide.md").exists(),
    reason="docs are not present in this checkout")
def test_the_install_guide_recommends_the_same_cuda_the_checker_does():
    """
    The guide used to send people to pytorch.org's installer page to pick a
    CUDA version. That page does not know CTranslate2 cannot leave CUDA 12,
    so it hands out a CUDA 13 build and transcription quietly drops to the
    CPU -- which is how at least one of us got there.

    So the guide names the tag, and this keeps it the tag the checker would
    give: when CTranslate2 ships CUDA 13 wheels and `MAX_CUDA` moves, this
    fails and the guide gets updated in the same breath.
    """
    from taters.ui.tasks.gpu import cuda_tag

    guide = (Path(__file__).resolve().parents[1] / "docs" / "install-guide.md"
             ).read_text(encoding="utf-8")
    recommended = cuda_tag(None)          # no driver named: the stack's own cap
    assert f"download.pytorch.org/whl/{recommended}" in guide, \
        f"the install guide does not tell anyone to install {recommended}"
    # and it must not send them off to choose for themselves again
    assert "pytorch.org/get-started" not in guide, \
        "the installer page hands out CUDA 13 builds; name the version instead"
