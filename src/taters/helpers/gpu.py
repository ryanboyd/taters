"""
The rules for anything that touches a GPU.

Every module here that can run a model on a GPU asks this module two questions,
and nothing else decides them:

1. **Which device?** ``"auto"`` has to become ``"cuda"`` or ``"cpu"``, and the
   answer has to be *true* -- not merely plausible. See :func:`resolve_device`.
2. **How many files at once?** More workers on a GPU step is not the free win it
   is on a CPU step, and how unfree depends on one fact about the module. See
   :data:`GPU_USE`.

Both used to be answered locally, module by module, with the predictable result:
``transcribe_with_whisper`` asked CTranslate2 how many CUDA devices it could see
while ``whisper_diar_wrapper`` asked torch, three modules had no device argument
at all, and the concurrency limit lived in the wizard's recipe catalog where the
pipeline runner could not read it. This module exists so there is one answer per
question and every entry point -- pipeline, facade, module CLI -- gets it.

Module CLIs override freely: ``--device cuda`` means what it says, and naming a
device explicitly turns a broken GPU into an error rather than a silent fallback.
The rules here are the *defaults*, not a cage.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GPU_USE", "WORKER_CAP", "DEVICE_CHOICES", "DEVICE_DEFAULT", "DEVICE_HELP",
    "resolve_device", "worker_cap", "add_device_argument", "hidden_gpu",
    "CPU_FALLBACK_NOTE", "cuda_libraries_ok", "ensure_cuda_libraries",
]


# ---------------------------------------------------------------------------
# Rule 2: how many files at once
# ---------------------------------------------------------------------------

# how many files a step gets to work on at once, decided by what a second
# worker costs us in GPU memory. one word per module, written once by whoever
# writes the module; nobody should be tuning these per run, machine, or model.
#
# the question we're really answering here: does a second worker mean a second
# copy of the model on the card?
#
#   "cpu"              no model on the GPU at all. nothing to run out of, so we
#                      honor the user's worker count in full.
#   "gpu_one_model"    one model serves every worker. we measured over four files
#                      with `tiny`: VRAM flat at 272 MiB from 1 worker to 4, and
#                      the wall clock stops improving after 2 -- the single model
#                      serializes the GPU work either way, and 2 is enough to
#                      overlap audio decoding with inference.
#   "gpu_model_each"   every worker loads its own copy. measured over the same
#                      four files: 240 -> 479 -> 893 MiB at 1, 2 and 4 workers,
#                      and *slower* at every step up (69s -> 79s -> 89s). worse
#                      on both axes, so we just run these one file at a time.
#
# scale matters more than those numbers suggest: 240 MiB is `tiny`. on
# `large-v3` the same four-worker run wants roughly 12 GB, which doesn't fit on
# an 8 GB card -- and a worker that dies mid-run can strand the others.
GPU_USE = ("cpu", "gpu_one_model", "gpu_model_each")

# what each answer caps concurrency at. `None` means "no cap -- whatever the
# user asked for, they get".
WORKER_CAP: Dict[str, Optional[int]] = {
    "cpu": None,
    "gpu_one_model": 2,
    "gpu_model_each": 1,
}


def worker_cap(gpu_use: Optional[str]) -> Optional[int]:
    """
    Most files a step may work on at once, or None for no limit.

    An unrecognized or missing value is treated as ``"gpu_model_each"`` -- one
    file at a time. Wrong-but-slow is a recoverable mistake; wrong-and-out-of-
    memory is not, and it fails halfway through a batch rather than at the start.
    """
    if gpu_use not in WORKER_CAP:
        return WORKER_CAP["gpu_model_each"]
    return WORKER_CAP[gpu_use]


# ---------------------------------------------------------------------------
# Rule 1: which device
# ---------------------------------------------------------------------------

DEVICE_CHOICES = ("auto", "cuda", "cpu")
DEVICE_DEFAULT = "auto"
DEVICE_HELP = ('Where to run the model: "auto" (default) uses the GPU when it '
               'is genuinely usable and falls back to the CPU when it is not; '
               '"cuda" insists, and fails if the GPU cannot be used; "cpu" '
               'never touches the GPU.')

CPU_FALLBACK_NOTE = (
    "A GPU that reports itself as present is not always a GPU that works. "
    "CTranslate2 and torch load their CUDA libraries lazily, so a missing "
    "cuBLAS or cuDNN surfaces at the first inference rather than when the "
    "model is built -- and CTranslate2 does not borrow torch's copies of them, "
    "so torch working here proves nothing about faster-whisper. "
    'Pass device="cuda" to turn this fallback into an error.'
)


def _cuda_visible(backend: str) -> Tuple[bool, Optional[str]]:
    """
    Whether `backend` can see a CUDA device, and why not when it cannot.

    The backend matters and is not an inconsistency to be tidied away: torch and
    CTranslate2 ship their own CUDA libraries and can genuinely disagree about
    the same card. A module built on faster-whisper must ask CTranslate2; one
    built on torch or sentence-transformers must ask torch. Asking the wrong one
    is how a run ends up believing in a GPU it cannot use.
    """
    if backend == "ctranslate2":
        try:
            import ctranslate2
        except Exception as e:
            return False, f"CTranslate2 would not import ({e})"
        try:
            return ctranslate2.get_cuda_device_count() > 0, None
        except Exception as e:
            return False, f"the CUDA query itself failed ({e})"

    try:
        import torch
    except Exception as e:
        return False, f"torch would not import ({e})"
    try:
        return bool(torch.cuda.is_available()), None
    except Exception as e:
        return False, f"the CUDA query itself failed ({e})"


def resolve_device(
    device: Optional[str],
    *,
    backend: str = "torch",
    probe: Optional[Callable[[], None]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Turn a device preference into the device that will actually be used.

    Parameters
    ----------
    device : {"auto", "cuda", "cpu"} | None
        What the caller asked for. ``None`` and ``""`` mean ``"auto"``.
    backend : {"torch", "ctranslate2"}, default "torch"
        Which library's view of CUDA to trust. See :func:`_cuda_visible`.
    probe : callable, optional
        Runs one real inference on the chosen device. Supply this wherever it is
        cheap, because "a device is visible" and "a device works" are different
        claims: both backends load cuBLAS/cuDNN at the first inference, so a
        broken install builds a model without complaint and fails later. Under
        ``"auto"`` a raising probe means the CPU; under ``"cuda"`` it is an error.

    Returns
    -------
    tuple[str, str | None]
        The resolved device, and a human-readable reason when ``"auto"`` chose
        the CPU over an apparently-present GPU. The reason is ``None`` when the
        caller named a device, and when CUDA was chosen -- neither needs
        explaining.

    Notes
    -----
    Naming ``"cuda"`` is an instruction, not a preference: it is returned as
    asked and any probe failure is raised. Silently doing something slower to
    the one person who has said they care about the GPU is worse than failing.
    """
    asked = (device or "auto").strip().lower() or "auto"
    if asked not in DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of {DEVICE_CHOICES}, not {device!r}"
        )

    if asked == "cpu":
        # no probe. the probe exists to find out whether the GPU really works,
        # and somebody who asked for the CPU has said they do not care -- every
        # caller hands us a *CUDA* probe, so running it here asks a question
        # nobody wanted the answer to. on a CPU-only build of torch it does not
        # merely waste time, it raises "Torch not compiled with CUDA enabled"
        # and takes the run down. found by CI; invisible on a machine whose
        # torch happens to have CUDA in it, which is why it survived this long.
        return "cpu", None

    if asked == "cuda":
        if probe is not None:
            probe()
        return "cuda", None

    visible, why = _cuda_visible(backend)
    if not visible:
        detail = f" ({why})" if why else ""
        return "cpu", f"{backend} reports no usable CUDA device{detail}. {CPU_FALLBACK_NOTE}"

    if probe is None:
        return "cuda", None
    try:
        probe()
    except Exception as e:
        return "cpu", (f"the GPU is visible but {backend} cannot use it: {e}. "
                       f"{CPU_FALLBACK_NOTE}")
    return "cuda", None


def device_note(doing: str, device_name: str, reason: Optional[str] = None) -> str:
    """
    One line naming the device a step is about to work on.

    Steps used to say this with a `print` behind `verbose`, which the app
    never turns on -- so a run that quietly took the CPU looked exactly like
    one on the card until somebody noticed it had been going for an hour.
    This goes to `on_progress` instead, where the run display shows it for as
    long as the work lasts. The reason rides along when `auto` had to fall
    back, because "on the CPU" invites the question this already answers.
    """
    where = "the GPU" if str(device_name).startswith("cuda") else "the CPU"
    return f"{doing} on {where}" + (f" -- {reason}" if reason else "")


def add_device_argument(parser, *, default: str = DEVICE_DEFAULT) -> None:
    """
    Give a module CLI the same ``--device`` flag as every other module CLI.

    Centralized so the choices, the default and the help text cannot drift
    apart between modules -- someone who has read one ``--device`` has read all
    of them.
    """
    parser.add_argument("--device", default=default, choices=DEVICE_CHOICES,
                        help=DEVICE_HELP)


# ---------------------------------------------------------------------------
# The CUDA math libraries, which are a separate question from the card
# ---------------------------------------------------------------------------

# what CTranslate2 needs beyond the driver, by platform. the driver
# (`nvcuda.dll` / `libcuda.so`) is what answers "is there a card", and it's
# always there on a machine that has one -- so counting devices tells us very
# little. these are what the *math* actually runs on; they ship separately
# from CTranslate2 on Windows, and they don't load until the first inference.
# put those together and a broken install looks perfectly healthy right up
# until we hit the first file.
_CUDA_LIBRARIES = {
    "Windows": ("cublas64_12.dll", "cublasLt64_12.dll"),
    "Linux": ("libcublas.so.12", "libcublasLt.so.12"),
}

# where the wheels put them. torch bundles its own copies, which is usually
# the only reason a Windows machine has them at all -- and also the usual
# reason CTranslate2 can't find them, since nothing puts torch's lib directory
# on the search path unless torch itself has been imported.
_LIBRARY_HOMES = (
    ("torch", "lib"),
    ("nvidia.cublas", "lib"),
    ("nvidia.cudnn", "lib"),
)


def _candidate_dirs() -> List[Path]:
    """Every installed package directory that might hold CUDA libraries."""
    found: List[Path] = []
    for module_name, subdir in _LIBRARY_HOMES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            paths = list(getattr(module, "__path__", []) or [])
            origin = f"{paths[0]}/__init__.py" if paths else None
        if not origin:
            continue
        candidate = Path(origin).parent / subdir
        if candidate.is_dir():
            found.append(candidate)
    return found


def _library_dirs(names: Sequence[str] = ()) -> List[Path]:
    """
    Directories that actually contain one of `names`.

    Containment, not existence. An earlier version returned any directory that
    was merely present, and so told a user "a copy appears to be in
    <torch>/lib" about a machine where that folder held no such file -- their
    torch was a CUDA 13 build shipping ``cublas64_13.dll`` while CTranslate2
    wanted ``cublas64_12.dll``. Pointing the loader at that folder was a no-op
    reported as a fix, which is the worst thing a diagnostic can do.

    With no names given, every candidate directory comes back -- for callers
    that want to look around rather than ask a question.
    """
    dirs = _candidate_dirs()
    if not names:
        return dirs
    return [d for d in dirs if any((d / name).exists() for name in names)]


def _nearby_versions(names: Sequence[str]) -> Dict[str, List[Path]]:
    """
    The same libraries at a *different* CUDA major, and where they are.

    This is the diagnosis worth reaching for when the wanted file is nowhere: a
    machine with ``cublas64_13.dll`` and none of ``cublas64_12.dll`` does not
    have a search-path problem, it has a version problem, and the two need
    opposite advice. Telling someone to fix their path when their torch is
    simply built against the wrong CUDA sends them in a circle.
    """
    import re

    out: Dict[str, List[Path]] = {}
    for name in names:
        # cublas64_12.dll -> cublas64_*.dll;  libcublas.so.12 -> libcublas.so.*
        pattern = re.sub(r"(?<=[._])\d+(?=(\.|$))", "*", name, count=1)
        if pattern == name:
            continue
        for directory in _candidate_dirs():
            for hit in sorted(directory.glob(pattern)):
                if hit.name != name:
                    out.setdefault(hit.name, []).append(directory)
    return out


def _can_load(name: str) -> bool:
    """Whether the dynamic loader can find and load one library by name."""
    import ctypes

    loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
    try:
        loader(name)
        return True
    except OSError:
        return False


def _import_ctranslate2() -> bool:
    """
    Import CTranslate2, so that a later library check asks a fair question.

    Without this the check is pessimistic to the point of being wrong. On Linux
    the math libraries usually come from the `nvidia-*` wheels and are found
    through the RPATH baked into CTranslate2's own extension module -- not
    through any path the loader searches by default. A bare `CDLL("libcublas
    .so.12")` therefore fails on a machine where transcription runs on the GPU
    perfectly well, which is a worse diagnostic than none at all.

    Importing first puts CTranslate2's dependencies in the process, after which
    asking for them by name answers the question actually worth asking: can the
    library CTranslate2 will use be loaded?
    """
    try:
        importlib.import_module("ctranslate2")
        return True
    except Exception:
        return False


def cuda_libraries_ok() -> Tuple[Optional[bool], str]:
    """
    Whether the CUDA math libraries are loadable, and what to say about it.

    This is the question ``get_cuda_device_count()`` does not answer, and the
    reason a run can report a healthy GPU and then fail at its first inference
    with ``Library cublas64_12.dll is not found or cannot be loaded``.

    Returns
    -------
    tuple[bool | None, str]
        ``None`` on a platform where this is not a separate concern (macOS, or
        anything not listed in :data:`_CUDA_LIBRARIES`) -- meaning "no opinion",
        not "broken".
    """
    wanted = _CUDA_LIBRARIES.get(platform.system())
    if not wanted:
        return None, "not applicable on this platform"
    if not _import_ctranslate2():
        return None, "ctranslate2 is not installed, so nothing needs them yet"

    missing = [name for name in wanted if not _can_load(name)]
    if not missing:
        return True, f"{wanted[0]} and friends load"

    # three different problems all wearing the same error message, and each one
    # needs different advice. telling them apart is most of the point here.
    homes = _library_dirs(missing)
    if homes:
        return False, (f"cannot load {missing[0]}, though a copy is in {homes[0]}. "
                       f"Nothing adds that folder to the library search path on "
                       f"its own.")

    nearby = _nearby_versions(missing)
    if nearby:
        other, where = sorted(nearby.items())[0]
        return False, (f"cannot load {missing[0]}, and nothing installed ships it. "
                       f"What is installed is {other} (in {where[0]}), which is a "
                       f"different CUDA major version -- so this is a version "
                       f"mismatch rather than a search-path problem, and pointing "
                       f"the loader at that folder would not help.")

    return False, (f"cannot load {missing[0]}, and no installed package appears "
                   f"to ship it")


def ensure_cuda_libraries() -> bool:
    """
    Make the CUDA math libraries loadable, if a copy can be found.

    They are usually present and merely unreachable: torch bundles them, but
    nothing puts torch's ``lib`` directory on the library search path, so
    CTranslate2 -- which does not depend on torch -- cannot see them. Pointing
    the loader at that directory costs nothing and is the difference between a
    GPU run and a CPU one.

    Two mechanisms, because the platforms differ in what is possible after the
    process has started. Windows has ``os.add_dll_directory``, which extends the
    search path for later loads. Linux reads ``LD_LIBRARY_PATH`` only at exec,
    so there is nothing to extend -- but loading a library by absolute path puts
    it in the process, and a later request for it by soname finds it already
    resident.

    Returns
    -------
    bool
        Whether the libraries are loadable now. False means they were not found,
        not that anything was broken by trying.
    """
    ok, _detail = cuda_libraries_ok()
    if ok is not False:
        return bool(ok)

    wanted = _CUDA_LIBRARIES.get(platform.system(), ())
    windows = platform.system() == "Windows"

    # only directories that actually hold one of the files we want. adding
    # anything else does nothing, and we'd still report it as an attempt.
    for directory in _library_dirs(wanted):
        if windows:
            try:
                # we hold this for the life of the process on purpose: releasing
                # the handle would undo the very thing we're doing here.
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
            except OSError:
                continue
        else:
            import ctypes

            for name in wanted:
                candidate = directory / name
                if candidate.exists():
                    try:
                        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    return bool(cuda_libraries_ok()[0])


# Windows DLL-directory handles, which we keep alive on purpose: they're
# context managers, and exiting one removes the directory again.
_DLL_DIRECTORIES: List[Any] = []


@contextlib.contextmanager
def hidden_gpu(hide: bool = True):
    """
    Hide the GPU from libraries that offer no way to ask for the CPU.

    Some third-party constructors pick a device themselves --
    ``ArchetypeQuantifier`` builds a ``SentenceTransformer`` internally and
    takes no ``device`` -- so the only way to honor ``device="cpu"`` is to make
    the GPU invisible while they run.

    Works by setting ``CUDA_VISIBLE_DEVICES`` to empty, which torch reads when
    it first initializes CUDA. That is the limitation worth knowing: in a
    process that has *already* initialized CUDA this cannot take effect, so it
    is a request rather than a guarantee. It is restored on the way out either
    way.
    """
    if not hide:
        yield
        return

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
