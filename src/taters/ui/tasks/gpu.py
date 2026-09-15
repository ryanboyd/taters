"""
"Check my setup" -- what this machine is, and what Taters can actually use.

The question people really have is never "is CUDA available". It is one of:

* do I need to install something,
* or is my hardware simply not going to work,
* or is it fine and the slowness is something else?

Those need opposite advice, and ``torch.cuda.is_available()`` collapses all
three into a single ``False``. So each layer is reported separately, and the
advice at the end depends on *which* layer is missing.

The report doubles as something to paste into a bug report, which is why it
names versions rather than just saying "installed".

Nothing here imports torch at module scope. On a machine without it that would
be an error, and "torch is not installed" is one of the answers this exists to
give. Package *versions* are read from installed metadata rather than by
importing, which is both faster and safe for packages that would fail to load.
"""

from __future__ import annotations

import platform
import re
import shutil
import sys
import subprocess
import textwrap
from typing import List, NamedTuple, Optional, Sequence, Tuple

from ...helpers.gpu import cuda_libraries_ok
from ..prompts import terminal_width
from . import Task, TaskContext

__all__ = ["TASK", "probe", "Finding", "Section", "reinstall_command", "cuda_tag",
           "unavailable_here"]

#: Advice lines starting with one of these are shown verbatim, unwrapped, so
#: they can be copied and pasted whole.
COMMAND_PREFIXES = ("pip ", "python ", "conda ", "uv ")


class Finding(NamedTuple):
    """One row: what it is, whether it works, and the detail."""

    layer: str
    ok: Optional[bool]      # True / False / None ("just information")
    detail: str = ""


class Section(NamedTuple):
    """A group of findings under a heading."""

    title: str
    findings: List[Finding]


# distribution name -> (what it's for, whether its absence is a problem). we
# read from metadata so a package that's installed but broken still shows a
# version -- which is exactly the state worth seeing in a bug report.
_PACKAGES = [
    ("torch", "sentence embeddings, diarization", False),
    ("ctranslate2", "transcription engine", True),
    ("faster-whisper", "transcription", True),
    ("transformers", "model loading", True),
    ("sentence-transformers", "sentence embeddings, archetypes", True),
    ("textstat", "readability", True),
    ("praat-parselmouth", "vocal acoustics", False),
    ("disvoice", "vocal acoustics", False),
    ("nemo-toolkit", "speaker diarization", False),
    ("gensim", "training word vectors", False),
]


def _version(dist: str) -> Optional[str]:
    """The installed version of a distribution, or None if it is absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _nvidia_smi() -> Optional[str]:
    """The driver's own description of the first GPU, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # old drivers reject `compute_cap` as a field -- and print the complaint to
    # stdout, which then showed up in the setup report as the GPU's name. so
    # only a clean exit with the four fields we asked for counts as an answer.
    if out.returncode != 0:
        return None
    line = out.stdout.strip()
    if not line or line.count(",") != 3:
        return None
    return line


def _driver_cuda_ceiling() -> Optional[str]:
    """
    The highest CUDA version the installed driver supports.

    Worth reporting on its own: a driver too old for the PyTorch build is a
    common failure that looks identical to "no GPU" from inside Python.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        if "CUDA Version:" in line:
            return line.split("CUDA Version:")[1].strip().split()[0]
    return None


# the newest CUDA the whole stack can use, not the newest that exists.
#
# the driver's ceiling is an upper bound on what will *run*, and following it
# was a mistake: we recommended a cu132 PyTorch on a machine whose driver
# reported CUDA 13.2, and that install shipped CUDA 13's cuBLAS -- which
# ctranslate2 (built against CUDA 12, and pulled in as a dependency nobody
# chose) then couldn't load. the checker went on to report that as the user's
# problem to solve. it was ours: we recommended the version.
#
# so we cap the recommendation at what the rest of the stack can live with.
# raise this when ctranslate2 ships CUDA 13 wheels, and not before.
MAX_CUDA = (12, 8)

# the wheel indexes PyTorch actually publishes (download.pytorch.org/whl/cuXYZ).
# we only ever recommend one of these -- see `cuda_tag`.
KNOWN_WHEEL_TAGS = ((11, 8), (12, 1), (12, 4), (12, 6), (12, 8))


def cuda_tag(driver_ceiling: Optional[str]) -> str:
    """
    The PyTorch wheel index to recommend.

    The lower of what the driver supports and what the rest of Taters can use.
    ``13.2`` -> ``cu128``, because ctranslate2 needs CUDA 12; ``12.4`` ->
    ``cu124``, because the driver is then the tighter constraint.

    Clamped to indexes PyTorch actually publishes. Deriving the tag from the
    driver ceiling alone produced advice like `cu122` -- a version that exists
    as a CUDA release but not as a wheel index, so the recommended install
    command 404'd. The table needs a new row when PyTorch adds an index, which
    is rare and visible; a fabricated URL is neither.
    """
    fallback = f"cu{MAX_CUDA[0]}{MAX_CUDA[1]}"
    if not driver_ceiling:
        return fallback
    parts = driver_ceiling.split(".")
    try:
        driver = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return fallback
    usable = [t for t in KNOWN_WHEEL_TAGS if t <= min(driver, MAX_CUDA)]
    if not usable:
        # a driver older than the oldest published index can't run any CUDA
        # wheel we could name; the CPU build is the honest recommendation.
        return "cpu"
    major, minor = max(usable)
    return f"cu{major}{minor}"


def reinstall_command(driver_ceiling: Optional[str]) -> str:
    """
    The exact line to run, force-reinstall included.

    The flag is the whole point. `pip install torch --index-url <cuda index>`
    looks correct and does nothing at all when a torch is already present:
    the requirement is already satisfied, so pip never consults the index. It
    reports success, and the CPU build stays exactly where it was.
    """
    return (
        "pip install --force-reinstall --no-cache-dir torch torchvision "
        f"--index-url https://download.pytorch.org/whl/{cuda_tag(driver_ceiling)}"
    )


def cards_without_kernels(capabilities: Sequence[Tuple[int, int]],
                          arch_list: Sequence[str]) -> List[str]:
    """
    The cards this torch build has no kernels for, as ``sm_120`` strings.

    A CUDA build only carries kernels for the architectures it was compiled
    for, and `torch.cuda.is_available()` does not check: a Blackwell card
    (sm_120) under a cu126 build reports itself present, names itself
    correctly, and then fails at the first real operation with "no kernel
    image is available". So the setup check asks the build what it was
    compiled for and compares. Empty means every card here is covered.
    """
    have = {str(a).strip() for a in arch_list}
    missing = []
    for major, minor in capabilities:
        tag = f"sm_{major}{minor}"
        if tag not in have:
            missing.append(tag)
    return missing


def _is_cuda_build(version: Optional[str]) -> bool:
    """Whether a torch version string names a CUDA build (`2.13.0+cu132`)."""
    return bool(version) and "+cu" in version


def _system_section() -> Section:
    from ...helpers.parallel_map import auto_workers, max_workers

    impl = platform.python_implementation()
    # the worker ceiling belongs here because this screen is where someone goes
    # to learn what their machine can do -- and "how high can I set workers?"
    # is exactly that question.
    cores = max_workers()
    return Section("System", [
        Finding("Taters", None, _version("taters") or "running from source"),
        Finding("Python", None, f"{platform.python_version()} ({impl})"),
        Finding("Operating system", None,
                f"{platform.system()} {platform.release()} ({platform.machine()})"),
        Finding("Processor", None,
                f"{cores} logical cores  ·  workers can be 1–{cores}; "
                f"automatic uses {auto_workers()}"),
        # where every transformer, embedding and Whisper model lands. it's the
        # first thing a shared-server user asks, so let's answer it where they
        # look.
        Finding("Downloaded models", None, _model_cache_line()),
    ])


def _model_cache_line() -> str:
    from ...helpers.settings import describe_model_cache

    try:
        return describe_model_cache()
    except Exception as e:      # a listing must never take the check down
        return f"could not be read ({e})"


def _hardware_section(smi: Optional[str]) -> Tuple[Section, bool, bool]:
    """Returns the section plus (has_nvidia, is_apple_silicon)."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        name = parts[0] if parts else "unknown"
        rows = [Finding("Graphics card", True, name)]
        if len(parts) > 1:
            # compute capability decides which CUDA builds will run at all --
            # sm_120 (Blackwell) needs cu128 or newer, and older wheels fail
            # with an error that names neither.
            rows.append(Finding("Compute capability", None,
                                f"{parts[1]}  (sm_{parts[1].replace('.', '')})"))
        if len(parts) > 2:
            ceiling = _driver_cuda_ceiling()
            detail = parts[2] + (f"  (supports CUDA up to {ceiling})" if ceiling else "")
            rows.append(Finding("Driver", None, detail))
        if len(parts) > 3:
            rows.append(Finding("GPU memory", None, parts[3]))
        return Section("Hardware", rows), True, False

    if apple:
        return Section("Hardware", [
            Finding("Graphics", True, "Apple silicon (Metal)"),
        ]), False, True

    return Section("Hardware", [
        Finding("Graphics card", False, "no NVIDIA driver found (nvidia-smi is absent)"),
    ]), False, False


def unavailable_here(dist: str) -> bool:
    """
    Whether Taters deliberately does not ask for `dist` on this Python.

    Read from our own metadata rather than from a table kept alongside it. A
    requirement carrying a marker like ``python_version < "3.14"`` is a
    statement that the package cannot be installed here, and asking the
    metadata means this answer cannot drift away from what pip actually did.

    NeMo is the case that prompted it: every 2.7.x `[asr]` needs
    `nv-one-logger-pytorch-lightning-integration`, capped at Python 3.13, and
    every 2.6.x needs `numpy<2`, which has no wheels past 3.13 either. On 3.14
    there is no version to install, so the extra does not ask for one -- and
    "not installed" alone would leave someone trying to fix it.
    """
    try:
        from importlib.metadata import requires
        from packaging.requirements import Requirement
    except Exception:
        return False

    try:
        declared = requires("taters") or []
    except Exception:
        # running from a source checkout that was never installed: there's no
        # distribution to ask. "don't know" mustn't crash the checker whose
        # whole job is telling people what's wrong.
        return False

    for raw in declared:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name.lower().replace("_", "-") != dist.lower().replace("_", "-"):
            continue
        marker = req.marker
        if marker is None:
            return False
        # `extra` has to be supplied or the marker can't be evaluated at all,
        # and it has to be the extra this line actually belongs to. we used to
        # try "all" and "diarization" only, which worked right up until the
        # first line pip listed for a package sat under [standard] or
        # [vocalacoustics] -- then `extra == "standard"` was false for both of
        # our guesses and a perfectly installable package got reported as
        # impossible on this Python. so we read the extra out of the marker
        # itself; a marker with no extra clause gets the empty one
        names = set(re.findall(r"""extra\s*==\s*['"]([^'"]*)['"]""", str(marker)))
        if any(marker.evaluate({"extra": name}) for name in names or {""}):
            return False
        return True
    return False


def _packages_section() -> Tuple[Section, Optional[str]]:
    """Every package that matters, with its version. Returns (section, torch_version)."""
    rows: List[Finding] = []
    torch_version = None
    for dist, purpose, required in _PACKAGES:
        found = _version(dist)
        if dist == "torch":
            torch_version = found
        if found:
            note = purpose
            if dist == "torch" and "+cpu" in found:
                note = f"{purpose} — CPU-only build"
            rows.append(Finding(dist, True, f"{found}  ·  {note}"))
        elif unavailable_here(dist):
            # not a problem to solve, and saying "not installed" would invite
            # someone to try. there's no version of it for this Python.
            rows.append(Finding(
                dist, None,
                f"not available for Python {sys.version_info.major}."
                f"{sys.version_info.minor}  ·  {purpose}"))
        else:
            rows.append(Finding(dist, None if not required else False,
                                f"not installed  ·  {purpose}"))
    return Section("Installed packages", rows), torch_version


def probe() -> Tuple[List[Section], List[str]]:
    """
    Inspect the machine, layer by layer.

    Returns
    -------
    (sections, advice)
        ``sections`` is what is true, grouped for display. ``advice`` is what to
        do about it -- and says so plainly when the answer is "nothing".
    """
    advice: List[str] = []
    smi = _nvidia_smi()

    system = _system_section()
    hardware, has_nvidia, apple = _hardware_section(smi)
    packages, torch_version = _packages_section()

    # what the runtime can actually reach. this is the layer people mean, and
    # it means nothing without the two above it.
    runtime: List[Finding] = []

    if torch_version is None:
        runtime.append(Finding("PyTorch", False, "not installed"))
        if has_nvidia:
            advice.append(
                "You have a working NVIDIA driver but no PyTorch. Install a CUDA "
                "build to use the card — see the install guide's 'Choose your "
                "PyTorch build' section."
            )
        elif apple:
            advice.append("Install PyTorch (`pip install torch`) to use the Apple GPU.")
        else:
            advice.append(
                "No GPU was found, so PyTorch is optional. Everything still runs "
                "on the CPU, just more slowly."
            )
    else:
        try:
            import torch
        except Exception as e:
            runtime.append(Finding("PyTorch", False, f"installed but will not load ({e})"))
            advice.append(
                "PyTorch is installed but cannot be imported. That usually means a "
                "build that does not match this machine; reinstall it."
            )
        else:
            if torch.cuda.is_available():
                names = [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())]
                # "available" is not "usable": see cards_without_kernels. this
                # was a false all-clear -- the card was listed, the run fell
                # back to the CPU anyway, and nothing joined the two up
                try:
                    missing = cards_without_kernels(
                        [torch.cuda.get_device_capability(i)
                         for i in range(torch.cuda.device_count())],
                        torch.cuda.get_arch_list())
                except Exception:
                    missing = []
                if missing:
                    runtime.append(Finding(
                        "PyTorch can use the GPU", False,
                        f"{', '.join(names)} is {', '.join(missing)}, which this "
                        f"build has no kernels for — it will run on the CPU"))
                    advice.append(
                        f"PyTorch {torch_version} was not built for this card "
                        f"({', '.join(missing)}); it sees the card and then has "
                        f"nothing to run on it. A newer CUDA build is needed "
                        f"(sm_120 wants cu128 or later). Run:")
                    advice.append(reinstall_command(_driver_cuda_ceiling()))
                else:
                    runtime.append(Finding("PyTorch can use the GPU", True,
                                           f"{len(names)} device(s): {', '.join(names)}"))
            elif getattr(getattr(torch, "backends", None), "mps", None) and \
                    torch.backends.mps.is_available():
                runtime.append(Finding("PyTorch can use the GPU", True, "Apple Metal (MPS)"))
            else:
                runtime.append(Finding("PyTorch can use the GPU", False,
                                       "it will run on the CPU"))
                if has_nvidia and not _is_cuda_build(torch_version):
                    # the specific, common case: the installed build has no
                    # CUDA in it at all. saying so beats "check your setup",
                    # and the force-reinstall is the part people leave off.
                    advice.append(
                        f"PyTorch {torch_version} has no CUDA in it — a version "
                        "with GPU support is numbered like 2.13.0+cu132. Run:"
                    )
                    advice.append(reinstall_command(_driver_cuda_ceiling()))
                    advice.append(
                        "--force-reinstall matters: without it pip sees a torch "
                        "already installed, decides there is nothing to do, and "
                        "never looks at the CUDA index. It reports success and "
                        "changes nothing."
                    )
                elif has_nvidia:
                    advice.append(
                        f"PyTorch {torch_version} is a CUDA build but cannot reach "
                        "the card. Usually its CUDA version is newer than the "
                        "driver supports. Run:"
                    )
                    advice.append(reinstall_command(_driver_cuda_ceiling()))

    # transcription goes through ctranslate2, not torch, so it can succeed or
    # fail on its own. reporting torch alone would mislead the person whose
    # actual complaint is that transcription is slow.
    if _version("ctranslate2") is None:
        runtime.append(Finding("Transcription on GPU", False, "ctranslate2 not installed"))
        advice.append("Reinstall Taters: ctranslate2 ships with the base install.")
    else:
        try:
            import ctranslate2
            count = ctranslate2.get_cuda_device_count()
        except Exception:
            count = 0

        # counting devices is the same too-shallow question that once let a run
        # believe in a GPU it couldn't use. the driver answers it, the driver is
        # always there on a machine with a card, and the math libraries --
        # which do the actual work, and which CTranslate2 only loads at the
        # first inference -- are a separate matter entirely. reporting "1 CUDA
        # device available" and then failing on the first file is worse than
        # reporting nothing, because it sends someone looking anywhere but here.
        libs_ok, libs_detail = cuda_libraries_ok()
        usable = bool(count) and libs_ok is not False

        runtime.append(Finding(
            "Transcription on GPU", usable,
            f"{count} CUDA device(s) available" if usable
            else ("will transcribe on the CPU" if not count
                  else "the card is visible but the CUDA math libraries are not "
                       "loadable, so this will fall back to the CPU")))

        if count and libs_ok is False:
            # one sentence and a command. this used to be four paragraphs about
            # the same problem, which read as four problems and left the reader
            # no better off -- it explained CTranslate2's library loading to
            # someone who never installed CTranslate2 and sent them off to find
            # a CUDA runtime. whatever's true about the internals, the only
            # thing worth saying here is what to type.
            if "version mismatch" in libs_detail:
                advice.append(
                    "Transcription cannot use your GPU: the PyTorch installed "
                    "here brings a newer CUDA than the transcription engine can "
                    "use. Reinstall PyTorch on the version they share:"
                )
                advice.append(reinstall_command(_driver_cuda_ceiling()))
                advice.append(
                    "--force-reinstall matters: without it pip sees a torch "
                    "already installed, decides there is nothing to do, and "
                    "never looks at the index."
                )
            elif "search path" in libs_detail:
                advice.append(
                    "Transcription may not be using your GPU. Taters points the "
                    "loader at the right folder before it loads a model, so runs "
                    "are probably fine even though this check says otherwise -- "
                    "start a transcription and see whether it mentions falling "
                    "back to the CPU."
                )
            else:
                advice.append(
                    "Transcription cannot use your GPU: the CUDA libraries it "
                    "needs are not installed. A CUDA build of PyTorch brings "
                    "them:"
                )
                advice.append(reinstall_command(_driver_cuda_ceiling()))

    sections = [system, hardware, packages, Section("What Taters can use", runtime)]
    if not advice:
        advice.append("Everything checks out. Taters will use the GPU where it can.")
    return sections, advice


def _verdict(ok: Optional[bool]) -> str:
    """
    ``yes`` / ``no`` / nothing at all, styled by how much it matters.

    ``None`` is "just information" -- a driver version, a GPU's memory -- and
    gets no word, because a blank cell says "nothing to decide here" more
    clearly than a third label would.
    """
    if ok is None:
        return ""
    return "[green]yes[/green]" if ok else "[bold red]no[/bold red]"


def _run(ctx: TaskContext) -> None:
    prompter = ctx.prompter
    prompter.note("\n  Checking what this machine can do…", style="cyan")

    sections, advice = probe()

    for section in sections:
        # color carries the verdict, so a row that needs attention jumps out
        # before it's read. a report where "no" looks exactly like "yes" makes
        # someone parse every line to find the one that matters -- and this
        # screen exists precisely for the person who doesn't know which line
        # that is.
        rows = [[f.layer, _verdict(f.ok), f.detail] for f in section.findings]
        prompter.table(section.title, rows, ["", "", ""])

    good = len(advice) == 1 and advice[0].startswith("Everything")
    for line in advice:
        prompter.note("")
        if line.startswith(COMMAND_PREFIXES):
            # a command has to survive being copied. folded across two lines it
            # pastes as two commands, the second of which is a bare URL.
            prompter.note(f"    {line}", style="cyan", wrap=False)
            continue
        # we wrap the prose by hand so the continuation keeps its indent. left
        # to the terminal, the second line starts hard against the left margin
        # and stops reading as part of the same paragraph.
        for wrapped in textwrap.wrap(line, width=max(40, terminal_width() - 6)):
            prompter.note(f"  {wrapped}", style="green" if good else "yellow")

    prompter.note("")
    # a pause, not a question. this screen only reports -- there's nothing to
    # decide and nothing to cancel -- and the "Done? yes/no" that used to sit
    # here went back to the same menu either way, while answering "no" printed
    # "Backed out. Nothing was changed." over a report that had changed nothing
    # in the first place.
    prompter.pause()


TASK = Task(
    id="gpu",
    label="Check my setup",
    help="Hardware, versions, and whether Taters can use your graphics card",
    run=_run,
)
