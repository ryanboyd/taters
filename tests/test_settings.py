"""
Tests for the settings submenu and the setup check behind it.

The setup check exists because `torch.cuda.is_available()` collapses three very
different situations into one `False`: you need to install something, your
hardware cannot work, or it is fine and the slowness is something else. Each
needs different advice, so each layer is probed and reported separately.
"""

from __future__ import annotations

import builtins
import sys
from unittest import mock


from taters.ui import hub
from taters.ui.prompts import ScriptedPrompter
from taters.ui.tasks import TaskContext, all_tasks
import pytest

from taters.ui.prompts import PAUSE_MESSAGE
from taters.ui.tasks import gpu, settings


def ctx_for(prompter, cwd):
    from pathlib import Path
    return TaskContext(prompter=prompter, cwd=Path(cwd))


def probe_with(*, smi, blocked=()):
    """
    Run the probe against a pretended machine.

    `blocked` names distributions that should look absent. Versions are read
    from installed metadata rather than by importing, so simulating absence
    means intercepting the metadata lookup *and* the import -- the report reads
    the first, the runtime checks use the second.

    Blocked modules are put back afterwards. Leaving `torch` out of
    `sys.modules` poisons every later test in the file: re-importing it in one
    process is not free, and with CUDA already initialized it can fail outright.
    """
    real_import = builtins.__import__
    real_version = gpu._version
    saved = {name: sys.modules[name] for name in blocked if name in sys.modules}

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    def fake_version(dist):
        return None if dist in blocked else real_version(dist)

    try:
        with mock.patch.object(gpu, "_nvidia_smi", lambda: smi), \
             mock.patch.object(gpu, "_version", fake_version), \
             mock.patch.object(builtins, "__import__", fake_import):
            for module in blocked:
                sys.modules.pop(module, None)
            return gpu.probe()
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------

def test_the_front_page_keeps_the_doing_verbs_in_order(tmp_path):
    """
    Housekeeping alongside the doing-verbs gave equal weight to tidying up
    and to the point of the program -- and the data row opens the list,
    because getting the data into shape is where every project starts (the
    user asked for it by position). That row holds both things you do to a
    spreadsheet you already have, wrangling and analyzing, which were two
    front-page rows and read as two unrelated errands. Extraction is two
    entries, not one: extracting features and answering a question about them
    are different intentions, and splitting them lets the second flow assume
    there is something to analyze.
    """
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=tmp_path)

    offered = [c.value for c in p.offered_choices("What would you like to do?")]
    assert offered == ["data", "extract", "extract_analyze", "train",
                       "run_saved", "settings", "quit"]


def test_managing_pipelines_moved_under_settings(tmp_path):
    from taters.ui.tasks import manage_data

    assert "manage" not in {t.id for t in all_tasks()}
    # one level further down than it used to be: Settings keeps three rows,
    # and everything Taters stores shares the middle one
    assert "manage" in {t.id for t in manage_data.entries()}
    assert [t.id for t in settings.entries()] == ["gpu", "manage_data",
                                                  "hashbrowns"]


def test_settings_returns_to_the_main_menu(tmp_path):
    p = ScriptedPrompter(["settings", "back", "quit"])
    hub.run_hub(p, cwd=tmp_path)

    assert [q for _, q in p.asked] == [
        "What would you like to do?", "Settings and tools", "Main menu",
    ]


def test_backing_out_of_a_tool_stays_in_settings(tmp_path, monkeypatch):
    """A mistyped path should cost the tool, not the whole detour."""
    from taters.ui.prompts import Cancelled
    from taters.ui.tasks import Task

    def explode(ctx):
        raise Cancelled()

    monkeypatch.setattr(settings, "entries",
                        lambda: [Task(id="t", label="T", help="", run=explode)])
    p = ScriptedPrompter(["t", "back"])
    settings._run(ctx_for(p, tmp_path))

    assert [q for _, q in p.asked] == ["Settings and tools", "Settings and tools"]


# ---------------------------------------------------------------------------
# the setup check
# ---------------------------------------------------------------------------

def flatten(sections):
    return [f for section in sections for f in section.findings]


def test_a_working_machine_says_so_plainly():
    sections, advice = gpu.probe()
    assert sections, "the probe reported nothing at all"
    assert advice, "silence is not an answer"


def test_the_report_names_the_machine_and_the_versions():
    """
    It doubles as something to paste into a bug report, so "installed" is not
    enough -- which version, on what, matters.
    """
    sections, _advice = gpu.probe()
    titles = [s.title for s in sections]
    assert titles == ["System", "Hardware", "Installed packages", "What Taters can use"]

    system = {f.layer: f.detail for f in sections[0].findings}
    assert system["Python"].startswith(("3.", "4."))
    assert system["Operating system"]

    packages = {f.layer: f.detail for f in sections[2].findings}
    assert "ctranslate2" in packages
    # a package that's there reports its version, not just a tick
    assert any(char.isdigit() for char in packages["ctranslate2"])


def test_a_missing_package_says_what_it_would_have_enabled():
    """
    "nemo-toolkit: not installed" is only useful if you know that means
    diarization.
    """
    sections, _advice = gpu.probe()
    packages = {f.layer: f.detail for f in sections[2].findings}
    for name, detail in packages.items():
        assert "·" in detail, f"{name} does not say what it is for"


def test_compute_capability_is_reported_when_there_is_a_card():
    """
    It decides which CUDA builds run at all -- sm_120 needs cu128 or newer, and
    older wheels fail with an error naming neither.
    """
    sections, _advice = probe_with(smi="NVIDIA RTX 4090, 8.9, 550.54, 24564 MiB")
    hardware = {f.layer: f.detail for f in sections[1].findings}
    assert "8.9" in hardware["Compute capability"]
    assert "sm_89" in hardware["Compute capability"]
    assert "550.54" in hardware["Driver"]
    assert "24564" in hardware["GPU memory"]


def test_a_gpu_with_no_pytorch_is_told_what_to_install():
    """The most actionable case, and the one a bare `False` hides."""
    _findings, advice = probe_with(smi="NVIDIA RTX 4090, 8.9, 550.54, 24564 MiB",
                                   blocked=("torch",))
    joined = " ".join(advice)
    assert "no PyTorch" in joined
    assert "install" in joined.lower()


def test_a_machine_with_no_gpu_is_told_it_is_fine():
    """
    "You have no GPU" is not a problem to be fixed, and telling someone to
    install CUDA when they cannot use it wastes an afternoon.
    """
    _findings, advice = probe_with(smi=None, blocked=("torch",))
    joined = " ".join(advice)
    assert "No GPU was found" in joined
    assert "optional" in joined


def test_each_layer_is_reported_separately():
    """
    Transcription goes through ctranslate2, not torch, so the two can differ.
    Reporting torch alone would mislead anyone whose complaint is that
    transcription is slow.
    """
    sections, _advice = gpu.probe()
    layers = {f.layer for f in flatten(sections)}
    assert any("PyTorch" in name for name in layers)
    assert any("Transcription" in name for name in layers)


def test_a_missing_transcription_engine_is_reported():
    sections, advice = probe_with(smi=None, blocked=("ctranslate2",))
    assert any(f.layer == "Transcription on GPU" and f.ok is False
               for f in flatten(sections))
    assert any("ctranslate2" in line for line in advice)


def test_the_probe_survives_a_machine_without_nvidia_smi():
    sections, _advice = probe_with(smi=None)
    assert any(f.layer == "Graphics card" and f.ok is False for f in flatten(sections))


def test_the_check_renders_without_a_terminal(tmp_path):
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))
    for title in ("System", "Hardware", "Installed packages", "What Taters can use"):
        assert title in p.text_output


def test_the_report_survives_the_question_that_follows_it(tmp_path):
    """
    Reported from a real run: the tables were drawn and then wiped by the
    "Done?" prompt, leaving only the advice line behind.
    """
    import io

    from rich.console import Console

    from taters.ui.live import LivePrompter

    buf = io.StringIO()

    class Probe(LivePrompter):
        def __init__(self):
            self._stages = []
            self._header = ""
            self._screen_notes = []
            self._title = "Taters"
            self._console = Console(file=buf, width=100)

        def _wait_for_key(self, message):
            # this only stands in for the keypress. the repaint is `pause`'s own
            # job and is what we're here to check, so overriding `pause` itself
            # would test nothing
            pass

    gpu._run(ctx_for(Probe(), tmp_path))

    # everything after the last wipe is what the user is left looking at
    final = buf.getvalue()
    assert "Installed packages" in final
    assert final.count("System") >= 2, "the report was not redrawn for the prompt"


def test_the_tables_carry_no_empty_header_band(tmp_path):
    """yes/no needs no column name, and a blank band on every table is noise."""
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))

    assert p.tables, "the report drew no tables at all"
    for title, _rows, headers in p.tables:
        assert not any(h.strip() for h in headers), f"{title} has a header band"


def test_every_reported_row_says_something(tmp_path):
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))

    for title, rows, _headers in p.tables:
        for row in rows:
            assert row[0].strip(), f"{title} has an unlabeled row"
            assert row[2].strip(), f"{title}: {row[0]} reports no detail"


# ---------------------------------------------------------------------------
# telling someone what to actually run
# ---------------------------------------------------------------------------

def test_the_wheel_index_follows_the_driver_but_not_past_the_stack():
    """
    Derived from what the driver supports rather than hard-coded -- the list of
    published CUDA indexes grows with every release -- but capped by what the
    rest of Taters can use.

    This test used to assert `cuda_tag("13.2") == "cu132"`, which is how a real
    machine came to have a cu132 PyTorch whose cuBLAS the transcription engine
    could not load. Following the driver was the bug.
    """
    assert gpu.cuda_tag("13.2") == "cu128"      # capped
    assert gpu.cuda_tag("12.8") == "cu128"
    assert gpu.cuda_tag("12.4") == "cu124"      # driver is the tighter bound


def test_an_unknown_driver_still_suggests_something_plausible():
    assert gpu.cuda_tag(None) == "cu128"
    assert gpu.cuda_tag("nonsense") == "cu128"


def test_the_suggested_command_forces_a_reinstall():
    """
    The flag is the entire point. `pip install torch --index-url <cuda index>`
    looks right and does nothing when a torch is already present: the
    requirement is satisfied, so pip never consults the index. It reports
    success and leaves the CPU build in place -- which is exactly how someone
    ends up with a CUDA index in their shell history and a CPU build installed.
    """
    command = gpu.reinstall_command("13.2")
    assert "--force-reinstall" in command
    assert "--no-cache-dir" in command
    assert "cu128" in command, "the command has to name a usable CUDA, not the driver's"
    assert "download.pytorch.org/whl/cu128" in command


def test_a_torch_without_cuda_is_recognized_by_its_version():
    assert gpu._is_cuda_build("2.13.0") is False
    assert gpu._is_cuda_build("2.13.0+cpu") is False
    assert gpu._is_cuda_build("2.13.0+cu132") is True
    assert gpu._is_cuda_build(None) is False


def test_a_cpu_build_beside_a_working_driver_is_named_precisely(tmp_path):
    """
    Reported from a real machine: a good driver, an sm_120 card, and a torch
    that simply had no CUDA in it. "Check your setup" is not an answer there.
    """
    import sys as _sys

    class FakeTorch:
        __version__ = "2.13.0"

        class cuda:
            @staticmethod
            def is_available():
                return False

        backends = None

    real_version = gpu._version
    with mock.patch.object(gpu, "_nvidia_smi",
                           lambda: "NVIDIA RTX PRO 500, 12.0, 596.58, 6113 MiB"), \
         mock.patch.object(gpu, "_driver_cuda_ceiling", lambda: "13.2"), \
         mock.patch.object(gpu, "_version",
                           lambda d: "2.13.0" if d == "torch" else real_version(d)), \
         mock.patch.dict(_sys.modules, {"torch": FakeTorch}):
        _sections, advice = gpu.probe()

    joined = " ".join(advice)
    assert "has no CUDA in it" in joined
    assert "2.13.0+cu132" in joined, "the shape of a good version should be shown"
    assert any(line.startswith("pip install --force-reinstall") for line in advice)


def test_a_command_is_never_wrapped(tmp_path):
    """Folded across two lines it pastes as two commands, the second a bare URL."""
    p = ScriptedPrompter([])
    with mock.patch.object(gpu, "probe",
                           lambda: ([], ["Run:", gpu.reinstall_command("13.2")])):
        gpu._run(ctx_for(p, tmp_path))

    command = next(line for line in p.output if "--index-url" in line)
    assert "\n" not in command
    assert command.strip().startswith("pip install")


# ---------------------------------------------------------------------------
# leaving the report
#
# it used to end with "Done? (Y/n)". both answers went back to the same menu,
# so the choice was decoration, and answering "no" raised `Cancelled`, which
# the hub reports as "Backed out. Nothing was changed." over a read-only report
# that had nothing to change in the first place
# ---------------------------------------------------------------------------

def test_the_report_ends_without_asking_anything(tmp_path):
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))

    kinds = {kind for kind, _q in p.asked}
    assert kinds <= {"pause"}, f"the report asked something: {p.asked}"


def test_the_report_waits_before_going_back(tmp_path):
    """
    Waiting is the part worth keeping. Dropping straight back to the menu would
    wipe a report someone has not finished reading.
    """
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))

    assert ("pause", PAUSE_MESSAGE) in p.asked


def test_the_pause_consumes_no_scripted_answer(tmp_path):
    """
    Why it is its own method rather than a `text` prompt with an empty default:
    a pause is not a decision, so it must not eat an answer meant for whatever
    is asked next. `ScriptedPrompter` raises when something reads past the end
    of its script, so an empty script is the assertion.
    """
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))        # would raise if the pause read one


def test_the_report_never_reports_itself_as_backed_out(tmp_path):
    """
    `Cancelled` is what the hub turns into "Backed out. Nothing was changed."
    A finished report has not been backed out of, and nothing about the machine
    was ever going to change.
    """
    from taters.ui.prompts import Cancelled

    p = ScriptedPrompter([])
    try:
        gpu._run(ctx_for(p, tmp_path))
    except Cancelled:                      # pragma: no cover - the regression
        pytest.fail("a finished report raised Cancelled")


def test_a_terminal_that_cannot_read_a_keypress_does_not_lose_the_report():
    """
    The keypress is a courtesy on top of work that is already done and on
    screen. A questionary version that has moved `press_any_key_to_continue`,
    or a terminal that cannot host it, must not take the screen down with it.
    """
    from taters.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter.__new__(QuestionaryPrompter)
    prompter._style = None

    class Broken:
        @staticmethod
        def press_any_key_to_continue(*a, **k):
            raise RuntimeError("no tty")

    prompter._q = Broken()
    prompter.pause()                       # shouldn't raise


# ---------------------------------------------------------------------------
# the checker has to catch what the runs were actually failing on
#
# it reported "Transcription on GPU: yes -- 1 CUDA device(s) available" and
# "Everything checks out" on a machine where every file then died with
# `Library cublas64_12.dll is not found or cannot be loaded`. it was asking
# `get_cuda_device_count()`, which the driver answers, and the driver is
# always there on a machine with a card
# ---------------------------------------------------------------------------

def _runtime_row(sections, layer):
    for section in sections:
        for finding in section.findings:
            if finding.layer == layer:
                return finding
    return None


def test_a_visible_card_with_unloadable_libraries_is_not_reported_as_usable(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_libraries_ok",
                        lambda: (False, "cannot load cublas64_12.dll"))

    sections, _advice = gpu.probe()
    row = _runtime_row(sections, "Transcription on GPU")
    if row is None or "not installed" in row.detail:
        pytest.skip("ctranslate2 is not installed here")

    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device to misreport")

    assert row.ok is False, "a card it cannot use was reported as usable"
    assert "not loadable" in row.detail or "fall back" in row.detail


def test_a_machine_that_cannot_transcribe_on_its_gpu_is_not_told_it_is_fine(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_libraries_ok",
                        lambda: (False, "... version mismatch ..."))

    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device here")

    _sections, advice = gpu.probe()
    assert not advice[0].startswith("Everything checks out")
    assert "Transcription cannot use your GPU" in advice[0]


def test_a_search_path_problem_gets_reassurance_rather_than_a_command(monkeypatch):
    """
    Different problem, different answer. Here the libraries exist and Taters
    already points the loader at them, so there is nothing for the reader to
    install -- and offering a `pip install` would send them to fix something
    that is not broken.
    """
    monkeypatch.setattr(gpu, "cuda_libraries_ok",
                        lambda: (False, "cannot load cublas64_12.dll, though a copy "
                                        "is in C:/x/torch/lib. Nothing adds that "
                                        "folder to the library search path on its own."))

    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device here")

    _sections, advice = gpu.probe()
    said = "\n".join(advice)
    assert "probably fine" in said
    assert not any(line.startswith("pip install") for line in advice)


def test_a_clean_machine_still_says_everything_checks_out(monkeypatch):
    """The counterweight: the check must not become a permanent warning."""
    monkeypatch.setattr(gpu, "cuda_libraries_ok", lambda: (True, "they load"))

    _sections, advice = gpu.probe()
    row = _runtime_row(_sections, "Transcription on GPU")
    if row is not None and row.ok is False:
        pytest.skip("this machine genuinely cannot transcribe on a GPU")
    assert not any("CTranslate2, which loads its own" in line for line in advice)


# ---------------------------------------------------------------------------
# advice someone can actually act on
#
# the version-mismatch case used to print four paragraphs about CTranslate2's
# library loading (to someone who never installed CTranslate2, since it comes
# along as a dependency of faster-whisper) and then asked them to go and find
# a CUDA runtime. it read like four separate problems and never named a command.
#
# worse, the mismatch was our own doing: this screen recommended a PyTorch
# build from the *driver's* CUDA ceiling, so a driver reporting 13.2 got a
# cu132 torch whose cuBLAS ctranslate2 can't load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ceiling,expected", [
    ("13.2", "cu128"),   # the reported machine: capped, not followed
    ("12.9", "cu128"),
    ("12.4", "cu124"),   # the driver is the tighter constraint here
    ("11.8", "cu118"),
    (None, "cu128"),
    ("nonsense", "cu128"),
])
def test_the_recommended_build_is_one_the_whole_stack_can_use(ceiling, expected):
    assert gpu.cuda_tag(ceiling) == expected


def test_no_recommendation_exceeds_what_the_transcription_engine_supports():
    """
    The rule, rather than the six examples above. Raise `MAX_CUDA` when
    ctranslate2 ships CUDA 13 wheels, and this keeps holding.
    """
    for ceiling in ("11.8", "12.4", "12.8", "13.2", "14.0", "99.9"):
        tag = gpu.cuda_tag(ceiling)
        major = int(tag[2:-1]) if len(tag) == 5 else int(tag[2:3])
        assert major <= gpu.MAX_CUDA[0], f"{ceiling} -> {tag}"


def test_the_mismatch_advice_is_short_and_ends_in_a_command(monkeypatch):
    monkeypatch.setattr(gpu, "cuda_libraries_ok",
                        lambda: (False, "cannot load cublas64_12.dll ... this is a "
                                        "version mismatch rather than a search-path "
                                        "problem"))
    monkeypatch.setattr(gpu, "_driver_cuda_ceiling", lambda: "13.2")

    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device here")

    _sections, advice = gpu.probe()

    assert len(advice) <= 3, f"{len(advice)} paragraphs for one problem:\n" + "\n".join(advice)
    assert any(line.startswith("pip install") for line in advice), (
        "told the user what is wrong without telling them what to type"
    )
    assert "cu128" in "\n".join(advice)


def test_the_advice_does_not_explain_a_library_the_user_never_installed(monkeypatch):
    """
    ctranslate2 is a dependency of faster-whisper, which is a base dependency of
    Taters. Nobody using this screen chose it, chose its version, or can be
    expected to know what it links against.
    """
    monkeypatch.setattr(gpu, "cuda_libraries_ok",
                        lambda: (False, "... version mismatch ..."))
    monkeypatch.setattr(gpu, "_driver_cuda_ceiling", lambda: "13.2")

    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        pytest.skip("no CUDA device here")

    _sections, advice = gpu.probe()
    said = " ".join(advice).lower()
    assert "ctranslate2" not in said
    assert "cublas" not in said


def test_a_failing_row_is_visible_before_it_is_read():
    """
    Color carries the verdict. A report where "no" looks exactly like "yes"
    makes someone parse every line to find the one that matters -- and this
    screen exists for the person who does not know which line that is.
    """
    assert gpu._verdict(False) == "[bold red]no[/bold red]"
    assert gpu._verdict(True) == "[green]yes[/green]"


def test_information_that_is_not_a_verdict_gets_no_word():
    """
    A driver version and a GPU's memory are neither good nor bad. A blank cell
    says "nothing to decide here" better than a third label would.
    """
    assert gpu._verdict(None) == ""


def test_the_verdict_reaches_the_table(tmp_path):
    p = ScriptedPrompter([])
    gpu._run(ctx_for(p, tmp_path))

    verdicts = {row[1] for _title, rows, _headers in p.tables for row in rows}
    assert "[green]yes[/green]" in verdicts
    assert all(v in ("", "[green]yes[/green]", "[bold red]no[/bold red]")
               for v in verdicts), verdicts


# ---------------------------------------------------------------------------
# a package that can't be installed here at all
#
# we hit this installing on Windows with Python 3.14. every NeMo 2.7.x `[asr]`
# requires `nv-one-logger-pytorch-lightning-integration`, which declares
# `Requires-Python: >=3.9,<3.14`, and every 2.6.x requires `numpy<2`, whose
# newest release predates 3.14. so on 3.14 there's no NeMo we can install, and
# the extra no longer asks for one. that means the report has to explain why
# it's missing rather than call it "not installed" and send someone off to fix it
# ---------------------------------------------------------------------------

def _marked(requirement: str):
    """Stand in for the installed metadata, with one marked requirement."""
    from unittest import mock

    return mock.patch("importlib.metadata.requires", lambda dist: [requirement])


def test_a_package_this_python_cannot_have_is_recognized():
    # a marker that's false on every Python, standing in for `< "3.14"` as seen
    # from 3.14 (we can't exercise the real one from an older runtime)
    with _marked('nemo-toolkit[asr]<3.0,>=2.7; python_version < "3.0" '
                 'and extra == "all"'):
        assert gpu.unavailable_here("nemo-toolkit") is True


def test_a_package_this_python_can_have_is_not_flagged():
    with _marked('nemo-toolkit[asr]<3.0,>=2.7; python_version >= "3.0" '
                 'and extra == "all"'):
        assert gpu.unavailable_here("nemo-toolkit") is False


def test_an_unconditional_requirement_is_never_flagged():
    with _marked('nemo-toolkit[asr]<3.0,>=2.7; extra == "all"'):
        assert gpu.unavailable_here("nemo-toolkit") is False


def test_a_package_listed_under_any_extra_is_judged_by_that_extra():
    """
    pip lists one requirement line per extra, and the first one for parselmouth
    is `extra == "standard"`. The reader used to evaluate every marker against
    "all" and "diarization" only, so that line came out false and an
    installable package was reported as impossible on this Python -- which
    then leaked into the wizard's preflight hint as "run Python 3.13 or older"
    for a package that installs fine everywhere.
    """
    with _marked('praat-parselmouth>=0.4.6; extra == "standard"'):
        assert gpu.unavailable_here("praat-parselmouth") is False
    with _marked('praat-parselmouth>=0.4.6; python_version < "3.0" '
                 'and extra == "vocalacoustics"'):
        assert gpu.unavailable_here("praat-parselmouth") is True


def test_a_package_we_do_not_require_at_all_is_not_flagged():
    """
    torch is deliberately undeclared -- users pick their own build -- so its
    absence is a normal state, not a Python-version verdict.
    """
    with _marked('nemo-toolkit[asr]<3.0,>=2.7; extra == "all"'):
        assert gpu.unavailable_here("torch") is False


def test_a_card_the_torch_build_has_no_kernels_for_is_not_an_all_clear():
    """
    The false all-clear that cost a real run: a Blackwell card (sm_120) under
    a cu126 torch reports itself available, names itself correctly, and then
    has no kernel to run -- so the work silently went to the CPU while this
    screen said the GPU was fine. The build says what it was compiled for, so
    we ask it.
    """
    older = ["sm_50", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert gpu.cards_without_kernels([(12, 0)], older) == ["sm_120"]
    assert gpu.cards_without_kernels([(8, 6)], older) == []
    assert gpu.cards_without_kernels([(12, 0)], older + ["sm_120"]) == []
    # two cards, one covered and one not: the uncovered one is still named
    assert gpu.cards_without_kernels([(8, 6), (12, 0)], older) == ["sm_120"]
    assert gpu.cards_without_kernels([], older) == []


def test_the_reason_reaches_the_report(monkeypatch):
    """
    "not installed" and "not available for Python 3.14" need opposite
    reactions, and only one of them is worth anyone's afternoon.
    """
    monkeypatch.setattr(gpu, "_version", lambda dist: None)
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "nemo-toolkit")

    sections, _advice = gpu.probe()
    rows = {f.layer: f for section in sections for f in section.findings}

    nemo = rows["nemo-toolkit"]
    assert "not available for Python" in nemo.detail
    assert nemo.ok is None, "an impossibility is not a failure to be fixed"

    # everything else still reads as absent, since it still could be present
    assert "not installed" in rows["praat-parselmouth"].detail


def test_gensim_is_reported_the_same_way_as_nemo(monkeypatch):
    """
    Same cap, same reason (no 3.14 wheels), same verdict. Without a row for
    gensim the check would say nothing at all, and the person who just watched
    pip fail on a C++ compiler would go looking for Visual Studio.
    """
    monkeypatch.setattr(gpu, "_version", lambda dist: None)
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "gensim")

    sections, _advice = gpu.probe()
    rows = {f.layer: f for section in sections for f in section.findings}

    assert "gensim" in rows, "the setup check has no row for gensim"
    assert "not available for Python" in rows["gensim"].detail
    assert rows["gensim"].ok is None


def test_the_metadata_is_the_single_source_of_the_answer():
    """
    Read from `requires("taters")` rather than from a table beside it. A second
    copy of "which Pythons can have NeMo" is a second copy to forget.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gpu.unavailable_here).strip())
    body = ast.unparse(tree.body[0].body[1:])   # everything but the docstring

    assert "requires(" in body
    assert "3.14" not in body, "the version belongs in pyproject.toml, not here"


# ---------------------------------------------------------------------------
# the setup checker's advice (code review issues 12 and 13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ceiling, tag", [
    ("13.2", "cu128"),   # capped by what ctranslate2 can use
    ("12.8", "cu128"),
    ("12.5", "cu124"),   # 12.5 has no index of its own, so nearest published below
    ("12.2", "cu121"),   # the review's case: cu122 doesn't exist
    ("12.0", "cu118"),
    ("11.4", "cpu"),     # older than every published CUDA index
    (None, "cu128"),
    ("weird", "cu128"),
])
def test_only_published_wheel_indexes_are_ever_recommended(ceiling, tag):
    """
    From the code review (issue 13): deriving the tag from the driver ceiling
    alone produced `cu122` -- a CUDA version with no wheel index behind it, so
    the recommended install command 404'd.
    """
    from taters.ui.tasks.gpu import cuda_tag

    assert cuda_tag(ceiling) == tag


def test_smi_error_text_is_not_a_gpu(monkeypatch):
    """
    From the code review (issue 12): drivers too old for `--query-gpu=compute_cap`
    print the complaint to stdout, which the report then rendered as the GPU's
    name. Only a clean exit with the four fields asked for counts.
    """
    from taters.ui.tasks import gpu as gpu_task

    class Fake:
        returncode = 6
        stdout = ('Field "compute_cap" is not a valid field to query.\n')
        stderr = ""

    monkeypatch.setattr(gpu_task.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_task.subprocess, "run", lambda *a, **k: Fake())

    assert gpu_task._nvidia_smi() is None


def test_a_clean_smi_answer_still_comes_through(monkeypatch):
    from taters.ui.tasks import gpu as gpu_task

    class Fake:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5090, 12.0, 570.86, 32607 MiB\n"
        stderr = ""

    monkeypatch.setattr(gpu_task.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_task.subprocess, "run", lambda *a, **k: Fake())

    assert "RTX 5090" in gpu_task._nvidia_smi()


def test_the_checker_survives_an_uninstalled_source_tree(monkeypatch):
    """Round-2 cut list: `requires('taters')` raises PackageNotFoundError when
    running from a source checkout that was never pip-installed -- crashing the
    very screen whose job is diagnosis."""
    from importlib.metadata import PackageNotFoundError

    from taters.ui.tasks import gpu as gpu_task

    def explode(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.requires", explode)
    assert gpu_task.unavailable_here("nemo-toolkit") is False


def test_the_setup_report_names_the_worker_ceiling():
    """"Check my setup" is where someone learns what their machine can do, so
    the System section says how high `workers` may go and what automatic
    picks -- the same numbers the resolver actually uses."""
    from taters.helpers.parallel_map import auto_workers, max_workers

    section = gpu._system_section()
    row = next(f for f in section.findings if f.layer == "Processor")
    cores = max_workers()
    assert f"{cores} logical cores" in row.detail
    assert f"1–{cores}" in row.detail
    assert f"automatic uses {auto_workers()}" in row.detail
    assert row.ok is None, "information, not a verdict"
