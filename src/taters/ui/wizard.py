"""
The Taters setup wizard: a console front door for people who do not write code.

Run ``taters`` with no arguments and this walks you through four questions --
where your data is, what kind it is, what you want out of it, and which options
to change -- then writes a pipeline folder (``<name>/<name>.yaml``, in the
working folder) and offers to run it.

Why it writes a file instead of just running
--------------------------------------------
The thing this produces is an ordinary preset, indistinguishable from the ones
that ship with Taters. That buys a lot for one design decision:

* the run gets the pipeline runner's concurrency, its resumability, its manifest
  and its per-file error isolation, none of which the wizard has to reimplement;
* the runner recognizes such folders, so the result turns up under
  ``--list-presets`` and can be re-run from the command line forever after;
* and the user ends up holding a small readable file they can edit, version, or
  send to a colleague -- which is how somebody graduates from the wizard to the
  rest of the tool.

Layering
--------
This module contains no terminal code. It asks questions through the
:class:`~taters.ui.prompts.Prompter` protocol, which is why the test suite can
drive the whole flow with a scripted list of answers. Swap the prompter and the
same logic backs a GUI.
"""

from __future__ import annotations

import csv as _csv
import importlib.util
import shutil
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from . import glyphs
from . import recipes as _recipes
from . import preflight as _checks
from .columns import (column_kind, constant_within, distinct_values,
                      kind_options, looks_numeric, peek_csv, plausible_labels,
                      read_columns, sniff_delimiter)
from .compose import (MODEL_WORK_DIR, ComposeError, compose,
                      feature_tables, table_names,
                      pending_choices, resolve_selection, slugify)
from .introspect import EMPTY, FunctionSpec, ParamSpec, describe, load_target
from .prompts import Cancelled, Choice, GoBack, Prompter, QuitRequested, ask_at_least_one

__all__ = ["run_wizard", "WizardResult", "main"]


# which import tells us an extra is installed. we use `importlib.util.find_spec`
# on these because it's near-instant; actually importing the analysis module
# would drag in torch and NeMo, which is way too slow to do for every step
# at preflight time.
EXTRA_PROBES: Dict[str, Tuple[str, ...]] = {
    "diarization": ("nemo",),
    # we leave `disvoice` out of here. it only backs one optional code path
    # that already warns and carries on without it, so if we probed for it
    # we'd make a perfectly good acoustics install look broken. it lives in
    # the `glottal` extra, which most people will never want anyway.
    "vocalacoustics": ("parselmouth", "soundfile"),
    "vectors": ("gensim",),
}
# and the distribution each extra hangs on -- the one whose Python-version
# marker in our own metadata decides whether "not installed" is fixable here
# or whether pip would just install nothing (see tasks.gpu.unavailable_here)
EXTRA_DISTS: Dict[str, str] = {
    "diarization": "nemo-toolkit",
    "vectors": "gensim",
    "vocalacoustics": "praat-parselmouth",
}
# no `readability` probe either. textstat moved into the base install, so if
# it's missing then the environment is busted, not just missing an extra --
# and `pip install taters[readability]` wouldn't install anything now anyway.

# the opening question. we keep media and text in one flat list rather than
# asking "media or text?" first and narrowing down from there. this is because
# that first question is really about how we've built things, not about the
# user's data, and if they guess wrong they end up on a transcription path
# they never needed.
#
# each entry maps to a (source, file_type) pair -- see SOURCE_KIND_MAP.
SOURCE_KINDS = [
    Choice("audio", "Audio files", "wav, mp3, m4a, flac, ogg, opus…"),
    Choice("video", "Video files", "mp4, mov, mkv, avi, webm… (the audio is pulled out)"),
    Choice("txt_dir", "Documents in a folder (.txt, .docx, .doc, .pdf)",
           "Essays, notes, transcripts you already have — nothing gets transcribed"),
    Choice("csv", "A spreadsheet, with text in one or more columns",
           "CSV with one row per document or response"),
    Choice("any", "Everything in a folder", "Media only; let ffmpeg try each file in turn"),
]

SOURCE_KIND_MAP = {
    "audio": ("media", "audio"),
    "video": ("media", "video"),
    "any": ("media", "any"),
    "txt_dir": ("txt_dir", "any"),
    "csv": ("csv", "any"),
}

# we keep this around for callers that only ever cared about the media filter.
FILE_TYPES = [c for c in SOURCE_KINDS if c.value in {"audio", "video", "any"}]


@dataclass
class WizardResult:
    """What the wizard did, for the caller and for the tests."""

    preset: dict
    preset_path: Optional[Path] = None
    folder: Optional[Path] = None
    root_dir: Optional[Path] = None
    file_type: str = "any"
    source: str = "media"
    inputs: List[Path] = field(default_factory=list)
    ran: bool = False
    manifest: Optional[dict] = None
    ok: Optional[bool] = None


# ---------------------------------------------------------------------------
# Step 1-2: where the data is
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    """Everything the rest of the wizard needs to know about the input."""

    source: str                      # "media" | "txt_dir" | "csv"
    path: Path                       # a folder, or the spreadsheet itself
    file_type: str = "any"           # only meaningful when source == "media"
    inputs: List[Path] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=lambda: ["text"])
    id_cols: List[str] = field(default_factory=list)
    text_mode: str = "concat"        # "concat" | "separate"; only for CSV
    group_by: List[str] = field(default_factory=list)   # combine rows sharing these
    level: str = ""                  # "" -> the source's default; see ask_level
    columns: List[str] = field(default_factory=list)    # a CSV's header row
    delimiter: str = ","             # sniffed from the file: , \t ; or |
    #: What each column is treated as -- "numbers", "labels", "text", "empty"
    #: -- detected from a sample and changed by the person with the left and
    #: right arrows on any column picker. Shown beside every column name, so
    #: a 1/2 gender code that reads as numbers is visibly a measurement until
    #: someone says otherwise, and the statistics stage reads the answer.
    kinds: Dict[str, str] = field(default_factory=dict)
    #: Columns the user named as the measures themselves, rather than as
    #: something to measure. Only the "Run analyses" flow sets these: the
    #: numbers are already in the spreadsheet, so these columns are the
    #: predictors and every other column is free to be an outcome, a group
    #: or a control. Spoken for either way, which is why the analysis stage
    #: subtracts them alongside `text_cols`.
    feature_cols: List[str] = field(default_factory=list)

    @property
    def root_dir(self) -> Optional[Path]:
        """
        What to hand the runner as ``root_dir``.

        ``None`` for text sources: those presets are GLOBAL-only, so the runner
        skips discovery entirely and the input path travels as a variable
        instead.
        """
        return self.path if self.source == "media" else None


def _suffixes_for(file_type: str) -> Tuple[str, ...]:
    """Which extensions to count next to each folder while browsing."""
    from ..pipelines.run_pipeline import _AUDIO_EXTS, _VIDEO_EXTS

    if file_type == "audio":
        return tuple(sorted(_AUDIO_EXTS))
    if file_type == "video":
        return tuple(sorted(_VIDEO_EXTS))
    return tuple(sorted(_AUDIO_EXTS | _VIDEO_EXTS))


def _ask_folder(prompter: Prompter, question: str,
                want_files: Optional[Sequence[str]] = None) -> Path:
    """
    Browse to a folder.

    Typing a path used to be the only option here, and it was the most
    error-prone thing the wizard asked for: a typo produces "no files found",
    which reads as "there is nothing here" rather than "you are looking in the
    wrong place". Browsing also shows how many usable files each folder holds,
    so the right one can be recognized instead of remembered. Typing is still
    offered inside the browser for anyone who already knows the path.
    """
    from .browse import browse_for_folder

    return browse_for_folder(prompter, question=question, want_files=want_files)


class _Restart(Exception):
    """The file or folder will not do: leave the source questions and start
    them over from the kind of data (see :func:`ask_source`)."""


def _ask_in_order(steps: Sequence[Callable[[], bool]]) -> None:
    """
    Ask a run of questions where Esc means "the previous question".

    Each step asks at most one question, stores its answer where later steps
    can read it, and returns True if it put a question on screen (a step
    that had nothing to ask -- no second text column, so no "how should they
    be measured" -- returns False and is skipped on the way back too). A
    GoBack from a step re-runs the nearest earlier step that actually asked;
    from the first one it propagates, because the stage before owns it.

    This exists because Esc used to be caught only at stage boundaries. From
    the fifth spreadsheet question it left the wizard; from the fourth
    analysis question it rewound to the feature checklist, throwing away the
    level and every analysis answer so far -- "sometimes the exact previous
    screen, sometimes five questions back" (a real report). Steps re-ask
    with the earlier answer pre-filled, so going back to change one thing
    costs one keypress per question, not a re-typing of all of them.
    """
    asked = [False] * len(steps)
    i = 0
    while i < len(steps):
        try:
            asked[i] = bool(steps[i]())
        except GoBack:
            j = i - 1
            while j >= 0 and not asked[j]:
                j -= 1
            if j < 0:
                raise
            i = j
            continue
        i += 1


def _try_again(prompter: Prompter, reason: str) -> None:
    """
    Explain why the input will not do, and offer another go.

    Returns None so the caller can `return _try_again(...)` -- every source
    asker signals "ask me again" by returning None, and this keeps the three of
    them from each spelling out the same explain-confirm-or-quit sequence.

    Raises
    ------
    Cancelled
        If the user would rather stop than pick again.
    """
    prompter.reason(reason)
    if not prompter.confirm("Try a different folder or file type?", default=True):
        raise Cancelled()
    return None


def _ask_media_source(prompter: Prompter, file_type: str) -> Optional[SourceSpec]:
    """
    Folder of audio/video, with discovery run before we go any further.

    An empty result loops rather than proceeding. Finding out that a run matched
    zero files after picking a dozen options and waiting for a model download is
    a miserable way to learn you typed the wrong folder.
    """
    from ..pipelines.run_pipeline import discover_inputs

    root = _ask_folder(prompter, "Where is your data?", _suffixes_for(file_type))
    inputs = discover_inputs(root, file_type)
    if inputs:
        prompter.note(f"  Found {len(inputs)} file(s) under {root}.", style="green")
        return SourceSpec(source="media", path=root, file_type=file_type, inputs=list(inputs))
    # returning None sends `ask_source` all the way back to the kind-of-data
    # question, and we want that. picking "Video files" for a folder of mp3s
    # is an easy mistake, and if we only re-asked the folder we'd keep the
    # wrong filter.
    return _try_again(
        prompter,
        f"No {file_type} files under {root}. Subfolders are searched too, "
        "so check the folder or pick a different type.")


def _ask_txt_source(prompter: Prompter) -> Optional[SourceSpec]:
    """Folder of documents. Counted up front, for the same reason as media."""
    from ..helpers.doc_text import DOCUMENT_SUFFIXES

    root = _ask_folder(prompter, "Where are your documents?",
                       DOCUMENT_SUFFIXES)
    # case-insensitive, same as the browser's counts. Windows will happily name
    # a file .TXT instead of .txt, and WSL reads that same folder
    # case-sensitively -- we had the browser say "12 files" and this find 0.
    found = sorted(f for f in root.rglob("*")
                   if f.is_file() and f.suffix.lower() in DOCUMENT_SUFFIXES)
    if found:
        prompter.note(f"  Found {len(found)} document(s) under {root}.",
                      style="green")
        return SourceSpec(source="txt_dir", path=root, inputs=found)
    return _try_again(
        prompter, f"No documents (.txt, .docx, .doc, .pdf) under {root} "
        "(subfolders included).")


def _delimiter_of(path: Path) -> str:
    """The wizard's one delimiter sniff; see :func:`columns.sniff_delimiter`."""
    return sniff_delimiter(path)


def _read_header(path: Path) -> List[str]:
    """
    The column names of a CSV, or [] if it cannot be read as one.

    utf-8-sig because spreadsheets exported from Excel carry a BOM, and a BOM
    left on the front of the first column name is exactly the kind of thing that
    turns "text" into "\ufefftext" and makes a later lookup fail for no visible
    reason.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in _csv.reader(fh, delimiter=_delimiter_of(path)):
                # raw, not stripped. the gather matches DictReader fieldnames
                # verbatim, so if we offer a cleaned-up 'text' for a header
                # that really says ' text', we build a pipeline that dies
                # minutes in with "Missing columns" (been there). we tidy up
                # the labels wherever the names get shown, but the *values*
                # have to be whatever the file actually says.
                names = [c for c in row if c.strip()]
                break
            else:
                return []
    except (OSError, UnicodeDecodeError, _csv.Error):
        return []

    # a .docx or .xlsx is really a zip, and its first few bytes decode as text
    # without complaint -- so `csv.reader` cheerfully hands us back one garbage
    # "column name" and we'd go and offer it to the user. control characters
    # are the tell here: no real column name has one.
    if any(ch < " " and ch != "\t" for name in names for ch in name):
        return []
    return names


#: How many column names the "I read your spreadsheet" note lists before it
#: stops. Enough to recognize the file by; the full list is the question
#: underneath it.
_NAMES_IN_NOTE = 8


def _ask_csv_source(prompter: Prompter, *,
                    needs_metadata: bool = False,
                    columns_are_measures: bool = False) -> Optional[SourceSpec]:
    """
    A spreadsheet, plus which of its columns hold the text and the identifiers.

    Asking about columns is not a nicety. The analyzers default to
    ``text_cols=("text",)``, so a file whose column is called ``response`` or
    ``Body`` would analyze nothing at all and report no error -- the single most
    likely way for a first run to fail silently.

    ``needs_metadata`` is the "+ run analyses" flow: those runs need at least
    one column that is *not* text, to group by or predict. Checked here
    because this is the last screen where it can still be fixed -- by
    unticking a text column, or by picking a different file.

    Five questions, asked through :func:`_ask_in_order` so Esc steps back one
    at a time with the earlier answers kept; from the file question it leaves
    for the kind-of-data question.
    """
    from .browse import browse_for_file

    state: Dict[str, Any] = {"path": None, "header": [], "text_cols": [],
                             "text_mode": "concat", "has_id": False,
                             "id_cols": [], "kinds": {}, "sample": [],
                             "feature_cols": []}

    def remaining() -> List[str]:
        # everything not already spoken for. a column picked as the text, or
        # (in the analyze-a-spreadsheet flow) as one of the measures, is not
        # also the row's identifier and is not left over to group by -- and
        # subtracting only the text once let a predictor be picked as the id,
        # which quietly dropped it from the feature table
        spoken_for = set(state["text_cols"]) | set(state["feature_cols"])
        return [c for c in state["header"] if c not in spoken_for]

    def ask_file() -> bool:
        path = browse_for_file(prompter, question="Where is your spreadsheet?",
                               suffixes=(".csv", ".tsv", ".txt"))
        header = _read_header(path)
        if not header:
            _try_again(
                prompter,
                f"I could not read column names from {path.name}. "
                "Is it a CSV with a header row?")
            raise _Restart
        path = path.resolve()
        if state["path"] != path:
            # a different file, so nothing we answered about the old one applies
            # anymore. wipe it.
            state.update(text_cols=[], text_mode="concat", has_id=False,
                         id_cols=[])
        state["path"], state["header"] = path, header
        # now we figure out what each column holds, from a sample. we show
        # this beside the name on every picker from here on out, so that people
        # can see how a column is going to be treated (and change it) before
        # it actually matters.
        try:
            _cols, sample = peek_csv(path, delimiter=_delimiter_of(path))
        except Exception:
            sample = []
        state["sample"] = sample
        state["kinds"] = {c: column_kind([r.get(c) for r in sample])
                          for c in header}
        # the count, plus enough names to recognize the file by -- not all of
        # them. every column shows up on the very next screen as a row you can
        # tick, so listing all 150 of them here tells you nothing new and just
        # squishes that screen down to a porthole (this bit us once with a
        # really wide file).
        shown = [c.strip() for c in header[:_NAMES_IN_NOTE]]
        rest = len(header) - len(shown)
        prompter.note(
            f"  {path.name}: {len(header)} column(s) — "
            + ", ".join(shown)
            + (f", and {rest} more" if rest > 0 else ""),
            style="green")
        return True

    def ask_measure_cols() -> bool:
        """The columns that *are* the measures, for the analyze flow."""
        numeric = [c for c in state["header"]
                   if state["kinds"].get(c) == "numbers"]
        if not numeric:
            _try_again(
                prompter,
                f"No column in {state['path'].name} reads as numbers, so "
                "there is nothing here to analyze as a predictor. The "
                "statistics need columns of numbers.")
            raise _Restart
        prompter.reason(
            "These are the measures themselves -- the columns the statistics "
            "will relate to your outcomes. Only columns that read as numbers "
            "are offered; press the left and right arrows on a row to change "
            "how a column is read. Everything you leave unticked is still "
            "available as an outcome, a group or a control.")
        state["feature_cols"] = ask_at_least_one(
            prompter, "Which columns are the predictors?",
            _kind_rows(numeric, state["kinds"],
                       checked=state["feature_cols"]), thing="one column",
            cycle=_cycler(state["kinds"], state["sample"], numeric))
        return True

    def ask_text_cols() -> bool:
        # a checkbox, not a single choice. a spreadsheet is the one source
        # where a row can carry several pieces of text (an open-ended answer
        # and a follow-up, a headline and a body), and without this there'd be
        # no way to analyze both short of going round twice.
        state["text_cols"] = ask_at_least_one(
            prompter, "Which column(s) hold the text you want analyzed?",
            _kind_rows(state["header"], state["kinds"],
                       checked=state["text_cols"]), thing="one column",
            cycle=_cycler(state["kinds"], state["sample"], state["header"]))
        return True

    def ask_mode() -> bool:
        if len(state["text_cols"]) <= 1:
            state["text_mode"] = "concat"
            return False
        state["text_mode"] = str(prompter.select(
            f"You picked {len(state['text_cols'])} columns. How should they "
            f"be measured?",
            [
                Choice("concat", "Together, as one piece of text",
                       "Joined per row — one set of measures per row"),
                Choice("separate", "Separately, one at a time",
                       "One set of measures per column per row"),
            ],
            default=state["text_mode"]))
        return True

    # we just ask this outright, since the two answers produce different
    # tables and nothing later in the run can tell which one was meant. we
    # used to infer it from the id column, which was actually worse than not
    # asking at all: naming a column whose values repeat gave several rows the
    # same `text_id` *without* combining them, so the id stopped identifying
    # anything -- and then the sentence-embedding merge (which groups on
    # `text_id`) averaged those rows while every other feature left them
    # apart. one spreadsheet, two tables, different numbers of rows, no
    # warning. fun.
    # note that we do *not* ask about combining rows here. that's the same
    # question as "what should one row of results describe?" on the media
    # path, so we let `ask_level` ask it once for every source -- it needs
    # the column names to do that, hence `columns` below.
    def ask_has_id() -> bool:
        if not remaining():
            state["has_id"], state["id_cols"] = False, []
            return False
        state["has_id"] = bool(prompter.confirm(
            "Is there a column that identifies each row (an ID or key)?",
            default=state["has_id"]))
        if not state["has_id"]:
            state["id_cols"] = []
        return True

    def ask_which_ids() -> bool:
        if not state["has_id"]:
            return False
        while True:
            state["id_cols"] = list(prompter.checkbox(
                "Which one(s)?",
                _kind_rows(remaining(), state["kinds"],
                           checked=state["id_cols"]),
                cycle=_cycler(state["kinds"], state["sample"], remaining())))
            # we check this on the whole file, not the sample, because a
            # repeating id gives several rows one text_id and the run dies at
            # the join, after every feature has already been measured (we had
            # a run go 938 rows in, 0 joined).
            finding = _ids_repeat(state["path"], state["id_cols"])
            if finding is None:
                return True
            prompter.note(f"  {finding.message}", style="yellow")

    try:
        # the analyze-a-spreadsheet flow measures nothing: its columns are
        # already the measures, so it picks those instead of text and never
        # needs the how-should-several-text-columns-be-read question
        questions = ([ask_file, ask_measure_cols] if columns_are_measures
                     else [ask_file, ask_text_cols, ask_mode])
        _ask_in_order(questions + [ask_has_id, ask_which_ids])
    except _Restart:
        return None

    # the analysis stage checks this too, four screens later, but by then
    # the only way back is the feature checklist -- and no answer there can
    # conjure up a metadata column, so the two screens just bounced off each
    # other until you hit Esc twice. if we catch it here, unticking a column
    # fixes it.
    if needs_metadata and not remaining():
        return _try_again(
            prompter,
            f"Every column in {state['path'].name} is being analyzed as "
            + ("a predictor" if columns_are_measures else "text")
            + ", so there is nothing left to group by or predict. "
            "Statistics need at least one column that is not "
            + ("a predictor." if columns_are_measures else "the text."))

    return SourceSpec(
        source="csv",
        path=state["path"],
        inputs=[state["path"]],
        columns=list(state["header"]),
        delimiter=_delimiter_of(state["path"]),
        text_cols=[str(c) for c in state["text_cols"]],
        feature_cols=[str(c) for c in state["feature_cols"]],
        id_cols=list(state["id_cols"]),
        text_mode=state["text_mode"],
        kinds=dict(state["kinds"]),
    )

def _ids_repeat(path: Path, id_cols: Sequence[str]):
    """The preflight finding for id columns that do not identify rows, or
    None -- also None when the file cannot be read here, since the gather
    will say so in its own words."""
    if not id_cols:
        return None
    try:
        values = read_columns(path, list(id_cols), delimiter=_delimiter_of(path))
    except Exception:
        return None
    return _checks.repeated_ids(values)


def ask_source(prompter: Prompter,
               analyses: Optional[bool] = None,
               text_only: bool = False,
               sources: Optional[Sequence[str]] = None,
               columns_are_measures: bool = False) -> SourceSpec:
    """
    Ask what kind of data the user has and where it is.

    The kind comes first: it decides whether the next question wants a folder
    or a file, and -- more importantly -- whether there is anything to
    transcribe at all.

    A source that turns up empty returns to the *top* of this loop rather than
    just re-asking the path. Picking "Video files" for a folder of mp3s is an
    easy mistake, and re-asking only the folder would leave the wrong filter in
    place however many times they retyped it.

    ``analyses=True`` means the user came here to run statistics, and only a
    spreadsheet carries the columns to run them against. Saying so here costs
    one screen; saying it after they have browsed to a folder, picked
    features and answered the level question costs all of that.

    ``text_only`` is the training flow: a model is trained on text that
    already exists, so recordings -- which would first have to be
    transcribed by a pipeline of their own -- are not offered. ``sources``
    narrows further to the named source ids (``csv``, ``txt_dir``,
    ``media``): a step that predicts outcome columns can only read a
    spreadsheet, and offering a folder of documents would fail after every
    question had been answered.
    """
    while True:
        if analyses:
            # this has to live inside the loop. the renderer clears a reason
            # once its question gets answered, so a rejected file used to come
            # back to a one-option menu with nothing on screen saying why.
            prompter.reason(
                "Statistics need a spreadsheet: the groups to compare or the "
                "outcomes to predict have to be columns sitting beside the "
                "text. A folder of documents or recordings has no such "
                "columns, so only the spreadsheet option is offered here.")
        kinds = ([c for c in SOURCE_KINDS if c.value == "csv"] if analyses
                 else [c for c in SOURCE_KINDS if c.value in ("csv", "txt_dir")]
                 if text_only else SOURCE_KINDS)
        if sources is not None:
            kinds = [c for c in kinds if SOURCE_KIND_MAP[c.value][0] in set(sources)]
        if text_only and not analyses and len(kinds) == 1 and kinds[0].value == "csv":
            prompter.reason(
                "This model predicts columns of a spreadsheet from its text, "
                "so only a spreadsheet -- the text in one column, the outcomes "
                "in others -- can train it.")
        elif text_only and not analyses:
            prompter.reason(
                "A model is trained on text you already have: a folder of "
                "documents or a spreadsheet with the text in one column. To "
                "train on recordings, transcribe them with a pipeline first "
                "and train on the transcripts it writes.")
        kind = str(prompter.select("What kind of data do you have?", kinds))
        source, file_type = SOURCE_KIND_MAP[kind]

        try:
            if source == "csv":
                spec = _ask_csv_source(
                    prompter, needs_metadata=bool(analyses),
                    columns_are_measures=columns_are_measures)
            elif source == "txt_dir":
                spec = _ask_txt_source(prompter)
            else:
                spec = _ask_media_source(prompter, file_type)
        except GoBack:
            # Esc on a source's first question (the file or folder browser)
            # means "not this kind of data after all", so we loop back around.
            # this used to climb right out of the wizard altogether.
            continue

        if spec is not None:
            return spec


# ---------------------------------------------------------------------------
# Step 3-4: what to extract, and how to make a transcript
# ---------------------------------------------------------------------------

#: How the feature checklist marks a row the statistics cannot use.
NOT_FOR_STATISTICS = "(not for statistics)"


def ask_features(prompter: Prompter, source: str = "media",
                 analyses: bool = False) -> List[str]:
    """
    Show the checklist of things a user can ask for, and take their picks.

    Filtered by source: there is no vocal pitch to measure in a folder of
    essays, and offering it would only let someone pick an option guaranteed to
    fail -- after a multi-gigabyte install to find out.

    Grayed out for the same reason where the row needs something the user
    does not have: scoring with a saved model needs a saved model, and a row
    that is offered plainly and then refused at the preflight screen was
    offered from a list that gave no hint it was unavailable.

    ``analyses=True`` is the "+ run analyses" flow. Statistics join per-text
    feature tables, and two rows here do not make one -- a document-term
    matrix and an n-gram frequency list describe the corpus, not each text
    -- so those are marked, and a pick made only of them is turned back
    here, at the screen that can change it. It used to be caught two
    screens later, after the level question, with a two-line note the
    layout then cut down to "Pick something else to extract, whose measures
    the statistics" (a real report).
    """
    choices = [
        Choice(r.id, r.label,
               (r.text_help or r.help) if source != "media" else r.help,
               disabled=_no_saved_model(r) or unavailable_reason(r),
               annotation=(NOT_FOR_STATISTICS
                           if analyses and not r.feature_table else ""))
        for r in _recipes.user_facing(source)
    ]
    # the question here used to be "What do you want out of it?", and people
    # found it vague -- it didn't say that the answer is a set of feature
    # tables, or that the steps they depend on come along for free.
    reason = ("Each row here is a table of measures Taters will produce, "
              f"one row per {'file' if source == 'media' else 'text'}. Tick "
              "as many as you like; anything a pick needs first -- "
              "converting audio, transcribing it, counting words -- is added "
              "for you.")
    if analyses:
        reason += (" You asked for statistics as well, so at least one pick "
                   f"has to be a per-text table; rows marked "
                   f"{NOT_FOR_STATISTICS} describe the whole corpus instead, "
                   "and can come along but not carry the statistics.")
    prompter.reason(reason)
    while True:
        picked = ask_at_least_one(
            prompter, "Which features do you want to extract?", choices,
            thing="one feature")
        if not analyses or any(_recipes.by_id(p).feature_table
                               for p in picked):
            return picked
        prompter.note("  None of those produces a per-text table for the "
                      "statistics to use; add at least one that does.",
                      style="yellow")


def resolve_providers(prompter: Prompter, selected: Sequence[str],
                      source: str = "media") -> Dict[str, str]:
    """
    Settle any capability that more than one recipe could satisfy.

    In practice this is one question: a transcript can come from plain
    transcription or from diarization. It gets asked when a chosen feature
    needs a transcript and the user did not tick either producer, and it gets
    asked *again* if they ticked both -- two transcription steps writing to the
    same place is never what anyone meant.
    """
    providers: Dict[str, str] = {}
    picked = set(selected)

    # first thing's first: capabilities where the user ticked more than one
    # producer. these collide, so we need a choice even though nothing's
    # actually missing.
    for capability, description in _recipes.CAPABILITIES.items():
        options = _recipes.providers_of(capability)
        ticked = [o for o in options if o.id in picked]
        if len(options) < 2 or len(ticked) < 2:
            continue
        prompter.reason(
            f"You ticked more than one way to get {description}. "
            "Pick one — they would collide."
        )
        providers[capability] = prompter.select(
            f"How should Taters produce {description}?",
            [Choice(o.id, o.label, o.help) for o in ticked],
            default=ticked[0].id,
        )

    # then we handle capabilities the selection needs but never named. we
    # resolve these transitively -- ticking only "Readability scores" needs a
    # transcript, via the merge step, without ever mentioning one.
    for capability, options in pending_choices(
        [s for s in selected if s not in {o.id
                                          for cap, keep in providers.items()
                                          for o in _recipes.providers_of(cap)
                                          if o.id != keep}],
        source=source,
    ).items():
        prompter.reason(
            f"What you picked needs {_recipes.CAPABILITIES[capability]}."
        )
        providers[capability] = prompter.select(
            f"How should Taters produce {_recipes.CAPABILITIES[capability]}?",
            # grayed for the same reason as on the checklist: diarization on a
            # Python with no NeMo release is not a way to get a transcript
            [Choice(o.id, o.label, o.help, disabled=unavailable_reason(o))
             for o in options],
            default=options[0].id,
        )
    return providers


# ---------------------------------------------------------------------------
# Step 5: preflight
# ---------------------------------------------------------------------------

def missing_extras(recipe: _recipes.Recipe) -> List[str]:
    """Extras this step needs that are not importable right now."""
    absent = []
    for extra in recipe.extras:
        probes = EXTRA_PROBES.get(extra, ())
        if any(importlib.util.find_spec(m) is None for m in probes):
            absent.append(extra)
    return absent


def unavailable_reason(recipe: _recipes.Recipe) -> str:
    """
    Why this step cannot run on this Python at all, or ``""``.

    A missing extra is usually one pip install away, and the preflight
    offers to keep the step anyway for exactly that reason. But when our own
    metadata says the package has no release for this Python (gensim and
    NeMo on 3.14), no install will ever fix it here -- so the row is grayed
    out with that reason, rather than offered plainly and refused later.
    Short, because it has to fit beside the row's label.
    """
    from .tasks.gpu import unavailable_here

    stuck = [EXTRA_DISTS[e] for e in missing_extras(recipe)
             if e in EXTRA_DISTS and unavailable_here(EXTRA_DISTS[e])]
    if not stuck:
        return ""
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"not available for Python {py} (no {' or '.join(stuck)} release yet)"


def preflight(prompter: Prompter, steps: Sequence[_recipes.Recipe]) -> List[str]:
    """
    Check the machine can actually run what was chosen.

    Returns
    -------
    list[str]
        Recipe ids to drop. Empty when everything is satisfied, or when the
        user chose to keep a step anyway -- they may be about to install the
        missing piece, and a preset that is slightly ahead of the machine is a
        perfectly reasonable thing to want.
    """
    drop: List[str] = []

    if any(r.needs_ffmpeg for r in steps) and shutil.which("ffmpeg") is None:
        prompter.reason(
            "ffmpeg was not found on your PATH. Every pipeline here needs it to "
            "read media files. Install it from https://ffmpeg.org/download.html "
            "(or `sudo apt install ffmpeg` / `brew install ffmpeg`)."
        )
        try:
            carry_on = prompter.confirm("Carry on and write the pipeline anyway?",
                                        default=True)
        except GoBack:
            # Esc mid-preflight shouldn't cost us the whole session (this was
            # round-2 issue 32). every answer so far survives, and we fall back
            # to the default, since "not this question" is the harmless read.
            carry_on = True
        if not carry_on:
            raise Cancelled()

    for recipe in steps:
        absent = missing_extras(recipe)
        if not absent:
            continue

        # we don't escape this for rich, and that's intentional. as a `note`
        # this string went through rich, which reads `[vocalacoustics]` as a
        # style tag and printed the command as `pip install "taters"` -- looks
        # right, runs cleanly, installs nothing that was missing. a description
        # goes to prompt_toolkit as plain text instead, so the brackets survive
        # on their own (and a rich escape would show its backslash).
        install = " ".join(f'"taters[{extra}]"' for extra in absent)
        why = f"Not installed. To add it:  pip install {install}"
        # unless this Python can't have it at all. somebody on 3.14 followed
        # the pip command above, watched it succeed, and still had no gensim,
        # because our metadata (rightly) doesn't ask for it there. so we say
        # what's actually going on instead of handing out a no-op
        from .tasks.gpu import unavailable_here
        stuck = [EXTRA_DISTS[e] for e in absent
                 if e in EXTRA_DISTS and unavailable_here(EXTRA_DISTS[e])]
        if stuck:
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            why = (f"Not available for Python {py}: {' and '.join(stuck)} has no "
                   f"release for it yet, so pip would install nothing. Run Taters "
                   f"on Python 3.13 or older to use this step.")
        if "diarization" in absent:
            why += ("  ·  Diarization also needs three packages that only exist "
                    "on GitHub — see the install guide. It is the one heavy extra.")

        # a two-option list rather than a yes/no, so that the explanation can
        # ride along as the question's description. as a `note` it got printed
        # before the screen was drawn and ended up above the progress rail --
        # a whole paragraph of context floating a long way from its question.
        try:
            keep = str(prompter.select(
                f"Keep '{recipe.label}' in the pipeline?",
                [Choice("keep", "Yes, keep it", why),
                 Choice("drop", "No, leave it out", why)],
            ))
        except GoBack:
            keep = "keep"       # Esc means the harmless answer, not a reset
        if keep == "drop":
            drop.append(recipe.id)

    # now the library case. a step whose dictionaries come from an empty
    # library would compose just fine and then read nothing at run time. same
    # idea as the missing-extra check above -- we say so while there's still
    # time to do something about it, and offer the fix right here.
    from ..helpers.library import entries as _lib_entries, kind_by_id
    from .library import import_files

    for recipe in steps:
        if recipe.id in drop:
            continue
        for param_name, kind_id in recipe.library.items():
            if param_name in recipe.library_defaults:
                # optional (see Recipe.library_defaults). the step runs fine
                # without one, so an empty library isn't a problem we need to
                # fix.
                continue
            kind = kind_by_id(kind_id)
            while not _lib_entries(kind):
                prompter.reason(
                    f"'{recipe.label}' needs at least one of your "
                    f"{kind.label.lower()}, and you have none imported."
                )
                try:
                    what = str(prompter.select(
                        "What would you like to do?",
                        [Choice("import", "Import one now…", kind.help),
                         Choice("drop", f"Leave '{recipe.label}' out"),
                         Choice("keep", "Keep it anyway",
                                "The pipeline will fail at this step unless "
                                "you import one before running it.")],
                    ))
                except GoBack:
                    what = "keep"
                if what == "import":
                    # straight into the file browser. "Import one now" used to
                    # open the library *manager*, whose first question is --
                    # again -- whether to import files. so we had a menu asking
                    # what the user had just finished telling us. not great.
                    import_files(prompter, kind)
                    continue    # go back and check: did an import actually land?
                if what == "drop":
                    drop.append(recipe.id)
                break
    return drop


# ---------------------------------------------------------------------------
# Step 6: per-step options
# ---------------------------------------------------------------------------

def _var_name(value: Any) -> Optional[str]:
    """The variable a template points at, or None if it is not a `{{var:}}`."""
    if isinstance(value, str) and value.startswith("{{var:") and value.endswith("}}"):
        return value[len("{{var:"):-2].strip()
    return None


def _var_behind(recipe: _recipes.Recipe, name: str) -> Optional[str]:
    """
    The preset variable a step parameter reads, if it reads one.

    Unwraps the one-element-list case (``dict_paths: ["{{var:...}}"]``) for the
    same reason :func:`_current_value` does: the function wants a sequence while
    the variable holds a single value, and it is still one variable.
    """
    raw = recipe.with_.get(name, EMPTY)
    direct = _var_name(raw)
    if direct is not None:
        return direct
    if isinstance(raw, list) and len(raw) == 1:
        return _var_name(raw[0])
    return None


def _current_value(recipe: _recipes.Recipe, param: ParamSpec,
                   var_specs: Dict[str, dict]) -> Tuple[Any, Optional[str], str]:
    """
    What a parameter is set to right now, which variable owns it, and how to ask.

    Recipe templates mostly read ``{{var:whisper_model}}`` rather than holding a
    literal. When that is the case we return the *variable's* current value and
    its name, so editing the parameter edits the variable -- which lands in the
    preset's ``vars:`` block where it is named, documented, and overridable from
    the command line later. Only genuinely step-local parameters get written
    into the step itself.

    Returns
    -------
    (current, variable_name, widget)
        `widget` is normally the parameter's own, but see the note below.

    Notes
    -----
    Two recipes wrap a variable in a one-element list --
    ``dict_paths: ["{{var:dictionaries_path}}"]`` -- because the function takes
    a sequence while the variable holds a single folder. Those are unwrapped and
    asked for as plain text. Without that, the user is shown the literal string
    ``{{var:dictionaries_path}}`` as the current value, which means nothing to
    them, and any answer would be written back as a list, rendering the template
    to a list nested inside a list.
    """
    raw = recipe.with_.get(param.name, EMPTY)

    var_name = _var_name(raw)
    if var_name is None and isinstance(raw, list) and len(raw) == 1:
        var_name = _var_name(raw[0])
        if var_name is not None:
            spec = var_specs.get(var_name, {})
            return spec.get("default"), var_name, "text"

    if var_name is not None:
        spec = var_specs.get(var_name, {})
        return spec.get("default"), var_name, param.widget

    if isinstance(raw, str) and "{{var:" in raw and not raw.startswith("{{pick"):
        # a composite like "{{var:features_dir}}/whisper-embeddings" -- built
        # from a variable rather than owned by one. we show it *rendered* with
        # the variables' current values. it used to display as "not set",
        # which a step with a perfectly good destination doesn't deserve.
        rendered = re.sub(
            r"\{\{var:([^}]+)\}\}",
            lambda m: str(var_specs.get(m.group(1).strip(), {}).get("default", "")),
            raw,
        )
        if rendered != raw:
            return rendered, None, param.widget

    if raw is EMPTY or (isinstance(raw, str) and raw.startswith("{{")):
        return (param.default if param.has_default else None), None, param.widget
    return raw, None, param.widget


# the progress rail, in order. renderers that can show it (LivePrompter) keep
# it pinned above the current question; everyone else just ignores stage().
STAGES = [
    ("source", "Source"),
    ("features", "Features"),
    ("analysis", "Analysis"),
    ("options", "Options"),
    ("review", "Review"),
    ("run", "Run"),
]


def _plural(n: int, noun: str) -> str:
    """`1 step`, `3 steps`. Small, but "1 steps" in a summary reads as a bug."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _and(items: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c` -- a list read aloud."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def ask_level(prompter: Prompter, src: SourceSpec,
              steps: Sequence[_recipes.Recipe] = ()) -> Tuple[str, List[str]]:
    """
    Ask what one row of the results should describe.

    The single most consequential answer in the whole wizard, and until now not
    a question at all: the level was fixed per recipe, differently for
    different recipes. Joining a speaker's utterances before measuring versus
    measuring each and averaging differ by about a third on vocabulary
    measures, so a level chosen on the user's behalf is a silent methodological
    decision in someone's results.

    Asked once, for the whole pipeline, rather than per feature. Mixed levels
    would give feature tables with different row counts that cannot be joined
    on anything -- which is precisely the defect this replaced on the CSV path,
    where readability emitted a row per spreadsheet row while sentence
    embeddings emitted one per participant.

    Returns
    -------
    (level_id, group_by)
        ``group_by`` is empty unless the chosen level leaves the columns to the
        user, which only a spreadsheet does.
    """
    # nothing to decide if nothing in this pipeline measures text. a run of
    # acoustics alone has its own grain (one row per speaker's audio) that
    # the level doesn't set, and shouldn't, so we skip the question.
    if steps and not any(_recipes.level_aware(r) for r in steps):
        return _recipes.DEFAULT_LEVEL[src.source], []

    options = _recipes.levels_for(src.source)
    if len(options) < 2:
        # a folder of .txt files is already one text per file. nothing to
        # decide here, and a question with one answer just wastes a screen.
        return options[0].id, []

    # we used to ask "what is your unit of analysis?" here, and it's the wrong
    # phrase. that one belongs to the statistics -- a researcher reads it as
    # "the thing my tests treat as an observation" -- and using it here,
    # before statistics have even come up, made answering feel like settling
    # the analysis question (people told us so). what this actually decides is
    # the grain of the *feature tables*, so we ask about a row of results
    # instead.
    ids = {o.id for o in options}
    state: Dict[str, Any] = {
        # when we get re-asked (Esc from the analysis stage) we want the
        # earlier answer sitting under the pointer, not the catalog default.
        "level": src.level if src.level in ids else _recipes.DEFAULT_LEVEL[src.source],
        "group_by": list(src.group_by),
    }

    def ask_which_level() -> bool:
        state["level"] = str(prompter.select(
            "What should one row of results describe?",
            [Choice(o.id, o.label, o.help) for o in options],
            default=state["level"],
        ))
        return True

    def ask_group_columns() -> bool:
        spec = _recipes.level_by_id(src.source, state["level"])
        if spec.group_by is not None:
            state["group_by"] = []
            return False
        # only a spreadsheet gets here. its grouping columns are the user's
        # own, so the catalog can't name them for us.
        spoken_for = set(src.text_cols) | set(src.feature_cols)
        remaining = [c for c in src.columns if c not in spoken_for]
        if not remaining:
            # a single-column file, or every column ticked as text (or, in the
            # analyze flow, as a predictor). an empty checkbox isn't a
            # question (questionary crashes on one), and there's really only
            # one honest outcome here anyway.
            prompter.reason(
                "Every column is already being analyzed, so there is "
                "nothing left to group rows by. Using one row per "
                "spreadsheet row instead."
            )
            state["level"], state["group_by"] = \
                _recipes.DEFAULT_LEVEL[src.source], []
            return False
        try:
            _cols, sample = peek_csv(src.path, delimiter=src.delimiter)
        except Exception:
            sample = []
        state["group_by"] = ask_at_least_one(
            prompter, "Combine rows that share which column(s)?",
            _kind_rows(remaining, src.kinds, checked=state["group_by"]),
            thing="one column",
            cycle=_cycler(src.kinds, sample, remaining))
        return True

    _ask_in_order([ask_which_level, ask_group_columns])
    return state["level"], list(state["group_by"])


@dataclass
class AnalysisSpec:
    """What the optional analysis stage decided.

    Empty ``analyses`` means the stage was skipped or answered "none", and
    nothing about the pipeline changes -- which is the common case: plenty of
    datasets have nothing to test, and plenty of users just want the feature
    tables.
    """

    analyses: List[str] = field(default_factory=list)
    group_col: str = ""
    outcome_cols: List[str] = field(default_factory=list)
    #: The categorical column(s) to predict. Separate from `outcome_cols`
    #: because predicting which and predicting how much are different
    #: questions: one run can sensibly do both, and each step refuses the
    #: other's column by name rather than fitting it badly.
    class_cols: List[str] = field(default_factory=list)
    control_cols: List[str] = field(default_factory=list)
    #: Which of `control_cols` are categories rather than measurements. Kept
    #: separately because a 1/2 gender code looks numeric and is not: read
    #: as a measurement it would say "each step from female to male adds
    #: 0.4 units", which is not a sentence about anything.
    categorical_controls: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)   # recipe ids; [] = all
    per_table: bool = False
    p_adjust: str = "fdr_bh"
    filters: List[list] = field(default_factory=list)
    #: Values to keep, per column, decided when a check found a class, group
    #: or control level too thin to use: ``{"gender": ["Female", "Male"]}``.
    #: Kept apart from `filters` so re-answering the filter question cannot
    #: silently undo a rescue made three screens earlier.
    value_filters: Dict[str, List[str]] = field(default_factory=dict)
    #: Recipe ids a filter answer implies -- asking to filter on word count
    #: is asking for the step that counts words.
    extra_features: List[str] = field(default_factory=list)
    #: Why the stage asked nothing, when it asked nothing. Shown on the rail
    #: so the answer is still there several screens later.
    why_not: str = ""
    #: Whether the stage actually put a question on screen. False when this
    #: run cannot support statistics at all -- and then the stage must not
    #: absorb an Esc from the screen after it, or "back" would land on a
    #: stage that asks nothing and bounce straight forward again.
    offered: bool = False

    @property
    def wanted(self) -> bool:
        return bool(self.analyses)


#: Filter operators offered for a numeric column, and what they say on screen.
_NUMERIC_OPS = [(">=", "is at least"), (">", "is more than"),
                ("<=", "is at most"), ("<", "is less than"),
                ("==", "equals"), ("!=", "is not")]

#: ... and for a text column. `in`/`not_in` are not offered: typing a list is
#: a worse experience than adding two filters, and the spec supports them for
#: YAML users who want them.
_TEXT_OPS = [("==", "is"), ("!=", "is not")]


def _no_saved_model(recipe: _recipes.Recipe) -> str:
    """Why an apply-a-saved-model step cannot run yet, if it cannot.

    Scoring with a model you fitted earlier needs that model to exist. The
    preflight screen offers to import one, but by then the user has already
    chosen it from a list that gave no hint it was unavailable -- and on a
    checklist that insists on an answer, an unusable row is one of only a
    few things they can pick.

    **Only** saved models, though. A model is the one library asset that
    cannot be obtained any other way: it comes out of a previous Taters run,
    so "import one" is no help to someone who has never fitted one. Every
    other kind is something the user brings or that ships with Taters --
    dictionaries, archetypes, connective lists, stop word lists -- and for
    those an empty library is an ordinary starting state that the preflight
    screen's "import one now" handles perfectly well.

    Applied to every library-backed row this grayed out *dictionary counts*
    and *archetype similarity*, and told the user they needed "a model you
    saved from an earlier run" (a real report). Neither is a model: an
    archetype run uses a sentence-transformers model to embed the text, but
    what it scores against is a dictionary of definitions the user supplies,
    exactly like LIWC-style counting. Nothing about it waits on an earlier
    run.
    """
    from ..helpers.model_spec import LIBRARY_KIND

    kinds = set(recipe.library.values())
    if LIBRARY_KIND not in kinds:
        return ""
    try:
        from ..helpers.library import KINDS, entries

        if entries(KINDS[LIBRARY_KIND]):
            return ""
    except Exception:      # pragma: no cover - a library that cannot be read
        return ""
    # short because it has to be: questionary draws this after the label on
    # the same line, and the label is already 33 characters wide.
    return "none saved yet; import one in Settings"


def _unrunnable(analyses: Sequence[_recipes.Recipe],
                numeric: Sequence[str],
                label_cols: Sequence[str] = ()) -> dict:
    """
    Why each analysis cannot run on this data, for the ones that cannot.

    Grayed out with a reason rather than hidden, and rather than accepted
    and quietly dropped afterwards -- which is what used to happen: ticking
    "Correlations" on a spreadsheet of nothing but labels was taken, then
    discarded with a note, leaving the stage with nothing chosen and the
    wizard bouncing back to a screen that could not fix it.
    """
    # we keep the reasons terse here. questionary draws them after the label
    # on the same line, and "Classification model (cross-validated logistic)"
    # leaves us about two dozen columns at 80. the sentence saying what the
    # spreadsheet lacks goes above the list instead.
    out: dict = {}
    if not numeric:
        why = "needs a column of numbers"
        for r in analyses:
            if r.outcome_kind == "numeric":
                out[r.id] = why
    if not label_cols:
        why = "needs a column of labels"
        for r in analyses:
            # classification needs the same thing group differences does:
            # labels that repeat. a column of unique strings would give us one
            # class per row, which isn't a category anyone (or anything) can
            # learn.
            if r.outcome_kind == "labels":
                out[r.id] = why
    for r in analyses:
        if r.id not in out:
            missing_model = _no_saved_model(r)
            if missing_model:
                out[r.id] = missing_model
    return out


def _analysis_possible(src: SourceSpec, steps: Sequence[_recipes.Recipe]
                       ) -> Tuple[bool, str]:
    """
    Whether to offer statistics at all, and why not when the answer is no.

    Three honest reasons to skip: the source has no metadata columns to test
    against (a folder of documents, or media whose transcripts carry only
    source and speaker); nothing selected produces a joinable per-text
    feature table; or the spreadsheet is being measured one text column at a
    time, which repeats each row per column and leaves the join ambiguous.
    The reason is shown rather than swallowed -- a stage that silently is not
    there reads as a stage that is broken.
    """
    if src.source != "csv":
        return False, ("Statistics need a spreadsheet with grouping or "
                       "outcome columns in it; this run reads its text from "
                       f"{'a folder of documents' if src.source == 'txt_dir' else 'media files'}.")
    if not any(r.feature_table for r in steps):
        return False, ("Nothing selected produces a table with one row per "
                       "text, so there is nothing to run statistics on yet.")
    spare = [c for c in src.columns
             if c not in src.text_cols and c not in src.feature_cols]
    if not spare:
        return False, ("Every column is being analyzed, so there is "
                       "nothing left to group by or predict.")
    return True, ""


#: Short menu labels for the corrections; the one-line explanation beside
#: each comes from the stats module, so the two can never disagree.
_ADJUST_LABELS = {
    "none": "None — raw p-values",
    "fdr_bh": "False discovery rate (Benjamini-Hochberg)",
    "fdr_by": "False discovery rate, any dependence (Benjamini-Yekutieli)",
    "holm": "Family-wise error (Holm-Bonferroni)",
    "bonferroni": "Family-wise error (Bonferroni)",
}

#: Rows on the filter list that are not spreadsheet columns. NUL-prefixed
#: like the other UI sentinels so they can never collide with a real name.
_MEASURED_COLUMN = "\x00measured"
_WORD_COUNT = "\x00word_count"

#: What ticking the word-count row asks for: the column, and the step that
#: writes it. A length filter is the commonest thing anyone wants here and it
#: used to be the one thing you could not simply pick -- you had to know that
#: readability calls its count `lexicon_count`, and to have selected
#: readability (a real report). Now it is the first row on the list and the
#: step that produces it is added for you.
_WORD_COUNT_COLUMN = "word_count"
_WORD_COUNT_RECIPE = "word_count"

#: The two places a filterable variable can come from.
_FILTER_SOURCES = [
    ("generated", "Measures this run will produce",
     "Word count, and anything else the steps you picked measure."),
    ("original", "Columns from your spreadsheet",
     "What was in the file before Taters touched it."),
]


def _ask_filters(prompter: Prompter, columns: Sequence[str],
                 sample: Sequence[dict],
                 tables: Sequence[str] = (),
                 values_of: Optional[Callable[[str], Optional[List[str]]]] = None,
                 kinds: Optional[Dict[str, str]] = None
                 ) -> Tuple[List[list], List[str]]:
    """
    The "select * where" conversation: which rows to keep before analyzing.

    Three steps rather than a loop of one-at-a-time questions. First, where
    the variables come from -- the run's own measures, the spreadsheet's
    columns, or both -- which is what keeps a 150-column file from filling
    the next screen. Then one list of everything in the chosen places, ticked
    as many at a time as you like. Then each ticked variable in turn, to set
    its condition. Picking the variables and setting the conditions are
    different kinds of decision, and interleaving them (the first version of
    this) meant re-answering "and another?" between every one.

    A column of labels is filtered by ticking the values to keep -- "only
    males and females" is one checkbox, not two typed conditions -- when
    ``values_of`` can read the whole column and it has few enough distinct
    values to list. Everything else takes an operator and a value.

    Returns the filters and any recipe ids the answers imply.
    """
    filters: List[list] = []
    extra: List[str] = []
    values_of = values_of or (lambda c: [str(r.get(c) or "") for r in sample])
    prompter.reason(
        "A filter drops rows before any statistics run -- the usual one is a "
        "minimum length, because a three-word answer makes most language "
        "measures meaningless. Rows with a blank in the filtered column are "
        "dropped too: an unknown word count is not evidence of a long text.")
    if not prompter.confirm("Ignore any rows before analyzing?", default=False):
        return filters, extra

    places = ask_at_least_one(
        prompter, "Filter on what?",
        [Choice(key, label, help, checked=True)
         for key, label, help in _FILTER_SOURCES],
        thing="one place")

    rows: List[Choice] = []
    if "generated" in places:
        # this one always goes first, since it's what people mean by "filter"
        # nearly every time.
        rows.append(Choice(_WORD_COUNT, "Word count",
                           "How many words each text has — counted for you, "
                           "whether or not you picked anything that measures "
                           "length."))
        rows.append(Choice(_MEASURED_COLUMN, "Something else this run "
                           "measures…",
                           ("A column from " + ", ".join(tables) if tables
                            else "A column from the features being extracted")
                           + " — type its name."))
    if "original" in places:
        rows += _kind_rows(
            columns, kinds or {},
            help_of=lambda c: ", ".join(distinct_values(sample, c, limit=4))
            or "no values sampled")

    picked = ask_at_least_one(prompter, "Which variable(s) do you want to "
                              "filter on?", rows, thing="one variable")

    for value in picked:
        if value == _WORD_COUNT:
            column, numeric = _WORD_COUNT_COLUMN, True
            if _WORD_COUNT_RECIPE not in extra:
                extra.append(_WORD_COUNT_RECIPE)
        elif value == _MEASURED_COLUMN:
            prompter.reason(
                "It has to be a column one of these tables will contain: "
                + (", ".join(tables) if tables else "the ones you selected")
                + ". The name is checked once the features exist, and a wrong "
                "one stops the run at the last step.")
            column = str(prompter.text(
                "Name of the measure to filter on:", default="")).strip()
            if not column:
                prompter.note("  Skipped: a filter needs a column.",
                              style="yellow")
                continue
            numeric = True
        else:
            column = value
            numeric = (kinds.get(column) == "numbers") if kinds \
                else looks_numeric(sample, column)
            whole = values_of(column)
            if not numeric and whole is not None:
                counts = _checks.level_counts(whole)
                if 2 <= len(counts) <= _checks.MAX_CLASSES \
                        and len(counts) < len(whole):
                    kept = ask_at_least_one(
                        prompter, f"Keep rows where {column} is…",
                        [Choice(v, v, annotation=f"{n} row{'' if n == 1 else 's'}",
                                checked=True)
                         for v, n in counts.most_common()],
                        thing="one value")
                    if len(kept) == len(counts):
                        prompter.note(f"  Every value of {column} kept, so "
                                      f"no filter.", style="dim")
                    else:
                        filters.append([column, "in", kept])
                        prompter.note(f"  Filter: keep rows where {column} is "
                                      f"one of {', '.join(kept)}", style="dim")
                    continue

        ops = _NUMERIC_OPS if numeric else _TEXT_OPS
        op = str(prompter.select(
            f"Keep rows where {column}…",
            [Choice(symbol, f"{symbol}  ({words})") for symbol, words in ops]))
        while True:
            raw = str(prompter.text(f"Keep rows where {column} {op}:",
                                    default="")).strip()
            if not raw:
                prompter.note("  A filter needs a value.", style="yellow")
                continue
            if numeric:
                try:
                    parsed: Any = float(raw)
                except ValueError:
                    prompter.note(f"  {raw!r} is not a number, and {column} "
                                  f"holds numbers.", style="yellow")
                    continue
                if parsed == int(parsed):
                    parsed = int(parsed)
            else:
                parsed = raw
            break
        filters.append([column, op, parsed])
        prompter.note(f"  Filter: keep rows where {column} {op} {parsed}",
                      style="dim")
    return filters, extra


def _label_choices(src: "SourceSpec", sample: Sequence[dict],
                   spare: Sequence[str], kinds: Dict[str, str],
                   label_cols: Sequence[str]) -> List[Choice]:
    """
    The columns that can name what a row *is*: a condition, a diagnosis, a
    moderator flag. Shared by the group-comparison and classification
    questions, which want exactly the same thing and would otherwise drift
    apart -- one offering a column the other refuses is the kind of
    inconsistency that reads as a bug in the screen.

    On a combined run the offer is NOT limited to the columns combined on. A
    column that is a fact about the thing being combined -- `is_moderator`
    for an author, `condition` for a participant -- has one value per group
    too, and restricting the offer to the keys made "combine each author's
    comments, then compare moderators with regular users" impossible to
    express (a real report). What cannot be used is a column that differs
    from row to row inside a group: the gatherer leaves those blank rather
    than reporting the first row's value, so choosing one would drop every
    row.
    """
    could = set(label_cols)
    if src.group_by:
        steady = [c for c in [*src.group_by,
                              *(c for c in spare if c not in src.group_by)]
                  if c in src.group_by
                  or constant_within(sample, src.group_by, c)]
        varies = [c for c in spare if c not in steady]
    else:
        steady, varies = list(spare), []
    offered = [c for c in steady if c in could or c in src.group_by]
    # grayed out rather than hidden, and with the reason attached. a column
    # the user can see in their spreadsheet but can't find on this screen
    # looks like a bug in the screen. a column of numbers with few repeating
    # values gets offered (the arrows make it labels); one with a value per
    # row is a measurement and gets grayed as such; free text can never name
    # a group.
    disabled = {c: f"differs within each {' + '.join(src.group_by)}; no single value"
                for c in varies}
    rest = [c for c in steady if c not in offered]
    for c in rest:
        disabled[c] = {"numbers": "a measurement, not labels",
                       "text": "free text", "empty": "empty"}.get(
            kinds.get(c, "text"), "not labels")
    return _kind_rows(
        [*offered, *rest, *varies], kinds,
        help_of=lambda c: ", ".join(distinct_values(sample, c, limit=6))
        or "no values sampled",
        disabled=disabled)


def _kind_rows(columns: Sequence[str], kinds: Dict[str, str], *,
               checked: Sequence[str] = (),
               help_of: Optional[Callable[[str], str]] = None,
               disabled: Optional[Dict[str, str]] = None) -> List[Choice]:
    """
    Column names as a two-column table: the name, then what it is treated as.

    The kind rides in the row's annotation -- its own color, lined up by
    padding every name to the widest -- so a picker reads as a table of
    ``name | numbers`` rather than a list of words, and the treatment that
    decides what a column can be used for is visible where the choice is
    made. The arrows change it in place (see `_cycler`).
    """
    ticked = set(checked)
    return [Choice(c, c.strip(),
                   help=help_of(c) if help_of else "",
                   checked=c in ticked,
                   annotation=_aligned(c, kinds.get(c, ""), columns),
                   disabled=(disabled or {}).get(c, ""))
            for c in columns]


def _aligned(column: str, kind: str, columns: Sequence[str]) -> str:
    """The kind, padded so it lines up under one heading whatever the name's
    length -- the padding is in the annotation, not the label, so the label
    stays the tidy column name the person reads back."""
    width = max((len(c.strip()) for c in columns), default=0)
    return " " * (width - len(column.strip())) + kind


def _cycler(kinds: Dict[str, str], sample: Sequence[dict],
            columns: Sequence[str] = ()):
    """
    The left/right arrows on a column row: change how it is treated.

    Numbers may become labels when the values repeat and are few (a coded
    gender or condition); labels may go back to numbers only when every
    value parses; free text stays free text. The change is written to the
    source spec, so every later screen -- and the statistics -- read it.
    Returns the row's new annotation, or None when the row cannot change.
    """
    def cycle(value: str, direction: int) -> Optional[str]:
        current = kinds.get(value, "text")
        options = kind_options(current, [r.get(value) for r in sample])
        if len(options) < 2:
            return None
        at = options.index(current)
        new = options[(at + direction) % len(options)]
        kinds[value] = new
        return _aligned(value, new, columns or [value])
    return cycle


def ask_analysis(prompter: Prompter, src: SourceSpec,
                 steps: Sequence[_recipes.Recipe],
                 required: bool = False,
                 picked: Optional[Sequence[str]] = None) -> AnalysisSpec:
    """
    The optional statistics stage: what to test, on what, over which features.

    Everything here is skippable -- the checklist takes an empty answer,
    unlike the feature one -- because the feature tables are the deliverable
    for plenty of runs and statistics are a bonus. When the source cannot
    support statistics at all, the stage says why instead of vanishing.

    ``required=True`` is the "+ run analyses" flow, where the user has
    already said statistics are the point: the checklist then insists on an
    answer, because an empty one there is not a decision but a dead end.
    """
    spec = AnalysisSpec()
    possible, why = _analysis_possible(src, steps)
    if not possible:
        # yellow, not dim. this is a thing you asked for and can't have, so
        # it's worth reading. it used to be dim, and a dim note gets drawn
        # above the *next* screen -- so you'd answer the question before this
        # one, land on the options screen, and it looked exactly like
        # statistics were never offered at all (we got a bug report saying
        # just that).
        prompter.note(f"\n  No statistics this time. {why}", style="yellow")
        spec.why_not = why
        return spec
    spec.offered = True

    # this is the same list the composer will wire into the assemble step,
    # named the same way. we used to build it separately, and the two
    # disagreed: the picker offered "Sentence embeddings" *and* "Merge
    # sentence embeddings" like they were two feature sets, when only one of
    # them was ever going to get joined (yep, someone noticed).
    tables = feature_tables(steps, src.source, src.level or None,
                            src.group_by, picked=picked)
    columns, sample = peek_csv(src.path, delimiter=src.delimiter)
    # the columns we've still got to describe a row with, i.e. not the text
    # itself.
    # what we have left to describe a row with: not the text itself, and not
    # the columns that *are* the measures in the analyze-a-spreadsheet flow.
    # a column cannot be both a predictor and the thing being predicted
    spare = [c for c in columns
             if c not in src.text_cols and c not in src.feature_cols]

    reason = ("Taters can also run the statistics for you, on the features "
              "it just extracted: differences between groups, correlations "
              "with an outcome. Results land in a stats_results folder as "
              "tidy tables plus a plain-English report."
              + ("" if required else
                 " Tick nothing to just get the feature tables."))
    if src.text_mode == "separate" and len(src.text_cols) > 1:
        reason += (" You asked for each text column to be measured "
                   "separately, so each is analyzed separately too -- one "
                   "set of results per column, which is what keeps one "
                   "person's several answers from counting as several "
                   "independent observations.")
    catalog = list(_recipes.user_facing(src.source, stage="analyze"))
    # what each column gets treated as, carried over from the source stage
    # (or detected right here for a spec built without it). `could_be` is the
    # wider question -- what a column can be *turned into* with the arrows --
    # and that's what decides whether an analysis is possible at all.
    kinds = src.kinds or {c: column_kind([r.get(c) for r in sample])
                          for c in columns}
    src.kinds = kinds

    def could_be(name: str, kind: str) -> bool:
        return kind in kind_options(kinds.get(name, "text"),
                                    [r.get(name) for r in sample])

    # offered as numbers: whatever holds numbers, or labels that all parse.
    # offered as labels: whatever is labels, or numbers that plausibly are
    # (few, repeating). any column of numbers can still be *made* into labels
    # with the arrows on an earlier picker (the kind then says so here), but
    # we don't go offering a column with a value per row as a group unasked.
    numeric = [c for c in spare if could_be(c, "numbers")]
    label_cols = [c for c in spare
                  if kinds.get(c) == "labels"
                  or (kinds.get(c) == "numbers"
                      and plausible_labels([r.get(c) for r in sample]))]
    blocked = _unrunnable(catalog, numeric, label_cols)

    def settle(name: str, kind: str) -> bool:
        """Treat `name` as `kind`, saying so; False when it cannot be."""
        if kinds.get(name) == kind:
            return True
        if could_be(name, kind):
            kinds[name] = kind
            prompter.note(f"  Treating {name} as {kind} for this.", style="dim")
            return True
        prompter.note(
            f"  {name} holds {kinds.get(name, 'text')} and cannot be treated "
            f"as {kind}"
            + (f": {len(set(str(r.get(name) or '').strip() for r in sample))}"
               f" distinct values in the sample is a measurement or an "
               f"identifier, not a set of categories." if kind == "labels"
               else "."), style="yellow")
        return False
    if blocked:
        # the row itself only has room to name what's missing, so this is
        # where we get to say that it's the spreadsheet that lacks it.
        reason += (" A grayed-out row needs a kind of column this "
                   "spreadsheet does not have"
                   + (": a column of numbers to relate the features to"
                      if not numeric else "")
                   + (", a column whose labels repeat" if not label_cols
                      else "") + ".")
    offered = [Choice(r.id, r.label, r.help, disabled=blocked.get(r.id, ""))
               for r in catalog]
    grouped = bool(src.group_by)
    # every question below is one step of `_ask_in_order`, so Esc means the
    # previous question and not the feature checklist. a step that doesn't
    # apply to what got ticked clears the answer it owns and asks nothing --
    # that way, going back to untick "Correlations" leaves no outcome column
    # lying around for `apply_analysis` to wire up later.
    state: Dict[str, Any] = {"stop": False, "controls": False, "chosen": []}

    # the checks below read whole columns, not the sample. a class of three
    # or an "n/a" on row 812 is invisible in 200 rows and fatal at the last
    # step. we read once per column, on demand. when rows are being combined
    # the raw column isn't what the analysis sees anyway, so we vet nothing.
    full: Dict[str, List[str]] = {}

    def column(name: str) -> Optional[List[str]]:
        if grouped:
            return None
        if name not in full:
            try:
                full.update(read_columns(src.path, [name],
                                         delimiter=src.delimiter))
            except Exception:
                return None
        return full.get(name)

    def vet(finding: Optional[_checks.Finding]) -> bool:
        """Show a finding; offer the fix when there is one. False means the
        question has to be asked again."""
        if finding is None:
            return True
        prompter.note(f"  {finding.message}", style="yellow")
        if finding.fatal:
            return False
        if not finding.keep:
            return True
        counts = _checks.level_counts(full.get(finding.column) or [])
        rows = [Choice(v, v, annotation=f"{n} row{'' if n == 1 else 's'}",
                       checked=v in finding.keep)
                for v, n in counts.most_common()]
        kept = ask_at_least_one(
            prompter, f"Which values of {finding.column} should stay in "
                      f"the analysis?", rows, thing="one value")
        if len(kept) == len(rows):
            spec.value_filters.pop(finding.column, None)
            prompter.note("  Keeping every row.", style="dim")
        else:
            spec.value_filters[finding.column] = kept
            prompter.note(f"  Filter: keep rows where {finding.column} is "
                          f"one of {', '.join(kept)}", style="dim")
        return True

    def ask_which() -> bool:
        prompter.reason(reason)
        rows = [Choice(c.value, c.label, c.help, checked=c.value in spec.analyses,
                       disabled=c.disabled) for c in offered]
        if required:
            picked = ask_at_least_one(
                prompter, "Which statistics do you want?", rows,
                thing="one analysis")
        else:
            picked = list(prompter.checkbox(
                "Run any statistics on the results? (optional)", rows))
        spec.analyses = picked
        state["stop"] = not picked
        return True

    def picked_labels(kind: Optional[str] = None,
                      only: Optional[str] = None) -> List[str]:
        """The analyses ticked that this question serves, by name."""
        return [_recipes.by_id(a).label for a in spec.analyses
                if (only is None or a == only)
                and (kind is None or _recipes.by_id(a).outcome_kind == kind)]

    def ask_group() -> bool:
        if state["stop"] or "stats_group_differences" not in spec.analyses:
            spec.group_col = ""
            return False
        prompter.reason(
            f"For {_and(picked_labels(only='stats_group_differences'))}. "
            "This is a different question from how the rows were combined. "
            "Combining decided what one row *is*; this decides what you want "
            "to *compare* -- a condition, a diagnosis, moderators against "
            "regular users. Every feature is then tested across those "
            "labels."
            + (f" Rows are being combined on "
               f"{', '.join(src.group_by)}, so a column can only be used "
               f"here if it has one value per combined row." if grouped
               else ""))
        while True:
            rows = _label_choices(src, sample, spare, kinds, label_cols)
            spec.group_col = str(prompter.select(
                "Which column separates the groups you want to compare?", rows,
                default=spec.group_col if any(
                    c.value == spec.group_col and not c.disabled for c in rows)
                else None,
                cycle=_cycler(kinds, sample, [c.value for c in rows])))
            if not settle(spec.group_col, "labels"):
                continue
            values = column(spec.group_col)
            if values is None or vet(_checks.small_groups(values,
                                                            spec.group_col)):
                return True

    def ask_outcomes() -> bool:
        if state["stop"] or not any(
                _recipes.by_id(a).outcome_kind == "numeric" for a in spec.analyses):
            spec.outcome_cols = []
            return False
        # no "what if there are no numeric columns" branch here. those rows
        # get grayed out when there are none, so if we've made it this far
        # `numeric` is non-empty. it used to be accept-then-drop, which left
        # the stage with nothing chosen.
        # we name the analyses this serves, and say what comes next. with a
        # classifier also ticked, this screen read as *the* outcome question,
        # so people picked their category column here (or hunted for it here
        # and came up empty), not knowing a separate question for categories
        # was one screen away.
        wants = picked_labels(kind="numeric")
        prompter.reason(
            f"For {_and(wants)}, which "
            f"{'relate' if len(wants) > 1 else 'relates'} the language to "
            "*numbers*: test scores, ratings, personality measures. Only "
            "columns that hold numbers are offered."
            + (" The classification model's categories are a separate "
               "question, asked next."
               if "stats_classify_fit" in spec.analyses else "")
            + (" Rows are being combined, so each outcome becomes the "
               "average for its group." if grouped else ""))
        while True:
            # only what holds numbers, or could be read as numbers. a column
            # of words has nothing to correlate, and the reason line above
            # already says as much.
            rows = _kind_rows(
                numeric, kinds, checked=spec.outcome_cols,
                help_of=lambda c: ", ".join(distinct_values(sample, c, limit=6))
                or "no values sampled")
            spec.outcome_cols = ask_at_least_one(
                prompter, "Which column(s) hold the outcomes?", rows,
                thing="one column",
                cycle=_cycler(kinds, sample, [c.value for c in rows]))
            if not all([settle(c, "numbers") for c in spec.outcome_cols]):
                continue
            findings = [_checks.non_numeric_outcome(column(c) or [], c)
                        for c in spec.outcome_cols if column(c) is not None]
            if all(vet(f) for f in findings if f is not None):
                return True

    def ask_classes() -> bool:
        if state["stop"] or "stats_classify_fit" not in spec.analyses:
            spec.class_cols = []
            return False
        prompter.reason(
            f"For {_and(picked_labels(only='stats_classify_fit'))}, which "
            "learns which *category* a row falls in rather than how much of "
            "something it has -- a diagnosis, a condition, which of two "
            "authors wrote it"
            + (". A different question from the outcomes you just chose: "
               "those were the numbers"
               if picked_labels(kind="numeric") else "")
            + ". Only columns whose labels repeat are offered, because a "
            "column of one-off values would be one class per row."
            + (" Rows are being combined, so the category has to describe "
               "the whole combined row." if grouped else ""))
        while True:
            rows = _label_choices(src, sample, spare, kinds, label_cols)
            for row in rows:
                row.checked = row.value in spec.class_cols
            spec.class_cols = ask_at_least_one(
                prompter, "Which column(s) hold the category to predict?", rows,
                thing="one column",
                cycle=_cycler(kinds, sample, [c.value for c in rows]))
            if not all([settle(c, "labels") for c in spec.class_cols]):
                continue
            findings = [_checks.thin_classes(column(c) or [], c)
                        for c in spec.class_cols if column(c) is not None]
            if all(vet(f) for f in findings if f is not None):
                return True

    def eligible_controls() -> List[str]:
        # not the identifier columns. `pid` is unique per row, so holding it
        # constant would either explain the outcome away entirely or mean
        # nothing, and offering it just invites that mistake. not the group
        # or the outcomes either -- controlling for the thing you're comparing
        # removes the comparison.
        used = (set(spec.outcome_cols) | set(spec.class_cols)
                | {spec.group_col} | set(src.id_cols))
        # a measurement can be a control, and so can a label that repeats.
        # what can't is a column of words unique to every row -- `pid`, a
        # response id, a filename. same test the grouping question uses, for
        # the same reason. `id_cols` on its own isn't enough here because a
        # run that declined to name its identifier columns still has one
        # sitting in the spreadsheet.
        return [c for c in spare
                if c not in used
                and kinds.get(c) in ("numbers", "labels")
                # on a combined run a control has to speak for the whole
                # combined row, same as a grouping column does.
                and (not grouped or c in src.group_by
                     or constant_within(sample, src.group_by, c))]

    def ask_controls_gate() -> bool:
        if state["stop"] or not eligible_controls():
            state["controls"] = False
            spec.control_cols, spec.categorical_controls = [], []
            return False
        prompter.reason(
            "A control is something you want held constant so it cannot "
            "explain your result: age, gender, how long the text is. Group "
            "comparisons become ANCOVA and report adjusted means, "
            "correlations become partial correlations, and the prediction "
            "step also fits the controls on their own -- so you can see "
            "what the language added over them rather than guessing.")
        state["controls"] = bool(prompter.confirm(
            "Control for any of your other columns?",
            default=bool(spec.control_cols)))
        if not state["controls"]:
            spec.control_cols, spec.categorical_controls = [], []
        return True

    def ask_which_controls() -> bool:
        if state["stop"] or not state["controls"]:
            return False
        prompter.reason(
            "Each column shows how it will be held constant: as numbers "
            "(a measurement) or as labels (a category, one coefficient per "
            "level). A category that happens to be numbered -- a 1/2 gender "
            "code -- reads as numbers; press → on its row to make it labels.")
        eligible = eligible_controls()
        spec.control_cols = ask_at_least_one(
            prompter, "Which column(s) should be held constant?",
            _kind_rows(eligible, kinds, checked=spec.control_cols,
                       help_of=lambda c: ", ".join(distinct_values(sample, c, limit=6))
                       or "no values sampled"),
            thing="one column", cycle=_cycler(kinds, sample, eligible))
        spec.categorical_controls = [c for c in spec.control_cols
                                     if kinds.get(c) == "labels"]
        return True

    def vet_controls() -> bool:
        """A categorical control's thin levels, offered for dropping. Asks
        only when there is something to say, so a clean column costs no
        screen."""
        if state["stop"] or not spec.categorical_controls:
            return False
        asked = False
        for c in spec.categorical_controls:
            values = column(c)
            finding = _checks.rare_levels(values, c) if values is not None \
                else None
            if finding is not None:
                asked = True
                vet(finding)
        return asked

    def ask_tables() -> bool:
        if state["stop"] or len(tables) <= 1:
            spec.tables = []
            state["chosen"] = [r.id for r, _ in tables]
            return False
        prompter.reason(
            "By default every feature table you extracted feeds the "
            "statistics together. You can narrow that -- and choose whether "
            "to analyze them as one set or one at a time.")
        pre = set(spec.tables) if spec.tables else {r.id for r, _ in tables}
        chosen = ask_at_least_one(
            prompter, "Which feature tables should feed the statistics?",
            [Choice(r.id, name, checked=r.id in pre) for r, name in tables],
            thing="one table")
        spec.tables = chosen if len(chosen) < len(tables) else []
        state["chosen"] = chosen
        return True

    def ask_together() -> bool:
        if state["stop"] or len(state["chosen"]) <= 1:
            spec.per_table = False
            return False
        spec.per_table = str(prompter.select(
            "Analyze them together, or one table at a time?",
            [Choice("together", "Together — one analysis over all of them",
                    "Features from every table side by side; a prediction "
                    "model is also fitted on each table alone and on every "
                    "combination, so you can see what each one adds."),
             Choice("separate", "Separately — one analysis per table",
                    "Repeats the analysis for each table, so you can "
                    "see which feature set does best.")],
            default="separate" if spec.per_table else "together")) == "separate"
        return True

    def ask_adjust() -> bool:
        if state["stop"] or not any(
                "p_adjust" in _recipes.by_id(a).with_ for a in spec.analyses):
            spec.p_adjust = "fdr_bh"
            return False
        from ..stats._common import P_ADJUST_METHODS

        prompter.reason(
            "Testing many features at once finds things by chance: 160 "
            "measures at p < .05 hands you eight 'findings' from noise "
            "alone. The correction decides what counts as a finding, and "
            "which one to use is a methodological choice, not a default "
            "worth hiding -- so it is asked, and recorded in the report.")
        spec.p_adjust = str(prompter.select(
            "How should p-values be corrected for multiple comparisons?",
            [Choice(key, _ADJUST_LABELS[key], words)
             for key, words in P_ADJUST_METHODS.items()],
            default=spec.p_adjust))
        return True

    def ask_row_filters() -> bool:
        if state["stop"]:
            spec.filters, spec.extra_features = [], []
            return False
        spec.filters, spec.extra_features = _ask_filters(
            prompter, spare, sample,
            tables=[name for r, name in tables
                    if not spec.tables or r.id in spec.tables],
            values_of=column, kinds=kinds)
        return True

    _ask_in_order([ask_which, ask_group, ask_outcomes, ask_classes,
                   ask_controls_gate, ask_which_controls, vet_controls,
                   ask_tables, ask_together, ask_adjust, ask_row_filters])
    return spec


def apply_analysis(spec: AnalysisSpec, src: SourceSpec,
                   selected: List[str], var_values: Dict[str, Any],
                   overrides: Dict[str, Dict[str, Any]]) -> None:
    """
    Write an :class:`AnalysisSpec` into the things ``compose`` reads.

    The metadata columns are the subtle part. The statistics read a metadata
    table gathered from the same spreadsheet with the same identity, and when
    rows are being combined a *numeric* column cannot ride along untouched --
    a group of twelve rows has twelve openness scores. Those become the
    group's average, named ``<column>_mean`` by the gatherer, so the outcome
    names handed to the analyses are rewritten to match. Getting that wrong
    is a step that runs happily and correlates nothing, so it is decided here,
    once, next to the reason.
    """
    if not spec.wanted:
        return

    grouped = bool(src.group_by)
    for analysis in spec.analyses:
        if analysis not in selected:
            selected.append(analysis)
    # a filter on a measure nothing produces is a run that dies at the last
    # step, so asking to filter on word count adds the step that counts them.
    # we add it as a *filter* table, not a feature one. wanting to drop short
    # texts isn't the same as wanting text length as a predictor, and a ridge
    # regression happily took it as one (this actually happened to someone).
    for recipe_id in spec.extra_features:
        if recipe_id not in selected:
            selected.append(recipe_id)
    if spec.extra_features:
        overrides.setdefault("stats_assemble", {})["filter_csvs"] = [
            "{{" + _recipes.by_id(rid).save_as + "}}"
            for rid in spec.extra_features]

    carry: List[str] = []
    agg: Dict[str, str] = {}

    if spec.group_col:
        var_values["stats_group_col"] = spec.group_col
        # a grouping column that *is* a grouping key is already sitting in
        # the metadata table as a key column, so if we carried it again we'd
        # write it twice under one heading.
        if spec.group_col not in src.group_by:
            carry.append(spec.group_col)

    if src.feature_cols:
        # the analyze-a-spreadsheet flow: the predictors are columns that are
        # already there. ungrouped we carry them through as they are; grouped
        # we average them, exactly as the outcomes below are averaged, and
        # they arrive in the table as <column>_mean
        if grouped:
            var_values["analysis_predictor_agg"] = {c: "mean"
                                                    for c in src.feature_cols}
            # averaging writes a count beside every column, plus one row
            # count for the group. those describe the combining rather than
            # the participant, so they go aside -- a predictor called
            # `f1_n` is the number of rows somebody had, and a model that
            # learns from it has learned about the data collection
            var_values["stats_bookkeeping_cols"] = (
                ["group_count"] + [f"{c}_n" for c in src.feature_cols])
        else:
            var_values["analysis_predictor_cols"] = list(src.feature_cols)

    if spec.outcome_cols:
        if grouped:
            agg = {c: "mean" for c in spec.outcome_cols}
            var_values["stats_outcome_cols"] = [f"{c}_mean"
                                                for c in spec.outcome_cols]
        else:
            carry += [c for c in spec.outcome_cols if c not in carry]
            var_values["stats_outcome_cols"] = list(spec.outcome_cols)

    if spec.class_cols:
        # we carry these raw even on a combined run, where a numeric outcome
        # gets averaged instead -- the mean of a diagnosis is not a diagnosis.
        # the gatherer leaves a column blank when it differs inside a group,
        # and the question only offered columns that don't, so what shows up
        # is one label per combined row.
        carry += [c for c in spec.class_cols if c not in carry]
        var_values["stats_class_cols"] = list(spec.class_cols)

    if spec.control_cols:
        var_values["stats_control_cols"] = list(spec.control_cols)
        if spec.categorical_controls:
            var_values["stats_categorical_controls"] = list(
                spec.categorical_controls)
        # a control has to make it to the analysis table like any other
        # metadata column, otherwise the step gets handed a column name that
        # isn't there. grouped runs keep the raw column rather than averaging
        # it -- the mean of a gender code is not a gender.
        carry += [c for c in spec.control_cols if c not in carry]

    if carry:
        var_values["stats_meta_carry"] = carry
    if agg:
        var_values["stats_meta_agg"] = agg
    if spec.per_table:
        var_values["stats_feature_sets"] = "per_table"
    if src.text_mode == "separate" and len(src.text_cols) > 1:
        # each text column got measured on its own, so we analyze each on its
        # own too. a participant has one row per column, and pooling them
        # would count one person's several answers as several independent
        # observations. the column the gather writes to say which is which
        # is `source_col`.
        var_values["stats_split_col"] = "source_col"
    if spec.p_adjust != "fdr_bh":
        var_values["stats_p_adjust"] = spec.p_adjust
    kept = [[col, "in", list(values)] for col, values in spec.value_filters.items()]
    if kept or spec.filters:
        var_values["stats_filters"] = kept + [list(f) for f in spec.filters]
    if spec.tables:
        # the composer's default is every selected feature table. narrowing
        # it down is just an ordinary per-step override, which outranks it.
        overrides.setdefault("stats_assemble", {})["feature_csvs"] = [
            "{{" + _recipes.by_id(rid).save_as + "}}" for rid in spec.tables]


def _init_stages(prompter: Prompter, analyses: Optional[bool] = None) -> None:
    """Paint the rail. A stage this run will never visit is left off it: a
    permanently-unvisited "Analysis" on a feature-only run reads as a step
    that got skipped rather than one that was never part of the journey."""
    stages = [s for s in STAGES
              if s[0] != "analysis" or analyses is not False]
    prompter.stage(stages[0][0], stages[0][1], status="active")
    for key, label in stages[1:]:
        prompter.stage(key, label, status="todo")


NEVER_TUNABLE = frozenset({
    "verbose", "run_in_subprocess", "extractor_module", "extra_env",
    # the runner supplies these, not whoever's reading this screen. their
    # docstrings are written for whoever maintains the pipeline, and offering
    # them just invites someone to break their own run by answering.
    "on_progress",
    # where the intermediate table goes. we already point it inside the
    # pipeline's own folder, and nobody needs to revisit that.
    "gathered_csv",
    # how to read a CSV this pipeline wrote itself, one step earlier. we know
    # the answer and the user can't -- getting either wrong turns a working
    # run into a parse error several minutes in.
    "delimiter", "encoding",
    # the text-gathering plumbing every text analyzer carries. our source
    # stage answers these once for the whole pipeline. offered per step they
    # read as questions ("recursive? pattern?") that mean nothing on a CSV
    # path and, worse, could be edited into *disagreement* between two steps
    # that share one gathered table -- where the second step's answers would
    # get silently ignored because the first step already wrote the file.
    "recursive", "id_from", "include_source_path", "joiner",
    "num_buckets", "max_open_bucket_files", "tmp_root",
})


def is_wired(recipe: _recipes.Recipe, name: str) -> bool:
    """
    Is this parameter carrying data from an earlier step?

    A recipe template like ``transcript_csv: "{{pick:diar.raw_files.csv}}"`` is
    the wiring that makes the pipeline a pipeline. Offering it as an editable
    option would let someone quietly disconnect their own run, so those
    parameters are withheld from the options screen entirely.

    ``{{var:...}}`` templates are deliberately *not* wiring: they point at a
    named variable that exists precisely so it can be changed.
    """
    if name in NEVER_TUNABLE or name in recipe.hidden:
        return True
    value = recipe.with_.get(name)
    if not isinstance(value, str):
        return False
    return value.startswith("{{") and not value.startswith("{{var:")


def _parse_as(text: str, param: ParamSpec) -> Any:
    """Turn a typed-in string back into the type the parameter wants."""
    text = text.strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    if param.widget == "int":
        return int(text)
    if param.widget == "float":
        return float(text)
    if param.widget == "list":
        return [piece.strip() for piece in text.split(",") if piece.strip()]
    if param.widget == "choice" and param.choices:
        for choice in param.choices:
            if str(choice) == text:
                return choice
    return text


def _explain(prompter: Prompter, param: ParamSpec, current: Any) -> None:
    """
    Say what a setting is before asking for it.

    Parameter names are written for the people who call these functions from
    Python. `rounding`, `relative_freq`, `mean_center_vectors` mean nothing to
    someone who has never opened the module, and this wizard exists precisely
    for that person -- so a question with only a name on it is a question they
    cannot answer.

    Three things go up, in the order they are useful:

    * what the setting does, in whole sentences from the function's own
      docstring -- previously only the first *line* was shown, which cut most
      descriptions mid-sentence because docstrings are hard-wrapped;
    * what it is set to now, since "leave it alone" is the right answer most of
      the time and needs to be visible;
    * what the valid answers are, when the set is small enough to name.
    """
    from rich.markup import escape

    prompter.note("")
    if param.desc:
        # blank line *between* paragraphs. these used to print back to back,
        # so a two-paragraph description came out as one unbroken slab of gray
        # -- the docstring had marked the break and the screen threw it away.
        for i, paragraph in enumerate(param.desc.split("\n\n")):
            if i:
                prompter.note("")
            # we escape here because docstrings write ``{name: [columns]}``
            # and ``[col, op, value]``, and rich reads a bracketed word as a
            # style tag and swallows it. we had the screen say "{name: }" once.
            prompter.note(f"    {escape(paragraph)}", style="dim")

    # the current value is the one fact that decides the answer, since "leave
    # it alone" is usually right. it used to be the tail of a dim parenthesis
    # at the end of the prose, styled exactly like the prose, so the thing you
    # most needed to see was the hardest thing to find. now we give it its own
    # line and a brighter color.
    shown = _describe_value(current)
    if shown or param.choices:
        prompter.note("")
    if shown:
        prompter.note(f"    Now: {escape(shown)}", style="cyan")
    if param.choices:
        prompter.note("    One of: "
                      + escape(", ".join(str(c) for c in param.choices)),
                      style="dim")


def _describe_library_value(kind_id: str, current: Any,
                            default_names: Sequence[str] = ()) -> str:
    """
    "3 of 5 dictionaries" instead of three absolute paths.

    The row exists to answer "is this set to something I do not want", and a
    slab of ``/home/...`` paths answers slower than a count -- the names are
    one keypress away in the picker.
    """
    from ..helpers.library import entries, expand, kind_by_id

    kind = kind_by_id(kind_id)
    in_library = {str(e.resolve()) for e in entries(kind)}
    total = len(in_library)
    label = kind.label.lower()
    if not isinstance(current, list) or any("{{" in str(v) for v in current):
        if default_names:
            # a curated default (see Recipe.library_defaults). left alone, we
            # apply exactly these when we compose -- so the row names them,
            # and the picker opens with the same set ticked.
            stems = ", ".join(Path(n).stem for n in default_names)
            return f"the default set ({stems})"
        # untouched: the template is still in place, and we resolve that to
        # "everything in the library" when we compose.
        return f"all {total} of your {label}" if total else "none imported yet"
    # a folder in the value means everything in it (that's how the analyzers
    # read it, and what a saved preset's default carries), so we count what a
    # run would actually use, not the raw values.
    picked_paths = {str(p.resolve()) for p in expand(kind, current)}
    if total and picked_paths == in_library:
        return f"all {total} of your {label}"
    picked = sorted(Path(p).stem for p in picked_paths)
    if len(picked) <= 2:
        return " and ".join(picked) or "none"
    return f"{len(picked)} of {total} {label}"


def _describe_value(value: Any) -> str:
    """How a current value should read in a one-line hint."""
    if value is EMPTY:
        return ""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "empty"
    text = str(value)
    return text if text.strip() else "empty"


#: Menu rows that are answers about the answer, not values. Colons keep them
#: from colliding with a real value, which never contains one.
_OTHER = ":other"
_SOME = ":some"
_UNSET = ""          # what `_parse_as` reads as None


def _table_choices(tables: Sequence[Tuple[str, str]], ticked: Sequence[str]
                   ) -> List[Choice]:
    """One tickable row per feature table: the step's name, and the name the
    results will use for it, which is the one the setting stores."""
    on = set(ticked)
    return [Choice(name, f"{label}  ({name})", checked=name in on)
            for name, label in tables]


def _names_in(current: Any) -> List[str]:
    """The feature-set names a `pca`-shaped value holds: a list, or the
    comma-separated string the command line spells it as."""
    if isinstance(current, (list, tuple)):
        return [str(n) for n in current]
    if isinstance(current, str) and current not in ("off", "all", ""):
        return [n.strip() for n in current.split(",") if n.strip()]
    return []


def _pick_pca(prompter: Prompter, label: str, current: Any,
              tables: Sequence[Tuple[str, str]]) -> Any:
    """
    off / all / some of the feature sets -- ticked, never spelled.

    The setting takes "off", "all", or the names of feature sets, and the
    names are file stems the user has never seen (``doc_term_matrix_count``).
    Asked as a text box it wanted exactly that spelling (a real report:
    "I had to type the name of a feature set, or all, or none"). The tables
    this run will make are known, so they are offered.
    """
    chosen = _names_in(current)
    choices = [Choice("off", "off — analyze the measures as they are"),
               Choice("all", "all — reduce every feature set to components")]
    if tables:
        choices.append(Choice(_SOME, "some — choose which feature sets to reduce"))
    default = current if current in ("off", "all") else (_SOME if chosen else "off")
    if default not in {c.value for c in choices}:
        default = "off"
    picked = str(prompter.select(f"{label}:", choices, default=default))
    if picked != _SOME:
        return picked
    ticked = prompter.checkbox("Which feature sets should be reduced to "
                               "components?", _table_choices(tables, chosen))
    if not ticked:
        prompter.note("  Nothing ticked, so this stays off.", style="yellow")
        return "off"
    return list(ticked)


def _pick_tables(prompter: Prompter, label: str, current: Any,
                 tables: Sequence[Tuple[str, str]]) -> Any:
    """A list of feature-table names, ticked from the tables this run makes."""
    if not tables:
        prompter.note("  This run makes no feature tables to choose from.",
                      style="yellow")
        return current
    return list(prompter.checkbox(f"{label} — which feature tables?",
                                  _table_choices(tables, _names_in(current))))


#: Settings whose answers are the run's own feature tables, and how each is
#: asked. Keyed by parameter name: `pca` is the same setting on all four
#: analyses, and no other step has a parameter of that name. A test walks the
#: catalog to keep that so.
_TABLE_PICKERS: Dict[str, Callable] = {
    "pca": _pick_pca,
    "unverified_ok": _pick_tables,
}


def _ask_one_param(prompter: Prompter, param: ParamSpec, current: Any,
                   widget: Optional[str] = None,
                   tables: Optional[Callable[[], List[Tuple[str, str]]]] = None
                   ) -> Any:
    """
    Render a single parameter, in whatever shape suits its type.

    `widget` overrides the parameter's own, for the unwrapped-list case in
    :func:`_current_value`. `tables` names the feature tables this run will
    make (lazily -- they depend on settings changed moments ago), for the
    settings in :data:`_TABLE_PICKERS`.

    Typing is the last resort. Anything with a known set of answers -- a
    ``{...}`` in its docstring, a ``Literal`` in its signature, the feature
    tables of this run -- is picked from a list, because spelling
    ``none_and_all`` or ``doc_term_matrix_count`` into a box is a test the
    person did not sign up for.
    """
    label = param.name.replace("_", " ")
    _explain(prompter, param, current)

    param = replace(param, widget=widget or param.widget)

    picker = _TABLE_PICKERS.get(param.name)
    if picker is not None and tables is not None:
        return picker(prompter, label, current, tables())

    if param.widget == "bool":
        return prompter.confirm(f"{label}?", default=bool(current))
    if param.widget == "choice" and param.choices:
        choices = [Choice(str(c), str(c)) for c in param.choices]
        if param.has_default and param.default is None:
            # a setting that can be left unset gets that as a row, since
            # "auto" is usually the best answer and there's no other way to
            # give it from a list.
            choices.insert(0, Choice(_UNSET, "automatic — leave it unset"))
        if param.open_ended:
            choices.append(Choice(_OTHER, "something else — type it"))
        default = _UNSET if current is None else str(current)
        if default not in {c.value for c in choices}:
            # a value that's off the list (a model folder typed in earlier).
            # the list can't point at it, so we start on "something else".
            default = _OTHER if param.open_ended else None
        picked = str(prompter.select(f"{label}:", choices, default=default))
        if picked != _OTHER:
            return _parse_as(picked, param)
        # fall through to the text box, with the current value there to edit.

    shown = "" if current is None else (
        ", ".join(str(v) for v in current) if isinstance(current, (list, tuple))
        else str(current)
    )
    while True:
        # the str() is there in case a Prompter implementation hands back a
        # non-string for a free-text question; the rest assumes text anyway.
        typed = str(prompter.text(f"{label}:", default=shown))
        if typed.strip() == shown.strip():
            return current      # unchanged, so we keep the original type exactly
        try:
            return _parse_as(typed, param)
        except ValueError:
            # typing "abc" into a number field is an easy slip, and this wizard
            # is aimed at people for whom a ValueError traceback is not a
            # useful reply. so we say what we wanted and ask again.
            wanted = {"int": "a whole number", "float": "a number"}.get(
                param.widget, f"a {param.widget} value")
            prompter.note(f"    '{typed}' is not {wanted}. Try again.", style="yellow")


# menu entries that are actions rather than settings. the colons keep them
# from ever colliding with a parameter name, which is always an identifier.
_DONE = ":done"
_SHARED = ":shared"
_SHARED_LABEL = "Settings used by more than one step"


def shared_variables(steps: Sequence[_recipes.Recipe]) -> Dict[str, Tuple[_recipes.Recipe, str]]:
    """
    The variables more than one step reads, and where to find a spec for each.

    ``overwrite_existing`` is referenced by thirteen of fifteen recipes, and
    ``device``, ``whisper_model`` and ``transcripts_dir`` by three apiece. They
    are one setting each, living in the preset's ``vars:`` block -- so offering
    them on every step's menu asks the same question over and over and implies
    an answer that is local when it is not. Changing ``device`` under
    "Transcript" changes it for the embeddings step too.

    Which ones are shared is counted, not listed, so this stays right as recipes
    come and go.

    Returns
    -------
    dict
        Variable name -> the (recipe, parameter) to borrow a description,
        widget and current value from. Any of the referencing steps would do;
        the first is taken so the order is stable.
    """
    uses: Dict[str, int] = {}
    owner: Dict[str, Tuple[_recipes.Recipe, str]] = {}
    for recipe in steps:
        for name in recipe.with_:
            var = _var_behind(recipe, name)
            if var is None:
                continue
            # we count this even where the parameter itself is held back.
            # `root_dir` on a merge step is wiring nobody should be re-pointing
            # by hand, but it still reads `{{var:transcripts_dir}}` -- so the
            # variable really is shared, and editing it under the step that
            # *does* offer it moves both. if we only counted the offered ends
            # we'd hide that.
            uses[var] = uses.get(var, 0) + 1
            if var not in owner and not is_wired(recipe, name):
                owner[var] = (recipe, name)
    return {var: owner[var] for var, n in uses.items() if n > 1 and var in owner}


def _editable_names(recipe: _recipes.Recipe, spec: FunctionSpec) -> Tuple[List[str], List[str]]:
    """
    The settings this step can offer, split into the ones to show and the rest.

    The cut is "did the recipe write this into its ``with:`` block". That is not
    a claim about which settings matter -- nothing in a signature or a docstring
    tells you that -- but it is a record of the settings someone actually made a
    decision about, in the order they made them, and it costs nobody any
    curation. Everything else is a default nobody has ever considered, and it is
    one keypress away rather than hidden.

    This used to be a hand-written `tunable` tuple per recipe. It read better
    for some steps and worse for others, went stale silently when a function
    grew a parameter, and had to be written again for every new module. Since
    the whole list is one keypress from the screen either way, the curation was
    not buying enough to be worth maintaining.

    Settings bound to *shared* variables are included too, marked as shared on
    the menu. They used to be left out entirely -- they belong to the pipeline,
    not to this step -- but "not on this step's menu" read as "does not exist":
    someone hunting for `lemmatize` under the frequency list concluded exactly
    that. Editing one edits the variable, so every reader updates together.
    """
    def _pipeline_handled(name: str) -> bool:
        # the one universal cut for input plumbing. `compose` REPLACES a text
        # step's whole input group (`TEXT_INPUT_KEYS`) with the pipeline's
        # own gather wiring, whatever the source -- so for a text step,
        # "which files to search for", "which column holds the text" and
        # friends are all decided before this step ever fires, and offering
        # them here would compose an answer the run silently ignores. one
        # rule, every module, no per-recipe patch lists.
        return recipe.text_input and name in _recipes.TEXT_INPUT_KEYS

    everyday = [name for name in recipe.with_
                if not is_wired(recipe, name)
                and not _pipeline_handled(name)
                and spec.get(name) is not None]
    rest = [param.name for param in spec.params
            if not param.required
            and param.name not in everyday
            and not is_wired(recipe, param.name)
            and not _pipeline_handled(param.name)]
    return everyday, rest


def _live_value(recipe: _recipes.Recipe, param: ParamSpec, var_specs: Dict[str, dict],
                overrides: Dict[str, Dict[str, Any]],
                var_values: Dict[str, Any]) -> Tuple[Any, Optional[str], str]:
    """
    What a setting is set to *now*, including changes made moments ago.

    :func:`_current_value` reads the recipe, which never changes. The menu is
    re-drawn after every edit, and a menu that still shows the old value after
    you have just changed it reads as an edit that did not take.
    """
    current, var_name, widget = _current_value(recipe, param, var_specs)
    if var_name is not None:
        if var_name in var_values:
            current = var_values[var_name]
    elif param.name in overrides.get(recipe.id, {}):
        current = overrides[recipe.id][param.name]
    return current, var_name, widget


#: Signatures already read, so opening a step's options twice is instant the
#: second time and does not depend on `sys.modules` for that.
_SPEC_CACHE: Dict[str, Optional[FunctionSpec]] = {}


def _spec_for(recipe: _recipes.Recipe, prompter: Optional[Prompter] = None
              ) -> Optional[FunctionSpec]:
    """The step function's signature, or None with a note when it is not installed."""
    if recipe.target in _SPEC_CACHE:
        return _SPEC_CACHE[recipe.target]
    try:
        spec = describe(load_target(recipe.target))
        _SPEC_CACHE[recipe.target] = spec
        return spec
    except ImportError as e:
        if prompter is not None:
            prompter.note(f"    Cannot read {recipe.label}'s settings until it is "
                          f"installed ({e}).", style="yellow")
        _SPEC_CACHE[recipe.target] = None
        return None


#: What a dependent setting's row starts with, so it reads as belonging to
#: the row above it: "  ↳ pca components — 0" under "pca — all".
# resolved at import for the same reason as live.py's markers: the glyph set
# is settled once per process, and this is a hot string.
INDENT = f"  {glyphs.INDENT} "


def _setting_choice(recipe: _recipes.Recipe, param: ParamSpec, var_specs: Dict[str, dict],
                    overrides: Dict[str, Dict[str, Any]], var_values: Dict[str, Any],
                    *, key: Optional[str] = None,
                    shared_marker: bool = False, indent: bool = False) -> Choice:
    """
    One row of a settings menu: what it is called, and what it is set to.

    The value is on the row because that is the whole question someone is
    answering here -- not "what settings exist" but "is anything set to
    something I do not want". A list of bare names makes them open each one to
    find out.

    `key` renames the row for a shared variable, where the variable's name
    (``transcripts_dir``) is the thing being changed and the parameter's
    (``out_dir``) is one step's local word for it.
    """
    current, _var, _widget = _live_value(recipe, param, var_specs, overrides, var_values)
    shown = _describe_value(current)
    if param.name in recipe.library:
        shown = _describe_library_value(
            recipe.library[param.name], current,
            default_names=recipe.library_defaults.get(param.name, ()))
    label = (key or param.name).replace("_", " ")
    if indent:
        label = INDENT + label
    return Choice(
        key or param.name,
        f"{label} — {shown}" if shown else label,
        # just the first sentence. the whole description goes up when the
        # setting is actually being answered; here it would bury the list.
        help=(param.desc or "").strip().split(". ")[0],
        # the green tag that says this row is the same setting other steps
        # read -- change it here and it changes everywhere. we wrap it in
        # parentheses so it can't read as part of the VALUE ("model: base.en
        # shared" looked like a model named "base.en shared" to someone).
        annotation="(shared)" if shared_marker else "",
    )


def _gate_open(recipe: _recipes.Recipe, spec, name: str,
               var_specs: Dict[str, dict], overrides: Dict[str, Dict[str, Any]],
               var_values: Dict[str, Any]) -> bool:
    """
    Whether a step's setting should be on the menu right now.

    One evaluator for both menus (a step's own and the shared section), so
    the two can never disagree about a row. Fails open twice over: no gate,
    or a gate parameter the step's signature does not have, shows the row.
    """
    gate = _recipes.gate_of(recipe, name)
    if gate is None:
        return True
    gate_param = spec.get(gate[0])
    if gate_param is None:
        return True
    current, _var, _widget = _live_value(recipe, gate_param, var_specs,
                                         overrides, var_values)
    return _recipes.gate_holds(gate, current)


def _menu_order(recipe: _recipes.Recipe, everyday: Sequence[str],
                rest: Sequence[str]) -> List[str]:
    """
    The rows of a step's menu, with every dependent setting directly under
    its gate.

    Everyday settings first, then the rest, as before -- take the dependents
    out and the order is exactly what it was. A dependent follows its gate
    even across that boundary (`pca_max_missing` lives in the rest and sits
    under `pca`, which is everyday). A dependent whose gate is not on the
    menu at all stays where it was.
    """
    base = list(everyday) + list(rest)
    gate_for = {n: _recipes.gate_of(recipe, n)[0] for n in base
                if _recipes.gate_of(recipe, n) is not None}
    out: List[str] = []
    for name in base:
        if name in gate_for and gate_for[name] in base:
            continue                       # this one goes after its gate, below
        out.append(name)
        out.extend(d for d in base if gate_for.get(d) == name)
    return out


def step_rows(recipe: _recipes.Recipe, spec, var_specs: Dict[str, dict],
              overrides: Dict[str, Dict[str, Any]], var_values: Dict[str, Any],
              shared: Sequence[str] = ()) -> List[Choice]:
    """One step's settings as menu rows: ordered, gated, indented."""
    everyday, rest = _editable_names(recipe, spec)
    names = _menu_order(recipe, everyday, rest)
    shared = set(shared)
    rows: List[Choice] = []
    for name in names:
        if spec.get(name) is None or not _gate_open(recipe, spec, name, var_specs,
                                                    overrides, var_values):
            continue
        gate = _recipes.gate_of(recipe, name)
        rows.append(_setting_choice(
            recipe, spec.get(name), var_specs, overrides, var_values,
            shared_marker=_var_behind(recipe, name) in shared,
            indent=gate is not None and gate[0] in names))
    return rows


def shared_rows(shared: Dict[str, Tuple[_recipes.Recipe, str]], var_specs, overrides,
                var_values, prompter=None) -> List[Tuple[str, _recipes.Recipe, Any, Choice]]:
    """
    The shared section's rows -- ``(var, owner, param, choice)`` -- under the
    same gates as the steps' own menus.

    A shared setting is one row for several steps, so it is gated by the
    step that lends it its spec (the owner). A dependent shared row sits
    indented under its gate's shared row when that row exists.
    """
    entries = []
    for var, (recipe, param_name) in shared.items():
        spec = _spec_for(recipe, prompter)
        param = spec.get(param_name) if spec else None
        if param is None:
            continue
        if not _gate_open(recipe, spec, param_name, var_specs, overrides, var_values):
            continue
        gate = _recipes.gate_of(recipe, param_name)
        gate_var = _var_behind(recipe, gate[0]) if gate else None
        entries.append((var, recipe, param, gate_var))
    present = {var for var, *_ in entries}
    ordered = []
    for var, recipe, param, gate_var in entries:
        if gate_var in present:
            continue
        ordered.append((var, recipe, param, False))
        ordered.extend((v, r, p, True) for v, r, p, g in entries if g == var)
    return [(var, recipe, param,
             _setting_choice(recipe, param, var_specs, overrides, var_values,
                             key=var, indent=indent))
            for var, recipe, param, indent in ordered]


def _edit(prompter: Prompter, recipe: _recipes.Recipe, param: ParamSpec,
          var_specs: Dict[str, dict], overrides: Dict[str, Dict[str, Any]],
          var_values: Dict[str, Any], is_shared: bool = False,
          tables: Optional[Callable[[], List[Tuple[str, str]]]] = None) -> None:
    """
    Ask for one setting and file the answer wherever it belongs.

    Esc here means "not this one after all". It leaves the setting as it was and
    returns to the list, keeping every other change made so far -- where before
    it unwound out of the options screen entirely and every change was thrown
    away, which is the opposite of going back.
    """
    current, var_name, widget = _live_value(recipe, param, var_specs, overrides, var_values)

    if is_shared:
        # you can edit this from any step that reads it, but it's one
        # pipeline-wide value -- we say so here, right when it's being edited,
        # so a change made under "Transcript" is never a surprise under
        # "Whisper embeddings".
        prompter.reason(
            "A shared setting: every step in this pipeline that reads it "
            "changes together."
        )

    if param.name in recipe.library:
        # this parameter's value is a set of library entries, not a typed
        # answer -- so we route it to the picker, and file the result as a
        # per-step override of explicit paths (which is exactly what the
        # analyzers accept). the var behind the template stays put for CLI
        # users; an override simply outranks it.
        from ..helpers.library import entries, kind_by_id
        from .library import pick_from_library

        kind = kind_by_id(recipe.library[param.name])
        existing = overrides.get(recipe.id, {}).get(param.name)
        if isinstance(existing, list):
            current = existing
        else:
            # no explicit choice yet, so the recipe's default entries
            # (stoplists: punctuation + English) start ticked, never the whole
            # library -- and with no declared defaults, the picker's own
            # "everything" default kicks in, which is right for dictionaries.
            names = recipe.library_defaults.get(param.name, ())
            current = [str(e) for e in entries(kind) if e.name in names]
            if names:
                # matches what the settings row promised: left alone, these
                # apply. we say it here too so the ticks read as the current
                # state, not as some suggestion the row disagreed with.
                prompter.reason(
                    "The default set, already applied — [enter] keeps it, "
                    "[space] adjusts it."
                )
        # a recipe that declares an *empty* default set (concept
        # dictionaries: optional) opens the picker empty; the plain
        # "everything" default is for parameters that mean it
        picked = pick_from_library(
            prompter, kind, current=current,
            start_unticked=(param.name in recipe.library_defaults and not names))
        if picked is not None:
            overrides.setdefault(recipe.id, {})[param.name] = picked
        return

    if param.name == recipe.encoder_param:
        # an encoder is mostly a thing the machine already has: the library's
        # encoders and predictors, the downloaded models. a text box made
        # people type names they couldn't remember, so now the picker lists
        # them and keeps the text box around for anything else.
        from .library import pick_encoder

        value = pick_encoder(prompter, str(current or ""))
        if value is None or value == current:
            return
        if var_name:
            var_values[var_name] = value
        else:
            overrides.setdefault(recipe.id, {})[param.name] = value
        return

    try:
        value = _ask_one_param(prompter, param, current, widget, tables)
    except GoBack:
        return
    if value == current:
        return
    if var_name:
        var_values[var_name] = value
    else:
        overrides.setdefault(recipe.id, {})[param.name] = value

    # validity cascades between the engine and its tokenizer -- preferences
    # don't. potts stays the tokenizer under either engine (comparable counts
    # is the whole point of it), but stanza's tokenizer can't run without the
    # stanza engine, and if we left the pair contradictory here we'd compose a
    # pipeline that only fails at run time.
    def _live_var(name: str):
        if name in var_values:
            return var_values[name]
        return (var_specs.get(name) or {}).get("default")

    if param.name == "engine" and value == "nltk" \
            and _live_var("tokenizer") == "stanza":
        var_values["tokenizer"] = "potts"
        prompter.note("  tokenizer switched back to potts — stanza's "
                      "tokenizer needs the stanza engine.", style="yellow")
    if param.name == "tokenizer" and value == "stanza" \
            and _live_var("engine") != "stanza":
        # choosing stanza's tokenizer *is* choosing stanza, so we follow the
        # intent upward rather than refusing the answer.
        var_values["engine"] = "stanza"
        prompter.note("  engine switched to stanza — its tokenizer comes "
                      "with it.", style="yellow")

    # same shape between translating and the English-only models. asking for
    # a translation *is* asking for a multilingual model, so the model
    # follows; picking an English-only model turns translation off. either
    # way we end up composing a pipeline that can actually run, and we tell
    # the person what moved and why right when they moved it.
    if param.name == "translate" and value is True:
        model = _live_var("whisper_model")
        if isinstance(model, str) and model.endswith(".en"):
            var_values["whisper_model"] = model[:-3]
            prompter.note(f"  whisper model switched to {model[:-3]} — "
                          f"{model} is English-only and cannot translate.",
                          style="yellow")
    if param.name == "whisper_model" and isinstance(value, str) \
            and value.endswith(".en") and _live_var("translate") is True:
        var_values["translate"] = False
        prompter.note(f"  translate switched off — {value} is English-only.",
                      style="yellow")

    if (param.name == "engine" and value == "stanza") or \
            (param.name == "tokenizer" and value == "stanza"):
        # the moment of choosing is the moment to say so. preflight has
        # already run by the time this screen exists, and a run-time failure
        # naming a pip line is an hour later than this note.
        import sys

        if "stanza" not in sys.modules:
            # importing stanza pulls in torch and takes several seconds the
            # first time. without a line here the screen just sits still after
            # the answer, which looks like a hang (and got reported as one).
            prompter.working("Checking that stanza is installed (importing "
                             "it takes a moment)…")
        try:
            import stanza  # noqa: F401
        except ImportError:
            prompter.note(
                "  stanza is not installed on this machine. The pipeline "
                "will save fine, but this step will fail at run time until "
                "you install it:  pip install taters[stanza]",
                style="yellow")


def tune_shared(
    prompter: Prompter,
    shared: Dict[str, Tuple[_recipes.Recipe, str]],
    var_specs: Dict[str, dict],
    overrides: Dict[str, Dict[str, Any]],
    var_values: Dict[str, Any],
    tables: Optional[Callable[[], List[Tuple[str, str]]]] = None,
) -> None:
    """Change the settings that belong to the pipeline rather than to one step."""
    if not shared_rows(shared, var_specs, overrides, var_values, prompter):
        return

    while True:
        rows = shared_rows(shared, var_specs, overrides, var_values, prompter)
        choices = [choice for _var, _recipe, _param, choice in rows]
        choices.insert(0, Choice(_DONE, f"{glyphs.TICK} Done with shared settings",
                                 tone="good"))

        prompter.note("")
        try:
            picked = str(prompter.select(f"{_SHARED_LABEL} — change a setting:",
                                         choices))
        except GoBack:
            return          # up one level, to the list of steps
        if picked == _DONE:
            return
        for var, recipe, param, _choice in rows:
            if var == picked:
                _edit(prompter, recipe, param, var_specs, overrides, var_values,
                      tables=tables)
                break


def tune_one_step(
    prompter: Prompter,
    recipe: _recipes.Recipe,
    var_specs: Dict[str, dict],
    overrides: Dict[str, Dict[str, Any]],
    var_values: Dict[str, Any],
    shared: Sequence[str] = (),
    tables: Optional[Callable[[], List[Tuple[str, str]]]] = None,
) -> None:
    """
    Work through one step's settings, one at a time, until the user says stop.

    A menu rather than a march through every parameter in turn. Being asked
    eleven questions to change one is the reason the old flow gated itself
    behind "do you want the common options or all of them?" -- a question
    nobody can answer before they have seen either list. Showing the list, with
    the current values on it, answers itself.
    """
    # we say so before reading, not after. reading a step's settings means
    # importing the module that implements it, and a module built on a
    # pre-trained model can take seconds to load however carefully its imports
    # are arranged. without a line here the screen just sits there blank, and
    # the only way to read that is "the program hung" -- which is exactly how
    # it got reported to us.
    if recipe.target not in _SPEC_CACHE:
        prompter.working(f"Reading {recipe.label}'s settings…")
    spec = _spec_for(recipe, prompter)
    if spec is None:
        return

    everyday, rest = _editable_names(recipe, spec)
    if not everyday and not rest:
        prompter.note(f"    {recipe.label} has nothing to change.", style="dim")
        return

    # settings bound to shared variables stay on this menu. when they were
    # missing it read as "doesn't exist" (someone hunting for `lemmatize`
    # under the frequency list concluded exactly that), so we keep them here,
    # marked, and edit them as the one pipeline-wide value they are.
    shared = set(shared)

    def _is_shared(name: str) -> bool:
        return _var_behind(recipe, name) in shared

    while True:
        # every setting, with dependents ordered under their gates and gated
        # against the live values. we re-evaluate on each repaint -- see
        # `step_rows`.
        choices = step_rows(recipe, spec, var_specs, overrides, var_values,
                            shared)
        choices.insert(0, Choice(_DONE, f"{glyphs.TICK} Done with {recipe.label}",
                                 tone="good"))
        if recipe.text_input:
            # we say that the input plumbing exists and where it went, rather
            # than hiding it silently -- "not on the menu" reads as "doesn't
            # exist" (the lesson the shared variables taught us earlier).
            handled = sorted(
                n for n in _recipes.TEXT_INPUT_KEYS
                if spec.get(n) is not None and n not in NEVER_TUNABLE)
            if handled:
                choices.append(Choice(
                    "_pipeline_handled_", "…input & gathering settings",
                    disabled=f"{len(handled)} set by the gather step, not here",
                ))

        prompter.note("")
        try:
            picked = str(prompter.select(f"{recipe.label} — change a setting:",
                                         choices))
        except GoBack:
            return          # up one level, to the list of steps
        if picked == _DONE:
            return

        param = spec.get(picked)
        if param is not None:
            _edit(prompter, recipe, param, var_specs, overrides, var_values,
                  is_shared=_is_shared(picked), tables=tables)


def _changed_count(recipe_id: str, names: Sequence[str],
                   overrides: Dict[str, Dict[str, Any]],
                   var_values: Dict[str, Any]) -> int:
    """How many of this entry's settings have been changed so far."""
    return (len(overrides.get(recipe_id, {}))
            + sum(1 for n in names if n in var_values))


def ask_tuning(
    prompter: Prompter,
    steps: Sequence[_recipes.Recipe],
    var_specs: Dict[str, dict],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    var_values: Optional[Dict[str, Any]] = None,
    ask_gate: bool = True,
    source: str = "media",
    level: Optional[str] = None,
    group_by: Sequence[str] = (),
    selected: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Offer to change the pipeline's settings, and collect what was changed.

    Two nested menus, each with a way back up:

        Which part of the pipeline?  ->  Which setting?  ->  answer it
              ^                              |                  |
              +---- "Done with X" -----------+                  |
              ^                                                 |
              +-------------------------------------------------+

    Returning to the list rather than leaving is the point. Changing one step's
    settings is not evidence that you are finished with the whole screen -- the
    previous shape asked once, up front, which steps you wanted (a checkbox),
    walked those in order, and then went straight to saving. Someone who picked
    the shared settings, changed one, and pressed "Done with shared settings"
    found their pipeline being written, with no way back to the step they had
    not thought of yet.

    Settings shared by several steps are gathered into one entry of their own
    rather than repeated under each. See :func:`shared_variables`.

    Parameters
    ----------
    overrides, var_values : dict, optional
        What has been changed already. Passed in and **mutated in place** so
        that leaving this screen and coming back keeps the work: they used to
        be built fresh on every call, so backing out of the question after it
        silently threw away every setting the user had just made.

    Returns
    -------
    (overrides, var_values)
        The same two dicts, for callers that would rather read than mutate.
        ``overrides`` is per-step, keyed by recipe id, and goes into the step's
        ``with:`` block. ``var_values`` goes into the preset's ``vars:`` block.
        See :func:`_current_value` for which changes end up where.

    Raises
    ------
    GoBack
        If the user backs out of the top of this screen. Raised rather than
        returned: returning meant Esc here carried *on* to naming the pipeline,
        so Esc at naming and Esc here bounced between the two screens with no
        way out in either direction.
    """
    overrides = {} if overrides is None else overrides
    var_values = {} if var_values is None else var_values

    if not steps:
        return overrides, var_values

    shared = shared_variables(steps)

    def tables() -> List[Tuple[str, str]]:
        # the feature tables this run will make, named the way the analyses
        # will see them. we read these at the moment of asking, since renaming
        # an output or changing the matrix weighting a row earlier changes the
        # names. (`selected`, not `picked`: the menu loop below rebinds
        # `picked` to the row chosen, and a closure reads the name at call
        # time.)
        return table_names(steps, source, level, group_by, selected,
                           overrides=overrides, var_values=var_values,
                           var_specs=var_specs)

    # only worth asking while the answer is still in doubt. coming back with
    # changes already made, it's obviously yes, and re-asking put the user
    # one screen further from the list they were actually working in -- this
    # is what made backing out of naming land somewhere so odd.
    if ask_gate and not (overrides or var_values):
        prompter.note("")
        prompter.note("  Your pipeline is ready to run as it is. You can also "
                      "change how any of its steps work.", style="dim")
        if not prompter.confirm("Change any settings first?", default=False):
            return overrides, var_values

    while True:
        total = sum(len(v) for v in overrides.values()) + len(var_values)
        choices: List[Choice] = [Choice(
            _DONE,
            f"{glyphs.TICK} Done — save and continue" if not total
            else f"{glyphs.TICK} Done — save {_plural(total, 'change')} and continue",
            tone="good",
        )]
        if shared:
            changed = _changed_count(_SHARED, list(shared), overrides, var_values)
            choices.append(Choice(
                _SHARED, _label_with_count(_SHARED_LABEL, changed),
                help=", ".join(v.replace("_", " ") for v in shared),
            ))
        for recipe in steps:
            # variables in `shared` get counted against the shared entry and
            # nowhere else. a step *reads* `whisper_model`, but that's not
            # where you changed it -- if we credited both, the row counts
            # wouldn't add up to the total on the "Done" line, which looks like
            # a bug.
            names = [var for var in (_var_behind(recipe, n) for n in recipe.with_)
                     if var and var not in shared]
            changed = _changed_count(recipe.id, names, overrides, var_values)
            choices.append(Choice(recipe.id,
                                  _label_with_count(recipe.label, changed),
                                  help=recipe.help))

        prompter.note("")
        try:
            picked = str(prompter.select("What would you like to change?", choices))
        except GoBack:
            # this is the top of this screen, so back means out of it -- and
            # out with the changes intact, since they live in the caller's
            # dicts. we re-raise rather than return so that "back" is actually
            # backwards.
            raise
        if picked == _DONE:
            return overrides, var_values
        if picked == _SHARED:
            tune_shared(prompter, shared, var_specs, overrides, var_values,
                        tables=tables)
            continue
        recipe = next((r for r in steps if r.id == picked), None)
        if recipe is not None:
            tune_one_step(prompter, recipe, var_specs, overrides, var_values,
                          shared, tables=tables)


def _label_with_count(label: str, changed: int) -> str:
    """
    `Readability scores` -> `Readability scores  (1 changed)`.

    So the list doubles as a record of what you have already been through. With
    the menu now returning here after every step, without this there is nothing
    on screen to distinguish the parts you have visited from the parts you have
    not.
    """
    return label if not changed else f"{label}  ({changed} changed)"


# ---------------------------------------------------------------------------
# Step 7-9: review, write, run
# ---------------------------------------------------------------------------

def _measures(recipe: _recipes.Recipe, step: dict) -> str:
    """
    How this step gets from many rows to one, in three words.

    Worth a column of its own because the default is deliberately *mixed*: the
    text analyzers join a speaker's utterances and measure the result once,
    while sentence embeddings measure each utterance and average the vectors.
    Those are different numbers -- about a third apart on vocabulary measures --
    and nothing in the output files says which happened. This is the one place
    it is visible before the run starts.
    """
    if not _recipes.level_aware(recipe):
        return ""
    keys = step["with"].get("group_by")
    if not keys or not isinstance(keys, list):
        return "each row"
    if "text_id" in keys:
        # grouping on the per-row identifier collapses nothing -- the merge is
        # just a tidy copy at that point. if we called it "averaged" we'd be
        # claiming a combining step that never happened.
        return "each row"
    if step["with"].get("aggregate") is True:
        return "scores averaged"
    return "words joined"


def _recipe_behind(step: dict):
    """
    The catalog recipe a composed step came from, or ``None``.

    Matched on ``(save_as, call)`` and nothing else -- the same strict rule
    ``run_saved._recipes_behind`` uses.

    Falling back to ``call`` alone looks like a kindness and defeats the
    purpose: a model's private extraction has by definition the *same* call as
    the user's step, since it measures the same feature with different
    settings. A call-alone match therefore finds the user's recipe, and the
    private step renders as an ordinary "added for you" row while its
    ``for_model`` is never read. Strictness is what makes ``None`` reachable,
    and ``None`` is the answer that gets the row rendered honestly.
    """
    return next((r for r in _recipes.RECIPES
                 if r.save_as == step.get("save_as")
                 and r.call == step.get("call")), None)


def review(prompter: Prompter, preset: dict, selected: Sequence[str],
           inputs: Sequence[Path]) -> None:
    """Show the composed pipeline as a table before anything is written."""
    picked = set(selected)
    rows = []
    for i, step in enumerate(preset["steps"], 1):
        recipe = _recipe_behind(step)
        if recipe is None:
            # a step the catalog doesn't know about. `next(...)` here used to
            # have no default, so this raised StopIteration on the happy path
            # of every run that composed one -- and the composer now splices in
            # private steps that no recipe owns, to extract a saved model's
            # features at the settings the model was fitted with.
            rows.append([
                str(i),
                step.get("label") or step.get("call", "").split(".")[-1],
                "each file" if step["scope"] == "item" else "once",
                step.get("why") or "",
                f"the model {step['for_model']!r} needs it"
                if step.get("for_model") else "added for you",
            ])
            continue
        rows.append([
            str(i),
            recipe.label,
            "each file" if step["scope"] == "item" else "once",
            _measures(recipe, step),
            "you picked" if recipe.id in picked else "added for you",
        ])
    prompter.note("")
    prompter.table(
        f"{preset['meta']['title']} — {_plural(len(preset['steps']), 'step')} "
        f"over {_plural(len(inputs), 'file')}",
        rows,
        ["#", "Step", "Runs", "Measures", "Why"],
    )


def pipeline_folder(preset: dict, cwd: Path) -> Path:
    """
    The folder this pipeline owns: ``<cwd>/<id>/``.

    Everything a run produces goes in here -- the preset itself, ``features/``,
    ``transcripts/``, the manifest. One folder per pipeline beats one shared
    ``features/`` that three different runs quietly overwrite in turn, and it
    means a whole analysis can be zipped up and sent to someone.

    ``run_pipeline._get_preset_dirs()`` recognizes a directory holding a YAML
    of the same name, so the pipeline is still visible to ``--list-presets``.
    """
    return Path(cwd) / str(preset["meta"]["id"])


def _resolve_name_collision(prompter: Prompter, title: str,
                            cwd: Path) -> Optional[str]:
    """
    Settle what to do when `title`'s folder already exists.

    Returns the title to use (possibly suffixed), or None for "let me rename
    it", which the review loop turns into re-asking the name. Any existing
    folder counts, pipeline or not -- but the two get different questions,
    because "Replace it" is only an honest offer for an actual pipeline.
    Writing a preset into some other existing folder would claim it as a
    pipeline folder and mix run outputs into whatever it holds, so for those
    the only ways forward are a suffixed name or a different one.
    """
    slug = slugify(title)
    target = Path(cwd) / slug
    if not target.exists():
        return title

    numbered = title
    for n in range(2, 1000):
        numbered = f"{title} ({n})"
        if not (Path(cwd) / slugify(numbered)).exists():
            break

    is_pipeline = (target / f"{slug}.yaml").exists()
    if is_pipeline:
        question = f"A pipeline called '{title}' already exists."
        options = [
            Choice("both", f"Keep both — save this one as '{numbered}'"),
            Choice("replace", "Replace it",
                   "Its old outputs stay in the folder, and steps will "
                   "reuse them unless 'overwrite existing' is turned on."),
            Choice("rename", "Let me pick a different name"),
        ]
    else:
        question = (f"'{title}' would use the folder '{slug}/', "
                    "which already exists and is not a pipeline.")
        options = [
            Choice("both", f"Save it as '{numbered}' instead"),
            Choice("rename", "Let me pick a different name"),
        ]

    try:
        picked = str(prompter.select(question, options))
    except GoBack:
        return None
    if picked == "rename":
        return None
    if picked == "both":
        return numbered
    return title


def write_preset(preset: dict, directory: Path) -> Path:
    """
    Write the preset to ``<directory>/<id>.yaml``.

    The filename matching the folder name is what makes the folder recognisable
    as a pipeline folder rather than any old directory with YAML in it.
    """
    from ..helpers.atomic import atomic_write

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{preset['meta']['id']}.yaml"
    # atomic, because a Ctrl-C mid-write leaves a truncated YAML -- which the
    # tolerant preset lister still shows, and which then crashes whatever
    # opens it. the helper exists for exactly this.
    with atomic_write(path, encoding="utf-8") as fh:
        fh.write(yaml.safe_dump(preset, sort_keys=False, allow_unicode=True,
                                width=88))
    return path


def default_workers() -> int:
    """
    A sensible number of files to work on at once.

    Three-quarters of the machine's logical cores -- the same "leave a
    quarter for the human" policy the runner's automatic resolve uses
    (`helpers.parallel_map.auto_workers`), so the wizard's recommendation
    and what `workers: 0` actually does are the same number.
    """
    from ..helpers.parallel_map import auto_workers

    return auto_workers()


def ask_workers(prompter: Prompter, *, fans_out: bool) -> int:
    """
    How many files to process at once.

    Only asked when the pipeline fans out over files (`fans_out`: it has
    item-scoped steps). Text pipelines parallelize too -- the analyzers spend
    the same shared ``workers`` variable on reader/scorer processes -- but
    their right default is automatic, so instead of a question they get 0
    ("let the dial decide"): the runner resolves it to one process per core,
    and the setting stays editable under Shared settings. Taken as a bool
    rather than a recipe list so a *saved* preset, which is plain dicts, can
    ask the very same question.
    """
    if not fans_out:
        return 0

    from ..helpers.parallel_map import max_workers

    suggested = default_workers()
    ceiling = max_workers()
    choices = [
        Choice(str(suggested), f"{suggested} at a time",
               f"recommended: three-quarters of this machine's {ceiling} "
               "cores, leaving some for you"),
        Choice("1", "One at a time", "slowest, but easiest to read if something fails"),
    ]
    if ceiling > suggested:
        choices.append(Choice(
            str(ceiling), f"{ceiling} at a time",
            "everything this machine has; other programs will feel it"))
    picked = prompter.select("How many files should Taters work on at once?", choices)
    try:
        # we clamp whatever gets typed in to what the machine can hold: 1 up
        # to its logical core count. past the cores, more processes is just
        # overhead.
        return min(ceiling, max(1, int(str(picked))))
    except ValueError:
        return suggested


def repro_command(preset_path: Path, *, root_dir, file_type: str,
                  workers: int, overrides: Optional[dict] = None) -> str:
    """
    The exact terminal command that reproduces a TUI run.

    Everything the menus decided is spelled out -- the preset file by path, the
    inputs, the worker count, and any variable changed at run time -- so
    re-running from a shell is a paste, not an afternoon of reverse-engineering
    the clicks. Written into the run manifest, which is where someone looks
    when they ask "what exactly ran here?".
    """
    def q(value) -> str:
        text = str(value)
        return f'"{text}"' if (" " in text or not text) else text

    parts = ["python -m taters.pipelines.run_pipeline",
             f"--preset-file {q(preset_path)}"]
    if root_dir is not None:
        parts.append(f"--root_dir {q(root_dir)}")
        parts.append(f"--file_type {file_type}")
    parts.append(f"--workers {workers}")
    for key, value in (overrides or {}).items():
        parts.append(f"--var {q(f'{key}={value}')}")
    return " ".join(parts)


def execute_preset(prompter: Prompter, preset: dict, *, root_dir, file_type,
                   workers: int, work_dir: Path, preset_name: str,
                   vars_ctx: Optional[dict] = None,
                   command: Optional[str] = None) -> Tuple[bool, dict]:
    """
    Run a composed or saved preset with the live display, and finish properly.

    The one path a run takes, whoever starts it. This block used to exist
    twice -- here and in "Run a saved pipeline" -- and the copies had already
    drifted: the wizard asked how many files to work on at once while the
    saved-pipeline path silently hardcoded four, so the same pipeline ran with
    different parallelism depending on which menu launched it.

    Returns
    -------
    (ok, manifest)
    """
    from ..pipelines.run_pipeline import run_preset, summarize_manifest
    from .run_display import RunDisplay, reporter_for

    prompter.note("")
    manifest_path = work_dir / "run_manifest.json"

    kwargs = dict(
        root_dir=root_dir,
        file_type=file_type,
        workers=workers,
        out_manifest=manifest_path,
        work_dir=work_dir,
        preset_name=preset_name,
        command=command,
        verbose=False,
    )
    if vars_ctx is not None:
        kwargs["vars_ctx"] = vars_ctx

    console = getattr(prompter, "_console", None)
    display = RunDisplay(console) if console is not None else None
    if display is not None:
        with display:
            manifest = run_preset(preset, on_event=reporter_for(display),
                                  **kwargs)
        failures = display.failures
    else:
        manifest = run_preset(preset, on_event=_progress_reporter(prompter),
                              **kwargs)
        failures = []

    ok = bool(summarize_manifest(manifest, verbose=False))
    prompter.stage("run", "Run", status="done",
                   detail="ok" if ok else "with problems")
    finish_screen(prompter, ok=ok, manifest=manifest, folder=work_dir,
                  manifest_path=manifest_path, failures=failures)
    return ok, manifest


#: How much of one failure message the finish screen shows. Long enough for
#: a sentence and the start of whatever it lists; the rest is in the
#: manifest, and the screen says where that is.
_ERROR_CHARS = 300


def _shortened(message: object, limit: int = _ERROR_CHARS) -> str:
    """One failure, trimmed to something readable on a terminal."""
    text = " ".join(str(message).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" ,") + f"… (+{len(text) - limit} more characters)"


def finish_screen(prompter: Prompter, *, ok: bool, manifest: dict,
                  folder: Optional[Path], manifest_path: Path,
                  failures: Sequence[str] = ()) -> None:
    """
    Report the outcome and ask what to do next.

    A run that simply stops, leaving the last progress bar on screen, gives no
    sense of having finished and no idea where the results went. This says what
    happened, names the folder, and offers the two things anyone wants at that
    point: do something else, or stop.

    Raises
    ------
    QuitRequested
        If the user chooses to finish.
    """
    items = manifest.get("items") or []
    failed = [i for i in items if i.get("status") == "error"]

    if ok:
        prompter.note(f"\n  {glyphs.TICK} Finished. Everything succeeded.", style="green")
    else:
        detail = f"{len(failed)} file(s) failed" if failed else "a step failed"
        prompter.note(f"\n  {glyphs.CROSS} Finished with problems: {detail}.",
                      style="bold red")

    # per-file failures first, then step-level ones. the step-level list is
    # where a failed GLOBAL step's message lives, and it used to get shown
    # nowhere -- the screen said "a step failed", the folder sat empty, and
    # the one line explaining why was only in the manifest JSON.
    # ...and we shorten each one. a step that refuses because a column is
    # missing can name every column it *does* have, and we had one
    # sentence-embedding table turn that into a thousand names -- printed
    # four times over, since the step-level list and the manifest both carry
    # it. the whole message is in the manifest, which the next line points at.
    seen: set = set()
    for line in list(failures)[:5]:
        seen.add(str(line))
        prompter.note(f"      {_shortened(line)}", style="red")
    if len(failures) > 5:
        prompter.note(f"      …and {len(failures) - 5} more.", style="red")
    for line in list(manifest.get("errors") or [])[:5]:
        # the same failure usually shows up twice -- once from the display,
        # once from the manifest -- and saying it twice looks like two
        # problems.
        if str(line) in seen:
            continue
        prompter.note(f"      {_shortened(line)}", style="red", wrap=True)

    if folder is not None:
        prompter.note(f"\n  Everything from this run is in:\n    {folder}",
                      style="cyan", wrap=False)
        outputs = _outputs_in(folder)
        for line in outputs:
            prompter.note(f"      {line}", style="dim")
    prompter.note(f"\n  Full details: {manifest_path}", style="dim", wrap=False)

    # the verdict rides on the question itself. the red lines above can
    # scroll past or sit above a busy folder listing; the question is the one
    # line the eye is guaranteed to land on (the cursor is there). we had a
    # failed run get read as a success because everything above it was.
    # a run that saved a model -- a ridge, a topic model, word vectors, a
    # fine-tuned predictor, an adapted encoder -- can hand it to the library
    # from here, so the next study can pick it. we make that a row on the
    # question rather than a question of its own, so the screen keeps the
    # same shape for a run that produced nothing worth keeping.
    from ..helpers.model_spec import models_produced

    produced = models_produced(folder) if folder is not None else []
    while True:
        question = "What now?" if ok else \
            "That run had problems (details above). What now?"
        rows = [Choice("menu", "Do something else", "Back to the main menu")]
        if produced:
            rows.append(Choice(
                "keep", f"Add {_plural(len(produced), 'model')} from this "
                        f"run to my library",
                ", ".join(label for _p, label in produced[:4])
                + (" …" if len(produced) > 4 else ""), tone="good"))
        rows.append(Choice("quit", "Finish", "Close Taters"))
        try:
            choice = str(prompter.select(question, rows))
        except (GoBack, Cancelled):
            # there's nothing above this question to go back to -- the run
            # already happened. Esc used to climb to the hub, which printed
            # "Backed out. Nothing was changed." over a folder full of fresh
            # outputs and threw the run's verdict away, so a failed run
            # exited 0. oops.
            choice = "menu"
        if choice == "keep":
            from .library import offer_library_import

            offer_library_import(prompter, produced)
            produced = []
            continue
        break
    if choice == "quit":
        raise QuitRequested(ok=ok)


def _outputs_in(folder: Path, limit: int = 8) -> List[str]:
    """The result files a run left behind, for the finished screen."""
    import itertools

    try:
        # bounded, because this draws the finish screen, and rglob over a run
        # that produced tens of thousands of files (or sits on /mnt/c) stalled
        # the one screen whose whole job is to show up promptly when the work
        # ends. the cap is generous -- we only show `limit` lines anyway.
        found = sorted(itertools.islice(
            (p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in {".csv", ".srt", ".txt", ".json"}
             and p.name != "run_manifest.json"
             # a model's private feature tables aren't results. they were
             # measured at somebody else's settings so that a saved model
             # could be scored, the user never asked for them, and listing
             # them first would shove the results they did ask for right off
             # the screen.
             and MODEL_WORK_DIR not in p.parts),
            2000,
        ))
    except OSError:
        return []
    lines = [str(p.relative_to(folder)) for p in found[:limit]]
    if len(found) > limit:
        lines.append(f"…and {len(found) - limit} more")
    return lines


def _progress_reporter(prompter: Prompter):
    """A ``run_preset`` event callback that narrates the run through the prompter."""
    state = {"done": 0, "total": 0}

    def on_event(name: str, **payload) -> None:
        if name == "step_start":
            state["done"] = 0
            state["total"] = payload.get("items", 1)
            forwhom = (f" (for {payload['for_model']})"
                       if payload.get("for_model") else "")
            prompter.note(
                f"  [{payload['index']}/{payload['total']}] "
                f"{payload['call']}{forwhom}",
                style="cyan",
            )
        elif name == "item_done":
            state["done"] += 1
            if payload.get("status") != "ok":
                prompter.note(
                    f"      failed: {Path(payload['input']).name} — {payload.get('error')}",
                    style="red",
                )
            elif state["done"] == state["total"]:
                prompter.note(f"      {state['done']}/{state['total']} done", style="dim")
        elif name == "step_done" and payload.get("status") == "error":
            prompter.note(f"      step failed: {payload.get('error')}", style="red")

    return on_event


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------

BANNER = """
  ╔══════════════════════════════════════════════════════════════╗
  ║   Taters — Takes All Things, Extracts Relevant Stuff          ║
  ╚══════════════════════════════════════════════════════════════╝

  This will ask a few questions, build a pipeline from your answers,
  save it so you can run it again later, and then offer to run it.

  Ctrl-C backs out at any point. Nothing is written until you say so.
"""


def _derive(prompter: Prompter, src: SourceSpec, selected: List[str],
            providers: Dict[str, str]) -> Tuple[List[str],
                                                List[_recipes.Recipe],
                                                Dict[str, dict], dict]:
    """
    Everything that follows from the chosen features, with nothing asked.

    Resolving the selection, warning about anything not installed, and
    composing a draft so the options screen can show real current values are
    one unit of work: they all follow from the selection and none of them
    should happen again while the user is only moving around downstream of it.
    Pulled out of `run_wizard` because it is the one part of that function that
    is not a stage -- it sits between them.

    Returns
    -------
    (selected, steps, var_specs, source_kwargs)
        `selected` comes back possibly shorter, if the user dropped a step
        whose extra is missing.
    """
    # source-agnostic; these just keep the compose call sites short.
    source_kwargs = dict(
        source=src.source,
        input_path=str(src.path),
        text_cols=src.text_cols,
        id_cols=src.id_cols,
        text_mode=src.text_mode,
        group_by=src.group_by,
        delimiter=src.delimiter,
        level=src.level or None,
    )

    def resolve() -> List[_recipes.Recipe]:
        try:
            return resolve_selection(selected, providers=providers,
                                     source=src.source)
        except ComposeError as e:
            # Cancelled, not a bare re-raise. nothing above here catches
            # ComposeError, so letting it climb ended the whole session with a
            # traceback. backing out to the menu is what every other dead end
            # in the wizard does, so we do the same.
            prompter.note(f"  {e}", style="red")
            raise Cancelled() from e

    from .compose import _is_pointless_merge

    def live_steps() -> List[_recipes.Recipe]:
        # the composer drops raw-level merges (they combine nothing and lose
        # columns), so we don't want the options screen offering settings for
        # a step that's never going to ship.
        return [r for r in resolve()
                if not _is_pointless_merge(r, src.source, src.level or None,
                                           src.group_by)]

    steps = live_steps()
    dropped = preflight(prompter, steps)
    if dropped:
        selected = [s for s in selected if s not in dropped]
        # a dropped step might be in here as the chosen *provider* (picked at
        # "How should Taters produce a transcript?") rather than as a tick of
        # its own. filtering `selected` alone left it in `providers`, and the
        # re-resolve quietly put the very step the user just declined straight
        # back into the pipeline.
        orphaned = [cap for cap, prov in providers.items() if prov in dropped]
        for cap in orphaned:
            del providers[cap]
        if not selected:
            prompter.note("  Nothing left to run.", style="yellow")
            raise Cancelled()
        steps = live_steps()
        # say who stepped in. the re-resolve fills an orphaned capability with
        # the first remaining provider, which is the right default -- but a
        # silent swap in a pipeline the user is about to trust is not.
        for cap in orphaned:
            stand_in = next((r for r in steps if cap in r.produces), None)
            if stand_in is not None:
                providers[cap] = stand_in.id
                prompter.note(
                    f"  Using '{stand_in.label}' for "
                    f"{_recipes.CAPABILITIES.get(cap, cap)} instead.",
                    style="dim")

    # provisional compose, purely so that tuning can show real current values.
    provisional = compose(selected, providers=providers, name="draft",
                          file_type=src.file_type, **source_kwargs)
    return selected, steps, provisional["meta"]["variables"], source_kwargs


def _ask_required_models(prompter: Prompter, steps: Sequence[_recipes.Recipe],
                         overrides: Dict[str, Dict[str, Any]]) -> None:
    """
    Ask which saved models a scoring step should use, when there is a choice.

    Dictionaries default to "everything in your library" and that is what
    people mean; for models it is not -- nobody wants every model they ever
    saved run over a dataset because they did not say otherwise. So with two
    or more models in the library the picker is asked here, at the features
    stage (tick as many as you like), and the answer is an ordinary
    override; with one model, or a choice already made, nothing is asked.
    Esc keeps the library-folder default, which means every model.
    """
    from ..helpers.library import entries, kind_by_id
    from .library import pick_from_library

    for recipe in steps:
        for param_name, kind_id in recipe.library.items():
            if kind_id != "models":
                continue
            if param_name in overrides.get(recipe.id, {}):
                continue
            kind = kind_by_id(kind_id)
            if len(entries(kind)) < 2:
                continue
            prompter.reason(
                f"'{recipe.label}' scores with the models you tick, and your "
                f"library holds {len(entries(kind))}. Tick one or several: each "
                f"gets its own file, and a merged table carries every model's "
                f"columns with the model's name in front.")
            picked = pick_from_library(prompter, kind)
            if picked is not None:
                overrides.setdefault(recipe.id, {})[param_name] = picked


def _ask_encoders(prompter: Prompter, steps: Sequence[_recipes.Recipe],
                  var_values: Dict[str, Any],
                  overrides: Dict[str, Dict[str, Any]]) -> None:
    """
    Ask which language model a transformer step should run on, at the
    features stage.

    A step that runs on an encoder ("Transformer embeddings (any encoder)")
    used to take its default silently and offer a change only on the
    options screen, where nobody looked for it -- so people got embeddings
    from a model they never chose. Now the picker is asked right after the
    checklist, once per such step, and the answer is the ordinary variable.
    Esc keeps the default.
    """
    from .library import pick_encoder

    for recipe in steps:
        if recipe.encoder_param is None:
            continue
        var = _var_behind(recipe, recipe.encoder_param)
        if var is not None and var in var_values:
            continue
        if var is None and recipe.encoder_param in overrides.get(recipe.id, {}):
            continue
        if var is not None:
            current = str(recipe.vars.get(var, {}).get("default", ""))
        else:
            current = str(recipe.with_.get(recipe.encoder_param, "") or "")
        prompter.reason(
            f"'{recipe.label}' runs on a language model. The default suits "
            f"English text; your own adapted encoders, and models already on "
            f"this machine, are listed first.")
        picked = pick_encoder(prompter, current)
        if picked is None:
            continue
        if var is not None:
            var_values[var] = picked
        else:
            overrides.setdefault(recipe.id, {})[recipe.encoder_param] = picked


def _library_defaults(steps: Sequence[_recipes.Recipe],
                      overrides: Dict[str, Dict[str, Any]]) -> None:
    """
    Point every untouched library parameter at the user's whole library.

    The analyzers expand a folder recursively, so handing them the kind's
    library folder means "use everything imported", which is what makes an
    import "always available" without opening the options screen. Injected
    here, as an ordinary override, rather than changed in the recipe defaults
    -- the CLI and the shipped presets keep their old cwd-relative paths, and
    the round-trip tests keep passing untouched.

    An explicit selection from the picker is also an override, so it simply
    wins by already being there.
    """
    from ..helpers.library import entries, kind_by_id, kind_dir

    for recipe in steps:
        for param_name, kind_id in recipe.library.items():
            if param_name in overrides.get(recipe.id, {}):
                continue
            if param_name in recipe.library_defaults:
                # a curated default, not the whole folder. left alone, the
                # frequency list applies punctuation + English -- what nearly
                # everyone wants -- whereas injecting the whole library would
                # apply all 22 languages' stopwords at once. entries the user
                # has deleted simply don't get injected; with none left, the
                # step runs without.
                names = set(recipe.library_defaults[param_name])
                paths = [str(e.resolve())
                         for e in entries(kind_by_id(kind_id))
                         if e.name in names]
                if paths:
                    overrides.setdefault(recipe.id, {})[param_name] = paths
                continue
            overrides.setdefault(recipe.id, {})[param_name] = [
                str(kind_dir(kind_by_id(kind_id)))
            ]


def _model_plans(steps: Sequence[_recipes.Recipe],
                 overrides: Dict[str, Dict[str, Any]]) -> list:
    """
    What each chosen model says about the features it was fitted on.

    Built here, not in ``compose``, because the composer is pure -- it reads
    no files and imports no analysis code, and is tested on that basis. This
    reads the model files and hands over plain data, which is also what lets a
    test drive the composer with a synthetic plan.

    A model whose features cannot be reproduced still yields a plan, carrying
    its problems. The composer refuses on those, by name, at a point where the
    user can still change their answer.
    """
    from ..helpers.library import expand, kind_by_id
    from ..helpers.model_spec import describe, feature_plan

    plans = []
    for recipe in steps:
        for param_name, kind_id in recipe.library.items():
            if kind_id != "models":
                continue
            chosen = overrides.get(recipe.id, {}).get(param_name)
            if not chosen:
                continue
            for path in expand(kind_by_id(kind_id), chosen):
                try:
                    plans.append(feature_plan(describe(path)))
                except Exception:
                    # an unreadable model is the import gate's problem, and
                    # the scoring step refuses it by name later. no reason to
                    # take the whole session down over it here.
                    continue
    return plans


def _check_controls(plans, src) -> None:
    """
    Say now if the spreadsheet lacks a column a model was fitted to control.

    The composer adds the metadata gather that carries a model's controls,
    and the run then refuses at scoring time if the spreadsheet has no such
    column -- after every feature has been measured. The header is known
    here, so the refusal can come while the user can still pick another
    model or another spreadsheet.
    """
    needed = sorted({str(c) for p in plans for c in getattr(p, "controls", ())})
    if not needed or src.source != "csv":
        return
    try:
        columns, _rows = peek_csv(src.path, 1, delimiter=src.delimiter)
    except Exception:
        return          # can't read it here; the run will say so in its own words
    missing = [c for c in needed if c not in columns]
    if missing:
        models = ", ".join(sorted(p.model for p in plans
                                  if getattr(p, "controls", ())))
        raise ComposeError(
            f"{models} was fitted with control column(s) "
            f"{', '.join(needed)}, and {Path(src.path).name} has no "
            f"{', '.join(missing)} column. Score a spreadsheet that carries "
            f"them, or use a model fitted without controls.")


def run_wizard(prompter: Prompter, *, cwd: Optional[Path] = None,
               banner: bool = True,
               analyses: Optional[bool] = None,
               preselected: Optional[Sequence[str]] = None,
               text_only: bool = False,
               before_options: Optional[Callable[..., None]] = None,
               var_defaults: Optional[Dict[str, Any]] = None
               ) -> WizardResult:
    """
    Ask the questions, compose a pipeline, write it, and optionally run it.

    Parameters
    ----------
    prompter : Prompter
        Where the questions go. Production passes
        :class:`~taters.ui.prompts.QuestionaryPrompter`; tests pass
        :class:`~taters.ui.prompts.ScriptedPrompter`.
    analyses : bool, optional
        What to do about the statistics stage.

        ``False`` leaves it out entirely -- no question, no stage on the
        rail. ``True`` means the user came here *to* run analyses, so the
        stage is not optional: at least one has to be picked, and a source
        that cannot support any is caught at the first question rather than
        after they have chosen features for it. ``None`` (the default) asks
        and lets them decline, which is what a programmatic caller with no
        opinion should get.

        The front menu offers the first two as separate entries. They are
        different intentions -- "turn my recordings into numbers" and "find
        out whether these groups differ" -- and one flow that tried to be
        both asked everyone the analysis question, most of whom had nothing
        to answer it with.
    cwd : Path, optional
        The working folder: pipelines are saved as subfolders of it.
        Defaults to the current directory.
    banner : bool, default True
        Whether to print the Taters banner first. False when the hub has
        already printed it, so it does not appear twice.
    preselected : sequence of str, optional
        Recipe ids chosen before the wizard opens, in place of the feature
        checklist. The "Train a model" task uses this: what to train was
        the task's own first question, so the checklist would be one
        already-ticked row. Everything else -- the source, the level, the
        options screen, naming, the run -- is the same flow, and the
        pipeline it saves is re-runnable like any other.
    text_only : bool, default False
        Offer only text sources (documents, a spreadsheet); see
        :func:`ask_source`. With ``preselected`` steps the sources are
        narrowed further to what those steps can read.
    before_options : callable, optional
        ``(prompter, src, var_values, overrides) -> None``, asked once
        after the level question and before the options screen, in the slot
        the analysis stage would take. The training task uses it to ask
        which columns a predictor should predict -- a question that needs
        the spreadsheet's columns and belongs to no step's options. Raising
        :class:`GoBack` from it returns to the source question.
    var_defaults : dict, optional
        Pipeline variables the run should start with already answered. A
        default rather than a decision: every one of them still appears on
        the options screen, so the user can change it. The
        analyze-a-spreadsheet task turns the word clouds off this way.

    Returns
    -------
    WizardResult

    Raises
    ------
    Cancelled
        If the user backs out. :func:`main` turns this into a clean exit.
    """
    cwd = Path(cwd or Path.cwd())
    if banner:
        prompter.note(BANNER)
    _init_stages(prompter, analyses)

    # which stage we enter next. Esc sets it back by one and we loop, and
    # that's what makes "back" mean the same thing on every screen -- the
    # stages are the units a user thinks in ("I picked the wrong folder", "I
    # meant to tick something else"), and re-asking a whole stage is both
    # simpler and less surprising than trying to rewind to some arbitrary
    # question inside one.
    #
    # we write this as a loop over a named stage rather than as nested loops
    # because the nesting could only ever express "back" as "carry on to the
    # next question", which is how Esc at the options screen ended up jumping
    # *forward* to naming the pipeline and bouncing against Esc there.
    at = "source"
    rebuild = False
    #: Set when Esc leaves the analysis stage: the screen before its first
    #: question is the level question, not the feature checklist, so the
    #: features stage re-enters there -- when there was a level question at
    #: all (a run of nothing but acoustics has none, and then the checklist
    #: really is the previous screen).
    resume_at_level = False
    selected: List[str] = []
    providers: Dict[str, str] = {}
    feature_picks: List[str] = []
    overrides: Dict[str, Dict[str, Any]] = {}
    # a task can start the run with a variable already answered -- the
    # analyze-a-spreadsheet flow turns the word clouds off, because a cloud of
    # column names is not what anyone came for. it is a *default*, not a
    # decision: the options screen still shows the row, so it can be turned
    # back on by anyone who joins a document-term matrix later
    var_values: Dict[str, Any] = dict(var_defaults or {})

    while True:
        if at == "source":
            prompter.stage("source", "Source", status="active")
            # no `except GoBack` here, and that's intentional. Esc means
            # "back", and at the first question of the wizard back is out of
            # the wizard -- the menu that launched it is the previous screen.
            # catching it and re-asking made the key look like it got ignored,
            # which is the same bug the main menu had.
            allowed: Optional[List[str]] = None
            if preselected:
                allowed = sorted(set.intersection(
                    *[set(_recipes.by_id(r).sources) for r in preselected]))
            # a preselected step whose columns *are* the measures asks for
            # predictors where the others ask for text. the catalog says so,
            # rather than the task, so the question follows the step
            measures = bool(preselected) and all(
                _recipes.by_id(r).takes_level and not _recipes.by_id(r).text_input
                for r in preselected)
            src = ask_source(prompter, analyses=analyses, text_only=text_only,
                             sources=allowed, columns_are_measures=measures)
            prompter.stage("source", "Source", status="done",
                           detail=f"{src.path}  ({len(src.inputs)} file(s))")
            at = "features"

        if at == "features":
            prompter.stage("features", "Features", status="active")
            # the checklist and the level are one stage, so Esc steps back
            # within it -- from the level question to the checklist, and only
            # from the checklist out to the source.
            back_to_source = False
            skip_checklist = False
            if resume_at_level:
                skip_checklist = any(
                    _recipes.level_aware(r) for r in resolve_selection(
                        feature_picks, providers=providers, source=src.source))
                selected = list(feature_picks)
                resume_at_level = False
            while True:
                if preselected:
                    # the choice was made before the wizard even opened, so
                    # there's no checklist to go back to. Esc at the level
                    # question is Esc out to the source.
                    selected = list(preselected)
                    providers = {}
                elif not skip_checklist:
                    try:
                        selected = ask_features(prompter, src.source,
                                                analyses=bool(analyses))
                        providers = resolve_providers(prompter, selected,
                                                      src.source)
                    except GoBack:
                        back_to_source = True
                        break
                skip_checklist = False
                try:
                    src.level, src.group_by = ask_level(
                        prompter, src,
                        resolve_selection(selected, providers=providers,
                                          source=src.source))
                except GoBack:
                    if preselected:
                        back_to_source = True
                        break
                    continue        # back to the checklist
                except ComposeError as e:
                    # an unsatisfiable selection is a wrong answer to *this*
                    # checklist, so we re-ask -- and this must never climb,
                    # since nothing above the wizard catches it and the
                    # session would die with a traceback.
                    prompter.note(f"  {e}", style="red")
                    continue
                break
            if back_to_source:
                at = "source"
                continue
            _ask_required_models(
                prompter, resolve_selection(selected, providers=providers,
                                            source=src.source), overrides)
            if not preselected:
                # the Train task asked for its encoder before the wizard even
                # opened (and seeds it just before the options screen), so
                # only the ordinary checklist route asks here
                _ask_encoders(
                    prompter, resolve_selection(selected, providers=providers,
                                                source=src.source),
                    var_values, overrides)
            prompter.stage("features", "Features", status="done",
                           detail=", ".join(_recipes.by_id(s).label
                                            for s in selected))
            at = "analysis"
            rebuild = True
            analysis = AnalysisSpec()
            feature_picks = list(selected)

        if at == "analysis" and analyses is False:
            # feature extraction was the whole ask. not asking beats asking a
            # question we already know the answer to.
            if before_options is not None:
                try:
                    before_options(prompter, src, var_values, overrides)
                except GoBack:
                    at = "source"
                    continue
            at = "options"

        if at == "analysis":
            prompter.stage("analysis", "Analysis", status="active")
            # we re-derive this from the feature picks each time through, so
            # that backing out of the stage and answering it differently can't
            # leave the previous answer's statistics steps in the selection.
            selected = list(feature_picks)
            var_values = {k: v for k, v in var_values.items()
                          if not k.startswith("stats_")}
            overrides.pop("stats_assemble", None)
            try:
                analysis = ask_analysis(
                    prompter, src,
                    resolve_selection(feature_picks, providers=providers,
                                      source=src.source),
                    required=bool(analyses), picked=feature_picks)
            except GoBack:
                at = "features"
                resume_at_level = True
                continue
            if analyses and not analysis.wanted:
                # they chose the analyses flow, so an empty answer is a
                # question that hasn't been answered, not a decision. each way
                # of ending up with no statistics gets refused at the screen
                # that can actually change it -- a spreadsheet that's all text
                # at the source question, a pick with nothing joinable at the
                # feature checklist, an analysis this data can't run as a
                # grayed-out row -- because sending someone to a screen that
                # can't fix their problem is how two screens bounce off each
                # other forever. so this is just a backstop. `why_not` was
                # already printed by the stage; repeating it here looked like
                # two separate problems.
                prompter.note(
                    "  Pick at least one feature that makes a per-text table, "
                    "or start again with plain feature extraction.",
                    style="yellow")
                at = "features"
                rebuild = True
                continue
            apply_analysis(analysis, src, selected, var_values, overrides)
            prompter.stage(
                "analysis", "Analysis", status="done",
                detail=", ".join(_recipes.by_id(a).label
                                 for a in analysis.analyses)
                or ("not possible here" if analysis.why_not else "none"))
            at = "options"
            rebuild = True

        # derived from the selection, so we redo it when the selection
        # changes and not otherwise. rebuilding on every pass would re-run
        # `preflight`, which asks whether to keep a step whose extra is
        # missing -- and getting asked that again on the way back from naming
        # the pipeline would be its own small mystery.
        if rebuild:
            try:
                selected, steps, var_specs, source_kwargs = _derive(
                    prompter, src, selected, providers)
            except ComposeError as e:
                # somebody ticked word vectors as their only feature, asked
                # for statistics, then said "leave it out" at the preflight.
                # that left the analysis table with nothing to assemble, and
                # the compose error climbed all the way out as a traceback.
                # the answer to "nothing left to extract" is the checklist
                prompter.note(f"  {e}", style="yellow")
                if preselected:
                    raise Cancelled() from e
                at = "features"
                continue
            rebuild = False

        if at == "options":
            prompter.stage("options", "Options", status="active")
            # `overrides` and `var_values` go in and come back mutated, so
            # stepping back out of this screen and returning to it keeps every
            # setting already changed. we used to rebind them from the return
            # value, which threw the lot away on the way past.
            try:
                ask_tuning(prompter, steps, var_specs, overrides, var_values,
                           source=src.source, level=src.level or None,
                           group_by=src.group_by, selected=selected)
            except GoBack:
                # a stage that asked nothing must not catch "back", otherwise
                # it hands control straight forward again and Esc looks
                # broken (it did, for every media run).
                at = "analysis" if analysis.offered else "features"
                continue
            changed = sum(len(v) for v in overrides.values()) + len(var_values)
            prompter.stage("options", "Options", status="done",
                           detail="defaults" if not changed
                           else f"{changed} changed")
            at = "review"

        # naming, the review table and the closing question are one stage, so
        # Esc moves within it before it leaves -- from the closing question
        # back to naming, and from naming back to the options screen.
        prompter.stage("review", "Review", status="active")
        try:
            title = str(prompter.text("Name this pipeline:",
                                      default="My pipeline")).strip()
        except GoBack:
            at = "options"
            continue
        title = title or "My pipeline"

        # we catch this at naming time, while renaming is still one keypress
        # away. saving used to overwrite a same-named pipeline without a word
        # -- and the replacement then *resumed against the old run's
        # outputs*, because steps skip work whose files already exist. a
        # results folder silently mixing two different pipeline definitions
        # is about the worst thing this screen can produce.
        collided = _resolve_name_collision(prompter, title, cwd)
        if collided is None:
            continue            # they chose to rename, so back to the question
        title = collided

        # on a copy, since these are compose-time defaults, not the user's
        # edits. when we injected them into the live dict they came back as a
        # phantom "(1 changed)" on the options screen after backing out of
        # naming.
        effective = {rid: dict(vals) for rid, vals in overrides.items()}
        _library_defaults(steps, effective)
        try:
            plans = _model_plans(steps, effective)
            for plan in plans:
                for problem in plan.problems:
                    raise ComposeError(problem)
            _check_controls(plans, src)
            preset = compose(
                selected,
                model_plans=plans,
                providers=providers,
                overrides=effective,
                var_values=var_values,
                name=title,
                file_type=src.file_type,
                root_dir=str(src.root_dir) if src.root_dir else None,
                **source_kwargs,
            )
        except ComposeError as e:
            # the composer refuses things a person can fix -- most of all a
            # saved model whose features can't be reproduced from this run's
            # settings. unguarded, that refusal left the session as a
            # traceback after the user had answered every single question. so
            # we say it at the options screen instead, where the answer that
            # caused it can be changed.
            prompter.note(f"  {e}", style="yellow")
            prompter.stage("options", "Options", status="active")
            at = "options"
            continue

        review(prompter, preset, selected, src.inputs)
        result = WizardResult(preset=preset, root_dir=src.root_dir,
                              file_type=src.file_type,
                              inputs=list(src.inputs), source=src.source)

        # one question, three real outcomes, instead of "Save?" then "Run?".
        #
        # naming a pipeline is already a decision to keep it, so asking again
        # read as a formality -- and the "no" branch wasn't "don't save" at
        # all, it threw the whole session away. two yes/no questions in a row
        # also made the common case (save it and run it) take two answers when
        # it's really one act.
        prompter.note("")
        try:
            choice = str(prompter.select("What next?", [
                Choice("run", "Save and run it now"),
                Choice("save", "Save it — I'll run it later"),
                Choice("discard", "Discard it and start over"),
            ]))
        except GoBack:
            at = "review"       # back to naming it
            continue
        break

    if choice == "discard":
        raise Cancelled()

    folder = pipeline_folder(preset, cwd)
    path = write_preset(preset, folder)
    result.preset_path = path
    result.folder = folder
    prompter.stage("review", "Review", status="done", detail=path.name)
    prompter.stage("run", "Run", status="active")
    prompter.note(f"\n  Saved to {path}", style="green")
    prompter.note(
        "  Run it again any time with:\n"
        f"    {preset['meta']['cli_example']}",
        style="dim",
        wrap=False,     # must stay one line to survive copy-paste
    )

    if choice != "run":
        prompter.note("  Saved. Run it whenever you are ready.")
        return result

    # we ask this here rather than earlier, since someone who declines to run
    # shouldn't have to answer a question about how the run would have gone.
    try:
        workers = ask_workers(
            prompter, fans_out=any(step.scope == "item" for step in steps))
    except (GoBack, Cancelled):
        # the preset is already on disk. letting this climb reached the hub's
        # "Backed out. Nothing was changed." -- a lie, over a saved file.
        prompter.note("  Saved, not run. Run it any time from the main menu.",
                      style="dim")
        return result

    result.ran = True
    result.ok, result.manifest = execute_preset(
        prompter, preset,
        root_dir=src.root_dir,
        file_type=src.file_type,
        workers=workers,
        work_dir=folder,
        preset_name=preset["meta"]["id"],
        command=repro_command(path, root_dir=src.root_dir,
                              file_type=src.file_type, workers=workers),
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point for the ``taters`` command."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="taters",
        description="Interactive setup wizard: build and run a Taters pipeline.",
    )
    ap.add_argument("--dir", default=None,
                    help="Working folder: pipelines are saved as subfolders "
                         "here (default: current folder)")
    ap.add_argument("--plain", action="store_true",
                    help="Ask one question at a time, without the progress rail. "
                         "Use this if your terminal renders the rail badly.")
    args = ap.parse_args(argv)

    try:
        if args.plain:
            from .prompts import QuestionaryPrompter
            prompter: Prompter = QuestionaryPrompter()
        else:
            from .live import LivePrompter
            prompter = LivePrompter()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2

    from .hub import run_hub

    try:
        ok = run_hub(prompter, cwd=Path(args.dir) if args.dir else None)
    except Cancelled:
        print("\nCancelled. Nothing was changed.")
        return 130
    except KeyboardInterrupt:
        print("\nCancelled. Nothing was changed.")
        return 130
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
