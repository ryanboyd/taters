"""
The library's two screens: manage it, and pick from it.

Storage lives in :mod:`taters.helpers.library`; this module is only the
conversation. Both flows are kind-agnostic -- they are handed a
:class:`~taters.helpers.library.LibraryKind` and never ask what it is -- which
is what lets a future kind (pretrained classifiers) reuse them unchanged.

``manage_library`` is shared deliberately: it is the Settings screen *and* the
"Manage dictionaries…" row inside the wizard's picker, so importing a file
mid-pipeline is the same experience as importing one from Settings, and there
is exactly one flow to learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from ..helpers import library as lib
from .browse import browse_and_tick, browse_for_folder, human_size
from .prompts import Cancelled, Choice, GoBack, Prompter

__all__ = ["import_files", "manage_library", "pick_from_library", "pick_encoder",
           "offer_library_import"]

#: Kinds where a step uses exactly one entry, so the picker is single-select:
#: "use all of them" is a question a model step cannot mean.
#: Kinds a step takes exactly one of. Models used to be here too; a scoring
#: step now takes as many as you tick, like dictionaries.
SINGLE_PICK_KINDS = frozenset({"encoders"})
#: Kinds whose picker opens with nothing ticked: "score with every model I
#: ever saved" is not what anyone means by default, where "use all of my
#: dictionaries" usually is.
START_UNTICKED_KINDS = frozenset({"models"})

# action rows. the colons keep them from colliding with entry names, which are
# filename stems (same convention as the wizard's :done / :shared).
_IMPORT = ":import"
_IMPORT_HF = ":import_hf"
_HF_FOLDER = ":folder"
_HF_NAME = ":name"
_EXPORT = ":export"
_MANAGE = ":manage"
_RENAME = ":rename"
_DELETE = ":delete"
_APPLY = ":apply"
_ALL = ":all"
_NONE = ":none"
_BACK = ":back"
#: The approve row a renderer without in-place ticks needs (see
#: Prompter.ticks_in_place): enter on an entry ticks it there, so approving
#: the ticked set has to be a row of its own.
_USE_TICKED = ":use-ticked"


def _size_of(path: Path) -> str:
    try:
        return human_size(path.stat().st_size)
    except OSError:
        # deleted between listing and drawing, or sitting on a drive that went
        # away. a row with no size beats a session with no screen.
        return ""


def manage_library(prompter: Prompter, kind: lib.LibraryKind) -> None:
    """
    List, import, export, rename and delete the entries of one kind.

    The same dialect as the import browser: entries are tick rows (enter
    toggles), and the actions apply to whatever is ticked -- so deleting six
    stale dictionaries is six ticks and one confirm, not six separate
    delete-and-confirm journeys. Rename, which can only ever mean one file,
    appears when exactly one is ticked.

    Esc leaves; every action returns to the list.
    """
    ticked: set = set()
    pointer: Optional[str] = None       # where the cursor was, kept across loops
    while True:
        entries = lib.entries(kind)
        names = {e.name for e in entries}
        ticked &= names             # renamed or deleted entries lose their tick
        if not entries:
            prompter.reason(
                f"No {kind.label.lower()} imported yet. {kind.help}"
            )

        # static rows, on purpose: space now ticks in place without rebuilding
        # the menu, so rows that appear/disappear (or carry counts) would be
        # lying the moment a tick happened. the counts live in the
        # confirmations instead, which always see the current selection.
        choices = [Choice(_BACK, "↩ Back", tone="nav")]
        # import is the forward act on this screen -- bringing dictionaries in
        # is the whole reason the library exists -- so it gets the green.
        # export is just an ordinary action, and painting it green because it
        # moves files turned the color language into noise.
        choices.append(Choice(_IMPORT, "＋ Import files…", kind.help,
                              tone="good"))
        if kind.id == "models":
            # the other way a model gets here: somebody else trained it and put
            # it on the hub, and it's probably already in this machine's cache
            choices.append(Choice(
                _IMPORT_HF, "＋ Import a classifier from Hugging Face…",
                "A sentiment, emotion or other text classifier (or a regression "
                "head) from the Hugging Face hub: one already in this computer's "
                "cache, a checkpoint folder, or a name to download.",
                tone="good"))
        if entries:
            choices.append(Choice(_EXPORT, "⇩ Export the ticked files…",
                                  "Copy them out to a folder of your choosing."))
            choices.append(Choice(_RENAME, "✎ Rename the ticked file",
                                  "Tick exactly one first."))
            choices.append(Choice(_DELETE, "✕ Delete the ticked files",
                                  "Removes them from the library.",
                                  tone="danger"))
            if kind.id == "models":
                choices.append(Choice(
                    _APPLY, "⚙ Change how the ticked model is applied",
                    "Its name, its output columns, what each predicted class "
                    "is called, and the settings it scores with. Tick exactly "
                    "one first."))
        # keyed by full filename, not stem: `foo.csv` and `foo.dic` can both
        # exist, and stem-keyed rows made every action resolve to whichever
        # sorted first -- so we deleted a file the user didn't pick. oops.
        for e in entries:
            mark = "x" if e.name in ticked else " "
            label, note = _row_of(kind, e)
            choices.append(Choice(e.name, f"[{mark}] {label}",
                                  annotation=note))

        try:
            picked = str(prompter.select(
                f"{kind.label}:", choices,
                # keep the pointer where it was: ticking five entries used to
                # mean five trips back down from the top of the list.
                default=pointer if any(c.value == pointer for c in choices)
                else None,
                toggle_values=names,
                ticked=ticked,
            ))
        except GoBack:
            return
        pointer = picked
        if picked == _BACK:
            return
        if picked == _IMPORT:
            import_files(prompter, kind)
        elif picked == _IMPORT_HF:
            import_hf_classifier_ui(prompter, kind)
        elif picked in (_EXPORT, _RENAME, _DELETE, _APPLY) and not ticked:
            # the backslash keeps rich from reading [space] as a markup tag.
            prompter.note("  Nothing is ticked yet — press \\[space] on a "
                          "file to tick it.", style="yellow")
        elif picked == _EXPORT:
            _export_files(prompter, kind, sorted(ticked))
        elif picked == _RENAME:
            if len(ticked) != 1:
                prompter.note("  Renaming needs exactly one file ticked.",
                              style="yellow")
            else:
                _rename_one(prompter, kind, next(iter(ticked)))
                ticked.clear()
        elif picked == _DELETE:
            if _delete_ticked(prompter, kind, sorted(ticked)):
                ticked.clear()
        elif picked == _APPLY:
            if len(ticked) != 1:
                prompter.note("  Changing a model's settings needs exactly one "
                              "file ticked.", style="yellow")
            else:
                edit_model_settings(prompter, lib.kind_dir(kind) / next(iter(ticked)))
        else:
            # enter on an entry still toggles -- a forgiving alias for space,
            # at the cost of a repaint.
            ticked.symmetric_difference_update({picked})


_APPLY_DONE = ":done"
_TYPE_ENCODER = ":type"


def pick_encoder(prompter: Prompter, current: str = "",
                 allow_predictors: bool = True) -> Optional[str]:
    """
    Choose a transformer encoder from what this machine already has.

    Three shelves, then a text box. Your library's text encoders (adapted
    to a corpus here) and, when ``allow_predictors``, your fine-tuned
    predictors (a warm start: their encoder continues and their heads are
    reused). The models already downloaded to the Hugging Face cache, so a
    second run costs no download. The curated names not yet downloaded,
    marked as such. And "something else", for any other Hugging Face name
    or checkpoint folder, typed. Returns the chosen name or path, or None
    for Esc (leave the setting as it was).
    """
    from ..helpers import library as _lib
    from ..helpers.model_spec import describe_all, describe_encoder
    from ..text._transformer_common import CURATED_ENCODERS, cached_hub_encoders

    rows: List[Choice] = []
    for entry in _lib.entries(_lib.KINDS["encoders"]):
        try:
            label, note = describe_encoder(entry)
        except Exception:
            continue
        rows.append(Choice(str(entry.resolve()), label,
                           f"your library · adapted encoder · {note}" if note
                           else "your library · adapted encoder"))
    if allow_predictors:
        for info in describe_all(_lib.entries(_lib.KINDS["models"])):
            if info.type_id == "text_predictor":
                rows.append(Choice(str(info.path.resolve()), info.display(),
                                   "your library · warm start: its encoder continues, "
                                   "its heads are reused for outcomes of the same name"))
    # "meaning-tuned" against "raw encoder" is the one distinction a non-expert
    # needs here: a sentence-transformers model was trained so that similar
    # texts get similar vectors, a plain encoder wasn't
    def kind_of(hub_id: str) -> str:
        return ("meaning-tuned" if hub_id.startswith("sentence-transformers/")
                else "raw encoder")

    cached = dict(cached_hub_encoders())
    for hub_id, note in cached.items():
        curated = dict(CURATED_ENCODERS).get(hub_id, "")
        rows.append(Choice(hub_id, hub_id, f"{kind_of(hub_id)} · {note}"
                           + (f" · {curated}" if curated else "")))
    for hub_id, note in CURATED_ENCODERS:
        if hub_id not in cached:
            rows.append(Choice(hub_id, hub_id, f"{kind_of(hub_id)} · not downloaded yet "
                                                f"(first run fetches it) · {note}"))
    rows.append(Choice(_TYPE_ENCODER, "Something else — type a name or path",
                       "Any Hugging Face encoder name, or a folder holding a checkpoint."))
    prompter.reason(
        "The encoder to start from. Your own come first, then what is already "
        "on this machine, then the usual names.")
    default = current if any(c.value == current for c in rows) else None
    try:
        picked = str(prompter.select("Which encoder?", rows, default=default))
        if picked == _TYPE_ENCODER:
            picked = str(prompter.text("Encoder name or folder:", default=current)).strip()
            if not picked:
                return None
    except GoBack:
        return None
    return picked


def edit_model_settings(prompter: Prompter, path: Path) -> None:
    """
    One saved model's editable defaults: name, output columns, class labels,
    apply settings.

    A model trained on a 0/1 condition column predicts "0" and "1" -- right,
    and unreadable in a results table a month later -- and a word-vector
    model that should weight by types has no way to say so from inside a
    pipeline. Both live in the model file (``class_names``, ``apply``) and
    both are edited here, one at a time, with the current value as the
    default; every edit is written through :func:`edit_model`, which
    validates and refuses by name. Esc or Done returns to the list.
    """
    from rich.markup import escape

    from ..helpers.model_spec import (BY_ID, UnknownModel, describe,
                                      edit_model)

    while True:
        try:
            info = describe(path)
        except (UnknownModel, FileNotFoundError, OSError) as e:
            prompter.note(f"  {escape(str(e))}", style="yellow")
            return
        spec = BY_ID[info.type_id]
        rows = [Choice(_APPLY_DONE, "↩ Done", tone="nav"),
                Choice("name", f"Name: {info.name}", "What every menu calls it.")]
        if spec.write_outputs is not None:
            shown = (f"{info.n_outputs} columns, {info.outputs[0]}…" if info.bulk_outputs
                     and info.n_outputs > 3 else ", ".join(info.outputs))
            rows.append(Choice("outputs", f"Output columns: {shown}",
                               "The column names its predictions get."))
        for outcome, classes in info.classes.items():
            rows.append(Choice(f"classes:{outcome}",
                               f"Classes of {outcome}: {', '.join(classes)}",
                               "What each predicted class is written as."))
        for key, setting in spec.apply_settings.items():
            shown = info.apply.get(key)
            if setting.kind.startswith("library:"):
                # the model stores the dictionaries' content, but the row
                # shows their names, since that's what a person would recognize
                shown = ", ".join(str(d.get("name")) for d in (shown or [])) or "none"
            rows.append(Choice(f"apply:{key}", f"{key}: {shown}", setting.help))
        prompter.reason(
            f"{info.display()}. These are the model's own defaults: every "
            "pipeline that applies it uses them unless a step says otherwise.")
        try:
            picked = str(prompter.select("Change what?", rows))
        except GoBack:
            return
        if picked == _APPLY_DONE:
            return
        try:
            if picked == "name":
                name = str(prompter.text("What do you want to call this model?",
                                         default=info.name)).strip()
                if name:
                    edit_model(path, name=name)
            elif picked == "outputs":
                if info.bulk_outputs:
                    prefix = str(prompter.text(
                        f"Column name prefix for its {info.n_outputs} outputs "
                        f"(they become <prefix>_1 … <prefix>_{info.n_outputs}):",
                        default=_prefix_of(info.outputs))).strip()
                    if prefix:
                        edit_model(path, prefix=prefix)
                else:
                    labels = [str(prompter.text(
                        f"Column name for '{o}' (the results get 'pred_<this>'):",
                        default=o)).strip() or o for o in info.outputs]
                    edit_model(path, outputs=labels)
            elif picked.startswith("classes:"):
                outcome = picked[len("classes:"):]
                fitted = spec.classes(_doc_of(path))[outcome] if spec.classes else []
                current = info.class_names.get(outcome, {})
                prompter.reason(
                    f"The classes of {outcome} as the data spelled them, and what "
                    "each is written as in the results. Enter keeps a label.")
                mapping = {}
                for cls in fitted:
                    mapping[str(cls)] = str(prompter.text(
                        f"Label for class '{cls}':",
                        default=current.get(str(cls), str(cls)))).strip()
                edit_model(path, class_names={outcome: mapping})
            elif picked.startswith("apply:"):
                key = picked[len("apply:"):]
                setting = spec.apply_settings[key]
                current = info.apply.get(key)
                if setting.kind.startswith("library:"):
                    kind_id = setting.kind.split(":", 1)[1]
                    prompter.reason(
                        f"Tick the {lib.kind_by_id(kind_id).label.lower()} this model "
                        f"should measure; none ticked means none.")
                    picked = pick_from_library(prompter, lib.kind_by_id(kind_id),
                                               start_unticked=True)
                    if picked is None:
                        continue
                    value: object = picked
                elif setting.kind == "bool":
                    value = prompter.confirm(f"{key}?", default=bool(current))
                elif setting.choices:
                    value = str(prompter.select(
                        f"{key}:", [Choice(c, c) for c in setting.choices],
                        default=str(current) if str(current) in setting.choices else None))
                else:
                    value = str(prompter.text(f"{key}:", default="" if current is None
                                              else str(current))).strip()
                edit_model(path, apply={key: value})
            prompter.note("  Saved.", style="green")
        except GoBack:
            continue
        except ValueError as e:
            prompter.note(f"  {escape(str(e))}", style="yellow")


def _doc_of(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_files(prompter: Prompter, kind: lib.LibraryKind) -> None:
    """
    One screen: browse folders, tick dictionary files where they sit, import.

    A browser that is also the selector -- files are toggle rows, folders open
    on enter, and the selection survives walking between folders, so a set
    spread across a whole DICTIONARIES tree is still one trip. Two earlier
    shapes (single-file browsing; folder-then-checkbox) both hid the files at
    the moment they mattered.
    """
    try:
        picked = browse_and_tick(
            prompter,
            question=f"[space] ticks, [enter] imports ({kind.help})",
            suffixes=kind.suffixes)
    except (GoBack, Cancelled):
        return      # changed their mind; back to the list
    if not picked:
        return

    # importing now means *parsing*: we hand each file to the kind's real
    # reader before accepting it, and the first file also pays for importing
    # that reader. usually instant -- but on a slow drive or a big .dicx the
    # screen would sit there silent looking hung, so let's say so first.
    prompter.working(
        f"Checking and importing {len(picked)} "
        f"file{'' if len(picked) == 1 else 's'}…"
    )
    imported = 0
    landed = []
    for src in picked:
        try:
            landed.append(lib.import_into(kind, src))
        except ValueError as e:
            prompter.note(f"  {e}", style="yellow")
            continue
        except lib.LibraryCollision as e:
            if not prompter.confirm(
                f"'{e.existing.stem}' is already in the library. Replace it?",
                default=False,
            ):
                continue
            landed.append(lib.import_into(kind, src, replace=True))
        except OSError as e:
            prompter.note(f"  {e}", style="yellow")
            continue
        imported += 1
    prompter.note(f"  Imported {imported} of {len(picked)}.", style="green")
    for path in landed:
        _name_a_model(prompter, path)


def import_hf_classifier_ui(prompter: Prompter, kind: lib.LibraryKind) -> Optional[Path]:
    """
    Bring a classifier from the Hugging Face hub into the model library.

    Three ways in, on one list: the text classifiers already in this
    machine's Hugging Face cache (each with its labels), a checkpoint folder
    browsed to, or a hub name typed and downloaded on first use. The
    checkpoint's config is read before anything is copied, so a bare
    encoder, a multi-label head or an audio model is refused right here,
    in words that say what it is. Then two names: what the model predicts
    (the stem of its columns) and what to call it. Returns the library
    entry, or None when nothing was imported.
    """
    import tempfile

    from rich.markup import escape

    from ..helpers.model_spec import describe, slug
    from ..text.hf_classifier import (cached_hub_classifiers, checkpoint_problem,
                                      import_hf_classifier, inspect_checkpoint)
    from .browse import browse_for_folder

    rows: List[Choice] = []
    try:
        cached = cached_hub_classifiers()
    except Exception:
        cached = []
    for hub_id, note in cached:
        rows.append(Choice(hub_id, hub_id, note))
    rows.append(Choice(_HF_FOLDER, "A checkpoint folder on this computer",
                       "Browse to a folder holding config.json and the weights "
                       "(a download from the hub, or something you trained)."))
    rows.append(Choice(_HF_NAME, "A Hugging Face model name",
                       "Typed, e.g. cardiffnlp/twitter-roberta-base-sentiment-latest; "
                       "downloaded on first use."))
    prompter.reason(
        "A classifier or regressor somebody else trained and published. Once "
        "imported it scores any dataset from the feature checklist, like a model "
        "you trained here. What is already in this computer's Hugging Face cache "
        "comes first.")
    try:
        picked = str(prompter.select("Which model?", rows))
        if picked == _HF_FOLDER:
            source = str(browse_for_folder(
                prompter, question="Which checkpoint folder?",
                want_files=(".safetensors", ".bin", ".json")))
        elif picked == _HF_NAME:
            source = str(prompter.text("Hugging Face model name:", default="")).strip()
            if not source:
                return None
        else:
            source = picked
    except GoBack:
        return None

    prompter.working(f"Reading {source}…")
    try:
        info = inspect_checkpoint(source)
        problem = checkpoint_problem(info)
    except (ValueError, OSError) as e:
        problem = str(e)
    if problem:
        prompter.note(f"  {escape(problem)}", style="yellow")
        return None

    if info.task == "regression":
        what = "a regression head: one number per text"
    elif info.task == "multi_label":
        what = (f"{len(info.labels)} labels, several of which can apply to one "
                f"text: {', '.join(info.labels)}")
    else:
        what = f"{len(info.labels)} classes: {', '.join(info.labels)}"
    default = slug(info.short_name, fallback="label")
    prompter.reason(
        f"{info.short_name} has {what}. What it predicts becomes the stem of "
        f"its columns (pred_<this>, p_<this>_<class>), so name it for what it "
        f"measures -- 'sentiment', say -- rather than after the model's file.")
    try:
        outcome = str(prompter.text("What does it predict? (the column stem)",
                                    default=default)).strip() or default
        name = str(prompter.text("What do you want to call this model?",
                                 default=default)).strip() or default
    except GoBack:
        return None

    # a folder gets copied into the library, so a big checkpoint takes a
    # moment; we write the manifest (and the copy) into a scratch folder and
    # let the ordinary import move it, so every rule the import screen
    # applies -- the gate, the collision question -- applies here too
    prompter.working("Importing…")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            manifest = import_hf_classifier(source=source, out_dir=tmp, name=name,
                                            outcome=outcome)
        except (ValueError, OSError) as e:
            prompter.note(f"  {escape(str(e))}", style="yellow")
            return None
        try:
            landed = lib.import_into(kind, manifest)
        except lib.LibraryCollision as e:
            if not prompter.confirm(
                    f"'{e.existing.stem}' is already in the library. Replace it?",
                    default=False):
                return None
            landed = lib.import_into(kind, manifest, replace=True)
        except (ValueError, OSError) as e:
            prompter.note(f"  {escape(str(e))}", style="yellow")
            return None
    prompter.note(f"  Imported '{describe(landed).display()}'; it adds pred_{outcome}.",
                  style="green")
    return landed


def _name_a_model(prompter: Prompter, path: Path) -> None:
    """
    Offer to name a freshly imported model, and its output columns.

    Asked at import because this is the one moment the user knows what the
    model is. A model file is named by whatever wrote it -- ``ridge__all__age``
    -- and its columns are named after the outcome it was fitted on, so a
    ridge that predicts age from a blog corpus arrives as ``ridge__all__age``
    and writes ``pred_age``. Score two such models in one run
    and the second column overwrites the first; score one against a dataset
    that already has an ``age`` column and the results are worse than a
    collision, because both readings are plausible.

    Skipped silently for anything that is not a model -- a dictionary has a
    filename that already says what it is -- and skippable for models too:
    the default is what the file already says, so enter three times leaves
    everything as it was.
    """
    from ..helpers.model_spec import describe, rename_model

    try:
        info = describe(path)
    except Exception:
        return          # not a model, or not one we can describe: no names
    if not prompter.confirm(
            f"Name this {info.type_label} model and its columns?",
            default=True):
        return
    try:
        name = str(prompter.text(
            "What do you want to call this model?",
            default=info.name)).strip()
        if info.bulk_outputs:
            # a hundred themes don't get named one at a time. the prefix is
            # what tells one topic model's columns from another's, which is the
            # whole problem here.
            prefix = str(prompter.text(
                f"Column name prefix for its {info.n_outputs} themes "
                f"(they become <prefix>_1 … <prefix>_{info.n_outputs}):",
                default=_prefix_of(info.outputs))).strip()
            rename_model(path, name=name or None, prefix=prefix or None)
        else:
            labels = []
            for output in info.outputs:
                labels.append(str(prompter.text(
                    f"Column name for what it predicts ('{output}') — "
                    f"the results get 'pred_<this>':",
                    default=output)).strip() or output)
            rename_model(path, name=name or None, outputs=labels)
    except GoBack:
        return          # leave it exactly as the file says
    except (ValueError, OSError) as e:
        from rich.markup import escape

        prompter.note(f"  {escape(str(e))}", style="yellow")
        return
    prompter.note(f"  Saved as '{describe(path).display()}'.", style="green")


def _prefix_of(outputs: Sequence[str]) -> str:
    """The stem the existing theme names already share, for a default.

    A model imported once and renamed keeps its own prefix as the default on
    the next import rather than reverting to `Theme`, which would silently
    undo the naming on a re-import.
    """
    first = str(outputs[0]) if outputs else "Theme"
    return first.rsplit("_", 1)[0] if "_" in first else first


def _rename_one(prompter: Prompter, kind: lib.LibraryKind, name: str) -> None:
    shown = Path(name).stem
    try:
        new_name = str(prompter.text("New name:", default=shown)).strip()
    except GoBack:
        return
    if not new_name or new_name == shown:
        return
    try:
        renamed = lib.rename(kind, name, new_name)
    except lib.LibraryCollision as e:
        prompter.note(f"  {e}.", style="yellow")
        return
    except (ValueError, OSError) as e:
        # ValueError included: a slash in the name gets refused with a reason,
        # and if we didn't show that reason it'd look like the rename silently
        # failed.
        from rich.markup import escape

        prompter.note(f"  {escape(str(e))}", style="yellow")
        return
    prompter.note(f"  Renamed to '{renamed.stem}'.", style="green")


def _delete_ticked(prompter: Prompter, kind: lib.LibraryKind,
                   names: Sequence[str]) -> bool:
    """Confirm once, delete all. Returns True if anything was deleted."""
    listed = ", ".join(Path(n).stem for n in names)
    try:
        sure = prompter.confirm(
            f"Delete {len(names)} from the library ({listed})?", default=False)
    except GoBack:
        return False
    if not sure:
        return False
    gone = 0
    for name in names:
        try:
            lib.delete(kind, name)
            gone += 1
        except OSError as e:
            prompter.note(f"  {e}", style="yellow")
    prompter.note(f"  Deleted {gone} of {len(names)}.", style="dim")
    return gone > 0


def _warn_about_embedded_content(prompter: Prompter, kind: lib.LibraryKind,
                                 names: Sequence[str]) -> None:
    """
    Say what is inside a model before it leaves this machine.

    A model carries the text of the word lists it was fitted with, which is
    what makes it portable -- a colleague can score with it having never seen
    your connective lists. It also means the file *contains* those lists, and
    the moment that matters is this one: exporting, emailing, or posting a
    model to a repository.

    Said here rather than decided at fit time, because it is the sharer's call
    and not the tool's. What is licensed, what is confidential and what is
    fine to publish is a judgment about the material, and only the person
    sharing it can make it.
    """
    from ..helpers.model_spec import LIBRARY_KIND, describe

    if kind.id != LIBRARY_KIND:
        return
    carried: dict = {}
    import json

    for name in names:
        path = lib.kind_dir(kind) / name
        try:
            describe(path)
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for asset_kind, entries in (doc.get("assets") or {}).items():
            for entry in entries:
                if entry.get("text") is not None:
                    carried.setdefault(asset_kind, set()).add(entry["name"])
    if not carried:
        return

    listed = "; ".join(f"{k}: {', '.join(sorted(v))}"
                       for k, v in sorted(carried.items()))
    prompter.note(
        "  These model files carry the word lists they were fitted with, so "
        "they work on a machine that does not have them.", style="yellow")
    prompter.note(f"  Inside: {listed}", style="yellow")
    prompter.note(
        "  Worth a look before you publish or send them on -- a word list "
        "can be licensed, or can contain material you would rather not "
        "share.", style="yellow")


def _export_files(prompter: Prompter, kind: lib.LibraryKind,
                  names: Sequence[str]) -> None:
    """Pick a destination once, copy every ticked entry out."""
    try:
        dest = browse_for_folder(prompter, question="Export to which folder?")
    except (GoBack, Cancelled):
        return

    _warn_about_embedded_content(prompter, kind, names)

    exported = 0
    for name in names:
        try:
            lib.export_to(kind, name, dest)
        except lib.LibraryCollision as e:
            if not prompter.confirm(
                f"'{e.existing.name}' already exists there. Replace it?",
                default=False,
            ):
                continue
            lib.export_to(kind, name, dest, replace=True)
        except OSError as e:
            prompter.note(f"  {e}", style="yellow")
            continue
        exported += 1
    prompter.note(f"  Exported {exported} of {len(names)} to {dest}",
                  style="green", wrap=False)


def _row_of(kind: lib.LibraryKind, entry) -> tuple:
    """
    How one entry reads in a list: its own words where it has them.

    Most kinds are fully described by their filename, and the suffix and
    size are the useful extra. Saved models are not: every one is a `.json`
    of no informative size, and the thing a person needs in order to pick
    the right one -- ridge or topic model, and what columns it adds -- is
    inside the file. So the kind supplies the row, and this stays the only
    place that knows a row can come from anywhere but the path.
    """
    if kind.describe_entry is not None:
        try:
            label, note = kind.describe_entry(entry)
            return str(label), str(note)
        except Exception:
            # a listing must not die on one damaged file: the import gate
            # already refused anything broken, so a file that fails here got in
            # some other way and is still worth showing by name.
            pass
    return entry.stem, f"{entry.suffix} · {_size_of(entry)}"


def pick_from_library(prompter: Prompter, kind: lib.LibraryKind,
                      current: Sequence[str] = (),
                      start_unticked: bool = False) -> Optional[List[str]]:
    """
    Choose which entries of one kind a pipeline step should use.

    The import browser's dialect, because the two screens are siblings and
    learning one should be learning both: space ticks in place, and **enter on
    an entry approves the selection** -- the ticked set plus the pointed row,
    the same union the browser returns. There is no Done row for the same
    reason the browser lost its confirm row: it sat off-screen above a long
    list, and enter meaning "approve" makes it redundant. The action rows
    (select all / select none / Manage…, which runs :func:`manage_library` and
    re-lists on return so a file imported mid-pick is offered immediately)
    sit under the entries and reloop.

    An empty selection cannot be approved -- enter always includes the pointed
    entry -- which is right: a step reading zero dictionaries fails minutes
    into a run. "None of these" is Esc, meaning "leave it as it was".

    Parameters
    ----------
    current
        The paths already chosen, if any. A folder means everything of this
        kind inside it -- the reading the analyzers give it, and what an
        untouched pipeline's saved preset carries -- so "all 10 selected" on
        the settings row opens as ten ticks, not ten empty boxes. With no
        current at all, everything starts ticked: "use all of my dictionaries"
        is the common case and the default the rest of the wizard assumes.

    ``start_unticked`` opens with nothing ticked even for a kind that
    usually starts full -- a setting that is *optional* (concept
    dictionaries for word vectors) must not quietly mean "all of them".

    Returns
    -------
    list of str, or None
        Absolute paths of the approved entries -- exactly what `dict_paths`
        and `archetype_csvs` accept -- or None for "leave it as it was" (Esc,
        or backing out of an empty library).
    """
    chosen = {str(p.resolve()) for p in lib.expand(kind, current)}
    single = kind.id in SINGLE_PICK_KINDS
    # nothing starts ticked for a single-pick kind (enter on an entry picks
    # that one alone) or for models (we don't want to quietly score with
    # every model in the library just because nobody chose)
    started_empty = (not chosen and not single and not start_unticked
                     and kind.id not in START_UNTICKED_KINDS)
    pointer: Optional[str] = None

    while True:
        entries = lib.entries(kind)
        if not entries:
            if not _offer_first_import(prompter, kind):
                return None
            continue

        if started_empty:
            chosen = {str(e.resolve()) for e in entries}
            started_empty = False

        keyed = {str(e.resolve()): e for e in entries}
        # a selection can point at an entry that's been renamed or deleted
        # since; dropping it here means the ticks always describe reality.
        chosen &= set(keyed)

        in_place = getattr(prompter, "ticks_in_place", True)

        def rows() -> list:
            out = []
            for key, e in keyed.items():
                label, note = _row_of(kind, e)      # once: it parses model JSON
                out.append(Choice(key, f"[{'x' if key in chosen else ' '}] {label}",
                                  note))
            if single:
                out.append(Choice(_MANAGE, f"Manage {kind.label.lower()}…",
                                  "Import, export, rename or delete entries."))
                return out
            if not in_place:
                out.insert(0, Choice(_USE_TICKED, "✓ Use the ticked ones",
                                     "Approves exactly the entries marked [x].",
                                     tone="good"))
            # "Use all of them" approves outright. its old shape ("Select
            # all", then find an entry and press enter) read as a button that
            # did nothing -- with everything ticked by default it changed no
            # visible state, and the reloop's repaint looked like a broken
            # blink back to the same screen.
            out.append(Choice(_ALL, "✓ Use all of them",
                              "Approves the full library for this step.",
                              tone="good"))
            out.append(Choice(_NONE, "Untick everything",
                              "Clear the ticks, then pick a few."))
            out.append(Choice(_MANAGE, f"Manage {kind.label.lower()}…",
                              "Import, export, rename or delete entries."))
            return out

        def navigate(value: str):
            # "Untick everything" happens inside the running prompt -- the
            # marks flip in place, no teardown, no flash -- just like the file
            # browser's folder walk.
            if value != _NONE:
                return None
            chosen.clear()
            return {"choices": rows(), "toggle_values": set(keyed),
                    "default": _NONE}

        choices = rows()
        if not any(c.value == pointer for c in choices):
            # start on the first ticked entry: a bare enter then approves the
            # selection exactly as it stands, since the union adds nothing.
            pointer = next((k for k in keyed if k in chosen), None)

        try:
            picked = str(prompter.select(
                (f"Which of your {kind.label.lower()} should this step use?"
                 if single else
                 f"Which {kind.label.lower()} should this step use? "
                 + ("[space] ticks, [enter] approves."
                    if in_place else "[enter] on an entry ticks it.")),
                choices,
                default=pointer,
                toggle_values=set() if single else set(keyed),
                ticked=None if single else chosen,
                navigate=None if single else navigate,
            ))
        except GoBack:
            return None         # keep whatever it was
        pointer = picked
        if single and picked in keyed:
            return [picked]
        if picked == _USE_TICKED:
            if chosen:
                return sorted(chosen)
            prompter.note("  Nothing is ticked yet.", style="yellow")
            continue
        if picked == _ALL:
            return sorted(keyed)
        elif picked == _NONE:
            # only a renderer that can't swap in place ever returns this; the
            # loop is its fallback.
            chosen.clear()
        elif picked == _MANAGE:
            manage_library(prompter, kind)
        else:
            return sorted(chosen | {picked})


def offer_library_import(prompter: Prompter, produced: Sequence[tuple]) -> List[Path]:
    """
    Offer to add the models a run produced to the library, one by one.

    ``produced`` is ``[(path, label), ...]`` from
    :func:`taters.helpers.model_spec.models_produced`. Each is a yes/no with
    the label on it; a yes imports into the kind the file belongs in
    (encoders or models), asks about a collision, and for a scoring model
    goes on to the naming questions. Returns the library paths landed.
    """
    from ..helpers.model_spec import library_kind_for

    landed: List[Path] = []
    for path, label in produced:
        path = Path(path)
        try:
            if not prompter.confirm(f"Add {label} to your library?", default=True):
                continue
        except (GoBack, Cancelled):
            break
        kind = lib.KINDS[library_kind_for(path)]
        try:
            dest = lib.import_into(kind, path)
        except lib.LibraryCollision as e:
            if not prompter.confirm(
                    f"{e.existing.name} is already in your library. Replace it?",
                    default=False):
                continue
            dest = lib.import_into(kind, path, replace=True)
        except ValueError as e:
            prompter.note(f"  Not added: {e}", style="yellow")
            continue
        prompter.note(f"  ✓ Added to your {kind.label.lower()}: {dest.stem}",
                      style="green")
        if kind.id == "models":
            _name_a_model(prompter, dest)
        landed.append(dest)
    return landed


def _offer_first_import(prompter: Prompter, kind: lib.LibraryKind) -> bool:
    """
    The empty-library contingency: explain, and offer a way forward.

    Returns False when the user backs out instead of importing, which the
    picker turns into "leave the setting as it was".
    """
    prompter.reason(
        f"You have no {kind.label.lower()} yet. This step needs at least one "
        "to do anything."
    )
    try:
        picked = str(prompter.select(
            "Import one now?",
            [
                Choice("import", "Import a file…", kind.help),
                Choice(_BACK, "Not now"),
            ],
        ))
    except GoBack:
        return False
    if picked == _BACK:
        return False
    import_files(prompter, kind)
    return True


def _plural(n: int) -> str:
    return f"{n} entr" + ("y" if n == 1 else "ies")
