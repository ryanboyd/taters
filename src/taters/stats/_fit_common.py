"""
The spine shared by the two model fitters -- ridge and the classifier.

They were written one after the other and each carried its own copy of the
same eighty lines: resolve the folders, validate the outcome list and the
penalty grid, read the table, refuse a missing outcome, resolve the feature
sets with the controls excluded, parse the numbers, reduce to components if
asked, build the controls and their subsets, split the rows. Then, per
feature set, the same sparse-predictor rule and the same model-file header;
and at scoring time the same read-match-design-write skeleton and the same
model-file prologue. Two 450-line functions that agreed on the day they were
written had already begun to disagree (one printed the worst sparse columns,
the other did not; one refused an empty penalty grid, the other fell over
inside numpy). This module is where those agreements live once.

What stays in each fitter is what genuinely differs: how an outcome is read
(a number or a class), the model that is fitted, and the tables it reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..helpers.model_spec import one_model_path
from ..helpers.progress import announce
from ._common import (MAX_MISSING, _write_csv, default_feature_cols,
                      drop_sparse_predictors, name_a_few, numeric_column,
                      reduce_sets, resolve_feature_sets, reusable,
                      source_tables, split_rows, taters_version,
                      zero_when_absent)

PathLike = Union[str, Path]


def random_folds(n: int, n_folds: int, seed: int):
    """Which fold each row belongs to -- a plain shuffle, deterministic by
    seed. The seed is stored in the model: a cross-validated number that
    cannot be reproduced is a number nobody can check."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % n_folds
    return folds


def balanced_folds(y, n_folds: int, seed: int):
    """
    Folds with the same spread of the outcome in each, and sizes within one.

    Rows are ordered by the outcome (ties broken by a seeded shuffle) and
    dealt in runs of ``n_folds``: each run hands one row to every fold, in a
    seeded order. Every fold therefore gets its share of the lowest, the
    middle and the highest values -- the regression analog of stratifying
    by class. A plain shuffle can put most of the high scores in one fold,
    and the fold-to-fold R-squared then says more about the deal than about
    the model. Falls back to the plain shuffle when there are too few rows
    to deal a full run to every fold.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    if n < 2 * n_folds:
        return random_folds(n, n_folds, seed)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    order = shuffled[np.argsort(y[shuffled], kind="stable")]
    folds = np.empty(n, dtype=int)
    for start in range(0, n, n_folds):
        run = order[start:start + n_folds]
        folds[run] = rng.permutation(n_folds)[:len(run)]
    return folds


def output_folders(table_csv: PathLike, out_dir: Optional[PathLike],
                   out_models_dir: Optional[PathLike]) -> Tuple[Path, Path, Path]:
    """``(table_csv, results folder, models folder)``, the folder created."""
    table_csv = Path(table_csv)
    folder = Path(out_dir) if out_dir else table_csv.parent
    models_dir = Path(out_models_dir) if out_models_dir else folder / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return table_csv, folder, models_dir


#: How feature tables analyzed together are combined into models. ``subsets``
#: fits every non-empty combination of the tables, ending with all of them,
#: so the report can say what each table adds to the others -- the question
#: "is the matrix worth having once the dictionary is in?" is a comparison
#: of two rows, not a new analysis. Bounded, because 2^k grows fast.
SET_COMBOS = ("none", "each_and_all", "subsets")
MAX_COMBO_SETS = 5


def source_sets(table_csv: Path, feature_cols: Sequence[str]):
    """
    The feature tables behind an analysis table, as the assemble step's
    sidecar names them, each restricted to the features this analysis may
    use. ``None`` without a sidecar or with fewer than two tables -- then
    there is nothing to combine.
    """
    import json

    sidecar = Path(table_csv).with_name(Path(table_csv).stem + "_sets.json")
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    allowed = set(feature_cols)
    sets = {}
    for name, cols in (payload.get("sets") or {}).items():
        kept = [str(c) for c in cols if str(c) in allowed]
        if kept:
            sets[str(name)] = kept
    return sets if len(sets) >= 2 else None


def combine_sets(sets, reductions, mode: str):
    """
    The models to fit from the source tables: each alone, every combination
    of them, and all of them, by ``mode``.

    Returns ``(sets, reductions, notes)``. A combination is named by its
    members joined with ``+`` in table order (``dictionary+readability``);
    the full combination keeps the name ``all`` that a single "together"
    analysis has always had, so nothing downstream moves. A member's
    reduction travels with it -- a combination of reduced tables carries a
    list of reductions, one per member, which the model file and the scorer
    both understand.

    ``subsets`` is bounded at :data:`MAX_COMBO_SETS` tables: six tables would
    be sixty-three models per outcome, so above the bound each table alone
    and all together are fitted and a note says so.
    """
    from itertools import combinations

    if mode not in SET_COMBOS:
        raise ValueError(f"set_combos must be one of {SET_COMBOS}, got {mode!r}")
    names = list(sets)
    notes = []
    if mode == "none" or len(names) < 2:
        cols = [c for n in names for c in sets[n]]
        reds = [reductions[n] for n in names if n in reductions]
        out = {"all": cols}
        out_red = {"all": reds[0] if len(reds) == 1 else reds} if reds else {}
        return out, out_red, notes
    sizes = range(1, len(names)) if mode == "subsets" else [1]
    if mode == "subsets" and len(names) > MAX_COMBO_SETS:
        total = 2 ** len(names) - 1
        notes.append(
            f"{len(names)} feature tables would make {total} combinations, "
            f"so each table was fitted alone and all together instead "
            f"(every combination is fitted for up to {MAX_COMBO_SETS} tables).")
        sizes = [1]
    out = {}
    out_red = {}
    for size in sizes:
        for members in combinations(names, size):
            name = "+".join(members)
            out[name] = [c for n in members for c in sets[n]]
            reds = [reductions[n] for n in members if n in reductions]
            if reds:
                out_red[name] = reds[0] if len(reds) == 1 else reds
    out["all"] = [c for n in names for c in sets[n]]
    reds = [reductions[n] for n in names if n in reductions]
    if reds:
        out_red["all"] = reds[0] if len(reds) == 1 else reds
    return out, out_red, notes


def members_of(set_names: Sequence[str], sources: Sequence[str]) -> Dict[str, List[str]]:
    """
    Which source tables each fitted set is made of.

    ``all`` is every table the sidecar knows; ``dictionary+readability`` its
    two members; anything else -- a single table, a hand-named set, a run
    with no sidecar -- is one set of its own. This is the ``n_feature_sets``
    column: how many tables went into the model, so a row reading 0.41 can
    be seen at a glance to be three tables' worth and not one.
    """
    known = set(sources)
    out: Dict[str, List[str]] = {}
    for name in set_names:
        if name == "all" and sources:
            out[name] = list(sources)
        elif "+" in name and all(part in known for part in name.split("+")):
            out[name] = name.split("+")
        else:
            out[name] = [name]
    return out


def comparison_lines(metrics_rows, metrics_header, *, score: str,
                     label: str, higher_is_better: bool = True) -> list:
    """
    The report's table of feature sets against each other: per outcome, one
    row per set with its best language-bearing score, ranked.

    Written for the reader of "what does each table add?" -- the rows for
    ``dictionary``, ``matrix`` and ``dictionary+matrix`` sit together and
    the difference is the answer.
    """
    at = {name: i for i, name in enumerate(metrics_header)}
    if not metrics_rows or score not in at:
        return []
    best: dict = {}
    for row in metrics_rows:
        if "language" not in row[at["model"]]:
            continue
        try:
            value = float(row[at[score]])
        except (TypeError, ValueError):
            continue
        key = (row[at["outcome"]], row[at["feature_set"]])
        n_pred = row[at["n_predictors"]] if "n_predictors" in at else ""
        if key not in best or (value > best[key][0]) == higher_is_better:
            best[key] = (value, n_pred)
    if len({s for _o, s in best}) < 2:
        return []
    lines = ["", "How the feature sets compare (the model with the language "
             "in it, alongside any controls):", "",
             f"| outcome | feature set | predictors | {label} |",
             "|---|---|---:|---:|"]
    outcomes = sorted({o for o, _s in best})
    for outcome in outcomes:
        rows = sorted(((s, v, n) for (o, s), (v, n) in best.items() if o == outcome),
                      key=lambda t: -t[1] if higher_is_better else t[1])
        for s, v, n in rows:
            lines.append(f"| `{outcome}` | {s} | {n} | {v:.3f} |")
    return lines


@dataclass
class FitInputs:
    """Everything a fitter needs before its first model, resolved once."""

    table: Any
    grid: List[float]
    sets: Dict[str, List[str]]
    columns: Dict[str, Any]
    pca_notes: List[str]
    #: Per reduced feature set, the fitted axes and the raw features behind
    #: them (see `_common.reduce_sets`); empty when nothing was reduced.
    reductions: Dict[str, dict]
    control_matrix: Any
    control_names: List[str]
    control_notes: List[str]
    control_spec: Any
    control_sets: List[tuple]
    subsets: List[tuple]
    lead: List[str]
    #: Models to fit in total, for the progress denominator.
    total: int = 0
    n_outcomes: int = field(default=0)
    #: What the report should say about how the sets were combined.
    set_notes: List[str] = field(default_factory=list)
    #: The source tables behind each fitted set, for the `n_feature_sets`
    #: column: `dictionary+readability` -> two, `all` -> every table.
    set_members: Dict[str, List[str]] = field(default_factory=dict)


def prepare_fit(*, table_csv: Path, folder: Path, stem: str,
                outcome_cols: Sequence[str], feature_sets, control_cols,
                categorical_controls, control_combos: str, pca,
                pca_components: int, pca_rotation: bool,
                split_col: Optional[str], alphas, n_folds: int,
                pca_retain: str = "parallel", set_combos: str = "subsets",
                max_missing: float, encoding: str, rounding: int,
                verbose: bool, on_progress, zero_alpha_reason: str) -> FitInputs:
    """
    The shared preamble of :func:`fit_ridge_csv` and :func:`fit_classifier_csv`.

    ``zero_alpha_reason`` finishes the sentence "every alpha must be greater
    than zero -- ": the two fitters refuse for different reasons and should
    say so.
    """
    from . import _controls
    from ._controls import control_subsets
    from ._common import read_str_csv
    from .ridge import default_alphas

    outcome_cols = [str(c) for c in outcome_cols]
    if not outcome_cols:
        raise ValueError("outcome_cols is empty: there is nothing to predict")
    grid = [float(a) for a in (alphas if alphas is not None else default_alphas())]
    if not grid:
        raise ValueError("alphas is empty: there is no penalty to choose from")
    if any(a <= 0 for a in grid):
        raise ValueError(
            "every alpha must be greater than zero -- " + zero_alpha_reason)
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    announce(on_progress, "reading the analysis table")
    table = read_str_csv(table_csv, encoding=encoding)
    missing = [c for c in outcome_cols if c not in table.columns]
    if missing:
        raise ValueError(
            f"outcome column(s) not in the analysis table: "
            f"{name_a_few(missing)}. It has {len(table.columns)}: "
            f"{name_a_few(list(table.columns))}.")

    # a control isn't a feature. it came in as a control, and if we left it
    # in the language block too we'd be counting it twice.
    feature_cols = default_feature_cols(
        table, table_csv,
        exclude=tuple(outcome_cols) + tuple(str(c) for c in control_cols))
    sets = resolve_feature_sets(feature_sets, table_csv=table_csv,
                                feature_cols=feature_cols)
    # now that we know which tables we've got, "together" means we fit them
    # alone AND in combination, so that the report can say what each one
    # adds. the reduction below then runs per table (a component that mixes
    # dictionary categories with readability indices can't really be given a
    # name), and the combinations get built from the reduced tables.
    known = source_sets(table_csv, feature_cols)
    sources = None
    if feature_sets in (None, "") and set_combos != "none":
        sources = known
        if sources:
            sets = sources
    columns = {c: numeric_column(table[c].tolist(), column=c)
               for cols in sets.values() for c in cols}
    # here's where we swap in components for raw measures, if this analysis
    # asked for them. we do the reduction per analysis and per feature set
    # because the right answer differs between them: raw variables read
    # better in a correlation table, whereas a ridge over hundreds of
    # collinear measures is exactly what components are for. one
    # `max_missing` governs both the features a model may be fit on and the
    # features a reduction gets built from.
    sets, columns, pca_notes, reductions = reduce_sets(
        sets, columns, pca=pca, n_components=pca_components,
        rotation=pca_rotation, out_stem=folder / stem, encoding=encoding,
        rounding=rounding, verbose=verbose, max_missing=max_missing,
        retain=pca_retain)
    set_notes: List[str] = []
    if sources:
        sets, reductions, set_notes = combine_sets(sets, reductions, set_combos)
    set_members = members_of(list(sets), list(known or []))

    control_matrix, control_names, control_notes, control_spec = \
        _controls.build(table, control_cols, categorical_controls)
    control_sets = control_subsets(control_cols, control_combos)
    subsets = list(split_rows(table, split_col))
    lead = [split_col] if split_col else []
    models_per_outcome = sum(2 if cs else 1 for cs in control_sets)
    return FitInputs(
        table=table, grid=grid, sets=sets, columns=columns,
        pca_notes=pca_notes, reductions=reductions,
        control_matrix=control_matrix,
        control_names=control_names, control_notes=control_notes,
        control_spec=control_spec, control_sets=control_sets,
        subsets=subsets, lead=lead,
        total=len(sets) * len(outcome_cols) * len(subsets) * models_per_outcome,
        n_outcomes=len(outcome_cols), set_notes=set_notes,
        set_members=set_members)


def set_design(set_name: str, all_cols: Sequence[str], columns, subset_mask,
               max_missing: float, *, verbose: bool, tag: str):
    """
    One feature set's design matrix, with the sparse-predictor rule applied.

    A predictor missing for most of the corpus is dropped instead of being
    allowed to delete most of the corpus -- see ``drop_sparse_predictors``:
    the difference between fitting on 904 rows and fitting on 52 while
    reporting neither. Returns ``(cols, sparse, design_all, design)``.
    """
    import numpy as np

    cols, sparse = drop_sparse_predictors(all_cols, columns, subset_mask,
                                          max_missing)
    if not cols:
        raise ValueError(
            f"every predictor in feature set {set_name!r} is missing for "
            f"more than {max_missing:.0%} of the rows, so there is nothing "
            f"left to fit on. The emptiest were "
            f"{name_a_few([c for c, _ in sparse[:6]])}.")
    if sparse and verbose:
        print(f"[{tag}] {set_name}: set aside {len(sparse)} predictor(s) "
              f"missing for more than {max_missing:.0%} of rows "
              f"(worst: {name_a_few([c for c, _ in sparse[:4]])})")
    design_all = np.column_stack([columns[c] for c in cols])
    design = design_all if subset_mask is None else design_all[subset_mask]
    return cols, sparse, design_all, design


def model_header(*, kind: str, fmt: int, set_name: str, cols: Sequence[str],
                 table_csv: Path, cv: dict, zscore: bool,
                 reduction: Optional[dict] = None, **extra) -> dict:
    """
    The part of a model file both fitters write the same way.

    ``reduction`` is the fitted PCA behind a set of component predictors
    (see `_common.reduce_sets`). A model fitted on components used to name
    ``Component_1`` as a predictor and nothing else -- no table produces a
    column by that name, so the model could never be applied. It now
    carries the axes and names the **raw** features as its inputs; the
    scorer rebuilds the components from those, standardized against the
    fitting sample, which is what makes the scores comparable at all.

    ``needs`` says where the predictors came from -- so a run a week later
    can be told what to extract rather than only what is missing -- and HOW
    they were measured: a predictor name is enough to refuse a missing
    column and no defense at all against a column of the right name holding
    numbers produced by different settings, which is what moved a real
    prediction by seven years in silence. ``assets`` embeds the word lists
    themselves, so a colleague can score with the model on a machine that
    has never seen them; whatever shares a model is responsible for saying so.
    """
    import datetime

    from ..helpers import provenance

    # what a run has to supply: the raw features behind any component, or
    # otherwise the predictors themselves. a combination of reduced tables
    # carries one reduction per table, so we've got a list to walk here.
    reductions = ([reduction] if isinstance(reduction, dict)
                  else list(reduction or []))
    components = {c for r in reductions for c in r["components"]}
    inputs = [c for c in cols if c not in components]
    for r in reductions:
        inputs += [f for f in r["features"] if f not in inputs]
    doc = {
        "kind": kind, "format": fmt,
        "created": datetime.date.today().isoformat(),
        "taters": taters_version(),
        "feature_set": set_name,
        "predictors": list(cols),
        "needs": {
            "feature_tables": source_tables(table_csv, inputs),
            "feature_provenance": provenance.for_columns(table_csv, inputs),
        },
        "assets": provenance.embedded_assets(table_csv, inputs),
        "zero_when_absent": zero_when_absent(table_csv, inputs),
        "cv": cv, "zscore": bool(zscore), "outcomes": {},
    }
    if reductions:
        doc["input_columns"] = inputs
        packed = [{
            "features": list(r["features"]),
            "components": list(r["components"]),
            "axes": {k: r["axes"][k] for k in ("kept", "mu", "sigma",
                                               "projection", "n_components")},
        } for r in reductions]
        if len(packed) == 1:
            doc["reduction"] = packed[0]
        else:
            doc["reductions"] = packed
    doc.update(extra)
    return doc


def load_model_doc(model_json: PathLike, *, kind: str, fmt: int, noun: str,
                   step: str, wrong_kind_hint: Optional[Callable[[str], str]] = None
                   ) -> Tuple[Path, dict, Callable[[str], ValueError]]:
    """
    Read a model file and refuse anything that is not one -- in words.

    The refusals are the point. A file that is the wrong kind of model, or
    was written by a newer Taters, or is structurally incoherent, must be
    turned away while the user is holding it and can act, not halfway
    through scoring a corpus. These messages are also what the library shows
    when someone imports a model file, so they are written for a person.

    Returns the path, the parsed document, and a ``broken(why)`` factory the
    caller uses for its own structural checks, so every "damaged" message has
    the same shape and names the same remedy.
    """
    path = Path(model_json)
    if not path.is_file():
        raise FileNotFoundError(f"no model file at {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            model = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"{path.name} is not readable as a model file: {e}") from None
    if not isinstance(model, dict):
        raise ValueError(f"{path.name} is not a Taters model file at all.")
    found = model.get("kind")
    if found != kind:
        what = f"a {found!r} file" if found else "not a Taters model file at all"
        hint = wrong_kind_hint(found) if (wrong_kind_hint and found) else ""
        raise ValueError(f"{path.name} is not a {noun} ({what}).{hint}")
    if int(model.get("format", 0)) > fmt:
        raise ValueError(
            f"{path.name} was written by a newer Taters "
            f"(model format {model.get('format')}; this one reads {fmt}). "
            f"Update Taters to use it.")

    def broken(why: str) -> ValueError:
        return ValueError(
            f"{path.name} is damaged or incomplete: {why}. Re-run the "
            f"{step} step to write a fresh one.")

    predictors = model.get("predictors")
    if not isinstance(predictors, list) or not predictors:
        raise broken("it names no predictors")
    outcomes = model.get("outcomes")
    if not isinstance(outcomes, dict) or not outcomes:
        raise broken("it holds no fitted outcomes")
    return path, model, broken


def score_design(table, model, predictors, input_csv, verbose: bool):
    """
    The design matrix for a table being scored, in the model's own order.

    Control columns are rebuilt from the model's stored recipe rather than
    looked up by their design names: the model knows a predictor called
    ``gender=male``, and the spreadsheet in front of us has a column called
    ``gender`` holding words. Everything else is matched by name, and a
    genuinely absent predictor is refused rather than imputed -- a model
    scored on a feature it never sees is not the model that was validated.

    The one exception is a column the model itself marks ``zero_when_absent``:
    a count of something -- a part-of-speech tag -- that simply never
    occurred in this corpus. Its absence *is* the measurement, and the
    column is scored as the zeros it would have held.

    A model fitted on components carries its reduction, and the components
    are rebuilt here from the raw features, standardized against the
    fitting sample. A row lacking any raw feature gets blank components and
    goes unscored, exactly as it would have at fit time.

    Returns ``(design, inputs, input_names)``: the design in the model's
    predictor order, and the matrix of what the table actually supplied
    (raw features and controls), for the unscored-row accounting.
    """
    import numpy as np

    from . import _controls

    spec = model.get("controls") or []
    control_matrix, control_names, unknown = _controls.rebuild(table, spec)
    known = dict(zip(control_names,
                     range(len(control_names)))) if control_names else {}
    if unknown and verbose:
        print(f"[stats] control value(s) never seen when fitting, so those "
              f"rows go unscored: {unknown}")
    reductions = list(model.get("reductions") or [])
    if model.get("reduction"):
        reductions.append(model["reduction"])
    components = [c for r in reductions for c in (r.get("components") or [])]
    wanted = [p for p in predictors if p not in components]
    for r in reductions:
        wanted += [f for f in (r.get("features") or []) if f not in wanted]
    zeroable = set(model.get("zero_when_absent") or ())
    zeros = [p for p in wanted
             if p not in known and p not in table.columns and p in zeroable]
    if zeros and verbose:
        print(f"[stats] {len(zeros)} column(s) the model knows never occur "
              f"in this table and count as zero: {name_a_few(zeros)}")
    absent = [p for p in wanted
              if p not in known and p not in table.columns and p not in zeroable]
    if absent:
        raise ValueError(
            f"{Path(input_csv).name} is missing {len(absent)} predictor(s) "
            f"this model needs: {name_a_few(absent)}. The model was fitted "
            f"on a different set of features; score a table produced the "
            f"same way.")
    n_rows = len(table.index)
    values = {
        p: control_matrix[:, known[p]] if p in known
        else np.zeros(n_rows) if p in zeros
        else numeric_column(table[p].tolist(), column=p)
        for p in wanted}
    inputs = np.column_stack([values[p] for p in wanted]) if wanted \
        else np.zeros((n_rows, 0))
    if components:
        from . import pca as _pca

        for reduction in reductions:
            names = list(reduction["components"])
            raw = np.column_stack([values[f] for f in reduction["features"]])
            complete = ~np.isnan(raw).any(axis=1)
            scores = np.full((n_rows, len(names)), np.nan)
            if complete.any():
                scores[complete] = _pca.apply_in_memory(raw[complete],
                                                        reduction["axes"])
            for i, name in enumerate(names):
                values[name] = scores[:, i]
    design = np.column_stack([values[p] for p in predictors])
    return design, inputs, list(wanted)


@dataclass
class ApplyInputs:
    model: dict
    table: Any
    design: Any
    id_cols: List[str]
    out_path: Path
    #: What the table supplied -- raw features and controls -- and their
    #: names, for the unscored-row accounting. The design itself when the
    #: predictors are those columns; the raw features when the predictors
    #: are components of them.
    input_names: List[str] = field(default_factory=list)
    inputs: Any = None


def prepare_apply(*, model_json: PathLike, load: Callable[[Path], dict],
                  input_csv: PathLike, out_csv: Optional[PathLike],
                  default_suffix: str, overwrite_existing: bool, on_progress,
                  verbose: bool, encoding: str,
                  id_cols: Optional[Sequence[str]] = None
                  ) -> Union[ApplyInputs, Path]:
    """
    The shared front half of scoring: the model, the table, the design.

    Returns the existing output path when the predictions are already there
    and ``overwrite_existing`` is False (the caller returns it unchanged),
    otherwise an :class:`ApplyInputs`.

    ``id_cols`` names the columns to carry beside the predictions. ``None``
    carries every column that is not a predictor, which is right for a
    table the user assembled -- their identifiers are whatever they are.
    The scoring step passes its join keys instead: its table is built from
    the model's own feature tables, so "everything else" there was a
    measure the fit had set aside as too sparse, and a raw control column,
    turning up in the scores file as if they were identifiers.
    """
    from ._common import read_str_csv

    model = load(one_model_path(model_json))
    input_csv = Path(input_csv)
    out_path = Path(out_csv) if out_csv else input_csv.with_name(
        input_csv.stem + default_suffix)
    if reusable(out_path, input_csv, model_json,
                overwrite_existing=overwrite_existing, verbose=verbose,
                what="the predictions"):
        if verbose:
            print(f"Predictions already exist; returning existing file: "
                  f"{out_path}")
        return out_path
    announce(on_progress, "reading the table to score")
    table = read_str_csv(input_csv, encoding=encoding)
    predictors = model["predictors"]
    design, inputs, input_names = score_design(table, model, predictors,
                                               input_csv, verbose)
    not_ids = set(predictors) | set(input_names)
    if id_cols is None:
        carried = [c for c in table.columns if c not in not_ids]
    else:
        carried = [str(c) for c in id_cols if str(c) in table.columns]
    return ApplyInputs(model=model, table=table, design=design,
                       id_cols=carried, out_path=out_path,
                       input_names=input_names, inputs=inputs)


def write_predictions(inputs: ApplyInputs, columns: Dict[str, list],
                      skipped: Dict[str, int], *, encoding: str,
                      on_progress, verbose: bool, tag: str) -> Path:
    """The shared back half: identifiers, then one column per prediction."""
    announce(on_progress, "writing the predictions")
    table = inputs.table
    header = inputs.id_cols + list(columns)
    rows = []
    for i in range(len(table)):
        row = [table[c].iloc[i] for c in inputs.id_cols]
        row += ["" if columns[c][i] is None else columns[c][i]
                for c in columns]
        rows.append(row)
    _write_csv(inputs.out_path, header, rows, encoding)
    unscored = [i for i in range(len(table))
                if any(columns[c][i] is None for c in columns)]
    entries = _blame(inputs, unscored)
    write_unscored(inputs.out_path, entries, encoding=encoding)
    if verbose:
        short = {k: v for k, v in skipped.items() if v}
        if short:
            print(f"[{tag}] rows left unscored for want of a predictor: "
                  f"{short}; which predictors, and how often, is in "
                  f"{unscored_path(inputs.out_path).name}")
        print(f"[{tag}] scored {len(rows)} row(s) -> {inputs.out_path}")
    return inputs.out_path


def unscored_path(out_path: PathLike) -> Path:
    """``model_scores.csv`` -> ``model_scores_unscored.csv``."""
    p = Path(out_path)
    return p.with_name(f"{p.stem}_unscored.csv")


def _blame(inputs: ApplyInputs, unscored: Sequence[int]) -> list:
    """
    Which inputs were blank in the rows that went unscored, and how often.

    A blank prediction has exactly one cause here -- a predictor the row
    did not have -- and the scorer knows which. In a real run half a study
    went unscored because one lexical measure is undefined under 42 tokens,
    and nothing but a verbose print said so; a wizard run showed nothing.
    """
    import numpy as np

    if not unscored:
        return []
    supplied = inputs.inputs if inputs.inputs is not None else inputs.design
    design = np.asarray(supplied, dtype=float)[list(unscored)]
    names = inputs.input_names or list(inputs.model.get("predictors") or [])
    entries = [("unscored", "rows with a blank prediction", len(unscored))]
    blanks = np.isnan(design).sum(axis=0) if design.size else []
    for name, n in sorted(zip(names, blanks), key=lambda t: (-t[1], t[0])):
        if n:
            entries.append(("blank predictor", name, int(n)))
    return entries


def write_unscored(out_path: PathLike, entries: Sequence[tuple], *,
                   encoding: str) -> Optional[Path]:
    """
    The row accounting beside a scores file: why each unscored row is blank.

    One small CSV, ``reason, detail, rows``, written only when something
    went unscored and removed otherwise, so its presence is itself the
    signal. The scoring step adds rows lost in the join (missing from a
    feature table altogether) under their own reason.
    """
    path = unscored_path(out_path)
    if not entries:
        path.unlink(missing_ok=True)
        return None
    _write_csv(path, ["reason", "detail", "rows"],
               [[r, d, str(n)] for r, d, n in entries], encoding)
    return path


def read_unscored(out_path: PathLike, *, encoding: str) -> list:
    """The entries :func:`write_unscored` wrote, or nothing."""
    import csv

    path = unscored_path(out_path)
    if not path.is_file():
        return []
    with path.open("r", encoding=encoding, newline="") as fh:
        return [(r["reason"], r["detail"], int(r["rows"]))
                for r in csv.DictReader(fh)]


__all__ = ["FitInputs", "ApplyInputs", "output_folders", "prepare_fit",
           "set_design", "model_header", "load_model_doc", "score_design",
           "prepare_apply", "write_predictions", "MAX_MISSING",
           "unscored_path", "write_unscored", "read_unscored"]
