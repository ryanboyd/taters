"""
Shared internals for the stats stage.

Small, private, and deliberately boring: the strict numeric reader every
analysis uses (NA is never zero, and a stray non-numeric token is refused by
name rather than coerced), Benjamini-Hochberg FDR, the resolution of
``feature_sets`` into named column groups, and the report-fragment writer that
lets each analysis contribute one section to ``report.md`` without any module
knowing about the others.

Nothing here is a pipeline step; nothing here is public API. The step
functions live in the sibling modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from ..helpers.atomic import atomic_write

PathLike = Union[str, Path]

#: Where the report fragments live, under the stats output folder.
SECTIONS_DIR = "_sections"

#: Fixed slot numbers, so fragments concatenate in analysis order however the
#: steps actually ran. A future analysis claims the next free slot.
SECTION_ORDER = {"assemble": 10, "group-differences": 20,
                 "correlations": 30, "ridge": 40, "text-predictor": 45,
                 "classifier": 50,
                 # these are just pictures of everything above, so they go last.
                 "word-clouds": 60}


#: How many column names a refusal lists before it stops counting. A table
#: of sentence embeddings has a thousand columns, and printing all of them
#: buried the one sentence that said what was wrong under four screens of
#: `e0, e1, e10, e100…` -- twice, because the finish screen repeats it (a
#: real report). Enough names to recognize the table by, then the count.
_NAMES_IN_REFUSAL = 12


def name_a_few(names: Sequence[str], limit: int = _NAMES_IN_REFUSAL) -> str:
    """
    Some of these names, and how many more there are.

    For error messages: a refusal has to be readable in a terminal, and a
    reader who needs the full list has the file itself.
    """
    names = [str(n) for n in names]
    if len(names) <= limit:
        return ", ".join(names)
    return (", ".join(names[:limit])
            + f", and {len(names) - limit} more")


def taters_version() -> str:
    """The installed version, or '' when running from a checkout."""
    try:
        from importlib.metadata import version

        return version("taters")
    except Exception:
        return ""


def parse_cell(raw: object) -> float:
    """
    One cell as a float, with blank meaning NaN -- never zero.

    Raises ValueError on anything that is neither blank nor a number; the
    caller adds the column name, because "could not parse 'n/a'" is useless
    without knowing where it sat.
    """
    text = ("" if raw is None else str(raw)).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    return float(text)


def numeric_column(values: Sequence[object], *, column: str):
    """
    A whole column as a float array, strictly.

    Blank cells become NaN. Any other unparseable value refuses, naming the
    column and the value -- silent coercion to NaN would quietly shrink every
    downstream N, which is the plausible-numbers failure this codebase refuses.
    """
    import numpy as np

    out = np.empty(len(values), dtype=float)
    for i, raw in enumerate(values):
        try:
            out[i] = parse_cell(raw)
        except ValueError:
            raise ValueError(
                f"column {column!r} holds {str(raw).strip()!r}, which is not "
                f"a number. Fix the value, or leave the cell blank to mean "
                f"'missing'."
            ) from None
    return out


def looks_numeric(values: Sequence[object]) -> bool:
    """Whether every non-blank value parses as a number, and at least one does."""
    seen = False
    for raw in values:
        text = ("" if raw is None else str(raw)).strip()
        if not text:
            continue
        try:
            float(text)
        except ValueError:
            return False
        seen = True
    return seen


#: The multiple-comparison adjustments offered, with the one-line
#: explanation each screen shows. Ordered from least to most conservative,
#: which is also the order a researcher weighs them in.
P_ADJUST_METHODS = {
    "none": "No adjustment -- raw p-values only",
    "fdr_bh": "Benjamini-Hochberg FDR (default) -- expected proportion of "
              "false discoveries",
    "fdr_by": "Benjamini-Yekutieli FDR -- valid under any dependence between "
              "features, and stricter for it",
    "holm": "Holm-Bonferroni -- family-wise error, uniformly stronger than "
            "Bonferroni",
    "bonferroni": "Bonferroni -- family-wise error, the strictest and "
                  "simplest",
}


def adjust_pvalues(pvalues: Sequence[float],
                   method: str = "fdr_bh") -> List[float]:
    """
    Adjusted p-values by the chosen method, or the raw ones under "none".

    Which adjustment to make is a methodological decision, not a default to
    be imposed: controlling the false discovery rate and controlling the
    family-wise error rate answer different questions, and there are analyses
    where correcting at all is the wrong move (a single planned contrast).
    So every method here, including doing nothing, is a choice the caller
    makes.

    ``fdr_by`` is worth knowing about for language features specifically:
    Benjamini-Hochberg assumes positive dependence between tests, and 160
    cohesion indices measuring overlapping things are dependent in ways
    nobody has characterized. Benjamini-Yekutieli is valid under *any*
    dependence, at the cost of being stricter by a factor of the harmonic
    number of the family size.

    NaN entries stay NaN and do not count toward the family size -- a
    feature whose test could not run is not one of the tests being corrected
    for. Adjusted values are monotone in the raw ones and capped at 1.

    Parameters
    ----------
    pvalues
        The raw p-values of one family of tests.
    method
        One of :data:`P_ADJUST_METHODS`.

    Returns
    -------
    list of float
        Adjusted p-values, aligned with the input.
    """
    import numpy as np

    # we refuse `None` here rather than coercing it. this is because
    # `str(None).lower()` is the string "none", which is a legal method -- so
    # a null sneaking in from a hand-edited preset (or a sloppy caller) would
    # quietly switch multiple-comparison correction OFF and we'd be handing
    # back raw p-values as if they were adjusted. callers need to turn an
    # unset value into their documented default before they get here.
    if method is None:
        raise ValueError(
            "p-value adjustment is None. Pass the string 'none' to correct "
            f"nothing on purpose, or one of {sorted(P_ADJUST_METHODS)}.")
    method = str(method).strip().lower()
    if method not in P_ADJUST_METHODS:
        raise ValueError(
            f"unknown p-value adjustment {method!r}; choose one of "
            f"{sorted(P_ADJUST_METHODS)}")

    p = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted.tolist()
    raw = p[valid]

    if method == "none":
        adjusted[valid] = raw
        return adjusted.tolist()

    if method == "bonferroni":
        adjusted[valid] = np.minimum(raw * m, 1.0)
        return adjusted.tolist()

    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    ranks = np.arange(1, m + 1)

    if method == "holm":
        # holm steps *down*: we multiply each p by how many hypotheses are
        # left, then take a running max from the smallest p upward so that
        # the sequence can't decrease.
        scaled = np.maximum.accumulate(ranked * (m - ranks + 1))
    else:
        # the FDR family steps *up*. BY is just BH with an extra harmonic
        # factor (sum of 1/rank) -- that factor is what buys us validity
        # under arbitrary dependence.
        factor = float(np.sum(1.0 / ranks)) if method == "fdr_by" else 1.0
        scaled = ranked * m * factor / ranks
        scaled = np.minimum.accumulate(scaled[::-1])[::-1]

    scaled = np.minimum(scaled, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = scaled
    adjusted[valid] = out
    return adjusted.tolist()


def bh_fdr(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (the default adjustment)."""
    return adjust_pvalues(pvalues, "fdr_bh")


def split_rows(table, split_col: Optional[str] = None):
    """
    The subsets an analysis should run over, as ``(label, mask)`` pairs.

    Yields one unlabeled pass with no mask when ``split_col`` is empty --
    the ordinary case, one analysis over every row.

    Splitting exists for spreadsheets measured one text column at a time. A
    participant who answered three open-ended questions contributes three
    rows there, and analyzing them together would treat one person's three
    answers as three independent observations: the degrees of freedom, and
    therefore every p-value, would be wrong in the direction that flatters.
    Running the analysis once per column keeps the rows within each analysis
    independent, which is what the tests assume -- and it matches what the
    user already said by asking for the columns to be measured separately.
    """
    import numpy as np

    if not split_col:
        yield "", None
        return
    if split_col not in table.columns:
        raise ValueError(
            f"cannot split the analysis by {split_col!r}: the analysis table "
            f"has no such column. It has {len(table.columns)}: "
            f"{name_a_few(list(table.columns))}.")
    values = [str(v).strip() for v in table[split_col].tolist()]
    labels = sorted({v for v in values if v})
    if not labels:
        raise ValueError(
            f"cannot split the analysis by {split_col!r}: every value in it "
            f"is blank.")
    column = np.array(values)
    for label in labels:
        yield label, (column == label)


def resolve_feature_sets(
    feature_sets: Union[None, str, PathLike, Mapping[str, Sequence[str]]],
    *,
    table_csv: PathLike,
    feature_cols: Sequence[str],
) -> Dict[str, List[str]]:
    """
    ``feature_sets`` as the analyses accept it, normalized to {name: [cols]}.

    ``None`` means one combined set named "all" over every feature column of
    the table. The string ``"per_table"`` reads the sidecar the assemble step
    wrote next to the table and repeats the analysis per source feature file.
    A mapping is taken as-is (validated against the table); a path is read as
    a sidecar JSON. Unknown columns refuse by name -- a misspelled column must
    not silently become an empty analysis.
    """
    import json

    available = set(feature_cols)

    if feature_sets is None or feature_sets == "":
        return {"all": list(feature_cols)}

    if isinstance(feature_sets, Mapping):
        sets = {str(k): [str(c) for c in v] for k, v in feature_sets.items()}
    else:
        if feature_sets == "per_table":
            sidecar = Path(table_csv).with_name(
                Path(table_csv).stem + "_sets.json")
        else:
            sidecar = Path(feature_sets)
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"{sidecar} does not exist. feature_sets='per_table' needs "
                f"the sidecar the assemble step writes next to the analysis "
                f"table; re-run the assemble step, or pass explicit sets."
            )
        with sidecar.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        sets = {str(k): [str(c) for c in v]
                for k, v in (payload.get("sets") or {}).items()}

    if not sets:
        raise ValueError("feature_sets resolved to nothing: no sets named")
    for name, cols in sets.items():
        missing = [c for c in cols if c not in available]
        if missing:
            raise ValueError(
                f"feature set {name!r} names columns the analysis table does "
                f"not have: {missing[:6]}{'…' if len(missing) > 6 else ''}"
            )
        if not cols:
            raise ValueError(f"feature set {name!r} is empty")
    return sets


def default_feature_cols(table, table_csv: PathLike,
                         exclude: Sequence[str] = ()) -> List[str]:
    """
    Which of the table's columns are features, when the caller didn't say.

    The assemble step's sidecar is the authority when it exists -- it knows
    which columns are keys and metadata. Without it (a hand-made table fed
    straight to an analysis), every numeric-looking column that isn't
    explicitly excluded counts as a feature; the group/outcome columns an
    analysis was pointed at are excluded by their caller.
    """
    import json

    sidecar = Path(table_csv).with_name(Path(table_csv).stem + "_sets.json")
    if sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cols = [c for cols in (payload.get("sets") or {}).values()
                for c in cols]
        return [c for c in cols if c not in set(exclude)]
    banned = set(exclude)
    return [c for c in table.columns
            if c not in banned and looks_numeric(table[c].tolist())]


def read_str_csv(path, *, encoding: str):
    """A CSV as an all-string DataFrame -- no NA magic, no dtype guessing.
    Every analysis reads its table this way; it lived in `assemble` and four
    siblings imported it from there."""
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding=encoding)


def components_appendix(pca_notes) -> str:
    """
    The report paragraph that says the features below are components.

    Said in the report, not only in the log: a reader who sees "2 features"
    has to be told they are components built from eight, which features were
    set aside, and how many rows dropped out for want of one. Four analyses
    pasted this text; it is one function now.
    """
    if not pca_notes:
        return ""
    return ("\n\n### Components, not measures\n\n"
            "The features below are principal components, not the original "
            "measures. Read the loadings beside these results before any "
            "number that names one.\n\n" + "\n".join(pca_notes))


def write_section(stats_dir: PathLike, slug: str, markdown: str) -> Path:
    """
    One analysis's contribution to the report, as a markdown fragment.

    Written (atomically) to ``<stats_dir>/_sections/NN-<slug>.md`` where NN is
    the slug's fixed slot in :data:`SECTION_ORDER`. Always rewritten when the
    analysis actually runs; a skipped step leaves its previous fragment, which
    is correct because its CSVs were left too.
    """
    order = SECTION_ORDER.get(slug)
    if order is None:
        raise ValueError(
            f"unknown report section {slug!r}; known: {sorted(SECTION_ORDER)}")
    folder = Path(stats_dir) / SECTIONS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{order:02d}-{slug}.md"
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write(markdown.rstrip() + "\n")
    return path


def fmt(value: float, rounding: int = 4) -> str:
    """
    A number for a CSV cell: rounded, blank when NaN.

    A value too small for the rounding (a p of 3e-12 rounded to 4 places)
    switches to significant digits instead of collapsing to 0 -- "p = 0" is a
    claim no test can make, and reviewers rightly pounce on it.
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if value != 0.0 and abs(value) < 10 ** -rounding:
        return format(value, f".{max(1, rounding - 2)}g")
    rounded = round(value, rounding)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


#: A predictor missing for more than this fraction of the rows is dropped
#: rather than allowed to drop the rows. Half is deliberately permissive:
#: it only ever bites a column that is structurally undefined for most
#: texts, and leaves ordinary patchy data to listwise deletion.
MAX_MISSING = 0.5


def drop_sparse_predictors(cols, columns, mask, max_missing: float):
    """
    Separate the predictors worth keeping from the ones that would cost rows.

    Returns ``(usable, dropped)``, where ``dropped`` is a list of
    ``(column, missing_fraction)`` sorted worst first.

    This exists because listwise deletion has a failure mode that looks like
    success. A cohesion run writes ``adjacent_overlap_2_*_para`` columns,
    which compare paragraphs two apart and are therefore undefined for any
    text with fewer than three paragraphs -- most texts. In a real run 60 of
    166 predictors were blank for 885 of 938 rows, every one of those rows
    was then deleted for being incomplete, and a ridge was fitted on **52
    rows with 165 predictors** and reported an R-squared as if nothing had
    happened. Nothing in the output said the sample had lost 94% of its
    data.

    Dropping the column instead of the rows is the right way round: a
    predictor available for 6% of the corpus cannot predict anything about
    the other 94%, whereas the rows it destroys carry every other feature
    intact. The same run keeps 106 predictors and 904 rows this way.

    A threshold cannot be avoided by being clever -- some column has to give
    -- so the caller reports what was dropped and why, and the threshold is
    a documented setting rather than a constant buried here.
    """
    import numpy as np

    usable, dropped = [], []
    for c in cols:
        values = np.asarray(columns[c], dtype=float)
        if mask is not None:
            values = values[mask]
        share = float(np.isnan(values).mean()) if values.size else 1.0
        if share > max_missing:
            dropped.append((c, share))
        else:
            usable.append(c)
    dropped.sort(key=lambda pair: -pair[1])
    return usable, dropped


def source_tables(table_csv, cols) -> list:
    """
    Which feature tables supplied these columns, by the assemble sidecar.

    A saved model records its predictors by name, which is enough to *score*
    a table that already has them and no help at all in getting there. A
    user who fitted on cohesion features, came back a week later and loaded
    the model had no way to know the run needed a cohesion step -- the model
    named 165 columns and none of them said where they came from (a real
    report).

    Returns the table stems, sorted, or an empty list when the sidecar is
    missing: this is provenance for a human to act on, so being unable to
    determine it is not an error.
    """
    import json

    wanted = set(str(c) for c in cols)
    sidecar = Path(table_csv).with_name(Path(table_csv).stem + "_sets.json")
    if not sidecar.is_file():
        return []
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []
    found = []
    for stem, owned in (doc.get("sets") or {}).items():
        if wanted.intersection(str(c) for c in (owned or ())):
            found.append(str(stem))
    return sorted(found)


def zero_when_absent(table_csv, cols) -> list:
    """
    The predictors a scorer may read as zero when a table lacks them.

    A part-of-speech table's columns are whichever tags occurred, so a tag
    no text in a second corpus uses has no column there -- and the count it
    stands for is exactly zero. The analyzer says so in its record
    (``absent_means_zero``); this reads that through the assemble sidecar
    and names the columns, so the model file carries the permission per
    predictor and the scorer never has to guess from a column's name.

    Empty when the sidecar or the records are missing: without a record
    nothing is known, and "unknown" must fall on the refusing side.
    """
    import json

    from ..helpers import provenance

    wanted = set(str(c) for c in cols)
    sidecar = Path(table_csv).with_name(Path(table_csv).stem + "_sets.json")
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    sources = doc.get("sources") or {}
    out = []
    for stem, owned in (doc.get("sets") or {}).items():
        mine = [str(c) for c in (owned or ()) if str(c) in wanted]
        if not mine or not sources.get(stem):
            continue
        rec = provenance.read(sources[stem])
        if rec and rec.get("absent_means_zero"):
            out.extend(mine)
    return sorted(out)




def wanted_sets(pca, sets) -> list:
    """
    Which feature sets this analysis should reduce to components.

    ``pca`` is deliberately three-valued rather than a boolean, because the
    useful answer is often neither "all of it" nor "none of it":

    * ``"off"`` / False / None -- analyze the raw measures.
    * ``"all"`` / True -- reduce every feature set.
    * a list of set names -- reduce those and leave the rest alone. This is
      the case a boolean cannot express and researchers actually want: a
      hundred dictionary categories are worth reducing and eight readability
      indices are not, and they differ so much in number and in kind that one
      answer for both is the wrong answer for one of them.

    Set per analysis, not once for the run: raw variables read better in a
    correlation table, where each row is a measure you can name, while a
    ridge over four hundred collinear measures is exactly what components are
    for. Those are different questions about the same features and they get
    different answers.
    """
    if pca in (None, False, "", "off", "none"):
        return []
    if pca is True or pca == "all":
        return list(sets)
    if isinstance(pca, str):
        names = [n.strip() for n in pca.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in pca]
    unknown = [n for n in names if n not in sets]
    if unknown:
        raise ValueError(
            f"pca names feature set(s) that do not exist: "
            f"{name_a_few(unknown)}. This run has: {name_a_few(list(sets))}.")
    return names


def reduce_sets(sets, columns, *, pca, n_components: int, rotation: bool,
                out_stem, encoding: str, rounding: int, verbose: bool,
                max_missing: float = MAX_MISSING, retain: str = "parallel"):
    """
    Replace the chosen feature sets with component scores, in place.

    Returns ``(new_sets, new_columns, report_lines, reductions)``, where
    ``reductions`` is ``{set: {"features": [...], "axes": {...},
    "components": [...]}}`` -- the fitted axes and the raw features they
    were fitted on, so a model fitted on the components can carry the
    reduction and be applied to raw features later. The loadings and
    eigenvalues are written beside the analysis's own results and named after
    it, because a component only means something alongside the table saying
    what loads on it -- and two analyses reducing different sets have
    genuinely different components under the same names.

    Missing values are handled the same way every other analysis here handles
    them, and deliberately not by filling anything in:

    1. A feature missing for more than ``max_missing`` of the rows is
       **dropped**, by the same rule and the same default the prediction
       steps use -- see :func:`drop_sparse_predictors`. A measure defined for
       6% of the corpus cannot contribute a direction the other 94% lie
       along.
    2. Of the rows that remain, the axes are fitted on those with **every**
       kept feature present, and a row still missing one is left unscored
       rather than guessed at. Its components come back empty, so it drops
       out of the analysis exactly as a row missing any other feature does.

    Step 2 is listwise where a correlation table would be pairwise, and that
    is a real difference worth stating. A correlation is one number per pair
    and can use whatever rows that pair has. A PCA eigendecomposes the whole
    matrix at once, and a correlation matrix assembled from different subsets
    of rows per cell is not guaranteed to be positive semi-definite -- when
    it is not, the decomposition returns components explaining *negative*
    variance, which is nonsense that looks exactly like output. Dropping the
    sparse columns first is what makes the cost of listwise small: on the run
    that motivated this, dropping 60 structurally-blank columns took the
    complete rows from 52 to 904 of 938.

    Filters do not need handling here: the analyses receive the table
    ``assemble`` already filtered, so a row excluded by "at least 25 words"
    was gone before any of this ran.
    """
    import numpy as np

    from . import pca as _pca

    chosen = wanted_sets(pca, sets)
    if not chosen:
        return sets, columns, [], {}

    new_sets = {k: list(v) for k, v in sets.items()}
    new_columns = dict(columns)
    loadings: dict = {}          # set -> (component names, rows)
    eigen, lines = [], []
    reductions: dict = {}
    for name in chosen:
        cols = list(sets[name])
        if len(cols) < 2:
            raise ValueError(
                f"cannot reduce feature set {name!r}: it has "
                f"{len(cols)} column(s), and a PCA needs at least two "
                f"features to find a shared direction in.")
        # rule 1: if a feature is missing for most of the corpus, we drop the
        # feature rather than let it drop the corpus. same rule (and same
        # default) as the prediction steps use.
        usable, sparse = drop_sparse_predictors(cols, columns, None,
                                               max_missing)
        if len(usable) < 2:
            raise ValueError(
                f"cannot reduce feature set {name!r}: only {len(usable)} of "
                f"its {len(cols)} feature(s) are present for more than "
                f"{1 - max_missing:.0%} of the rows, and a PCA needs at "
                f"least two. The emptiest were "
                f"{name_a_few([c for c, _ in sparse[:6]])}.")
        matrix = np.column_stack([np.asarray(columns[c], dtype=float)
                                  for c in usable])
        # rule 2: we fit on the rows that have everything, and only score the
        # rest if they've got everything too. nothing gets filled in -- see
        # the docstring for why this is listwise where a correlation would
        # be pairwise.
        complete = ~np.isnan(matrix).any(axis=1)
        if int(complete.sum()) < 3:
            raise ValueError(
                f"cannot reduce feature set {name!r}: only "
                f"{int(complete.sum())} row(s) have every one of its "
                f"{len(usable)} features, and a PCA needs at least three.")
        axes = _pca.fit_in_memory(matrix[complete],
                                  n_components=n_components,
                                  rotation=rotation, retain=retain)
        scores = np.full((matrix.shape[0], axes["n_components"]), np.nan)
        scores[complete] = _pca.apply_in_memory(matrix[complete], axes)
        unscored = int((~complete).sum())
        cols_used = usable
        # the set's name, always -- even when it is the only set in the run and
        # nothing here needs telling apart. it used to be dropped in that case,
        # on the grounds that `Supertopic_1` is unambiguous when there is only
        # one thing it could have come from. it is, inside that run. it stops
        # being so the moment the column is quoted in a paper, merged into
        # another study's table, or read a year later, and the name is the only
        # thing that travels with it.
        prefix = f"{name}_"
        # a component built out of a topic model's topics is a supertopic, and
        # calling it one costs nothing here and saves a lookup every time
        # somebody reads the results.
        from ..helpers.feature_columns import reduced_name

        names = [f"{prefix}{c}" for c in _pca.component_names(
            axes["n_components"], reduced_name(usable, set_name=name))]
        new_sets[name] = names
        reductions[name] = {"features": list(usable), "axes": axes,
                            "components": list(names)}
        for i, column in enumerate(names):
            new_columns[column] = scores[:, i]
        loadings[name] = (names, [
            [name, feature] + [round(axes["loadings"][f][c], rounding)
                               for c in range(axes["n_components"])]
            for f, feature in enumerate([cols_used[i] for i in axes["kept"]])])
        for c in range(axes["n_components"]):
            eigen.append([name, names[c],
                          round(axes["eigenvalues"][c], rounding),
                          round(axes["pct_variance"][c], rounding)])
        explained = sum(axes["pct_variance"][:axes["n_components"]])
        lines.append(
            f"- **{name}**: {len(usable)} features → "
            f"{axes['n_components']} component(s), explaining "
            f"{explained:.1f}% of their variance, over "
            f"{int(complete.sum()):,} row(s). The count was "
            f"{_pca.describe_retention(axes.get('retention', {}))}.")
        if sparse:
            lines.append(
                f"  - {len(sparse)} feature(s) were set aside as missing for "
                f"more than {max_missing:.0%} of rows: "
                f"{name_a_few([c for c, _ in sparse[:6]])}.")
        if unscored:
            lines.append(
                f"  - {unscored:,} row(s) were missing at least one feature "
                f"and have no component scores, so they drop out of this "
                f"analysis.")
        if verbose:
            print(f"[pca] {name}: {len(usable)} features -> "
                  f"{axes['n_components']} component(s) over "
                  f"{int(complete.sum())} row(s)"
                  + (f"; {len(sparse)} sparse feature(s) set aside"
                     if sparse else ""))

    if loadings:
        # now we write one loadings table per reduced set, with the columns
        # named exactly the way the components are named in the results. we
        # used to write one wide table for every set under a shared
        # `Component_1..N` header, and it read like a single PCA over all of
        # them -- someone who reduced dictionary and cohesion separately saw
        # both sets of rows under the same eighteen columns and figured the
        # two had been mixed. they hadn't, but the file sure made it look
        # that way.
        several = len(loadings) > 1
        written = []
        for name, (comp_names, rows) in loadings.items():
            path = Path(f"{out_stem}_pca_loadings__{name}.csv" if several
                        else f"{out_stem}_pca_loadings.csv")
            _write_csv(path, ["feature_set", "feature"] + list(comp_names),
                       rows, encoding)
            written.append(path.name)
        if several:
            lines.insert(0, (
                "- Each feature set was reduced **on its own**: a component "
                "is built only from the measures of the set it is named "
                "after, and never mixes sets. Loadings, one table per set: "
                + ", ".join(f"`{w}`" for w in written) + "."))
        _write_csv(Path(f"{out_stem}_pca_eigenvalues.csv"),
                   ["feature_set", "component", "eigenvalue", "pct_variance"],
                   eigen, encoding)
    left = [n for n in sets if n not in chosen]
    if left:
        lines.append(f"- Left as raw measures: {', '.join(sorted(left))}.")
    return new_sets, new_columns, lines, reductions


def _write_csv(path: Path, header, rows, encoding: str) -> None:
    import csv

    from ..helpers.atomic import atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def gap_lines(set_name: str, cols, design) -> list:
    """
    The kept predictors that are blank in some rows, worst first.

    A predictor missing for *most* rows is dropped (see
    :func:`drop_sparse_predictors`); one missing for some rows is kept, and
    every row lacking it is unscorable -- now, and in any study the model
    is later applied to. A lexical measure undefined under 42 tokens was
    kept on a study of long texts and then left half of a shorter second
    study unscored, with nothing at fit time saying it could.

    Returns ``[(set, column, fraction_blank)]``; the report renders it.
    """
    import numpy as np

    a = np.asarray(design, dtype=float)
    if a.size == 0:
        return []
    frac = np.isnan(a).mean(axis=0)
    out = [(set_name, c, float(f)) for c, f in zip(cols, frac) if f > 0]
    return sorted(out, key=lambda t: (-t[2], t[1]))


def gaps_section(gaps, *, max_missing: float) -> list:
    """Markdown for :func:`gap_lines`, or nothing when every predictor is
    complete."""
    if not gaps:
        return []
    worst = gaps[:8]
    listed = ", ".join(f"`{c}` ({f:.0%})" for _s, c, f in worst)
    more = f", and {len(gaps) - len(worst)} more" if len(gaps) > len(worst) else ""
    return [
        "", "### Predictors with gaps", "",
        f"{len(gaps)} predictor(s) are blank in some rows and were kept, "
        f"because fewer than {max_missing:.0%} of rows lacked them: "
        f"{listed}{more}. A row lacking any of them cannot be scored -- "
        f"here, or in a study this model is later applied to, where "
        f"shorter texts may lack more of them. To score every text, leave "
        f"those measures out of the feature set or raise the minimum text "
        f"length."]


def reusable(output, *inputs, overwrite_existing: bool, verbose: bool,
             what: str) -> bool:
    """
    Whether an existing output may be handed back instead of being remade.

    The resume contract -- an existing result is returned untouched unless
    ``overwrite_existing`` -- has one qualification a spreadsheet person
    takes for granted and the code did not: a result older than the table
    it was computed from is not a result, it is a leftover. A real run
    rebuilt its metadata table (right columns this time), then the assemble
    step handed back the analysis table joined from the *old* one and the
    ridge could not find its outcomes. So: reusable when it exists, was not
    asked to be redone, and is at least as new as every input that exists.
    Says so when it is not.
    """
    output = Path(output)
    if overwrite_existing or not output.is_file():
        return False
    made = output.stat().st_mtime
    newer = []
    for path in inputs:
        if not path:
            continue
        for one in (path if isinstance(path, (list, tuple, set)) else [path]):
            p = Path(str(one))
            try:
                if p.is_file() and p.stat().st_mtime > made:
                    newer.append(p.name)
            except OSError:
                continue
    if newer:
        if verbose:
            print(f"[stats] {what} exists but {', '.join(newer)} changed "
                  f"since it was made; redoing it.")
        return False
    return True


def sample_warnings(metrics_rows, metrics_header, *, model: str = "ridge") -> list:
    """
    Say out loud when a model was fitted on much less data than existed.

    A silent sample loss is the failure this whole section guards against.
    In a real run a ridge reported an R-squared computed from 52 of 938
    rows -- with 165 predictors -- and every number in the table looked
    ordinary. A reader cannot ask "how much of my data went into this?"
    unless the answer is printed next to the answer they came for.

    Shared by the fitters rather than owned by ridge: the classifier wrote
    neither ``n_available`` nor this section, so a classifier fitted on a
    fifth of the corpus was reported with no warning at all while the guide
    promised one for "every metrics table".
    """
    if not metrics_rows or not metrics_header:
        return []
    at = {name: i for i, name in enumerate(metrics_header)}
    for key in ("n_used", "n_available", "n_predictors", "outcome"):
        if key not in at:                    # pragma: no cover - older header
            return []
    overfit = {
        "ridge": ("a ridge can fit almost anything; the out-of-fold scores "
                  "are the only ones worth reading, and a negative R-squared "
                  "here means the model is worse than the mean."),
        "classifier": ("a classifier can separate almost anything; the "
                       "out-of-fold scores are the only ones worth reading, "
                       "and an accuracy at the baseline means it learned "
                       "nothing."),
    }[model]
    lines = []
    for row in metrics_rows:
        used, available = int(row[at["n_used"]]), int(row[at["n_available"]])
        predictors = int(row[at["n_predictors"]])
        outcome = row[at["outcome"]]
        if available and used < 0.8 * available:
            lines.append(
                f"- **`{outcome}` used {used:,} of {available:,} rows.** The "
                f"rest were missing at least one predictor and were dropped "
                f"entirely. Treat the scores as describing those {used:,} "
                f"texts, not your corpus.")
        if predictors >= used:
            lines.append(
                f"- **`{outcome}` had {predictors:,} predictors for {used:,} "
                f"rows.** With at least as many predictors as observations "
                + overfit)
    if lines:
        lines = ["", "### About the sample", ""] + lines
    return lines
