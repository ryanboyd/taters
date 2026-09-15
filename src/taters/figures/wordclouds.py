"""
Which words go into which cloud, for every kind of result Taters writes.

Three entry points, one per stage of a run:

* :func:`stats_wordclouds` reads the tables under a ``stats_results`` folder
  -- ridge and classifier coefficients, correlations, pairwise group
  differences, PCA loadings, and the analysis table itself for per-group
  frequencies -- and draws one cloud per direction per outcome, plus a
  report section that shows them.
* :func:`theme_wordclouds` draws one cloud per theme of a topic model from
  its loadings table.
* :func:`frequency_wordclouds` draws the corpus's most frequent words from
  a frequency list.

Every function reads finished tables and computes no statistic of its own.
The choosing -- which rows, which model, which threshold -- is in pure
functions over lists of rows, tested without a picture ever being drawn; the
drawing is :mod:`taters.figures.render`. A cloud that would be empty is not
drawn and the report says why, because a missing picture reads as a bug and
"no feature passed p < .05" reads as a finding.

The folders follow the analyses: ``<analysis>/<feature set>/`` holds that
set's clouds, with ``components/`` and ``themes/`` inside it for what its
predictors are made of -- a ridge over topic-model themes shows "Theme_5"
as a word, and the picture that says what Theme 5 *is* belongs beside it,
not three folders away.

Pictures are redrawn only when the table they came from is newer than they
are, the same rule every statistics step follows, and the report section is
rewritten only when its text changes -- so a second run of an unchanged
pipeline writes nothing, which is the resume contract the whole pipeline
keeps.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..helpers.cliargs import CliSpec
from ..helpers.model_spec import slug
from ..helpers.progress import announce
from ..stats._common import SECTIONS_DIR, reusable, write_section
from .render import pillow_missing_reason, render_wordcloud

__all__ = ["CloudSpec", "stats_wordclouds", "theme_wordclouds",
           "frequency_wordclouds", "neighbor_wordclouds",
           "clouds_for_ridge", "clouds_for_classifier",
           "clouds_for_correlations", "clouds_for_group_differences",
           "clouds_for_components", "clouds_for_frequencies_by_group",
           "clouds_for_theme_predictors", "clouds_for_themes",
           "clouds_for_frequencies", "clouds_for_neighbors", "skip_redundant",
           "FIGURES_DIR",
           "SECTION_SLUG", "ANALYSES"]

PathLike = Union[str, Path]
Row = Dict[str, str]
Word = Tuple[str, float]

#: Where figures live under a results folder, and the report slot they fill.
FIGURES_DIR = Path("figures") / "wordclouds"
SECTION_SLUG = "word-clouds"

#: The analyses whose tables get clouds: the stem their files start with ->
#: (the report heading, the folder). Component clouds of an analysis land
#: under the same heading and folder as its other clouds.
ANALYSES: Dict[str, Tuple[str, str]] = {
    "ridge": ("Ridge regression", "ridge-regression"),
    "classifier": ("Classification", "classification"),
    "correlations": ("Correlations", "correlations"),
    "group_differences": ("Group differences", "group-differences"),
}
_FREQ_BY_GROUP = ("Most frequent terms by group", "most-frequent-terms-by-group")

#: The model a coefficient cloud is drawn from, in order of preference: the
#: language features fitted alongside every control (the language's own
#: contribution, which is the question), else language alone.
_FULL_MODELS = ("controls+language", "language")


@dataclass
class CloudSpec:
    """
    One picture to draw, or one sentence about why there is none.

    ``folder`` is where it goes under the figures root and ``name`` the file
    name without extension; ``analysis`` the heading it lists under in the
    report and ``set_name`` the feature set it belongs to; ``words`` the
    (label, weight) pairs; ``note`` replaces the picture when ``words`` is
    empty; ``stat`` names the statistic the weights are, for the sentence
    that says where a theme appeared ("TIPI_Open: β = +0.32"); ``kind`` is
    ``cloud``, ``component`` or ``theme``; ``inputs`` the tables it was drawn
    from, which decide whether it needs redrawing.
    """

    analysis: str
    name: str
    title: str
    legend: str
    words: List[Word] = field(default_factory=list)
    note: str = ""
    folder: str = ""
    set_name: str = ""
    stat: str = ""
    kind: str = "cloud"
    inputs: List[Path] = field(default_factory=list)

    @property
    def relative_png(self) -> str:
        """The picture's path under the figures root, POSIX style."""
        return (Path(self.folder) / f"{self.name}.png").as_posix()


# ---------------------------------------------------------------------------
# reading the tables in
# ---------------------------------------------------------------------------

def _rows(path: Path, encoding: str) -> List[Row]:
    with Path(path).open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def _num(value: Optional[str]) -> Optional[float]:
    """A cell as a number, or None for blank and non-numeric cells."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _split_col(rows: Sequence[Row], first: str = "feature_set") -> Optional[str]:
    """The split column's name, when the table has one: it is whatever comes
    before ``feature_set`` in the header."""
    if not rows:
        return None
    cols = list(rows[0].keys())
    if first not in cols:
        return None
    ahead = cols[:cols.index(first)]
    return ahead[0] if ahead else None


def _label(feature: str, feature_set: str) -> str:
    """The name a feature is shown under: assemble's ``<stem>.`` collision
    prefix comes off, nothing else does."""
    prefix = f"{feature_set}."
    return feature[len(prefix):] if feature.startswith(prefix) else feature


def _where(split: Optional[str], value: str) -> str:
    """The words that name a split's half in a title, or nothing."""
    return f" [{split} = {value}]" if split and value else ""


def _tag(split: Optional[str], value: str) -> str:
    """The same for a file name."""
    return f"__{slug(split)}-{slug(value)}" if split and value else ""


def _full_model(rows: Sequence[Row]) -> List[Row]:
    """
    The rows of the one model a coefficient cloud is drawn from.

    Fitters write every control combination they tried. The cloud shows the
    full one: ``controls+language`` with the most controls (the language's
    contribution once everything is held constant), and ``language`` alone
    when there were no controls. Ties keep the first seen.
    """
    best_key = None
    best_rank = None
    for r in rows:
        model = r.get("model", "")
        if model not in _FULL_MODELS:
            continue
        controls = [c for c in (r.get("controls") or "").split("|") if c]
        rank = (model == "controls+language", len(controls))
        key = (model, r.get("controls") or "")
        if best_rank is None or rank > best_rank:
            best_rank, best_key = rank, key
    if best_key is None:
        return []
    return [r for r in rows
            if (r.get("model", ""), r.get("controls") or "") == best_key]


def _grouped(rows: Iterable[Row], *keys: str) -> Dict[tuple, List[Row]]:
    out: Dict[tuple, List[Row]] = {}
    for r in rows:
        out.setdefault(tuple(r.get(k, "") for k in keys), []).append(r)
    return out


def _feature_filter(sets: Optional[Dict[str, List[str]]],
                    components: Dict[str, List[str]]
                    ) -> Callable[[str, str], bool]:
    """
    Whether a predictor is a language feature of a set, as opposed to a
    control.

    Coefficient tables list controls beside the features (``age``, the
    dummies of ``gender``); a cloud of predictors would show ``age`` as the
    biggest word, which is true and not what the picture is for. With the
    assemble sidecar a predictor counts when it is one of the set's columns
    or one of the set's PCA components; without a sidecar every predictor
    counts, since nothing says otherwise.
    """
    if not sets:
        return lambda feature_set, predictor: True
    every = {c for cols in sets.values() for c in cols}

    def allowed(feature_set: str, predictor: str) -> bool:
        cols = every if feature_set == "all" else set(sets.get(feature_set, ()))
        return predictor in cols or predictor in components.get(feature_set, ()) \
            or (feature_set == "all" and any(
                predictor in c for c in components.values()))
    return allowed


def _signed(words: Sequence[Word]) -> Tuple[List[Word], List[Word]]:
    pos = sorted([w for w in words if w[1] > 0], key=lambda w: -w[1])
    neg = sorted([w for w in words if w[1] < 0], key=lambda w: w[1])
    return pos, neg


def _pair(analysis: str, folder: str, fs: str, base: str, pos: List[Word],
          neg: List[Word], *, titles: Tuple[str, str], legends: Tuple[str, str],
          notes: Tuple[str, str], stat: str) -> List[CloudSpec]:
    """The positive and the negative cloud of one result, as two specs."""
    out = []
    for side, chosen, title, legend, note in (
            ("positive", pos, titles[0], legends[0], notes[0]),
            ("negative", neg, titles[1], legends[1], notes[1])):
        out.append(CloudSpec(
            analysis=analysis, folder=f"{folder}/{slug(fs)}", set_name=fs,
            name=f"{base}__{side}", title=title, legend=legend, words=chosen,
            note="" if chosen else note, stat=stat))
    return out


# ---------------------------------------------------------------------------
# choosing what to draw: pure functions over rows
# ---------------------------------------------------------------------------

def clouds_for_ridge(rows: Sequence[Row], allowed: Callable[[str, str], bool]
                     ) -> List[CloudSpec]:
    """
    Two clouds per outcome per feature set from ``ridge_coefficients.csv``:
    the predictors of a higher score and of a lower one, sized by their
    standardized coefficient in the full model.
    """
    if not rows:
        return []
    heading, folder = ANALYSES["ridge"]
    split = _split_col(rows)
    cols = list(rows[0].keys())
    outcomes = cols[cols.index("predictor") + 1:]
    out: List[CloudSpec] = []
    for (sv, fs), group in _grouped(rows, split or "", "feature_set").items():
        if "+" in fs:
            continue        # a combination of tables; its members get the clouds
        model_rows = _full_model(group)
        for outcome in outcomes:
            words = [(_label(r["predictor"], fs), _num(r.get(outcome)))
                     for r in model_rows if allowed(fs, r["predictor"])]
            pos, neg = _signed([(t, w) for t, w in words if w])
            where = _where(split, sv)
            out += _pair(
                heading, folder, fs, f"{slug(outcome)}{_tag(split, sv)}", pos, neg,
                titles=(f"Ridge: {outcome} — features predicting higher scores "
                        f"({fs}){where}",
                        f"Ridge: {outcome} — features predicting lower scores "
                        f"({fs}){where}"),
                legends=(f"Bigger and darker = larger standardized coefficient. "
                         f"Blue: predicts a higher {outcome}.",
                         f"Bigger and darker = larger standardized coefficient. "
                         f"Red: predicts a lower {outcome}."),
                notes=(f"no feature had a positive coefficient for {outcome} ({fs}).",
                       f"no feature had a negative coefficient for {outcome} ({fs})."),
                stat=f"{outcome}: β")
    return out


def clouds_for_classifier(rows: Sequence[Row], allowed: Callable[[str, str], bool]
                          ) -> List[CloudSpec]:
    """Two clouds per class per outcome per feature set from
    ``classifier_coefficients.csv``: what pushes a text toward the class,
    and what pushes it away."""
    if not rows:
        return []
    heading, folder = ANALYSES["classifier"]
    split = _split_col(rows)
    out: List[CloudSpec] = []
    keyed = _grouped(rows, split or "", "feature_set", "outcome", "class")
    for (sv, fs, outcome, klass), group in keyed.items():
        if "+" in fs:
            continue        # a combination of tables; its members get the clouds
        model_rows = _full_model(group)
        words = [(_label(r["predictor"], fs), _num(r.get("coef")))
                 for r in model_rows if allowed(fs, r["predictor"])]
        pos, neg = _signed([(t, w) for t, w in words if w])
        where = _where(split, sv)
        out += _pair(
            heading, folder, fs,
            f"{slug(outcome)}__{slug(klass)}{_tag(split, sv)}", pos, neg,
            titles=(f"Classifier: {outcome} = {klass} — features pushing toward "
                    f"this class ({fs}){where}",
                    f"Classifier: {outcome} = {klass} — features pushing away "
                    f"from this class ({fs}){where}"),
            legends=(f"Bigger and darker = larger standardized coefficient. "
                     f"Blue: pushes toward {klass}.",
                     f"Bigger and darker = larger standardized coefficient. "
                     f"Red: pushes away from {klass}."),
            notes=(f"no feature pushed toward {outcome} = {klass} ({fs}).",
                   f"no feature pushed away from {outcome} = {klass} ({fs})."),
            stat=f"{outcome} = {klass}: coefficient")
    return out


def _p_of(row: Row, outcome: str, max_p: float) -> bool:
    """Whether a correlation row passes the threshold, on the adjusted p when
    the table has one and the raw p otherwise."""
    for col in (f"{outcome}_p_adj", f"{outcome}_p"):
        if col in row:
            p = _num(row.get(col))
            return p is not None and p < max_p
    return False


def clouds_for_correlations(rows: Sequence[Row], *, max_p: float,
                            method: str = "") -> List[CloudSpec]:
    """Two clouds per outcome per feature set from a correlations table:
    features correlated positively and negatively, sized by r, only those
    under ``max_p`` (adjusted when the table has adjusted p-values)."""
    if not rows:
        return []
    heading, folder = ANALYSES["correlations"]
    split = _split_col(rows)
    cols = list(rows[0].keys())
    outcomes = [c[:-2] for c in cols
                if c.endswith("_r") and f"{c[:-2]}_p" in cols]
    adjusted = any(c.endswith("_p_adj") for c in cols)
    which = f"{method} " if method else ""
    adj = " (adjusted)" if adjusted else ""
    out: List[CloudSpec] = []
    for (sv, fs), group in _grouped(rows, split or "", "feature_set").items():
        for outcome in outcomes:
            words = [(_label(r["feature"], fs), _num(r.get(f"{outcome}_r")))
                     for r in group if _p_of(r, outcome, max_p)]
            pos, neg = _signed([(t, w) for t, w in words if w])
            where = _where(split, sv)
            legend = (f"Bigger and darker = larger |r|; only {which}correlations "
                      f"with p < {max_p:g}{adj} are shown.")
            out += _pair(
                heading, folder, fs, f"{slug(outcome)}{_tag(split, sv)}", pos, neg,
                titles=(f"Correlations: {outcome} — features correlated "
                        f"positively ({fs}){where}",
                        f"Correlations: {outcome} — features correlated "
                        f"negatively ({fs}){where}"),
                legends=(legend, legend),
                notes=(f"no feature was positively correlated with {outcome} at "
                       f"p < {max_p:g}{adj} ({fs}).",
                       f"no feature was negatively correlated with {outcome} at "
                       f"p < {max_p:g}{adj} ({fs})."),
                stat=f"{outcome}: r")
    return out


def clouds_for_group_differences(rows: Sequence[Row], *, max_p: float
                                 ) -> List[CloudSpec]:
    """Two clouds per pair of groups per feature set from
    ``group_differences_pairwise.csv``: what is higher in the first group
    and what is higher in the second, sized by Cohen's d."""
    if not rows:
        return []
    heading, folder = ANALYSES["group_differences"]
    split = _split_col(rows)
    cols = list(rows[0].keys())
    pcol = "p_adj" if "p_adj" in cols else "p"
    shown_p = pcol.replace("_", " ")
    out: List[CloudSpec] = []
    keyed = _grouped(rows, split or "", "feature_set", "group_1", "group_2")
    for (sv, fs, g1, g2), group in keyed.items():
        words = []
        for r in group:
            p = _num(r.get(pcol))
            d = _num(r.get("d"))
            if p is not None and p < max_p and d:
                words.append((_label(r["feature"], fs), d))
        pos, neg = _signed(words)
        base = f"{slug(g1)}_vs_{slug(g2)}{_tag(split, sv)}"
        where = _where(split, sv)
        for chosen, higher, other, color in ((pos, g1, g2, "Blue"),
                                              (neg, g2, g1, "Red")):
            out.append(CloudSpec(
                analysis=heading, folder=f"{folder}/{slug(fs)}", set_name=fs,
                name=f"{base}__higher_in_{slug(higher)}",
                title=f"Group differences: {g1} vs {g2} — higher in {higher} "
                      f"({fs}){where}",
                legend=f"Bigger and darker = larger Cohen's d; only pairs "
                       f"with {shown_p} < {max_p:g} are shown. "
                       f"{color}: higher in {higher}.",
                words=chosen,
                note="" if chosen else
                f"no feature was higher in {higher} than in {other} at "
                f"{shown_p} < {max_p:g} ({fs}).",
                stat=f"{g1} vs {g2}: d"))
    return out


def clouds_for_components(rows: Sequence[Row], *, analysis: str,
                          component_words: int) -> List[CloudSpec]:
    """One mixed-sign cloud per component from a ``*_pca_loadings*.csv``:
    the ``component_words`` features that load most strongly, blue for a
    positive loading and red for a negative one. Filed under the analysis
    and set they belong to, in a ``components/`` folder of their own."""
    if not rows:
        return []
    heading, folder = ANALYSES.get(
        analysis, (analysis.replace("_", " ").capitalize(), slug(analysis)))
    cols = list(rows[0].keys())
    components = cols[cols.index("feature") + 1:]
    out: List[CloudSpec] = []
    for (fs,), group in _grouped(rows, "feature_set").items():
        for comp in components:
            words = [(_label(r["feature"], fs), _num(r.get(comp))) for r in group]
            words = [(t, w) for t, w in words if w]
            words.sort(key=lambda tw: -abs(tw[1]))
            chosen = words[:max(0, int(component_words))]
            if not chosen:
                continue        # this set has fewer components than the widest one
            out.append(CloudSpec(
                analysis=heading, folder=f"{folder}/{slug(fs)}/components",
                set_name=fs, kind="component", name=slug(comp),
                title=f"{heading}: {comp} of {fs} — what loads on it",
                legend="Bigger and darker = larger loading. Blue loads "
                       "positively, red negatively.",
                words=chosen))
    return out


def clouds_for_theme_predictors(loadings: Sequence[Row],
                                mentions: Dict[str, List[Tuple[str, float]]], *,
                                analysis: str, folder: str, set_name: str,
                                top_words: int) -> List[CloudSpec]:
    """
    What each theme that appears in a set's clouds is made of.

    A ridge over topic-model themes draws "Theme_5" as a word, which says
    nothing until the theme's own words are in view. For every theme named
    in ``mentions`` (theme -> where it appeared, as (statistic, value) pairs
    such as ``("TIPI_Open: β", 0.32)``) that has a column in the loadings
    table, one mixed-sign cloud of its ``top_words`` strongest terms, filed
    under the set's ``themes/`` folder, its legend naming the three places
    it mattered most.
    """
    if not loadings or not mentions:
        return []
    out: List[CloudSpec] = []
    for theme in sorted(mentions, key=lambda t: (len(t), t)):
        # a name with no column in the loadings (a control, a component)
        # just gathers no words below, and we skip it there.
        words = []
        for r in loadings:
            v = _num(r.get(theme))
            if not v:
                continue
            label = r.get("term", "")
            if r.get("pos"):
                label = f"{label} ({r['pos']})"
            words.append((label, v))
        words.sort(key=lambda tw: -abs(tw[1]))
        chosen = words[:max(0, int(top_words))]
        if not chosen:
            continue
        seen = sorted(mentions[theme], key=lambda sv: -abs(sv[1]))
        said = "; ".join(f"{stat} {v:+.2f}" for stat, v in seen[:3])
        out.append(CloudSpec(
            analysis=analysis, folder=f"{folder}/{slug(set_name)}/themes",
            set_name=set_name, kind="theme", name=slug(theme),
            title=f"{theme.replace('_', ' ')} — what it is made of ({set_name})",
            legend="Bigger and darker = larger loading; blue loads positively, "
                   "red negatively. Where it mattered most here: " + said
                   + (f" (and {len(seen) - 3} more)" if len(seen) > 3 else "") + ".",
            words=chosen))
    return out


def clouds_for_frequencies_by_group(table_rows: Sequence[Row],
                                    sets: Dict[str, List[str]], *,
                                    group_col: str, max_words: int
                                    ) -> List[CloudSpec]:
    """
    One cloud per group level per document-term set: the terms used most in
    that group's texts, sized by their summed count in the analysis table.

    The analysis table already joins the matrix and the grouping column, so
    this is a column sum per level and nothing is re-tokenized.
    """
    if not table_rows or not group_col or group_col not in table_rows[0]:
        return []
    heading, folder = _FREQ_BY_GROUP
    out: List[CloudSpec] = []
    for fs, cols in sets.items():
        if not fs.startswith("doc_term_matrix"):
            continue
        totals: Dict[str, Dict[str, float]] = {}
        for r in table_rows:
            level = str(r.get(group_col) or "").strip()
            if not level:
                continue
            bucket = totals.setdefault(level, {})
            for c in cols:
                v = _num(r.get(c))
                if v:
                    bucket[c] = bucket.get(c, 0.0) + v
        for level in sorted(totals):
            words = sorted(((_label(c, fs), v) for c, v in totals[level].items()),
                           key=lambda tw: -tw[1])[:max(0, int(max_words))]
            if not words:
                continue
            out.append(CloudSpec(
                analysis=heading, folder=f"{folder}/{slug(fs)}", set_name=fs,
                name=f"{slug(group_col)}-{slug(level)}",
                title=f"Most frequent terms — {group_col} = {level} ({fs})",
                legend=f"Bigger and darker = used more in the texts of "
                       f"{group_col} = {level}.",
                words=words))
    return out


def clouds_for_themes(rows: Sequence[Row], *, top_words: int,
                      min_abs_loading: float = 0.0,
                      shares: Optional[Dict[str, float]] = None) -> List[CloudSpec]:
    """One mixed-sign cloud per theme from a topic model's loadings table
    (``term, [pos,] Theme_1…``). A ``pos`` column joins the label, so
    ``felt (VBD)`` reads as the tagged term it is."""
    if not rows:
        return []
    cols = list(rows[0].keys())
    themes = [c for c in cols if c not in ("term", "pos")]
    out: List[CloudSpec] = []
    for theme in themes:
        words = []
        for r in rows:
            v = _num(r.get(theme))
            if v is None or abs(v) < min_abs_loading or v == 0:
                continue
            label = r.get("term", "")
            if r.get("pos"):
                label = f"{label} ({r['pos']})"
            words.append((label, v))
        words.sort(key=lambda tw: -abs(tw[1]))
        chosen = words[:max(0, int(top_words))]
        share = (shares or {}).get(theme)
        out.append(CloudSpec(
            analysis="Themes",
            name=slug(theme),
            title=f"{theme.replace('_', ' ')} — the words that define it"
                  + (f" ({share:.1f}% of variance)" if share is not None else ""),
            legend="Bigger and darker = larger loading. Blue loads "
                   "positively, red negatively.",
            words=chosen,
            note="" if chosen else f"{theme} has no term loading above "
                                   f"{min_abs_loading:g}."))
    return out


def skip_redundant(phrases: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Drop an n-gram that only repeats words the cloud already shows.

    Walking down by frequency, a phrase every one of whose words is already
    in the cloud on its own -- "of the" beside "of" and "the" -- says nothing
    new and is skipped; a phrase that brings at least one new word stays
    ("going to be" when "going" is not there yet). Single words are always
    kept. A frequency-based test was tried first and kept "of the" because
    "of" is rarer than "the", which is true and not the point.
    """
    kept: List[Tuple[str, float]] = []
    shown: set = set()
    for text, weight in sorted(phrases, key=lambda tw: -tw[1]):
        words = text.split()
        if len(words) > 1 and all(w in shown for w in words):
            continue
        kept.append((text, weight))
        shown.update(words)
    return kept


def clouds_for_frequencies(rows: Sequence[Row], *, top_words: int) -> List[CloudSpec]:
    """The corpus cloud from a frequency list: the ``top_words`` most
    frequent n-grams after :func:`skip_redundant`, sized by frequency."""
    if not rows:
        return []
    phrases = []
    for r in rows:
        f = _num(r.get("frequency"))
        text = (r.get("ngram") or "").strip()
        if not text or not f:
            continue
        phrases.append((text, f))
    chosen = skip_redundant(phrases)[:max(0, int(top_words))]
    if rows and rows[0].get("pos") is not None:
        tags = {r.get("ngram", ""): r.get("pos", "") for r in rows}
        chosen = [(f"{t} ({tags[t]})" if tags.get(t) else t, f) for t, f in chosen]
    return [CloudSpec(
        analysis="Most frequent terms",
        name="top_words",
        title=f"The {len(chosen)} most frequent terms in the corpus",
        legend="Bigger and darker = more frequent. Phrases that only repeat "
               "a more frequent word are left out.",
        words=chosen,
        note="" if chosen else "the frequency list is empty.")]


def clouds_for_neighbors(rows: Sequence[Row], *, top_words: int) -> List[CloudSpec]:
    """One cloud per probe from a word-vector model's neighbors table
    (``probe, rank, word, similarity``), sized by similarity. A probe the
    model does not know has a row with blanks, and gets a sentence."""
    if not rows:
        return []
    by_probe: Dict[str, List[Word]] = {}
    for r in rows:
        probe = (r.get("probe") or "").strip()
        if not probe:
            continue
        by_probe.setdefault(probe, [])
        sim = _num(r.get("similarity"))
        word = (r.get("word") or "").strip()
        if word and sim is not None:
            by_probe[probe].append((word, sim))
    out: List[CloudSpec] = []
    for probe, words in by_probe.items():
        words.sort(key=lambda tw: -tw[1])
        chosen = [(w, s) for w, s in words if s > 0][:max(0, int(top_words))]
        out.append(CloudSpec(
            analysis="Nearest neighbors",
            name=slug(probe),
            title=f"Words closest to “{probe}” in the model",
            legend="Bigger and darker = more similar (cosine).",
            words=chosen,
            note="" if chosen else f"“{probe}” is not in the model's vocabulary."))
    return out


# ---------------------------------------------------------------------------
# drawing and reporting
# ---------------------------------------------------------------------------

def _draw_all(specs: Sequence[CloudSpec], root: Path, *, max_words: int,
              overwrite_existing: bool, verbose: bool, on_progress
              ) -> Tuple[int, int]:
    """Render every spec with words, unless its picture is newer than the
    tables it comes from. Returns (drawn, reused)."""
    drawn = reused = 0
    todo = [s for s in specs if s.words]
    for i, spec in enumerate(todo):
        png = root / spec.folder / f"{spec.name}.png"
        if reusable(png, *spec.inputs, overwrite_existing=overwrite_existing,
                    verbose=verbose, what=f"the word cloud {png.name}"):
            reused += 1
            continue
        if on_progress is not None:
            on_progress(i, len(todo), f"drawing {png.name}")
        render_wordcloud(spec.words, png, title=spec.title, legend=spec.legend,
                         max_words=max_words, verbose=verbose)
        drawn += 1
    return drawn, reused


def _write_section_if_changed(stats_dir: Path, markdown: str) -> bool:
    """The report fragment, through the registered writer -- but only when
    its text changed, so an unchanged re-run leaves its mtime alone and the
    report is not rebuilt for nothing."""
    from ..stats._common import SECTION_ORDER
    path = stats_dir / SECTIONS_DIR / f"{SECTION_ORDER[SECTION_SLUG]:02d}-{SECTION_SLUG}.md"
    text = markdown.rstrip() + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    write_section(stats_dir, SECTION_SLUG, markdown)
    return True


def _write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` unless the file already says exactly that. Keeps the
    mtime of an unchanged file, which is what lets a re-run write nothing."""
    text = text.rstrip() + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    from ..helpers.atomic import atomic_write
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write(text)
    return True


def _picture(s: CloudSpec) -> List[str]:
    return [f"**{s.title}**  ",
            f"![{s.title}]({(FIGURES_DIR / s.relative_png).as_posix()})", ""]


def _section_md(specs: Sequence[CloudSpec], skipped: str = "") -> str:
    """The report fragment: one heading per analysis, one sub-heading per
    feature set, a caption and the picture for every cloud, then what the
    set's components and themes are made of, and a sentence for every cloud
    there was nothing to draw."""
    lines = ["## Word clouds", "",
             "Pictures of the results above, under `figures/wordclouds/`, one "
             "folder per analysis and one per feature set inside it. In every "
             "cloud the size and shade of a word follow the size of its "
             "statistic, blue is a positive value and red a negative one, and "
             "the table the cloud is drawn from is the record -- the picture "
             "is a way to read it quickly, not a result of its own."]
    if skipped:
        lines += ["", f"Figures skipped: {skipped}"]
        return "\n".join(lines)
    by_analysis: Dict[str, Dict[str, List[CloudSpec]]] = {}
    for s in specs:
        by_analysis.setdefault(s.analysis, {}).setdefault(s.set_name, []).append(s)
    for analysis, by_set in by_analysis.items():
        lines += ["", f"### {analysis}"]
        for set_name, group in by_set.items():
            if set_name:
                lines += ["", f"#### {set_name}", ""]
            else:
                lines += [""]
            for s in group:
                if s.words and s.kind == "cloud":
                    lines += _picture(s)
            components = [s for s in group if s.words and s.kind == "component"]
            if components:
                lines += [f"What each component of {set_name} is made of:", ""]
                for s in components:
                    lines += _picture(s)
            themes = [s for s in group if s.words and s.kind == "theme"]
            if themes:
                lines += ["What the themes that appear above are made of "
                          "(every theme, not only these, is drawn under "
                          "`features/figures/wordclouds/`):", ""]
                for s in themes:
                    lines += _picture(s)
            notes = [s.note for s in group if not s.words and s.note]
            if notes:
                lines += ["Nothing to draw: " + " ".join(notes), ""]
    return "\n".join(lines)


def _sidecar(table_csv: Path) -> Tuple[Optional[Dict[str, List[str]]], Dict[str, str]]:
    """The assemble step's sets and the source file of each, when written."""
    sidecar = table_csv.with_name(table_csv.stem + "_sets.json")
    if not sidecar.is_file():
        return None, {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sets = {str(k): [str(c) for c in v] for k, v in (payload.get("sets") or {}).items()}
    sources = {str(k): str(v) for k, v in (payload.get("sources") or {}).items()}
    return sets, sources


def _components_by_set(stats_dir: Path, encoding: str) -> Dict[str, List[str]]:
    """Component names per feature set, from every PCA eigenvalues table, so
    a coefficient on ``dictionary_Component_1`` counts as a feature."""
    out: Dict[str, List[str]] = {}
    for path in sorted(stats_dir.glob("*_pca_eigenvalues.csv")):
        for r in _rows(path, encoding):
            out.setdefault(r.get("feature_set", ""), []).append(r.get("component", ""))
    return out


def _theme_loadings_for(set_name: str, cols: Sequence[str], sources: Dict[str, str],
                        stats_dir: Path, encoding: str) -> Optional[Path]:
    """
    The loadings table that says what this set's features are made of, when
    the set is a model's themes.

    The topic model writes ``<stem>_loadings.csv`` beside its features CSV,
    whose path the assemble sidecar records. That path is taken as written
    first, then looked for under the pipeline's folder -- a sidecar written
    on Windows and read from WSL records a path that does not resolve, and
    the file is right there under ``features/``. A candidate counts when it
    has a ``term`` column and a column for one of the set's features.
    """
    candidates: List[Path] = []
    src = sources.get(set_name)
    if src:
        p = Path(src.replace("\\", "/"))
        candidates.append(p.with_name(p.stem + "_loadings.csv"))
    candidates += sorted(stats_dir.parent.rglob(f"{set_name}_loadings.csv"))
    wanted = {_label(c, set_name) for c in cols}
    for path in candidates:
        try:
            if not path.is_file():
                continue
            with path.open("r", encoding=encoding, newline="") as fh:
                header = next(csv.reader(fh), [])
        except OSError:
            continue
        if "term" in header and wanted & set(header):
            return path
    return None


def stats_wordclouds(
    stats_dir: PathLike = "stats_results",
    *,
    max_words: int = 80,
    max_p: float = 0.05,
    component_words: int = 30,
    group_col: str = "",
    enabled: bool = True,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Draw word clouds of every statistics result in a folder, and a report
    section showing them.

    Parameters
    ----------
    stats_dir : str or Path, default "stats_results"
        The statistics output folder. Whatever tables it holds are drawn:
        ridge and classifier coefficients, correlations, pairwise group
        differences, PCA loadings, and per-group term frequencies when the
        analysis table joins a document-term matrix and ``group_col`` names
        the grouping column.
    max_words : int, default 80
        The most words in any one cloud.
    max_p : float, default 0.05
        Correlations and pairwise differences are drawn only when they pass
        this, on the adjusted p-value where the table has one.
    component_words : int, default 30
        How many features to show for each principal component, and how
        many terms for each topic-model theme a set's clouds name.
    group_col : str, default ""
        The metadata column whose levels get a frequency cloud each. Empty
        draws none.
    enabled : bool, default True
        Off draws nothing and writes nothing -- the switch behind
        ``wordclouds: false`` in a pipeline.
    overwrite_existing : bool, default False
        Redraw pictures that are newer than their tables anyway.
    encoding : str, default "utf-8-sig"
        The tables' encoding.
    verbose, on_progress
        The usual.

    Returns
    -------
    Path
        ``<stats_dir>/figures/wordclouds``. Under it, one folder per analysis
        and one per feature set inside it -- ``ridge-regression/dictionary/``
        -- with ``components/`` and ``themes/`` folders for what that set's
        predictors are made of.

    Notes
    -----
    Coefficient clouds come from one model per outcome: language fitted
    with every control when there were controls, language alone otherwise.
    Controls themselves are not drawn -- ``age`` as the biggest word in a
    cloud about language is true and beside the point. A cloud that would be
    empty is not drawn; the report says so instead.
    """
    stats_dir = Path(stats_dir)
    root = stats_dir / FIGURES_DIR
    if not enabled:
        if verbose:
            print("[figures] word clouds are off for this run.")
        return root

    missing = pillow_missing_reason()
    if missing:
        if verbose:
            print(f"[figures] {missing}")
        _write_section_if_changed(stats_dir, _section_md([], skipped=missing))
        return root

    announce(on_progress, "choosing the words for each cloud")
    table_csv = stats_dir / "analysis_table.csv"
    sets, sources = _sidecar(table_csv) if table_csv.is_file() else (None, {})
    components = _components_by_set(stats_dir, encoding)
    allowed = _feature_filter(sets, components)

    specs: List[CloudSpec] = []

    def take(path: Path, made: List[CloudSpec]) -> None:
        for s in made:
            s.inputs = [path]
        specs.extend(made)

    p = stats_dir / "group_differences_pairwise.csv"
    if p.is_file():
        take(p, clouds_for_group_differences(_rows(p, encoding), max_p=max_p))
    for method in ("pearson", "spearman"):
        p = stats_dir / f"correlations_{method}.csv"
        if p.is_file():
            take(p, clouds_for_correlations(_rows(p, encoding), max_p=max_p,
                                            method=method))
    p = stats_dir / "ridge_coefficients.csv"
    if p.is_file():
        take(p, clouds_for_ridge(_rows(p, encoding), allowed))
    p = stats_dir / "classifier_coefficients.csv"
    if p.is_file():
        take(p, clouds_for_classifier(_rows(p, encoding), allowed))
    for p in sorted(stats_dir.glob("*_pca_loadings*.csv")):
        analysis = p.name.split("_pca_loadings")[0]
        take(p, clouds_for_components(_rows(p, encoding), analysis=analysis,
                                      component_words=component_words))
    if sets and group_col and table_csv.is_file():
        take(table_csv, clouds_for_frequencies_by_group(
            _rows(table_csv, encoding), sets, group_col=group_col,
            max_words=max_words))

    # lastly, what the themes a set's clouds name are made of, filed next to
    # them.
    if sets:
        mentions: Dict[Tuple[str, str, str], Dict[str, List[Tuple[str, float]]]] = {}
        for s in specs:
            if s.kind != "cloud" or not s.words or not s.set_name:
                continue
            folder_root = s.folder.rsplit("/", 1)[0]
            bucket = mentions.setdefault((s.analysis, folder_root, s.set_name), {})
            for label, w in s.words:
                bucket.setdefault(label, []).append((s.stat, w))
        loadings_cache: Dict[str, Optional[Path]] = {}
        for (analysis, folder_root, fs), named in mentions.items():
            if fs not in loadings_cache:
                loadings_cache[fs] = _theme_loadings_for(
                    fs, sets.get(fs, []), sources, stats_dir, encoding)
            path = loadings_cache[fs]
            if path is None:
                continue
            made = clouds_for_theme_predictors(
                _rows(path, encoding), named, analysis=analysis,
                folder=folder_root, set_name=fs, top_words=component_words)
            for s in made:
                s.inputs = [path] + [t for t in (stats_dir / "analysis_table.csv",)
                                     if t.is_file()]
            specs.extend(made)

    drawn, reused = _draw_all(specs, root, max_words=max_words,
                              overwrite_existing=overwrite_existing,
                              verbose=verbose, on_progress=on_progress)
    if specs:
        _write_section_if_changed(stats_dir, _section_md(specs))
    if verbose:
        print(f"[figures] {drawn} word cloud(s) drawn, {reused} already current"
              f" -> {root}" if specs else
              "[figures] no statistics tables to draw word clouds from.")
    return root


def theme_wordclouds(
    loadings_csv: PathLike,
    out_dir: Optional[PathLike] = None,
    *,
    top_words: int = 30,
    min_abs_loading: float = 0.0,
    enabled: bool = True,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    One word cloud per theme of a topic model, from its loadings table.

    Parameters
    ----------
    loadings_csv : str or Path
        The ``*_loadings.csv`` the topic-model step writes: ``term``, an
        optional ``pos``, then one column per theme.
    out_dir : str or Path, optional
        Where the pictures go. Defaults to
        ``<features folder>/figures/wordclouds/topic_model_mem``.
    top_words : int, default 30
        How many terms per theme, by the size of their loading.
    min_abs_loading : float, default 0.0
        Leave out terms loading less than this in either direction.
    enabled, overwrite_existing, encoding, verbose, on_progress
        As for :func:`stats_wordclouds`.

    Returns
    -------
    Path
        The output folder; it also holds an ``index.md`` listing the themes.
    """
    loadings_csv = Path(loadings_csv)
    folder = Path(out_dir) if out_dir else (
        loadings_csv.parent / FIGURES_DIR / "topic_model_mem")
    if not enabled:
        if verbose:
            print("[figures] theme word clouds are off for this run.")
        return folder
    missing = pillow_missing_reason()
    if missing:
        if verbose:
            print(f"[figures] {missing}")
        return folder

    shares: Dict[str, float] = {}
    eigen = loadings_csv.with_name(
        loadings_csv.name.replace("_loadings", "_eigenvalues"))
    if eigen != loadings_csv and eigen.is_file():
        for r in _rows(eigen, encoding):
            v = _num(r.get("pct_variance"))
            if v is not None:
                shares[r.get("theme", "")] = v
    specs = clouds_for_themes(_rows(loadings_csv, encoding), top_words=top_words,
                              min_abs_loading=min_abs_loading, shares=shares)
    for s in specs:
        s.inputs = [loadings_csv]
    drawn, reused = _draw_all(specs, folder, max_words=top_words,
                              overwrite_existing=overwrite_existing,
                              verbose=verbose, on_progress=on_progress)
    lines = ["# Themes", "", f"Drawn from `{loadings_csv.name}`: the {top_words} "
             "terms loading most strongly on each theme. Blue loads positively, "
             "red negatively; bigger and darker is a larger loading.", ""]
    for s in specs:
        if s.words:
            lines += [f"**{s.title}**  ", f"![{s.title}]({s.name}.png)", ""]
        elif s.note:
            lines += [f"{s.note}", ""]
    _write_if_changed(folder / "index.md", "\n".join(lines))
    if verbose:
        print(f"[figures] {drawn} theme cloud(s) drawn, {reused} already current "
              f"-> {folder}")
    return folder


def frequency_wordclouds(
    freq_csv: PathLike,
    out_dir: Optional[PathLike] = None,
    *,
    top_words: int = 100,
    enabled: bool = True,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    The corpus's most frequent words and phrases as one cloud.

    Parameters
    ----------
    freq_csv : str or Path
        The n-gram frequency list (``ngram``, ``frequency``, …).
    out_dir : str or Path, optional
        Where the picture goes. Defaults to
        ``<features folder>/figures/wordclouds/ngram_frequencies``.
    top_words : int, default 100
        How many terms, by frequency, after phrases that only repeat a more
        frequent word are left out.
    enabled, overwrite_existing, encoding, verbose, on_progress
        As for :func:`stats_wordclouds`.

    Returns
    -------
    Path
        The output folder holding ``top_words.png``.
    """
    freq_csv = Path(freq_csv)
    folder = Path(out_dir) if out_dir else (
        freq_csv.parent / FIGURES_DIR / "ngram_frequencies")
    if not enabled:
        if verbose:
            print("[figures] the frequency word cloud is off for this run.")
        return folder
    missing = pillow_missing_reason()
    if missing:
        if verbose:
            print(f"[figures] {missing}")
        return folder
    specs = clouds_for_frequencies(_rows(freq_csv, encoding), top_words=top_words)
    for s in specs:
        s.inputs = [freq_csv]
    drawn, reused = _draw_all(specs, folder, max_words=top_words,
                              overwrite_existing=overwrite_existing,
                              verbose=verbose, on_progress=on_progress)
    if verbose:
        print(f"[figures] {drawn} frequency cloud(s) drawn, {reused} already "
              f"current -> {folder}")
    return folder


def neighbor_wordclouds(
    neighbors_csv: PathLike,
    out_dir: Optional[PathLike] = None,
    *,
    top_words: int = 20,
    enabled: bool = True,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    One word cloud per probe of a word-vector model's neighbors table.

    Parameters
    ----------
    neighbors_csv : str or Path
        The ``*_neighbors.csv`` the word-vector steps write:
        ``probe, rank, word, similarity``.
    out_dir : str or Path, optional
        Where the pictures go. Defaults to
        ``<features folder>/figures/wordclouds/word_vectors``.
    top_words : int, default 20
        How many neighbors per probe, by similarity.
    enabled, overwrite_existing, encoding, verbose, on_progress
        As for :func:`stats_wordclouds`.

    Returns
    -------
    Path
        The output folder; it also holds an ``index.md`` listing the probes.
    """
    neighbors_csv = Path(neighbors_csv)
    folder = Path(out_dir) if out_dir else (
        neighbors_csv.parent.parent / FIGURES_DIR / "word_vectors")
    if not enabled:
        if verbose:
            print("[figures] neighbor word clouds are off for this run.")
        return folder
    missing = pillow_missing_reason()
    if missing:
        if verbose:
            print(f"[figures] {missing}")
        return folder
    specs = clouds_for_neighbors(_rows(neighbors_csv, encoding), top_words=top_words)
    for s in specs:
        s.inputs = [neighbors_csv]
    drawn, reused = _draw_all(specs, folder, max_words=top_words,
                              overwrite_existing=overwrite_existing,
                              verbose=verbose, on_progress=on_progress)
    lines = ["# Nearest neighbors", "",
             f"Drawn from `{neighbors_csv.name}`: the {top_words} words closest "
             "to each probe by cosine similarity. Bigger and darker is more "
             "similar. These are the evidence that the model learned what the "
             "study assumes it learned.", ""]
    for s in specs:
        if s.words:
            lines += [f"**{s.title}**  ", f"![{s.title}]({s.name}.png)", ""]
        elif s.note:
            lines += [f"{s.note}", ""]
    _write_if_changed(folder / "index.md", "\n".join(lines))
    if verbose:
        print(f"[figures] {drawn} neighbor cloud(s) drawn, {reused} already "
              f"current -> {folder}")
    return folder


# ---------------------------------------------------------------------------
# command line -- we derive it from the functions above; see helpers.cliargs.CliSpec.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"stats": stats_wordclouds, "themes": theme_wordclouds,
     "frequencies": frequency_wordclouds, "neighbors": neighbor_wordclouds},
    description="Word clouds of finished results: a statistics folder, a "
                "topic model's themes, a frequency list, or a word-vector "
                "model's nearest neighbors.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
