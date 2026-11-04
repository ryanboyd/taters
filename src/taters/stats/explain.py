"""
What a prediction model tracks, in words.

A ridge or a classifier fitted on embeddings is drawn as a coefficient cloud
of ``e238``, ``e10``, ... which says nothing. This step explains such a
model in words two ways, and writes each as a table first (the record) so
that a cloud drawn from it afterwards is a way to read the table, never a
result of its own:

* **Word correlates.** The n-grams whose relative frequency across texts
  correlates with the model's held-out predictions -- what the scores
  track in language -- and with its residuals: what the outcome has that
  the model missed. Works for any model on any feature set.
* **Token attributions.** The exact share of every token in a prediction
  made over a feature set that is a sum over its tokens (:mod:`attribution`:
  mean-pooled embeddings, in context; word vectors, word by word), summed
  per word across the corpus and written out token by token for the
  highest and lowest scored texts. Exact with or without PCA in front; a
  set that cannot be decomposed that way is said so and gets the
  correlates only. A word-vector model also gets its whole vocabulary
  ranked by the direction each word would push the prediction.

Which sets are explained is ``which``: ``auto`` explains the sets whose
columns are not words (embeddings, model scores, components of those),
``all`` explains every set, ``off`` none. Every number a cloud is drawn from
is in the tables under the results folder -- a CSV on a folder run, a table
in the database on a database run -- and ``explain_manifest.json`` names
them for the figure step.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..helpers import tables
from ..helpers.cliargs import CliSpec
from ..helpers.model_spec import slug
from ..helpers.progress import FlightReporter, announce, count_rows
from ._common import (SECTION_ORDER, adjust_pvalues, analysis_table_of, leftover,
                      opaque_sets, opaque_share, reusable, set_members, sets_sidecar,
                      write_section_if_changed)

__all__ = ["explain_predictions", "Target", "targets_in", "word_correlates",
           "read_manifest", "EXPLAIN_MANIFEST", "SECTION_SLUG"]

PathLike = Union[str, Path]
Row = Dict[str, str]

EXPLAIN_MANIFEST = "explain_manifest.json"
#: Fewer texts than this with a held-out value, and a target gets no
#: correlates: the p-values would be real and the correlations meaningless.
MIN_TEXTS = 10
SECTION_SLUG = "prediction-words"
assert SECTION_SLUG in SECTION_ORDER

#: The model files and predictions tables of each analysis, by their stems.
_ANALYSES = {"ridge": ("ridge__", "ridge_predictions__"),
             "classifier": ("classifier__", "classifier_predictions__")}


# ---------------------------------------------------------------------------
# What there is to explain: one continuous target per model and kind
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """
    One column of numbers a model made, keyed by text: the thing the words
    get correlated with.

    ``kind`` is ``prediction`` (the held-out prediction), ``residual``
    (observed minus held-out prediction: what the model missed) or
    ``p_<class>`` (a classifier's held-out probability of that class).
    ``model`` is the model file it belongs to, which is also where the
    feature set, the subset and the predictors are read from.
    """

    analysis: str
    feature_set: str
    subset: str
    outcome: str
    kind: str
    values: Dict[str, float]
    model: Path
    opaque: bool = False
    #: the kept model's own (in-sample) prediction per text, for the ridge:
    #: what an attribution explains, since it explains the kept model
    fitted: Dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.outcome} ({self.kind})"


def _num(value: Any) -> Optional[float]:
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


def _model_files(stats_dir: Path, analysis: str) -> List[Path]:
    prefix = _ANALYSES[analysis][0]
    return sorted((stats_dir / "models").glob(f"{prefix}*.json"))


def _read_model(path: Path) -> Optional[dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("outcomes") else None


def _predictions_for(stats_dir: Path, analysis: str, model: Path, outcome: str) -> Optional[Path]:
    """The predictions table a model file's rows came from: the two share a
    stem, ``<set>[__<subset>]``."""
    model_prefix, pred_prefix = _ANALYSES[analysis]
    stem = model.stem[len(model_prefix):]
    tail = f"__{slug(outcome)}"
    if stem.endswith(tail):
        stem = stem[:-len(tail)]
    path = stats_dir / f"{pred_prefix}{stem}.csv"
    return path if path.is_file() else None


def targets_in(stats_dir: PathLike, *, encoding: str = "utf-8-sig",
               opaque: Optional[Dict[str, bool]] = None) -> List[Target]:
    """
    Every explainable column the fitters left in a results folder.

    For each model file: the held-out prediction of its outcome and the
    residual against the observed value (a ridge), or the held-out
    probability of each class (a classifier). Rows without a held-out
    value -- the fit could not use them -- are left out of the target.
    """
    stats_dir = Path(stats_dir)
    opaque = opaque or {}
    out: List[Target] = []
    for analysis in _ANALYSES:
        for model_path in _model_files(stats_dir, analysis):
            # the models of a set fitted in an earlier run sit beside
            # today's; a model older than the analysis table is theirs
            if leftover(model_path, stats_dir):
                continue
            doc = _read_model(model_path)
            if doc is None:
                continue
            (outcome, block), = doc["outcomes"].items()
            preds = _predictions_for(stats_dir, analysis, model_path, outcome)
            if preds is None:
                continue
            rows = list(tables.read_rows(preds, encoding=encoding))
            if not rows:
                continue
            key = "text_id" if "text_id" in rows[0] else list(rows[0])[0]
            fs = str(doc.get("feature_set") or "")
            subset = str(doc.get("subset") or "")
            # a set is one table, several, or all of them; one member a
            # reader cannot read is enough to want the words
            common = dict(analysis=analysis, feature_set=fs, subset=subset,
                          outcome=outcome, model=model_path,
                          opaque=opaque_share(fs, opaque)[0] > 0)
            if analysis == "ridge":
                col = f"oof_pred_{outcome}"
                if col not in rows[0]:
                    continue
                pred = {r[key]: v for r in rows if (v := _num(r.get(col))) is not None}
                fitted = {r[key]: v for r in rows
                          if (v := _num(r.get(f"pred_{outcome}"))) is not None}
                resid = {}
                for r in rows:
                    p, y = _num(r.get(col)), _num(r.get(outcome))
                    if p is not None and y is not None:
                        resid[r[key]] = y - p
                if pred:
                    out.append(Target(kind="prediction", values=pred, fitted=fitted, **common))
                if resid:
                    out.append(Target(kind="residual", values=resid, **common))
            else:
                for klass in block.get("classes") or []:
                    col = f"oof_p_{outcome}_{klass}"
                    if col not in rows[0]:
                        continue
                    vals = {r[key]: v for r in rows if (v := _num(r.get(col))) is not None}
                    if vals:
                        out.append(Target(kind=f"p_{klass}", values=vals, **common))
    return out


# ---------------------------------------------------------------------------
# The correlates: one streaming pass, vocabulary-sized memory
# ---------------------------------------------------------------------------

@dataclass
class _Sums:
    """Running sums for one n-gram, per target: how many of that target's
    texts it is in, Σx, Σx² and Σxy. x is the n-gram's relative frequency
    in a text, zero where it is absent, so the sums over texts it appears
    in are the sums over every text. Per target, because a text with no
    held-out value for a target is not in that target's sample at all --
    summing its x in and not its y gave correlations of exactly ±1."""

    docs: List[int] = field(default_factory=list)
    sx: List[float] = field(default_factory=list)
    sxx: List[float] = field(default_factory=list)
    sxy: List[float] = field(default_factory=list)


def _pearson(n: int, sx: float, sxx: float, sy: float, syy: float, sxy: float
             ) -> Tuple[float, float]:
    """r and its two-sided p from sums alone; (nan, nan) when undefined."""
    if n < 3:
        return float("nan"), float("nan")
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if vx <= 0 or vy <= 0:
        return float("nan"), float("nan")
    r = (n * sxy - sx * sy) / math.sqrt(vx * vy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0
    from scipy.stats import t as _t

    stat = r * math.sqrt((n - 2) / (1.0 - r * r))
    return r, float(2.0 * _t.sf(abs(stat), n - 2))


def _grams_with_id_in_worker(pair: Tuple[str, str]):
    """The pool's half of the pass: the text's n-gram counts, with its id
    riding along so the parent can look its targets up. Module-level
    because a worker function is pickled to the pool; the closure this used
    to be took down a real run with "Can't get local object"."""
    from ..text.analyze_ngram_frequencies import _grams_in_worker

    return pair[0], _grams_in_worker(pair)


def word_correlates(texts: PathLike, targets: Sequence[Target], *,
                    ngram_n: int = 1, min_docs: int = 10, min_doc_pct: float = 1.0,
                    p_adjust: str = "fdr_bh", lemmatize: bool = False,
                    keep_punctuation: bool = False, pos_tagged: bool = False,
                    engine: str = "nltk", tokenizer: str = "potts",
                    stanza_lang: str = "en", device: str = "auto",
                    stoplist_paths: Sequence[PathLike] = (), workers: int = 0,
                    encoding: str = "utf-8-sig", rounding: int = 6,
                    on_progress: Optional[Callable[..., None]] = None,
                    ) -> Tuple[List[dict], Dict[str, int]]:
    """
    Correlate every n-gram's relative frequency with every target, in one
    pass over the texts.

    The tokens are the n-gram steps' own (same tokenizer, same settings),
    so a word here is the word the frequency list counts. Memory is one
    small record per distinct n-gram; no document-term matrix is built.
    Returns the rows (one per n-gram per target that passes the document
    floor) and a count of texts used and skipped.
    """
    from ..helpers.parallel_map import ordered_parallel_map
    from ..text.analyze_ngram_frequencies import (_counting_workers, _doc_grams,
                                                  _init_gram_worker)
    from ..text.ngram_prep import load_stoplist, make_token_stream, stoplisted

    if not targets:
        return [], {"texts": 0, "skipped": 0}
    texts_ref = tables.parse_ref(texts)
    fields = tables.columns_of(texts_ref, encoding=encoding)
    if "text_id" not in fields or "text" not in fields:
        raise ValueError(f"Expected columns 'text_id' and 'text' in {texts_ref}; found {fields}")
    total_rows = count_rows(texts_ref, on_progress=on_progress)
    stream_args = dict(lemmatize=lemmatize, pos_tagged=pos_tagged, engine=engine,
                       tokenizer=tokenizer, stanza_lang=stanza_lang, device=device,
                       keep_punctuation=keep_punctuation)
    stream = make_token_stream(**stream_args)
    n_workers = _counting_workers(workers, total_rows, engine=engine, tokenizer=tokenizer,
                                  lemmatize=lemmatize, pos_tagged=pos_tagged)
    stopset = load_stoplist(stoplist_paths) if stoplist_paths else set()
    reporter = FlightReporter(on_progress, total_rows, "reading the texts for their words")

    # the targets as columns of one array per text, so a text pays one dict
    # lookup per target and the sums are plain arithmetic
    k = len(targets)
    thin: List[Tuple[Target, int]] = []
    sums: Dict[str, _Sums] = {}
    n_used = 0
    skipped = 0
    sy = [0.0] * k
    syy = [0.0] * k
    n_per = [0] * k           # texts with a value for each target

    def pairs():
        for row in tables.read_rows(texts_ref, encoding=encoding, columns=["text_id", "text"]):
            yield (str(row.get("text_id", "") or ""), str(row.get("text", "") or ""))

    def with_id(pair):
        return pair[0], _doc_grams(stream, pair[1], ngram_n, 1)

    for text_id, result in ordered_parallel_map(
            with_id, pairs(), workers=n_workers, pool_fn=_grams_with_id_in_worker,
            initializer=_init_gram_worker, initargs=(stream_args, ngram_n, 1),
            on_start=lambda pair: reporter.start(pair[0]),
            on_finish=lambda pair: reporter.finish(pair[0])):
        reporter.consumed()
        ys = [t.values.get(text_id) for t in targets]
        if result is None or all(v is None for v in ys):
            skipped += 1
            continue
        n_tokens, counts = result
        if not n_tokens:
            skipped += 1
            continue
        n_used += 1
        for j, y in enumerate(ys):
            if y is not None:
                n_per[j] += 1
                sy[j] += y
                syy[j] += y * y
        for gram, c in counts.items():
            x = c / n_tokens
            rec = sums.get(gram)
            if rec is None:
                rec = sums[gram] = _Sums(docs=[0] * k, sx=[0.0] * k, sxx=[0.0] * k,
                                         sxy=[0.0] * k)
            for j, y in enumerate(ys):
                if y is not None:
                    rec.docs[j] += 1
                    rec.sx[j] += x
                    rec.sxx[j] += x * x
                    rec.sxy[j] += x * y

    # a target's n is the texts that had a value for it; an n-gram's sums
    # over texts without that value were never added, so they agree
    rows: List[dict] = []
    for j, target in enumerate(targets):
        n = n_per[j]
        if n < MIN_TEXTS:
            # a correlation over a handful of texts is a coin toss with a
            # p-value; the section says the target was too thin
            thin.append((target, n))
            continue
        floor = max(int(min_docs), math.ceil(n * float(min_doc_pct) / 100.0))
        family: List[dict] = []
        for gram, rec in sums.items():
            if rec.docs[j] < floor or stoplisted(gram, stopset):
                continue
            r, p = _pearson(n, rec.sx[j], rec.sxx[j], sy[j], syy[j], rec.sxy[j])
            if r != r:
                continue
            family.append({
                "analysis": target.analysis, "feature_set": target.feature_set,
                "subset": target.subset, "outcome": target.outcome,
                "target": target.kind, "ngram": gram, "r": round(r, rounding),
                "p": p, "n_texts": n, "docs": rec.docs[j],
                "mean_relfreq": round(rec.sx[j] / n, rounding),
            })
        adjusted = adjust_pvalues([row["p"] for row in family], p_adjust) if family else []
        for row, q in zip(family, adjusted):
            row["p_adj"] = q
        family.sort(key=lambda row: (-abs(row["r"]), row["ngram"]))
        rows.extend(family)
    return rows, {"texts": n_used, "skipped": skipped,
                  "thin": [(t.label, n) for t, n in thin]}


# ---------------------------------------------------------------------------
# Where the tables go, and the manifest that names them
# ---------------------------------------------------------------------------

def _table_ref(out_tables: PathLike, name: str):
    """``<out_tables>/<name>``: a CSV under a folder, a table under a
    database address."""
    text = str(out_tables).rstrip("/").rstrip("\\")
    probe = tables.parse_ref(text + "/x")
    if probe.is_db:
        return probe.under("/".join([*probe.parts[:-1], name]))
    return tables.parse_ref(Path(text) / f"{name}.csv")


def _write_rows(ref, header: Sequence[str], rows: Iterable[dict], *, encoding: str,
                hints: Optional[Dict[str, str]] = None) -> int:
    n = 0
    with tables.write_table(ref, list(header), encoding=encoding, hints=hints) as w:
        for row in rows:
            w.writerow(["" if row.get(c) is None else row.get(c) for c in header])
            n += 1
    return n


def read_manifest(stats_dir: PathLike) -> dict:
    """The explanation tables a results folder holds, by name, or ``{}``."""
    path = Path(stats_dir) / EXPLAIN_MANIFEST
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    # a manifest written on Windows spells its file addresses with
    # backslashes, which are one long file name on this side; a run folder
    # is read from both (a real one was), so the slashes are made the same
    tables_ = doc.get("tables")
    if isinstance(tables_, dict):
        doc["tables"] = {k: (v.replace("\\", "/") if isinstance(v, str)
                             and not v.startswith("db://") else v)
                         for k, v in tables_.items()}
    return doc


_CORRELATE_COLUMNS = ["subset", "feature_set", "outcome", "target", "ngram", "r", "p",
                      "p_adj", "n_texts", "docs", "mean_relfreq"]


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def explain_predictions(
    stats_dir: PathLike = "stats_results",
    *,
    texts: Optional[PathLike] = None,
    out_tables: Optional[PathLike] = None,
    which: str = "auto",
    how: str = "both",
    ngram_n: int = 1,
    min_docs: int = 10,
    min_doc_pct: float = 1.0,
    p_adjust: str = "fdr_bh",
    lemmatize: bool = False,
    keep_punctuation: bool = False,
    pos_tagged: bool = False,
    engine: str = "nltk",
    tokenizer: str = "potts",
    stanza_lang: str = "en",
    stoplist_paths: Sequence[PathLike] = (),
    top_texts: int = 5,
    size_by: str = "mean",
    device: str = "auto",
    workers: int = 0,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    rounding: int = 6,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Explain the prediction models in a results folder in words.

    Parameters
    ----------
    stats_dir : str or Path, default "stats_results"
        The statistics output folder: its ``models/`` and predictions
        tables are what gets explained.
    texts : str or Path
        The gathered text table the models were fitted over (``text_id``,
        ``text``): a file, or a ``db://`` table.
    out_tables : str or Path, optional
        Where the tables go: a folder (default ``stats_dir``) or a database
        address, under which each table takes its name.
    which : {"auto", "all", "off"}, default "auto"
        Which feature sets to explain. ``auto`` is the sets whose columns are
        not words -- embeddings, model scores, components of those -- which
        is where a coefficient cloud says nothing; ``all`` is every set;
        ``off`` writes nothing.
    how : {"both", "correlates", "attributions"}, default "both"
        Word correlates of the predictions and residuals, token
        attributions (where the model allows them), or both.
    ngram_n : int, default 1
        The n-gram order of the correlates.
    min_docs, min_doc_pct : int, float
        An n-gram must appear in at least this many texts, and this percent
        of them, to be correlated.
    p_adjust : str, default "fdr_bh"
        The multiple-comparison correction, one family per target.
    top_texts : int, default 5
        How many of the highest and of the lowest scored texts are written
        out word by word (and drawn shaded) for each outcome.
    size_by : {"mean", "total"}, default "mean"
        What orders the attribution table and sizes its cloud: a word's
        mean share when it occurs, or its total share over the corpus
        (which the frequent words dominate).
    lemmatize, keep_punctuation, pos_tagged, engine, tokenizer, stanza_lang, stoplist_paths
        How the texts become words: the n-gram steps' own settings, so a
        word here is the word the frequency list counts.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs when it is the engine; otherwise unused.
    workers, overwrite_existing, encoding, rounding, verbose, on_progress
        The usual.

    Returns
    -------
    Path
        ``stats_dir``. The tables are named in ``explain_manifest.json``.
    """
    stats_dir = Path(stats_dir)
    out_root = out_tables or stats_dir
    manifest_path = stats_dir / EXPLAIN_MANIFEST
    which = str(which or "auto").strip().lower()
    how = str(how or "both").strip().lower()
    if which not in ("auto", "all", "off"):
        raise ValueError("which must be 'auto', 'all' or 'off'")
    if how not in ("both", "correlates", "attributions"):
        raise ValueError("how must be 'both', 'correlates' or 'attributions'")
    size_by = str(size_by or "mean").strip().lower()
    if size_by not in ("mean", "total"):
        raise ValueError("size_by must be 'mean' or 'total'")
    if which == "off":
        if verbose:
            print("[explain] explaining the predictions in words is off for this run.")
        return stats_dir

    announce(on_progress, "reading the models and their predictions")
    sets, sources = sets_sidecar(analysis_table_of(stats_dir))
    opaque = opaque_sets(stats_dir, sets=sets, sources=sources)
    targets = targets_in(stats_dir, encoding=encoding, opaque=opaque)
    chosen = [t for t in targets if which == "all" or t.opaque]
    decisions = _decisions(targets, chosen, which=which)
    if not chosen:
        _write_manifest(manifest_path, tables_={}, decisions=decisions,
                        which=which, how=how, used={})
        write_section_if_changed(stats_dir, SECTION_SLUG, _section_md(
            [], decisions, which=which, how=how, files=[], used={}))
        if verbose:
            print("[explain] no prediction model to explain in words"
                  + (" (every feature set is already words; which='all' explains "
                     "those too)." if targets else "."))
        return stats_dir
    if texts is None:
        raise ValueError("explain_predictions needs the gathered text table (texts=) to "
                         "read the words from")

    # the resume contract: the tables are remade when any model file or
    # predictions table is newer than them, or on request
    inputs = sorted({t.model for t in chosen}) + sorted(
        {p for t in chosen if (p := _predictions_for(stats_dir, t.analysis, t.model,
                                                      t.outcome)) is not None})
    made: Dict[str, str] = {}
    used: Dict[str, int] = {}
    thin_targets: Dict[str, int] = {}
    previous = read_manifest(stats_dir)
    if how in ("both", "correlates"):
        by_analysis: Dict[str, List[Target]] = {}
        for t in chosen:
            by_analysis.setdefault(t.analysis, []).append(t)
        for analysis, group in by_analysis.items():
            name = f"{analysis}_word_correlates"
            ref = _table_ref(out_root, name)
            fresh = ref.path if not ref.is_db else None
            if fresh is not None and reusable(fresh, *inputs, overwrite_existing=overwrite_existing,
                                              verbose=verbose, what=f"the {name} table"):
                made[name] = str(ref)
                # the count of texts it was made from rides along in the
                # manifest, so a reused table still says so
                kept = (previous.get("texts_used") or {}).get(name)
                if kept is not None:
                    used[name] = kept
                thin_targets.update(previous.get("thin_targets") or {})
                continue
            rows, counts = word_correlates(
                texts, group, ngram_n=ngram_n, min_docs=min_docs, min_doc_pct=min_doc_pct,
                p_adjust=p_adjust, lemmatize=lemmatize, keep_punctuation=keep_punctuation,
                pos_tagged=pos_tagged, engine=engine, tokenizer=tokenizer,
                stanza_lang=stanza_lang, device=device, stoplist_paths=stoplist_paths,
                workers=workers, encoding=encoding, rounding=rounding, on_progress=on_progress)
            announce(on_progress, f"writing {name}")
            _write_rows(ref, _CORRELATE_COLUMNS, rows, encoding=encoding,
                        hints={"r": "float", "p": "float", "p_adj": "float", "n_texts": "int",
                               "docs": "int", "mean_relfreq": "float"})
            made[name] = str(ref)
            used[name] = counts["texts"]
            for label, n in counts.get("thin", []):
                thin_targets[f"{analysis}: {label}"] = n
            if verbose:
                print(f"[explain] {name}: {len(rows)} n-gram × target rows from "
                      f"{counts['texts']} text(s) -> {ref}")
    attributed: Dict[Tuple[str, str, str], dict] = {}
    if how in ("both", "attributions"):
        attributed = _attribute_all(
            stats_dir, chosen, sources=sources, texts=texts, out_root=out_root, made=made,
            used=used, inputs=inputs, device=device, top_texts=top_texts, min_docs=min_docs,
            min_doc_pct=min_doc_pct, size_by=size_by, overwrite_existing=overwrite_existing,
            encoding=encoding, rounding=rounding, verbose=verbose, on_progress=on_progress,
            previous=previous)
        for d in decisions:
            key = (d["analysis"], d["feature_set"], d["subset"])
            if key in attributed:
                d["attributions"] = attributed[key]["status"]
            elif d["explained"]:
                d["attributions"] = "not attempted: no attributable model behind this set"
    _write_manifest(manifest_path, tables_=made, decisions=decisions, which=which, how=how,
                    used=used, thin=thin_targets)
    strongest = _strongest(made, encoding=encoding)
    write_section_if_changed(stats_dir, SECTION_SLUG, _section_md(
        chosen, decisions, which=which, how=how, files=list(made.values()), used=used,
        strongest=strongest, attributed=attributed, thin=thin_targets))
    return stats_dir


_ATTRIBUTION_COLUMNS = ["subset", "feature_set", "outcome", "target", "word",
                        "mean_contribution", "total_contribution", "sd", "count", "docs",
                        "share_of_abs_total"]
_TEXT_COLUMNS = ["subset", "feature_set", "table", "outcome", "target", "text_id", "rank",
                 "prediction", "recomputed", "sentence", "window", "position", "token", "word",
                 "contribution"]
_SUMMARY_COLUMNS = ["subset", "feature_set", "outcome", "target", "kind", "model", "method",
                    "exact", "constant", "n_texts", "skipped", "max_gap", "device", "note"]
_DIRECTION_COLUMNS = ["subset", "feature_set", "outcome", "target", "word", "direction",
                      "count", "rank"]
_ATTRIBUTION_TABLES = ("token_attributions", "attributed_texts", "attribution_summary",
                       "word_directions")


def _attribute_all(stats_dir: Path, chosen: Sequence[Target], *, sources: Dict[str, str],
                   texts, out_root, made: Dict[str, str], used: Dict[str, int],
                   inputs: Sequence[Path], device: str, top_texts: int, min_docs: int,
                   min_doc_pct: float, size_by: str, overwrite_existing: bool, encoding: str,
                   rounding: int, verbose: bool, on_progress,
                   previous: Optional[dict] = None) -> Dict[Tuple[str, str, str], dict]:
    """
    Token attributions for every chosen set made by an attributable step,
    one pass per set over the texts, every outcome of that set at once.
    Returns, per (analysis, set, subset), a status sentence and the rows
    it produced; the tables are written per analysis.
    """
    import numpy as np

    from ..helpers.gpu import device_note
    from . import attribution as attr
    from ._fit_common import effective_weights

    status: Dict[Tuple[str, str, str], dict] = {}
    rows_by_analysis: Dict[str, Dict[str, List[dict]]] = {}
    by_set: Dict[Tuple[str, str, str], List[Target]] = {}
    for t in chosen:
        if t.kind == "residual":
            continue
        by_set.setdefault((t.analysis, t.feature_set, t.subset), []).append(t)
    # the resume contract: an analysis whose attribution tables are newer
    # than every model and predictions file keeps them, and the statuses
    # the last run recorded for its sets
    previous = previous or {}
    reused: set = set()
    for analysis in {a for a, _fs, _sv in by_set}:
        summary = _table_ref(out_root, f"{analysis}_attribution_summary")
        if not reusable(summary, *inputs, overwrite_existing=overwrite_existing,
                        verbose=verbose, what=f"the {analysis} attribution tables"):
            continue
        reused.add(analysis)
        for suffix in _ATTRIBUTION_TABLES:
            name = f"{analysis}_{suffix}"
            ref = _table_ref(out_root, name)
            if tables.exists(ref):
                made[name] = str(ref)
        for d in previous.get("sets") or []:
            if d.get("analysis") == analysis and d.get("attributions"):
                status[(analysis, d.get("feature_set", ""), d.get("subset", ""))] = {
                    "status": d["attributions"]}
    analysis_rows: Optional[Dict[str, Row]] = None
    for (analysis, fs, subset), group in by_set.items():
        if analysis in reused:
            continue
        # the set's member tables: every one made by an attributable step
        # reads the text into units; the others' columns are read from the
        # analysis table into the [features] row, so the text still adds up
        members = []
        problems = []
        tables_ = set_members(fs, list(sources))
        for table in tables_:
            origin = attr.set_origin(table, sources, root=stats_dir.parent)
            why = origin.problem
            if not why:
                announce(on_progress, f"loading the {origin.kind} model behind {table} to "
                                      f"attribute the predictions")
                one = attr.backend_for(origin, device=device, verbose=verbose)
                if one.exact:
                    members.append((table, one))
                    continue
                why = one.why
            problems.append(f"{table}: {why}" if len(tables_) > 1 else why)
        if not members:
            status[(analysis, fs, subset)] = {"status": "not attributed: " + "; ".join(problems)}
            continue
        backend = attr.Composite(members)
        # one weight vector per target over the unit columns, by name; a
        # weighted column that is not a sum over units (a cosine to a
        # concept) keeps its term per text, read from the analysis table
        weights: Dict[str, tuple] = {}
        predicted: Dict[str, Dict[str, float]] = {}
        labels: Dict[str, Target] = {}
        other: Dict[str, Dict[str, float]] = {}          # label -> column -> weight
        for t in group:
            doc = _read_model(t.model)
            if doc is None:
                continue
            if analysis == "ridge":
                lin = effective_weights(doc, t.outcome)
                label = f"{t.outcome}"
                target_name = "prediction"
                scores = t.fitted or t.values
            else:
                klass = t.kind[2:]
                lin = effective_weights(doc, t.outcome, klass=klass)
                label = f"{t.outcome}:{klass}"
                target_name = f"logit_{klass}"
                scores = t.values
            w = np.zeros(len(backend.columns))
            for name, value in lin.weights.items():
                i = backend.locate(name)
                if i is None:
                    other.setdefault(label, {})[name] = float(value)
                else:
                    w[i] += value
            weights[label] = (w, lin.constant)
            predicted[label] = scores
            labels[label] = (t, target_name)
        if not weights:
            continue
        extra: Dict[str, Dict[str, float]] = {}
        if other:
            if analysis_rows is None:
                analysis_rows = _rows_by_text(analysis_table_of(stats_dir), encoding=encoding)
            for label, cols in other.items():
                term: Dict[str, float] = {}
                for tid in predicted.get(label, {}):
                    row = analysis_rows.get(tid)
                    if row is None:
                        continue
                    values = [(_num(row.get(c)), wt) for c, wt in cols.items()]
                    if any(v is None for v, _wt in values):
                        continue
                    term[tid] = sum(v * wt for v, wt in values)
                extra[label] = term
        # the texts a reader is shown: the highest and lowest scored, per target
        chosen_ids: Dict[str, List[Tuple[int, str]]] = {}
        for label, scores in predicted.items():
            ordered = sorted(scores.items(), key=lambda kv: kv[1])
            k = max(0, int(top_texts))
            ranked = [(i + 1, tid) for i, (tid, _v) in enumerate(reversed(ordered[-k:]))]
            ranked += [(-(i + 1), tid) for i, (tid, _v) in enumerate(ordered[:k])]
            chosen_ids[label] = ranked
        per_label = {label: [tid for _r, tid in ranked] for label, ranked in chosen_ids.items()}
        announce(on_progress, device_note(
            f"re-reading every text through {backend.model_label} to attribute the predictions",
            backend.device_name, backend.reason))
        result = attr.attribute(
            texts, backend=backend, weights=weights, predicted=predicted,
            chosen_ids=per_label, extra=extra, encoding=encoding, on_progress=on_progress)
        if not result.exact:
            status[(analysis, fs, subset)] = {"status": f"not attributed: {result.why}"}
            continue
        bucket = rows_by_analysis.setdefault(
            analysis, {"words": [], "texts": [], "summary": [], "directions": []})
        n_texts = result.n_texts
        floor = max(int(min_docs), math.ceil(n_texts * float(min_doc_pct) / 100.0))
        key = "mean_contribution" if size_by == "mean" else "total_contribution"
        for label, (t, target_name) in labels.items():
            table = result.words.get(label, {})
            abs_total = sum(abs(r.total) for r in table.values()) or 1.0
            words = []
            for word, rec in table.items():
                if rec.docs < floor:
                    continue
                mean = rec.total / rec.count if rec.count else 0.0
                var = rec.sumsq / rec.count - mean * mean if rec.count > 1 else 0.0
                words.append({
                    "subset": subset, "feature_set": fs, "outcome": t.outcome,
                    "target": target_name, "word": word,
                    "mean_contribution": round(mean, rounding),
                    "total_contribution": round(rec.total, rounding),
                    "sd": round(math.sqrt(max(0.0, var)), rounding),
                    "count": rec.count, "docs": rec.docs,
                    "share_of_abs_total": round(abs(rec.total) / abs_total, rounding),
                })
            words.sort(key=lambda r: (-abs(r[key]), r["word"]))
            bucket["words"].extend(words)
            rank_of = {tid: r for r, tid in chosen_ids.get(label, [])}
            for row in result.texts:
                if row["target"] != label:
                    continue
                bucket["texts"].append({
                    "subset": subset, "feature_set": fs, "table": row["table"],
                    "outcome": t.outcome,
                    "target": target_name, "text_id": row["text_id"],
                    "rank": rank_of.get(row["text_id"], ""),
                    "prediction": row["prediction"],
                    "recomputed": round(row["recomputed"], rounding),
                    "sentence": row["sentence"] if row["sentence"] is not None else "",
                    "window": row["window"] if row["window"] is not None else "",
                    "position": row["position"], "token": row["token"], "word": row["word"],
                    "contribution": round(row["contribution"], rounding + 2),
                })
            folded = sorted(other.get(label, {}))
            note = ("the class-versus-rest logit, not a probability" if analysis != "ridge"
                    else "the kept model's prediction")
            if folded:
                note += (f"; the term of {', '.join(folded)}, which are not sums over the "
                         f"words, is each text's {attr.FEATURES} row")
            bucket["summary"].append({
                "subset": subset, "feature_set": fs, "outcome": t.outcome,
                "target": target_name, "kind": backend.kind, "model": backend.model_label,
                "method": backend.describe(), "exact": True,
                "constant": round(float(weights[label][1]), rounding),
                "n_texts": n_texts, "skipped": result.skipped,
                "max_gap": f"{result.audit.get(label, 0.0):.3g}", "device": result.device,
                "note": note,
            })
        bucket["directions"].extend(_directions(
            backend, weights, labels, subset=subset, feature_set=fs, rounding=rounding))
        status[(analysis, fs, subset)] = {
            "status": f"attributed over {n_texts} text(s) with {backend.model_label}",
            "labels": list(labels)}
    for analysis, bucket in rows_by_analysis.items():
        for suffix, header, rows in (("token_attributions", _ATTRIBUTION_COLUMNS, bucket["words"]),
                                     ("attributed_texts", _TEXT_COLUMNS, bucket["texts"]),
                                     ("attribution_summary", _SUMMARY_COLUMNS, bucket["summary"]),
                                     ("word_directions", _DIRECTION_COLUMNS, bucket["directions"])):
            if suffix == "word_directions" and not rows:
                continue
            name = f"{analysis}_{suffix}"
            ref = _table_ref(out_root, name)
            announce(on_progress, f"writing {name}")
            _write_rows(ref, header, rows, encoding=encoding)
            made[name] = str(ref)
            if verbose:
                print(f"[explain] {name}: {len(rows)} row(s) -> {ref}")
    return status


def _rows_by_text(table, *, encoding: str) -> Dict[str, Row]:
    """The analysis table keyed by text id (``text_id`` when it has one,
    else its first column), for the columns no unit can carry."""
    out: Dict[str, Row] = {}
    for row in tables.read_rows(table, encoding=encoding):
        key = row.get("text_id")
        if key is None:
            key = next(iter(row.values()), "")
        out[str(key)] = row
    return out


def _directions(backend, weights: Dict[str, tuple], labels: Dict[str, tuple], *,
                subset: str, feature_set: str, rounding: int) -> List[dict]:
    """
    Every word a model knows, by the direction it would push each target:
    the prediction a text of that one word gets, minus the constant. Only
    a static model has units without a text; the encoders yield nothing.
    """
    import numpy as np

    chunks = backend.vocabulary()
    if chunks is None:
        return []
    names = list(weights)
    W = np.stack([np.asarray(weights[label][0], dtype=np.float64) for label in names], axis=1)
    per_label: Dict[str, List[Tuple[str, Optional[int], float]]] = {label: [] for label in names}
    for words, counts, block in chunks:
        D = block @ W
        for j, label in enumerate(names):
            per_label[label].extend(zip(words, counts, D[:, j].tolist()))
    out: List[dict] = []
    for label, (t, target_name) in labels.items():
        ranked = sorted(per_label[label], key=lambda wcd: (-abs(wcd[2]), wcd[0]))
        for rank, (word, count, direction) in enumerate(ranked, start=1):
            out.append({"subset": subset, "feature_set": feature_set, "outcome": t.outcome,
                        "target": target_name, "word": word,
                        "direction": round(float(direction), rounding),
                        "count": "" if count is None else count, "rank": rank})
    return out


def _decisions(targets: Sequence[Target], chosen: Sequence[Target], *, which: str
               ) -> List[dict]:
    """One line per (analysis, feature set): explained or not, and why."""
    seen: Dict[Tuple[str, str, str], dict] = {}
    picked = {(t.analysis, t.feature_set, t.subset) for t in chosen}
    for t in targets:
        key = (t.analysis, t.feature_set, t.subset)
        if key in seen:
            continue
        explained = key in picked
        seen[key] = {
            "analysis": t.analysis, "feature_set": t.feature_set, "subset": t.subset,
            "opaque": bool(t.opaque), "explained": explained,
            "reason": ("its columns are not words" if t.opaque else
                       "every feature set" if which == "all" else
                       "its features are words already, so its coefficient cloud "
                       "reads on its own (which='all' explains it too)"),
            "attributions": "not attempted",
        }
    return list(seen.values())


def _write_manifest(path: Path, *, tables_: Dict[str, str], decisions: List[dict],
                    which: str, how: str, used: Dict[str, int],
                    thin: Optional[Dict[str, int]] = None) -> None:
    from ..helpers.atomic import atomic_write

    doc = {"tables": tables_, "sets": decisions, "which": which, "how": how, "texts_used": used,
           "thin_targets": thin or {}}
    text = json.dumps(doc, indent=1)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write(text)


def _strongest(made: Dict[str, str], *, encoding: str, top: int = 6
               ) -> Dict[Tuple[str, str, str, str], Dict[str, List[Tuple[str, float, float]]]]:
    """Per (analysis, set, outcome, target): the strongest positive and
    negative correlates that survive the correction, for the section."""
    out: Dict[Tuple[str, str, str, str], Dict[str, List[Tuple[str, float, float]]]] = {}
    for name, address in made.items():
        if not name.endswith("_word_correlates"):
            continue
        analysis = name[:-len("_word_correlates")]
        for r in tables.read_rows(address, encoding=encoding):
            key = (analysis, r.get("feature_set", ""), r.get("outcome", ""), r.get("target", ""))
            bucket = out.setdefault(key, {"up": [], "down": []})
            rv, q = _num(r.get("r")), _num(r.get("p_adj"))
            if rv is None or q is None or q >= 0.05:
                continue
            side = bucket["up" if rv > 0 else "down"]
            if len(side) < top:
                side.append((r.get("ngram", ""), rv, q))
    return out


def _section_md(chosen: Sequence[Target], decisions: Sequence[dict], *, which: str,
                how: str, files: Sequence[str], used: Dict[str, int],
                strongest: Optional[dict] = None, attributed: Optional[dict] = None,
                thin: Optional[Dict[str, int]] = None) -> str:
    lines = ["## What the predictions track in words", ""]
    if not chosen:
        if not decisions:
            lines.append("No prediction model was fitted in this run, so there is "
                         "nothing to explain.")
        else:
            lines.append("Every prediction model here is fitted on features that are "
                         "words already (a dictionary, readability, n-grams), so its "
                         "coefficient cloud reads on its own. Set "
                         "`explain_predictions: all` to correlate words with these "
                         "predictions too.")
        return "\n".join(lines)
    lines.append(
        "A model fitted on embeddings is a weight per dimension, and a cloud of "
        "`e238` says nothing. So for the feature sets below, every word's "
        "relative frequency across texts was correlated with the model's "
        "**held-out** prediction (what the scores track in language) and with "
        "its **residual**, observed minus predicted (what the outcome has that "
        "the model missed). Held-out, because the in-sample prediction has "
        "seen the answer. Positive means the word goes with a higher score.")
    strongest = strongest or {}
    by_set: Dict[Tuple[str, str, str], List[Target]] = {}
    for t in chosen:
        by_set.setdefault((t.analysis, t.feature_set, t.subset), []).append(t)
    for (analysis, fs, subset), group in by_set.items():
        where = f" [{subset}]" if subset else ""
        lines += ["", f"### {analysis}: {fs}{where}"]
        outcomes = sorted({t.outcome for t in group})
        for outcome in outcomes:
            kinds = [t.kind for t in group if t.outcome == outcome]
            for kind in kinds:
                found = strongest.get((analysis, fs, outcome, kind), {})
                up, down = found.get("up", []), found.get("down", [])
                what = ("higher predicted" if kind == "prediction" else
                        "a larger residual (under-predicted)" if kind == "residual"
                        else f"a higher held-out probability of {kind[2:]}")
                if not up and not down:
                    lines.append(f"- `{outcome}`, {kind}: no word survived the correction.")
                    continue
                said_up = ", ".join(f"*{w}* ({r:+.2f})" for w, r, _q in up) or "none"
                said_down = ", ".join(f"*{w}* ({r:+.2f})" for w, r, _q in down) or "none"
                lines.append(f"- `{outcome}`: words going with {what}: {said_up}; "
                             f"the other way: {said_down}.")
    if thin:
        lines += ["", "No correlates for " + "; ".join(
            f"{label} ({n} text(s) with a held-out value)" for label, n in thin.items())
            + f": fewer than {MIN_TEXTS} texts make a correlation a coin toss with a p-value."]
    skipped = [d for d in decisions if not d["explained"]]
    if skipped:
        lines += ["", "Not explained here: " + "; ".join(
            f"{d['feature_set']} ({d['analysis']}): {d['reason']}" for d in skipped) + "."]
    done = [d for d in decisions if d["explained"]]
    if done:
        lines += ["", "**Token attributions.** A prediction over a feature set that is a "
                      "sum over its tokens -- mean-pooled embeddings (each token in its "
                      "context), word vectors (each word) -- is, exactly, a sum of one "
                      "share per token, so the share of every word in every text was "
                      "computed and summed per word (the `*_token_attributions` table: "
                      "mean and total share, with how often the word occurs) and, for "
                      "the highest and lowest scored texts, written out word by word "
                      "(`*_attributed_texts`, drawn as shaded texts under the word "
                      "clouds). The `*_attribution_summary` table records the model, "
                      "the method and the largest gap between a text's summed shares "
                      "and its recomputed prediction, which is the exactness audit. A "
                      "word-vector model's whole vocabulary is ranked in "
                      "`*_word_directions` by the push each word would give the "
                      "prediction on its own."]
        for d in done:
            lines.append(f"- {d['feature_set']} ({d['analysis']}): {d['attributions']}.")
    if used:
        lines += ["", "Texts read: " + ", ".join(f"{n} for `{Path(k).name if '://' not in k else k}`"
                                                   for k, n in used.items()) + "."]
    if files:
        lines += ["", "Details: " + ", ".join(f"`{f}`" for f in files) + "."]
    return "\n".join(lines)


CLI = CliSpec({"explain": explain_predictions},
              description="Explain the prediction models in a results folder in words.")


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
