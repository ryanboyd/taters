# taters/text/topic_model_nmf.py
"""
Non-negative matrix factorization: topics without the probability story.

What it does
------------
NMF asks something simpler than LDA. Split the document-term matrix into two
non-negative pieces -- documents by topics, topics by terms -- whose product is
as close to the original as it can get. No generative story, no priors, nothing
that has to sum to one. A document's topic weights are just weights: one can be
large without forcing another to be small.

That simplicity is why it is worth having beside LDA rather than instead of it.
On short texts -- tweets, open-ended survey answers, single utterances -- LDA
often struggles because there is not enough of each document to infer a mixture
from, and NMF's topics come out sharper and easier to name. On long documents
the two tend to agree.

Which weighting, and why it differs from LDA
--------------------------------------------
NMF defaults to **tf-idf**, and that is not an accident of taste: without it,
the factorization spends its first topic on whatever words are simply common,
because those are the cells with the most mass to explain. Down-weighting what
is everywhere is how NMF gets topics rather than a frequency ranking.

LDA is the opposite -- it is only defined over integer counts, and refuses
anything else. The two engines genuinely want different matrices, which is why
each builds its own instead of sharing one.

What comes out
--------------
One column per topic, ``Factor_1..Factor_k``. They are *weights*, not
proportions: bigger means more of that topic, zero means none, and they do not
add up to anything in particular. That makes them easier to use as ordinary
predictors than LDA's proportions, which are compositional.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.feature_columns import ColumnSpec
from ..helpers.model_spec import one_model_path
from ..helpers.progress import Ticker, announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from . import _topics
from .ngram_prep import tags_of, words_of

__all__ = ["topic_model_nmf", "apply_nmf_model", "MODEL_KIND", "MODEL_FORMAT",
           "FEATURE_COLUMNS"]

PathLike = Union[str, Path]

MODEL_KIND = "taters-nmf-model"
MODEL_FORMAT = 1

#: One column per factor. `Factor_` rather than LDA's `Topic_` or MEM's `Theme_`
#: so that a study running more than one topic model gets three readable sets of
#: columns instead of a collision -- see `helpers/feature_columns.py`.
FEATURE_COLUMNS = ColumnSpec(label="Topic model (NMF)", patterns=("Factor_{n}",),
                             reduces_to="Supertopic")


def _model_version() -> str:
    try:
        from importlib.metadata import version
        return version("taters")
    except Exception:
        return ""


def _load_model(model_json: PathLike) -> dict:
    """Read a saved NMF model, refusing anything that is not one."""
    path = Path(model_json)
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"{path} is not readable as a saved model: {e}") from e

    def broken(why: str) -> ValueError:
        return ValueError(f"{path} is not a usable NMF model: {why}")

    if not isinstance(model, dict):
        raise broken("it is not a JSON object")
    if model.get("kind") != MODEL_KIND:
        raise broken(f"it says kind={model.get('kind')!r}, not {MODEL_KIND!r}")
    if int(model.get("format", 0)) > MODEL_FORMAT:
        raise broken(
            f"it was written in format {model['format']}, and this Taters "
            f"understands up to {MODEL_FORMAT}. Upgrade Taters to use it.")

    for key in ("text", "matrix", "model"):
        if not isinstance(model.get(key), dict):
            raise broken(f"it has no {key!r} section")

    terms = model["matrix"].get("terms") or []
    h = model["model"].get("components") or []
    if not terms or not h:
        raise broken("it carries no vocabulary or no factors")
    if any(len(row) != len(terms) for row in h):
        raise broken("its factors and its vocabulary disagree in size")
    if len(model["model"].get("topics") or []) != len(h):
        raise broken("its factor names do not match its factors")
    return model


def _topic_names(k: int) -> List[str]:
    return [f"Factor_{i + 1}" for i in range(k)]


@records_settings(
    binding=TEXT_INPUT, grain=TEXT_GRAIN,
    outputs=("out_features_csv", "out_model_json", "out_loadings_csv",
             "out_top_terms_csv"),
    # the topics are fitted to this corpus, so the honest way to measure them on
    # another is to apply the saved model rather than to fit again. refitting
    # gives a second study its own topics in its own order, and a ridge trained
    # on the first one cannot be scored with them at all.
    replay=(f"{__name__}:apply_nmf_model", {"model_json": "out_model_json"}),
    bookkeeping=("token_count",))
def topic_model_nmf(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    workers: int = 0,
    device: str = "auto",
    on_progress: Optional[Callable[..., None]] = None,

    # ----- Output -----
    out_features_csv: Optional[PathLike] = None,
    out_model_json: Optional[PathLike] = None,
    out_loadings_csv: Optional[PathLike] = None,
    out_top_terms_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ----- the vocabulary this model builds for itself -----
    ngram_n: int = 1,
    stoplist_paths: Optional[Sequence[PathLike]] = None,
    min_freq: int = 5,
    min_obs_pct: float = 0.10,
    min_token_count: int = 10,
    min_npmi: Optional[float] = None,
    lemmatize: bool = False,
    pos_tagged: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    keep_punctuation: bool = False,
    weighting: Literal["tfidf", "count"] = "tfidf",
    matrix_rounding: int = 4,
    vocab_min_freq: float = 0,
    vocab_min_obs_pct: float = 0,
    vocab_rule: Literal["top_n", "min_obs_pct", "min_freq"] = "top_n",
    vocab_top_n: int = 2000,
    vocab_rank_by: Literal["obs_pct", "frequency"] = "obs_pct",

    # ----- NMF options -----
    n_topics: int = 20,
    beta_loss: Literal["frobenius", "kullback-leibler"] = "frobenius",
    iters: int = 200,
    tol: float = 1e-4,
    top_terms: int = 15,
    rounding: int = 4,
) -> Path:
    """
    Fit NMF factors to a corpus; write per-document weights and a model.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The corpus, given exactly one of these ways: a spreadsheet, a folder of
        documents, an already-gathered analysis-ready table, or a gathered
        table to write and reuse.
    workers : int, default=0
        Worker processes for gathering, counting and scoring. 0 picks a
        sensible number for the machine and the job.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs, when ``engine="stanza"``. The fit itself is CPU
        arithmetic and does not use a GPU.
    out_features_csv : str or pathlib.Path, optional
        Per-document topic proportions. Defaults to
        ``./features/topic_model_nmf/<gathered filename>``. The model, the
        loadings and the top-terms table are written beside it unless given
        their own paths.
    out_model_json, out_loadings_csv, out_top_terms_csv : str or Path, optional
        Where the reusable model, the term-by-topic table the word clouds are
        drawn from, and the readable top-terms summary go.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, skip the work and return the path.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns that identify each row when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Whether multiple text columns are joined into one document or measured
        separately.
    group_by : Sequence[str] or None, optional
        Optional grouping keys used during CSV gathering -- one document per
        group instead of one per row.
    pattern : str, default=every document type
        Which files to read when gathering from a folder. Only with ``txt_dir``.
    ngram_n : int, default=1
        Highest n-gram order to consider for the vocabulary.
    stoplist_paths : Sequence[str or pathlib.Path] or None, optional
        Word lists to drop before counting. Worth using: function words are
        the most frequent words in any corpus and a topic made of them tells
        you nothing.
    min_freq : int, default=5
        Drop terms rarer than this from the vocabulary.
    min_obs_pct : float, default=0.10
        Drop terms appearing in fewer than this *percent* of documents.
    min_token_count : int, default=10
        Skip documents shorter than this many tokens entirely.
    min_npmi : float, optional
        Optional collocation threshold for orders above one.
    lemmatize : bool, default=False
        Lemmatize before counting, so "run" and "running" are one term.
    pos_tagged : bool, default=False
        Keep part-of-speech tags on terms, so "book/NOUN" and "book/VERB" are
        different terms.
    engine : {"nltk", "stanza"}, default="nltk"
        Which tokenizer and tagger to use.
    tokenizer : {"potts", "stanza"}, default="potts"
        Which tokenizer rules to apply.
    stanza_lang : str, default="en"
        Language for the Stanza engine.
    keep_punctuation : bool, default=False
        Count punctuation as terms.
    weighting : {"tfidf", "count"}, default="tfidf"
        tf-idf by default, and that is a real recommendation rather than a
        shrug: without it the factorization spends its first factor on whatever
        words are merely common, because those cells hold the most mass. Raw
        counts are allowed for anyone who wants NMF and LDA over the same
        matrix. ``binary`` and ``relfreq`` are refused -- the first throws away
        how much a word was used, the second rescales every document to the
        same total, which is the thing the factorization is trying to see.
    matrix_rounding : int, default=4
        Decimal places in the written matrix.
    vocab_min_freq, vocab_min_obs_pct, vocab_rule, vocab_top_n
        How the vocabulary is cut from the frequency list. ``vocab_top_n``
        defaults to 2000 here rather than MEM's 500: more factors need more
        vocabulary to separate them.
    vocab_rank_by : {"obs_pct", "frequency"}, default="obs_pct"
        What ``top_n`` ranks by when it cuts the vocabulary. ``obs_pct`` is the
        share of documents a term appears in; ``frequency`` is its raw count.

        Spread, not volume, is what a topic model needs: a word one document
        repeats five hundred times outranks everything on frequency and
        cannot distinguish a thing, because it only ever describes that one
        document. A word used once each across half the corpus is what topics
        are made of. (The standalone document-term matrix still ranks by
        frequency, because a *feature* table usually does want the commonest
        terms.)
    n_topics : int, default=20
        How many topics to fit. The one setting with no good default -- see
        :func:`taters.text.topic_count_sweep.sweep_topic_count` for help
        choosing, and expect to read the topics before believing a number.
    beta_loss : {"frobenius", "kullback-leibler"}, default="frobenius"
        What "close to the original" means. Frobenius is least squares: faster,
        steadier, and the usual choice. Kullback-Leibler matches the way counts
        actually vary and tends to give sparser, more readable factors on short
        texts -- worth trying if the Frobenius factors all look alike.
    iters : int, default=200
        Most passes to make. It stops early when the error settles.
    tol : float, default=1e-4
        How small a relative improvement counts as settled.
    top_terms : int, default=15
        How many words to list per factor in the readable summary.
    rounding : int, default=4
        Decimal places in the written weights and loadings.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``token_count``, then
        ``Factor_1..Factor_k``.

    Notes
    -----
    There is no ``seed``. NMF starts from NNDSVD, which is derived from the
    matrix's own singular vectors and is therefore deterministic -- two runs on
    one corpus give one answer, with nothing random to record.

    The weights are not proportions. They do not sum to anything in particular,
    which makes them more straightforward as regression predictors than LDA's
    topic shares -- those are compositional, and using all of them at once is a
    known trap. Factor *scale* is arbitrary, though: doubling a factor in H and
    halving it in W is the same model, so compare a factor against itself across
    documents rather than against another factor.
    """
    if weighting not in ("tfidf", "count"):
        raise ValueError(
            f"NMF needs tf-idf or raw counts, not {weighting!r}. A binary or "
            "relative-frequency matrix factors without complaint and gives "
            "topics of something nobody asked about -- binary throws away how "
            "much a word was used, and relative frequency rescales every "
            "document to the same total, which is the one thing the "
            "factorization is trying to see.")

    import numpy as np

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers)

    if out_features_csv is None:
        out_features_csv = (Path.cwd() / "features" / "topic_model_nmf"
                            / analysis_ready.name)
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    stem = out_features_csv.stem
    model_path = Path(out_model_json) if out_model_json else \
        out_features_csv.with_name(f"{stem}_model.json")
    loadings_path = Path(out_loadings_csv) if out_loadings_csv else \
        out_features_csv.with_name(f"{stem}_loadings.csv")
    terms_path = Path(out_top_terms_csv) if out_top_terms_csv else \
        out_features_csv.with_name(f"{stem}_top_terms.csv")

    if not overwrite_existing and out_features_csv.is_file():
        print("NMF factor output file already exists; returning existing file.")
        return out_features_csv

    text_settings = dict(lemmatize=lemmatize, pos_tagged=pos_tagged,
                         engine=engine, tokenizer=tokenizer,
                         stanza_lang=stanza_lang,
                         keep_punctuation=keep_punctuation, device=device)
    vocab_settings = dict(vocab_min_freq=vocab_min_freq,
                          vocab_min_obs_pct=vocab_min_obs_pct,
                          vocab_rule=vocab_rule, vocab_top_n=vocab_top_n,
                          vocab_rank_by=vocab_rank_by)
    freq_list_csv, dtm_csv, terms, columns, vocab = _topics.build_matrix(
    # the matrix folder is derived from the output's *stem*, not from its
    # folder. a fixed-name sibling (`parent / "matrix"`) is the same path for
    # every step that writes into one folder, and that produced the two worst
    # bugs in this feature: three topic models building over each other's
    # matrix, and an apply leaving a frozen vocabulary where a later fit picked
    # it up and modeled the wrong corpus. Deriving from the stem makes both
    # impossible rather than merely wired-around -- and it is the idiom the
    # rest of the codebase already uses for companion files.
        analysis_ready=analysis_ready,
        out_dir=out_features_csv.with_name(f"{stem}_matrix"),
        weighting=weighting, matrix_rounding=matrix_rounding,
        text_settings=text_settings, vocab_settings=vocab_settings,
        ngram_settings=dict(ngram_n=ngram_n, stoplist_paths=stoplist_paths,
                            min_freq=min_freq, min_obs_pct=min_obs_pct,
                            min_token_count=min_token_count, min_npmi=min_npmi),
        overwrite_existing=overwrite_existing, workers=workers,
        on_progress=on_progress, encoding=encoding)

    def batches():
        return _topics.stream_counts(dtm_csv, encoding=encoding, skip_cols=2)

    # NMF needs the whole matrix at once -- the multiplicative updates touch
    # every cell on every pass -- so unlike LDA it cannot stream. say how much
    # that will cost before spending minutes reading it in, not after.
    n_documents = sum(len(block) for block in batches())
    wanted = _topics.nmf_memory_gb(n_documents, len(terms), n_topics)
    if wanted >= 1.0:
        import warnings
        warnings.warn(
            f"NMF holds the whole matrix in memory: about {wanted:.1f} GB for "
            f"{n_documents:,} documents by {len(terms):,} terms. Reduce "
            "`vocab_top_n` if that is more than this machine has.")

    announce(on_progress, "fitting factors")
    matrix = _topics.read_matrix(dtm_csv, encoding=encoding, skip_cols=2)
    _w, h, trace = _topics.fit_nmf(matrix, n_topics, beta_loss=beta_loss,
                                   iters=iters, tol=tol, on_progress=on_progress)
    topic_names = _topic_names(n_topics)

    # the loadings table, in the shape `figures.wordclouds.theme_wordclouds`
    # already reads -- term, an optional pos, then one column per topic. it is
    # the same file for every topic model on purpose, so the word clouds are
    # written once and not once per engine.
    with atomic_write(loadings_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        head = ["term", "pos", *topic_names] if pos_tagged else ["term", *topic_names]
        writer.writerow(head)
        # scaled to sum to one per factor purely so the numbers are readable
        # and comparable between factors. NMF's H carries no such constraint,
        # and the scale of a factor is arbitrary anyway -- W absorbs it.
        weights = h / (h.sum(axis=1)[:, None] + 1e-100)
        for j, gram in enumerate(terms):
            row = [words_of(gram)]
            if pos_tagged:
                row.append(tags_of(gram))
            row.extend(round(float(weights[i, j]), rounding)
                       for i in range(n_topics))
            writer.writerow(row)

    # and the readable one: what each topic is actually about, in order
    with atomic_write(terms_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "rank", "term", "weight"])
        weights = h / (h.sum(axis=1)[:, None] + 1e-100)
        for i, name in enumerate(topic_names):
            order = np.argsort(weights[i])[::-1][:top_terms]
            for rank, j in enumerate(order, start=1):
                writer.writerow([name, rank, words_of(terms[j]),
                                 round(float(weights[i, j]), rounding)])

    from datetime import date

    model = {
        "kind": MODEL_KIND,
        "format": MODEL_FORMAT,
        "created": date.today().isoformat(),
        "taters": _model_version(),
        # `device` is deliberately absent: where Stanza ran is a runtime choice,
        # not part of the instrument, and recording it would make two otherwise
        # identical models compare as different.
        "text": {"lemmatize": lemmatize, "pos_tagged": pos_tagged,
                 "engine": engine, "tokenizer": tokenizer,
                 "stanza_lang": stanza_lang,
                 "keep_punctuation": keep_punctuation},
        "matrix": {"weighting": weighting, "rounding": matrix_rounding,
                   "terms": terms, "columns": columns,
                   "idf": [vocab[g]["idf"] for g in terms]},
        # no seed: NNDSVD initialization is deterministic, so there is nothing
        # random here to pin down and nothing to forget to record.
        "model": {"n_documents": n_documents, "topics": topic_names,
                  "beta_loss": beta_loss, "iters": iters, "tol": tol,
                  "error_trace": [round(float(x), 4) for x in trace],
                  "components": h.tolist()},
    }
    with atomic_write(model_path, encoding="utf-8") as f:
        json.dump(model, f, indent=1)

    _score_matrix(dtm_csv, out_features_csv, h=h, beta_loss=beta_loss,
                  topic_names=topic_names, rounding=rounding, encoding=encoding,
                  on_progress=on_progress, n_documents=n_documents)
    return out_features_csv


def _score_matrix(dtm_csv: Path, out_csv: Path, *, h, beta_loss: str,
                  topic_names: Sequence[str], rounding: int, encoding: str,
                  on_progress, n_documents: int) -> None:
    """Stream the matrix through inference and write one row per document.

    Shared by the fit and by :func:`apply_nmf_model` so the two cannot drift:
    if they did, applying a model to its own training texts would stop
    reproducing the training scores, and that is the test that would notice.
    """
    ticker = Ticker(on_progress, n_documents)
    with Path(dtm_csv).open("r", newline="", encoding=encoding) as f, \
            atomic_write(out_csv, newline="", encoding=encoding) as out:
        reader = csv.reader(f)
        next(reader)
        writer = csv.writer(out)
        writer.writerow(["text_id", "token_count", *topic_names])
        for row in reader:
            ticker.tick(message="scoring documents")
            counts = [float(x or 0) for x in row[2:]]
            weights = _topics.infer_nmf(h, counts, beta_loss=beta_loss)[0]
            writer.writerow([row[0], row[1],
                             *(round(float(v), rounding) for v in weights)])


@records_settings(
    # the model file is the whole instrument -- vocabulary, tokenizer settings
    # and topics -- so it is compared by content. two models with the same name
    # and different topics have to come out as different measurements.
    binding=TEXT_INPUT, grain=TEXT_GRAIN, assets={"model_json": None},
    outputs=("out_features_csv",), bookkeeping=("token_count",))
def apply_nmf_model(
    *,
    model_json: PathLike,

    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    workers: int = 0,
    device: str = "auto",
    on_progress: Optional[Callable[..., None]] = None,

    # ----- Output -----
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    rounding: int = 4,
) -> Path:
    """
    Score a corpus with topics fitted somewhere else.

    Takes no vocabulary, tokenizer or weighting settings at all: they come out
    of the model, because the model *is* the instrument. That is what makes two
    studies comparable -- measure the second corpus with the first one's topics,
    rather than fitting new topics and hoping Topic_3 means the same thing.

    Parameters
    ----------
    model_json : str or pathlib.Path
        A model written by :func:`topic_model_nmf`, or a library folder holding
        exactly one.
    csv_path, txt_dir, analysis_csv, gathered_csv
        The corpus to score, exactly one way.
    workers : int, default=0
        Worker processes. 0 picks a sensible number.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs, if the model was built with it.
    out_features_csv : str or pathlib.Path, optional
        Defaults to ``./features/topic_model_nmf/<gathered filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, return it unchanged.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols, id_cols, mode, group_by, pattern
        Gathering options, as in :func:`topic_model_nmf`.
    rounding : int, default=4
        Decimal places in the written proportions.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``token_count``, then one column per
        topic of the model.
    """
    import numpy as np

    model = _load_model(one_model_path(model_json))
    text_cfg = model["text"]
    matrix_cfg = model["matrix"]
    fit = model["model"]

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers)

    if out_features_csv is None:
        # `_applied`, so that fitting and applying with the defaults do not
        # write to one file -- and so their matrix folders, derived from this
        # stem, can never be the same one.
        out_features_csv = (Path.cwd() / "features" / "topic_model_nmf"
                            / f"{analysis_ready.stem}_applied{analysis_ready.suffix}")
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        print("NMF factor output file already exists; returning existing file.")
        return out_features_csv

    # rebuild the matrix the model's way, not the caller's. the vocabulary is
    # pinned to the model's own term list, so a word the training corpus never
    # saw simply does not count -- which is the honest thing: the model has no
    # topic for it.
    from .build_doc_term_matrix import build_doc_term_matrix

    # the apply gets its own matrix folder, never the fit's: the frozen
    # frequency list written below is deliberately *not* provenance-
    # recorded -- it is the model's vocabulary, not a measurement of this
    # corpus. Sharing one folder meant a later fit found that file, saw no
    # record to disagree with, reused it, and modeled the applied model's
    # vocabulary instead of its own corpus's. Fit on A, apply to B, fit on B,
    # and the second model came out holding A's words, with no error anywhere.
    # the matrix folder is derived from the output's *stem*, not from its
    # folder. a fixed-name sibling (`parent / "matrix"`) is the same path for
    # every step that writes into one folder, and that produced the two worst
    # bugs in this feature: three topic models building over each other's
    # matrix, and an apply leaving a frozen vocabulary where a later fit picked
    # it up and modeled the wrong corpus. Deriving from the stem makes both
    # impossible rather than merely wired-around -- and it is the idiom the
    # rest of the codebase already uses for companion files.
    matrix_dir = out_features_csv.with_name(
        f"{out_features_csv.stem}_matrix")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    dtm_csv = Path(build_doc_term_matrix(
        freq_list_csv=_frozen_freq_list(model, matrix_dir, encoding=encoding),
        analysis_csv=analysis_ready,
        out_features_csv=matrix_dir / "dtm.csv",
        overwrite_existing=True, workers=workers, on_progress=on_progress,
        encoding=encoding, weighting=matrix_cfg["weighting"],
        rounding=int(matrix_cfg["rounding"]),
        vocab_min_freq=0, vocab_min_obs_pct=0, vocab_rule="top_n",
        vocab_top_n=len(matrix_cfg["terms"]), vocab_rank_by="frequency",
        lemmatize=text_cfg["lemmatize"], pos_tagged=text_cfg["pos_tagged"],
        engine=text_cfg["engine"], tokenizer=text_cfg["tokenizer"],
        stanza_lang=text_cfg["stanza_lang"],
        keep_punctuation=bool(text_cfg.get("keep_punctuation", True)),
        device=device))

    h = np.asarray(fit["components"], dtype=np.float64)
    n_documents = sum(1 for _ in Path(dtm_csv).open(encoding=encoding)) - 1
    # the divergence the model was fitted with, not this run's default: the
    # two have different update rules, so scoring a KL model with the
    # least-squares one gives weights the fit never produced.
    _score_matrix(dtm_csv, out_features_csv, h=h,
                  beta_loss=str(fit.get("beta_loss", "frobenius")),
                  topic_names=list(fit["topics"]), rounding=rounding,
                  encoding=encoding, on_progress=on_progress,
                  n_documents=max(n_documents, 0))
    return out_features_csv


def _frozen_freq_list(model: dict, out_dir: Path, *, encoding: str) -> Path:
    """Write the model's own vocabulary out as a frequency list.

    `build_doc_term_matrix` chooses its columns from a frequency list, and when
    applying a model there is nothing to choose: the columns are the model's
    terms, in the model's order. So we hand it a list that can only produce
    that answer, rather than counting the new corpus and hoping the cut lands
    in the same place.
    """
    matrix_cfg = model["matrix"]
    terms = list(matrix_cfg["terms"])
    idf = list(matrix_cfg.get("idf") or [0.0] * len(terms))
    pos_tagged = bool(model["text"].get("pos_tagged"))

    path = out_dir / "freq_list.csv"
    with atomic_write(path, newline="", encoding=encoding) as f:
        # `_load_vocabulary` reads by name and needs ngram/frequency/obs_pct/idf,
        # plus `pos` exactly when the model was tagged -- it refuses a mismatch
        # both ways, because a tagged vocabulary scanned without tags matches
        # almost nothing and does it silently.
        head = ["ngram", "frequency", "obs_pct", "idf"]
        if pos_tagged:
            head.insert(1, "pos")
        writer = csv.DictWriter(f, fieldnames=head)
        writer.writeheader()
        for rank, gram in enumerate(terms):
            # a descending, distinct frequency so that whatever cut the vocab
            # rule applies, it keeps everything and in the model's own order.
            # obs_pct is 100 for the same reason: nothing here may be dropped.
            row = {"ngram": words_of(gram),
                   "frequency": float(len(terms) - rank),
                   "obs_pct": 100.0,
                   "idf": float(idf[rank]) if rank < len(idf) else 0.0}
            if pos_tagged:
                row["pos"] = tags_of(gram)
            writer.writerow(row)
    return path


CLI = CliSpec(
    {"fit": topic_model_nmf, "apply": apply_nmf_model},
    description="Fit NMF factors to a corpus, or score a corpus with saved factors.",
    aliases={"csv_path": ["--csv"], "out_features_csv": ["--out"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
