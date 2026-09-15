"""
Topic model: the Meaning Extraction Method (MEM).

Chung & Pennebaker's MEM (2008): take a document-term matrix over frequent
content terms, run a PCA with varimax rotation, and read the rotated
components as *themes* -- clusters of words that rise and fall together
across documents. Each document gets a score per theme; each term gets a
loading per theme.

Two commitments shape this module:

* **Exact and out-of-core, without heavy dependencies.** A MEM matrix is
  long (documents) but narrow (hundreds to a few thousand terms), so the
  fit streams the rows once to accumulate the column means and the p-by-p
  cross-product matrix, then eigendecomposes the correlation matrix in
  memory. That is the *exact* full PCA -- no randomized SVD, no seed, no
  dask -- with memory bounded by the vocabulary width, never the corpus
  length. Should a vocabulary ever be too wide for p-by-p to fit, that is
  the moment to reach for an out-of-core SVD, behind this same interface.

* **A fitted model is a reusable instrument.** ``topic_model_mem`` writes a
  model file carrying everything needed to score *new* text on the same
  themes: the exact vocabulary (tagged keys included), the tokenizer/engine
  settings that produced it, the weighting, the training means and standard
  deviations, and the projection. ``apply_mem_model`` rebuilds the same
  document-term matrix for new texts -- through the very same scan and
  weighting code the DTM step uses, imported rather than copied -- and
  projects it. Applying a model to its own training texts reproduces the
  training scores exactly.

Eigenvalues here are those of the **correlation matrix** (they average 1.0),
which is what both retention rules are stated over: ``n_components=0`` picks
the count by parallel analysis (a theme is kept while its eigenvalue beats
what a random matrix of the same size gives at that rank) or, on request,
by the Kaiser criterion (keep eigenvalue >= 1). Rotation redistributes variance,
so the reported per-theme eigenvalues and %-variance follow the rotated
loadings, ordered largest first, exactly as classic MEM tooling reports them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.progress import Ticker, announce
from ..helpers.model_spec import one_model_path
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import (resolve_analysis_ready)
from ..stats.pca import fit_axes, project_row, stream_moments, warn_if_wide
from .build_doc_term_matrix import (
    _load_vocabulary,
    column_names,
)
from .ngram_prep import make_token_stream, tags_of, words_of
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: Model file format, bumped on any incompatible change so an older Taters
#: refuses a newer model instead of misreading it.
MODEL_FORMAT = 1


# ---------------------------------------------------------------------------
# linear algebra: taters.stats.pca does the heavy lifting; this module is just
# the MEM instrument wrapped around it (vocabulary, tokenizer settings, and
# applying the model again later)
# ---------------------------------------------------------------------------

def _theme_row(text_id: str, token_count, cells, kept, mu, sigma, projection,
               rounding: int) -> list:
    """One output row, shared by fit and apply so the two can never drift."""
    return [text_id, token_count,
            *project_row(cells, kept, mu, sigma, projection, rounding)]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

@records_settings(
    # the matrix and the vocabulary it came from are our binding, so that the
    # chain can walk back through the matrix's own record to the gather. we
    # used to record nothing here, and the theme scores (a feature table, by
    # declaration) came up as "unavailable" to every model fitted on them.
    # the refusal's advice ("re-run the extraction so a record is written")
    # couldn't even be followed, because nothing ever wrote one
    binding=("dtm_csv", "freq_list_csv"),
    outputs=("out_features_csv", "out_model_json", "out_loadings_csv",
             "out_eigenvalues_csv"),
    # the themes are fitted to this corpus, so the honest way to measure them
    # on another is to apply the saved model. when we refit instead, a second
    # study picked a different number of themes, and a ridge fitted on the
    # first one couldn't be scored at all
    replay=(f"{__name__}:apply_mem_model", {"model_json": "out_model_json"}),
    bookkeeping=("token_count",))
def topic_model_mem(
    *,
    dtm_csv: PathLike,
    freq_list_csv: PathLike,
    out_features_csv: Optional[PathLike] = None,
    out_model_json: Optional[PathLike] = None,
    out_loadings_csv: Optional[PathLike] = None,
    out_eigenvalues_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
    encoding: str = "utf-8-sig",

    # ----- how the matrix was made. we record these into the model, and the
    # ----- vocab settings also rebuild the term list (checked vs. the matrix)
    lemmatize: bool = False,
    pos_tagged: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    keep_punctuation: bool = False,
    weighting: Literal["count", "binary", "relfreq", "tfidf"] = "count",
    matrix_rounding: int = 4,
    vocab_min_freq: float = 0,
    vocab_min_obs_pct: float = 0,
    vocab_rule: Literal["top_n", "min_obs_pct", "min_freq"] = "top_n",
    vocab_top_n: int = 500,
    vocab_rank_by: Literal["frequency", "obs_pct"] = "frequency",

    # ----- MEM options -----
    n_components: int = 0,
    retain: Literal["parallel", "kaiser"] = "parallel",
    rotation: bool = True,
    rounding: int = 4,
) -> Path:
    """
    Fit MEM themes to a document-term matrix; write scores and a reusable model.

    Parameters
    ----------
    dtm_csv
        A matrix written by
        :func:`taters.text.build_doc_term_matrix.build_doc_term_matrix`. In a
        pipeline this is wired automatically from that step's output.
    freq_list_csv
        The frequency list the matrix was built over. The vocabulary is
        re-derived from it (with the ``vocab_*`` settings below) and checked
        against the matrix's columns, so a mismatch refuses loudly instead of
        modeling the wrong terms.
    out_features_csv : str or pathlib.Path, optional
        Per-document theme scores. Defaults to
        ``./features/topic_model_mem/<dtm filename>``. The model, loadings,
        and eigenvalue files are written next to it unless given their own
        paths (``<name>_model.json``, ``<name>_loadings.csv``,
        ``<name>_eigenvalues.csv``).
    out_model_json, out_loadings_csv, out_eigenvalues_csv : optional
        The reusable model (see :func:`apply_mem_model`), the term-by-theme
        rotated loadings, and the per-theme eigenvalue / %-variance table.
    overwrite_existing : bool, default=False
        If ``False`` and the scores file already exists, skip and return it.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    lemmatize, pos_tagged, engine, tokenizer, stanza_lang, keep_punctuation
        Must match the settings the matrix was built with -- the pipeline
        drives both steps from the same shared variables. Recorded in the
        model so a later apply reads new text the same way. (``pos_tagged``
        also decides how the vocabulary's tagged terms are parsed here.)
    weighting : {"count", "binary", "relfreq", "tfidf"}, default="count"
        The matrix's cell weighting, recorded so apply weights new text the
        same way.
    matrix_rounding : int, default=4
        The ``rounding`` the matrix step used (its default). Only affects
        relfreq/tfidf cells; recorded so apply reproduces them digit for
        digit.
    vocab_rule, vocab_min_freq, vocab_min_obs_pct, vocab_top_n, vocab_rank_by
        The vocabulary rule the matrix was built with, and its setting; used
        to re-derive (and verify) the exact term list, which the model then
        carries. These have to match the matrix step exactly -- the check
        below is what makes a mismatch an error rather than a model fitted
        on the wrong terms -- so ``vocab_rule`` is a shared setting in the
        app, not a per-step one.
    n_components : int, default=0
        How many themes to keep. ``0`` picks automatically by ``retain``;
        an explicit number is honored up to the number of non-constant
        terms.
    retain : {"parallel", "kaiser"}, default="parallel"
        How ``0`` chooses. ``parallel`` is parallel analysis: a theme is
        kept while its eigenvalue beats the 95th percentile of what fifty
        random matrices of the same size produce at the same rank -- the
        eigenvalues chance alone would give -- so a theme has to explain
        more than noise does. ``kaiser`` keeps every eigenvalue above 1,
        the older rule of thumb, which on a wide matrix keeps most of them
        (one real corpus gave 101 themes). Both are generic PCA rules, not
        MEM doctrine; the eigenvalues file and the model record how the
        count was decided, so read them before trusting it.
    rotation : bool, default=True
        Varimax-rotate the components. MEM is defined over rotated loadings;
        turn off only to inspect the raw principal axes.
    rounding : int, default=4
        Decimal places for theme scores, loadings, and eigenvalues.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``token_count``, then
        ``Theme_1..Theme_k``.
    """
    dtm_csv = Path(dtm_csv)
    if not dtm_csv.exists():
        raise FileNotFoundError(f"dtm_csv not found: {dtm_csv}")

    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "topic_model_mem" / dtm_csv.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    stem = out_features_csv.stem
    model_path = Path(out_model_json) if out_model_json else \
        out_features_csv.with_name(f"{stem}_model.json")
    loadings_path = Path(out_loadings_csv) if out_loadings_csv else \
        out_features_csv.with_name(f"{stem}_loadings.csv")
    eigen_path = Path(out_eigenvalues_csv) if out_eigenvalues_csv else \
        out_features_csv.with_name(f"{stem}_eigenvalues.csv")

    if not overwrite_existing and out_features_csv.is_file():
        print("MEM theme scores output file already exists; returning existing file.")
        return out_features_csv

    # 1) first, we re-derive the vocabulary and hold it up against the
    #    matrix's header
    vocab = _load_vocabulary(
        Path(freq_list_csv), encoding=encoding, pos_tagged=pos_tagged,
        vocab_rule=vocab_rule,
        vocab_min_freq=vocab_min_freq, vocab_min_obs_pct=vocab_min_obs_pct,
        vocab_top_n=vocab_top_n, vocab_rank_by=vocab_rank_by,
    )
    terms = sorted(vocab,
                   key=lambda g: (-vocab[g]["frequency"], words_of(g), tags_of(g)))
    columns = column_names(terms, pos_tagged)

    with dtm_csv.open("r", newline="", encoding=encoding) as f:
        header = next(csv.reader(f))
    if header != ["text_id", "token_count", *columns]:
        raise ValueError(
            f"{dtm_csv} does not match the vocabulary derived from "
            f"{freq_list_csv} with these settings (pos_tagged={pos_tagged}, "
            f"vocab_rule={vocab_rule!r}, "
            f"vocab_top_n={vocab_top_n}, vocab_min_freq={vocab_min_freq}, "
            f"vocab_min_obs_pct={vocab_min_obs_pct}, "
            f"vocab_rank_by={vocab_rank_by!r}). The matrix and this step must "
            "agree on the vocabulary, or the model would score the wrong terms."
        )

    # 2) one streaming pass to get the moments, then we find the axes in memory
    warn_if_wide(len(terms))
    n, sums, cross = stream_moments(dtm_csv, encoding=encoding, skip_cols=2,
                                    on_progress=on_progress,
                                    message="measuring the matrix")
    if n < 3:
        raise ValueError(f"Only {n} document(s) in {dtm_csv}; MEM needs a corpus.")
    announce(on_progress, "extracting themes")
    kept, mu, sigma, loadings, projection, theme_eig, pct, retention = fit_axes(
        n, sums, cross, n_components=n_components, rotation=rotation,
        retain=retain, on_progress=on_progress)
    k = projection.shape[1]
    theme_names = [f"Theme_{i + 1}" for i in range(k)]

    kept_set = set(kept.tolist())
    dropped = [columns[j] for j in range(len(terms)) if j not in kept_set]
    if dropped:
        import warnings
        warnings.warn(
            f"{len(dropped)} constant term column(s) carried no signal and "
            f"were left out of the model: {', '.join(dropped[:8])}"
            f"{'…' if len(dropped) > 8 else ''}")

    # 3) the loadings and eigenvalue tables (small, so we do them in memory)
    kept_terms = [terms[j] for j in kept.tolist()]
    with atomic_write(loadings_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        head = ["term", "pos", *theme_names] if pos_tagged else ["term", *theme_names]
        writer.writerow(head)
        for i, gram in enumerate(kept_terms):
            row = [words_of(gram)]
            if pos_tagged:
                row.append(tags_of(gram))
            row.extend(round(float(v), rounding) for v in loadings[i])
            writer.writerow(row)
    with atomic_write(eigen_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["theme", "eigenvalue", "pct_variance"])
        for name, eig, p in zip(theme_names, theme_eig, pct):
            writer.writerow([name, round(float(eig), rounding),
                             round(float(p), rounding)])

    # 4) the model itself: everything apply needs and nothing it doesn't.
    #    `device` is a runtime choice, not part of the instrument, so we do
    #    NOT store it
    from datetime import date

    def _version() -> str:
        try:
            from importlib.metadata import version
            return version("taters")
        except Exception:
            return ""

    model = {
        "kind": "taters-mem-model",
        "format": MODEL_FORMAT,
        "created": date.today().isoformat(),
        "taters": _version(),
        "text": {"lemmatize": lemmatize, "pos_tagged": pos_tagged,
                 "engine": engine, "tokenizer": tokenizer,
                 "stanza_lang": stanza_lang,
                 "keep_punctuation": keep_punctuation},
        "matrix": {"weighting": weighting, "rounding": matrix_rounding,
                   "terms": terms, "columns": columns,
                   "idf": [vocab[g]["idf"] for g in terms]},
        "model": {"n_documents": n, "rotation": bool(rotation),
                  "retention": retention,
                  "themes": theme_names,
                  "kept": kept.tolist(),
                  "mu": mu.tolist(), "sigma": sigma.tolist(),
                  "projection": projection.tolist(),
                  "eigenvalues": theme_eig.tolist(),
                  "pct_variance": pct.tolist()},
    }
    with atomic_write(model_path, encoding="utf-8") as f:
        json.dump(model, f, indent=1)

    # 5) lastly, a second streaming pass: we score every document through the
    #    shared projection helper (the very same one apply uses)
    ticker = Ticker(on_progress, n)
    with dtm_csv.open("r", newline="", encoding=encoding) as f, \
            atomic_write(out_features_csv, newline="", encoding=encoding) as out:
        reader = csv.reader(f)
        next(reader)
        writer = csv.writer(out)
        writer.writerow(["text_id", "token_count", *theme_names])
        for row in reader:
            ticker.tick(message="scoring themes")
            writer.writerow(_theme_row(row[0], row[1], row[2:], kept, mu,
                                       sigma, projection, rounding))

    return out_features_csv


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _load_model(model_json: PathLike) -> dict:
    """
    Load and *vet* a model file: this is the import gate's authority too.

    A model is an instrument someone will publish results from, so a file
    that merely parses is not enough -- the kind tag keeps out other
    procedures' models (a `stats.pca` model file, say), and the structural
    checks keep out a truncated or hand-edited file whose matrices no longer
    agree with each other. Every refusal names what is wrong.
    """
    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"model_json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    kind = model.get("kind")
    if kind != "taters-mem-model":
        what = f"a {kind!r} file" if kind else "not a Taters model file at all"
        raise ValueError(
            f"{path.name} is not a MEM topic model ({what}). Only models "
            "written by the 'Topic model: meaning extraction method' step "
            "can score texts here.")
    if int(model.get("format", 0)) > MODEL_FORMAT:
        raise ValueError(
            f"{path} was written by a newer Taters (model format "
            f"{model.get('format')}; this build reads up to {MODEL_FORMAT}). "
            "Update Taters to use it.")

    def _broken(why: str) -> ValueError:
        return ValueError(f"{path.name} is damaged or incomplete: {why}. "
                          "Re-run the topic-model step to write a fresh one.")

    text = model.get("text")
    matrix = model.get("matrix")
    fit = model.get("model")
    if not (isinstance(text, dict) and isinstance(matrix, dict)
            and isinstance(fit, dict)):
        raise _broken("its text/matrix/model sections are missing")
    for key in ("lemmatize", "pos_tagged", "engine", "tokenizer", "stanza_lang"):
        if key not in text:
            raise _broken(f"the text settings lack {key!r}")
    terms = matrix.get("terms") or []
    if not terms:
        raise _broken("it carries no vocabulary terms")
    if len(matrix.get("idf") or []) != len(terms):
        raise _broken("its idf table does not match its vocabulary")
    kept = fit.get("kept") or []
    themes = fit.get("themes") or []
    projection = fit.get("projection") or []
    if not themes or not kept:
        raise _broken("it names no themes or keeps no terms")
    if max(kept) >= len(terms):
        raise _broken("its kept-term indices point outside the vocabulary")
    if not (len(fit.get("mu") or []) == len(fit.get("sigma") or [])
            == len(kept) == len(projection)):
        raise _broken("its means, deviations and projection disagree in size")
    if any(len(row) != len(themes) for row in projection):
        raise _broken("its projection does not match its theme count")
    return model


@records_settings(
    # the model file is the whole instrument: vocabulary, loadings, and the
    # tokenizer settings the matrix was built with. we compare it by content
    # so that a table scored with a same-named model with different themes
    # gets told apart from one scored with the model a ridge was fitted beside
    binding=TEXT_INPUT, grain=TEXT_GRAIN, assets={"model_json": None},
    outputs=("out_features_csv",), bookkeeping=("token_count",))
def apply_mem_model(
    *,
    model_json: PathLike,

    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,

    # ----- Output -----
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS (used when csv_path is provided) ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,

    # ====== TXT FOLDER GATHER OPTIONS (used when txt_dir is provided) ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== APPLY OPTIONS ======
    device: str = "auto",
    rounding: int = 4,
) -> Path:
    """
    Score new texts on the themes of a saved MEM model.

    The model file (from :func:`topic_model_mem`) carries the vocabulary,
    tokenizer settings, weighting, and projection of the original fit; this
    function rebuilds the same document-term matrix for the new texts --
    through the same scan and weighting code the matrix step uses -- then
    standardizes with the *training* means and deviations and projects.
    Applying a model to its own training texts reproduces the training
    scores exactly.

    Parameters
    ----------
    model_json
        A ``*_model.json`` written by :func:`topic_model_mem`. A folder (or a
        list, as the library hands over) is accepted when it resolves to
        exactly one model file; anything else refuses and says how to pick.
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as the other text analyzers: a spreadsheet of
        texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/topic_model_mem/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and
        return the path.
    workers : int, default=0
        Parallel processes for reading documents during the gather. ``0``
        means automatic: three-quarters of the logical cores.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns that identify each row when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple text columns are provided.
    group_by : Sequence[str] or None, optional
        Optional grouping keys used during CSV gathering.
    delimiter : str, default=","
        Column separator of the *input* CSV.
    pattern : str, default=every document type
        Which files to read when gathering from a folder of documents.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs, if the model was built with the stanza engine --
        a runtime choice, deliberately not stored in the model.
    rounding : int, default=4
        Decimal places for the theme scores.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``token_count``, then one column
        per theme of the model.
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
        out_features_csv = Path.cwd() / "features" / "topic_model_mem" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        print("MEM theme scores output file already exists; returning existing file.")
        return out_features_csv

    # 2) now we rebuild the instrument from the model, exactly as it was trained
    if text_cfg["engine"] == "stanza":
        announce(on_progress, "loading the stanza pipeline (first use "
                              "downloads its model)")
    # a model saved before this setting existed was fitted on a vocabulary
    # that still had its punctuation in it, so a missing key means "keep it"
    keep_punctuation = bool(text_cfg.get("keep_punctuation", True))
    stream = make_token_stream(
        text_cfg["lemmatize"], text_cfg["pos_tagged"],
        engine=text_cfg["engine"], tokenizer=text_cfg["tokenizer"],
        stanza_lang=text_cfg["stanza_lang"], device=device,
        keep_punctuation=keep_punctuation)
    terms: List[str] = list(matrix_cfg["terms"])
    term_index = {g: i for i, g in enumerate(terms)}
    max_n = max(g.count(" ") + 1 for g in terms)
    idf: List[float] = list(matrix_cfg["idf"])
    weighting = matrix_cfg["weighting"]
    matrix_rounding = int(matrix_cfg["rounding"])
    kept = np.asarray(fit["kept"], dtype=int)
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)
    projection = np.asarray(fit["projection"], dtype=np.float64)
    theme_names = list(fit["themes"])

    # 3) scan and weight on the shared pooled-row driver. this is the same
    #    worker the DTM step runs, so new text gets scored through the exact
    #    same code. then we standardize with the TRAINING statistics and
    #    project. stanza-backed models stay single-process (see
    #    `pooled_text_workers`)
    from ..helpers.row_map import map_text_rows
    from .build_doc_term_matrix import _cells_of, _init_scan_worker, _scan_in_worker
    from .ngram_prep import pooled_text_workers

    stream_args = dict(lemmatize=text_cfg["lemmatize"],
                       pos_tagged=text_cfg["pos_tagged"],
                       engine=text_cfg["engine"],
                       tokenizer=text_cfg["tokenizer"],
                       stanza_lang=text_cfg["stanza_lang"], device=device,
                       keep_punctuation=keep_punctuation)
    with atomic_write(out_features_csv, newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(["text_id", "token_count", *theme_names])
        for row, (n_tokens, cells) in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pooled_text_workers(
                    workers, n_rows, engine=text_cfg["engine"],
                    tokenizer=text_cfg["tokenizer"],
                    lemmatize=text_cfg["lemmatize"],
                    pos_tagged=text_cfg["pos_tagged"]),
                message="scoring themes", on_progress=on_progress,
                inline_fn=lambda pair: _cells_of(
                    stream, term_index, max_n, weighting, idf,
                    matrix_rounding, pair[1]),
                pool_fn=_scan_in_worker,
                initializer=_init_scan_worker,
                initargs=(stream_args, terms, weighting, idf, matrix_rounding)):
            writer.writerow(_theme_row(row.get("text_id", ""), n_tokens,
                                       cells, kept, mu, sigma, projection,
                                       rounding))

    return out_features_csv


# --- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"fit": topic_model_mem, "apply": apply_mem_model},
    description='The Meaning Extraction Method: fit themes and save the model, or score new texts with a saved one.',
    aliases={},
    legacy={
        '--no-rotation': ['--rotation', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
