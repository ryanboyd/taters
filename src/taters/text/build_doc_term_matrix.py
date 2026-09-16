"""
Document-term matrix built over a vocabulary taken from a frequency list.

The companion step to :mod:`analyze_ngram_frequencies`, and the on-ramp for
the topic-modeling steps planned behind it: one row per document, one column
per retained term, cells weighted as counts, binary, relative frequency, or
TF-IDF. "Document" means one analysis-ready row, exactly as in the frequency
list, so the two steps agree about units by construction.

Two agreements with the frequency list are load-bearing:

* **The token stream must match.** The vocabulary was built from prepared
  (possibly lemmatized) tokens; scanning unprepared text against it fails
  *silently* -- near-zero matches, no error. Both steps therefore share
  :mod:`taters.text.ngram_prep`, and the pipeline recipes drive both steps'
  ``lemmatize`` from one shared variable. (The stoplist needs no such
  agreement: the vocabulary already encodes it.)
* **Longest match wins.** Scanning prefers the highest-order n-gram at each
  position and consumes its tokens: "I study health behaviors" scores
  "health behaviors" once and "health" zero times, not both. A frequency
  list counts every window; a DTM must not double-count nested terms.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from .ngram_prep import TAG_SEP, make_token_stream, tags_of, words_of
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec

FEATURE_COLUMNS = ColumnSpec(
    label="Document-term matrix",
    dynamic="one column per vocabulary term, so the corpus names them",
)

PathLike = Union[str, Path]


def _tagged_key(words: str, tags: str) -> str:
    """Rebuild the internal tagged form ("felt\\x1fVBD") from the two columns."""
    return " ".join(f"{w}{TAG_SEP}{t}"
                    for w, t in zip(words.split(" "), tags.split(" ")))


def _load_vocabulary(
    freq_list_csv: Path,
    *,
    encoding: str,
    pos_tagged: bool,
    vocab_rule: str,
    vocab_min_freq: float,
    vocab_min_obs_pct: float,
    vocab_top_n: int,
    vocab_rank_by: str,
) -> Dict[str, dict]:
    """
    Read the frequency list and subset it to the DTM's vocabulary.

    Columns are read by name (the C# plugin read them by position, which
    broke the moment the list gained or lost a column). ``vocab_top_n`` keeps
    ties at the cut-off -- everything whose ranking value equals the Nth
    value survives -- so the choice between tied terms is never arbitrary.

    A POS mismatch between the two steps is refused up front, both ways: a
    tagged vocabulary scanned without tags (or vice versa) fails *silently*
    otherwise -- near-zero matches, no error -- which is the worst outcome.
    """
    rows: Dict[str, dict] = {}
    with Path(freq_list_csv).open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        needed = {"ngram", "frequency", "obs_pct", "idf"}
        missing = sorted(needed - set(fields))
        if missing:
            raise ValueError(
                f"{freq_list_csv} does not look like a Taters n-gram frequency "
                f"list: missing column(s) {', '.join(missing)}."
            )
        has_pos = "pos" in fields
        if pos_tagged and not has_pos:
            raise ValueError(
                f"pos_tagged=True, but {freq_list_csv} has no 'pos' column -- "
                "the frequency list was built without pos_tagged. The two "
                "steps must agree, or scanning quietly matches nothing."
            )
        if has_pos and not pos_tagged:
            raise ValueError(
                f"{freq_list_csv} carries a 'pos' column (built with "
                "pos_tagged=True), but this run has pos_tagged=False. The two "
                "steps must agree, or scanning quietly matches nothing."
            )
        for row in reader:
            gram = row["ngram"]
            if not gram:
                continue
            key = _tagged_key(gram, row["pos"]) if pos_tagged else gram
            if key not in rows:
                rows[key] = {
                    "frequency": float(row["frequency"]),
                    "obs_pct": float(row["obs_pct"]),
                    "idf": float(row["idf"]),
                }

    # ONE rule decides the vocabulary. we used to have three that all applied
    # at once -- a minimum frequency, a minimum percent-of-documents, and a
    # top N -- plus a fourth setting that only mattered for the third. working
    # out what a corpus would be left with meant intersecting all of them in
    # your head, and the screen gave no hint that they even combined ("it gets
    # super confusing to try to decide how all three of those can intersect
    # with each other"). picking the rule and then its number gives you the
    # same power with none of that arithmetic
    if vocab_rule not in _VOCAB_RULES:
        raise ValueError(
            f"vocab_rule must be one of {sorted(_VOCAB_RULES)}, got "
            f"{vocab_rule!r}.")

    if vocab_rule == "min_freq":
        kept = {g: r for g, r in rows.items()
                if r["frequency"] >= vocab_min_freq}
        applied = f"frequency >= {vocab_min_freq}"
    elif vocab_rule == "min_obs_pct":
        kept = {g: r for g, r in rows.items()
                if r["obs_pct"] >= vocab_min_obs_pct}
        applied = f"appears in >= {vocab_min_obs_pct}% of documents"
    else:
        key = "obs_pct" if vocab_rank_by == "obs_pct" else "frequency"
        kept = dict(rows)
        if vocab_top_n and len(kept) > vocab_top_n:
            # we keep ties at the cut-off, so that the choice between two terms
            # with identical standing is never left up to sort order
            cutoff = sorted((r[key] for r in kept.values()),
                            reverse=True)[vocab_top_n - 1]
            kept = {g: r for g, r in kept.items() if r[key] >= cutoff}
        applied = f"top {vocab_top_n} by {key}"

    if not kept:
        raise ValueError(
            f"No term in {freq_list_csv} survived the vocabulary rule "
            f"({applied}). Loosen it, or pick a different vocab_rule.")
    return kept


#: The vocabulary rules, exactly one of which applies to a run.
_VOCAB_RULES = ("top_n", "min_obs_pct", "min_freq")


def column_names(terms: Sequence[str], pos_tagged: bool) -> list:
    """
    Display names for the term columns, unique after the two reserved ones.

    "text_id" is a perfectly plausible *token*, and left as-is it duplicated
    the header -- reading the file back by name then returned the term's count
    where the document id should be. Colliding terms get a trailing underscore
    (repeatedly, in case that name is somehow taken too). Tagged terms read
    "felt (VBD)", which also keeps the verb and the noun apart as columns.
    """
    used = {"text_id", "token_count"}
    columns = []
    for gram in terms:
        name = f"{words_of(gram)} ({tags_of(gram)})" if pos_tagged else gram
        while name in used:
            name += "_"
        used.add(name)
        columns.append(name)
    return columns


def scan_tokens(tokens: Sequence[str], term_index: Dict[str, int],
                max_n: int) -> list:
    """
    Raw match counts for one document, longest match first.

    Matched tokens are consumed, so a nested term is not also counted
    ("health behaviors" beats "health"). The C# plugin's window guard was off
    by one and could never match an n-gram that ran to the end of the
    document; ``i + n <= len`` can. Shared with the MEM topic model's apply
    path, so a saved model scores new text with the *same* scan, not a copy.
    """
    scores = [0.0] * len(term_index)
    i = 0
    while i < len(tokens):
        for n in range(min(max_n, len(tokens) - i), 0, -1):
            gram = " ".join(tokens[i:i + n])
            idx = term_index.get(gram)
            if idx is not None:
                scores[idx] += 1
                i += n
                break
        else:
            i += 1
    return scores


def weight_scores(scores: list, weighting: str, n_tokens: int,
                  idf: Sequence[float], rounding: int) -> list:
    """Turn raw match counts into the requested cell values."""
    if weighting == "binary":
        return [1 if s else 0 for s in scores]
    if weighting == "relfreq":
        return [round(s / n_tokens, rounding) if n_tokens else 0
                for s in scores]
    if weighting == "tfidf":
        return [round(s * idf[j], rounding) for j, s in enumerate(scores)]
    return [int(s) for s in scores]


def _cells_of(stream, term_index: Dict[str, int], max_n: int, weighting: str,
              idf: Sequence[float], rounding: int, text: str):
    """One document's row: (token_count, weighted cells). The single scorer
    both the inline path and the worker processes run -- and the MEM topic
    model's apply path imports it, so a saved model scores new text through
    exactly this code."""
    tokens = stream(text)
    return len(tokens), weight_scores(scan_tokens(tokens, term_index, max_n),
                                      weighting, len(tokens), idf, rounding)


#: Per-process state for scan workers: the token stream and vocabulary are
#: built once per worker (spawn cannot pickle the stream's closures).
_SCAN_WORKER: Dict[str, object] = {}


def _init_scan_worker(stream_args: Dict[str, object], terms: Sequence[str],
                      weighting: str, idf: Sequence[float],
                      rounding: int) -> None:
    _SCAN_WORKER["stream"] = make_token_stream(**stream_args)  # type: ignore[arg-type]
    _SCAN_WORKER["term_index"] = {g: i for i, g in enumerate(terms)}
    _SCAN_WORKER["max_n"] = max(g.count(" ") + 1 for g in terms)
    _SCAN_WORKER["weighting"] = weighting
    _SCAN_WORKER["idf"] = list(idf)
    _SCAN_WORKER["rounding"] = rounding


def _scan_in_worker(pair):
    _tid, text = pair
    s = _SCAN_WORKER
    return _cells_of(s["stream"], s["term_index"], s["max_n"], s["weighting"],
                     s["idf"], s["rounding"], text)


@records_settings(
    # `freq_list_csv` is the vocabulary we scanned this matrix against, and
    # the chain walk follows it: the frequency list's own settings (lemmatize,
    # the stoplists, the vocabulary rule) decide these numbers, and we don't
    # restate them here.
    # ... and it's an *asset* rather than a binding, because the vocabulary is
    # the instrument: two matrices scanned against different lists have
    # different columns, and a model fitted on one can't be scored on the
    # other. since we record it by content, the list travels inside any model
    # fitted on this matrix, and a later corpus gets scanned against the same
    # words
    binding=TEXT_INPUT, grain=TEXT_GRAIN,
    assets={"freq_list_csv": None},
    outputs=("out_features_csv",),
    # how many tokens the row had is just bookkeeping, not a term count
    bookkeeping=("token_count",))
def build_doc_term_matrix(
    *,
    # ----- The vocabulary -----
    freq_list_csv: PathLike,

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

    # ====== DTM OPTIONS ======
    lemmatize: bool = False,
    pos_tagged: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    keep_punctuation: bool = False,
    device: str = "auto",
    weighting: Literal["count", "binary", "relfreq", "tfidf"] = "count",
    vocab_min_freq: float = 0,
    vocab_min_obs_pct: float = 0,
    vocab_rule: str = "top_n",
    vocab_top_n: int = 500,
    vocab_rank_by: Literal["frequency", "obs_pct"] = "frequency",
    rounding: int = 4,
) -> Path:
    """
    Score every document against a frequency-list vocabulary; write a wide CSV.

    Parameters
    ----------
    freq_list_csv
        A frequency list written by
        :func:`taters.text.analyze_ngram_frequencies.analyze_ngram_frequencies`.
        In a pipeline this is wired automatically from that step's output.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/doc_term_matrix/<analysis_ready_stem>_<weighting>.csv``
        -- the weighting is part of the name, so rerunning with a different
        one writes a sibling file instead of skipping or clobbering.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and
        return the path. Also controls whether the intermediate analysis-ready
        CSV is rebuilt from the current source or reused.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns that identify each row when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple text columns are provided:
        ``"concat"`` joins them into one text per row; ``"separate"`` measures
        each column on its own.
    group_by : Sequence[str] or None, optional
        Optional grouping keys used during CSV gathering (e.g. ``["speaker"]``)
        -- one document per group instead of one per row.
    delimiter : str, default=","
        Column separator of the *input* CSV.
    pattern : str, default=every document type
        Which files to read when gathering from a folder of documents
        (globs, ``;``-separated). Only used with ``txt_dir``.
    lemmatize : bool, default=False
        Must match the frequency list's setting -- the pipeline drives both
        from one shared variable for exactly this reason.
    pos_tagged : bool, default=False
        Must also match the frequency list (checked against its ``pos``
        column, both ways). Tagged term columns read "felt (VBD)".
    engine : {"nltk", "stanza"}, default="nltk"
        Who tags: NLTK's perceptron tagger, or Stanza's neural pipeline.
        Stanza is slower, more accurate, multilingual, and GPU-optional.
    tokenizer : {"potts", "stanza"}, default="potts"
        Who splits text into tokens. The default (the Potts social-media
        tokenizer) keeps counts comparable across engines and keeps
        emoticons, hashtags and URLs whole; "stanza" (Stanza engine only)
        hands Stanza the whole job.
    stanza_lang : str, default="en"
        Language for the Stanza engine; that language's model is downloaded
        once on first use (can be a few hundred MB). Only with engine="stanza".
    keep_punctuation : bool, default=False
        Count punctuation (and emoticons) as terms. Off, only tokens with a
        letter or digit are counted, so "." and "," never become vocabulary.
        Must match across the frequency list, the matrix and the topic model.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs: "auto", "cuda", or "cpu". Only with engine="stanza";
        the NLTK engine is CPU-only either way.
    weighting : {"count", "binary", "relfreq", "tfidf"}, default="count"
        Cell values: raw matches; 0/1; matches over the document's token
        count; or matches times the term's IDF from the frequency list.
        Counts are the neutral default and what topic models want.
    vocab_rule : {"top_n", "min_obs_pct", "min_freq"}, default="top_n"
        Which single rule decides the vocabulary. Exactly one applies, and
        the setting below that belongs to it is the only one read:

        * ``top_n`` -- keep the ``vocab_top_n`` highest-ranked terms.
          The usual choice: it fixes the width of the matrix, so you know
          what you are getting whatever the corpus looks like.
        * ``min_obs_pct`` -- keep terms appearing in at least
          ``vocab_min_obs_pct`` percent of documents. Use it when you want
          terms that are *widespread* rather than merely common, which is
          the right idea for a topic model.
        * ``min_freq`` -- keep terms used at least ``vocab_min_freq`` times
          in total.
    vocab_top_n : int, default=500
        How many terms to keep, when ``vocab_rule`` is ``"top_n"``. Zero
        means no limit. Ties at the cut-off are kept, so the choice between
        two equally-ranked terms is never arbitrary.
    vocab_min_obs_pct : float, default=0
        The percentage of documents a term must appear in, when
        ``vocab_rule`` is ``"min_obs_pct"``.
    vocab_min_freq : float, default=0
        The total number of uses a term must have, when ``vocab_rule`` is
        ``"min_freq"``.
    vocab_rank_by : {"frequency", "obs_pct"}, default="frequency"
        What ``top_n`` ranks by. Ignored by the other two rules.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.
    rounding : int, default=4
        Decimal places for the derived values.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``token_count``, then one column
        per term, in descending frequency-list order.
    """
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
        # the weighting goes in the FILENAME, not just the cells. with a bare
        # name, rerunning with a different weighting either handed back the
        # old matrix untouched (overwrite off -- silently the wrong numbers)
        # or destroyed it (overwrite on). this way the four matrices of one
        # corpus can all live side by side
        out_features_csv = (Path.cwd() / "features" / "doc_term_matrix"
                            / f"{analysis_ready.stem}_{weighting}.csv")
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print("Doc-term matrix output file already exists; returning existing file.")
        return out_features_csv

    vocab = _load_vocabulary(
        Path(freq_list_csv),
        encoding=encoding,
        pos_tagged=pos_tagged,
        vocab_min_freq=vocab_min_freq,
        vocab_min_obs_pct=vocab_min_obs_pct,
        vocab_rule=vocab_rule,
        vocab_top_n=vocab_top_n,
        vocab_rank_by=vocab_rank_by,
    )
    # column order: descending frequency, ties alphabetical -- same order the
    # frequency list is written in, so that the two files read side by side
    terms = sorted(vocab,
                   key=lambda g: (-vocab[g]["frequency"], words_of(g), tags_of(g)))
    term_index = {g: i for i, g in enumerate(terms)}
    max_n = max(g.count(" ") + 1 for g in terms)

    columns = column_names(terms, pos_tagged)
    idf = [vocab[g]["idf"] for g in terms]

    if engine == "stanza" and (lemmatize or pos_tagged or tokenizer == "stanza"):
        announce(on_progress, "loading the stanza pipeline (first use "
                              "downloads its model)")
    stream = make_token_stream(lemmatize, pos_tagged, engine=engine,
                               tokenizer=tokenizer,
                               stanza_lang=stanza_lang, device=device,
                               keep_punctuation=keep_punctuation)

    # 2) scan and write, one document per row, on the shared pooled-row
    #    driver. that gives us results in file order no matter the worker
    #    count, plus one live sub-bar per document in flight. stanza-backed
    #    streams stay single-process (see `pooled_text_workers`)
    from ..helpers.row_map import map_text_rows
    from .ngram_prep import pooled_text_workers

    stream_args = dict(lemmatize=lemmatize, pos_tagged=pos_tagged,
                       engine=engine, tokenizer=tokenizer,
                       stanza_lang=stanza_lang, device=device,
                       keep_punctuation=keep_punctuation)
    with atomic_write(out_features_csv, newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(["text_id", "token_count", *columns])
        for row, (n_tokens, scores) in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pooled_text_workers(
                    workers, n_rows, engine=engine, tokenizer=tokenizer,
                    lemmatize=lemmatize, pos_tagged=pos_tagged),
                message="building the matrix", on_progress=on_progress,
                inline_fn=lambda pair: _cells_of(
                    stream, term_index, max_n, weighting, idf, rounding,
                    pair[1]),
                pool_fn=_scan_in_worker,
                initializer=_init_scan_worker,
                initargs=(stream_args, terms, weighting, idf, rounding)):
            writer.writerow([row.get("text_id", ""), n_tokens, *scores])

    return out_features_csv


# --- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    build_doc_term_matrix,
    description='Build a document-term matrix from a Taters n-gram frequency list.',
    aliases={},
    legacy={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
