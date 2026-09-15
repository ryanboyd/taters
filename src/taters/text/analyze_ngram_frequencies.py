"""
Corpus-level n-gram frequency list, with collocation statistics.

One output row per retained n-gram (all orders 1..``ngram_n``), not one row
per text: this is a *corpus tool*. "Document" means one analysis-ready row,
so the same unit-of-analysis machinery the other text steps use decides what
NPMI's document counts mean here too.

The statistics follow Bouma (2009), "Normalized (Pointwise) Mutual Information
in Collocation Extraction", with the chain split NPMI((w1..wn-1), wn) for
orders above two. Three corrections over the C# plugin this replaces
(BUTTER's Frequency List), all silent-result bugs rather than crashes:

* Every probability shares **one sample space: the corpus token count** --
  p(g) = freq(g) / N for the n-gram and both of its parts, which is the
  formulation Bouma's bounds are proved for (and what e.g. gensim's phrase
  scorer computes). The plugin normalized each order by its own total
  (bigrams by the bigram count, words by the word count), mixing sample
  spaces -- under which NPMI is not bounded by 1 and "0.9" means nothing.
* Those counts are the **full** counts, taken before any minimum-frequency
  or document-share filter removes rows. Computing totals after filtering
  (as the plugin did) inflates every probability by exactly what was
  filtered, so NPMI values drifted with the user's filter settings.
* Nothing is pruned during counting, so every retained n-gram's subgram
  counts exist and every retained n-gram gets a score. The plugin pruned
  periodically for memory and then silently discarded any n-gram whose
  subgram had been pruned out from under it. Memory is bounded a different
  way: past a RAM budget (``max_ram_mb``) the counting table spills sorted
  batches to disk and merges them once at the end (`helpers.spill_counter`)
  -- the counts stay exact and the output file stays identical.

Filters (minimum frequency, minimum document share, the stoplist, and the
optional NPMI/logDice thresholds) apply only at write time, to output rows --
never to the counts the statistics are computed from.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import (Callable, Dict, List, Literal, NamedTuple, Optional,
                    Sequence, Tuple, Union)

from ..helpers.atomic import atomic_write
from ..helpers.parallel_map import ordered_parallel_map
from ..helpers.progress import FlightReporter, announce, count_rows
from ..helpers.spill_counter import SpillCounter
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from .ngram_prep import (
    iter_ngrams,
    load_stoplist,
    make_token_stream,
    split_for_collocation,
    stoplisted,
    tags_of,
    words_of,
)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: The output columns, in order. Everything BUTTER's plugin wrote, snake_cased:
#: rank (1-based, by descending frequency), the n-gram itself, its order, raw
#: frequency, the number and percent of documents it appears in, ln(N/df), and
#: -- for orders above one -- NPMI and logDice. With ``pos_tagged=True`` a
#: ``pos`` column appears right after ``ngram`` (and only then: an always-empty
#: column is exactly the clutter this file kept out of BUTTER's output).
HEADER = ["rank", "ngram", "phrase_length", "frequency", "documents",
          "obs_pct", "idf", "npmi", "logdice"]


def header_for(pos_tagged: bool) -> List[str]:
    """The header this run will write; ``pos`` exists only when asked for."""
    if not pos_tagged:
        return list(HEADER)
    return HEADER[:2] + ["pos"] + HEADER[2:]


def _doc_grams(stream, text: str, ngram_n: int,
               min_token_count: int) -> Optional[Tuple[int, Dict[str, int]]]:
    """
    One document's contribution: its token count and its gram tallies.

    ``None`` for a document under ``min_token_count`` -- the caller must not
    count it toward any total. This is the unit of work the counting pool
    distributes; everything order-sensitive (folding into the shared counter)
    stays with the caller.
    """
    tokens = stream(text)
    if len(tokens) < min_token_count:
        return None
    counts: Dict[str, int] = {}
    for n in range(1, ngram_n + 1):
        for gram in iter_ngrams(tokens, n):
            counts[gram] = counts.get(gram, 0) + 1
    return len(tokens), counts


#: Per-process state for counting workers: the token stream is built once per
#: worker (spawn cannot pickle a closure over an NLTK tagger), not per text.
_GRAM_WORKER: Dict[str, object] = {}


def _init_gram_worker(stream_args: Dict[str, object], ngram_n: int,
                      min_token_count: int) -> None:
    _GRAM_WORKER["stream"] = make_token_stream(**stream_args)  # type: ignore[arg-type]
    _GRAM_WORKER["ngram_n"] = ngram_n
    _GRAM_WORKER["min_token_count"] = min_token_count


def _grams_in_worker(pair: Tuple[str, str]) -> Optional[Tuple[int, Dict[str, int]]]:
    _tid, text = pair
    return _doc_grams(_GRAM_WORKER["stream"], text,
                      _GRAM_WORKER["ngram_n"], _GRAM_WORKER["min_token_count"])


def _counting_workers(workers: int, total_rows: int, *, engine: str,
                      tokenizer: str, lemmatize: bool, pos_tagged: bool) -> int:
    """
    Processes for the counting pass.

    The Stanza *model* never fans out: a process per worker means a model copy
    per worker in memory, and its lever is batching, not processes -- the same
    rule the runner applies to every model-backed step. That path is engaged
    only when something actually needs the model; plain potts tokenization
    under engine="stanza" short-circuits to potts and pools happily. Small
    corpora stay serial on the automatic setting (`pool_workers`): the pool
    costs more to start than it saves.
    """
    from .ngram_prep import pooled_text_workers

    return pooled_text_workers(workers, total_rows, engine=engine,
                               tokenizer=tokenizer, lemmatize=lemmatize,
                               pos_tagged=pos_tagged)


class _FilterOpts(NamedTuple):
    """Everything the write-time filters need, bundled so the two scoring
    paths (in-memory and spilled-to-disk) cannot drift apart on defaults."""

    n_docs: int
    total_tokens: int
    min_freq: int
    min_obs_pct: float
    stopset: set
    min_npmi: Optional[float]
    min_logdice: Optional[float]


def _collocation_stats(freq: int, left_freq: int, right_freq: int,
                       total_tokens: int) -> Tuple[float, float]:
    """
    NPMI and logDice for one n-gram from raw counts.

    One sample space (the corpus token count) for all three probabilities. An
    order-n n-gram can cover at most 1/n of the tokens, so p_xy <= 1/2 for
    every scored gram and the -log normalizer is always strictly positive.
    """
    p_xy = freq / total_tokens
    p_left = left_freq / total_tokens
    p_right = right_freq / total_tokens
    npmi = math.log(p_xy / (p_left * p_right)) / -math.log(p_xy)
    logdice = 14 + math.log2(2 * freq / (left_freq + right_freq))
    return npmi, logdice


def _passes(gram: str, freq: int, docs: int, opts: _FilterOpts) -> Optional[float]:
    """The cheap write-time filters; the doc share when they pass, else None."""
    obs_pct = docs / opts.n_docs * 100
    if freq < opts.min_freq or obs_pct < opts.min_obs_pct:
        return None
    if stoplisted(gram, opts.stopset):
        return None
    return obs_pct


def _score_in_memory(freqs: Dict[str, List[int]], opts: _FilterOpts) -> list:
    """Score straight off the full table: subgram counts are dict lookups."""
    kept = []
    for gram, (freq, docs) in freqs.items():
        obs_pct = _passes(gram, freq, docs, opts)
        if obs_pct is None:
            continue
        n = gram.count(" ") + 1
        npmi = logdice = None
        if n > 1:
            left, right = split_for_collocation(gram)
            npmi, logdice = _collocation_stats(
                freq, freqs[left][0], freqs[right][0], opts.total_tokens)
            if opts.min_npmi is not None and npmi < opts.min_npmi:
                continue
            if opts.min_logdice is not None and logdice < opts.min_logdice:
                continue
        kept.append((gram, n, freq, docs, obs_pct, npmi, logdice))
    return kept


def _score_streaming(counter: SpillCounter, opts: _FilterOpts) -> list:
    """
    Score without the table in RAM, off the counter's word-ordered stream.

    The chain split means an order-n gram needs exactly two subgram counts:
    its (n-1)-word prefix and its last word. In word order the prefix arrives
    before its extensions and nothing that is not an extension sits between
    them, so prefix counts live on a stack no deeper than ``ngram_n``. Last
    words are unigrams -- the one slice of the vocabulary that stays small --
    served by the counter's indexed lookup.
    """
    right_count = counter.unigram_lookup()
    kept = []
    stack: List[Tuple[str, int]] = []       # (gram, freq) prefixes, shallow -> deep
    for gram, freq, docs in counter.sorted_items():
        while stack and not gram.startswith(stack[-1][0] + " "):
            stack.pop()
        obs_pct = _passes(gram, freq, docs, opts)
        n = gram.count(" ") + 1
        npmi = logdice = None
        if obs_pct is not None and n > 1:
            left, right = split_for_collocation(gram)
            if not stack or stack[-1][0] != left:
                # the stream isn't in the order it promised. if we scored with
                # the wrong prefix count we'd get silently wrong numbers, which
                # is exactly the kind of bug this module is supposed to prevent
                raise RuntimeError(
                    f"prefix {left!r} was not on the stack while scoring "
                    f"{gram!r} -- the merged count stream is out of order")
            npmi, logdice = _collocation_stats(
                freq, stack[-1][1], right_count(right), opts.total_tokens)
            if opts.min_npmi is not None and npmi < opts.min_npmi:
                obs_pct = None
            elif opts.min_logdice is not None and logdice < opts.min_logdice:
                obs_pct = None
        stack.append((gram, freq))
        if obs_pct is not None:
            kept.append((gram, n, freq, docs, obs_pct, npmi, logdice))
    return kept


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  assets={"stoplist_paths": "stoplists"})
def analyze_ngram_frequencies(
    *,
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

    # ====== N-GRAM OPTIONS ======
    ngram_n: int = 1,
    lemmatize: bool = False,
    pos_tagged: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    keep_punctuation: bool = False,
    device: str = "auto",
    stoplist_paths: Optional[Sequence[PathLike]] = None,
    min_freq: int = 5,
    min_obs_pct: float = 0.10,
    min_token_count: int = 10,
    min_npmi: Optional[float] = None,
    min_logdice: Optional[float] = None,
    max_ram_mb: int = 1024,
    rounding: int = 4,
) -> Path:
    """
    Build a corpus frequency list of 1..``ngram_n``-grams and write it as CSV.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as the other text analyzers: a spreadsheet of
        texts, a folder of ``.txt`` files, or a prebuilt analysis-ready CSV.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/ngram_frequencies/<analysis_ready_filename>``.
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
    ngram_n : int, default=1
        Highest n-gram order. The output lists every order from 1 up to this,
        and the lower orders are needed internally for NPMI in any case.
    lemmatize : bool, default=False
        Lemmatize tokens (WordNet, POS-guided) before counting. Always
        applied *before* the stoplist, so "be" catches "is/was/were".
    pos_tagged : bool, default=False
        Treat word+tag as the unit: the verb "felt" and the noun "felt"
        become separate rows, and a ``pos`` column (after ``ngram``) names
        each term's tag(s). Off, the column does not exist at all.
    stoplist_paths : sequence of paths, optional
        Stop word/character lists (.txt, one entry per line; folders are
        expanded). None means a vanilla list. Applied to finished n-grams --
        a row is dropped when any of its tokens is a stop entry -- never to
        the token stream, so counts and NPMI keep their true values.
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
    min_freq : int, default=5
        Drop n-grams rarer than this from the output.
    min_obs_pct : float, default=0.10
        Drop n-grams appearing in fewer than this *percent* of documents.
    min_token_count : int, default=10
        Skip documents shorter than this many tokens entirely (they do not
        count toward document totals either).
    min_npmi, min_logdice : float, optional
        Optional collocation thresholds for orders above one. Default None:
        the metrics are reported and filtering stays an analysis decision.
    max_ram_mb : int, default=1024
        Approximate RAM budget (in MB) for the n-gram counting table. A
        corpus whose vocabulary outgrows it has sorted batches cached to a
        temporary folder and merged once at the end: the output file is
        identical, memory stays bounded, and the disk is touched in a few
        big sequential passes rather than per gram. A small corpus never
        reaches the budget and never notices; raise it on a big machine for
        speed on a huge corpus.
    workers : int, default=0
        Parallel processes, spent twice: reading documents during the gather,
        then tokenizing-and-tallying during the count. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off.
        Output files are identical whatever the worker count. Counting under
        a Stanza model stays single-process (its lever is batching, not
        processes). Each in-flight document holds its gram tallies in memory,
        so many workers over very large documents costs RAM beyond
        ``max_ram_mb``'s table.
    rounding : int, default=4
        Decimal places for the derived statistics.

    Returns
    -------
    Path
        ``out_features_csv``, sorted by descending frequency (ties broken
        alphabetically), rank assigned after all filters.

    Raises
    ------
    ValueError
        If no document met ``min_token_count`` -- an empty frequency list is
        indistinguishable from a misconfigured run, so it refuses instead.
    """
    if ngram_n < 1:
        raise ValueError(f"ngram_n must be >= 1, got {ngram_n}")

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
        out_features_csv = Path.cwd() / "features" / "ngram_frequencies" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print("N-gram frequency output file already exists; returning existing file.")
        return out_features_csv

    if engine == "stanza" and (lemmatize or pos_tagged or tokenizer == "stanza"):
        # building the stanza pipeline might download its model, so we name
        # the phase. otherwise the display just sits on the previous one
        # while that happens
        announce(on_progress, "loading the stanza pipeline (first use "
                              "downloads its model)")
    stream = make_token_stream(lemmatize, pos_tagged, engine=engine,
                               keep_punctuation=keep_punctuation,
                               tokenizer=tokenizer,
                               stanza_lang=stanza_lang, device=device)
    stopset = load_stoplist(stoplist_paths) if stoplist_paths else set()

    # 2) one pass over everything: we count every n-gram of every order, plus
    #    the corpus token count that every probability hangs off of. no pruning
    #    here, whatever the filters say -- see the module docstring for why.
    #    the counter keeps the table in RAM up to `max_ram_mb` and spills
    #    sorted batches to disk past that; either way, the counts are exact
    counter = SpillCounter(
        max(0, int(max_ram_mb)) * 1024 * 1024, tmp_root=tmp_root,
        on_spill=lambda batch: announce(
            on_progress, f"cached n-gram counts to disk (batch {batch})"),
    )
    total_tokens = 0
    n_docs = 0

    try:
        # tokenizing is the expensive (and embarrassingly parallel) half, so
        # that and the per-document tallying go to the pool. folding into the
        # shared counter stays here, in submission order -- that's what keeps
        # the output identical no matter the worker count, spills included
        total_rows = count_rows(analysis_ready, on_progress=on_progress)
        count_workers = _counting_workers(workers, total_rows, engine=engine,
                                          tokenizer=tokenizer,
                                          lemmatize=lemmatize,
                                          pos_tagged=pos_tagged)
        stream_args = dict(lemmatize=lemmatize, pos_tagged=pos_tagged,
                           engine=engine, tokenizer=tokenizer,
                           stanza_lang=stanza_lang, device=device,
                           keep_punctuation=keep_punctuation)
        reporter = FlightReporter(on_progress, total_rows, "counting n-grams")
        with analysis_ready.open("r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if "text_id" not in fields or "text" not in fields:
                raise ValueError(
                    f"Expected columns 'text_id' and 'text' in {analysis_ready}; found {fields}"
                )

            def pairs():
                for row in reader:
                    yield (str(row.get("text_id", "") or ""),
                           str(row.get("text", "") or ""))

            for result in ordered_parallel_map(
                    lambda pair: _doc_grams(stream, pair[1], ngram_n,
                                            min_token_count),
                    pairs(),
                    workers=count_workers,
                    pool_fn=_grams_in_worker,
                    initializer=_init_gram_worker,
                    initargs=(stream_args, ngram_n, min_token_count),
                    on_start=lambda pair: reporter.start(pair[0]),
                    on_finish=lambda pair: reporter.finish(pair[0])):
                reporter.consumed()
                if result is None:
                    continue
                n_tokens, doc_counts = result
                n_docs += 1
                total_tokens += n_tokens
                counter.add_doc(doc_counts)

        if n_docs == 0:
            raise ValueError(
                f"No document reached min_token_count={min_token_count}; "
                "an empty frequency list would look like a run that worked."
            )

        # 3) score and filter. the statistics always come from the full counts
        #    above; the filters only pick which rows make it to the file. both
        #    paths share the same arithmetic (`_collocation_stats`) and give
        #    the same rows -- the disk path just looks up subgram counts
        #    without holding the full table in memory
        opts = _FilterOpts(n_docs=n_docs, total_tokens=total_tokens,
                           min_freq=min_freq, min_obs_pct=min_obs_pct,
                           stopset=stopset, min_npmi=min_npmi,
                           min_logdice=min_logdice)
        if counter.spilled:
            announce(on_progress, "merging n-gram counts cached on disk")
            kept = _score_streaming(counter, opts)
        else:
            kept = _score_in_memory(counter.data, opts)
    finally:
        counter.close()

    # ties break on the words (then the tags), so that "felt (NN)" and
    # "felt (VBD)" at the same frequency land in the same order every run
    kept.sort(key=lambda row: (-row[2], words_of(row[0]), tags_of(row[0])))

    with atomic_write(out_features_csv, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header_for(pos_tagged))
        for rank, (gram, n, freq, docs, obs_pct, npmi, logdice) in enumerate(kept, 1):
            row = [
                rank, words_of(gram), n, freq, docs,
                round(obs_pct, rounding),
                round(math.log(n_docs / docs), rounding),
                "" if npmi is None else round(npmi, rounding),
                "" if logdice is None else round(logdice, rounding),
            ]
            if pos_tagged:
                row.insert(2, tags_of(gram))
            writer.writerow(row)

    return out_features_csv


# --- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_ngram_frequencies,
    description='Build a corpus n-gram frequency list with NPMI/logDice.',
    aliases={
        'stoplist_paths': ['--stoplist'],
    },
    legacy={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
