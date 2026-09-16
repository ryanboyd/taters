"""
Part-of-speech features, one row per document -- the content coder's shape.

Each document is tokenized (happierfuntokenizing, the same tokenizer as the
n-gram tools), POS-tagged with NLTK, and summarized as one column per tag:
relative frequencies by default, raw counts on request. Syntactic n-grams --
sequences of tags, "DT_NN", "PRP_VBD_JJ" -- are available the same way: all
orders from 1 up to ``sngram_n``, each order normalized by its own number of
windows in the document, so within a document every order's columns sum to 1.

Two tag sets: Penn Treebank (``tagset="penn"``, NLTK's native ~45 tags) and
the Universal tagset (12 coarse categories: NOUN, VERB, ADJ, ...), which is
often the better unit for psychological work.

The tagged text itself can also be written out (``tagged_text_csv``), one row
per document with tokens as ``word_TAG`` -- off by default, for the rare
downstream that wants the tags rather than the summary.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.feature_columns import ColumnSpec
from ..helpers.text_gather import (resolve_analysis_ready)
from .ngram_prep import iter_ngrams, make_tagged_stream
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: Every column is a part-of-speech tag, or an n-gram of them, under a `pos_`
#: prefix -- `pos_NN`, `pos_DT_NN`. Which tags appear depends on the corpus and
#: the tagger, so the prefix is the declarable part and it is what keeps these
#: out of everybody else's way.
FEATURE_COLUMNS = ColumnSpec(
    label="Parts of speech",
    # `{*}` and not `{n}`: the tail is a tag, `pos_NN` and `pos_DT_NN`, so the
    # numeric placeholder matched none of the columns this module writes -- and
    # a pattern matching nothing cannot catch a future measure colliding with
    # it, which is the one job the registry has.
    patterns=("pos_{*}",),
    dynamic="one column per tag or tag n-gram actually seen in the corpus",
)


# the tagging path lives over in ngram_prep (one factory for both engines) so
# that the POS features, the frequency list and the DTM can't drift apart in
# how they read text


def _tag_counts(tag, sngram_n: int, keep_tagged: bool, text: str):
    """One document's tagging: (tag_count, sequence Counter, tagged text or
    None). The single tagger both the inline path and the workers run."""
    pairs = tag(str(text or ""))
    tags = [t for _, t in pairs]
    counts: Counter = Counter()
    for n in range(1, sngram_n + 1):
        counts.update((n, gram) for gram in iter_ngrams(tags, n))
    tagged_text = " ".join(f"{w}_{t}" for w, t in pairs) if keep_tagged else None
    return len(tags), counts, tagged_text


#: Per-process state for tagging workers: the tagger is built once per worker
#: (spawn cannot pickle a closure over an NLTK tagger), not per text.
_TAG_WORKER: Dict[str, object] = {}


def _init_tag_worker(tag_args: Dict[str, object], sngram_n: int,
                     keep_tagged: bool) -> None:
    _TAG_WORKER["tag"] = make_tagged_stream(**tag_args)  # type: ignore[arg-type]
    _TAG_WORKER["sngram_n"] = sngram_n
    _TAG_WORKER["keep_tagged"] = keep_tagged


def _tags_in_worker(pair):
    _tid, text = pair
    return _tag_counts(_TAG_WORKER["tag"], _TAG_WORKER["sngram_n"],
                       _TAG_WORKER["keep_tagged"], text)


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  # `tagged_text_csv` is a second output path, not a setting:
                  # it just says where the tagged text goes
                  outputs=("out_features_csv", "tagged_text_csv"),
                  # our columns are whichever tags (and tag n-grams) actually
                  # show up. so, a tag that no text in a second corpus uses
                  # has no column there, and a model that knows that tag
                  # should read it as zero -- because that's what the count
                  # was. if we refused instead, a trained model couldn't score
                  # a new study just because one particle column was missing
                  absent_means_zero=True,
                  bookkeeping=("token_count",))
def analyze_parts_of_speech(
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

    # ====== POS OPTIONS ======
    tagset: Literal["penn", "universal"] = "penn",
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    device: str = "auto",
    relative_freq: bool = True,
    sngram_n: int = 1,
    tagged_text_csv: Optional[PathLike] = None,
    rounding: int = 4,
) -> Path:
    """
    Tag every document and write one row of POS features per document.

    Parameters
    ----------
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/pos/<analysis_ready_filename>``.
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
    tagset : {"penn", "universal"}, default="penn"
        Penn Treebank's ~45 tags, or the Universal tagset's 12 coarse ones.
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
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs: "auto", "cuda", or "cpu". Only with engine="stanza";
        the NLTK engine is CPU-only either way.
    relative_freq : bool, default=True
        Each order-n column is that sequence's count divided by the number of
        length-n windows in the document, so an order's columns sum to 1 per
        document. False writes raw counts.
    sngram_n : int, default=1
        Highest syntactic-n-gram order. All orders from 1 up are written;
        an order-2 column reads ``pos_DT_NN``.
    tagged_text_csv : path, optional
        Also write the tagged text itself -- ``text_id, tagged_text`` with
        tokens as ``word_TAG`` -- for the rare downstream that wants the tags
        rather than the summary. Off (None) by default.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.
    rounding : int, default=4
        Decimal places for the derived values.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, any pass-through ``id_cols``,
        ``token_count``, then one ``pos_*`` column per observed tag sequence
        (orders ascending, alphabetical within an order).
    """
    if sngram_n < 1:
        raise ValueError(f"sngram_n must be >= 1, got {sngram_n}")

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
        out_features_csv = Path.cwd() / "features" / "pos" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print("POS features output file already exists; returning existing file.")
        return out_features_csv

    if engine == "stanza":
        announce(on_progress, "loading the stanza pipeline (first use "
                              "downloads its model)")
    tag = make_tagged_stream(engine=engine, tokenizer=tokenizer,
                             tagset=tagset, stanza_lang=stanza_lang,
                             device=device)   # this is the inline path's tagger


    # 2) one pooled pass: tag and count, with results in file order and one
    #    live sub-bar per document in flight. we hold the rows in memory
    #    afterwards because the column set is the union of every sequence any
    #    document produced -- we can't know that until we've read the last
    #    document. stanza tagging stays single-process (see
    #    `pooled_text_workers`); NLTK pools just fine
    from ..helpers.row_map import map_text_rows
    from .ngram_prep import pooled_text_workers

    # there's one shared rule for what rides along beside text_id -- see
    # resolve_passthrough_columns. it also does the header check that
    # map_text_rows can't do for us
    from ..helpers.row_map import resolve_passthrough_columns

    with analysis_ready.open("r", newline="", encoding=encoding) as f:
        fields = csv.DictReader(f).fieldnames or []
    passthrough = resolve_passthrough_columns(
        fields, id_cols=id_cols, group_by=group_by,
        analysis_ready=analysis_ready)

    tag_args = dict(engine=engine, tokenizer=tokenizer, tagset=tagset,
                    stanza_lang=stanza_lang, device=device)
    docs: List[tuple] = []       # (text_id, meta, token_count, Counter)
    tagged_rows: List[tuple] = []
    for row, (n_tags, counts, tagged_text) in map_text_rows(
            analysis_ready, encoding=encoding,
            workers=lambda n_rows: pooled_text_workers(
                workers, n_rows, engine=engine, tokenizer=tokenizer,
                lemmatize=False, pos_tagged=True),
            message="tagging documents", on_progress=on_progress,
            inline_fn=lambda pair: _tag_counts(
                tag, sngram_n, tagged_text_csv is not None, pair[1]),
            pool_fn=_tags_in_worker,
            initializer=_init_tag_worker,
            initargs=(tag_args, sngram_n, tagged_text_csv is not None)):
        meta = {c: str(row.get(c, "") or "") for c in passthrough}
        docs.append((str(row.get("text_id", "") or ""), meta, n_tags, counts))
        if tagged_text_csv is not None:
            tagged_rows.append((str(row.get("text_id", "") or ""), tagged_text))

    # 3) the column plan: orders ascending, alphabetical within an order
    seen = sorted({key for *_ , counts in docs for key in counts})
    columns = [(n, gram, "pos_" + gram.replace(" ", "_")) for n, gram in seen]

    with atomic_write(out_features_csv, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", *passthrough, "token_count",
                         *(name for *_ , name in columns)])
        for text_id, meta, token_count, counts in docs:
            cells = []
            for n, gram, _name in columns:
                value = counts.get((n, gram), 0)
                if relative_freq:
                    windows = max(token_count - n + 1, 0)
                    value = round(value / windows, rounding) if windows else 0
                cells.append(value)
            writer.writerow([text_id, *(meta[c] for c in passthrough),
                             token_count, *cells])

    if tagged_text_csv is not None:
        tagged_text_csv = Path(tagged_text_csv)
        tagged_text_csv.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(tagged_text_csv, newline="", encoding=encoding) as f:
            writer = csv.writer(f)
            writer.writerow(["text_id", "tagged_text"])
            writer.writerows(tagged_rows)

    return out_features_csv


# --- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_parts_of_speech,
    description='Part-of-speech features per document (NLTK tagger).',
    aliases={},
    legacy={
        '--counts': ['--relative-freq', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
