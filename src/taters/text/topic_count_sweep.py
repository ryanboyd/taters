# taters/text/topic_count_sweep.py
"""
How many topics? Fit several, score them, and look at the curve.

The one setting a topic model cannot pick for you
-------------------------------------------------
Everything else about LDA and NMF has a defensible default. The number of
topics does not. Ask for five and you get broad ones ("food", "work"); ask for
fifty on the same corpus and you get narrow ones ("breakfast", "restaurant
complaints"). Neither is wrong -- they are different questions -- and no
statistic can tell you which question you meant to ask.

What this does is narrower and still useful: fit the model at each number you
name, score how well each one's topics hang together, and write the curve out
with the top words for every fit. Then you read them.

Topic coherence, and what it is not
-----------------------------------
Coherence asks whether the words that define a topic actually turn up in the
same documents. A topic of *bread, butter, cheese* scores well because those
co-occur; a topic of *bread, deadline, quarterly* scores badly because they do
not. That correlates with topics a person would call meaningful, and it is not
the same thing -- a model can score beautifully and still carve the corpus
somewhere useless.

So this reports and recommends. It does not decide, and the report says so.
Anyone who picks the peak of this curve without reading the words has replaced
a judgment call with a number that was never meant to carry it.

Both metrics come out of the document-term matrix in a single pass -- see
`_topics.coherence`. C_v, the one most papers quote, is deliberately missing:
it needs sliding windows over the raw text, and wanting it is the usual reason
people end up adding gensim.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from . import _topics

__all__ = ["sweep_topic_count", "ENGINES"]

PathLike = Union[str, Path]

#: The engines a sweep can run. Both read the same matrix; which weighting that
#: matrix uses follows the engine, as it does when you fit one for real.
ENGINES = ("lda", "nmf")


def _parse_counts(k_values) -> list:
    """``"5,10,20"`` or ``[5, 10, 20]`` -> ``[5, 10, 20]``.

    A string as well as a list because this arrives from a YAML pipeline and
    from a command line, and typing a list on a command line is miserable.
    """
    if isinstance(k_values, str):
        parts = [p.strip() for p in k_values.replace(";", ",").split(",")]
        k_values = [p for p in parts if p]
    counts = []
    for value in k_values:
        try:
            k = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{value!r} is not a number of topics. Give something like "
                "'5,10,20,40'.") from None
        if k < 2:
            raise ValueError(f"{k} topics is not a topic model; ask for 2 or more.")
        counts.append(k)
    if not counts:
        raise ValueError("no topic counts to try; give something like '5,10,20,40'.")
    return sorted(set(counts))


@records_settings(
    binding=TEXT_INPUT, grain=TEXT_GRAIN,
    outputs=("out_csv", "out_chart_png", "out_report_md"))
def sweep_topic_count(
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
    out_csv: Optional[PathLike] = None,
    out_chart_png: Optional[PathLike] = None,
    out_report_md: Optional[PathLike] = None,
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

    # ----- the vocabulary, built once and shared by every fit -----
    ngram_n: int = 1,
    stoplist_paths: Optional[Sequence[PathLike]] = None,
    min_freq: int = 5,
    min_obs_pct: float = 0.10,
    min_token_count: int = 10,
    min_npmi: Optional[float] = None,
    lemmatize: bool = False,
    pos_tagged: bool = False,
    engine_nlp: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    keep_punctuation: bool = False,
    matrix_rounding: int = 4,
    vocab_min_freq: float = 0,
    vocab_min_obs_pct: float = 0,
    vocab_rule: Literal["top_n", "min_obs_pct", "min_freq"] = "top_n",
    vocab_top_n: int = 2000,
    vocab_rank_by: Literal["obs_pct", "frequency"] = "obs_pct",

    # ----- the sweep -----
    engine: Literal["lda", "nmf"] = "lda",
    k_values: Union[str, Sequence[int]] = "5,10,20,40",
    metric: Literal["npmi", "umass"] = "npmi",
    top_terms: int = 10,
    passes: int = 10,
    seed: int = 42,
    beta_loss: Literal["frobenius", "kullback-leibler"] = "frobenius",
    rounding: int = 4,
) -> Path:
    """
    Fit a topic model at several topic counts and score each one's coherence.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The corpus, exactly one way.
    workers : int, default=0
        Worker processes for gathering and counting. 0 picks a sensible number.
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs, with ``engine_nlp="stanza"``.
    out_csv : str or pathlib.Path, optional
        One row per topic count: the coherence, and the spread across topics.
        Defaults to ``./features/topic_count_sweep/<gathered filename>``.
    out_chart_png, out_report_md : str or pathlib.Path, optional
        The curve, and a short write-up naming the best score and listing the
        top words of every fit. Written beside ``out_csv`` by default.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, return it unchanged.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols, id_cols, mode, group_by, pattern
        Gathering options, as elsewhere.
    ngram_n, stoplist_paths, min_freq, min_obs_pct, min_token_count, min_npmi
        How the vocabulary is counted. Built **once** and shared by every fit:
        changing the vocabulary between fits would mean comparing coherence
        scores computed over different word lists, which compares nothing.
    lemmatize, pos_tagged, engine_nlp, tokenizer, stanza_lang, keep_punctuation
        Tokenizer settings. ``engine_nlp`` rather than ``engine`` because
        ``engine`` here already means which topic model to fit.
    matrix_rounding : int, default=4
        Decimal places in the shared matrix.
    vocab_min_freq, vocab_min_obs_pct, vocab_rule, vocab_top_n, vocab_rank_by
        How the vocabulary is cut from the frequency list.
    engine : {"lda", "nmf"}, default="lda"
        Which model to fit. The matrix follows: counts for LDA, tf-idf for NMF,
        exactly as when fitting one for real.
    k_values : str or sequence of int, default="5,10,20,40"
        The topic counts to try, as a list or a comma-separated string.
    metric : {"npmi", "umass"}, default="npmi"
        Which coherence to score with. ``npmi`` is normalized to [-1, 1], which
        is what makes scores comparable **between** topic counts -- and comparing
        between topic counts is the entire point here. ``umass`` is the older
        asymmetric measure, for comparison against papers that report it.
    top_terms : int, default=10
        How many words of each topic the coherence is computed over, and how
        many the report lists. Keep it well below ``vocab_top_n / n_topics``:
        if the vocabulary cannot give each topic that many distinct words, the
        lists overlap and coherence starts measuring words from different
        topics against each other. A warning says so when it happens.
    passes, seed : int
        LDA's settings; ignored for NMF, which needs no seed.
    beta_loss : {"frobenius", "kullback-leibler"}
        NMF's divergence; ignored for LDA.
    rounding : int, default=4
        Decimal places in the written scores.

    Returns
    -------
    Path
        ``out_csv``: ``n_topics``, ``coherence``, ``coherence_sd``, ``worst_topic``.

    Notes
    -----
    Coherence is a guide, not a verdict, and this is not an optimizer. Fitting
    is repeated once per topic count, so a sweep over four counts costs about
    four fits -- the matrix, which is the slow part on a big corpus, is built
    once and reused.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; have {ENGINES}")
    if metric not in _topics.COHERENCE_METRICS:
        raise ValueError(
            f"unknown coherence metric {metric!r}; have "
            f"{_topics.COHERENCE_METRICS}. c_v is not available -- it needs "
            "sliding windows over the raw text rather than the matrix.")
    counts = _parse_counts(k_values)

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

    if out_csv is None:
        out_csv = (Path.cwd() / "features" / "topic_count_sweep"
                   / analysis_ready.name)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stem = out_csv.stem
    chart_path = Path(out_chart_png) if out_chart_png else \
        out_csv.with_name(f"{stem}_coherence.png")
    report_path = Path(out_report_md) if out_report_md else \
        out_csv.with_name(f"{stem}_report.md")

    if not overwrite_existing and out_csv.is_file():
        print("Topic-count sweep output already exists; returning existing file.")
        return out_csv

    # one matrix, every fit. rebuilding it per topic count would mean scoring
    # coherence over a different vocabulary each time, and those numbers are
    # not comparable -- which would defeat the only thing a sweep is for.
    weighting = "count" if engine == "lda" else "tfidf"
    _freq, dtm_csv, terms, _columns, _vocab = _topics.build_matrix(
    # the matrix folder is derived from the output's *stem*, not from its
    # folder. a fixed-name sibling (`parent / "matrix"`) is the same path for
    # every step that writes into one folder, and that produced the two worst
    # bugs in this feature: three topic models building over each other's
    # matrix, and an apply leaving a frozen vocabulary where a later fit picked
    # it up and modeled the wrong corpus. Deriving from the stem makes both
    # impossible rather than merely wired-around -- and it is the idiom the
    # rest of the codebase already uses for companion files.
        analysis_ready=analysis_ready,
        out_dir=out_csv.with_name(f"{stem}_matrix"),
        weighting=weighting, matrix_rounding=matrix_rounding,
        text_settings=dict(lemmatize=lemmatize, pos_tagged=pos_tagged,
                           engine=engine_nlp, tokenizer=tokenizer,
                           stanza_lang=stanza_lang,
                           keep_punctuation=keep_punctuation, device=device),
        vocab_settings=dict(vocab_min_freq=vocab_min_freq,
                            vocab_min_obs_pct=vocab_min_obs_pct,
                            vocab_rule=vocab_rule, vocab_top_n=vocab_top_n,
                            vocab_rank_by=vocab_rank_by),
        ngram_settings=dict(ngram_n=ngram_n, stoplist_paths=stoplist_paths,
                            min_freq=min_freq, min_obs_pct=min_obs_pct,
                            min_token_count=min_token_count, min_npmi=min_npmi),
        overwrite_existing=overwrite_existing, workers=workers,
        on_progress=on_progress, encoding=encoding)

    def batches():
        return _topics.stream_counts(dtm_csv, encoding=encoding, skip_cols=2)

    announce(on_progress, "counting how often terms appear together")
    pair, single, n_docs = _topics.codocument_counts(batches, len(terms))

    # a trap worth naming, because it produces confident nonsense rather than
    # an error: coherence is computed over each topic's top `top_terms` words,
    # and if the vocabulary cannot supply that many *distinct* words per topic,
    # the lists necessarily spill into each other. Every score then measures
    # pairs of words from different topics, which do not co-occur, so every
    # count scores badly and the "best" one is arbitrary. Found the hard way on
    # a sixteen-term test corpus asking for ten words across eight topics.
    crowded = [k for k in counts if k * top_terms > len(terms)]
    if crowded:
        import warnings
        warnings.warn(
            f"a {len(terms)}-term vocabulary cannot give {top_terms} distinct "
            f"words to each of {', '.join(str(k) for k in crowded)} topics, so "
            "their coherence is measuring words from different topics against "
            "each other. Raise `vocab_top_n`, lower `top_terms`, or drop the "
            "larger topic counts.")

    # NMF needs the whole matrix and cannot stream, so it is read once here
    # rather than once per topic count -- a four-count sweep was reading and
    # materializing it four times -- and the same memory warning the fit gives
    # is given here, where the cost is paid repeatedly.
    matrix = None
    if engine == "nmf":
        wanted = _topics.nmf_memory_gb(n_docs, len(terms), max(counts))
        if wanted >= 1.0:
            import warnings
            warnings.warn(
                f"NMF holds the whole matrix in memory: about {wanted:.1f} GB "
                f"for {n_docs:,} documents by {len(terms):,} terms. Reduce "
                "`vocab_top_n` if that is more than this machine has.")
        matrix = _topics.read_matrix(dtm_csv, encoding=encoding, skip_cols=2)

    # a sweep is the one place where a topic count that cannot be fitted is
    # not a mistake: it is the question. `fit_lda` refuses k > n_docs and
    # NNDSVD refuses k > min(n_docs, n_terms), and both fire inside the loop
    # below -- so on a 12-document corpus the default 5,10,20,40 threw away
    # three good fits and the whole run, after paying for the matrix. We skip
    # what cannot be fitted, say so, and model the rest.
    ceiling = n_docs if engine == "lda" else min(n_docs, len(terms))
    fittable = [k for k in counts if k <= ceiling]
    skipped = [k for k in counts if k > ceiling]
    if skipped:
        import warnings
        warnings.warn(
            f"{n_docs:,} documents and {len(terms):,} terms cannot support "
            f"{', '.join(str(k) for k in skipped)} topics, so "
            f"{'those counts were' if len(skipped) > 1 else 'that count was'} "
            "skipped. A topic model needs more documents than topics, and in "
            "practice many more.")
    if not fittable:
        raise ValueError(
            f"none of the topic counts {', '.join(str(k) for k in counts)} can "
            f"be fitted to {n_docs:,} document(s) over {len(terms):,} terms: "
            f"the most this corpus can support is {ceiling}. Pass smaller "
            "`k_values`, or model a bigger corpus.")
    counts = fittable

    rows = []
    words = {}
    for i, k in enumerate(counts):
        announce(on_progress, f"fitting {k} topics ({i + 1} of {len(counts)})")
        if engine == "lda":
            components, _trace = _topics.fit_lda(
                batches, len(terms), k, passes=passes, seed=seed)
        else:
            _w, components, _trace = _topics.fit_nmf(
                matrix, k, beta_loss=beta_loss)

        leading = [list(np.argsort(row)[::-1][:top_terms]) for row in components]
        scores = _topics.coherence(leading, pair, single, n_docs, metric=metric)
        rows.append({
            "n_topics": k,
            "coherence": round(float(np.mean(scores)), rounding),
            "coherence_sd": round(float(np.std(scores)), rounding),
            "worst_topic": round(float(np.min(scores)), rounding),
        })
        words[k] = [[terms[j] for j in topic] for topic in leading]

    with atomic_write(out_csv, newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from ..figures.charts import line_chart
    line_chart({f"{metric} coherence": [(r["n_topics"], r["coherence"])
                                        for r in rows]},
               chart_path, title="Topic coherence by number of topics",
               x_label="topics", y_label=f"{metric} coherence",
               on_progress=on_progress)

    _write_report(report_path, rows=rows, words=words, engine=engine,
                  metric=metric, n_docs=n_docs, n_terms=len(terms),
                  skipped=skipped, encoding=encoding)
    return out_csv


def _write_report(path: Path, *, rows, words, engine: str, metric: str,
                  n_docs: int, n_terms: int, skipped=(), encoding: str) -> None:
    """The write-up: the numbers, then the words, then the caveat."""
    best = max(rows, key=lambda r: r["coherence"])
    lines = [
        "# How many topics?",
        "",
        f"Fitted **{engine.upper()}** at {len(rows)} topic counts over "
        f"{n_docs:,} documents and a {n_terms:,}-term vocabulary, scoring each "
        f"with **{metric}** coherence.",
        "",
        f"The highest average coherence was at **{best['n_topics']} topics** "
        f"({best['coherence']}).",
        "",
    ]
    # said here and not only in a warning, because the report is what somebody
    # reads a week later, and a curve that silently stops short of what they
    # asked for reads as a curve that peaked there.
    if skipped:
        lines += [
            f"{', '.join(str(k) for k in skipped)} "
            f"{'topics were' if len(skipped) > 1 else 'topics was'} asked for "
            f"and skipped: this corpus is too small to fit "
            f"{'them' if len(skipped) > 1 else 'it'}.",
            "",
        ]
    lines += [
        "| topics | coherence | spread | worst topic |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        marker = " **<-**" if row is best else ""
        lines.append(f"| {row['n_topics']}{marker} | {row['coherence']} | "
                     f"{row['coherence_sd']} | {row['worst_topic']} |")

    lines += [
        "",
        "## Read the words before believing the number",
        "",
        "Coherence measures whether a topic's words turn up in the same "
        "documents. That is related to whether a topic is *meaningful*, and it "
        "is not the same thing: a model can score well and still cut the "
        "corpus somewhere useless, and the highest score is often at more "
        "topics than anybody wants to interpret.",
        "",
        "Use the curve to rule out counts that are clearly too few or too "
        "many, then pick from what is left by reading the topics below.",
        "",
        "## The topics at each count",
        "",
    ]
    for count in sorted(words):
        lines.append(f"### {count} topics")
        lines.append("")
        for i, topic in enumerate(words[count], start=1):
            lines.append(f"{i}. {', '.join(topic)}")
        lines.append("")

    with atomic_write(path, encoding=encoding) as f:
        f.write("\n".join(lines) + "\n")


CLI = CliSpec(
    sweep_topic_count,
    description="Fit a topic model at several topic counts and score each.",
    aliases={"csv_path": ["--csv"], "out_csv": ["--out"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
