"""
Entropy and diversity, one row per document.

Lexical richness (:mod:`analyze_lexical_richness`) already answers "how varied
is the vocabulary" with a dozen indices. This answers the same family of
questions from information theory instead, which buys three things those
indices do not give you.

**One family rather than a dozen names.** Type-token ratio, Simpson's D and
Yule's K are not unrelated measures; they are points on one curve, the Rényi
entropies, indexed by an order *q* that says how much weight to put on common
words versus rare ones. At *q* = 0 the answer is the number of types, at 1 it
is Shannon, at 2 it is Simpson, and as *q* grows it approaches the commonest
word's share alone. So the order is reported as a setting of the measure
rather than buried in a formula named after somebody, and you can read the
profile across orders instead of picking one index and hoping.

**An honest answer on short texts.** Plug-in entropy is biased *downward*, and
the bias depends on how many tokens you had -- worse than type-token ratio,
which is the usual cautionary example. A 50-word answer and a 5,000-word essay
are not comparable on the plug-in number even when their vocabularies are
equally varied. Four bias corrections are computed beside it (Miller-Madow,
Chao-Shen, Grassberger, NSB), so the difference between them is visible rather
than assumed away. Where they disagree, the text was too short to say.

**Structure as well as variety.** Entropy over single tokens measures how
varied the vocabulary is. Conditional entropy over pairs and triples measures
how *predictable* the next token is given the last one or two, which is a
different construct -- a text can have a wide vocabulary and be highly
formulaic. The compression ratios are a crude estimate of the same quantity
that makes no assumption about tokenization at all.

Everything is computed over two units: words, and characters. Character-level
measures need no tokenizer and survive languages the word tokenizer handles
badly.

References
----------
* Hill, M. O. (1973). Diversity and evenness: a unifying notation and its
  consequences. *Ecology, 54*(2), 427-432.
* Rényi, A. (1961). On measures of entropy and information. *Berkeley
  Symposium on Mathematical Statistics and Probability*.
* Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics.
  *Journal of Statistical Physics, 52*, 479-487.
* Miller, G. A. (1955). Note on the bias of information estimates.
  *Information Theory in Psychology*.
* Chao, A., & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index
  of diversity when there are unseen species. *Environmental and Ecological
  Statistics, 10*, 429-443.
* Grassberger, P. (2003). Entropy estimates from insufficient samplings.
  *arXiv:physics/0307138*.
* Nemenman, I., Shafee, F., & Bialek, W. (2002). Entropy and inference,
  revisited. *NIPS 14*.
* Pielou, E. C. (1966). The measurement of diversity in different types of
  biological collections. *Journal of Theoretical Biology, 13*, 131-144.
"""
from __future__ import annotations

import bz2
import csv
import lzma
import re
import string
import zlib
from collections import Counter
from math import log, log2
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Sequence,
                    Tuple, Union)

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.feature_columns import ColumnSpec
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.row_map import map_text_rows, resolve_passthrough_columns
from ..helpers.text_gather import resolve_analysis_ready

PathLike = Union[str, Path]

__all__ = ["analyze_entropy", "FEATURE_COLUMNS"]

LN2 = log(2.0)

#: The units a document is cut into. Words need a tokenizer and carry meaning;
#: characters need nothing and survive a language the tokenizer handles badly.
UNITS = ("word", "char")

#: Which orders of the Rényi/Hill profile to report. 1 is left out because it
#: *is* Shannon, which is reported at length under its own name, and a second
#: column holding the same number under another name is how a feature set
#: starts to double-count.
ORDERS: Tuple[Tuple[str, float], ...] = (
    ("q0", 0.0), ("q0_5", 0.5), ("q2", 2.0), ("q3", 3.0), ("qinf", float("inf")),
)

#: Tsallis at the same orders, minus 0 (which is just types - 1) and infinity
#: (which does not converge).
TSALLIS_ORDERS: Tuple[Tuple[str, float], ...] = (
    ("q0_5", 0.5), ("q2", 2.0), ("q3", 3.0),
)

#: How deep the block and conditional entropies go by default.
DEFAULT_MAX_ORDER = 3

_PUNCT = str.maketrans({p: " " for p in string.punctuation})
_DIGITS = re.compile(r"[0-9]+")


def _measure_names(max_order: int) -> List[str]:
    """Every column this writes, in the order it writes them."""
    cols: List[str] = []
    for unit in UNITS:
        u = f"ent_{unit}"
        cols += [f"{u}_shannon_bits", f"{u}_shannon_effective",
                 f"{u}_shannon_evenness",
                 f"{u}_mm_bits", f"{u}_mm_effective",
                 f"{u}_chao_shen_bits", f"{u}_chao_shen_effective",
                 f"{u}_grassberger_bits", f"{u}_grassberger_effective",
                 f"{u}_nsb_bits", f"{u}_nsb_effective"]
        for name, _q in ORDERS:
            cols += [f"{u}_renyi_{name}_bits", f"{u}_hill_{name}"]
        cols += [f"{u}_tsallis_{name}" for name, _q in TSALLIS_ORDERS]
        cols += [f"{u}_berger_parker", f"{u}_coverage", f"{u}_chao1"]
        for n in range(2, max_order + 1):
            cols += [f"{u}_block{n}_bits", f"{u}_conditional{n}_bits"]
        cols += [f"{u}_rate_bits", f"{u}_tokens"]
    cols += ["ent_compress_zlib_bpc", "ent_compress_bz2_bpc",
             "ent_compress_lzma_bpc"]
    return cols


#: Declared from the same function that writes the header, so the registry and
#: the file cannot drift. The order-carrying names are patterns because how
#: deep the block entropies go is a setting: `block2` and `block4` are not the
#: same measure and must not share a column.
FEATURE_COLUMNS = ColumnSpec(
    label="Entropy",
    names=tuple(c for c in _measure_names(DEFAULT_MAX_ORDER)
                if "block" not in c and "conditional" not in c),
    patterns=tuple(f"ent_{u}_{kind}{{n}}_bits"
                   for u in UNITS for kind in ("block", "conditional")),
)

#: Counts, not measurements. `assemble` keeps them in the analysis table to
#: filter on and out of the feature sets -- entropy is the most length-
#: sensitive family of measures in common use, so the length is exactly what
#: somebody needs to hold constant, and exactly what nobody should regress on
#: by accident.
BOOKKEEPING = tuple(f"ent_{u}_tokens" for u in UNITS)


# ---------------------------------------------------------------------------
# Cutting a document into units
# ---------------------------------------------------------------------------

def _words(text: str, *, lowercase: bool, strip_punctuation: bool,
           strip_digits: bool) -> List[str]:
    """
    Words, preprocessed the way `analyze_lexical_richness` preprocesses them.

    Deliberately the same rules, so an entropy column and a type-token ratio
    from the same run are answers about the same token stream and can be
    compared.
    """
    if lowercase:
        text = text.lower()
    if strip_digits:
        text = _DIGITS.sub("", text)
    if strip_punctuation:
        text = text.translate(_PUNCT)
    return text.split()


def _chars(text: str, *, lowercase: bool) -> List[str]:
    """Characters, with runs of whitespace flattened to one space so that
    indentation does not read as vocabulary."""
    text = " ".join(text.split())
    return list(text.lower() if lowercase else text)


def _blocks(seq: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    """Every window of `n` consecutive units."""
    return (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


# ---------------------------------------------------------------------------
# The estimators. Everything takes a vector of counts and returns bits.
# ---------------------------------------------------------------------------

def _counts(seq: Sequence) -> List[int]:
    return sorted(Counter(seq).values())


def shannon_bits(counts: Sequence[int]) -> float:
    """Plug-in (maximum likelihood) Shannon entropy, in bits."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    # clamped at zero: one repeated word gives -0.0, and a rounding wobble can
    # give a tiny negative. an entropy is never below zero.
    return max(0.0, float(-sum((c / total) * log(c / total)
                               for c in counts) / LN2))


def miller_madow_bits(counts: Sequence[int]) -> float:
    """
    Plug-in plus ``(V - 1) / 2N``: the leading term of the bias.

    The cheapest correction and the least effective on a badly undersampled
    text, because it only knows how many types you *saw*.
    """
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return max(0.0, shannon_bits(counts)
               + (len(counts) - 1) / (2.0 * total * LN2))


def chao_shen_bits(counts: Sequence[int]) -> float:
    """
    Coverage-adjusted, after Chao and Shen (2003).

    Estimates what share of the distribution you actually saw -- from how many
    words appeared exactly once -- and reweights accordingly. The one that
    holds up best on the short, Zipfian samples that text usually is.
    """
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    singles = float(sum(1 for c in counts if c == 1))
    if singles >= total:
        # every word appeared once, so coverage estimates as zero and the
        # formula divides by it. back off by one rather than refuse.
        singles = total - 1.0
    coverage = 1.0 - singles / total
    if coverage <= 0.0:
        return shannon_bits(counts)
    out = 0.0
    for c in counts:
        p = coverage * c / total
        seen = 1.0 - (1.0 - p) ** total
        if seen > 0.0:
            out -= p * log(p) / seen
    return max(0.0, float(out / LN2))


def grassberger_bits(counts: Sequence[int]) -> float:
    """Grassberger (2003), which corrects each count by a digamma term."""
    from scipy.special import digamma

    total = float(sum(counts))
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts:
        g = digamma(c) + 0.5 * ((-1.0) ** c) * (digamma((c + 1) / 2.0)
                                                - digamma(c / 2.0))
        out += c * g
    return max(0.0, float((log(total) - out / total) / LN2))


def chao1(counts: Sequence[int]) -> float:
    """
    How many types the text would have had if you had kept reading.

    The bias-corrected Chao1 estimator: observed types plus a term built from
    how many appeared once and twice. Reported in its own right -- it is the
    q = 0 diversity you cannot see -- and used as the alphabet size NSB needs.
    """
    seen = float(len(counts))
    f1 = float(sum(1 for c in counts if c == 1))
    f2 = float(sum(1 for c in counts if c == 2))
    return seen + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))


def nsb_bits(counts: Sequence[int], alphabet: Optional[int] = None) -> float:
    """
    Nemenman-Shafee-Bialek: Bayesian, integrated over the Dirichlet prior.

    The only estimator here that needs to be told how many types the text
    *could* have used, because its prior is over distributions on a known
    alphabet. Text has no such number, and the answer moves by three bits
    across plausible guesses -- more than any two other estimators differ --
    so guessing badly is worse than not using it.

    So the guess is not left to anybody: ``alphabet`` defaults to
    :func:`chao1`, which estimates the unseen types from the seen ones. That
    makes it self-tuning and, on samples where the truth is known, accurate to
    within a tenth of a bit.
    """
    import numpy as np
    from scipy.special import digamma, gammaln, polygamma

    n = np.asarray(list(counts), dtype=float)
    total = float(n.sum())
    if total <= 0 or len(n) <= 1:
        return 0.0
    size = int(max(len(n), round(alphabet if alphabet else chao1(counts))))
    if size <= 1:
        return 0.0

    betas = np.exp(np.linspace(log(1e-4), log(1e3), 400))
    weights = np.empty_like(betas)
    means = np.empty_like(betas)
    for i, b in enumerate(betas):
        # the likelihood of this concentration, and NSB's prior on it (chosen
        # so the implied prior on the entropy itself is close to flat)
        like = (gammaln(size * b) - gammaln(total + size * b)
                + float((gammaln(n + b) - gammaln(b)).sum()))
        slope = size * polygamma(1, size * b + 1) - polygamma(1, b + 1)
        weights[i] = like + log(max(slope, 1e-300))
        seen = float(((n + b) / (total + size * b)
                      * digamma(n + b + 1)).sum())
        # the types that never turned up still carry prior weight. leaving
        # them out let the estimate climb above log2(alphabet), which is
        # impossible.
        unseen = (size - len(n)) * (b / (total + size * b)) * digamma(b + 1)
        means[i] = digamma(total + size * b + 1) - seen - unseen
    weights -= weights.max()
    # times beta, because the grid is spaced in log(beta)
    w = np.exp(weights) * betas
    if not w.sum():
        return shannon_bits(counts)
    return float((w * means).sum() / w.sum() / LN2)


def renyi_bits(counts: Sequence[int], order: float) -> float:
    """Rényi entropy of the given order, in bits. Order 1 is Shannon."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts]
    if order == 1.0:
        return shannon_bits(counts)
    if order == 0.0:
        return log2(len(ps))
    if order == float("inf"):
        return -log2(max(ps))
    return float(log(sum(p ** order for p in ps)) / (1.0 - order) / LN2)


def hill_number(counts: Sequence[int], order: float) -> float:
    """
    The Rényi entropy as an effective number of types.

    Reported alongside every entropy because "4.2 bits" is not a quantity
    anybody has intuitions about and "18 equally common words" is.
    """
    return float(2.0 ** renyi_bits(counts, order))


def tsallis(counts: Sequence[int], order: float) -> float:
    """Tsallis entropy: the same family under a different, non-logarithmic
    way of combining independent parts."""
    total = float(sum(counts))
    if total <= 0 or order == 1.0:
        return shannon_bits(counts) * LN2
    ps = [c / total for c in counts]
    return float((1.0 - sum(p ** order for p in ps)) / (order - 1.0))


def coverage(counts: Sequence[int]) -> float:
    """Good-Turing sample coverage: the share of the distribution the text
    actually showed you. Low coverage is the signal that the entropy numbers
    below it are estimates rather than measurements."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    singles = float(sum(1 for c in counts if c == 1))
    return float(max(0.0, 1.0 - singles / total))


def berger_parker(counts: Sequence[int]) -> float:
    """The commonest unit's share -- dominance, the q -> infinity end."""
    total = float(sum(counts))
    return float(max(counts) / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# One document
# ---------------------------------------------------------------------------

def _unit_row(units: Sequence[str], prefix: str, max_order: int) -> Dict[str, float]:
    """Every measure for one unit of one document."""
    out: Dict[str, float] = {}
    counts = _counts(units)
    total = len(units)
    if not counts:
        return {name: "" for name in _measure_names(max_order)
                if name.startswith(prefix)}

    h = shannon_bits(counts)
    out[f"{prefix}_shannon_bits"] = h
    out[f"{prefix}_shannon_effective"] = 2.0 ** h
    # Pielou's evenness: how close to flat, where 1 means every type equally
    # common. Undefined on one type, which is even but says nothing.
    out[f"{prefix}_shannon_evenness"] = (h / log2(len(counts))
                                         if len(counts) > 1 else "")
    for key, fn in (("mm", miller_madow_bits), ("chao_shen", chao_shen_bits),
                    ("grassberger", grassberger_bits), ("nsb", nsb_bits)):
        value = fn(counts)
        out[f"{prefix}_{key}_bits"] = value
        out[f"{prefix}_{key}_effective"] = 2.0 ** value
    for name, q in ORDERS:
        out[f"{prefix}_renyi_{name}_bits"] = renyi_bits(counts, q)
        out[f"{prefix}_hill_{name}"] = hill_number(counts, q)
    for name, q in TSALLIS_ORDERS:
        out[f"{prefix}_tsallis_{name}"] = tsallis(counts, q)
    out[f"{prefix}_berger_parker"] = berger_parker(counts)
    out[f"{prefix}_coverage"] = coverage(counts)
    out[f"{prefix}_chao1"] = chao1(counts)

    # block entropies, and the conditional entropy each one implies: how many
    # bits the next unit still costs once you know the previous n - 1.
    previous = h
    rate = h
    for n in range(2, max_order + 1):
        windows = list(_blocks(units, n))
        block = shannon_bits(_counts(windows)) if windows else ""
        out[f"{prefix}_block{n}_bits"] = block
        if block == "":
            out[f"{prefix}_conditional{n}_bits"] = ""
        else:
            rate = block - previous
            out[f"{prefix}_conditional{n}_bits"] = rate
            previous = block
    out[f"{prefix}_rate_bits"] = rate
    out[f"{prefix}_tokens"] = total
    return out


def _compression_row(text: str) -> Dict[str, float]:
    """
    Bits per character after three general-purpose compressors.

    A crude estimate of the same thing the conditional entropies estimate, and
    the only one here that assumes nothing at all about tokens -- which is why
    it is worth having beside them rather than instead of them.
    """
    raw = text.encode("utf-8")
    if not raw:
        return {k: "" for k in ("ent_compress_zlib_bpc", "ent_compress_bz2_bpc",
                                "ent_compress_lzma_bpc")}
    n = float(len(raw))
    return {
        "ent_compress_zlib_bpc": 8.0 * len(zlib.compress(raw, 9)) / n,
        "ent_compress_bz2_bpc": 8.0 * len(bz2.compress(raw, 9)) / n,
        "ent_compress_lzma_bpc": 8.0 * len(lzma.compress(raw)) / n,
    }


class _Scorer:
    """Settings plus the per-document work, in one picklable object."""

    def __init__(self, *, lowercase: bool, strip_punctuation: bool,
                 strip_digits: bool, max_order: int):
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.strip_digits = strip_digits
        self.max_order = max_order

    def __call__(self, job: Tuple[str, str]) -> Dict[str, float]:
        _text_id, text = job
        text = text or ""
        row: Dict[str, float] = {}
        row.update(_unit_row(
            _words(text, lowercase=self.lowercase,
                   strip_punctuation=self.strip_punctuation,
                   strip_digits=self.strip_digits),
            "ent_word", self.max_order))
        row.update(_unit_row(_chars(text, lowercase=self.lowercase),
                             "ent_char", self.max_order))
        row.update(_compression_row(text))
        return row


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",), bookkeeping=BOOKKEEPING)
def analyze_entropy(
    *,
    # ----- input: the same three ways every text step takes one -----
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,

    # ----- output -----
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,

    # ----- how to cut the text up -----
    lowercase: bool = True,
    strip_punctuation: bool = True,
    strip_digits: bool = True,
    max_order: int = DEFAULT_MAX_ORDER,

    # ----- gathering, for when the input is a spreadsheet or a folder -----
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = False,
    pass_through_cols: Optional[Sequence[str]] = None,

    workers: int = 0,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
) -> Path:
    """
    Entropy and diversity measures for each text, over words and characters.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The usual input contract: a spreadsheet of texts, a folder of
        documents, or a table somebody already gathered.
    out_features_csv : str or Path, optional
        Where to write. Default ``./features/entropy.csv``.
    overwrite_existing : bool, default False
        Rebuild the table if it is already there.
    lowercase : bool, default True
        Fold case before counting. Off, "The" and "the" are two types.
    strip_punctuation, strip_digits : bool, default True
        Applied to the *word* units only, matching what
        `analyze_lexical_richness` does, so the two are comparable. The
        character units always keep punctuation, since that is most of what
        distinguishes character-level style.
    max_order : int, default 3
        How far the block and conditional entropies go. Order 3 over
        characters already wants a few thousand characters to be worth
        reading; higher orders on short texts measure the sample rather than
        the text.
    text_cols : sequence of str, default ``("text",)``
        Which spreadsheet columns hold the text, when one has to be gathered.
    id_cols : sequence of str, optional
        Columns that compose each row's identifier.
    mode : {"concat", "separate"}, default "concat"
        Measure several text columns joined together, or one at a time.
    group_by : sequence of str, optional
        Combine rows sharing these columns before measuring.
    delimiter, encoding, joiner, num_buckets, max_open_bucket_files, tmp_root
        The gatherer's own settings: how to read the spreadsheet, what to join
        combined texts with, and how much to spill to disk on a large one.
    recursive : bool, default True
        Search subfolders when the input is a folder of documents.
    pattern : str
        Which files in that folder count as documents.
    id_from : {"stem", "name", "path"}, default "stem"
        What to call each document, when the input is a folder.
    include_source_path : bool, default False
        Carry each document's path into the gathered table.
    pass_through_cols : sequence of str, optional
        Columns of the gathered table to copy into the output.
    workers : int, default 0
        Processes. 0 picks a sensible number for the job's size.

    Returns
    -------
    pathlib.Path
        ``out_features_csv``.

    Notes
    -----
    Read `coverage` before anything else. It says what share of the
    distribution the text actually showed you, and when it is low the four
    Shannon estimates will disagree -- that disagreement is the honest width
    of the answer, not noise to average away.

    Every entropy is reported in bits and again as an effective number of
    types, which is the same number in units people can hold in their head.

    A conditional entropy can come out slightly *negative* on a short text.
    That is impossible in truth and is the estimate telling on itself: the
    block entropy above it is more undersampled than the one below, so the
    difference goes the wrong way. It is left as it falls rather than clamped
    at zero, because a negative number is a visible sign that the text was too
    short for that order and a zero is not.
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
        workers=workers, verbose=verbose)

    out = Path(out_features_csv) if out_features_csv \
        else Path("features") / "entropy.csv"
    if out.exists() and not overwrite_existing:
        if verbose:
            print(f"[entropy] exists, leaving alone: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    if int(max_order) < 1:
        raise ValueError("max_order must be at least 1: order 1 is the "
                         "single-unit entropy, which is the point of it.")

    with Path(analysis_ready).open("r", newline="", encoding=encoding) as fin:
        header_fields = csv.DictReader(fin).fieldnames or []
    if "text_id" not in header_fields or "text" not in header_fields:
        raise ValueError(f"Expected columns 'text_id' and 'text' in "
                         f"{analysis_ready}; found {header_fields}")
    # one shared rule for what rides along beside text_id
    passthrough = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)
    names = _measure_names(int(max_order))
    scorer = _Scorer(lowercase=lowercase, strip_punctuation=strip_punctuation,
                     strip_digits=strip_digits, max_order=int(max_order))

    announce(on_progress, "measuring entropy")
    with atomic_write(out, newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["text_id", *passthrough,
                                                  *names])
        writer.writeheader()
        for row, values in map_text_rows(
                analysis_ready, encoding=encoding, workers=workers,
                message="entropy", on_progress=on_progress,
                inline_fn=scorer, pool_fn=scorer):
            out_row = {"text_id": row.get("text_id", "")}
            for col in passthrough:
                out_row[col] = row.get(col, "")
            for name in names:
                value = values.get(name, "")
                out_row[name] = (f"{value:.6g}" if isinstance(value, float)
                                 else value)
            writer.writerow(out_row)

    if verbose:
        print(f"[entropy] wrote {out}")
    return out


CLI = CliSpec(
    {"run": analyze_entropy},
    description="Entropy and diversity measures for each text.",
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
