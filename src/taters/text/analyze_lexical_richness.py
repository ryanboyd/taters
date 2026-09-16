# taters/text/analyze_lexical_richness.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union, Sequence, Literal, Dict, Any, Iterable, List, Tuple
import csv
import re
import string
import random
from collections import Counter
from itertools import islice
from math import log, sqrt, comb
from statistics import mean

from ..helpers.atomic import atomic_write
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.feature_columns import ColumnSpec
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.cliargs import CliSpec

# ------------------------------------------------------------------------------
# Lightweight tokenization (lower, strip digits, strip punctuation)
# ------------------------------------------------------------------------------

#: The measures whose names never change. A module constant so the column
#: registry can read them without running anything -- see
#: `helpers/feature_columns.py` for why two measures must never agree on a name.
METRICS_FIXED = ("ttr", "rttr", "cttr", "herdan_c", "summer_s", "dugast",
                 "maas", "yule_k", "yule_i", "herdan_vm", "simpson_d")

#: The other five carry their window or threshold in the name, so two runs at
#: different settings cannot produce the same column with different meanings --
#: `msttr_50` and `msttr_100` are not the same measure.
FEATURE_COLUMNS = ColumnSpec(
    label="Lexical richness",
    names=METRICS_FIXED,
    # mtld carries a *threshold*, so its tail is `0_72`, not a plain integer
    patterns=("msttr_{n}", "mattr_{n}", "mtld_{*}", "hdd_{n}", "vocd_{n}"),
)

_punct_table = str.maketrans({p: " " for p in string.punctuation})
_digit_re = re.compile(r"[0-9]+")

def _preprocess(text: str) -> str:
    text = text.lower()
    text = _digit_re.sub("", text)
    return text.translate(_punct_table)

def _tokenize(text: str) -> List[str]:
    text = _preprocess(text)
    words = text.split()
    return words

# ------------------------------------------------------------------------------
# Helpers shared by multiple metrics
# ------------------------------------------------------------------------------

def _segment_generator(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def _list_sliding_window(seq: List[str], window_size: int) -> Iterable[Tuple[str, ...]]:
    it = iter(seq)
    window = tuple(islice(it, window_size))
    if len(window) == window_size:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window

def _freq_wordfrequency_table(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Return pairs (freq, count_of_terms_with_that_freq), sorted by freq asc.
    """
    term_freq = Counter(tokens)
    bucket: Dict[int, int] = {}
    for _, f in term_freq.items():
        bucket[f] = bucket.get(f, 0) + 1
    return sorted(bucket.items())

# ------------------------------------------------------------------------------
# Core lexical richness measures
# ------------------------------------------------------------------------------

def ttr(tokens: List[str]) -> Optional[float]:
    w = len(tokens)
    if w == 0:
        return None
    return len(set(tokens)) / w

def rttr(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 0:
        return None
    return t / sqrt(w)

def cttr(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 0:
        return None
    return t / sqrt(2 * w)

def herdan_c(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 1 or t <= 1:
        return None
    return log(t) / log(w)

def summer_s(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 2 or t <= 2:
        return None
    return log(log(t)) / log(log(w))

def dugast(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 1 or t <= 1 or w == t:
        return None
    return (log(w) ** 2) / (log(w) - log(t))

def maas(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w <= 1 or t <= 1:
        return None
    return (log(w) - log(t)) / (log(w) ** 2)

def yule_k(tokens: List[str]) -> Optional[float]:
    w = len(tokens)
    if w <= 1:
        return None
    fi = _freq_wordfrequency_table(tokens)  # [(freq, count)]
    total_sum = sum(cnt * (freq ** 2) for freq, cnt in fi)
    return 1e4 * (total_sum / (w ** 2) - 1 / w)

def yule_i(tokens: List[str]) -> Optional[float]:
    w = len(tokens)
    if w == 0:
        return None
    t = len(set(tokens))
    fi = _freq_wordfrequency_table(tokens)
    total_sum = sum(cnt * (freq ** 2) for freq, cnt in fi)
    denom = (total_sum - t)
    if denom <= 0:
        return None
    return (t ** 2) / denom

def herdan_vm(tokens: List[str]) -> Optional[float]:
    w = len(tokens); t = len(set(tokens))
    if w == 0 or t == 0:
        return None
    fi = _freq_wordfrequency_table(tokens)
    sum_element = sum(cnt * (freq / w) ** 2 for freq, cnt in fi)
    val = sum_element - (1 / t)
    return sqrt(val) if val > 0 else 0.0

def simpson_d(tokens: List[str]) -> Optional[float]:
    w = len(tokens)
    if w <= 1:
        return None
    fi = _freq_wordfrequency_table(tokens)
    total = sum(cnt * freq * (freq - 1) for freq, cnt in fi)
    return total / (w * (w - 1))

def msttr(tokens: List[str], segment_window: int = 100, discard: bool = True) -> Optional[float]:
    w = len(tokens)
    if w <= segment_window or segment_window < 1:
        return None
    scores = []
    for seg in _segment_generator(tokens, segment_window):
        if not seg:
            continue
        scores.append(len(set(seg)) / len(seg))
    if not scores:
        return None
    if discard and len(tokens) % segment_window != 0 and len(scores) > 0:
        scores = scores[:-1]
    return mean(scores) if scores else None

def mattr(tokens: List[str], window_size: int = 100) -> Optional[float]:
    w = len(tokens)
    if w < window_size or window_size < 1:
        return None
    scores = [len(set(win)) / window_size for win in _list_sliding_window(tokens, window_size)]
    return mean(scores) if scores else None

def mtld(tokens: List[str], threshold: float = 0.72) -> Optional[float]:
    if not tokens:
        return None

    def _sub(seq: List[str], thr: float) -> float:
        terms: set[str] = set()
        wc = 0
        factors = 0.0
        ttr_last = 1.0
        for w in seq:
            wc += 1
            terms.add(w)
            ttr_now = len(terms) / wc
            ttr_last = ttr_now
            if ttr_now <= thr:
                factors += 1
                wc = 0
                terms.clear()
        if wc > 0:
            # the leftover bit counts as a partial factor
            factors += (1 - ttr_last) / (1 - thr)
        if factors == 0:
            # we never dipped below the threshold, so we approximate a partial
            # factor instead
            ttr_overall = len(set(seq)) / len(seq)
            if ttr_overall == 1:
                factors += 1
            else:
                factors += (1 - ttr_overall) / (1 - thr)
        return len(seq) / factors

    forward = _sub(tokens, threshold)
    backward = _sub(list(reversed(tokens)), threshold)
    return mean([forward, backward])

def _hypergeom_pmf_zero(N: int, K: int, n: int) -> float:
    """
    PMF for drawing zero successes: C(N-K, n) / C(N, n)
    """
    if n > N or K < 0 or N <= 0 or n < 0:
        return 1.0
    num = comb(N - K, n)
    den = comb(N, n)
    return 0.0 if den == 0 else (num / den)

def hdd(tokens: List[str], draws: int = 42) -> Optional[float]:
    """
    HD-D (McCarthy & Jarvis): sum over types of (1 - P(X=0)) / draws,
    where X ~ Hypergeom(N, K, n) with N=len(tokens), K=freq(term), n=draws.
    """
    N = len(tokens)
    if N == 0 or draws <= 0 or draws > N:
        return None
    term_freq = Counter(tokens)
    contribs = []
    for K in term_freq.values():
        p0 = _hypergeom_pmf_zero(N, K, draws)
        contribs.append((1 - p0) / draws)
    return sum(contribs)

# ---- VOCD: ttr(N; D) model and simple 1-D fit --------------------------------

def _ttr_nd(N: float, D: float) -> float:
    """
    McKee et al.'s functional form relating TTR to text length N and latent D.
    """
    if D <= 0 or N <= 0:
        return 0.0
    return (D / N) * (sqrt(1 + 2 * (N / D)) - 1)

def vocd(tokens: List[str], ntokens: int = 50, within_sample: int = 100,
         iterations: int = 3, seed: int = 42) -> Optional[float]:
    """
    Estimate D by:
      - for N in 35..ntokens:
          * sample 'within_sample' subsets of size N, compute TTR, average
      - grid search D over a reasonable range to minimize squared error to _ttr_nd
      - repeat 'iterations' times and average the best D
    """
    if len(tokens) <= ntokens or ntokens < 35:
        return None
    rng = random.Random(seed)
    Ds: List[float] = []

    # set up our search grid for D (5..200)
    grid: List[float] = []
    # this covers where typical D values live (10..120) with room to spare
    for d in range(5, 201):
        grid.append(float(d))

    for it in range(iterations):
        x_vals: List[int] = []
        y_means: List[float] = []
        for N in range(35, ntokens + 1):
            ttrs: List[float] = []
            for _ in range(within_sample):
                sample = rng.sample(tokens, k=N)
                ttrs.append(len(set(sample)) / N)
            x_vals.append(N)
            y_means.append(mean(ttrs))

        # now we find the D that minimizes the squared error
        best_D = None
        best_err = float("inf")
        for D in grid:
            err = 0.0
            for N, y in zip(x_vals, y_means):
                yhat = _ttr_nd(N, D)
                diff = (y - yhat)
                err += diff * diff
            if err < best_err:
                best_err = err
                best_D = D
        if best_D is not None:
            Ds.append(best_D)

    return mean(Ds) if Ds else None

# ------------------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------------------

def _richness_of(text: str, params: Dict[str, Any]) -> list:
    """
    Every metric's value for one text, in the output's column order (the
    eleven fixed measures, then the five parameterized ones). One scorer for
    the inline path and the worker processes, so they cannot drift.
    """
    txt = (text or "").strip()
    toks = _tokenize(txt) if txt else []
    return [
        ttr(toks), rttr(toks), cttr(toks), herdan_c(toks), summer_s(toks),
        dugast(toks), maas(toks), yule_k(toks), yule_i(toks),
        herdan_vm(toks), simpson_d(toks),
        msttr(toks, segment_window=params["msttr_window"]),
        mattr(toks, window_size=params["mattr_window"]),
        mtld(toks, threshold=params["mtld_threshold"]),
        hdd(toks, draws=params["hdd_draws"]),
        vocd(toks, ntokens=params["vocd_ntokens"],
             within_sample=params["vocd_within_sample"],
             iterations=params["vocd_iterations"],
             seed=params["vocd_seed"]),
    ]


#: Per-process state for scoring workers: the parameter dict arrives once
#: via the initializer, not pickled per text.
_RICHNESS_WORKER: Dict[str, Any] = {}


def _init_richness_worker(params: Dict[str, Any]) -> None:
    _RICHNESS_WORKER["params"] = dict(params)


def _richness_in_worker(pair):
    _tid, text = pair
    return _richness_of(text, _RICHNESS_WORKER["params"])


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",))
def analyze_lexical_richness(
    *,
    # ----- Input source (exactly one unless analysis_csv is provided) ----------
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,  # if given, we skip gathering
    gathered_csv: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,

    # ----- Output --------------------------------------------------------------
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    workers: int = 0,

    # ====== SHARED I/O OPTIONS ======
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
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== Metric hyperparameters (optional) ======
    msttr_window: int = 100,
    mattr_window: int = 100,
    mtld_threshold: float = 0.72,
    hdd_draws: int = 42,
    vocd_ntokens: int = 50,
    vocd_within_sample: int = 100,
    vocd_iterations: int = 3,
    vocd_seed: int = 42,

    # ====== NEW: passthrough control ======
    pass_through_cols: Optional[Sequence[str]] = None,
) -> Path:
    """
    Compute lexical richness/diversity metrics for each text row and write a features CSV.
    Draws heavily from https://github.com/LSYS/lexicalrichness but makes several key changes 
    with the goals of minimizing dependencies, attempting to make some speed optimizations with 
    grid search instead of precise curve specifications, and making some principled decisions 
    around punctuation/hyphenization that differ from the original Note that these decisions are 
    not objectively "better" than the original but, instead, reflect my own experiences/intuitions 
    about what makes sense.

    This function accepts (a) an *analysis-ready* CSV (with columns `text_id,text`), (b) a
    raw CSV plus instructions for gathering/aggregation, or (c) a folder of `.txt` files.
    For each resulting row of text, it tokenizes words and computes a suite of classical
    lexical richness measures (e.g., TTR, Herdan's C, Yule's K, MTLD, MATTR, HDD, VOCD).
    Results are written as a wide CSV whose rows align with the rows in the analysis-ready
    table (or the gathered `group_by` rows), preserving any non-text metadata columns.

    Parameters
    ----------
    csv_path : str or Path, optional
        Source CSV to *gather* from. Use with `text_cols`, optional `id_cols`, and
        optional `group_by`. Exactly one of `csv_path`, `txt_dir`, or `analysis_csv`
        must be provided (unless `analysis_csv` is given, which skips gathering).
    txt_dir : str or Path, optional
        Folder of `.txt` files to gather. File identifiers are created from filenames
        via `id_from` and (optionally) a `source_path` column when `include_source_path=True`.
    analysis_csv : str or Path, optional
        Existing analysis-ready CSV with columns `text_id,text`. When provided, all
        gathering options are ignored and the file is used as-is.
    gathered_csv : str or pathlib.Path, optional
        Where to write the intermediate "analysis-ready" table built from
        ``csv_path`` or ``txt_dir``.

        By default it lands beside the *source* -- which means analyzing a
        spreadsheet in someone's Downloads folder writes a file into their
        Downloads folder. Pass this to keep the intermediate with the rest of a
        run's output instead. Ignored when ``analysis_csv`` is given, because
        then no gathering happens.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)`` so a UI can show a
        real bar instead of a spinner.

        ``total`` is the row count of the analysis-ready table, which is fully
        written before measuring starts, so it is known up front. Until it is,
        ``total`` is ``None`` and ``done`` is a running tally -- the long silent
        passes (reading the input, counting its rows) report through the same
        callback with a ``message`` saying which one is running.

        Injected automatically by the pipeline runner for any step function that
        declares this parameter.
    out_features_csv : str or Path, optional
        Output CSV path. If omitted, defaults to
        `./features/lexical-richness/<analysis_ready_filename>`.
    overwrite_existing : bool, default False
        If `False` and `out_features_csv` exists, the function short-circuits and
        returns the existing path without recomputation. This also controls the
        intermediate analysis-ready CSV: when `True`, it is rebuilt from the current
        source instead of reusing a stale copy from an earlier run.
    encoding : str, default "utf-8-sig"
        Encoding for reading/writing CSVs.
    text_cols : sequence of str, default ("text",)
        Text column(s) to use when `csv_path` is provided. When multiple columns are
        given, they are combined according to `mode` (`concat` or `separate`).
    id_cols : sequence of str, optional
        Columns to carry through unchanged into the analysis-ready CSV prior to analysis
        (e.g., `["source","speaker"]`). These will also appear in the output features CSV.
    mode : {"concat", "separate"}, default "concat"
        Gathering behavior when multiple `text_cols` are provided. `"concat"` joins
        values using `joiner`; `"separate"` produces separate rows per text column.
    group_by : sequence of str, optional
        If provided, texts are grouped by these columns before analysis (e.g.,
        `["source","speaker"]`). With `mode="concat"`, all texts in a group are joined
        into one blob per group; with `mode="separate"`, they remain separate rows.
    delimiter : str, default ","
        Column separator of the *input* spreadsheet. The gathered table and
        the output are always comma-separated.
    joiner : str, default " "
        String used to join text fields when `mode="concat"`.
    num_buckets : int, default 512
        Internal streaming/gather parameter to control temporary file bucketing
        (passed through to the gatherer).
    max_open_bucket_files : int, default 64
        Maximum number of temporary files simultaneously open during gathering.
    tmp_root : str or Path, optional
        Temporary directory root for the gatherer. Defaults to a system temp location.
    recursive : bool, default True
        When `txt_dir` is provided, whether to search subdirectories for `.txt` files.
    pattern : str, default "*.txt"
        Glob pattern for discovering text files under `txt_dir`.
    id_from : {"stem", "name", "path"}, default "stem"
        How to construct `text_id` for `.txt` inputs: file stem, full name, or relative path.
    include_source_path : bool, default True
        When `txt_dir` is used, include a `source_path` column in the analysis-ready CSV.
    msttr_window : int, default 100
        Window size for MSTTR (Mean Segmental TTR). Must be smaller than the number of tokens
        in the text to produce a value.
    mattr_window : int, default 100
        Window size for MATTR (Moving-Average TTR). Must be smaller than the number of tokens.
    mtld_threshold : float, default 0.72
        MTLD threshold for factor completion. A higher threshold yields shorter factors and
        typically lower MTLD values; the default follows common practice.
    hdd_draws : int, default 42
        Sample size `n` for HD-D (Hypergeometric Distribution Diversity). Must be less than
        the number of tokens to produce a value.
    vocd_ntokens : int, default 50
        Maximum sample size used to estimate VOCD (D). For each `N` in 35..`vocd_ntokens`,
        the function computes the average TTR over many random samples (`vocd_within_sample`).
    vocd_within_sample : int, default 100
        Number of random samples drawn per `N` when estimating VOCD.
    vocd_iterations : int, default 3
        Repeat-estimate count for VOCD. The best-fit D from each repetition is averaged.
    vocd_seed : int, default 42
        Seed for the VOCD random sampler (controls reproducibility across runs).

    pass_through_cols : Sequence[str] or None, optional
        Extra input columns to copy into the output beside ``text_id``.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.

    Returns
    -------
    Path
        Path to the written features CSV.

    Output shape
    ------------
    The features CSV starts with::

        text_id, <pass-through columns...>, ttr, rttr, cttr, ...

    Pass-through behavior:

    - If ``pass_through_cols`` is provided, those columns are included in that order.
    - Otherwise, if ``id_cols`` were used during gathering, they are included in that order.
    - Otherwise (backward compatible), *all* non-``text`` columns from the analysis-ready CSV are passed through.

    Metrics emitted per row (``None`` if the text is too short):
    ``ttr, rttr, cttr, herdan_c, summer_s, dugast, maas, yule_k, yule_i, herdan_vm, simpson_d,
    msttr_{msttr_window}, mattr_{mattr_window}, mtld_{mtld_threshold}, hdd_{hdd_draws}, vocd_{vocd_ntokens}``.

    Notes
    -----
    **Tokenization and preprocessing.**
    Texts are lowercased, digits are removed, and punctuation characters are
    replaced with spaces prior to tokenization. As a result, hyphenated forms such
    as `"state-of-the-art"` will be split into separate tokens (`"state"`, `"of"`,
    `"the"`, `"art"`). This choice yields robust behavior across corpora but can
    produce different numeric results than implementations that *remove* hyphens
    (treating `"state-of-the-art"` as a single token). If you require strict parity
    with a hyphen-removal scheme, adapt the internal preprocessing accordingly.

    **Metrics.**
    The following measures are emitted per row (values are `None` when a text is
    too short to support the computation):
    - ``ttr``: Type-Token Ratio (|V| / N)
    - ``rttr``: Root TTR (|V| / sqrt(N))
    - ``cttr``: Corrected TTR (|V| / sqrt(2N))
    - ``herdan_c``: Herdan's C (log |V| / log N)
    - ``summer_s``: Summer's S (log log |V| / log log N)
    - ``dugast``: Dugast's U ((log N)^2 / (log N − log |V|))
    - ``maas``: Maas a^2 ((log N − log |V|) / (log N)^2)
    - ``yule_k``: Yule's K (dispersion of frequencies; higher = less diverse)
    - ``yule_i``: Yule's I (inverse of K, scaled)
    - ``herdan_vm``: Herdan's Vm
    - ``simpson_d``: Simpson's D (repeat-probability across tokens)
    - ``msttr_{msttr_window}``: Mean Segmental TTR over fixed segments
    - ``mattr_{mattr_window}``: Moving-Average TTR over a sliding window
    - ``mtld_{mtld_threshold}``: Measure of Textual Lexical Diversity (bidirectional)
    - ``hdd_{hdd_draws}``: HD-D (expected proportion of types in a sample of size ``hdd_draws``)
    - ``vocd_{vocd_ntokens}``: VOCD (D) estimated by fitting TTR(N) to a theoretical curve

    **VOCD estimation.**
    VOCD is fit without external optimization libraries: the function performs a
    coarse grid search over candidate D values (minimizing squared error between
    observed mean TTRs and a theoretical TTR(N; D) curve) for multiple repetitions,
    then averages the best D across repetitions. This generally tracks SciPy-based
    curve fits closely; you can widen the search grid or add a fine local search
    if tighter agreement is desired.

    Raises
    ------
    FileNotFoundError
        If `analysis_csv` is provided but the file does not exist.
    ValueError
        If none or more than one of `csv_path`, `txt_dir`, or `analysis_csv` are provided,
        or if the analysis-ready CSV is missing required columns (`text_id`, `text`).

    Examples
    --------
    Analyze an existing analysis-ready CSV (utterance-level):

    >>> analyze_lexical_richness(
    ...     analysis_csv="transcripts_all.csv",
    ...     out_features_csv="features/lexical-richness.csv",
    ...     overwrite_existing=True,
    ... )

    Gather from a transcript CSV and aggregate per (source, speaker):

    >>> analyze_lexical_richness(
    ...     csv_path="transcripts/session.csv",
    ...     text_cols=["text"],
    ...     id_cols=["source", "speaker"],
    ...     group_by=["source", "speaker"],
    ...     mode="concat",
    ...     out_features_csv="features/lexical-richness.csv",
    ... )

    See Also
    --------
    analyze_readability : Parallel analyzer producing readability indices.
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
        out_features_csv = Path.cwd() / "features" / "lexical-richness" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print(f"Lexical richness output file already exists; returning existing file: {out_features_csv}")
        return out_features_csv

    # 3) name our metrics
    metrics_fixed = list(METRICS_FIXED)
    m_msttr = f"msttr_{msttr_window}"
    m_mattr = f"mattr_{mattr_window}"
    m_mtld  = f"mtld_{str(mtld_threshold).replace('.', '_')}"
    m_hdd   = f"hdd_{hdd_draws}"
    m_vocd  = f"vocd_{vocd_ntokens}"
    metric_names = metrics_fixed + [m_msttr, m_mattr, m_mtld, m_hdd, m_vocd]

    # 4) figure out the output's shape from the input's header alone
    with analysis_ready.open("r", newline="", encoding=encoding) as fin:
        header_fields = csv.DictReader(fin).fieldnames or []

    if "text_id" not in header_fields or "text" not in header_fields:
        raise ValueError(
            f"Expected columns 'text_id' and 'text' in {analysis_ready}; found {header_fields}"
        )

    # there's one shared rule for what rides along beside text_id -- see
    # resolve_passthrough_columns
    from ..helpers.row_map import resolve_passthrough_columns

    passthrough_cols = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)
    fieldnames = ["text_id", *passthrough_cols, *metric_names]

    # 5) scoring is CPU-bound per row and the rows don't depend on each other,
    #    so we run it on the shared pooled-row driver. that gives us results
    #    in file order no matter the worker count, plus one live sub-bar per
    #    document in flight
    from ..helpers.parallel_map import pool_workers
    from ..helpers.row_map import map_text_rows

    params = dict(msttr_window=msttr_window, mattr_window=mattr_window,
                  mtld_threshold=mtld_threshold, hdd_draws=hdd_draws,
                  vocd_ntokens=vocd_ntokens,
                  vocd_within_sample=vocd_within_sample,
                  vocd_iterations=vocd_iterations, vocd_seed=vocd_seed)
    with atomic_write(out_features_csv, newline="", encoding=encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row, values in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pool_workers(workers, n_rows),
                message="measuring lexical richness", on_progress=on_progress,
                inline_fn=lambda pair: _richness_of(pair[1], params),
                pool_fn=_richness_in_worker,
                initializer=_init_richness_worker, initargs=(params,)):
            out_row: Dict[str, Any] = {
                "text_id": row.get("text_id"),
                **{k: row.get(k) for k in passthrough_cols},
                **dict(zip(metric_names, values)),
            }
            writer.writerow(out_row)
    return out_features_csv

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_lexical_richness,
    description='Compute lexical richness/diversity metrics for an analysis-ready CSV.',
    aliases={
        'csv_path': ['--csv'],
        'out_features_csv': ['--out'],
    },
    legacy={
        '--no-include-source-path': ['--include-source-path', 'false'],
        '--no-recursive': ['--recursive', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
