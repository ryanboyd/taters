"""
Streaming, exact PCA with varimax rotation, over CSVs of any length.

The engine behind the MEM topic model, and a standalone tool for reducing any
feature table (dictionary scores, readability measures, embeddings) before a
downstream model. Two design facts:

* **Exact, one pass, bounded memory.** Feature tables are long (rows) but
  narrow (columns), so the fit streams the rows once, accumulating the
  column sums and the p-by-p cross-product in chunks (BLAS does the work:
  ~50x faster than row-at-a-time), then eigendecomposes the correlation
  matrix in memory. No randomized SVD, no seed, no dask: the decomposition
  is verified against scikit-learn's full SVD to machine precision, and the
  finished rotated solution against R's ``psych::principal`` to ~2 decimal
  places on a real 1,000-term MEM (see `varimax` -- Kaiser normalization
  plus iterate-to-convergence, deliberately correcting the dask script this
  module replaces, whose rotation was unnormalized and capped at 20
  sweeps). Randomized methods earn their keep when p is tens of thousands;
  if that day comes, it is a solver option behind this same interface.

* **A fit is a reusable instrument.** ``fit_pca_csv`` writes a model file
  with the feature names, the training means and deviations, and the
  projection; ``apply_pca_csv`` matches features *by name* in any new table,
  standardizes with the TRAINING statistics, and projects. Applying a model
  to its own training table reproduces the training scores exactly.

Eigenvalues are those of the correlation matrix (they average 1.0), so both
retention rules mean what the textbooks say: parallel analysis (the default
for ``n_components=0``) keeps a component while its eigenvalue beats what
random data of the same size produce at that rank, and the Kaiser criterion
keeps every eigenvalue >= 1. (dask-ml and scikit-learn
scale variances by n-1 rather than n; that uniform sqrt((n-1)/n) on loadings
is a bookkeeping convention, not a disagreement.)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import Ticker, announce, count_rows
from ..helpers.cliargs import CliSpec
from ._common import taters_version

PathLike = Union[str, Path]

#: Model file format, bumped on any incompatible change so an older Taters
#: refuses a newer model instead of misreading it.
PCA_MODEL_FORMAT = 1

#: Rows accumulated per BLAS call in the streaming pass. Big enough that the
#: matrix product dominates, small enough that a chunk of wide rows stays
#: comfortably in memory.
_CHUNK_ROWS = 4096

#: How the number of components is chosen when none is asked for. Parallel
#: analysis (Horn, 1965; the 95th-percentile form of Glorfeld, 1995) keeps a
#: component only while its eigenvalue beats what random data of the same
#: size produce at the same rank. The Kaiser rule keeps every eigenvalue
#: above 1, and on a wide matrix that is most of them -- 101 themes from one
#: real corpus -- because chance alone lifts the leading eigenvalues of
#: random data well past 1.
RETAIN_RULES = ("parallel", "kaiser")
PARALLEL_DRAWS = 50
PARALLEL_PERCENTILE = 95


def parallel_thresholds(n: int, p: int, *, draws: int = PARALLEL_DRAWS,
                        percentile: float = PARALLEL_PERCENTILE, seed: int = 0,
                        chunk_rows: int = _CHUNK_ROWS, on_progress=None):
    """
    The eigenvalues chance alone produces: for each rank, the ``percentile``
    of the correlation-matrix eigenvalues of ``draws`` random standard-normal
    data sets with ``n`` rows and ``p`` columns.

    Memory-safe by construction: no random data set is ever held whole. Each
    draw is generated ``chunk_rows`` rows at a time and folded into a running
    ``p``-by-``p`` cross-product, exactly as :func:`stream_moments` folds a
    real table, so the cost in memory is one more ``p``-by-``p`` matrix --
    the same order as the fit already holds -- whatever ``n`` is. The cost in
    time is ``draws`` decompositions; fifty is the usual number and takes
    seconds at a few hundred features.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n, p = int(n), int(p)
    eigs = np.empty((draws, p), dtype=np.float64)
    for d in range(draws):
        if on_progress is not None:
            on_progress(d, draws, "parallel analysis: eigenvalues of random data")
        sums = np.zeros(p)
        cross = np.zeros((p, p))
        done = 0
        while done < n:
            rows = min(chunk_rows, n - done)
            block = rng.standard_normal((rows, p))
            sums += block.sum(axis=0)
            cross += block.T @ block
            done += rows
        mu = sums / n
        cov = cross / n - np.outer(mu, mu)
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(sd, sd)
        eigs[d] = np.linalg.eigvalsh((corr + corr.T) / 2)[::-1]
    return np.percentile(eigs, percentile, axis=0)


def warn_if_wide(n_features: int, *, warn_gb: float = 1.0) -> None:
    """
    Memory truth-telling before the work starts.

    The fit is memory-safe against corpus LENGTH by construction -- rows
    stream through in chunks and are gone. Its one memory cost grows with
    table WIDTH: the p-by-p cross-product/correlation matrix plus the
    eigendecomposition's workspace, roughly three p*p float64 buffers. At the
    default 250-term vocabulary that is under 2 MB; at 5,000 features about
    0.6 GB; at 20,000 it would be ~10 GB, which deserves a warning before
    hours of streaming, not an OOM after.
    """
    approx_gb = 3 * (n_features ** 2) * 8 / 1024 ** 3
    if approx_gb >= warn_gb:
        import warnings
        warnings.warn(
            f"{n_features} feature columns: the exact fit holds about "
            f"{approx_gb:.1f} GB of p-by-p working memory. Consider fewer "
            "features (e.g. a smaller vocab_top_n).")


def varimax(loadings, gamma: float = 1.0, q: int = 2000, tol: float = 1e-13,
            normalize: bool = True):
    """
    Orthogonal varimax rotation of an (n_features, n_components) loadings
    matrix; returns (rotated_loadings, rotation_matrix).

    ``normalize=True`` is **Kaiser normalization** -- each feature's loading
    row is scaled to unit communality while the rotation is *chosen*, which
    is what ``stats::varimax``, ``psych::principal`` and SPSS all do by
    default. Without it, high-communality features dominate the criterion
    and the solution genuinely differs (a real MEM run diverged from its R
    twin until this matched; the worst theme correlated at 0.39 unnormalized
    and 0.996 normalized). Normalization only steers the choice of rotation:
    the returned loadings are still ``loadings @ R`` exactly, so a scorer
    using the same ``R`` stays coherent with them.

    The iteration budget is a ceiling, not a schedule -- the loop stops on
    convergence, and the predecessor's cap of 20 sweeps was itself a source
    of divergence at MEM sizes (1,000 terms x 25 themes converges in ~100).
    """
    import numpy as np

    p, k = loadings.shape
    if normalize:
        h = np.sqrt((loadings ** 2).sum(axis=1))
        h[h == 0] = 1.0
        working = loadings / h[:, None]
    else:
        working = loadings
    R = np.eye(k)
    d = 0.0
    converged = False
    for _ in range(q):
        Lambda = working @ R
        u, s, vh = np.linalg.svd(
            working.T @ (Lambda ** 3 - (gamma / p) * Lambda
                         @ np.diag(np.sum(Lambda ** 2, axis=0)))
        )
        R_new = u @ vh
        d_old = d
        d = float(np.sum(s))
        R = R_new
        if d_old != 0 and d / d_old < 1 + tol:
            converged = True
            break
    if not converged:
        # running out of sweeps, in practice, means near-tied components: the
        # criterion is almost flat and no rotation is uniquely "the" varimax
        # solution. the rotation we hand back is still exact and
        # deterministic, but quietly passing it off as converged is exactly
        # the kind of plausible-looking-numbers failure we're trying to
        # avoid, so we say so.
        import warnings
        warnings.warn(
            f"varimax did not converge within {q} sweeps; the rotation "
            "returned is the best found. Near-tied components usually cause "
            "this -- the themes are valid but their exact rotation is not "
            "unique, and a different tool may land elsewhere. Fewer "
            "components often stabilizes it.")
    return loadings @ R, R


def stream_moments(csv_path: Path, *, encoding: str, skip_cols: int,
                   on_progress=None,
                   message: str = "measuring the table") -> Tuple[int, "object", "object"]:
    """
    One pass over the rows: n, per-column sums, and X^T X for the columns
    after the first ``skip_cols``. Rows are accumulated in chunks so the
    cross-product is a handful of BLAS calls per chunk rather than an outer
    product per row -- same numbers, a large constant factor faster.
    """
    import numpy as np

    sums = cross = None
    n = 0
    chunk: List[List[str]] = []

    def _flush():
        nonlocal sums, cross, n
        if not chunk:
            return
        block = np.asarray(chunk, dtype=np.float64)
        if sums is None:
            sums = np.zeros(block.shape[1], dtype=np.float64)
            cross = np.zeros((block.shape[1], block.shape[1]), dtype=np.float64)
        sums += block.sum(axis=0)
        cross += block.T @ block
        n += block.shape[0]
        chunk.clear()

    ticker = Ticker(on_progress, count_rows(csv_path, on_progress=on_progress))
    with Path(csv_path).open("r", newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader)                                    # skip the header; caller's job
        for row in reader:
            ticker.tick(message=message)
            chunk.append(row[skip_cols:])
            if len(chunk) >= _CHUNK_ROWS:
                _flush()
        _flush()
    return n, sums, cross


def _decompose(n: int, sums, cross):
    """
    The expensive half of a PCA: the correlation matrix and its eigen-
    decomposition, which do not depend on how many components you keep.

    Split out so a sweep over component counts pays for this once. MEM
    scoring twenty candidate theme counts was otherwise twenty eigen-
    decompositions of the same matrix.

    Returns ``(kept, mu, sigma, eigvals, eigvecs)``, eigen-pairs descending.
    """
    import numpy as np

    mu_all = sums / n
    var_all = np.maximum(cross.diagonal() / n - mu_all ** 2, 0.0)
    kept = np.flatnonzero(var_all > 0)
    if kept.size == 0:
        raise ValueError("Every feature column is constant; there is nothing "
                         "for a PCA to explain.")
    mu = mu_all[kept]
    sigma = np.sqrt(var_all[kept])

    cov = cross[np.ix_(kept, kept)] / n - np.outer(mu, mu)
    corr = cov / np.outer(sigma, sigma)
    corr = np.clip((corr + corr.T) / 2, -1.0, 1.0)      # force it to be symmetric

    eigvals, eigvecs = np.linalg.eigh(corr)             # these come out ascending, so flip
    eigvals = np.maximum(eigvals[::-1], 0.0)
    eigvecs = eigvecs[:, ::-1]
    return kept, mu, sigma, eigvals, eigvecs


def _retain_count(n: int, n_features: int, eigvals, *, n_components: int,
                  retain: str, kaiser_cutoff: float = 1.0, on_progress=None):
    """How many components to keep, and the record of why. See
    :func:`fit_axes` for what the rules mean."""
    import numpy as np

    if n_components > 0:
        return (min(int(n_components), n_features),
                {"rule": "asked", "n_components": int(n_components)})
    if retain == "parallel":
        thresholds = parallel_thresholds(n, n_features, on_progress=on_progress)
        # we keep going while the eigenvalue beats chance at its rank, and
        # stop at the first one that doesn't (that's Horn's procedure). a
        # later one that happens to beat its threshold isn't a component
        # worth naming.
        k = 0
        while k < n_features and eigvals[k] > thresholds[k]:
            k += 1
        k = max(1, k)
        shown = min(k + 1, n_features)
        return k, {"rule": "parallel", "draws": PARALLEL_DRAWS,
                   "percentile": PARALLEL_PERCENTILE,
                   "unrotated_eigenvalues": [float(v) for v in eigvals[:shown]],
                   "thresholds": [float(v) for v in thresholds[:shown]]}
    if retain == "kaiser":
        # Kaiser's rule: keep the components that explain more than one
        # feature's worth of variance -- or more than `kaiser_cutoff` of
        # one, since on a wide matrix a cutoff of exactly 1 keeps almost
        # everything (one real corpus gave 101 themes).
        cutoff = float(kaiser_cutoff)
        k = max(1, int(np.sum(eigvals >= cutoff)))
        return k, {"rule": "kaiser", "cutoff": cutoff,
                   "unrotated_eigenvalues": [float(v) for v in eigvals[:k + 1]]}
    raise ValueError(f"retain must be one of {RETAIN_RULES}, got {retain!r}")


def _axes_at(kept, eigvals, eigvecs, k: int, *, rotation: bool):
    """
    The cheap half: the loadings and the projection at one component count.

    Called once by :func:`fit_axes` and once per candidate count by a sweep,
    over an eigen-decomposition computed once.
    """
    import numpy as np

    unrotated = eigvecs[:, :k] * np.sqrt(eigvals[:k])   # the loadings, before rotation
    if rotation and k > 1:
        loadings, R = varimax(unrotated)
    else:
        loadings, R = unrotated, np.eye(k)

    comp_eig = (loadings ** 2).sum(axis=0)
    order = np.argsort(comp_eig)[::-1]
    loadings = loadings[:, order]
    comp_eig = comp_eig[order]
    flips = np.where(loadings[np.abs(loadings).argmax(axis=0),
                              np.arange(k)] < 0, -1.0, 1.0)
    loadings = loadings * flips

    # the scores have to go through the same rotation, order, and sign flips.
    projection = (eigvecs[:, :k] @ R)[:, order] * flips
    pct = comp_eig / kept.size * 100
    return loadings, projection, comp_eig, pct


def fit_axes(n: int, sums, cross, *, n_components: int, rotation: bool,
             retain: str = "parallel", kaiser_cutoff: float = 1.0,
             on_progress=None):
    """
    From streamed moments to finished axes.

    Returns ``(kept, mu, sigma, loadings, projection, eigenvalues, pct,
    retention)``: ``kept`` indexes the non-constant columns; ``loadings`` is
    the rotated feature-by-component matrix (interpretation); ``projection``
    scores a row -- standardized kept cells @ projection -- and both share
    one component order (descending rotated eigenvalue) and one sign
    convention (each component's strongest feature loads positive), so
    reruns cannot come back mirror-flipped or shuffled. ``retention`` says
    how the count was chosen: the rule, and for parallel analysis the
    unrotated eigenvalues beside the chance thresholds up to the first one
    that fell short, so the decision can be read and argued with.

    ``n_components > 0`` is honored as asked; ``0`` chooses by ``retain``,
    one of :data:`RETAIN_RULES`. ``kaiser_cutoff`` is the eigenvalue a
    component has to beat under the Kaiser rule; 1.0 is the textbook value
    (one feature's worth of variance, since these are correlation-matrix
    eigenvalues), and raising it is the usual answer to that rule keeping
    almost everything on a wide matrix.
    """
    kept, mu, sigma, eigvals, eigvecs = _decompose(n, sums, cross)
    k, retention = _retain_count(
        n, kept.size, eigvals, n_components=n_components, retain=retain,
        kaiser_cutoff=kaiser_cutoff, on_progress=on_progress)
    loadings, projection, comp_eig, pct = _axes_at(
        kept, eigvals, eigvecs, k, rotation=rotation)
    return kept, mu, sigma, loadings, projection, comp_eig, pct, retention


def describe_retention(retention: dict) -> str:
    """One sentence on how the component count was chosen, for a report."""
    rule = retention.get("rule")
    if rule == "parallel":
        eig = retention.get("unrotated_eigenvalues") or []
        thr = retention.get("thresholds") or []
        k = max(1, len(eig) - 1) if len(eig) > 1 else len(eig)
        tail = ""
        if len(eig) > k:
            tail = (f"; the next eigenvalue, {eig[k]:.2f}, fell short of its "
                    f"chance level of {thr[k]:.2f}")
        return (f"chosen by parallel analysis: a component is kept while its "
                f"eigenvalue beats the {retention.get('percentile', 95):g}th "
                f"percentile of {retention.get('draws', PARALLEL_DRAWS)} random "
                f"data sets of the same size{tail}")
    if rule == "kaiser":
        cutoff = retention.get("cutoff", 1.0)
        return ("chosen by the Kaiser rule: every component with an "
                f"eigenvalue above {cutoff:g}")
    return "the number asked for"


def project_row(cells, kept, mu, sigma, projection, rounding: int) -> list:
    """One row's component scores from raw feature cells -- shared by every
    fit and every apply, so the two can never drift."""
    import numpy as np

    x = np.asarray(cells, dtype=np.float64)
    scores = ((x[kept] - mu) / sigma) @ projection
    return [round(float(s), rounding) for s in scores]


# ---------------------------------------------------------------------------
# the generic tool: reduce any old feature table
# ---------------------------------------------------------------------------

def _read_header(path: Path, encoding: str) -> List[str]:
    with path.open("r", newline="", encoding=encoding) as f:
        return next(csv.reader(f))


def fit_pca_csv(
    input_csv: PathLike,
    *,
    start_col: int = 2,
    n_components: int = 0,
    retain: Literal["parallel", "kaiser"] = "parallel",
    kaiser_cutoff: float = 1.0,
    rotation: bool = True,
    out_scores_csv: Optional[PathLike] = None,
    out_model_json: Optional[PathLike] = None,
    out_loadings_csv: Optional[PathLike] = None,
    out_eigenvalues_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
) -> Path:
    """
    Fit a varimax-rotated PCA to the numeric columns of any feature CSV.

    Parameters
    ----------
    input_csv
        Any wide feature table: identifier column(s) first, numeric feature
        columns after. Every Taters feature file (dictionary scores,
        readability, a document-term matrix) has this shape.
    start_col : int, default=2
        1-based position of the first *feature* column, exactly as in the
        house dask script this replaces. Everything before it is carried
        into the scores file unchanged as identifiers. The default fits
        tables shaped ``text_id, <features...>``.
    n_components : int, default=0
        ``0`` picks automatically by ``retain``; an explicit number is
        honored up to the count of non-constant features.
    retain : {"parallel", "kaiser"}, default="parallel"
        How ``0`` chooses. ``parallel`` is parallel analysis: a component is
        kept while its eigenvalue beats the 95th percentile of what fifty
        random data sets of the same size produce at the same rank, so
        chance structure is not kept. ``kaiser`` keeps every eigenvalue above
        ``kaiser_cutoff``, which at the textbook 1.0 keeps most of them on a
        wide table. The decision is written into the model file so it can be
        read and argued with.
    kaiser_cutoff : float, default=1.0
        The eigenvalue a component has to beat under ``retain="kaiser"``.
        1.0 is the textbook value -- one feature's worth of variance -- and
        raising it is the usual answer to that rule keeping almost
        everything on a wide matrix.
    rotation : bool, default=True
        Varimax-rotate. Off gives the raw principal axes.
    out_scores_csv, out_model_json, out_loadings_csv, out_eigenvalues_csv
        Defaults: ``<input>_pca_scores.csv`` (identifiers + ``Component_1..Component_k``)
        beside the input, with the model / loadings / eigenvalue files named
        the same way.
    overwrite_existing : bool, default=False
        If ``False`` and the scores file already exists, skip and return it.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    rounding : int, default=4
        Decimal places for scores, loadings, and eigenvalues.

    Returns
    -------
    Path
        ``out_scores_csv``.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    if start_col < 1:
        raise ValueError(f"start_col is 1-based; got {start_col}")

    stem = input_csv.stem
    scores_path = Path(out_scores_csv) if out_scores_csv else \
        input_csv.with_name(f"{stem}_pca_scores.csv")
    model_path = Path(out_model_json) if out_model_json else \
        input_csv.with_name(f"{stem}_pca_model.json")
    loadings_path = Path(out_loadings_csv) if out_loadings_csv else \
        input_csv.with_name(f"{stem}_pca_loadings.csv")
    eigen_path = Path(out_eigenvalues_csv) if out_eigenvalues_csv else \
        input_csv.with_name(f"{stem}_pca_eigenvalues.csv")
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    from ._common import reusable

    if reusable(scores_path, input_csv, overwrite_existing=overwrite_existing,
                verbose=True, what="the PCA scores"):
        print("PCA scores output file already exists; returning existing file.")
        return scores_path

    header = _read_header(input_csv, encoding)
    if start_col > len(header):
        raise ValueError(f"start_col={start_col} but {input_csv} has only "
                         f"{len(header)} column(s).")
    id_names = header[:start_col - 1]
    features = header[start_col - 1:]

    warn_if_wide(len(features))
    n, sums, cross = stream_moments(input_csv, encoding=encoding,
                                    skip_cols=start_col - 1,
                                    on_progress=on_progress)
    if n < 3:
        raise ValueError(f"Only {n} row(s) in {input_csv}; PCA needs data.")
    announce(on_progress, "extracting components")
    kept, mu, sigma, loadings, projection, eig, pct, retention = fit_axes(
        n, sums, cross, n_components=n_components, rotation=rotation,
        retain=retain, kaiser_cutoff=kaiser_cutoff,
        on_progress=on_progress)
    k = projection.shape[1]
    comp_names = component_names(k)

    kept_list = kept.tolist()
    dropped = [features[j] for j in range(len(features))
               if j not in set(kept_list)]
    if dropped:
        import warnings
        warnings.warn(
            f"{len(dropped)} constant feature column(s) carried no signal "
            f"and were left out: {', '.join(dropped[:8])}"
            f"{'…' if len(dropped) > 8 else ''}")

    with atomic_write(loadings_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["feature", *comp_names])
        for i, j in enumerate(kept_list):
            writer.writerow([features[j],
                             *(round(float(v), rounding) for v in loadings[i])])
    with atomic_write(eigen_path, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["component", "eigenvalue", "pct_variance"])
        for name, e, p in zip(comp_names, eig, pct):
            writer.writerow([name, round(float(e), rounding),
                             round(float(p), rounding)])

    from datetime import date

    # this kind is deliberately *not* in `helpers/model_spec.MODEL_TYPES`, so
    # it cannot be imported into the library or picked by "score with a saved
    # model". everywhere else in taters a PCA rides inside the instrument that
    # used it -- MEM's themes are its projection, and a ridge fitted on
    # components carries its reduction and replays it when it scores. this
    # pair is the one freestanding PCA, and it is a Python-API tool. a test
    # pins the exemption, so registering it is a deliberate decision rather
    # than something to discover later.
    model = {
        "kind": "taters-pca-model",
        "format": PCA_MODEL_FORMAT,
        "created": date.today().isoformat(),
        "taters": taters_version(),
        "features": features,
        "model": {"n_rows": n, "rotation": bool(rotation), "retention": retention,
                  "components": comp_names, "kept": kept_list,
                  "mu": mu.tolist(), "sigma": sigma.tolist(),
                  "projection": projection.tolist(),
                  "eigenvalues": eig.tolist(),
                  "pct_variance": pct.tolist()},
    }
    with atomic_write(model_path, encoding="utf-8") as f:
        json.dump(model, f, indent=1)

    ticker = Ticker(on_progress, n)
    with input_csv.open("r", newline="", encoding=encoding) as f, \
            atomic_write(scores_path, newline="", encoding=encoding) as out:
        reader = csv.reader(f)
        next(reader)
        writer = csv.writer(out)
        writer.writerow([*id_names, *comp_names])
        for row in reader:
            ticker.tick(message="scoring components")
            writer.writerow([*row[:start_col - 1],
                             *project_row(row[start_col - 1:], kept, mu,
                                          sigma, projection, rounding)])
    return scores_path


def load_pca_model(model_json: PathLike) -> dict:
    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"model_json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("kind") != "taters-pca-model":
        raise ValueError(f"{path} is not a Taters PCA model file.")
    if int(model.get("format", 0)) > PCA_MODEL_FORMAT:
        raise ValueError(
            f"{path} was written by a newer Taters (model format "
            f"{model.get('format')}; this build reads up to "
            f"{PCA_MODEL_FORMAT}). Update Taters to use it.")
    return model


def apply_pca_csv(
    model_json: PathLike,
    input_csv: PathLike,
    *,
    out_scores_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
) -> Path:
    """
    Score a new feature table on a saved PCA model.

    The model's features are matched **by name** in the new table's header
    (any column order; extra columns pass through as identifiers), then each
    row is standardized with the *training* means and deviations and
    projected. A feature the model needs that the table lacks is refused --
    filling it with anything would silently move every score.

    Parameters
    ----------
    model_json
        A ``*_pca_model.json`` written by :func:`fit_pca_csv`.
    input_csv
        Any CSV whose header contains every feature the model was fit on.
    out_scores_csv : optional
        Defaults to ``<input>_pca_scores.csv`` beside the input.
    overwrite_existing : bool, default=False
        If ``False`` and the output already exists, skip and return it.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    rounding : int, default=4
        Decimal places for the component scores.

    Returns
    -------
    Path
        ``out_scores_csv``: the input's non-feature columns, then
        the model's components.
    """
    import numpy as np

    model = load_pca_model(model_json)
    features: List[str] = list(model["features"])
    fit = model["model"]
    kept = np.asarray(fit["kept"], dtype=int)
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)
    projection = np.asarray(fit["projection"], dtype=np.float64)
    comp_names = list(fit["components"])

    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    scores_path = Path(out_scores_csv) if out_scores_csv else \
        input_csv.with_name(f"{input_csv.stem}_pca_scores.csv")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    from ._common import reusable

    if reusable(scores_path, input_csv, model_json,
                overwrite_existing=overwrite_existing, verbose=True,
                what="the PCA scores"):
        print("PCA scores output file already exists; returning existing file.")
        return scores_path

    header = _read_header(input_csv, encoding)
    positions = {name: i for i, name in enumerate(header)}
    missing = [f for f in features if f not in positions]
    if missing:
        raise ValueError(
            f"{input_csv} is missing {len(missing)} feature(s) the model was "
            f"fit on ({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}). "
            "The new table must carry every original feature, by name."
        )
    feature_pos = [positions[f] for f in features]
    id_pos = [i for i, name in enumerate(header) if name not in set(features)]

    ticker = Ticker(on_progress, count_rows(input_csv, on_progress=on_progress))
    with input_csv.open("r", newline="", encoding=encoding) as f, \
            atomic_write(scores_path, newline="", encoding=encoding) as out:
        reader = csv.reader(f)
        next(reader)
        writer = csv.writer(out)
        writer.writerow([*(header[i] for i in id_pos), *comp_names])
        for row in reader:
            ticker.tick(message="scoring components")
            cells = [row[i] for i in feature_pos]
            writer.writerow([*(row[i] for i in id_pos),
                             *project_row(cells, kept, mu, sigma, projection,
                                          rounding)])
    return scores_path


# --- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# in-memory reduction, for an analysis that wants components instead of features
# ---------------------------------------------------------------------------

def fit_in_memory(x, *, n_components: int = 0, rotation: bool = True,
                  retain: str = "parallel", kaiser_cutoff: float = 1.0):
    """
    Fit the same axes as :func:`fit_pca_csv`, from a matrix already in memory.

    The analyses hold their design matrix in memory already -- the whole
    filtered table, since a reduction is fitted once per analysis before the
    analysis runs (see the planning decisions) -- so they cannot go through
    the CSV-streaming path, which reads a file. Same moments, same
    :func:`fit_axes`, same sign and order conventions, so a reduction fitted
    here and one fitted from a file agree.

    Returns the dict :func:`apply_in_memory` consumes, plus the loadings and
    eigenvalues a reader wants to inspect.
    """
    import numpy as np

    a = np.asarray(x, dtype=np.float64)
    n = int(a.shape[0])
    if n < 3:
        raise ValueError(
            f"a PCA needs at least three rows to find a direction in, got {n}.")
    sums = a.sum(axis=0)
    cross = a.T @ a
    kept, mu, sigma, loadings, projection, eigenvalues, pct, retention = fit_axes(
        n, sums, cross, n_components=n_components, rotation=rotation,
        retain=retain, kaiser_cutoff=kaiser_cutoff)
    return {
        "retention": retention,
        "kept": [int(i) for i in kept],
        "mu": [float(v) for v in mu],
        "sigma": [float(v) for v in sigma],
        "projection": [[float(v) for v in row] for row in projection],
        "loadings": [[float(v) for v in row] for row in loadings],
        "eigenvalues": [float(v) for v in eigenvalues],
        "pct_variance": [float(v) for v in pct],
        "n_components": int(projection.shape[1]),
    }


def apply_in_memory(x, axes: dict):
    """Project rows onto axes from :func:`fit_in_memory`.

    Standardized with the axes' **own** means and deviations, not ones
    recomputed from the rows being projected -- which is what lets a saved
    reduction score a new table on the original sample's scale.
    """
    import numpy as np

    a = np.asarray(x, dtype=np.float64)[:, axes["kept"]]
    mu = np.asarray(axes["mu"], dtype=np.float64)
    sigma = np.asarray(axes["sigma"], dtype=np.float64)
    return ((a - mu) / sigma) @ np.asarray(axes["projection"],
                                           dtype=np.float64)


def component_names(n: int, kind: str = "Component") -> list:
    """``Component_1 … Component_n`` -- one name everywhere. The file API
    wrote ``PC_1`` and the in-analysis reduction ``Component_1`` for the same
    thing, so a reader met two names for one idea in one results folder.
    ``Component_1 … Component_n`` -- what the analyses call them, so a
    loadings table and a results table can be read side by side.

    ``kind`` names them something more specific when the caller knows what is
    being reduced. Reducing a topic model's topics does not give you
    "components", it gives you supertopics, and a results table full of
    ``Component_3`` is one more thing to look up -- see
    :func:`taters.helpers.feature_columns.reduced_name`, which is what decides.
    """
    return [f"{kind}_{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"fit": fit_pca_csv, "apply": apply_pca_csv},
    description='Principal components of a feature table: fit and save, or score a new table with a saved model.',
    aliases={
    },
    positional=("model_json", "input_csv"),
    legacy={"--no-rotation": ["--rotation", "false"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
