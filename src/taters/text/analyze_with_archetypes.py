from pathlib import Path
from typing import Callable, Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from ..helpers.find_files import find_files
from ..helpers.nltk_data import ensure_punkt
from ..helpers.progress import Ticker, count_rows
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec

FEATURE_COLUMNS = ColumnSpec(
    label="Archetypes",
    dynamic="one column per archetype in the file the user picked",
)

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  bookkeeping=("WC",),
                  assets={"archetype_csvs": "archetypes"})
def analyze_with_archetypes(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,   # if given, we skip gathering
    gathered_csv: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    workers: int = 0,  # if the file already exists, let's not overwrite by default

    # ----- Archetype CSVs (one or more) -----
    archetype_csvs: Sequence[Union[str, Path]],

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",
    delimiter: str = ",",

    # ====== CSV GATHER OPTIONS (when csv_path is provided) ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS (when txt_dir is provided) ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== Archetyper scoring options ======
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    device: Optional[str] = "auto",
    mean_center_vectors: bool = True,
    fisher_z_transform: bool = False,
    rounding: int = 4,
) -> Path:
    """
    Compute archetype scores for text rows and write a wide, analysis-ready features CSV.

    This function supports three input modes:

    1. ``analysis_csv`` — Use a prebuilt CSV with exactly two columns: ``text_id`` and ``text``.
    2. ``csv_path`` — Gather text from an arbitrary CSV by specifying ``text_cols`` (and optionally
    ``id_cols`` and ``group_by``) to construct an analysis-ready CSV on the fly.
    3. ``txt_dir`` — Gather text from a folder of ``.txt`` files.

    Archetype scoring is delegated to a middle layer that embeds text with a Sentence-Transformers
    model and evaluates cosine similarity to one or more archetype CSVs. If ``out_features_csv`` is
    omitted, the default path is ``./features/archetypes/<analysis_ready_filename>``.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV for gathering. Mutually exclusive with ``txt_dir`` and ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder of ``.txt`` files to gather from. Mutually exclusive with the other input modes.
    analysis_csv : str or pathlib.Path, optional
        Precomputed analysis-ready CSV containing exactly the columns ``text_id`` and ``text``.
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
        real bar instead of a spinner. Injected automatically by the pipeline
        runner for any step function that declares this parameter. See
        :mod:`taters.helpers.progress` for the contract.
    out_features_csv : str or pathlib.Path, optional
        Output path for the features CSV. If ``None``, defaults to
        ``./features/archetypes/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip recomputation and return the existing path.
        This also controls the intermediate analysis-ready CSV: when ``True``, it is rebuilt
        from the current source instead of reusing a stale copy from an earlier run.
    archetype_csvs : Sequence[str or pathlib.Path]
        One or more archetype CSVs (name → seed phrases). Directories are allowed and expanded
        recursively to all ``.csv`` files.
    encoding : str, default="utf-8-sig"
        Text encoding for CSV I/O.
    delimiter : str, default=","
        Field delimiter for CSV I/O.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV: column(s) that contain text. Used only if ``csv_path`` is provided.
    id_cols : Sequence[str], optional
        When gathering from a CSV: optional ID columns to carry into grouping (e.g., ``["speaker"]``).
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple ``text_cols`` are provided. ``"concat"`` joins into a single
        text field; ``"separate"`` creates one row per text column.
    group_by : Sequence[str], optional
        Optional grouping keys used during gathering (e.g., ``["speaker"]``). In ``"concat"`` mode,
        members are concatenated into one row per group.
    joiner : str, default=" "
        Separator used when concatenating multiple text chunks.
    num_buckets : int, default=512
        Number of temporary hash buckets used for scalable CSV gathering.
    max_open_bucket_files : int, default=64
        Maximum number of bucket files to keep open concurrently during gathering.
    tmp_root : str or pathlib.Path, optional
        Root directory for temporary files used by gathering.
    recursive : bool, default=True
        When gathering from a text folder, whether to recurse into subdirectories.
    pattern : str, default="*.txt"
        Filename glob used when gathering from a text folder.
    id_from : {"stem", "name", "path"}, default="stem"
        How to derive the ``text_id`` when gathering from a text folder.
    include_source_path : bool, default=True
        Whether to include the absolute source path as an additional column when gathering from a text folder.
    device : {"auto", "cuda", "cpu"} | None, default "auto"
        Where to run the embedding model. "auto" uses the GPU when torch reports
        one that works and falls back to the CPU when it does not; "cuda"
        insists and raises if it cannot; "cpu" never touches the GPU.
    model_name : str, default="sentence-transformers/all-roberta-large-v1"
        Sentence-Transformers model used to embed text for archetype scoring.
    mean_center_vectors : bool, default=True
        If ``True``, mean-center embedding vectors prior to scoring.
    fisher_z_transform : bool, default=False
        If ``True``, apply the Fisher z-transform to correlations.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.
    rounding : int, default=4
        Number of decimal places to round numeric outputs. Use ``None`` to disable rounding.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If an input file or folder does not exist, or an archetype CSV path is invalid.
    ValueError
        If required arguments are incompatible or missing (e.g., no input mode chosen),
        or if the analysis-ready CSV lacks ``text_id``/``text`` columns.

    Examples
    --------
    Run on a transcript CSV, grouped by speaker:

    >>> analyze_with_archetypes(
    ...     csv_path="transcripts/session.csv",
    ...     text_cols=["text"],
    ...     id_cols=["speaker"],
    ...     group_by=["speaker"],
    ...     archetype_csvs=["dictionaries/archetypes"],
    ...     model_name="sentence-transformers/all-roberta-large-v1",
    ... )
    PosixPath('.../features/archetypes/session.csv')

    Notes
    -----
    If ``out_features_csv`` exists and ``overwrite_existing=False``, the existing path is returned
    without recomputation. Directories passed in ``archetype_csvs`` are expanded recursively to
    all ``.csv`` files and deduplicated before scoring.
    """


    # archetyper splits text with nltk.sent_tokenize, which needs data that NLTK
    # doesn't ship. we grab it up front so that we don't fail mid-pipeline with
    # a wall of asterisks after the expensive steps have already run
    ensure_punkt(verbose=True)

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
        out_features_csv = Path.cwd() / "features" / "archetypes" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        print("Archetypes output file already exists; returning existing file.")
        return out_features_csv


    # 2) resolve/validate the archetype CSVs
    # we allow passing either:
    #   • one or more CSV files, or
    #   • one or more directories containing CSVs (recursively).
    #
    # we lean on the shared find_files helper so we're not reinventing it here

    # 2) resolve/validate the archetype CSVs
    resolved_archetype_csvs: list[Path] = []

    for src in archetype_csvs:
        src_path = Path(src)
        if src_path.is_dir():
            # find all *.csv under this folder (recursive)
            found = find_files(
                root_dir=src_path,
                extensions=[".csv"],
                recursive=True,
                absolute=True,
                sort=True,
            )
            resolved_archetype_csvs.extend(Path(f) for f in found)
        else:
            resolved_archetype_csvs.append(src_path)

    # de-dup, normalize, and sort
    archetype_csvs = sorted({p.resolve() for p in resolved_archetype_csvs})

    if not archetype_csvs:
        raise ValueError(
            "No archetype CSVs found. Pass one or more CSV files, or a directory containing CSV files with your archetypes."
        )
    for p in archetype_csvs:
        if not p.exists():
            raise FileNotFoundError(f"Archetype CSV not found: {p}")



        # 3) stream (text_id, text, meta) → middle layer → features CSV
    def _iter_items_from_csv_with_meta(
        path: Path,
        *,
        id_col: str = "text_id",
        text_col: str = "text",
        wanted: Optional[Sequence[str]] = None,
    ) -> Iterable[Tuple[str, str, dict]]:
        """
        Stream (text_id, text, meta) from an analysis-ready CSV.

        Enforces that all requested `wanted` columns exist (fail fast).
        """
        wanted = list(wanted or [])
        with path.open("r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = reader.fieldnames or []
            if id_col not in fields or text_col not in fields:
                raise ValueError(
                    f"Expected columns '{id_col}' and '{text_col}' in {path}; found {fields}"
                )
            missing = [c for c in wanted if c not in fields]
            if missing:
                raise ValueError(
                    f"Requested id_cols not present in analysis-ready CSV {path}: {missing}"
                )

            for row in reader:
                tid = str(row.get(id_col, "") or "")
                txt = str(row.get(text_col, "") or "")
                meta = {c: str(row.get(c, "") or "") for c in wanted}
                # we name each tick because these rows are wildly uneven: one
                # book-sized document can take minutes where its neighbors
                # take milliseconds, and an unnamed pause that long looks
                # like a hang
                _ticker.tick(message=f"scoring {tid}")
                yield tid, txt, meta


    # the middle layer pulls the generator above lazily and writes as it goes,
    # so one yield is one row's worth of work handed over
    _ticker = Ticker(on_progress, count_rows(analysis_ready, on_progress=on_progress))

    # we import this late for the same reason the analyzer defers `archetypes`:
    # this module gets imported to read its signature far more often than it
    # actually runs
    from .dictionary_analyzers import multi_archetype_analyzer as maa

    # there's one shared rule for what rides along beside text_id -- see
    # resolve_passthrough_columns: same order of preference that every per-row
    # analyzer uses, and the same two columns that never get carried
    from ..helpers.row_map import resolve_passthrough_columns

    with analysis_ready.open("r", newline="", encoding=encoding) as _fh:
        _header = csv.DictReader(_fh).fieldnames or []
    passthrough = resolve_passthrough_columns(
        _header, id_cols=id_cols, group_by=group_by,
        analysis_ready=analysis_ready)

    maa.analyze_texts_to_csv(
        items=_iter_items_from_csv_with_meta(analysis_ready, wanted=passthrough),
        archetype_csvs=archetype_csvs,
        out_csv=out_features_csv,
        model_name=model_name,
        device=device,
        mean_center_vectors=mean_center_vectors,
        fisher_z_transform=fisher_z_transform,
        rounding=rounding,
        encoding=encoding,
        delimiter=delimiter,
        id_col_name="text_id",
        pass_through_cols=passthrough,  # these land right after text_id
        verbose=on_progress is None,
    )

    return out_features_csv



# --- CLI ------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_with_archetypes,
    description='Archetype scoring into a single CSV (globals once + per-archetype blocks).',
    aliases={
        'archetype_csvs': ['--archetype'],
        'csv_path': ['--csv'],
        'out_features_csv': ['--out'],
    },
    legacy={
        '--no-include-source-path': ['--include-source-path', 'false'],
        '--no-mean-center-vectors': ['--mean-center-vectors', 'false'],
        '--no-recursive': ['--recursive', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
