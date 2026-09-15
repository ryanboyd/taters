from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Optional, List, Dict
import csv
import re

# we only ever touch archetyper's public API
from ...helpers.atomic import atomic_write
from ...helpers.gpu import hidden_gpu, resolve_device
from ...helpers.csvio import widen_csv_field_limit

widen_csv_field_limit()

PathLike = Union[str, Path]

# ---------------- helpers ----------------

def _prefix_from_path(p: PathLike) -> str:
    stem = Path(p).stem
    return re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_") or "archetypes"

def _wc_weighted_mean(sentence_results: Sequence[object], names: Sequence[str], rounding: int) -> Tuple[int, List[float]]:
    """
    Collapse sentence-level results to document-level via WC-weighted mean.
    Returns (total_wc, scores_in_names_order).
    """
    total_wc = 0.0
    sums: Dict[str, float] = {n: 0.0 for n in names}

    for r in sentence_results:
        wc = float(getattr(r, "WC", 0) or 0)
        scores = getattr(r, "archetype_scores", {}) or {}
        if wc <= 0:
            continue
        total_wc += wc
        for n in names:
            sums[n] += wc * float(scores.get(n, 0.0))

    if total_wc <= 0:
        return 0, [0.0 for _ in names]

    out = [round(sums[n] / total_wc, rounding) for n in names]
    return int(total_wc), out

# ---------------- public API ----------------

def analyze_texts_to_csv(
    items: Iterable[Tuple[str, str]] | Iterable[Tuple[str, str, dict]],
    archetype_csvs: Sequence[PathLike],
    out_csv: PathLike,
    *,
    # archetyper options
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    device: Optional[str] = "auto",
    mean_center_vectors: bool = True,
    fisher_z_transform: bool = False,
    # file options
    encoding: str = "utf-8-sig",
    file_has_headers: bool = True,
    delimiter: str = ",",
    # output options
    id_col_name: str = "text_id",
    pass_through_cols: Sequence[str] = (),
    rounding: int = 4,
    newline: str = "",
    verbose: bool = True,
) -> Path:
    """
    Write a single CSV with one row per input text.

    Header layout:
        [id_col_name] + [<pass_through_cols...>] + ["WC"] + [<prefix>__<ArchetypeName> ...] × each input CSV

    Parameters
    ----------
    items
        Either (text_id, text) or (text_id, text, meta_dict). If meta_dict is provided,
        values are written in the same order as `pass_through_cols` (missing keys → "").
    pass_through_cols
        Column names to inject immediately after `id_col_name` (e.g., ["source","speaker"]).
    """
    # we import this here rather than at the top of the module. `archetypes`
    # pulls in sentence-transformers (and so torch), which takes about ten
    # seconds, and the wizard imports this module just to read the function
    # signature when it builds the options screen. paying for torch just to
    # learn the parameter names made that screen look like it had hung
    from archetypes.archetypes import (  # be sure to pip install archetyper
        ArchetypeCollection, ArchetypeQuantifier,
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) first, we build a quantifier per CSV and hang on to the archetype
    #    names in a stable order
    blocks: List[Tuple[str, List[str], ArchetypeQuantifier]] = []
    # our header starts with the id, then the pass-through columns, then WC
    header: List[str] = [id_col_name, *list(pass_through_cols), "WC"]

    for csv_path in archetype_csvs:
        pref = _prefix_from_path(csv_path)

        arch = ArchetypeCollection()
        arch.add_archetypes_from_CSV(
            filepath=str(csv_path),
            file_encoding=encoding,
            file_has_headers=file_has_headers
        )

        # `ArchetypeQuantifier` builds its own `SentenceTransformer` and doesn't
        # take a device, so the only way we can honor device="cpu" is to hide
        # the GPU while it runs. it also encodes the archetype sentences right
        # in its constructor, so we have to wrap the construction itself --
        # moving the model afterwards would be too late
        resolved, fallback_reason = resolve_device(device, backend="torch")
        if verbose:
            print(f"[archetypes] Loading {model_name} on {resolved}")
            if fallback_reason:
                print(f"[archetypes] {fallback_reason}")
        with hidden_gpu(resolved == "cpu"):
            q = ArchetypeQuantifier(archetypes=arch, model=model_name)
        names = list(q.get_list_of_archetypes())  # the library's order; it's stable

        blocks.append((pref, names, q))
        header.extend([f"{pref}__{n}" for n in names])

    # 2) now we stream: analyze each item, aggregate, and write as we go
    with atomic_write(out_csv, newline=newline, encoding=encoding) as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)

        for it in items:
            # we take either (text_id, text) or (text_id, text, meta)
            if len(it) == 2:
                text_id, text = it  # type: ignore[misc]
                meta: Dict[str, str] = {}
            elif len(it) == 3:
                text_id, text, meta = it  # type: ignore[misc]
                if not isinstance(meta, dict):
                    raise ValueError("items third element must be a dict of pass-through values.")
            else:
                raise ValueError("Each item must be (text_id, text) or (text_id, text, meta_dict).")

            if verbose:
                print(f"Analyzing for archetypes: {text_id}")

            # WC comes from the first block's sentence parse (it's the same
            # across blocks anyway)
            total_wc_written = False
            total_wc_val = 0

            all_scores: List[float] = []
            for pref, names, q in blocks:
                if verbose:
                    print("\tAnalyzing for:\n\t\t" + "\n\t\t".join(names))
                q.analyze(
                    text,
                    mean_center_vectors=mean_center_vectors,
                    fisher_z_transform=fisher_z_transform,
                )
                wc, scores = _wc_weighted_mean(q.results, names, rounding=rounding)
                if not total_wc_written:
                    total_wc_val = wc
                    total_wc_written = True
                all_scores.extend(scores)

            # lastly, the row: id, pass-through values (in order), WC, scores...
            pt_vals = [str(meta.get(c, "")) for c in pass_through_cols]
            writer.writerow([text_id, *pt_vals, total_wc_val, *all_scores])

    return out_csv

