from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Optional, List, Dict
import csv
import sys
import re

# Use the public archetyper API only
from archetypes.archetypes import ArchetypeCollection, ArchetypeQuantifier  # be sure to pip install archetyper

# Allow very large CSV fields (handles huge text columns safely).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # macOS / some platforms can’t take sys.maxsize; fall back to 2^31-1
    csv.field_size_limit(2**31 - 1)

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
    items: Iterable[Tuple[str, str]],                 # (text_id, text)
    archetype_csvs: Sequence[PathLike],               # one or more CSV files of prototypes
    out_csv: PathLike,
    *,
    # archetyper options
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    mean_center_vectors: bool = True,
    fisher_z_transform: bool = False,
    # file options
    encoding: str = "utf-8-sig",
    file_has_headers: bool = True,
    delimiter: str = ",",
    # output options
    id_col_name: str = "text_id",
    rounding: int = 4,
    newline: str = "",
) -> Path:
    """
    Write a single CSV with one row per input text.
      Columns: [text_id, WC, <prefix>__<ArchetypeName>, ...] for each input CSV.
      Column order is deterministic: file order × quantifier.get_list_of_archetypes() order.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build a quantifier per CSV and capture stable archetype name order
    blocks: List[Tuple[str, List[str], ArchetypeQuantifier]] = []
    header: List[str] = [id_col_name, "WC"]

    for csv_path in archetype_csvs:
        pref = _prefix_from_path(csv_path)

        arch = ArchetypeCollection()
        arch.add_archetypes_from_CSV(
            filepath=str(csv_path),
            file_encoding=encoding,
            file_has_headers=file_has_headers
        )

        q = ArchetypeQuantifier(archetypes=arch, model=model_name)
        names = list(q.get_list_of_archetypes())  # stable, library-defined order

        blocks.append((pref, names, q))
        header.extend([f"{pref}__{n}" for n in names])

    # 2) Stream items → analyze → aggregate → write
    with out_csv.open("w", newline=newline, encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for text_id, text in items:
            # total WC from the first block’s sentence parse (consistent across blocks)
            total_wc_written = False
            total_wc_val = 0

            print(f"Analyzing for archetypes: {text_id}")

            all_scores: List[float] = []
            for pref, names, q in blocks:
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

            writer.writerow([text_id, total_wc_val, *all_scores])

    return out_csv
