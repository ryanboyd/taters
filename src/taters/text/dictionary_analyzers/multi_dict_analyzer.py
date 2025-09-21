from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, Optional
import csv
import re
import sys

# Public API from ContentCoder only.
try:
    from contentcoder.ContentCoder import ContentCoder
except ImportError:  # pragma: no cover
    from ContentCoder import ContentCoder  # type: ignore

PathLike = Union[str, Path]

# Allow very large CSV fields (handles huge text columns safely).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # macOS / some platforms can’t take sys.maxsize; fall back to 2^31-1
    csv.field_size_limit(2**31 - 1)


# ---- globals handled once (from FIRST dict) --------------------------------
GLOBAL_ONLY_FIELDS = {
    "WC",
    "BigWords",
    "Numbers",
    "AllPunct",
    "Period",
    "Comma",
    "QMark",
    "Exclam",
    "Apostro",
}

# ---- helpers ---------------------------------------------------------------

def _prefix_from_path(p: PathLike) -> str:
    stem = Path(p).stem
    return re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_") or "dict"

def _load_coders(dict_files: Sequence[PathLike]) -> List[Tuple[str, ContentCoder]]:
    coders: List[Tuple[str, ContentCoder]] = []
    for d in dict_files:
        d = Path(d)
        cc = ContentCoder(dicFilename=str(d), fileEncoding="utf-8-sig")
        coders.append((_prefix_from_path(d), cc))
    return coders

def _partition_indices(header: Sequence[str], *, keep_globals: bool) -> Tuple[List[int], List[int]]:
    """
    Return (global_idxs, per_dict_idxs) for a single dictionary header.
    - global_idxs: indices for WC/BigWords/Numbers/punctuation — included only when keep_globals=True
    - per_dict_idxs: indices for 'Dic' + categories (i.e., everything NOT in GLOBAL_ONLY_FIELDS)
    """
    global_idxs: List[int] = []
    per_idxs: List[int] = []
    for i, name in enumerate(header):
        if name in GLOBAL_ONLY_FIELDS:
            if keep_globals:
                global_idxs.append(i)
        else:
            per_idxs.append(i)
    return global_idxs, per_idxs

# ---- public API ------------------------------------------------------------

def build_header_for_dicts(
    dict_files: Sequence[PathLike], *, include_id_col: bool = True, id_col_name: str = "text_id"
) -> List[str]:
    """
    Return a stable header: unprefixed globals once (from first dict), then each
    dictionary's 'Dic'+category block prefixed by the dict filename stem.
    """
    coders = _load_coders(dict_files)
    if not coders:
        raise ValueError("No dictionaries provided.")
    
    dict_names = [pref for pref, _ in coders]
    print(
        f"Analyzing with dictionaries: {text_id if text_id is not None else ''}\n\t"
        + "\n\t".join(dict_names),
        flush=True,
    )

    header: List[str] = [id_col_name] if include_id_col else []

    # First dictionary: globals (unprefixed) + its per-dict block
    first_pref, first_cc = coders[0]
    h0 = list(first_cc.GetResultsHeader())
    g0, p0 = _partition_indices(h0, keep_globals=True)
    header.extend([h0[i] for i in g0])                       # globals once, unprefixed
    header.extend([f"{first_pref}__{h0[i]}" for i in p0])    # first dict block

    # Remaining dictionaries: only per-dict blocks
    for pref, cc in coders[1:]:
        h = list(cc.GetResultsHeader())
        _, p = _partition_indices(h, keep_globals=False)
        header.extend([f"{pref}__{h[i]}" for i in p])

    return header

def analyze_text_one_row(
    text: str,
    dict_files: Sequence[PathLike],
    *,
    include_id_col: bool = False,
    text_id: Optional[str] = None,
    relative_freq: bool = True,
    drop_punct: bool = True,
    rounding: int = 4,
    retain_captures: bool = False,
    wildcard_mem: bool = True,
) -> Tuple[List[str], List[Union[str, float]]]:
    """
    Analyze one text across N dictionaries; return (header, row).
    Globals appear once (from the first dict). Each dict contributes its 'Dic'+categories block.
    """
    coders = _load_coders(dict_files)
    if not coders:
        raise ValueError("No dictionaries provided.")

    header: List[str] = []
    row: List[Union[str, float]] = []

    if include_id_col:
        header.append("text_id")
        row.append(text_id if text_id is not None else "")

    # First dict
    first_pref, first_cc = coders[0]
    h0 = list(first_cc.GetResultsHeader())
    g0, p0 = _partition_indices(h0, keep_globals=True)
    res0 = first_cc.Analyze(
        text,
        relativeFreq=relative_freq,
        dropPunct=drop_punct,
        retainCaptures=retain_captures,
        returnTokens=False,
        wildcardMem=wildcard_mem,
    )
    v0 = list(first_cc.GetResultsArray(res0, rounding=rounding))

    header.extend([h0[i] for i in g0])                     # globals (unprefixed)
    row.extend([v0[i] for i in g0])

    header.extend([f"{first_pref}__{h0[i]}" for i in p0])  # per-dict block
    row.extend([v0[i] for i in p0])

    # Remaining dicts: per-dict blocks only
    for pref, cc in coders[1:]:
        h = list(cc.GetResultsHeader())
        _, p = _partition_indices(h, keep_globals=False)
        res = cc.Analyze(
            text,
            relativeFreq=relative_freq,
            dropPunct=drop_punct,
            retainCaptures=retain_captures,
            returnTokens=False,
            wildcardMem=wildcard_mem,
        )
        v = list(cc.GetResultsArray(res, rounding=rounding))
        header.extend([f"{pref}__{h[i]}" for i in p])
        row.extend([v[i] for i in p])

    return header, row

def analyze_texts_to_csv(
    items: Iterable[Tuple[str, str]],
    dict_files: Sequence[PathLike],
    out_csv: PathLike,
    *,
    relative_freq: bool = True,
    drop_punct: bool = True,
    rounding: int = 4,
    retain_captures: bool = False,
    wildcard_mem: bool = True,
    id_col_name: str = "text_id",
    newline: str = "",
    encoding: str = "utf-8-sig",
) -> Path:
    """
    Write a single CSV with one row per (text_id, text).
    Header = [text_id] + globals-once + per-dict blocks (prefixed).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    coders = _load_coders(dict_files)
    if not coders:
        raise ValueError("No dictionaries provided.")
    
    dict_names = [pref for pref, _ in coders]

    # Header + index plans
    first_pref, first_cc = coders[0]
    h0 = list(first_cc.GetResultsHeader())
    g0, p0 = _partition_indices(h0, keep_globals=True)

    header: List[str] = [id_col_name]
    header.extend([h0[i] for i in g0])
    header.extend([f"{first_pref}__{h0[i]}" for i in p0])

    # Plans for remaining dicts
    plans: List[Tuple[str, List[int], List[str], ContentCoder]] = []
    for pref, cc in coders[1:]:
        h = list(cc.GetResultsHeader())
        _, p = _partition_indices(h, keep_globals=False)
        header.extend([f"{pref}__{h[i]}" for i in p])
        plans.append((pref, p, h, cc))

    with out_csv.open("w", newline=newline, encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for text_id, text in items:
            row_out: List[Union[str, float]] = [text_id]

            print(
                f"Analyzing with dictionaries: {text_id}\n\t" + "\n\t".join(dict_names),
                flush=True,
            )
            row_out: List[Union[str, float]] = [text_id]

            # First dict (globals + per-dict)
            res0 = first_cc.Analyze(
                text,
                relativeFreq=relative_freq,
                dropPunct=drop_punct,
                retainCaptures=retain_captures,
                returnTokens=False,
                wildcardMem=wildcard_mem,
            )
            v0 = list(first_cc.GetResultsArray(res0, rounding=rounding))
            row_out.extend([v0[i] for i in g0])
            row_out.extend([v0[i] for i in p0])

            # Remaining dicts (per-dict only)
            for _, per_idxs, _, cc in plans:
                res = cc.Analyze(
                    text,
                    relativeFreq=relative_freq,
                    dropPunct=drop_punct,
                    retainCaptures=retain_captures,
                    returnTokens=False,
                    wildcardMem=wildcard_mem,
                )
                v = list(cc.GetResultsArray(res, rounding=rounding))
                row_out.extend([v[i] for i in per_idxs])

            writer.writerow(row_out)

    return out_csv
