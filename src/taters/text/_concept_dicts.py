"""
Concepts for word vectors, spelled as LIWC-22 dictionaries.

A concept is a weighted set of words -- *death, dying, grave* -- and the
similarity between a text's vector and the concept's mean vector is one
number a hypothesis can rest on. The natural way to write such sets down
is the dictionary format researchers already keep their word lists in:
columns are categories, rows are terms, a cell is the term's weight in
that category (``X`` is 1.0; ``2`` counts a word twice as heavily), with
``*`` wildcards and multi-word phrases exactly as LIWC reads them. One
dictionary gives one ``sim_`` column per category; several dictionaries
give several sets, each prefixed with the dictionary's name the way the
dictionary analyzer prefixes its categories.

The files are read by ``contentcoder`` -- the same package the dictionary
analyzer scores with -- so a ``.dic``, ``.dicx`` or ``.csv`` that counts
words there means the same thing here, wildcards, phrases and weights
included. Nothing in this module parses a dictionary itself.

How a term meets a vocabulary
-----------------------------
LIWC resolves a term against *text*; here it is resolved against the
model's *vocabulary*, the same way:

* a plain word is tokenized with the model's own stream (so a lemmatised
  model matches ``dying`` as ``die``) and looked up;
* a wildcard word (``abrad*``) matches every vocabulary word its compiled
  pattern accepts -- a trailing-asterisk prefix by bisection on the sorted
  vocabulary, anything fancier by the regex contentcoder compiled;
* a phrase (``see-through``, ``butt bag*``) is split into words, each word
  resolved as above, and the phrase's vector is the mean of its words'.

Each dictionary *term* contributes one vector -- a wildcard that matches
forty words is still one term, not forty votes -- and the category's
vector is the weighted mean over its matched terms, normalized by the sum
of the absolute weights (so a negative weight pushes away, and ``1`` against
``-1`` is a contrast rather than a cancellation). A category that matches
nothing, or whose matched terms all weigh zero, is refused by name: its
column would be blank for every text.
"""
from __future__ import annotations

import bisect
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

__all__ = ["ConceptDict", "load_concept_dicts", "concept_dicts_from_json",
           "concept_dicts_to_json", "concept_column", "resolve_concepts",
           "expand_dict_paths", "DICT_SUFFIXES"]

PathLike = Union[str, Path]
DICT_SUFFIXES = (".dic", ".dicx", ".csv")


@dataclass
class ConceptDict:
    """One dictionary: its name (the file stem, tidied) and, per category
    in file order, the ``(term, weight)`` pairs that belong to it."""

    name: str
    categories: "OrderedDict[str, List[Tuple[str, float]]]" = field(
        default_factory=OrderedDict)


def _prefix_from_name(stem: str) -> str:
    """The dictionary analyzer's spelling of a dictionary's name as a column
    prefix: letters, digits and underscores, nothing else."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(stem)).strip("_") or "dict"


def concept_column(dict_name: str, category: str) -> str:
    """``sim_<dictionary>__<category>`` -- always both, so a table with two
    dictionaries says which one each column came from."""
    return f"sim_{_prefix_from_name(dict_name)}__{category}"


def expand_dict_paths(paths: Sequence[PathLike]) -> List[Path]:
    """Files as given, folders as every dictionary file inside them (sorted),
    duplicates dropped, order kept."""
    out: List[Path] = []
    seen = set()
    for item in paths or ():
        p = Path(str(item))
        found = sorted(f for f in p.rglob("*") if f.is_file()
                       and f.suffix.lower() in DICT_SUFFIXES) if p.is_dir() else [p]
        for f in found:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


def load_concept_dicts(paths: Sequence[PathLike]) -> List[ConceptDict]:
    """
    Read dictionaries through ``contentcoder`` and keep their terms and
    weights per category.

    A file that is not a dictionary is refused with the library gate's own
    words (the same check the dictionary analyzer runs), so a bare word
    list saved as ``.dic`` says so instead of contributing nothing.
    """
    from ..helpers.library import asset_problem

    out: List[ConceptDict] = []
    for path in expand_dict_paths(paths):
        if not path.is_file():
            raise FileNotFoundError(f"concept dictionary not found: {path}")
        problem = asset_problem(path)
        if problem:
            raise ValueError(problem)
        try:
            import contextlib
            import io

            from contentcoder.ContentCoder import ContentCoder

            # contentcoder prints "Dictionary loaded." to stdout every single time,
            # which would land right in the middle of the wizard's live display.
            # so, we swallow it
            with contextlib.redirect_stdout(io.StringIO()):
                coder = ContentCoder(dicFilename=str(path), fileEncoding="utf-8-sig")
        except Exception as e:      # contentcoder's errors don't say which file
            raise ValueError(f"'{path.name}' could not be read as a dictionary "
                             f"({type(e).__name__}: {e}).") from None
        loaded = coder.dict
        categories: "OrderedDict[str, List[Tuple[str, float]]]" = OrderedDict(
            (str(cat), []) for cat in loaded.catNames)
        # dictTermCatMap is term -> {category: weight}. we flip it around so
        # that each category lists its terms in the dictionary's own order
        for term, cats in loaded.dictTermCatMap.items():
            for cat, weight in cats.items():
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    w = 1.0
                categories.setdefault(str(cat), []).append((str(term), w))
        out.append(ConceptDict(name=path.stem, categories=categories))
    return out


def concept_dicts_to_json(dicts: Sequence[ConceptDict]) -> List[dict]:
    """The shape a model file stores, so a model travels with its concepts."""
    return [{"name": d.name,
             "categories": {cat: [[t, w] for t, w in terms]
                            for cat, terms in d.categories.items()}}
            for d in dicts]


def concept_dicts_from_json(items) -> List[ConceptDict]:
    out: List[ConceptDict] = []
    for item in items or ():
        cats: "OrderedDict[str, List[Tuple[str, float]]]" = OrderedDict()
        for cat, terms in (item.get("categories") or {}).items():
            cats[str(cat)] = [(str(t), float(w)) for t, w in terms]
        out.append(ConceptDict(name=str(item.get("name") or "dict"), categories=cats))
    return out


# ---------------------------------------------------------------------------
# Resolving terms against a vocabulary
# ---------------------------------------------------------------------------

class _Vocabulary:
    """The model's words, indexed for exact, prefix and pattern lookups."""

    def __init__(self, index: Dict[str, int]):
        self.index = index
        self.sorted = sorted(index)

    def exact(self, word: str) -> List[int]:
        i = self.index.get(word)
        return [i] if i is not None else []

    def prefix(self, stem: str) -> List[int]:
        # a trailing asterisk is by far the most common wildcard, and a bisect
        # over our sorted vocabulary beats running a regex over 200k words
        lo = bisect.bisect_left(self.sorted, stem)
        out = []
        while lo < len(self.sorted) and self.sorted[lo].startswith(stem):
            out.append(self.index[self.sorted[lo]])
            lo += 1
        return out

    def pattern(self, regex) -> List[int]:
        return [i for w, i in self.index.items() if regex.match(w)]


def _word_rows(word: str, vocab: _Vocabulary, stream, lowercase: bool) -> List[int]:
    """Vocabulary rows one dictionary word stands for."""
    from contentcoder.ContentCodingDictionary import compileWildcard, containsWildcard

    w = word.lower() if lowercase else word
    if containsWildcard(w):
        body = w.replace("*", "")
        if w.endswith("*") and "*" not in w[:-1] and body:
            return vocab.prefix(body)
        return vocab.pattern(compileWildcard(w))
    # a plain word goes through the model's own tokenizer so that a lemmatised
    # model sees "dying" as "die" -- the same way it saw the corpus
    forms = list(stream(w)) or [w]
    forms = [f.lower() for f in forms] if lowercase else forms
    rows: List[int] = []
    for form in forms:
        rows += vocab.exact(form)
    if not rows:
        rows = vocab.exact(w)
    return rows


def resolve_concepts(index: Dict[str, int], vectors, dicts: Sequence[ConceptDict],
                     stream: Callable[[str], List[str]], lowercase: bool
                     ) -> Tuple[List[str], list, List[dict]]:
    """
    Every category of every dictionary as a column name and a vector.

    Returns ``(columns, vectors, rows)``: the ``sim_`` column names in
    dictionary-then-category order, one mean vector per column, and one
    report row per category (``dictionary, category, n_terms, n_matched,
    weight_sum, matched, missed``) -- the evidence that the concept is made
    of words the model actually has.
    """
    import numpy as np

    vocab = _Vocabulary(index)
    columns: List[str] = []
    out_vectors: list = []
    rows: List[dict] = []
    for d in dicts:
        for cat, terms in d.categories.items():
            weighted: List[Tuple[float, object]] = []
            matched_forms: List[str] = []
            missed: List[str] = []
            for term, weight in terms:
                words = [w for w in str(term).split() if w]
                per_word = []
                for word in words:
                    idx = _word_rows(word, vocab, stream, lowercase)
                    if not idx:
                        per_word = []
                        break
                    per_word.append(np.asarray(vectors[sorted(set(idx))],
                                               dtype=np.float64).mean(axis=0))
                if not per_word:
                    missed.append(term)
                    continue
                # a phrase is just the mean of its words; a single word is itself
                weighted.append((float(weight), np.mean(per_word, axis=0)))
                matched_forms.append(term)
            total = sum(abs(w) for w, _v in weighted)
            column = concept_column(d.name, cat)
            if not weighted:
                raise ValueError(
                    f"none of the {len(terms)} term(s) of {cat!r} in "
                    f"{d.name} is in the model's vocabulary, so {column} would be "
                    f"blank for every text. Use a dictionary whose words the "
                    f"corpus has, or drop that category.")
            if total == 0:
                raise ValueError(
                    f"the weights of {cat!r} in {d.name} add up to zero over the "
                    f"terms the model knows, so {column} has no direction.")
            vector = sum(w * v for w, v in weighted) / total
            columns.append(column)
            out_vectors.append(vector)
            rows.append({"dictionary": d.name, "category": cat, "column": column,
                         "n_terms": len(terms), "n_matched": len(weighted),
                         "weight_sum": total,
                         "matched": matched_forms, "missed": missed})
    return columns, out_vectors, rows
