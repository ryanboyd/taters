"""
Text cohesion features (TAACO-style), one row per document.

The measure families of Crossley, Kyle & McNamara's TAACO (2016) and the
Coh-Metrix lineage behind it (Graesser, McNamara, Louwerse & Cai 2004),
reimplemented from their published definitions: lexical overlap between
adjacent sentences and paragraphs across nine word classes, lemma-based
type-token ratios and lexical density, and givenness. Column names follow
TAACO 2.1.3 wherever the measure survives, so results can be read against
the TAACO literature.

This is a REIMPLEMENTATION, not a port. The reference implementation
(`reference/TAACO-main`, CC BY-NC-SA -- nothing was copied) was audited
line by line and several of its defects are deliberately not reproduced.
Where behavior differs, the code comments say exactly what TAACO did and
what this does instead, and `COHESION_MEASURES.md` (shipped next to this
module) documents every column: formula, range, interpretation, lineage,
and differences. The headline corrections:

* Documents with too few segments emit **empty cells (NA), not 0.0** --
  TAACO scores a one-paragraph essay as "zero paragraph cohesion", which
  poisons any downstream average (5.6% of its own sample corpus).
* Real punctuation filtering: a token must contain a letter or digit.
  TAACO's punctuation list mixed POS tags with literal characters that can
  never match a tag, so ``%`` (Penn-tagged NN) counted as a *noun* in every
  noun, content and argument index, and stray quotes/hyphens inflated the
  word counts underneath every ratio.
* The two-segment windows are built by concatenation into fresh lists.
  TAACO appended the third segment *into its shared sentence list*, so
  computing one index corrupted the input of the next -- its published
  values depend on which checkboxes were ticked.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union)

from ..helpers.atomic import atomic_write
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.progress import announce
from ..helpers.text_gather import (resolve_analysis_ready)
from .ngram_prep import make_sentence_stream, pooled_text_workers
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Word classes
# ---------------------------------------------------------------------------

#: A token participates in any measure only if it contains a letter or a
#: digit. This is the punctuation filter TAACO meant to have: its list mixed
#: Penn tags with literal characters that can never equal a tag, so quotes,
#: hyphens and ``%`` flowed into the counts (``%`` as a noun, via tag NN).
_ALNUM = re.compile(r"[A-Za-z0-9]")

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_ADJ_TAGS = {"JJ", "JJR", "JJS"}
_ADV_TAGS = {"RB", "RBR", "RBS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}
_PRONOUN_TAGS = {"PRP", "PRP$"}
_DEMONSTRATIVES = {"this", "that", "these", "those"}

#: The nine classes, in TAACO's column order. Values are the printable names.
CLASS_ORDER = ("all", "cw", "fw", "noun", "verb", "adj", "adv",
               "pronoun", "argument")


def _adverb_is_content(word: str, lemma: str) -> bool:
    """
    Deadjectival adverbs ("quickly", "happily") count as content words;
    other adverbs ("very", "not", "again") as function words -- TAACO's
    distinction, kept. TAACO tested membership in a COCA-derived adjective
    list we cannot ship (license); instead: an ``-ly`` adverb whose stem has
    an adjective sense in WordNet is content. ("happily" -> "happi" ->
    "happy" via the -ily rule.)
    """
    if not word.endswith("ly") or len(word) <= 3:
        return False
    from nltk.corpus import wordnet

    stem = word[:-2]
    candidates = [stem]
    if stem.endswith("i"):
        candidates.append(stem[:-1] + "y")
    for cand in candidates:
        try:
            if wordnet.synsets(cand, pos="a") or wordnet.synsets(cand, pos="s"):
                return True
        except Exception:
            return False
    return False


def _demonstrative_attended(rest) -> bool:
    """
    Whether a demonstrative ("this", "that", ...) is *attended*: followed, past
    any punctuation, by a noun or adjective ("this car") rather than standing
    alone ("this is"). TAACO used the dependency parse; the next content tag
    is the tagger-only stand-in, and it was written out twice -- once keyed on
    the lemma, once on the word -- before it lived here.
    """
    for later_word, _later_lemma, later_tag in rest:
        if not _ALNUM.search(later_word):
            continue
        return later_tag in _NOUN_TAGS or later_tag in _ADJ_TAGS
    return False


def classify_sentence(triples: Sequence[Tuple[str, str, str]]) -> Dict[str, List[str]]:
    """
    One sentence's ``(word, lemma, tag)`` triples -> lemma lists per class.

    Class rules (all deviations from TAACO documented):

    * noun: NN/NNS/NNP/NNPS -- proper nouns included, as in TAACO.
    * pronoun: PRP/PRP$ plus *unattended demonstratives* -- this/that/these/
      those NOT followed by a noun or adjective ("I like *that*" yes,
      "*that* car" no). TAACO used the dependency parse for attendedness;
      this uses the next content-bearing token's tag, documented as a
      heuristic. "that" as a complementizer (tag IN) is neither.
    * verb: content verbs only, exactly as TAACO's verb indices are -- a
      modal (MD) or a form of "be" is a function word. TAACO additionally
      demoted auxiliary "have"/"do" via spaCy's AUX tag; without a parse we
      keep have/do as content, a documented deviation.
    * adv: deadjectival adverbs are content (see `_adverb_is_content`).
    * cw: nouns + adjectives + content verbs + content adverbs.
      fw: every other counted token.
    * argument: nouns + pronouns (TAACO's definition).
    """
    out: Dict[str, List[str]] = {c: [] for c in CLASS_ORDER}
    n = len(triples)
    for i, (word, lemma, tag) in enumerate(triples):
        if not _ALNUM.search(word):
            continue
        out["all"].append(lemma)

        content = False
        if tag in _NOUN_TAGS:
            out["noun"].append(lemma)
            out["argument"].append(lemma)
            content = True
        elif tag in _ADJ_TAGS:
            out["adj"].append(lemma)
            content = True
        elif tag in _VERB_TAGS:
            if tag != "MD" and lemma != "be":
                out["verb"].append(lemma)
                content = True
        elif tag in _ADV_TAGS:
            out["adv"].append(lemma)
            if _adverb_is_content(word, lemma):
                content = True
        elif tag in _PRONOUN_TAGS:
            out["pronoun"].append(lemma)
            out["argument"].append(lemma)
        elif lemma in _DEMONSTRATIVES and tag in ("DT", "WDT"):
            if not _demonstrative_attended(triples[i + 1:n]):
                # this is a pronominal use, so it joins the pronoun (and
                # argument) classes -- same as TAACO does with its unattended
                # demonstratives
                out["pronoun"].append(lemma)
                out["argument"].append(lemma)

        if content:
            out["cw"].append(lemma)
        else:
            out["fw"].append(lemma)
    return out


# ---------------------------------------------------------------------------
# Connectives
# ---------------------------------------------------------------------------

#: Where the shipped category lists live; the same files seed the library
#: kind, so "the shipped lists" and "an untouched library" are one thing.
_SHIPPED_CONNECTIVES = (Path(__file__).resolve().parent.parent
                        / "data" / "library" / "connectives")

#: TAACO's column order for the categories we ship; custom lists a user adds
#: to the library follow alphabetically.
_CONNECTIVE_ORDER = (
    "basic_connectives", "conjunctions", "disjunctions",
    "lexical_subordinators", "coordinating_conjuncts", "addition",
    "sentence_linking", "order", "reason_and_purpose", "all_causal",
    "positive_causal", "opposition", "determiners", "all_additive",
    "all_logical", "positive_logical", "negative_logical", "all_temporal",
    "positive_intentional", "all_positive", "all_negative", "all_connective",
)

#: One parsed entry: the phrase as a word tuple, plus an optional Penn-tag
#: constraint (single-word entries only) for ambiguous items -- "yet" counts
#: as opposition only when tagged CC, so "not yet" stays out. The constraint
#: is what replaces TAACO's dependency-parse caveats; TAACO's own caveat
#: mechanism was part dead code (causal caveats computed then ignored) and
#: part broken (an unsplit caveat STRING, so membership tests were substring
#: matches: "in" matched inside "since").
Entry = Tuple[Tuple[str, ...], Optional[frozenset]]


def load_connective_lists(paths: Optional[Sequence[PathLike]] = None
                          ) -> List[Tuple[str, List[Entry]]]:
    """
    Read connectives category files: one entry per line, ``#`` comments,
    optional TAB + ``TAG`` or ``TAG|TAG`` Penn constraint. ``None`` means the
    shipped categories; a directory means every ``.txt`` inside it. Each
    file becomes one output column named by its stem, ordered canonically
    (TAACO's order) with unknown stems appended alphabetically.
    """
    if paths is None:
        files = sorted(_SHIPPED_CONNECTIVES.glob("*.txt"))
    else:
        files = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files.extend(sorted(f for f in p.rglob("*.txt") if f.is_file()))
            else:
                files.append(p)
    rank = {name: i for i, name in enumerate(_CONNECTIVE_ORDER)}
    files.sort(key=lambda f: (rank.get(f.stem, len(rank)), f.stem))

    lists: List[Tuple[str, List[Entry]]] = []
    for path in files:
        entries: List[Entry] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            phrase, _, tags = line.partition("\t")
            words = tuple(phrase.lower().split())
            if not words:
                continue
            constraint = frozenset(t.strip() for t in tags.split("|")
                                   if t.strip()) if tags.strip() else None
            if constraint and len(words) > 1:
                raise ValueError(
                    f"{path.name}: a tag constraint is only meaningful on a "
                    f"single word, but {' '.join(words)!r} carries one.")
            entries.append((words, constraint))
        if not entries:
            raise ValueError(f"{path} contains no connective entries.")
        lists.append((path.stem, entries))
    if not lists:
        raise ValueError("No connectives lists were found.")
    return lists


def count_connectives(sentences: Sequence[Sequence[Tuple[str, str]]],
                      entries: Sequence[Entry]) -> int:
    """
    Occurrences of a category's entries over ``(word, tag)`` sentences.

    Matching is per position WITHIN a sentence -- two of TAACO's counting
    bugs are thereby structurally impossible: its `str.count` on the joined,
    punctuation-stripped document both undercounted adjacent repeats
    ("so so" counted once) and manufactured phrase matches across sentence
    boundaries ("...ends in. Fact is..." matched "in fact").
    """
    total = 0
    singles: Dict[str, List[Optional[frozenset]]] = {}
    phrases: List[Entry] = []
    for words, constraint in entries:
        if len(words) == 1:
            singles.setdefault(words[0], []).append(constraint)
        else:
            phrases.append((words, constraint))
    for sentence in sentences:
        words = [w for w, _t in sentence]
        for i, (word, tag) in enumerate(sentence):
            for constraint in singles.get(word, ()):
                if constraint is None or tag in constraint:
                    total += 1
        for phrase, _none in phrases:
            n = len(phrase)
            for i in range(len(words) - n + 1):
                if tuple(words[i:i + n]) == phrase:
                    total += 1
    return total


# ---------------------------------------------------------------------------
# The overlap engine
# ---------------------------------------------------------------------------

def overlap_indices(segments: Sequence[Sequence[str]],
                    window: int) -> Tuple[Optional[float], Optional[float],
                                          Optional[float]]:
    """
    TAACO's adjacent-overlap triple over a list of per-segment lemma lists.

    For each segment ``i``, the *types* of segment ``i`` are looked up in the
    next ``window`` segments (their concatenation -- built as a fresh list;
    TAACO's version appended segment i+2 into its shared input list, so the
    order its indices were computed in changed their values). Returns:

    * proportion: total overlapping types / total types of the source
      segments (TAACO's ``adjacent_overlap_X``: type-normalized, not
      word-normalized, despite the name -- kept, and documented).
    * per-pair mean: total overlapping types / number of comparisons
      (``_div_seg``; unbounded above).
    * binary: share of comparisons with at least one overlapping type.

    A document with too few segments returns ``(None, None, None)`` -- NA,
    where TAACO wrote 0.0 and made "one paragraph" look like "zero
    cohesion".
    """
    n = len(segments)
    comparisons = n - window
    if comparisons < 1:
        return None, None, None
    overlap_total = 0
    type_total = 0
    hits = 0
    for i in range(comparisons):
        source_types = set(segments[i])
        target = set()
        for j in range(1, window + 1):
            target.update(segments[i + j])
        overlapping = len(source_types & target)
        overlap_total += overlapping
        type_total += len(source_types)
        if overlapping:
            hits += 1
    proportion = overlap_total / type_total if type_total else 0.0
    return proportion, overlap_total / comparisons, hits / comparisons


# ---------------------------------------------------------------------------
# Synonym overlap (WordNet)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=100_000)
def _synonyms(lemma: str, pos: str) -> frozenset:
    """
    Every single-word WordNet lemma name sharing a synset with ``lemma`` (the
    word itself included); an out-of-vocabulary lemma maps to itself, as in
    TAACO. WordNet is consulted directly through nltk -- TAACO shipped
    WordNet-derived files keyed by inflected forms, whose entries were noisy
    enough that "was" mapped to "washington".
    """
    out = {lemma}
    try:
        from nltk.corpus import wordnet

        for syn in wordnet.synsets(lemma, pos=pos):
            for name in syn.lemma_names():
                if "_" not in name:
                    out.add(name.lower())
    except Exception:
        pass
    return frozenset(out)


def synonym_overlap(segments: Sequence[Sequence[str]],
                    pos: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Adjacent-segment synonym overlap; returns ``(count, proportion)``.

    ``count`` is TAACO's definition, kept for comparability and documented
    as what it is (audit finding 4.15): for each *type* of segment i, one
    hit per word of segment i+1 whose synonym set contains it -- a multiply-
    counting, unbounded count, averaged over segment pairs, NOT comparable
    across texts with different sentence lengths. ``proportion`` is the
    normalized companion this implementation adds: source types matched by
    at least one next-segment word, over total source types -- the same
    scale as the lexical overlap indices. Too few segments: ``(None, None)``.
    """
    n = len(segments)
    if n < 2:
        return None, None
    count_total = 0
    matched_types = 0
    type_total = 0
    for i in range(n - 1):
        source_types = set(segments[i])
        next_synonyms = [_synonyms(w, pos) for w in segments[i + 1]]
        type_total += len(source_types)
        for t in source_types:
            hits = sum(1 for s in next_synonyms if t in s)
            count_total += hits
            if hits:
                matched_types += 1
    proportion = matched_types / type_total if type_total else 0.0
    return count_total / (n - 1), proportion


# ---------------------------------------------------------------------------
# Semantic similarity (sentence embeddings)
# ---------------------------------------------------------------------------

#: Deliberately lighter than the embeddings step's document-level default
#: (all-roberta-large): cohesion encodes every sentence of every document,
#: and MiniLM at ~5x the speed is the standard choice for sentence-pair
#: similarity. This replaces TAACO's frozen 2014 COCA LSA/word2vec spaces
#: (unshippable license, fixed vocabulary, OOV words silently dropped) and
#: its LDA index (destroyed by an implementation bug: audit finding 4.1 --
#: the shipped tool divides the loop INDEX, not the topic value, so its
#: lda_* columns are a near-constant carrying no topical information).
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def adjacent_semantic(embeddings) -> Tuple[Optional[float], Optional[float]]:
    """
    Mean adjacent cosine over row-normalized segment embeddings: window one
    (segment i vs i+1) and window two (segment i vs the normalized mean of
    i+1 and i+2 -- a fresh vector, never TAACO's in-place concatenation of
    shared state, audit finding 4.2). Too few segments: None.
    """
    import numpy as np

    E = np.asarray(embeddings, dtype=np.float64)
    n = len(E)
    if n < 2:
        return None, None
    sim1 = float(np.mean([E[i] @ E[i + 1] for i in range(n - 1)]))
    if n < 3:
        return sim1, None
    sims2 = []
    for i in range(n - 2):
        pair = E[i + 1] + E[i + 2]
        norm = np.linalg.norm(pair)
        sims2.append(E[i] @ (pair / norm) if norm else 0.0)
    return sim1, float(np.mean(sims2))


def _semantic_values(encoder, sent_texts: List[str],
                     para_texts: List[str]) -> List[Optional[float]]:
    """The four semantic columns for one document, sentences then paragraphs."""
    out: List[Optional[float]] = []
    for texts in (sent_texts, para_texts):
        if len(texts) < 2:
            out.extend((None, None))
            continue
        embeddings = encoder.encode(list(texts), convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)
        out.extend(adjacent_semantic(embeddings))
    return out


# ---------------------------------------------------------------------------
# TTR family
# ---------------------------------------------------------------------------

def _ttr(tokens: Sequence[str]) -> Optional[float]:
    return len(set(tokens)) / len(tokens) if tokens else None


def _mattr(tokens: Sequence[str], window: int) -> Optional[float]:
    """Moving-average TTR (Covington & McFall 2010). Documents shorter than
    the window fall back to plain TTR, as TAACO's does."""
    if not tokens:
        return None
    if len(tokens) < window + 1:
        return _ttr(tokens)
    counts: Counter = Counter(tokens[:window])
    total = 0.0
    windows = 0
    total += len(counts) / window
    windows += 1
    for i in range(window, len(tokens)):
        out_tok, in_tok = tokens[i - window], tokens[i]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        total += len(counts) / window
        windows += 1
    return total / windows


def _ngram_ttr(sentences: Sequence[Sequence[str]], n: int) -> Optional[float]:
    """Lemma n-gram TTR, n-grams built WITHIN sentences. TAACO built them
    over the flat document, manufacturing n-grams that span sentence
    boundaries; a documented deviation."""
    grams = [" ".join(sent[i:i + n])
             for sent in sentences for i in range(len(sent) - n + 1)]
    return _ttr(grams)


# ---------------------------------------------------------------------------
# One document -> one row of values
# ---------------------------------------------------------------------------

def _overlap_block(segments_by_class: Dict[str, List[List[str]]]) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for cls in CLASS_ORDER:
        segs = segments_by_class[cls]
        for window in (1, 2):
            prop, per_seg, binary = overlap_indices(segs, window)
            values.extend((prop, per_seg, binary))
    return values


def cohesion_row(document: List[List[List[Tuple[str, str, str]]]],
                 *, mattr_window: int,
                 connectives: Sequence[Tuple[str, List[Entry]]] = (),
                 keep_texts: bool = False,
                 ) -> Tuple[int, List[Optional[float]],
                            Optional[List[str]], Optional[List[str]]]:
    """
    Every lexical index for one parsed document, in column order (see
    `header_columns`, called with the same connectives; the semantic columns
    are computed by the caller, which holds the embedding model). Returns
    ``(token_count, values, sentence_texts, paragraph_texts)`` -- the texts
    only when ``keep_texts`` (the semantic pass needs them; pickling them
    back from workers is otherwise wasted weight).
    """
    # first, we build per-sentence and per-paragraph segments for each class.
    # note that a sentence with no members of a class still gets a slot: "no
    # adjectives here" is an absence that the adjective-overlap denominators
    # are defined over. that's how TAACO does it, and we keep it that way (and
    # say so in the guide)
    sent_segments: Dict[str, List[List[str]]] = {c: [] for c in CLASS_ORDER}
    para_segments: Dict[str, List[List[str]]] = {c: [] for c in CLASS_ORDER}
    doc_sentences_all: List[List[str]] = []
    #: (word, tag) sentences for connective matching, and the demonstrative
    #: tallies, gathered in the same pass.
    tagged_sentences: List[List[Tuple[str, str]]] = []
    dem_attended = dem_unattended = 0

    for paragraph in document:
        para_acc: Dict[str, List[str]] = {c: [] for c in CLASS_ORDER}
        for sentence in paragraph:
            classes = classify_sentence(sentence)
            doc_sentences_all.append(classes["all"])
            for c in CLASS_ORDER:
                sent_segments[c].append(classes[c])
                para_acc[c].extend(classes[c])
            counted = [(w, t) for w, _l, t in sentence if _ALNUM.search(w)]
            tagged_sentences.append(counted)
            for i, (word, _lemma, tag) in enumerate(sentence):
                if word in _DEMONSTRATIVES and tag in ("DT", "WDT"):
                    if _demonstrative_attended(sentence[i + 1:]):
                        dem_attended += 1
                    else:
                        dem_unattended += 1
        for c in CLASS_ORDER:
            para_segments[c].append(para_acc[c])

    tokens = {c: [t for seg in sent_segments[c] for t in seg]
              for c in CLASS_ORDER}
    nwords = len(tokens["all"])

    values: List[Optional[float]] = []

    # --- TTR / density family (TAACO's 15 columns, in its order) ---
    values.append(_ttr(tokens["all"]))                          # lemma_ttr
    values.append(_mattr(tokens["all"], mattr_window))          # lemma_mattr
    values.append(len(tokens["cw"]) / nwords if nwords else None)
    all_types = len(set(tokens["all"]))
    values.append(len(set(tokens["cw"])) / all_types if all_types else None)
    for cls in ("cw", "fw"):
        values.append(_ttr(tokens[cls]))                        # content/function ttr
    values.append(_mattr(tokens["fw"], mattr_window))           # function_mattr
    for cls in ("noun", "verb", "adj", "adv", "pronoun", "argument"):
        values.append(_ttr(tokens[cls]))
    values.append(_ngram_ttr(doc_sentences_all, 2))             # bigram_lemma_ttr
    values.append(_ngram_ttr(doc_sentences_all, 3))             # trigram_lemma_ttr

    # --- Lexical overlap: sentences, then paragraphs ---
    values.extend(_overlap_block(sent_segments))
    values.extend(_overlap_block(para_segments))

    # --- Synonym overlap (WordNet), sentences then paragraphs ---
    for unit_segments in (sent_segments, para_segments):
        for cls, pos in (("noun", "n"), ("verb", "v")):
            count, proportion = synonym_overlap(unit_segments[cls], pos)
            values.extend((count, proportion))

    # --- connectives: incidence per word for each category list, then the
    # --- three demonstrative columns (which come from the class machinery,
    # --- not from lists) ---
    for _name, entries in connectives:
        count = count_connectives(tagged_sentences, entries)
        values.append(count / nwords if nwords else None)
    n_dem = dem_attended + dem_unattended
    for dem in (n_dem, dem_attended, dem_unattended):
        values.append(dem / nwords if nwords else None)

    # --- Givenness ---
    n_pron = len(tokens["pronoun"])
    n_noun = len(tokens["noun"])
    values.append(n_pron / nwords if nwords else None)          # pronoun_density
    values.append(n_pron / n_noun if n_noun else None)          # pronoun_noun_ratio
    # repeated-lemma ratios. we do two things differently from TAACO here: the
    # denominator is the same class as the numerator (TAACO divided a
    # content-word count by ALL words), and we count with a Counter so that
    # it's O(n) (TAACO called list.count() per token, which is O(n^2) and gets
    # painfully slow on long documents)
    cw_counts = Counter(tokens["cw"])
    n_cw = len(tokens["cw"])
    repeated_cw = sum(1 for t in tokens["cw"] if cw_counts[t] > 1)
    values.append(repeated_cw / n_cw if n_cw else None)
    cw_pron = tokens["cw"] + tokens["pronoun"]
    cw_pron_counts = Counter(cw_pron)
    repeated_both = sum(1 for t in cw_pron if cw_pron_counts[t] > 1)
    values.append(repeated_both / len(cw_pron) if cw_pron else None)

    sent_texts = para_texts = None
    if keep_texts:
        sent_texts = []
        para_texts = []
        for paragraph in document:
            parts = []
            for sentence in paragraph:
                text = " ".join(w for w, _l, _t in sentence)
                sent_texts.append(text)
                parts.append(text)
            para_texts.append(" ".join(parts))
    return nwords, values, sent_texts, para_texts


def header_columns(connective_names: Optional[Sequence[str]] = None,
                   *, semantic: bool = False) -> List[str]:
    """The output columns, in order -- TAACO 2.1.3's names for every measure
    that survives, so results read against the TAACO literature. Connective
    columns are named by their list files; ``None`` means the shipped set.
    The four semantic columns exist only when the embedding pass runs."""
    if connective_names is None:
        connective_names = [name for name, _ in load_connective_lists()]
    cols = ["lemma_ttr", "lemma_mattr", "lexical_density_tokens",
            "lexical_density_types", "content_ttr", "function_ttr",
            "function_mattr", "noun_ttr", "verb_ttr", "adj_ttr", "adv_ttr",
            "prp_ttr", "argument_ttr", "bigram_lemma_ttr",
            "trigram_lemma_ttr"]
    for unit in ("sent", "para"):
        for cls in CLASS_ORDER:
            cols.extend((
                f"adjacent_overlap_{cls}_{unit}",
                f"adjacent_overlap_{cls}_{unit}_div_seg",
                f"adjacent_overlap_binary_{cls}_{unit}",
                f"adjacent_overlap_2_{cls}_{unit}",
                f"adjacent_overlap_2_{cls}_{unit}_div_seg",
                f"adjacent_overlap_binary_2_{cls}_{unit}",
            ))
    for unit in ("sent", "para"):
        for cls in ("noun", "verb"):
            cols.extend((f"syn_overlap_{unit}_{cls}",
                         f"syn_overlap_{unit}_{cls}_prop"))
    if semantic:
        cols.extend(("semantic_1_all_sent", "semantic_2_all_sent",
                     "semantic_1_all_para", "semantic_2_all_para"))
    cols.extend(connective_names)
    cols.extend(("all_demonstratives", "attended_demonstratives",
                 "unattended_demonstratives"))
    cols.extend(("pronoun_density", "pronoun_noun_ratio",
                 "repeated_content_lemmas",
                 "repeated_content_and_pronoun_lemmas"))
    return cols


# ---------------------------------------------------------------------------
# Worker plumbing (the pooled-row pattern every text scorer uses)
# ---------------------------------------------------------------------------

def _measure_text(parse, text: str, mattr_window: int, connectives,
                  keep_texts: bool):
    return cohesion_row(parse(text), mattr_window=mattr_window,
                        connectives=connectives, keep_texts=keep_texts)


_COHESION_WORKER: Dict[str, object] = {}


def _init_cohesion_worker(stream_args: Dict[str, object], mattr_window: int,
                          connectives, keep_texts: bool) -> None:
    _COHESION_WORKER["parse"] = make_sentence_stream(**stream_args)  # type: ignore[arg-type]
    _COHESION_WORKER["mattr_window"] = mattr_window
    _COHESION_WORKER["connectives"] = connectives
    _COHESION_WORKER["keep_texts"] = keep_texts


def _cohesion_in_worker(pair):
    _tid, text = pair
    return _measure_text(_COHESION_WORKER["parse"], text,
                         _COHESION_WORKER["mattr_window"],
                         _COHESION_WORKER["connectives"],
                         _COHESION_WORKER["keep_texts"])


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  bookkeeping=("nwords",),
                  assets={"connective_lists": "connectives"})
def analyze_cohesion(
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

    # ====== COHESION OPTIONS ======
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    device: str = "auto",
    mattr_window: int = 50,
    connective_lists: Optional[Sequence[PathLike]] = None,
    semantic_model: str = DEFAULT_SEMANTIC_MODEL,
    rounding: int = 4,
) -> Path:
    """
    Compute TAACO-style cohesion indices; one row per document.

    See ``COHESION_MEASURES.md`` (shipped next to this module) for what every
    column measures, how to interpret it, and where this implementation
    deliberately differs from TAACO 2.1.3.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as the other text analyzers: a spreadsheet of
        texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/cohesion/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and
        return the path.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns that identify each row when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple text columns are provided.
    group_by : Sequence[str] or None, optional
        Optional grouping keys used during CSV gathering.
    delimiter : str, default=","
        Column separator of the *input* CSV.
    pattern : str, default=every document type
        Which files to read when gathering from a folder of documents.
    engine : {"nltk", "stanza"}, default="nltk"
        Who tags and lemmatizes: NLTK (fast, English), or Stanza (neural,
        multilingual, GPU-optional). Sentence splitting is punkt under NLTK
        and the model's own under Stanza.
    tokenizer : {"potts", "stanza"}, default="potts"
        Who splits text into tokens, exactly as in the n-gram steps.
    stanza_lang : str, default="en"
        Language for the Stanza engine; the model downloads once on first
        use. Only with engine="stanza".
    device : {"auto", "cuda", "cpu"}, default="auto"
        Where Stanza runs: "auto", "cuda", or "cpu". Only with
        engine="stanza".
    mattr_window : int, default=50
        Window (tokens) for the moving-average TTRs. TAACO hardcodes 50;
        this default matches, and shorter documents fall back to plain TTR.
    connective_lists : sequence of paths, optional
        Connectives category files (.txt, one word or phrase per line, ``#``
        comments, optional TAB + Penn tag constraint; folders are expanded).
        Each file becomes one incidence column named by its stem. ``None``
        (the default) means the shipped categories; in a pipeline, the
        library's connectives shelf is wired in, so lists you import there
        become columns automatically.
    semantic_model : str, default the MiniLM sentence-transformer
        The embedding model for the four semantic-similarity columns
        (adjacent sentence/paragraph cosine). ``"none"`` skips them and the
        step runs without any model. Replaces TAACO's frozen COCA LSA and
        its LDA index (the latter provably broken in the reference tool).
    workers : int, default=0
        Parallel processes for reading and measuring documents. ``0`` means
        automatic: three-quarters of the logical cores. Output files are
        identical whatever the worker count; Stanza-backed runs stay
        single-process (its lever is batching, not processes).
    rounding : int, default=4
        Decimal places for every index.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id``, ``nwords``, then one column per
        index (~130). Indices that are undefined for a document -- paragraph
        cohesion of a one-paragraph text -- are empty cells, never 0.
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
        out_features_csv = Path.cwd() / "features" / "cohesion" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        print("Cohesion output file already exists; returning existing file.")
        return out_features_csv

    if engine == "stanza":
        # the model might download (hundreds of MB) or load right here, so we
        # say so. otherwise the step just sits on "reading the input" while
        # it's clearly off doing something else
        announce(on_progress, "loading the stanza pipeline (first use "
                              "downloads its model)")
    parse = make_sentence_stream(engine=engine, tokenizer=tokenizer,
                                 stanza_lang=stanza_lang, device=device)
    stream_args = dict(engine=engine, tokenizer=tokenizer,
                       stanza_lang=stanza_lang, device=device)
    connectives = load_connective_lists(connective_lists)
    semantic_on = str(semantic_model).strip().lower() not in ("", "none")
    columns = header_columns([name for name, _ in connectives],
                             semantic=semantic_on)
    encoder = None
    if semantic_on:
        # the model lives HERE, in the parent, exactly once -- the workers stay
        # model-free. our house rule: a model's speed lever is batching, and
        # one copy per worker process is a memory bill nobody wants to pay
        from sentence_transformers import SentenceTransformer

        from ..helpers.gpu import resolve_device

        announce(on_progress, "loading the sentence-embedding model")
        encoder = SentenceTransformer(semantic_model,
                                      device=resolve_device(device)[0])
    insert_at = columns.index("semantic_1_all_sent") if semantic_on else -1

    from ..helpers.row_map import map_text_rows

    def _fmt(value: Optional[float]) -> object:
        # NA is an EMPTY CELL. TAACO writes 0.0 for indices that don't exist
        # for a document, but downstream you can't tell that apart from a
        # measured zero cohesion, so we don't do that
        if value is None:
            return ""
        return round(float(value), rounding)

    with atomic_write(out_features_csv, newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(["text_id", "nwords", *columns])
        for row, (nwords, values, sent_texts, para_texts) in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pooled_text_workers(
                    workers, n_rows, engine=engine, tokenizer=tokenizer,
                    lemmatize=True, pos_tagged=True),
                message="measuring cohesion", on_progress=on_progress,
                inline_fn=lambda pair: _measure_text(
                    parse, pair[1], mattr_window, connectives, semantic_on),
                pool_fn=_cohesion_in_worker,
                initializer=_init_cohesion_worker,
                initargs=(stream_args, mattr_window, connectives, semantic_on)):
            if encoder is not None:
                semantic = _semantic_values(encoder, sent_texts or [],
                                            para_texts or [])
                # the worker computed the lexical row without the semantic
                # slots (workers are model-free, remember), so we splice them
                # in at their column position. `columns` and the worker's
                # values share the same prefix, so the index in the full header
                # is exactly where they go
                values = values[:insert_at] + semantic + values[insert_at:]
            writer.writerow([row.get("text_id", ""), nwords,
                             *(_fmt(v) for v in values)])
    return out_features_csv


# --- CLI ---------------------------------------------------------------------


def measures_guide() -> str:
    """The full text of COHESION_MEASURES.md, shipped beside this module."""
    return (Path(__file__).resolve().parent
            / "COHESION_MEASURES.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_cohesion,
    description="TAACO-style cohesion indices, one row per document. See COHESION_MEASURES.md for every column's meaning.",
    aliases={},
    legacy={},
)


def main(argv=None) -> int:
    if argv is not None and "--explain" in argv or argv is None and "--explain" in __import__("sys").argv:
        print(measures_guide())          # just the full measures guide, nothing else
        return 0
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
