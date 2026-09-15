"""
Shared text preparation for the n-gram tools.

The frequency list, the document-term matrix, and the topic models planned
behind them must all see *identical* token streams -- a vocabulary built from
lemmatized text is unfindable in unlemmatized text, and the failure is silent:
near-zero matches, no error. So the tokenize -> lemmatize -> n-gram pipeline
lives here, once, and every consumer calls the same functions.

Two rules are deliberate and load-bearing:

* **Lemmatize before stoplisting.** A stoplist entry like "be" is expected to
  catch "is/was/were"; filtering the raw tokens first would let them through.
  The stoplist is therefore always matched against the *prepared* stream.
* **Stoplists filter output rows, never the token stream.** Removing stopwords
  before building n-grams manufactures adjacencies that never occurred
  ("the cat sat" would yield the bigram "cat sat"), which corrupts counts and
  every collocation statistic computed from them. Counting happens on the
  intact stream; :func:`stoplisted` is applied to finished n-grams.

The tokenizer is :mod:`taters.text.happierfuntokenizing` (Potts/Schwartz),
conceptually the same one the content coder uses. Making the tokenizer
user-selectable is planned; when that lands, this module is the seam.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Set, Tuple, Union

__all__ = ["make_token_stream", "make_tagged_stream", "load_stoplist",
           "stoplisted", "iter_ngrams", "split_for_collocation", "TAG_SEP",
           "words_of", "tags_of"]

#: The valid engine/tokenizer combinations. `tokenizer="stanza"` under the
#: NLTK engine is refused: NLTK tagging over Stanza's tokens is a combination
#: nobody means, and keeping the matrix at three honest cells is cheaper than
#: supporting a fourth nobody asked for.
ENGINES = ("nltk", "stanza")
TOKENIZERS = ("potts", "stanza")

PathLike = Union[str, Path]

#: Joins a word to its POS tag inside a token when ``pos_tagged`` streams are
#: in use ("felt\x1fVBD"). The unit separator, because it is the one character
#: no tokenizer output can contain -- words can legitimately hold "/", "_",
#: and "-" (URLs, hashtags, hyphenations), so none of those were safe.
TAG_SEP = "\x1f"


def words_of(gram: str) -> str:
    """The word half of a (possibly tagged) space-joined n-gram."""
    return " ".join(tok.split(TAG_SEP, 1)[0] for tok in gram.split(" "))


def tags_of(gram: str) -> str:
    """The tag half of a tagged n-gram ("JJ NN"), or "" for untagged."""
    parts = [tok.split(TAG_SEP, 1) for tok in gram.split(" ")]
    if any(len(p) < 2 for p in parts):
        return ""
    return " ".join(p[1] for p in parts)


def _check_combo(engine: str, tokenizer: str) -> None:
    """Refuse unknown names and the one senseless pairing, loudly and early."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if tokenizer not in TOKENIZERS:
        raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got {tokenizer!r}")
    if engine == "nltk" and tokenizer == "stanza":
        raise ValueError(
            "tokenizer='stanza' needs engine='stanza': NLTK tagging over "
            "Stanza's tokens is not a supported combination."
        )


def _split_spaced_tokens(tokens: List[str]) -> List[str]:
    """
    No token may contain a space: it is the n-gram separator.

    The Potts tokenizer's phone-number pattern matches page ranges --
    ``"pp. 1467- 1470"`` in any academic PDF -- as ONE token with a space in
    it. Every layer above joins n-grams with spaces, so that "unigram" reads
    back as a bigram whose halves were never counted: a KeyError at scoring
    time (how this was found), or silently wrong statistics where the halves
    happen to exist. Splitting on the whitespace keeps the pieces as the
    tokens they visibly are.
    """
    if not any(" " in token for token in tokens):
        return tokens
    out: List[str] = []
    for token in tokens:
        if " " in token:
            out.extend(token.split())
        else:
            out.append(token)
    return out


def _require(ensure, what: str, package: str) -> None:
    """Refuse, in words, when an NLTK resource is missing and cannot be fetched.
    The same sentence used to be written out at every call site."""
    if not ensure():
        raise RuntimeError(
            f"{what} is not available and could not be downloaded. Run once "
            f"with network access, or install it with: "
            f"python -m nltk.downloader {package}")


def _potts_tokenize() -> Callable[[str], List[str]]:
    from .happierfuntokenizing import Tokenizer

    tokenize = Tokenizer(preserve_case=False).tokenize

    def stream(text: str) -> List[str]:
        return _split_spaced_tokens(tokenize(text))

    return stream


def is_word(token: str) -> bool:
    """
    Whether a token is a word rather than punctuation.

    A token counts as a word when it has at least one letter or digit --
    ``don't``, ``#tag``, ``2nd`` and ``felt\x1fVBD`` do; ``.``, ``--``
    and ``:)`` do not. The tag is ignored, so a tagged period is still
    punctuation.
    """
    return any(ch.isalnum() for ch in words_of(token))


def make_token_stream(lemmatize: bool = False,
                      pos_tagged: bool = False,
                      *,
                      engine: str = "nltk",
                      tokenizer: str = "potts",
                      stanza_lang: str = "en",
                      device: str = "auto",
                      keep_punctuation: bool = False
                      ) -> Callable[[str], List[str]]:
    """
    Build the text -> tokens function every n-gram consumer shares.

    Punctuation is dropped unless ``keep_punctuation`` is True: a frequency
    list whose top term is ``.`` and a topic model whose vocabulary spends
    a column on ``,`` are counting sentence boundaries, not words, and the
    period was the one term a real topic model found constant and set
    aside with a warning. Emoticons go with the punctuation (they have no
    letter in them); anyone studying them turns this on. The filter sits
    here so the frequency list, the matrix and a saved topic model all read
    text the same way -- the one thing a vocabulary scan cannot survive
    disagreement on.
    """
    stream = _unfiltered_token_stream(
        lemmatize, pos_tagged, engine=engine, tokenizer=tokenizer,
        stanza_lang=stanza_lang, device=device)
    if keep_punctuation:
        return stream

    def words(text: str) -> List[str]:
        return [t for t in stream(text) if is_word(t)]
    return words


def _unfiltered_token_stream(lemmatize: bool, pos_tagged: bool, *,
                             engine: str, tokenizer: str, stanza_lang: str,
                             device: str) -> Callable[[str], List[str]]:
    """
    The tokens as the tokenizer produced them; see :func:`make_token_stream`.

    Tokenization is happierfuntokenizing with case folded. With
    ``lemmatize=True``, tokens are POS-tagged and lemmatized under their tag
    -- "cats" -> "cat", "was" -> "be", "running" -> "run" -- and tokens whose
    tag WordNet has no category for (pronouns, determiners, prepositions) are
    left alone. The POS step is not optional polish: a bare noun-first
    lemmatizer mangles exactly the most frequent words in the language --
    "was" -> "wa", "us" -> "u", "does" -> "doe" -- and a frequency list is
    the one place such garbage cannot hide. Found in a real end-to-end run.

    With ``pos_tagged=True``, each emitted token carries its tag joined by
    :data:`TAG_SEP` ("felt\\x1fVBD"), so the verb "felt" and the noun "felt"
    are different tokens everywhere downstream -- different frequency rows,
    different NPMI partners, different DTM columns. One tagging pass serves
    both options when they are combined.

    The engine decides who tags and lemmatizes: ``"nltk"`` (the default, and
    byte-identical to what this function always produced) or ``"stanza"``
    (neural, context-aware lemmas, GPU-optional). The tokenizer is a separate
    choice on purpose -- ``tokenizer="potts"`` under either engine yields the
    *same token strings*, so counts stay comparable and emoticons, hashtags
    and URLs survive; ``tokenizer="stanza"`` hands Stanza the whole job.

    Raises
    ------
    RuntimeError
        If lemmatization or tagging was asked for and the engine's data cannot
        be made available. The caller asked for prepared counts; silently
        returning raw ones would corrupt results without a word.
    """
    _check_combo(engine, tokenizer)

    if tokenizer == "potts" and not lemmatize and not pos_tagged:
        # nothing to tag or lemmatize, and the tokens are potts's either way, so
        # both engines would give identical output. no point loading a model
        return _potts_tokenize()

    if engine == "stanza":
        return _stanza_token_stream(lemmatize, pos_tagged,
                                    tokenizer=tokenizer,
                                    stanza_lang=stanza_lang, device=device)

    tokenize = _potts_tokenize()

    from ..helpers.nltk_data import ensure_pos_tagger, ensure_wordnet

    _require(ensure_pos_tagger, "the POS tagger data", "averaged_perceptron_tagger_eng")
    if lemmatize:
        _require(ensure_wordnet, "lemmatize=True, but NLTK's WordNet data", "wordnet")
    import nltk

    if lemmatize:
        from nltk.stem import WordNetLemmatizer

        wnl = WordNetLemmatizer()
        # penn treebank tag families -> wordnet categories
        wordnet_pos = {"J": "a", "N": "n", "V": "v", "R": "r"}

    def stream(text: str) -> List[str]:
        out = []
        for token, tag in nltk.pos_tag(tokenize(text)):
            if lemmatize:
                pos = wordnet_pos.get(tag[:1])
                if pos:
                    token = wnl.lemmatize(token, pos)
            out.append(f"{token}{TAG_SEP}{tag}" if pos_tagged else token)
        return out

    return stream


#: Built pipelines, keyed by (lang, pretokenized, use_gpu). Construction is
#: the expensive part (~2s on CPU, a model load on GPU); the streams a factory
#: hands out are called once per document.
_STANZA_PIPELINES: dict = {}


def _stanza_pipeline(lang: str, pretokenized: bool, device: str):
    """Build (or reuse) a Stanza pipeline, failing with actionable messages."""
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except ImportError:
        raise RuntimeError(
            "engine='stanza', but the stanza package is not installed. "
            "Install it with: pip install taters[stanza]"
        ) from None

    from ..helpers.gpu import resolve_device
    from ..helpers.stanza_data import ensure_stanza

    use_gpu = str(resolve_device(device)[0]).startswith("cuda")
    key = (lang, pretokenized, use_gpu)
    pipe = _STANZA_PIPELINES.get(key)
    if pipe is None:
        if not ensure_stanza(lang):
            raise RuntimeError(
                f"Stanza's '{lang}' model is not available and could not be "
                "downloaded. Run once with network access, or fetch it with: "
                f"python -c \"import stanza; stanza.download('{lang}')\""
            )
        pipe = stanza.Pipeline(
            lang=lang, processors="tokenize,pos,lemma",
            tokenize_pretokenized=pretokenized, use_gpu=use_gpu,
            verbose=False, download_method=DownloadMethod.REUSE_RESOURCES,
        )
        _STANZA_PIPELINES[key] = pipe
    return pipe


def _stanza_words(pipe, tokenize, text: str):
    """One document through Stanza; yields its Word objects."""
    if tokenize is not None:
        tokens = tokenize(text)
        if not tokens:
            # stanza raises IndexError on an empty pretokenized sentence (found
            # this out the hard way) -- and an empty document has nothing to
            # say anyway
            return
        doc = pipe([tokens])
    else:
        if not text.strip():
            return
        doc = pipe(text)
    for sentence in doc.sentences:
        yield from sentence.words


def _stanza_tag(word, tagset: str) -> str:
    """The requested tag family; xpos is Penn for English, upos is universal.

    Some languages ship no xpos; falling back to upos beats an empty tag."""
    if tagset == "universal":
        return word.upos or ""
    return word.xpos or word.upos or ""


def _stanza_token_stream(lemmatize: bool, pos_tagged: bool, *,
                         tokenizer: str, stanza_lang: str,
                         device: str) -> Callable[[str], List[str]]:
    """The Stanza half of :func:`make_token_stream`; same output contract."""
    tokenize = _potts_tokenize() if tokenizer == "potts" else None
    pipe = _stanza_pipeline(stanza_lang, pretokenized=tokenize is not None,
                            device=device)

    def stream(text: str) -> List[str]:
        out = []
        for word in _stanza_words(pipe, tokenize, text):
            # we lowercase here so that this matches every other stream: the
            # potts path arrives lowercased already, but stanza's own tokenizer
            # preserves case
            token = (word.lemma or word.text).lower() if lemmatize \
                else word.text.lower()
            # stanza's own tokenizer can spit out multi-word tokens in some
            # languages, and a space inside a token poisons the n-gram machinery
            # (see _split_spaced_tokens). so, we split it and each part gets the
            # tag
            parts = token.split() if " " in token else (token,)
            if pos_tagged:
                tag = _stanza_tag(word, 'penn')
                for part in parts:
                    out.append(f"{part}{TAG_SEP}{tag}")
            else:
                out.extend(parts)
        return out

    return stream


def make_tagged_stream(*, engine: str = "nltk", tokenizer: str = "potts",
                       tagset: str = "penn", stanza_lang: str = "en",
                       device: str = "auto") -> Callable[[str], List[tuple]]:
    """
    Text -> ``[(token, tag), ...]`` -- the parts-of-speech analyzer's shape.

    One factory for both engines so there is exactly one tagging path in the
    package. ``tagset`` is honored by both: NLTK maps "universal" through its
    own tag mapper; Stanza serves xpos (Penn, for English) or upos.

    Raises the same way :func:`make_token_stream` does when an engine's data
    cannot be made available.
    """
    _check_combo(engine, tokenizer)
    if tagset not in ("penn", "universal"):
        raise ValueError(f"tagset must be 'penn' or 'universal', got {tagset!r}")

    if engine == "stanza":
        tokenize = _potts_tokenize() if tokenizer == "potts" else None
        pipe = _stanza_pipeline(stanza_lang, pretokenized=tokenize is not None,
                                device=device)

        def tagged(text: str) -> List[tuple]:
            return [(word.text.lower(), _stanza_tag(word, tagset))
                    for word in _stanza_words(pipe, tokenize, text)]

        return tagged

    from ..helpers.nltk_data import ensure_pos_tagger, ensure_universal_tagset

    _require(ensure_pos_tagger, "the POS tagger data", "averaged_perceptron_tagger_eng")
    if tagset == "universal":
        _require(ensure_universal_tagset,
                 "tagset='universal', but the universal_tagset mapping",
                 "universal_tagset")
    import nltk

    tokenize = _potts_tokenize()
    kwargs = {"tagset": "universal"} if tagset == "universal" else {}

    def tagged(text: str) -> List[tuple]:
        return nltk.pos_tag(tokenize(text), **kwargs)

    return tagged


def load_stoplist(paths: Sequence[PathLike]) -> Set[str]:
    """
    Read stop entries from files (or folders of ``.txt``), one per line.

    The built-in lists are BOM-prefixed CRLF files; both are handled. Entries
    are lowercased to match the case-folded token stream. A named file that
    does not exist raises -- a run silently missing its stoplist would look
    exactly like a run that worked.
    """
    stopset: Set[str] = set()
    for p in paths or ():
        p = Path(p)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*.txt") if f.is_file())
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"Stoplist not found: {p}")
        for f in files:
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                entry = line.strip().lower()
                if entry:
                    stopset.add(entry)
    return stopset


def stoplisted(ngram: str, stopset: Set[str]) -> bool:
    """
    Whether this (space-joined) n-gram should be dropped from an output.

    An n-gram is dropped when *any* of its tokens is a stop entry -- applied
    to finished n-grams, after counting, so statistics are computed on the
    intact stream (see the module docstring for why). Stop entries are words,
    so a tagged token is matched by its word half: "the\\x1fDT" is still "the".
    """
    if not stopset:
        return False
    return any(tok.split(TAG_SEP, 1)[0] in stopset for tok in ngram.split(" "))


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    """Every space-joined n-gram of exactly order ``n``, in document order."""
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


#: Paragraph boundary: one or more blank lines. Deliberately NOT "any
#: newline" (TAACO's rule): hard-wrapped text -- every line of a PDF
#: extraction -- would make each visual line a "paragraph" and reduce every
#: paragraph-level cohesion index to noise.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

#: Fallback sentence splitter for when punkt cannot be downloaded: naive but
#: honest, and only ever used after ensure_punkt() said no.
_SENTENCE_FALLBACK = re.compile(r"(?<=[.!?])\s+")


def make_sentence_stream(*, engine: str = "nltk", tokenizer: str = "potts",
                         stanza_lang: str = "en", device: str = "auto"):
    """
    A document parser for sentence-aware measures (cohesion): text ->
    paragraphs -> sentences -> ``(word, lemma, penn_tag)`` triples,
    lowercased, empty sentences/paragraphs dropped.

    The same engine/tokenizer matrix as :func:`make_token_stream`, so a
    pipeline's shared engine settings mean one thing everywhere. Lemmas are
    always produced (cohesion measures are lemma-based by definition):
    WordNet POS-guided on the nltk path, the model's lemma under stanza.
    Sentences never span paragraphs; paragraphs are blank-line delimited
    (see ``_PARAGRAPH_BREAK`` for why not TAACO's any-newline rule).
    """
    _check_combo(engine, tokenizer)

    if engine == "stanza":
        return _stanza_sentence_stream(tokenizer=tokenizer,
                                       stanza_lang=stanza_lang, device=device)

    from ..helpers.nltk_data import ensure_pos_tagger, ensure_punkt, ensure_wordnet

    _require(ensure_pos_tagger, "the POS tagger data", "averaged_perceptron_tagger_eng")
    _require(ensure_wordnet, "NLTK's WordNet data", "wordnet")
    have_punkt = ensure_punkt(verbose=False)
    import nltk
    from nltk.stem import WordNetLemmatizer

    tokenize = _potts_tokenize()
    wnl = WordNetLemmatizer()
    wordnet_pos = {"J": "a", "N": "n", "V": "v", "R": "r"}

    def _sentences(paragraph: str) -> List[str]:
        if have_punkt:
            return nltk.sent_tokenize(paragraph)
        return _SENTENCE_FALLBACK.split(paragraph)

    def parse(text: str) -> List[List[List[Tuple[str, str, str]]]]:
        paragraphs = []
        for para in _PARAGRAPH_BREAK.split(text or ""):
            if not para.strip():
                continue
            sentences = []
            for sent in _sentences(para):
                tokens = tokenize(sent)
                if not tokens:
                    continue
                triples = []
                for word, tag in nltk.pos_tag(tokens):
                    pos = wordnet_pos.get(tag[:1])
                    lemma = wnl.lemmatize(word, pos) if pos else word
                    triples.append((word, lemma, tag))
                sentences.append(triples)
            if sentences:
                paragraphs.append(sentences)
        return paragraphs

    return parse


def _stanza_sentence_stream(*, tokenizer: str, stanza_lang: str, device: str):
    """The stanza half of :func:`make_sentence_stream`; same output contract.

    With tokenizer="potts", sentences are split by punkt (or the fallback)
    and handed to stanza pre-tokenized one sentence per list, so token counts
    stay comparable with the potts-based streams; with tokenizer="stanza"
    the model does the whole job on the raw paragraph."""
    from ..helpers.nltk_data import ensure_punkt

    pretokenize = _potts_tokenize() if tokenizer == "potts" else None
    have_punkt = ensure_punkt(verbose=False) if pretokenize is not None else False
    if pretokenize is not None and have_punkt:
        import nltk
    pipe = _stanza_pipeline(stanza_lang, pretokenized=pretokenize is not None,
                            device=device)

    def _split(paragraph: str) -> List[str]:
        if have_punkt:
            return nltk.sent_tokenize(paragraph)
        return _SENTENCE_FALLBACK.split(paragraph)

    def parse(text: str) -> List[List[List[Tuple[str, str, str]]]]:
        paragraphs = []
        for para in _PARAGRAPH_BREAK.split(text or ""):
            if not para.strip():
                continue
            if pretokenize is not None:
                token_lists = [t for t in (pretokenize(s) for s in _split(para)) if t]
                if not token_lists:
                    continue
                doc = pipe(token_lists)
            else:
                doc = pipe(para)
            sentences = []
            for sentence in doc.sentences:
                triples = []
                for word in sentence.words:
                    token = word.text.lower()
                    lemma = (word.lemma or word.text).lower()
                    tag = _stanza_tag(word, "penn")
                    for part, lpart in zip(token.split() or (token,),
                                           lemma.split() or (lemma,)):
                        triples.append((part, lpart, tag))
                if triples:
                    sentences.append(triples)
            if sentences:
                paragraphs.append(sentences)
        return paragraphs

    return parse


def pooled_text_workers(workers: int, n_rows: int, *, engine: str,
                        tokenizer: str, lemmatize: bool,
                        pos_tagged: bool) -> int:
    """
    Processes for a pooled per-row pass over prepared text.

    The Stanza *model* never fans out: a process per worker means a model
    copy per worker in memory, and its lever is batching, not processes --
    the same rule the runner applies to every model-backed step. The model
    path is engaged only when something actually needs it; plain potts
    tokenization under engine="stanza" short-circuits to potts and pools
    happily. Small row counts stay serial on the automatic setting
    (`pool_workers`): the pool costs more to start than it saves.
    """
    from ..helpers.parallel_map import pool_workers

    if engine == "stanza" and (lemmatize or pos_tagged or tokenizer == "stanza"):
        return 1
    return pool_workers(workers, n_rows)


def split_for_collocation(ngram: str) -> Tuple[str, str]:
    """
    The chain split collocation statistics are computed over:
    ``(w1 ... wn-1)`` and ``(wn)`` -- NPMI((w1..wn-1), wn), per Bouma (2009).
    """
    words = ngram.split(" ")
    return " ".join(words[:-1]), words[-1]
