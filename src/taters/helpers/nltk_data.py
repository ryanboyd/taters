"""Make sure NLTK's data packages are present before they are needed.

NLTK ships code but not data. The first call to `sent_tokenize` (or the
WordNet lemmatizer) on a fresh machine raises a `LookupError` wrapped in a
wall of asterisks — technically accurate, thoroughly unhelpful, and, when it
happens six steps into a pipeline, expensive. Anything in Taters that will end
up in `sent_tokenize` should call :func:`ensure_punkt` first; anything that
lemmatizes calls :func:`ensure_wordnet`.
"""

from __future__ import annotations


def _ensure(works, packages, *, verbose: bool, what: str, missing: str) -> bool:
    """
    Ask the real question -- does it work? -- download once if not, never raise.

    Every ``ensure_*`` below is this loop with a different probe and package
    list. They were four copies of it: the probe is the part that matters
    (checking that a package *directory* exists reports success while the call
    still raises on newer NLTK releases), and the download-then-recheck loop is
    the same for all of them.
    """
    try:
        import nltk
    except Exception:
        return False

    if works():
        return True
    for package in packages:
        try:
            if verbose:
                print(f"Downloading NLTK '{package}' data ...")
            nltk.download(package, quiet=True)
        except Exception:
            continue
        if works():
            return True
    if verbose:
        print(f"{what} NOT available: {missing}")
    return False


def ensure_punkt(verbose: bool = True) -> bool:
    """
    Ensure NLTK's ``punkt`` sentence tokenizer is available, downloading it once
    if necessary. Handles both the classic ``punkt`` package and ``punkt_tab``,
    which newer NLTK releases look for instead.

    Returns True if ``nltk.sent_tokenize`` can be used; False if callers should
    fall back to their own splitting. Never raises -- a download failure
    (offline machine, locked-down environment) returns False rather than
    taking the pipeline down with it.
    """
    try:
        from nltk.tokenize import sent_tokenize
    except Exception:
        if verbose:
            print("NLTK is not installed; using a regex sentence splitter.")
        return False

    def works() -> bool:
        try:
            sent_tokenize("A short sentence. And another one.")
            return True
        except Exception:
            return False

    ok = _ensure(works, ("punkt_tab", "punkt"), verbose=verbose,
                 what="Sentence tokenizer", missing="using regex fallback.")
    if ok and verbose:
        print("Sentence tokenizer available: using NLTK sent_tokenize.")
    return ok


def ensure_wordnet(verbose: bool = True) -> bool:
    """
    Ensure NLTK's WordNet data is available, downloading it once if necessary.

    Returns True if ``WordNetLemmatizer`` works. Never raises; the *caller*
    decides whether missing lemmatization is fatal. For the n-gram tools it
    is: the user asked for lemmatized counts, and silently returning
    unlemmatized ones would corrupt the results without a word.
    """
    try:
        from nltk.stem import WordNetLemmatizer
    except Exception:
        if verbose:
            print("NLTK is not installed; lemmatization is unavailable.")
        return False

    def works() -> bool:
        try:
            return WordNetLemmatizer().lemmatize("cats") == "cat"
        except Exception:
            return False

    return _ensure(works, ("wordnet", "omw-1.4"), verbose=verbose,
                   what="WordNet data", missing="lemmatization is unavailable.")


def ensure_pos_tagger(verbose: bool = True) -> bool:
    """
    Ensure NLTK's part-of-speech tagger is available, downloading it once.

    The lemmatizer needs it: WordNet lemmatization without a POS is noun-first
    guessing, and the noun pass mangles high-frequency function words --
    "was" -> "wa", "us" -> "u", "does" -> "doe" -- garbage that then tops a
    frequency list.
    """
    try:
        import nltk
    except Exception:
        if verbose:
            print("NLTK is not installed; POS tagging is unavailable.")
        return False

    def works() -> bool:
        try:
            return bool(nltk.pos_tag(["hello"]))
        except Exception:
            return False

    return _ensure(works, ("averaged_perceptron_tagger_eng",
                           "averaged_perceptron_tagger"), verbose=verbose,
                   what="POS tagger", missing="lemmatization is unavailable.")


def ensure_universal_tagset(verbose: bool = True) -> bool:
    """Ensure the Penn -> Universal tag mapping is available, downloading once.
    ``nltk.pos_tag(..., tagset="universal")`` needs it."""
    try:
        import nltk
    except Exception:
        return False

    def works() -> bool:
        try:
            return nltk.pos_tag(["hello"], tagset="universal")[0][1] != ""
        except Exception:
            return False

    return _ensure(works, ("universal_tagset",), verbose=verbose,
                   what="Universal tagset", missing="tagset='universal' cannot be used.")
