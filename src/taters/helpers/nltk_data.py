"""Make sure NLTK's sentence tokenizer data is present before it is needed.

NLTK ships code but not data. The first call to `sent_tokenize` on a fresh
machine raises a `LookupError` wrapped in a wall of asterisks — technically
accurate, thoroughly unhelpful, and, when it happens six steps into a pipeline,
expensive. Anything in Taters that will end up in `sent_tokenize` (directly, or
through a library like archetyper) should call :func:`ensure_punkt` first.
"""

from __future__ import annotations


def ensure_punkt(verbose: bool = True) -> bool:
    """
    Ensure NLTK's ``punkt`` sentence tokenizer is available, downloading it once
    if necessary.

    Handles both the classic ``punkt`` package and ``punkt_tab``, which newer
    NLTK releases look for instead.

    Parameters
    ----------
    verbose : bool, default=True
        Print a short status line about what happened.

    Returns
    -------
    bool
        True if `nltk.sent_tokenize` can be used; False if callers should fall
        back to their own splitting. Never raises — a download failure (offline
        machine, locked-down environment) returns False rather than taking the
        pipeline down with it.
    """
    try:
        import nltk  # noqa: F401
    except Exception:
        if verbose:
            print("NLTK is not installed; using a regex sentence splitter.")
        return False

    import nltk
    from nltk.tokenize import sent_tokenize

    def _works() -> bool:
        """
        Ask the real question — can we tokenize? — instead of guessing which
        data package this NLTK version wants. `punkt` was enough for years;
        NLTK 3.8.2+ wants `punkt_tab`, and checking for the presence of `punkt`
        reports success while `sent_tokenize` still raises.
        """
        try:
            sent_tokenize("A short sentence. And another one.")
            return True
        except Exception:
            return False

    if _works():
        if verbose:
            print("Sentence tokenizer available: using NLTK sent_tokenize.")
        return True

    for package in ("punkt_tab", "punkt"):
        try:
            if verbose:
                print(f"Downloading NLTK '{package}' tokenizer data ...")
            nltk.download(package, quiet=True)
        except Exception:
            continue
        if _works():
            if verbose:
                print("Sentence tokenizer available: using NLTK sent_tokenize.")
            return True

    if verbose:
        print("Sentence tokenizer NOT available: using regex fallback.")
    return False
