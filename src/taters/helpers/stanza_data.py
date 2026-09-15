"""Make sure Stanza's language models are present before they are needed.

The sibling of :mod:`taters.helpers.nltk_data`, with the same posture: ask the
real question -- can a pipeline be built? -- download once if not, and never
raise. The *caller* decides whether a missing model is fatal; for the n-gram
tools it is, because the user asked for Stanza's tags and silently returning
something else would corrupt results without a word.

Models live under ``~/stanza_resources`` by default; Stanza honors the
``STANZA_RESOURCES_DIR`` environment variable, so nothing here manages paths.
"""

from __future__ import annotations


def ensure_stanza(lang: str = "en", verbose: bool = True) -> bool:
    """
    Ensure Stanza and its model for ``lang`` are available.

    Returns
    -------
    bool
        True when a ``tokenize,pos,lemma`` pipeline for ``lang`` can be built.
        False when the stanza package is missing, or the model is absent and
        could not be downloaded (offline machine, locked-down environment).
    """
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except Exception:
        if verbose:
            print("stanza is not installed; install it with: "
                  "pip install taters[stanza]")
        return False

    def _works() -> bool:
        try:
            stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma",
                            use_gpu=False, verbose=False,
                            download_method=DownloadMethod.NONE)
            return True
        except Exception:
            return False

    if _works():
        return True

    try:
        if verbose:
            print(f"Downloading Stanza's '{lang}' model (one time; can be "
                  "a few hundred MB -- it downloads quietly, so expect a "
                  "few minutes without a progress bar) ...")
        # the download itself has to be MUTE. `verbose=False` silences stanza's
        # own logging, but the underlying huggingface-hub fetch draws raw tqdm
        # bars straight onto stderr -- and during a pipeline run, those sliced
        # right through our live progress display. the env var asks the hub
        # politely; the stderr redirect covers hub versions that don't honor
        # it. exceptions still raise through both.
        import contextlib
        import io
        import os

        before = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                stanza.download(lang, verbose=False)
        finally:
            if before is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = before
    except Exception:
        pass
    return _works()
