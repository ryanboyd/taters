"""
One sentence splitter for every step that embeds sentence by sentence.

The sentence-transformers step and the transformer-embeddings step have to
agree on what a sentence is, or the same text gets a different number of
vectors averaged into it depending on which model was chosen -- a
difference that would look like a property of the models. So the splitter
lives here and both import it.

NLTK's ``punkt`` is used when it can be loaded (``ensure_punkt`` downloads
it once); otherwise a regex that splits on end punctuation followed by
whitespace, which under- and over-segments a little but never fails.
"""
from __future__ import annotations

import re
from typing import List

__all__ = ["split_sentences"]

_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """
    The non-empty, stripped sentences of ``text``; ``[]`` for nothing.

    Prefers ``nltk.tokenize.sent_tokenize``; falls back to splitting on end
    punctuation plus whitespace when NLTK or its data is unavailable.
    """
    txt = (text or "").strip()
    if not txt:
        return []
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore

        return [s for s in sent_tokenize(txt) if s.strip()]
    except Exception:
        return [p.strip() for p in _END.split(txt) if p.strip()]
