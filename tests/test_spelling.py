"""
The prose is in American English.

Comments, docstrings and the docs were respelled from British to American
on 2026-09-13 (the word list is in ``american_spelling.py``). These keep it
that way: any British spelling that lands in a comment, a docstring or a
Markdown page fails here with the file, the line and the word. Identifiers,
string literals and quoted code are not the prose's business and are never
read -- ``neighbour_wordclouds`` is a name, and a name keeps its spelling.
"""
from __future__ import annotations

from pathlib import Path

from american_spelling import PAIRS, british_words, files_under, scan_markdown, scan_python

ROOT = Path(__file__).resolve().parents[1]


def _report(hits) -> str:
    lines = [f"  {path.relative_to(ROOT)}:{line}: {word!r} -> {PAIRS[word.lower()]!r}"
             for path, line, word in hits[:25]]
    if len(hits) > 25:
        lines.append(f"  ... and {len(hits) - 25} more")
    return "\n".join(lines)


def test_comments_and_docstrings_are_in_american_english():
    hits = []
    for path in files_under([ROOT / "src", ROOT / "tests"], (".py",)):
        for line, word in scan_python(path):
            hits.append((path, line, word))
    assert not hits, "British spellings in comments or docstrings:\n" + _report(hits)


def test_the_docs_and_planning_notes_are_in_american_english():
    hits = []
    roots = [ROOT / "docs", ROOT / "planning", ROOT / "README.MD", ROOT / "CODE_REVIEW.md"]
    for path in files_under(roots, (".md",)):
        for line, word in scan_markdown(path):
            hits.append((path, line, word))
    assert not hits, "British spellings in the docs:\n" + _report(hits)


def test_the_word_list_covers_prefixes_and_the_words_this_codebase_uses():
    """
    The list is generated from stems, and two holes let spellings straight
    back through the door they had just been shown out of.

    `recognised` was caught and `unrecognised` was not, because a prefix made
    it a different word. And `lemma` was a stem while `lemmat` was not -- so
    `lemmatised` sailed past in a project where lemmatizing is on half the
    screens. Both were found by the words that were already in the tree, not
    by reading the list.
    """
    for british, american in (
            ("unrecognised", "unrecognized"),
            ("unnormalised", "unnormalized"),
            ("renormalised", "renormalized"),
            ("lemmatised", "lemmatized"),
            ("lemmatising", "lemmatizing"),
            ("lemmatisation", "lemmatization"),
            ("canonicalised", "canonicalized"),
            ("canonicalisation", "canonicalization"),
            ("recolour", "recolor")):
        assert PAIRS.get(british) == american, f"{british} is not caught"
    # and the prefix rule has not invented a correction for a word that is
    # already American
    assert "unrecognized" not in PAIRS and "relocalize" not in PAIRS


def test_the_scanner_reads_prose_and_leaves_code_alone():
    """The word list is only as useful as the scanner's aim: a quoted
    parameter name and a code span must never count, and a plain word must."""
    assert british_words("we grey the row out; see top_neighbours and ``colour``") == ["grey"]
    assert british_words("`neighbour_wordclouds` draws the neighbours") == ["neighbours"]
    assert british_words("neighbours.csv is the file; the Colour column") == ["Colour"]
    assert british_words("run the analyses; raise Cancelled") == []
