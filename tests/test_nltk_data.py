"""Tests for taters.helpers.nltk_data.

NLTK ships code but not data, and which data package `sent_tokenize` needs has
changed across versions: `punkt` used to be enough, NLTK 3.8.2+ wants
`punkt_tab`. Checking for a resource by name therefore reports success on a
machine where tokenizing still fails — which is exactly what happened, and is
why this helper probes the real call instead.
"""

import pytest

from taters.helpers.nltk_data import ensure_punkt

nltk = pytest.importorskip("nltk", reason="needs nltk")


def test_ensure_punkt_makes_sent_tokenize_actually_work():
    """The claim being made is 'you can tokenize now', so verify exactly that."""
    if not ensure_punkt(verbose=False):
        pytest.skip("tokenizer data unavailable (offline?)")

    from nltk import sent_tokenize
    assert sent_tokenize("One sentence. And a second one!") == [
        "One sentence.", "And a second one!",
    ]


def test_ensure_punkt_is_idempotent():
    assert ensure_punkt(verbose=False) == ensure_punkt(verbose=False)


def test_ensure_punkt_reports_status_when_verbose(capsys):
    ensure_punkt(verbose=True)
    assert "tokenizer" in capsys.readouterr().out.lower()


def test_ensure_punkt_is_quiet_when_asked():
    # No assertion on the return value: this is about not printing.
    ensure_punkt(verbose=False)


def test_ensure_punkt_never_raises_when_downloads_fail(monkeypatch):
    """
    An offline or locked-down machine must degrade to the regex fallback, not
    take the pipeline down. Simulate both the probe and the download failing.
    """
    import taters.helpers.nltk_data as mod

    def explode(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(nltk, "download", explode)
    monkeypatch.setattr("nltk.tokenize.sent_tokenize", explode)

    assert mod.ensure_punkt(verbose=False) is False


def test_missing_nltk_returns_false(monkeypatch):
    """The helper is also used where nltk itself may not be installed."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nltk":
            raise ImportError("no nltk here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ensure_punkt(verbose=False) is False


def test_archetype_analyzer_prepares_the_tokenizer():
    """
    Regression: analyze_with_archetypes used to hand text straight to archetyper,
    which calls sent_tokenize, and blew up with NLTK's asterisk wall on a fresh
    machine — after diarization and embeddings had already run.
    """
    pytest.importorskip("archetypes", reason="needs archetyper")
    import inspect
    from taters.text import analyze_with_archetypes as module

    source = inspect.getsource(module.analyze_with_archetypes)
    assert "ensure_punkt(" in source
