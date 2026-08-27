"""The 'don't overwrite unless asked' contract, checked in one place.

Taters promises that every function which writes a file will leave an existing
output alone unless you pass `overwrite_existing=True`. That promise is spread
across a dozen modules, which makes it exactly the kind of rule that quietly
develops holes — one did: the text analyzers honored it for their own output
but silently reused a stale *intermediate* file, so `overwrite_existing=True`
produced features computed from last week's data.

Two kinds of check here:

  1. A structural check — every public entry point that writes files must
     expose the parameter. This one catches a *new* function that forgets.
  2. A behavioral check — for each writer we can run cheaply, prove that the
     file is preserved by default and rebuilt on request.
"""

import csv
import inspect
from pathlib import Path

import pytest

from taters import Taters

# Functions that write files, as (namespace, method). Kept explicit so adding a
# feature extractor means consciously adding it here too.
WRITERS = [
    ("audio", "convert_to_wav"),
    ("audio", "extract_wavs_from_video"),
    ("text", "convert_subtitles"),
    ("audio", "split_wav_by_speaker"),
    ("audio", "extract_whisper_embeddings"),
    ("audio", "diarize_with_thirdparty"),
    ("audio", "analyze_vocal_acoustics"),
    ("text", "analyze_with_dictionaries"),
    ("text", "analyze_with_archetypes"),
    ("text", "analyze_readability"),
    ("text", "analyze_lexical_richness"),
    ("text", "extract_sentence_embeddings"),
    ("helpers", "txt_folder_to_analysis_ready_csv"),
    ("helpers", "csv_to_analysis_ready_csv"),
    ("helpers", "feature_gather"),
]

# `extract_wavs_from_video` still accepts the old `overwrite` spelling, but only
# as a deprecated alias — the real parameter is `overwrite_existing` like
# everywhere else.
DEPRECATED_ALIASES = {("audio", "extract_wavs_from_video"): "overwrite"}


def underlying(namespace: str, method: str):
    """
    Resolve a facade method to the real function it forwards to.

    Several targets sit behind optional dependencies (parselmouth, contentcoder,
    archetyper, ...). Where those are not installed we skip rather than fail:
    "cannot check here" is a different statement from "the contract is broken".
    """
    import importlib
    import re
    src = inspect.getsource(getattr(type(getattr(Taters(), namespace)), method))
    rel, name = re.search(r"from (\S+) import (\w+)", src).groups()
    try:
        module = importlib.import_module("taters" + rel)
    except ImportError as exc:
        pytest.skip(f"{namespace}.{method} needs an optional dependency: {exc}")
    return getattr(module, name)


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("namespace,method", WRITERS, ids=lambda v: str(v))
def test_every_writer_exposes_overwrite_existing(namespace, method):
    params = inspect.signature(underlying(namespace, method)).parameters
    assert "overwrite_existing" in params, (
        f"{namespace}.{method} writes files but has no overwrite_existing parameter"
    )


@pytest.mark.parametrize("namespace,method", WRITERS, ids=lambda v: str(v))
def test_overwrite_existing_defaults_to_false(namespace, method):
    """Never destroy work by default."""
    param = inspect.signature(underlying(namespace, method)).parameters["overwrite_existing"]
    assert param.default is False, f"{namespace}.{method} defaults to {param.default!r}"


@pytest.mark.parametrize("key,alias", list(DEPRECATED_ALIASES.items()), ids=lambda v: str(v))
def test_deprecated_aliases_are_keyword_only_and_default_to_none(key, alias):
    """
    An alias kept for back-compat must not become the recommended spelling: it
    stays keyword-only, and defaults to None so "not passed" is distinguishable
    from "passed False".
    """
    param = inspect.signature(underlying(*key)).parameters[alias]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


# ---------------------------------------------------------------------------
# Behavioral — the writers that need no optional dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def source_csv(tmp_path) -> Path:
    p = tmp_path / "src.csv"
    p.write_text(
        "speaker,text\nalice,the first thing said here\nbob,a second utterance\n",
        encoding="utf-8",
    )
    return p


def rows_in(path) -> int:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def test_gather_preserves_then_rebuilds(source_csv, tmp_path):
    t = Taters()
    out = tmp_path / "gathered.csv"
    out.write_text("sentinel\n", encoding="utf-8")

    t.helpers.csv_to_analysis_ready_csv(
        csv_path=source_csv, out_csv=out, text_cols=["text"]
    )
    assert out.read_text(encoding="utf-8") == "sentinel\n"

    t.helpers.csv_to_analysis_ready_csv(
        csv_path=source_csv, out_csv=out, text_cols=["text"], overwrite_existing=True
    )
    assert rows_in(out) == 2


def test_lexical_richness_preserves_then_rebuilds(source_csv, tmp_path):
    t = Taters()
    out = tmp_path / "features.csv"
    out.write_text("sentinel\n", encoding="utf-8")

    t.text.analyze_lexical_richness(
        csv_path=source_csv, out_features_csv=out, text_cols=["text"]
    )
    assert out.read_text(encoding="utf-8") == "sentinel\n"

    t.text.analyze_lexical_richness(
        csv_path=source_csv, out_features_csv=out, text_cols=["text"],
        overwrite_existing=True,
    )
    assert rows_in(out) == 2


def test_overwrite_also_refreshes_the_intermediate_gather_file(source_csv, tmp_path):
    """
    The regression itself. An analyzer builds an analysis-ready CSV on the way
    to its features file; with overwrite_existing=True *both* must be rebuilt,
    or new rows in the source never reach the features.
    """
    t = Taters()
    out = tmp_path / "features.csv"

    t.text.analyze_lexical_richness(
        csv_path=source_csv, out_features_csv=out, text_cols=["text"],
        group_by=["speaker"],
    )
    assert rows_in(out) == 2

    with source_csv.open("a", encoding="utf-8") as f:
        f.write("carol,a brand new third speaker appears\n")

    t.text.analyze_lexical_richness(
        csv_path=source_csv, out_features_csv=out, text_cols=["text"],
        group_by=["speaker"], overwrite_existing=True,
    )
    assert rows_in(out) == 3, "stale intermediate CSV was reused"


def test_txt_folder_gather_preserves_then_rebuilds(tmp_path):
    t = Taters()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("first document", encoding="utf-8")

    out = tmp_path / "gathered.csv"
    out.write_text("sentinel\n", encoding="utf-8")

    t.helpers.txt_folder_to_analysis_ready_csv(root_dir=corpus, out_csv=out)
    assert out.read_text(encoding="utf-8") == "sentinel\n"

    t.helpers.txt_folder_to_analysis_ready_csv(
        root_dir=corpus, out_csv=out, overwrite_existing=True
    )
    assert rows_in(out) == 1


def test_feature_gather_preserves_then_rebuilds(tmp_path):
    pytest.importorskip("pandas", reason="feature_gather requires pandas")
    t = Taters()
    features = tmp_path / "features"
    features.mkdir()
    (features / "a.csv").write_text("speaker,f1\nalice,1\n", encoding="utf-8")

    out = tmp_path / "all.csv"
    out.write_text("sentinel\n", encoding="utf-8")

    t.helpers.feature_gather(root_dir=features, out_csv=out)
    assert out.read_text(encoding="utf-8") == "sentinel\n"

    t.helpers.feature_gather(root_dir=features, out_csv=out, overwrite_existing=True)
    assert rows_in(out) == 1
