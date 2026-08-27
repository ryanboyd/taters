"""Tests for the Taters facade (taters/Taters.py) and the package surface.

The facade is thin, but it is the thing every user and every pipeline preset
touches first, so its contracts are worth pinning down: the import works, the
namespaces exist, nothing heavy is imported until you call something, and a
typo produces a useful message instead of a confusing one.
"""

import inspect
import os
import subprocess
import sys

import pytest

import taters
from taters import Taters


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_the_documented_import_actually_works():
    """`from taters import Taters` must give the class, not the module.

    Regression test: the package __init__ used to be empty, so this bound the
    submodule and `Taters()` raised "'module' object is not callable" — which
    is the first line of every example in the README.
    """
    assert inspect.isclass(Taters)
    assert Taters().audio is not None


def test_both_import_spellings_give_the_same_class():
    from taters.Taters import Taters as FromModule
    assert Taters is FromModule


def test_package_exposes_a_version():
    assert isinstance(taters.__version__, str) and taters.__version__


@pytest.mark.parametrize("namespace", ["audio", "text", "helpers"])
def test_namespaces_exist(namespace):
    assert hasattr(Taters(), namespace)


# ---------------------------------------------------------------------------
# Forwarding contracts
# ---------------------------------------------------------------------------

FACADE_METHODS = [
    ("audio", "convert_to_wav"),
    ("audio", "extract_wavs_from_video"),
    ("audio", "split_wav_by_speaker"),
    ("audio", "extract_whisper_embeddings"),
    ("audio", "diarize_with_thirdparty"),
    ("audio", "analyze_vocal_acoustics"),
    ("text", "analyze_with_dictionaries"),
    ("text", "analyze_with_archetypes"),
    ("text", "analyze_readability"),
    ("text", "analyze_lexical_richness"),
    ("text", "extract_sentence_embeddings"),
    ("text", "convert_subtitles"),
    ("helpers", "txt_folder_to_analysis_ready_csv"),
    ("helpers", "csv_to_analysis_ready_csv"),
    ("helpers", "find_files"),
    ("helpers", "feature_gather"),
]


@pytest.mark.parametrize("namespace,method", FACADE_METHODS, ids=lambda v: str(v))
def test_every_namespaced_method_exists(namespace, method):
    assert callable(getattr(getattr(Taters(), namespace), method))


@pytest.mark.parametrize("namespace,method", FACADE_METHODS, ids=lambda v: str(v))
def test_every_method_has_a_top_level_alias(namespace, method):
    """Back-compat: t.convert_to_wav(...) must keep working alongside t.audio.…"""
    assert callable(getattr(Taters(), method))


def test_facade_calls_are_keyword_only():
    """Positional arguments are not supported — the error should say so early."""
    with pytest.raises(TypeError, match="positional"):
        Taters().helpers.find_files("some/dir")


def test_unknown_parameter_raises_a_helpful_error(tmp_path):
    with pytest.raises(TypeError) as excinfo:
        Taters().helpers.find_files(root_dir=str(tmp_path), file_typo="video")
    message = str(excinfo.value)
    assert "file_typo" in message
    assert "Allowed params" in message          # tells you what you could pass
    assert "file_type" in message               # ...including the one you meant


def test_forwarding_actually_reaches_the_implementation(tmp_path):
    (tmp_path / "a.mp4").touch()
    found = Taters().helpers.find_files(root_dir=str(tmp_path), file_type="video")
    assert [p.name for p in found] == ["a.mp4"]


def test_top_level_alias_returns_the_same_thing(tmp_path):
    (tmp_path / "a.mp4").touch()
    t = Taters()
    assert t.find_files(root_dir=str(tmp_path), file_type="video") == \
           t.helpers.find_files(root_dir=str(tmp_path), file_type="video")


# ---------------------------------------------------------------------------
# Laziness
# ---------------------------------------------------------------------------

def test_importing_taters_does_not_drag_in_the_heavy_stack(repo_root):
    """
    Constructing the facade must not import torch, whisper, transformers, and
    friends. That is what keeps `import taters` fast and lets a CPU-only box
    use the text tools without a CUDA stack installed.

    Run in a subprocess so this test sees a clean interpreter regardless of
    what other tests already imported.
    """
    code = (
        "import sys; import taters; taters.Taters()\n"
        "heavy = [m for m in ('torch','transformers','faster_whisper','librosa',"
        "'sentence_transformers','pandas','nemo','parselmouth') if m in sys.modules]\n"
        "print(','.join(heavy))"
    )
    # The child is a fresh interpreter, so it needs to be told where the
    # package lives (pytest's `pythonpath` setting only applies to this process).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert res.stdout.strip() == "", f"eagerly imported: {res.stdout.strip()}"
