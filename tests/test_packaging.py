"""Checks on pyproject.toml itself.

Packaging bugs are invisible during development — your working copy has every
package installed for unrelated reasons, so an undeclared dependency only
surfaces when someone installs from PyPI into a clean environment and gets
`ModuleNotFoundError`. These tests read the metadata and compare it against
what the code actually imports.

This suite already caught three: pandas, numpy and PyYAML were imported at
module level and declared nowhere, so a fresh install could not run
`feature_gather` (which the conversation_video preset uses four times).
"""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SRC = PROJECT_ROOT / "src" / "taters"

# import name -> distribution name, for the ones where they differ
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "faster_whisper": "faster-whisper",
    "sentence_transformers": "sentence-transformers",
    "parselmouth": "praat-parselmouth",
    "contentcoder": "contentcoder",
    "archetypes": "archetyper",
    "ContentCoder": "contentcoder",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
}

# import-time requirements of the modules behind an optional extra. importing
# these is supposed to fail without the extra installed
OPTIONAL = {
    "parselmouth": "vocalacoustics",
    "soundfile": "vocalacoustics",
    "disvoice": "vocalacoustics",
    "textstat": None,       # a base dependency, since it's tiny and pure Python
    "nemo": "diarization",
    "gensim": "vectors",
    "torch": None,          # not declared on purpose: users pick their own build
    "nvidia": "cuda",
    "chardet": None,        # optional nicety, guarded by try/except
}


@pytest.fixture(scope="module")
def metadata() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def top_level_imports(path: Path) -> set[str]:
    """Third-party modules imported when `path` is imported (not lazily)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n not in sys.stdlib_module_names and n != "taters"}


def source_files() -> list[Path]:
    return [p for p in sorted(SRC.rglob("*.py")) if "whisper-diarization" not in str(p)]


def declared_distributions(metadata: dict) -> set[str]:
    """Every distribution named in dependencies or any extra, normalized."""
    import re
    names: set[str] = set()
    project = metadata["project"]
    groups = [project.get("dependencies", [])]
    groups += list(project.get("optional-dependencies", {}).values())
    for group in groups:
        for spec in group:
            name = re.split(r"[<>=!\[;\s]", spec, maxsplit=1)[0]
            names.add(name.strip().lower().replace("_", "-"))
    return names


def test_pyproject_parses(metadata):
    assert metadata["project"]["name"] == "taters"


def test_every_import_time_dependency_is_declared(metadata):
    """
    Anything imported at module scope must be installable from the metadata —
    either as a core dependency or via a documented extra.
    """
    declared = declared_distributions(metadata)
    missing: list[str] = []

    for path in source_files():
        for module in top_level_imports(path):
            if module in OPTIONAL:
                continue
            dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
            if dist not in declared:
                rel = path.relative_to(PROJECT_ROOT)
                missing.append(f"{module} (imported by {rel}) -> needs '{dist}'")

    assert not missing, "undeclared dependencies:\n  " + "\n  ".join(sorted(missing))


def test_core_dependencies_cover_the_core_helpers(metadata):
    """
    helpers/ and pipelines/ are the parts every workflow touches; they must not
    depend on an optional extra.
    """
    core = {
        name.lower().replace("_", "-")
        for name in (
            __import__("re").split(r"[<>=!\[;\s]", spec, maxsplit=1)[0]
            for spec in metadata["project"]["dependencies"]
        )
    }
    for path in source_files():
        if path.parent.name not in {"helpers", "pipelines"}:
            continue
        for module in top_level_imports(path):
            dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
            assert dist in core, (
                f"{path.name} imports {module} at module level, but '{dist}' is "
                f"not a core dependency"
            )


def test_console_script_targets_exist(metadata):
    """An entry point pointing at a module that does not exist fails on first use."""
    import importlib
    for name, target in (metadata["project"].get("scripts") or {}).items():
        module_path, _, attribute = target.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            pytest.skip(f"{name}: {module_path} needs an optional dependency ({exc})")
        assert hasattr(module, attribute), f"{name}: {module_path} has no {attribute}()"
        assert callable(getattr(module, attribute))


def test_declared_python_version_matches_what_the_code_uses(metadata):
    """
    The CLI uses argparse.BooleanOptionalAction and the code uses `X | Y` type
    syntax, both of which need 3.10+.
    """
    assert metadata["project"]["requires-python"] == ">=3.10"


def test_packaging_does_not_ship_the_tests(metadata):
    """Tests live outside src/, so the wheel should not contain them."""
    find = metadata["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert all(pattern.startswith("taters") for pattern in find["include"])


def test_the_wizard_is_reachable_as_bare_taters(metadata):
    """
    `taters` with no arguments is the front door for people who do not write
    code. If the entry point is missing or renamed, the install guide's very
    first instruction stops working.
    """
    scripts = metadata["project"].get("scripts") or {}
    assert scripts.get("taters") == "taters.ui.wizard:main"


def test_the_wizard_ui_libraries_are_core_dependencies(metadata):
    """
    `questionary`, `rich` and `prompt_toolkit` are all imported lazily -- inside
    `QuestionaryPrompter.__init__` and `LivePrompter._wrap` -- so the
    module-level import scan above cannot see any of them. They still have to be
    in the base install: a wizard that greets a new user with
    `ModuleNotFoundError` is worse than no wizard.

    `prompt_toolkit` is listed even though questionary depends on it. Relying on
    a transitive dependency means the day questionary drops or swaps it, the
    live renderer breaks with no warning from anything here.
    """
    core = {req.split(">=")[0].split("[")[0].split("<")[0].strip().lower()
            for req in metadata["project"]["dependencies"]}
    assert {"questionary", "rich", "prompt_toolkit"} <= core


def test_setup_py_is_a_shim_and_not_a_second_source_of_truth():
    """
    setup.py exists so that `python setup.py install` works for anyone who
    clones the repo and reaches for the command out of habit. It must stay a
    bare `setup()` call that reads pyproject.toml.

    The failure this guards against is duplication, not absence: the moment
    someone adds `install_requires=[...]` or `version="0.2.1"` here, there are
    two places to edit and one of them will be forgotten. That drift is silent
    -- pip prefers the pyproject metadata, so a stale setup.py only bites the
    person using the legacy command.
    """
    setup_py = PROJECT_ROOT / "setup.py"
    assert setup_py.exists(), "setup.py is documented in the install guide"

    tree = ast.parse(setup_py.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "setup"]

    assert len(calls) == 1, "expected exactly one setup() call"
    assert not calls[0].args and not calls[0].keywords, (
        "setup() must take no arguments -- metadata belongs in pyproject.toml"
    )


def test_standard_is_everything_except_diarization(metadata):
    """
    `standard` exists so someone can ask for "everything that will just work"
    without knowing which extras happen to be wheel-only.

    Adding a new extra and forgetting to fold it into `standard` is the drift
    this catches: the name would quietly stop meaning what it says, and the
    person it was built for is exactly the person who would not notice.
    """
    extras = metadata["project"]["optional-dependencies"]
    diarization = set(extras["diarization"])

    assert set(extras["standard"]) == set(extras["all"]) - diarization
    assert not set(extras["standard"]) & diarization


# the extras we leave out of `all` on purpose, and why. adding to this needs a
# real reason: people assume `all` means all, and every exception makes the
# name a little less true
NOT_IN_ALL = {
    # `disvoice` needs `phonet`, which publishes no wheels and no static
    # metadata, so pip builds every candidate version just to read its
    # dependencies, then backtracks the rest of the resolution around whatever
    # it finds. on Python 3.14 that walk ends at a source-only `contourpy` and
    # a missing Visual Studio, and `pip install taters[all]` just fails outright.
    #
    # all that buys us is three features (HRF, NAQ, OQ) from one code path that
    # already warns and carries on without them. trading a working install for
    # that is the wrong way round, so it's opt-in: `pip install taters[glottal]`
    "glottal",
}


def test_every_feature_extra_is_reachable_from_all(metadata):
    """`all` has to actually mean all, apart from documented exceptions."""
    extras = metadata["project"]["optional-dependencies"]
    # `dev` is tooling, not a feature, and `standard`/`all` are the aggregates
    features = set(extras) - {"dev", "all", "standard"} - NOT_IN_ALL

    for name in sorted(features):
        missing = set(extras[name]) - set(extras["all"])
        assert not missing, f"extra {name!r} has {sorted(missing)}, absent from 'all'"


def test_the_exceptions_to_all_are_real_extras(metadata):
    """
    A name in `NOT_IN_ALL` that no longer exists is an exemption still being
    granted to nothing -- and the next extra to need one will not get it.
    """
    extras = set(metadata["project"]["optional-dependencies"])
    assert NOT_IN_ALL <= extras, f"stale exemptions: {sorted(NOT_IN_ALL - extras)}"


def test_the_optional_extras_are_genuinely_optional(metadata):
    """
    The rule that earns `glottal` its exemption: anything kept out of `all` has
    to degrade gracefully when absent, not raise at import.
    """
    source = (SRC / "audio" / "analyze_vocal_acoustics.py").read_text(encoding="utf-8")

    # imported inside a function, inside a try, and never at module scope
    assert "\nfrom disvoice" not in source and "\nimport disvoice" not in source
    assert "        from disvoice.glottal import Glottal" in source
    assert "DisVoice not installed or failed" in source


# ---------------------------------------------------------------------------
# version floors that exist to keep the resolver out of a ditch
#
# this bit us on a Windows install with Python 3.14: `pip install .[all]` spent
# minutes backtracking through six NeMo versions and died with
#
#     ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ...]
#     × Preparing metadata (pyproject.toml) did not run successfully.  (numpy)
#
# NeMo 2.6's `[asr]` extra pins `numpy<2.0.0`, and the newest numpy under 2 is
# 1.26.4, which predates Python 3.14 and so has no wheel for it. pip's answer
# was to try building numpy from source, which needs a C toolchain, and then
# it reported that as a problem with numpy. nothing in the message named NeMo
# ---------------------------------------------------------------------------

def _requirement(metadata: dict, name: str) -> str:
    extras = metadata["project"]["optional-dependencies"]
    for deps in extras.values():
        for dep in deps:
            if dep.startswith(name):
                return dep
    raise AssertionError(f"{name} is not required by any extra")


def test_the_diarization_extra_avoids_the_numpy_pin(metadata):
    """
    The floor is load-bearing, not tidiness. Below 2.7, NeMo's `[asr]` extra
    drags in `numpy<2` and a source-only `ctc_segmentation`, either of which
    turns a wheel-only install into one that needs a compiler.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    req = Requirement(_requirement(metadata, "nemo-toolkit"))
    floors = [Version(spec.version) for spec in req.specifier if spec.operator == ">="]
    assert floors and min(floors) >= Version("2.7"), (
        f"{req} can resolve to a NeMo whose asr extra pins numpy<2"
    )


def test_the_diarization_extra_still_excludes_nemo_3(metadata):
    """
    The other half of the range, and the reason it is a range: NeMo 3.0 removed
    `nemo.collections.asr.models.msdd_models`, which the vendored
    whisper-diarization code imports.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    req = Requirement(_requirement(metadata, "nemo-toolkit"))
    assert not req.specifier.contains(Version("3.0.0"))


def test_every_extra_pins_nemo_the_same_way(metadata):
    """
    `diarization` and `all` both carry it. A floor raised in one and not the
    other is a floor that is not there.
    """
    extras = metadata["project"]["optional-dependencies"]
    pins = {name: [d for d in deps if d.startswith("nemo-toolkit")]
            for name, deps in extras.items()}
    declared = {tuple(v) for v in pins.values() if v}
    assert len(declared) == 1, f"NeMo is pinned inconsistently: {pins}"


def test_the_diarization_extra_declines_pythons_it_cannot_serve(metadata):
    """
    Every NeMo from 2.7 through 3.0 requires
    `nv-one-logger-pytorch-lightning-integration`, which declares
    `Requires-Python: >=3.9,<3.14`; every 2.6.x requires `numpy<2`, whose newest
    release predates 3.14. So on 3.14 there is no installable NeMo, and asking
    for one takes the *whole* install down with `ResolutionImpossible` -- losing
    transcription, embeddings and the text analyzers, none of which have
    anything to do with diarization.

    The marker turns that into a partial install plus an honest report. Drop it
    when NVIDIA lifts the cap.
    """
    from packaging.requirements import Requirement

    req = Requirement(_requirement(metadata, "nemo-toolkit"))
    assert req.marker is not None, "nemo is requested on every Python, including 3.14"

    on_314 = req.marker.evaluate({"extra": "all", "python_version": "3.14"})
    on_312 = req.marker.evaluate({"extra": "all", "python_version": "3.12"})
    assert not on_314, "3.14 has no installable NeMo, so it must not be requested"
    assert on_312, "the marker excludes Pythons that can run diarization perfectly well"


def test_the_vectors_extra_declines_pythons_it_cannot_serve(metadata):
    """
    gensim publishes wheels up to Python 3.13 and none for 3.14. Asked for on
    3.14, pip tries to compile it and stops 700 lines later on "Microsoft
    Visual C++ 14.0 or greater is required" -- after taters itself built fine,
    so the whole install fails for a step most people never tick.

    The marker turns that into "grayed out, and the setup check says why".
    Every extra that lists gensim has to carry it, or `standard` and `all`
    keep failing while `vectors` alone is fixed. Drop it when gensim ships
    3.14 wheels.
    """
    from packaging.requirements import Requirement

    extras = metadata["project"]["optional-dependencies"]
    listing = {name: [d for d in deps if d.startswith("gensim")]
               for name, deps in extras.items()}
    listing = {k: v for k, v in listing.items() if v}
    assert listing, "no extra lists gensim at all"
    for name, deps in listing.items():
        req = Requirement(deps[0])
        assert req.marker is not None, f"[{name}] asks for gensim on every Python"
        assert not req.marker.evaluate({"extra": name, "python_version": "3.14"}), (
            f"[{name}] still asks for gensim on 3.14, where there is none to install")
        assert req.marker.evaluate({"extra": name, "python_version": "3.13"}), (
            f"[{name}] excludes a Python that can run gensim perfectly well")


# ---------------------------------------------------------------------------
# the docs make specific promises about these pins
# ---------------------------------------------------------------------------

DOCS = PROJECT_ROOT / "docs" / "install-guide.md"
HELP = PROJECT_ROOT / "docs" / "install-troubleshooting.md"
README = PROJECT_ROOT / "README.MD"

# terms that only mean something to somebody who already knows the answer
JARGON = ("nv-one-logger", "Requires-Python", "ctc_segmentation", "cublas",
          "numpy<2", "backtrack", "resolver", "ResolutionImpossible", "msdd",
          "sdist", "RPATH", "ABI")


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_install_page_stays_short_enough_to_read():
    """
    The property that matters most and is easiest to lose: someone landing here
    should think "that looks easy", not scroll. This page reached 400 lines and
    22 headings before it was split, at which point the first thing a new user
    saw was a wall.

    Depth is not banned -- it lives in install-troubleshooting.md, which nobody
    reads until something breaks.
    """
    text = DOCS.read_text(encoding="utf-8")
    headings = [ln for ln in text.splitlines() if ln.startswith("#")]

    assert len(text.splitlines()) < 160, f"{len(text.splitlines())} lines"
    assert len(headings) <= 10, f"{len(headings)} headings: {headings}"


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_install_page_explains_nothing_a_beginner_did_not_ask():
    """
    Not just the top of the page -- the whole of it. A short page earns its
    shortness by sending the diagnosis elsewhere, not by compressing it.
    """
    import re

    text = DOCS.read_text(encoding="utf-8").lower()
    for term in JARGON:
        # word boundaries, otherwise "ABI" matches inside "readability"
        assert not re.search(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])", text), (
            f"the install page explains {term!r} before anyone has hit it"
        )


@pytest.mark.skipif(not HELP.exists(), reason="docs are not present in this checkout")
def test_the_detail_moved_rather_than_vanished():
    """
    All of it is real and someone mid-failure needs it. The split is about
    *where*, not about deleting hard-won explanations.
    """
    text = HELP.read_text(encoding="utf-8")
    for term in ("nv-one-logger", "numpy<2", "CTranslate2", "cu128"):
        assert term in text, f"{term} was dropped in the split, not relocated"

    folded = "\n".join(text.split("??? ")[1:])
    assert "nv-one-logger" in folded, "the deepest detail is not behind a fold"
    assert "CTranslate2" in folded


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_docs_name_the_python_cap_the_metadata_enforces(metadata):
    """
    The docs promise diarization needs 3.13 or older. That is only true while
    the marker in pyproject.toml says so, and the two are edited months apart.
    """
    from packaging.requirements import Requirement

    req = Requirement(_requirement(metadata, "nemo-toolkit"))
    assert not req.marker.evaluate({"extra": "all", "python_version": "3.14"})

    pages = DOCS.read_text(encoding="utf-8") + HELP.read_text(encoding="utf-8")
    assert "3.13" in pages, "nothing names the newest Python that runs diarization"
    assert "not available for Python 3.14" in pages, (
        "nothing shows what the app actually reports"
    )
    # word vectors have the same cap for the same reason, and somebody who hit
    # the compiler error needs to find the answer under a heading of its own
    assert "gensim" in HELP.read_text(encoding="utf-8"), (
        "the troubleshooting page never says which Pythons can train word vectors"
    )


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_docs_do_not_call_the_app_a_wizard():
    """
    "Setup wizard" frames it as a one-time thing you get through on the way to
    the real program. It *is* the program -- and a name that says otherwise
    stops people opening it again.
    """
    for path in (DOCS, HELP, README):
        text = path.read_text(encoding="utf-8").lower()
        assert "setup wizard" not in text, f"{path.name} calls the app a wizard"


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_docs_do_not_lean_on_one_adverb():
    """
    "Quietly" and "silently" are useful once. Across a page they stop carrying
    meaning and start reading as a tic.
    """
    import re

    for path in (DOCS, HELP, README):
        text = path.read_text(encoding="utf-8").lower()
        hits = len(re.findall(r"\b(?:quietly|silently)\b", text))
        assert hits <= 1, f"{path.name} uses quietly/silently {hits} times"


@pytest.mark.skipif(not DOCS.exists(), reason="docs are not present in this checkout")
def test_the_guide_does_not_contradict_itself_about_python():
    """
    The guide said "3.11 or 3.12", "≤ 3.12" and "3.10–3.13" in different
    sections at the same time, because each was written when a different thing
    was true. A reader hitting two of them has no way to know which is current.
    """
    import re

    text = DOCS.read_text(encoding="utf-8")

    # claims of the form "Python <= 3.x" / "3.x or older" about diarization
    stale = re.findall(r"(?:Python )?(?:≤ ?|<= ?)3\.(\d+)|3\.(\d+) or older", text)
    ceilings = {int(a or b) for a, b in stale}
    assert ceilings <= {13}, (
        f"the guide names more than one newest-supported Python: {sorted(ceilings)}"
    )




def test_preset_writers_go_through_the_atomic_helper():
    """
    From the code review (issue 20): `write_text` straight onto a preset path
    leaves a truncated YAML behind a Ctrl-C -- which the tolerant lister still
    shows, and which then breaks whatever opens it. `helpers/atomic.py` exists
    for exactly this, so writing a preset any other way is pinned as an error.
    """
    from pathlib import Path

    for name in ("src/taters/ui/wizard.py", "src/taters/ui/tasks/manage.py"):
        text = (Path(__file__).resolve().parent.parent / name).read_text(
            encoding="utf-8")
        offenders = [
            line.strip() for line in text.splitlines()
            if ".write_text(" in line and "yaml" in line.lower()
        ]
        assert not offenders, f"{name}: {offenders}"


def test_the_builtin_stoplists_ship_with_the_package(metadata):
    """
    The stoplists seed the library on first use, which only works installed if
    the wheel actually carries them: the package-data glob must cover the
    data/library tree, and the files must be where the seeder looks.
    """
    patterns = metadata["tool"]["setuptools"]["package-data"]["taters"]
    assert any(p.startswith("data/library") for p in patterns), patterns

    shipped = PROJECT_ROOT / "src" / "taters" / "data" / "library" / "stoplists"
    names = {f.name for f in shipped.glob("*.txt")}
    assert {"stopwords-en.txt", "_chars.txt", "_chars_extended.txt"} <= names
    assert len(names) >= 24


def test_the_builtin_norms_ship_with_the_package(metadata):
    """
    The norm tables are gitignored, the same way the dictionaries are, so a
    fresh clone has an empty folder and nothing complains until somebody
    installs the wheel and finds no norms in it. This is the check that turns
    that into a build failure instead.
    """
    patterns = metadata["tool"]["setuptools"]["package-data"]["taters"]
    assert any(p.startswith("data/library") for p in patterns), patterns

    shipped = PROJECT_ROOT / "src" / "taters" / "data" / "library" / "norms"
    names = {f.stem for f in shipped.glob("*.csv")}
    assert {"Concreteness (Brysbaert)", "Affective Norms (Warriner)",
            "Lancaster Sensorimotor"} <= names, sorted(names)
    assert len(names) >= 14


def test_no_shipped_norm_table_carries_an_intercept_row(metadata):
    """
    An `_intercept` row means the weights were fitted with a constant term,
    and norm scoring reports a plain mean, so the constant would be silently
    dropped. The converter strips them; this is what keeps them stripped.
    """
    import csv

    from taters.helpers.library import INTERCEPT_TERM

    shipped = PROJECT_ROOT / "src" / "taters" / "data" / "library" / "norms"
    offenders = []
    for f in sorted(shipped.glob("*.csv")):
        with f.open("r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.reader(fh):
                if row and row[0].strip().lower() == INTERCEPT_TERM:
                    offenders.append(f.name)
                    break
    assert not offenders, offenders


def test_stereotype_content_is_a_norm_table_not_a_dictionary(metadata):
    """
    Its cells are ratings, so scoring it as a word-counting dictionary gives a
    rate where a mean belongs. A stale copy left in the dictionaries folder
    would be scored the wrong way without anything looking wrong, which is the
    exact failure the two separate libraries exist to prevent.
    """
    library = PROJECT_ROOT / "src" / "taters" / "data" / "library"
    assert not (library / "dictionaries" / "Stereotype Content.dicx").exists()
    assert (library / "norms" / "Stereotype Content.csv").exists()


def test_the_wordcloud_font_ships_with_the_package(metadata):
    """
    The word clouds are drawn in one bundled font. An installed wheel without
    it would draw nothing, so the package-data glob must cover the fonts
    folder, the file must be there, and its license must travel with it --
    the Bitstream Vera license requires the notice to accompany the font.
    """
    patterns = metadata["tool"]["setuptools"]["package-data"]["taters"]
    assert any(p.startswith("figures/fonts") for p in patterns), patterns
    fonts = SRC / "figures" / "fonts"
    assert (fonts / "DejaVuSans-Bold.ttf").is_file()
    assert (fonts / "LICENSE").is_file()
    assert "Bitstream" in (fonts / "LICENSE").read_text(encoding="utf-8")
    assert "src/taters/figures/fonts/LICENSE" in \
        metadata["tool"]["setuptools"]["license-files"]


def test_the_renderer_does_not_import_pillow_at_module_scope():
    """Pillow is loaded when a picture is drawn, not when taters is imported:
    a machine without it still runs every analysis and is told, once, that
    the figures were skipped."""
    assert "PIL" not in top_level_imports(SRC / "figures" / "render.py")


def test_no_source_file_needs_a_python_newer_than_we_claim():
    """
    `finetune_predictor` carried an f-string that reused the same quote inside
    a replacement field. That only became legal in 3.12 (PEP 701), so on 3.11
    it was a SyntaxError at import -- which takes the whole test run down at
    collection rather than failing one test, and which nothing local caught
    because the developer's Python is newer than the floor we advertise.

    `ast.parse(feature_version=...)` does NOT catch it: feature_version tunes
    only a handful of grammar decisions and f-string quoting is not one of
    them. So this reads the declared floor out of pyproject and refuses the
    construct textually.
    """
    import re

    floor = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["requires-python"]
    lowest = tuple(int(n) for n in re.search(r"(\d+)\.(\d+)", floor).groups())
    if lowest >= (3, 12):
        pytest.skip(f"requires-python is {floor}; nested f-string quotes are legal")

    offenders = []
    for path in SRC.rglob("*.py"):
        if "diarizer" in path.parts:          # vendored, not ours to restyle
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for quote in ("'", '"'):
                # an f-string opened with `quote` whose replacement field uses
                # the same `quote` again before the string closes
                if re.search(rf"f{quote}[^{quote}]*\{{[^}}]*{quote}", line):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert not offenders, (
        f"these reuse a quote inside an f-string, which needs 3.12 but "
        f"requires-python says {floor}: {offenders}")
