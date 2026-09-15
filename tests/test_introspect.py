"""
Tests for the layer that reads a function and describes it to a user interface.

`taters.ui.introspect` is what lets the wizard offer options for a module
without anyone hand-writing a list of them. It reads two things: the signature
(names, types, defaults) and the numpydoc `Parameters` block (prose, and any
enumerated set of allowed values).

The docstrings in this package are not uniform, and that is the interesting
part. Three genuinely different spellings appear in the tree, and the parser
has to handle all three or the wizard silently loses help text for whole
modules. Each one gets a test below, pinned to the real function that uses it —
so if someone reformats that docstring, this tells you the UI just got worse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import pytest

from taters.ui.introspect import (
    EMPTY,
    describe,
    load_target,
    parse_numpydoc_params,
    widget_for,
)

# ---------------------------------------------------------------------------
# Docstring shapes
# ---------------------------------------------------------------------------

def test_parses_the_common_name_colon_type_form():
    """`sample_rate : int, default 16000` — what most of the codebase uses."""
    docs = parse_numpydoc_params("""
        Summary line.

        Parameters
        ----------
        sample_rate : int, default 16000
            Desired sample rate (Hz).
        overwrite_existing : bool, default False
            Overwrite the output if it already exists.
    """)
    assert set(docs) == {"sample_rate", "overwrite_existing"}
    assert docs["sample_rate"].type_str == "int, default 16000"
    assert docs["sample_rate"].desc == "Desired sample rate (Hz)."


def test_parses_a_bare_name_with_no_type():
    """`helpers/find_files.py` documents parameters without the type half."""
    docs = parse_numpydoc_params("""
        Summary.

        Parameters
        ----------
        root_dir
            Folder to scan.
        recursive
            Recurse into subfolders. Default: `True`.
    """)
    assert docs["root_dir"].desc == "Folder to scan."
    assert docs["root_dir"].type_str == ""


def test_expands_two_names_sharing_one_entry():
    """`include_globs / exclude_globs` is one paragraph documenting two params."""
    docs = parse_numpydoc_params("""
        Summary.

        Parameters
        ----------
        include_globs / exclude_globs
            Additional glob filters applied after extension filtering.
    """)
    assert "include_globs" in docs and "exclude_globs" in docs
    assert docs["include_globs"].desc == docs["exclude_globs"].desc


def test_stops_at_the_next_section():
    """Returns/Raises/Notes must not be swallowed as more parameters."""
    docs = parse_numpydoc_params("""
        Summary.

        Parameters
        ----------
        alpha : int
            The only parameter.

        Returns
        -------
        Path
            Not a parameter.

        Raises
        ------
        ValueError
            Also not a parameter.
    """)
    assert list(docs) == ["alpha"]


def test_multi_line_descriptions_are_joined():
    docs = parse_numpydoc_params("""
        Parameters
        ----------
        channels : int | None, default 1
            If provided, set number of output channels.
            If None, keep original channel count.
    """)
    assert "keep original channel count" in docs["channels"].desc


@pytest.mark.parametrize("doc", [None, "", "Just a summary, no sections."])
def test_missing_parameters_section_is_not_an_error(doc):
    """An undocumented function must still be usable, just with less help."""
    assert parse_numpydoc_params(doc) == {}


# ---------------------------------------------------------------------------
# Enumerated choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_str,expected", [
    ("{16,24,32}, default 16", [16, 24, 32]),
    ('{"ms", "s"}, default "ms"', ["ms", "s"]),
    ('{"concat", "separate"}, default="concat"', ["concat", "separate"]),
    ("int, default 5", None),
    ("", None),
])
def test_choice_sets_are_lifted_from_the_type_string(type_str, expected):
    """
    A closed set of values should render as a picker, not a text box.

    The type string is the only place this information exists — the annotation
    is just `int` or `str`.
    """
    spec = describe(_fn_with_doc(f"""
        Parameters
        ----------
        thing : {type_str}
            A thing.
    """)).get("thing")
    assert spec.choices == expected
    assert (spec.widget == "choice") is (expected is not None)


def test_a_literal_annotation_is_a_picker_even_with_no_docstring_type():
    """
    `bookkeeping: Literal["aside", "features"]` with a docstring header of
    just its name was a text box asking someone to spell "features". The
    signature said what the answers were all along; it counts, Optional or
    not, and the docstring's own set still wins when both are present.
    """
    # (we import `Literal` at module level on purpose: with postponed
    # annotations the signature gets evaluated in the function's globals, and
    # a name imported inside the test wouldn't be there. that's also why the
    # analyzers import it at the top of their modules.)
    def f(mode: Literal["aside", "features"] = "aside",
          maybe: Optional[Literal["a", "b"]] = None,
          both: Literal["x", "y"] = "x"):
        """
        Parameters
        ----------
        mode
            Which.
        maybe
            Perhaps.
        both : {"p", "q"}
            The docstring disagrees, and is the one written for the reader.
        """

    spec = describe(f)
    assert spec.get("mode").widget == "choice"
    assert spec.get("mode").choices == ["aside", "features"]
    assert spec.get("maybe").choices == ["a", "b"]
    assert spec.get("both").choices == ["p", "q"]
    assert not spec.get("mode").open_ended


@pytest.mark.parametrize("type_str,expected", [
    ('{"tiny", "base"} or str, default "base"', True),
    ('{"tiny", "base"} or a path, default "base"', True),
    ('{"auto", "cuda", "cpu"} | None, default "auto"', False),
    ('{"count", "tfidf"}, default="count"', False),
])
def test_a_set_followed_by_or_str_is_open_ended(type_str, expected):
    """
    `{"tiny", "base"} or str` names the stock models and still takes a model
    folder, so its picker must keep a way to type. `{...} | None` does not:
    "unset" is one of the answers, not a license for any string.
    """
    spec = describe(_fn_with_doc(f"""
        Parameters
        ----------
        thing : {type_str}
            A thing.
    """)).get("thing")
    assert spec.widget == "choice"
    assert spec.open_ended is expected


def _fn_with_doc(doc: str):
    """Build a throwaway function carrying `doc`, for parser tests."""
    def f(thing: int = 1):
        pass
    f.__doc__ = doc
    return f


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def test_string_annotations_are_evaluated():
    """
    Every module here starts with `from __future__ import annotations`, which
    makes annotations plain strings at runtime. Without `eval_str=True` the
    widget picker would be dispatching on the *text* "Optional[Union[str, Path]]"
    and would fall through to a generic text box for every path in the project.
    """
    spec = describe(load_target("taters.audio.convert_to_wav:convert_audio_to_wav"))
    output_path = spec.get("output_path")
    assert not isinstance(output_path.annotation, str)
    assert output_path.widget == "path"


@pytest.mark.parametrize("annotation,name,expected", [
    (bool, "flag", "bool"),
    (int, "count", "int"),
    (float, "ratio", "float"),
    (str, "label", "text"),
    (Path, "somewhere", "text"),
    (Optional[int], "count", "int"),
    (Union[str, Path], "input_path", "path"),
    (Optional[Sequence[str]], "cols", "list"),
    (str, "out_dir", "dir"),
    (str, "transcript_csv", "path"),
])
def test_widget_for_maps_annotations(annotation, name, expected):
    assert widget_for(annotation, name) == expected


def test_a_concrete_type_beats_the_name_heuristic():
    """
    `analyze_readability` has a real `include_source_path: bool`. The `_path`
    suffix must not win, or the wizard renders a file picker for a yes/no
    question.
    """
    assert widget_for(bool, "include_source_path") == "bool"


def test_widget_falls_back_to_the_default_value():
    """Unannotated parameters still get a sensible widget."""
    assert widget_for(None, "n", default=5) == "int"
    assert widget_for(None, "flag", default=True) == "bool"
    assert widget_for(None, "cols", default=["a"]) == "list"


# ---------------------------------------------------------------------------
# describe() over the real modules
# ---------------------------------------------------------------------------

def test_required_and_optional_are_distinguished():
    spec = describe(load_target("taters.audio.convert_to_wav:convert_audio_to_wav"))
    assert spec.get("input_path").required is True
    assert spec.get("sample_rate").required is False
    assert spec.get("sample_rate").default == 16000


def test_a_default_of_none_is_not_the_same_as_no_default():
    """
    Both are falsy and both mean "you did not pass this", but only one may be
    omitted from a call. `EMPTY` keeps them apart.
    """
    spec = describe(load_target("taters.audio.convert_to_wav:convert_audio_to_wav"))
    assert spec.get("output_path").default is None
    assert spec.get("output_path").has_default is True
    assert spec.get("input_path").default is EMPTY
    assert spec.get("input_path").has_default is False


def test_varargs_are_not_offered_as_fields():
    def f(a, *args, **kwargs):
        pass
    assert describe(f).names == ["a"]


def test_summary_is_the_first_docstring_line():
    spec = describe(load_target("taters.audio.transcribe_with_whisper:transcribe_with_whisper"))
    assert spec.summary.startswith("Transcribe an audio file")


def test_undocumented_functions_still_produce_fields():
    """
    Three helpers document their parameters in prose rather than a numpydoc
    block. They must still be renderable — types and defaults alone are enough
    to draw a form, just without help text.
    """
    spec = describe(load_target("taters.helpers.text_gather:csv_to_analysis_ready_csv"))
    assert spec.get("csv_path") is not None
    assert spec.get("num_buckets").widget == "int"


def test_as_dict_is_json_shaped():
    """A web UI or a tool schema needs these to serialize."""
    import json
    spec = describe(load_target("taters.audio.convert_to_wav:convert_audio_to_wav"))
    json.dumps(spec.as_dict())          # this just has to not blow up


def test_load_target_accepts_both_spellings():
    colon = load_target("taters.audio.convert_to_wav:convert_audio_to_wav")
    dotted = load_target("taters.audio.convert_to_wav.convert_audio_to_wav")
    assert colon is dotted


# ---------------------------------------------------------------------------
# Descriptions a person can read
# ---------------------------------------------------------------------------

def test_a_wrapped_description_becomes_one_sentence():
    """
    Docstrings are hard-wrapped, so a description arrives as several short
    lines. Anything showing only the first line shows half a sentence -- which
    is what the wizard used to do.
    """
    from taters.ui.introspect import clean_doc

    assert clean_doc("Output file path. If None,\ndefaults to ./features/x.csv.") == (
        "Output file path. If None, defaults to ./features/x.csv."
    )


def test_restructured_text_markup_is_removed():
    """``None`` renders as literal backticks in a terminal."""
    from taters.ui.introspect import clean_doc

    assert clean_doc("Use ``None`` to disable.") == "Use None to disable."
    assert clean_doc("See :func:`taters.helpers.progress.count_rows`.") == (
        "See taters.helpers.progress.count_rows."
    )
    assert clean_doc("See :mod:`~taters.ui.compose`.") == "See taters.ui.compose."


def test_paragraph_breaks_survive():
    """A blank line is a deliberate break, not wrapping."""
    from taters.ui.introspect import clean_doc

    out = clean_doc("First thing.\nStill first.\n\nSecond thing.")
    assert out == "First thing. Still first.\n\nSecond thing."


def test_an_empty_description_stays_empty():
    from taters.ui.introspect import clean_doc

    assert clean_doc("") == ""
    assert clean_doc("   \n  ") == ""


def test_real_descriptions_come_through_whole():
    """The end-to-end case: a live function's docstring, ready to display."""
    spec = describe(load_target(
        "taters.text.analyze_with_dictionaries:analyze_with_dictionaries"))

    desc = spec.get("rounding").desc
    assert "``" not in desc
    assert desc.endswith("disable rounding.")


def test_comma_separated_names_share_one_description():
    """
    From the code review (issue 15): numpydoc's own multi-name form is
    `a, b : type`, but only the slash variant was recognized -- a comma header
    silently matched nothing and both parameters reached the wizard with no
    help text at all.
    """
    doc = """Summary.

    Parameters
    ----------
    start, end : float
        Where the window begins and ends.
    """
    params = parse_numpydoc_params(doc)

    assert set(params) >= {"start", "end"}
    assert "window begins" in params["start"].desc
    assert params["start"].desc == params["end"].desc
    assert params["start"].type_str == "float"
