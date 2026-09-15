"""
Every module's command line offers every parameter of the function it drives.

The parsers used to be written by hand beside the functions, and had drifted
in every way a parser can: two flag dialects, booleans that could be switched
on but not off, and each missing between two and eighteen of its function's
parameters. They are now derived from the signature (helpers.cliargs.CliSpec);
this pins that the derivation covers everything and speaks one dialect.
"""

import importlib
import inspect

import pytest

MODULES = [
    "taters.stats.assemble", "taters.stats.group_differences",
    "taters.stats.correlations", "taters.stats.ridge", "taters.stats.classify",
    "taters.stats.pca", "taters.stats.report", "taters.stats.describe",
    "taters.figures.wordclouds",
    "taters.text.analyze_readability", "taters.text.analyze_lexical_richness",
    "taters.text.analyze_word_count", "taters.text.analyze_parts_of_speech",
    "taters.text.analyze_cohesion", "taters.text.analyze_with_dictionaries",
    "taters.text.analyze_with_archetypes", "taters.text.extract_sentence_embeddings",
    "taters.text.analyze_ngram_frequencies", "taters.text.build_doc_term_matrix",
    "taters.text.topic_model_mem", "taters.text.word_vectors",
    "taters.text.transformer_embeddings", "taters.text.adapt_encoder",
    "taters.text.finetune_predictor", "taters.text.hf_classifier", "taters.score_model",
    "taters.helpers.text_gather", "taters.helpers.feature_gather",
    "taters.audio.convert_to_wav", "taters.audio.transcribe_with_whisper",
    "taters.audio.extract_whisper_embeddings", "taters.audio.extract_wav_from_video",
    "taters.audio.split_wav_by_speaker", "taters.audio.diarizer.whisper_diar_wrapper",
]


def _import_or_skip(module):
    """
    The module, or a skip naming what is missing.

    A module that cannot be imported here says nothing about its command
    line. `split_wav_by_speaker` imports pydub, which on 3.13 needs
    `audioop-lts` because the standard library dropped `audioop` -- an
    environment fact, and one every other test that loads these modules
    already steps around.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pytest.skip(f"{module} needs an optional dependency: {exc}")


@pytest.mark.parametrize("module", MODULES)
def test_help_renders_for_every_module_and_subcommand(module):
    """`--help` is the one thing every user runs first. A `%` in a docstring
    ("at least 5% of texts") made argparse's formatter raise, so the topic
    model's command line crashed on `--help` while its parser built fine."""
    mod = _import_or_skip(module)
    for parser in mod.CLI.parsers_for_test().values():
        assert parser.format_help()
    assert mod.CLI.parser().format_help()


@pytest.mark.parametrize("module", MODULES)
def test_every_parameter_has_a_flag_and_every_flag_is_dashed(module):
    mod = _import_or_skip(module)
    cli = mod.CLI
    for command, parser in cli.parsers_for_test().items():
        fn = cli.targets[command]
        params = set(inspect.signature(fn).parameters) - cli.skip
        offered = {a.dest for a in parser._actions if a.dest != "help"}
        assert params <= offered, (module, command, sorted(params - offered))
        for action in parser._actions:
            if action.dest == "help":
                continue
            primary = action.option_strings[0] if action.option_strings else action.dest
            assert "_" not in primary or primary.startswith("--" + action.dest.replace("_", "-")) or not primary.startswith("--"), \
                (module, primary)


def test_booleans_can_be_turned_off_and_lists_repeat():
    from taters.stats.correlations import CLI

    _fn, kw = CLI.kwargs(["--table-csv", "t.csv", "--outcome-col", "a",
                          "--outcome-col", "b,c", "--overwrite-existing", "false",
                          "--pca-rotation"])
    assert kw["outcome_cols"] == ["a", "b", "c"]
    assert kw["overwrite_existing"] is False and kw["pca_rotation"] is True
    # a parameter we don't give doesn't get passed, so the function's default wins.
    assert "rounding" not in kw


def test_the_old_spellings_still_work():
    from taters.stats.classify import CLI
    from taters.stats.ridge import CLI as RIDGE

    _fn, kw = CLI.kwargs(["fit", "--table-csv", "t.csv", "--outcome-col", "y",
                          "--no-zscore", "--overwrite-existing"])
    assert kw["zscore"] is False and kw["overwrite_existing"] is True
    _fn, kw = RIDGE.kwargs(["apply", "--model-json", "m.json", "--input-csv",
                            "in.csv", "--out", "scores.csv"])
    assert kw["out_csv"] == "scores.csv"


def test_the_gatherers_old_selector_flags_pick_the_subcommand():
    """`python -m taters.helpers.text_gather --csv x.csv --text-col text` had
    no subcommand; the flag said which gatherer. The rewrite keeps that."""
    from taters.helpers.text_gather import CLI

    fn, kw = CLI.kwargs(["--csv", "x.csv", "--text-col", "text", "--out", "g.csv"])
    assert fn.__name__ == "csv_to_analysis_ready_csv"
    assert kw["csv_path"] == "x.csv" and kw["text_cols"] == ["text"] and kw["out_csv"] == "g.csv"
    fn, kw = CLI.kwargs(["--txt-dir", "docs", "--no-source-path"])
    assert fn.__name__ == "txt_folder_to_analysis_ready_csv"
    assert kw["root_dir"] == "docs" and kw["include_source_path"] is False


def test_a_parts_of_speech_line_written_for_the_old_parser_still_works():
    from taters.text.analyze_parts_of_speech import CLI

    _fn, kw = CLI.kwargs(["--csv-path", "s.csv", "--id-col", "text_id",
                          "--counts", "--overwrite-existing"])
    assert kw["relative_freq"] is False and kw["id_cols"] == ["text_id"]


def test_a_parameter_that_takes_one_or_many_repeats_on_the_command_line():
    """`model_json` is `Union[PathLike, Sequence[PathLike]]` -- one model or
    several. The union used to render as a text box, so a second
    `--model-json` silently replaced the first."""
    from pathlib import Path
    from typing import Sequence, Union

    from taters.score_model import CLI
    from taters.ui.introspect import widget_for

    assert widget_for(Union[str, Path, Sequence[Union[str, Path]]], "model_json") == "list"
    assert widget_for(Union[str, Path], "model_json") == "text"
    ns = CLI.parser().parse_args(["--model-json", "a.json", "--model-json", "b.json"])
    assert ns.model_json == ["a.json", "b.json"]
