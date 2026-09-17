"""
The declared data-flow catalog: what a user can ask for, and what it needs.

:mod:`taters.ui.introspect` can read every knob off a function's signature, but
it cannot read the *wiring*. Nothing in ``analyze_vocal_acoustics``'s signature
says its ``transcript_csv`` should be fed ``{{pick:diar.raw_files.csv}}`` from
an earlier step. That knowledge lives here, declared by hand, one
:class:`Recipe` per pipeline step.

Every ``with_`` block below is transcribed from the two shipped presets
(``conversation_video.yaml`` and ``single_speaker_media.yaml``). That is
deliberate: those presets are the known-good wiring, and
``tests/test_compose.py`` asserts that selecting the right recipes reproduces
them. If you change a template here, that test tells you.

How dependencies work
---------------------
Steps are linked by **capability strings** -- ``"wav"``, ``"transcript_csv"``,
and so on -- rather than by naming each other directly. A recipe declares what
it ``requires`` and what it ``produces``, and :mod:`taters.ui.compose` walks the
graph. The indirection buys one important thing: ``transcript_csv`` has *two*
providers (plain transcription and diarization), so the user gets to choose how
a requirement is met without any recipe knowing that a choice exists.

``auto_with`` covers the one relationship capabilities cannot express. A gather
step does not *provide* anything the feature step needs -- it tidies up
afterwards -- so it cannot be pulled in by ``requires``. Naming it in
``auto_with`` says "whenever you include me, include this too."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..helpers.gpu import worker_cap

__all__ = [
    "Recipe", "RECIPES", "by_id", "user_facing", "providers_of",
    "gate_of", "gates_of", "gate_holds",
    "CAPABILITIES", "SOURCES", "TEXT_INPUT_KEYS", "TEXT_IDENTITY",
    "LEVELS", "DEFAULT_LEVEL", "Level", "levels_for", "level_by_id",
    "level_aware",
    "text_binding",
]


# human-readable names for the capability strings. the wizard drops these into
# prompts like "Acoustic features needs a transcript. How should we make one?".
CAPABILITIES: Dict[str, str] = {
    "wav": "a WAV version of each input file",
    "transcript_csv": "a transcript",
    "speaker_wavs": "one WAV per speaker",
    "unified_transcripts_csv": "all transcripts merged into one table",
    "acoustics_csv": "per-file acoustic measures",
    "whisper_embeddings_csv": "per-file Whisper embeddings",
    "sentence_embeddings_csv": "row-level sentence embeddings",
    "ngram_freq_csv": "a corpus n-gram frequency list",
    "doc_term_matrix_csv": "a document-term matrix",
    "stats_metadata_csv": "your spreadsheet's grouping/outcome columns, one row per analyzed text",
    "analysis_table_csv": "one wide table joining every feature table with your metadata",
    "mem_loadings_csv": "the topic model's term-by-theme loadings",
    "lda_loadings_csv": "the LDA topic model's term-by-topic loadings",
    "nmf_loadings_csv": "the NMF topic model's term-by-factor loadings",
    "transformer_embeddings_csv": "row-level transformer embeddings",
    "adapted_encoder_json": "a text encoder adapted to the corpus",
    "text_predictor_json": "a fine-tuned text predictor",
    "word_vectors_model_json": "a trained word-vector model",
    "word_vectors_neighbors_csv": "the word-vector model's nearest-neighbor table",
}


# the kinds of input a run can start from.
#
# "media" is the original path: audio or video that we have to convert and
# transcribe before there's any text to measure. the other two skip all of that
# since the text already exists. nice side effect: every text analyzer is
# already GLOBAL-scoped, so a text preset has no ITEM-scoped steps at all and
# `run_preset` skips input discovery entirely. no root_dir, no ffmpeg, no
# model download.
SOURCES: Dict[str, str] = {
    "media": "audio or video files that need transcribing first",
    "txt_dir": "a folder of documents (.txt, .docx, .doc, .pdf)",
    "csv": "a spreadsheet with a column of text in it",
}

# every argument on a text analyzer that has a say in choosing an input.
#
# the analyzers take exactly one of three input modes -- `analysis_csv`,
# `csv_path` (+ the CSV-gathering args) or `txt_dir` (+ the folder-walking
# args) -- and passing args from two modes at once is how you get a step that
# quietly reads the wrong thing. so `text_binding` below rebuilds this whole
# group from scratch for the chosen source instead of editing it in place;
# that way a leftover key from another mode can't survive.
@dataclass(frozen=True)
class Level:
    """
    One answer to "what should a row of results describe?".

    Attributes
    ----------
    id : str
        Stable identifier, stored in the preset's ``meta``.
    label, help : str
        What the option says, and the sentence under it. The label alone is
        never enough: "one row per speaker" does not say what happened to the
        utterances, and joining a speaker's words before measuring versus
        averaging their per-utterance scores differ by ~35% on vocabulary
        measures. The help says which happened.
    group_by : tuple of str, or None
        The columns to aggregate on. Empty means no aggregation -- the raw row
        is the unit. ``None`` means the user picks the columns, which is only
        the case for a spreadsheet, where the catalog cannot know their names.
    """

    id: str
    label: str
    help: str
    group_by: Optional[Tuple[str, ...]]


#: What a row of results can describe, per source.
#:
#: `media` transcripts carry `source` (the file) and `speaker`, so all three
#: levels are real columns. A spreadsheet's grouping columns belong to the
#: user, hence ``group_by=None``. A folder of .txt files is already one text
#: per file, so it has exactly one level and the question is not worth asking.
#:
#: A fourth media level -- ``("speaker",)``, one row per person across all of
#: their recordings, for a longitudinal design -- is deliberately absent to
#: keep the question to three options. It stays reachable by editing `group_by`
#: on the options screen.
LEVELS: Dict[str, Tuple[Level, ...]] = {
    # most to least granular, so the list reads like zooming out. the default
    # is still the speaker -- `default=` moves the pointer there. order and
    # default are two separate things.
    "media": (
        Level("utterance", "Utterance",
              "One row per turn, measured on its own. Use this to look at "
              "change over the course of a conversation.",
              ()),
        Level("speaker", "Person (speaker)",
              "One row per speaker per recording. Their utterances are "
              "combined into a single measurement.",
              ("source", "speaker")),
        Level("conversation", "Whole conversation (transcript)",
              "One row per recording, with everyone's speech together.",
              ("source",)),
    ),
    "csv": (
        Level("row", "Each spreadsheet row",
              "Every row measured on its own.", ()),
        Level("group", "Each group",
              "Combine rows that share a column -- all of one participant's "
              "answers measured together as a single text.",
              None),
    ),
    "txt_dir": (
        Level("file", "Each file", "One row per document.", ()),
    ),
}

#: The level assumed when nobody chose one. For media this reproduces what
#: Taters has always done, which the shipped presets and their round-trip test
#: depend on.
DEFAULT_LEVEL: Dict[str, str] = {
    "media": "speaker", "csv": "row", "txt_dir": "file",
}


def level_aware(recipe: "Recipe") -> bool:
    """
    Does the analysis level decide this step's grain?

    Derived rather than declared, so a new module needs no wizard work:

    * a text analyzer (``text_input``) measures text, and the level says what
      a piece of text is;
    * a merge (``aggregate: True``) that consumes a capability some text step
      produces is the "measure each, then average" half of a text feature, so
      it has to collapse on the same key the analyzers grouped by.

    One step declares it instead (``takes_level``): the one that reads the
    user's own spreadsheet columns as the measures. It analyzes no text, so
    nothing derives it, but "one row per participant rather than one per
    spreadsheet row" is exactly the question the level asks, and the answer
    decides whether its columns are carried or averaged.

    Everything else keeps the grain it declares. That deliberately excludes the
    audio features: ``acoustics`` groups by ``speaker`` within one file because
    it is item-scoped, and ``gather_whisper_embeddings`` aggregates audio
    segments. "One row per utterance" has no meaning for either -- there is no
    per-utterance WAV to measure -- and overwriting their keys would silently
    change what they average.
    """
    if recipe.text_input or recipe.takes_level:
        return True
    if recipe.with_.get("aggregate") is not True:
        return False
    produced_by_text = {c for r in RECIPES if r.text_input for c in r.produces}
    return bool(recipe.requires & produced_by_text)


def levels_for(source: str) -> Tuple[Level, ...]:
    """Every level that makes sense for `source`."""
    if source not in LEVELS:
        raise KeyError(f"unknown source {source!r}. Known: {', '.join(sorted(LEVELS))}")
    return LEVELS[source]


def level_by_id(source: str, level: Optional[str] = None) -> Level:
    """One level, by id, falling back to the source's default."""
    wanted = level or DEFAULT_LEVEL[source]
    for candidate in levels_for(source):
        if candidate.id == wanted:
            return candidate
    known = ", ".join(c.id for c in levels_for(source))
    raise KeyError(f"unknown level {wanted!r} for {source!r}. Known: {known}")


#: Placeholder for "whatever makes one row of this analyzer's output unique".
#: A merge over CSV or txt_dir input is meant to tidy columns, not to collapse
#: rows -- the analyzers already emit one row per gathered text -- so its
#: ``group_by`` has to be the full identity of a row. What that is depends on
#: answers the catalog cannot see (``mode="separate"`` adds ``source_col``), so
#: the recipe names the intent and :func:`taters.ui.compose._bind_source`
#: resolves it. Grouping on a partial key here silently averages rows that were
#: meant to stay apart, and the output looks perfectly well-formed either way.
TEXT_IDENTITY = "{{text_identity}}"


TEXT_INPUT_KEYS = frozenset({
    "csv_path", "txt_dir", "analysis_csv", "gathered_csv",
    "text_cols", "id_cols", "group_by", "mode", "delimiter", "joiner",
    "recursive", "pattern", "id_from", "include_source_path",
    "pass_through_cols",
})


# the rules themselves live in `taters.helpers.gpu`, not here. the pipeline
# runner needs them too, and we don't want it importing the wizard to get
# them. all this module does is record which answer each step gives.
@dataclass(frozen=True)
class Recipe:
    """
    One pipeline step, plus everything a UI needs to offer it.

    Attributes
    ----------
    id : str
        Stable identifier. This is what the wizard passes to
        :func:`taters.ui.compose.compose`.
    label, help : str
        What the checkbox says, and the one-line explanation under it.
    call : str
        The preset ``call:`` value, e.g. ``"potato.audio.convert_to_wav"``.
    target : str
        ``"module:function"`` for the function ``call`` ultimately reaches.
        Kept as a string so importing it -- which can pull in torch, NeMo, or
        parselmouth -- happens only when a user actually selects this step.
    scope : {"item", "global"}
        ``item`` steps run once per input file; ``global`` steps run once.
    save_as : str
        Name the step's result is bound to for later ``{{templates}}``.
    requires, produces : frozenset[str]
        Capability strings. See the module docstring.
    auto_with : tuple[str, ...]
        Recipe ids to include alongside this one -- used for the gather steps
        that follow a feature step.
    extras : tuple[str, ...]
        pip extras this step needs, e.g. ``("vocalacoustics",)``.
    needs_ffmpeg : bool
        Whether the step shells out to ffmpeg.
    gpu_use : str
        One of :data:`GPU_USE`. Declares what a second worker costs in GPU
        memory, which is the one thing no amount of inspection can work out
        from the outside: ``transcribe`` and ``whisper_embeddings`` have nearly
        identical signatures and differ fourfold in how their VRAM scales.

        Defaults to ``"gpu_model_each"`` -- the cautious answer -- for any step
        that reads the ``device`` variable and has not said otherwise, so
        forgetting to declare it on a new module costs speed rather than a
        crashed run.
    with_ : dict
        The preset ``with:`` block, ``{{templates}}`` already written.
    hidden : tuple[str, ...]
        Parameters never offered, even under "show advanced". These are the
        alternate-input arguments -- ``txt_dir`` and ``analysis_csv`` on the
        text analyzers -- which are mutually exclusive with the ``csv_path``
        the pipeline wires in, so setting one silently detaches the step from
        the run.
    vars : dict[str, dict]
        Contributions to the preset's ``vars:`` block, each entry shaped
        ``{"default": ..., "desc": ...}`` so it can feed ``meta.variables`` too.
    user_facing : bool
        Whether this appears in the feature checklist. Prerequisites and
        gathers are ``False`` -- they get added for you.
    sources : tuple[str, ...]
        Which of :data:`SOURCES` this step makes sense for. Defaults to
        ``("media",)`` because most of the catalog is audio machinery: there is
        nothing to convert to WAV in a folder of essays. The feature checklist
        is filtered by this, so a user who says "I have text files" is never
        offered vocal acoustics.
    text_input : bool
        Whether this step's input binding is rewritten by :func:`text_binding`
        when the source is not media. True for the five text analyzers, which
        read gathered transcripts on the media path but read the user's own
        files or spreadsheet directly otherwise.
    source_with : dict[str, dict]
        Per-source patches merged into ``with_`` last, keyed by source. Covers
        steps that are not themselves text inputs but still have to change --
        the embedding gather groups by speaker on the media path, and there is
        no speaker column when the input was a folder of essays.
    text_help : str
        Replaces ``help`` when the source is not media. The catalog describes
        results "per speaker per file", which is exactly right for a recorded
        conversation and simply untrue for a folder of essays -- and the help
        line under a checkbox is most of what a non-programmer has to go on.
    """

    id: str
    label: str
    help: str
    call: str
    target: str
    scope: str
    save_as: str
    with_: dict
    requires: FrozenSet[str] = frozenset()
    produces: FrozenSet[str] = frozenset()
    auto_with: Tuple[str, ...] = ()
    #: What the options screen calls a setting *in this step*, when the shared
    #: wording in `SETTING_LABELS` would be wrong here. `topic_count_sweep`
    #: needs it: its `engine` is which topic model to fit, not who tags.
    labels: Dict[str, str] = field(default_factory=dict)

    extras: Tuple[str, ...] = ()
    needs_ffmpeg: bool = False
    gpu_use: Optional[str] = None       # None -> inferred; see `resolved_gpu_use`
    hidden: Tuple[str, ...] = ()
    vars: Dict[str, dict] = field(default_factory=dict)
    user_facing: bool = True
    sources: Tuple[str, ...] = ("media",)
    text_input: bool = False
    source_with: Dict[str, dict] = field(default_factory=dict)
    text_help: str = ""
    #: Parameters whose value comes from the user's library rather than from a
    #: typed answer: ``{parameter name: library kind id}``. Declaring one gets
    #: the step a dictionary picker on the options screen, an "everything in
    #: your library" default, and a preflight warning when that library is
    #: empty -- the whole flow, from these two words. Kinds live in
    #: :data:`taters.helpers.library.KINDS`.
    library: Dict[str, str] = field(default_factory=dict)
    #: Default library *entries* for a library parameter, by filename:
    #: ``{parameter name: ("_chars.txt", "stopwords-en.txt")}``. Untouched,
    #: the wizard applies exactly these (instead of ``library``'s usual
    #: whole-folder default), the settings row names them, and the picker
    #: opens with them ticked. Built for stoplists, where "all 22 languages"
    #: is not a default anyone means -- German's "die" would silently strip
    #: English text -- but punctuation + English is what nearly everyone
    #: wants without having to set it up each time. Entries the user deleted
    #: are skipped; with none left, the step runs without.
    library_defaults: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Settings that only matter under some value of another setting:
    #: ``{param: (gate, value)}`` -- shown while the gate *equals* the value
    #: (`stanza_lang` is noise on the menu while the engine is nltk) -- or
    #: ``{param: (gate, "!=", value)}`` -- shown while the gate is *anything
    #: but* the value (the PCA component count means nothing while `pca` is
    #: off). The options screen indents a gated row under its gate and
    #: applies the same rule on the shared section. See `gate_of`.
    param_when: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Extra tags for the preset index, on top of the ones derived from where
    #: the step lives (`audio`/`text` from its module). Declared here so a new
    #: module can describe itself; the alternative was a chain of `if
    #: recipe.id == ...` in the composer, which every new step had to be added
    #: to and none of them could see.
    tags: Tuple[str, ...] = ()
    #: Which wizard stage offers this recipe: "extract" (the feature
    #: checklist) or "analyze" (the optional statistics stage that follows
    #: it). A field rather than a tag because the wizard *branches* on it.
    stage: str = "extract"
    #: Whether this step's `save_as` artifact is a per-text feature CSV,
    #: joinable on the run's identity key -- the mark the stats stage uses to
    #: find its inputs. Deliberately False on corpus-level tables (the n-gram
    #: frequency list) and on the raw document-term matrix, whose thousands
    #: of term columns would swamp every results table; a user can still feed
    #: the DTM in explicitly through the assemble step's options.
    feature_table: bool = False
    #: For an analysis: the kind of outcome column it needs -- "numeric"
    #: (correlations, ridge) or "labels" (group differences, the classifier).
    #: The wizard grays out and asks from this rather than from a list of ids.
    outcome_kind: Optional[str] = None
    #: Whether compose should hand this step the list of selected feature
    #: tables (as `feature_csvs`). True on the stats assemble step and on
    #: the model-scoring step.
    consumes_feature_tables: bool = False
    #: The parameter naming a transformer encoder, if the step takes one:
    #: the options screen offers a list of what is already on this machine
    #: (the library's encoders and predictors, the models in the Hugging
    #: Face cache, the curated names) with a text entry for anything else.
    encoder_param: Optional[str] = None
    #: Whether an empty feature-table list is acceptable. False for the
    #: statistics, which have nothing to analyze without one. True for
    #: model scoring, where whether features are needed at all is a fact
    #: about the chosen model -- a topic model reads text and needs none,
    #: and refusing to compose the pipeline would be refusing the commonest
    #: case.
    feature_tables_optional: bool = False
    #: Whether this step's grain is the analysis level's to set, even though
    #: it is not a text analyzer. See :func:`level_aware`, which is otherwise
    #: derived. Only the step that reads the user's own spreadsheet as the
    #: measures sets this: there is no text to combine, but "one row per
    #: participant rather than per row" is still the user's to choose.
    takes_level: bool = False
    #: Whether compose should bind this step's row key the way it binds the
    #: metadata gather's: the same group keys when rows are combined, the
    #: same id columns otherwise. Every table that has to join on ``text_id``
    #: needs it, and the join is structural rather than lucky only because
    #: the binding is written once. See `compose._bind_source`.
    keys_like_metadata: bool = False

    @property
    def resolved_gpu_use(self) -> str:
        """
        This step's declared GPU behavior, or the cautious guess.

        A step that reads the ``device`` variable and has not declared itself is
        treated as ``"gpu_model_each"`` -- one file at a time. Wrong-but-slow is
        a recoverable mistake; wrong-and-out-of-memory is not, and it fails
        halfway through a batch rather than at the start.
        """
        if self.gpu_use is not None:
            return self.gpu_use
        reads_device = any(
            isinstance(v, str) and "{{var:device}}" in v for v in self.with_.values()
        )
        return "gpu_model_each" if reads_device else "cpu"

    @property
    def worker_cap(self) -> Optional[int]:
        """Most files this step may work on at once, or None for no limit."""
        return worker_cap(self.resolved_gpu_use)

    def to_step(self) -> dict:
        """Render this recipe as a preset step dict."""
        step: dict = {"scope": self.scope, "call": self.call, "save_as": self.save_as}
        step["with"] = dict(self.with_)
        # we write the cap into the preset rather than leaving it implicit, so
        # a preset someone edits by hand (or sends to a colleague) carries the
        # limit with it instead of depending on a catalog they might not have.
        cap = self.worker_cap
        if cap is not None and self.scope == "item":
            step["max_workers"] = cap
        return step


# the vars every preset carries, no matter what got selected.
BASE_VARS: Dict[str, dict] = {
    "device": {
        "default": "auto",
        "desc": "Where to run the models (cpu | cuda | auto).",
    },
    # our one parallelism dial. per-file steps spend it as files-at-once
    # (still capped by each step's max_workers), text steps spend it as
    # reader/scorer processes. `device` says where, `workers` says how much.
    "workers": {
        "default": 0,
        "desc": "How much runs in parallel: files at once for per-file "
                "steps, reader/scorer processes inside the text steps. "
                "0 = automatic: three-quarters of this machine's cores, "
                "leaving the rest for you. Any number is kept between 1 "
                "and the machine's core count.",
    },
    "overwrite_existing": {
        "default": False,
        "desc": "Re-do work whose output files already exist. Off by default, "
                "so an interrupted run resumes instead of starting over.",
    },
}

_TRANSCRIPTS_DIR_VAR = {
    "transcripts_dir": {
        "default": "transcripts",
        "desc": "Where per-file transcripts are written.",
    }
}
_FEATURES_DIR_VAR = {
    "features_dir": {
        "default": "features",
        "desc": "Where feature CSVs are written.",
    }
}
#: One switch for every figure step, and one size. On by default: a word
#: cloud is the quickest read of a result there is, and the steps that draw
#: them cost seconds and change no number.
_WORDCLOUD_VARS: Dict[str, dict] = {
    "wordclouds": {
        "default": True,
        "desc": "Draw word clouds of the results: the features predicting "
                "higher and lower scores, correlating with each outcome, "
                "differing between groups, loading on each component and "
                "theme, and the most frequent terms. Pictures only; no "
                "number changes. Off draws none.",
    },
    "wordcloud_words": {
        "default": 80,
        "desc": "The most words in any one cloud.",
    },
}
_ACOUSTICS_VARS = {
    "acoustics_mode": {"default": "simple",
                       # keep the "One of:" sentence last, after the colon --
                       # a test parses it to check every mode named exists.
                       "desc": "Acoustic feature set. tremor and advanced "
                               "need a Praat tremor script (set tremor_script "
                               "on the step's own settings) or the step "
                               "refuses rather than quietly producing the "
                               "simple set. One of: simple, tremor, or "
                               "advanced."},
    "acoustics_voiced_ms": {"default": 100, "desc": "Ignore voiced runs shorter than this (ms)."},
    "acoustics_preprocess": {"default": True, "desc": "Normalize audio before measuring."},
    "acoustics_target_sr": {"default": 44100, "desc": "Resample rate used for measurement (Hz)."},
    "acoustics_target_dbfs": {"default": -20.0, "desc": "Loudness normalization target (dBFS)."},
    "acoustics_remove_dc": {"default": True, "desc": "Remove DC offset before measuring."},
    "acoustics_pause_top_db": {"default": 30, "desc": "Silence threshold for pause detection (dB below peak)."},
    "acoustics_pause_frame_length": {"default": 2048, "desc": "Frame length for pause detection."},
    "acoustics_pause_hop_length": {"default": 512, "desc": "Hop length for pause detection."},
}

# the id/group columns every text analyzer carries through. defined once,
# because letting these drift out of step with each other is how you end up
# with feature tables that won't join.
_ID_COLS = ["source", "speaker"]

#: One shared variable for every step that prepares tokens the n-gram way.
#: The doc-term matrix scans text against the frequency list's vocabulary;
#: if one lemmatized and the other did not, matching would fail *silently*.
_LEMMATIZE_VAR = {
    # on by default. for the three steps that read this (the frequency list,
    # the document-term matrix and the topic model) the question is always
    # "which words does this text use", and `cats`/`cat` and `was`/`be` are
    # the same word for that purpose. leave it off and a topic model spends
    # part of its vocabulary on inflections of words it already has, and a
    # frequency list splits one word's count across its forms.
    "default": True,
    "desc": "Lemmatize words before counting (WordNet; 'cats' -> 'cat', "
            "'was' -> 'be'), so a word's forms count as one word. Applied "
            "before any stop list.",
}

#: The settings that decide what a topic model's *vocabulary* is, named per
#: model rather than shared.
#:
#: These used to be shared with the document-term-matrix step, and had to be:
#: the model scanned a matrix somebody else built, and a recorded setting that
#: disagreed with that matrix would have been a lie. Each model builds its own
#: matrix now, so nothing can disagree -- and sharing them actively gets in the
#: way, because the whole reason each model has its own matrix is that they
#: want different ones. One study can reasonably want LDA over lemmatized
#: unigrams and NMF over raw bigrams.
#:
#: The tokenizer settings (`engine`, `tokenizer`, `stanza_lang`) stay shared:
#: which toolkit splits and tags the text is a decision about the corpus and
#: the machine, not about any one model, and Stanza's download is shared anyway.
def _vocab_vars(prefix: str) -> Dict[str, dict]:
    return {
        f"{prefix}_lemmatize": dict(_LEMMATIZE_VAR),
        f"{prefix}_pos_tagged": dict(_POS_TAGGED_VAR),
        f"{prefix}_keep_punctuation": dict(_KEEP_PUNCTUATION_VAR),
        f"{prefix}_ngram_n": {
            "default": 1,
            "desc": "Longest phrase to consider as a term: 1 for single "
                    "words, 2 to let two-word phrases compete with them. "
                    "Per model, so one can use phrases and another not.",
        },
    }


def _vocab_with(prefix: str) -> Dict[str, object]:
    return {
        "lemmatize": f"{{{{var:{prefix}_lemmatize}}}}",
        "pos_tagged": f"{{{{var:{prefix}_pos_tagged}}}}",
        "keep_punctuation": f"{{{{var:{prefix}_keep_punctuation}}}}",
        "ngram_n": f"{{{{var:{prefix}_ngram_n}}}}",
    }


#: What an untouched pipeline applies, and what the stoplist picker opens
#: with: punctuation characters and English. Never all 22 languages -- see
#: the recipe's comment.
_DEFAULT_STOPLISTS = ("_chars.txt", "_chars_extended.txt", "stopwords-en.txt")

#: Shared for the same reason as lemmatize: the DTM scans text against the
#: frequency list's vocabulary, and a tagged vocabulary is unfindable in an
#: untagged stream (the analyzers refuse the mismatch, but one shared answer
#: means it cannot arise from the wizard at all).
_POS_TAGGED_VAR = {
    "default": False,
    "desc": "Treat word+part-of-speech as the unit: the verb 'felt' and the "
            "noun 'felt' become separate terms, with a 'pos' column naming "
            "each term's tag.",
}
_KEEP_PUNCTUATION_VAR = {
    # off by default: a frequency list whose top term is "." and a topic
    # model spending a column on "," are counting sentence boundaries, not
    # words. shared by the same three steps as lemmatize, for the same reason
    # -- scan a vocabulary with a different rule from the one that built it
    # and it fails silently.
    "default": False,
    "desc": "Count punctuation (and emoticons) as terms. Off: only tokens "
            "with a letter or digit are counted.",
}

#: What the options screen calls a setting, when the parameter's own name is a
#: programmer's word for it.
#:
#: The API keeps its names and the wizard gets these, and that split is the
#: right way round rather than a compromise: somebody reading ``engine`` in a
#: docstring is writing code and has the signature in front of them, and
#: somebody meeting it on the options screen is not.
#:
#: Renaming the parameters instead would break every saved pipeline that names
#: one -- loudly for a ``with:`` key, and *silently* for a shared variable,
#: where the old name would simply go unread and somebody's setting be lost.
#:
#: Only names that genuinely do not say what they are belong here. `encoding`,
#: `workers` and `rounding` already mean what they look like.
SETTING_LABELS: Dict[str, str] = {
    "engine": "tagging engine",
    # the vocabulary is built in two passes and the parameter names do not say
    # so: `min_freq` decides which words are counted at all, `vocab_min_freq`
    # decides how many of the survivors the model actually sees. Four rows
    # apart on screen and near-identically named, they read as a typo.
    "min_freq": "drop words used fewer times than",
    "min_obs_pct": "drop words found in fewer texts than (%)",
    "min_token_count": "skip texts shorter than",
    "min_npmi": "phrase strength cutoff",
    "ngram_n": "longest phrase to consider",
    "vocab_rule": "how to pick the model's vocabulary",
    "vocab_top_n": "how many words the model gets",
    "vocab_rank_by": "rank the vocabulary by",
    "vocab_min_freq": "keep words used at least this many times",
    "vocab_min_obs_pct": "keep words found in at least this % of texts",
    "matrix_rounding": "decimal places in the matrix",
    "rounding": "decimal places",
    "rotation": "rotate the themes (varimax)",
    "n_components": "how many themes",
    "stoplist_paths": "stop word lists",
    "tokenizer": "word splitter",
    "stanza_lang": "stanza language",
    "joiner": "text joiner",
    "pattern": "which files to read",
    "weighting": "how cells are counted",
    "pca": "reduce to components",
    "zscore": "standardize predictors",
    "stratify": "balance the folds",
    "pooling": "how tokens become one vector",
    "precision": "number precision",
    "layers": "which encoder layers",
}


#: Shared for the same reason as lemmatize: every text-preparing step in a
#: pipeline must read text the same way, or a vocabulary built by one step is
#: silently unfindable by another.
def _k_sweep_vars(prefix: str) -> Dict[str, dict]:
    """The settings the two scoring rules need, one copy per model.

    Per model rather than shared, for the same reason each builds its own
    matrix: a sweep scored over one model's vocabulary says nothing about
    another's, and sharing the setting would quietly imply it did.
    """
    return {
        f"{prefix}_k_values": {
            "default": ",".join(str(k) for k in _topics_default_k()),
            "desc": "Which topic counts to try, when the count is being "
                    "chosen by fitting. Counts this corpus is too small to "
                    "support are skipped and named.",
        },
        f"{prefix}_coherence_metric": {
            "default": "npmi",
            "desc": "Which coherence to score with. npmi is bounded, which "
                    "is what lets it be balanced against exclusivity; umass "
                    "is the older measure and cannot be balanced.",
        },
    }


def _topics_default_k():
    from ..text._topics import DEFAULT_K_VALUES
    return DEFAULT_K_VALUES


#: Shown only while a topic count is being chosen by fitting, on all three
#: models. Two conditions where the model has more than one rule.
_K_SELECTION_DESC = (
    "How 0 decides the number of topics. coherence: fit at each count and "
    "keep the one whose topics' words most often turn up in the same "
    "documents. coherence_exclusivity: the harmonic mean of that and "
    "exclusivity -- whether those are this topic's words rather than "
    "everybody's -- because coherence alone prefers a few topics made of "
    "common words. Both cost one fit per count."
)


#: MEM is the one model with four rules, two of which read eigenvalues the
#: fit already has and two of which fit at every candidate count. The sweep's
#: own settings only mean anything under the latter pair.
#: The vocabulary is cut one of three ways, and each way reads one setting.
#: Showing all three settings at once is most of why this screen is confusing:
#: two of them are inert, and the inert ones are the ones whose names collide
#: with the frequency-list thresholds higher up.
_VOCAB_PARAM_WHEN = {
    "vocab_top_n": ("vocab_rule", "top_n"),
    "vocab_rank_by": ("vocab_rule", "top_n"),
    "vocab_min_freq": ("vocab_rule", "min_freq"),
    "vocab_min_obs_pct": ("vocab_rule", "min_obs_pct"),
}


_MEM_SWEEPING = [("n_components", 0),
                 ("k_selection", "!=", "kaiser"),
                 ("k_selection", "!=", "parallel")]


_ENGINE_VAR = {
    "default": "nltk",
    "desc": "Who tags and lemmatizes: nltk (fast, English) or stanza "
            "(neural, more accurate, multilingual, GPU-optional).",
}
_TOKENIZER_VAR = {
    "default": "potts",
    "desc": "Who splits text into tokens: potts (social-media-aware, keeps "
            "emoticons and URLs whole) or stanza (stanza engine only).",
}
_STANZA_LANG_VAR = {
    "default": "en",
    "desc": "Language for the stanza engine; that language's model downloads "
            "once on first use (can be a few hundred MB).",
}

#: The vars and with_ entries every engine-aware text step carries.
_ENGINE_VARS = {
    "engine": _ENGINE_VAR,
    "tokenizer": _TOKENIZER_VAR,
    "stanza_lang": _STANZA_LANG_VAR,
}
_ENGINE_WITH = {
    "engine": "{{var:engine}}",
    "tokenizer": "{{var:tokenizer}}",
    "stanza_lang": "{{var:stanza_lang}}",
    "device": "{{var:device}}",
}

#: Which settings only matter under the stanza engine -- the wizard hides
#: their rows while the engine is nltk (see Recipe.param_when).
_ENGINE_PARAM_WHEN = {
    "stanza_lang": ("engine", "stanza"),
    "device": ("engine", "stanza"),
}

#: The PCA settings mean nothing while `pca` is off: how many components,
#: whether to rotate them, what to do with a mostly-empty feature.
_STATS_PCA_PARAM_WHEN = {
    "pca_components": ("pca", "!=", "off"),
    "pca_rotation": ("pca", "!=", "off"),
    "pca_retain": ("pca", "!=", "off"),
}
_STATS_PCA_PARAM_WHEN_WITH_MISSING = {
    **_STATS_PCA_PARAM_WHEN,
    "pca_max_missing": ("pca", "!=", "off"),
}


def gate_of(recipe: "Recipe", param: str) -> Optional[Tuple[str, str, str]]:
    """
    A gated setting's rule as ``(gate, op, value)``, or None when ungated.

    Where a setting has more than one condition this is the *first* one --
    what the row nests under on screen. :func:`gates_of` is the whole
    predicate.

    Two spellings are accepted -- ``(gate, value)`` means ``==`` -- and
    anything else raises, so a typo in a declaration is a test failure
    rather than a row that never shows.
    """
    gates = gates_of(recipe, param)
    return gates[0] if gates else None


def gates_of(recipe: "Recipe",
             param: str) -> Optional[Tuple[Tuple[str, str, str], ...]]:
    """
    Every condition on a setting, all of which must hold for it to show.

    A list of rules rather than one, because some settings depend on two
    answers: the Kaiser cutoff is worth asking about only when the theme
    count is being chosen automatically *and* the rule doing the choosing is
    Kaiser. One rule stays the common case and is written as one.
    """
    raw = recipe.param_when.get(param)
    if raw is None:
        return None
    rules = raw if (raw and isinstance(raw[0], (tuple, list))) else [raw]
    out = []
    for rule in rules:
        if len(rule) == 2:
            out.append((str(rule[0]), "==", str(rule[1])))
        elif len(rule) == 3 and rule[1] in ("==", "!="):
            out.append((str(rule[0]), str(rule[1]), str(rule[2])))
        else:
            raise ValueError(
                f"{recipe.id}: param_when[{param!r}] must be (gate, value) or "
                f"(gate, '!='|'==', value), or a list of those, got {raw!r}")
    return tuple(out)


def gate_holds(gate: Tuple[str, str, str], current: object) -> bool:
    """
    Whether a gated setting should be shown, given its gate's live value.

    Fails **open**: a gate whose value is unknown (`None`, or the wizard's
    EMPTY sentinel) shows the row, because a hidden setting that should be
    visible is the one mistake this screen must never make. The comparison
    is on the value's text, as the recipes spell it.
    """
    if current is None or type(current).__name__ == "_Empty":
        return True
    _gate, op, value = gate
    return (str(current) == value) if op == "==" else (str(current) != value)

#: Shared for the same reason as lemmatize, one level up: these decide the
#: *shape* of the document-term matrix -- which terms become columns and what
#: the cells hold -- and the MEM topic model re-derives that shape to verify
#: the matrix and to record it in its reusable model. Two steps with two
#: different answers would refuse at run time; one shared answer means the
#: disagreement cannot be composed at all.
_MATRIX_SHAPE_VARS = {
    "weighting": {
        "default": "count",
        "desc": "What each matrix cell holds: count (raw matches), binary "
                "(0/1), relfreq (matches over the document's tokens), or "
                "tfidf (matches times the term's rarity).",
    },
    # one rule, then its number. this used to be three thresholds applied at
    # once plus a fourth setting that only affected one of them, and working
    # out what a corpus would be left with meant intersecting all of them in
    # your head -- with nothing on screen saying they combined at all. now
    # exactly one rule applies and we only read its own setting.
    "vocab_rule": {
        "default": "top_n",
        "desc": "Which single rule decides the matrix vocabulary: top_n "
                "(the N highest-ranked terms), min_obs_pct (terms appearing "
                "in at least X% of documents), or min_freq (terms used at "
                "least X times). Only the setting belonging to the rule you "
                "pick is read.",
    },
    "vocab_top_n": {
        # 500, not 250. the matrix is the whole vocabulary a topic model or
        # an open-vocabulary prediction gets to work with, and at 250 the tail
        # of a corpus's ordinary words was already gone (someone went looking
        # for a word by name and it wasn't there).
        "default": 500,
        "desc": "How many terms to keep, when the rule is top_n (ties at "
                "the cut-off stay). 0 = no limit.",
    },
    "vocab_min_freq": {
        "default": 0,
        "desc": "How many times a term must be used in total, when the "
                "rule is min_freq.",
    },
    "vocab_min_obs_pct": {
        "default": 0,
        "desc": "What percent of documents a term must appear in, when the "
                "rule is min_obs_pct.",
    },
    "vocab_rank_by": {
        "default": "frequency",
        "desc": "What top_n ranks by: frequency (raw count) or obs_pct "
                "(share of documents). Ignored by the other rules.",
    },
}
_MATRIX_SHAPE_WITH = {name: f"{{{{var:{name}}}}}" for name in _MATRIX_SHAPE_VARS}

# the stuff a merge step must not let anyone rearrange. `feature_gather` gets
# told how to fold a table down (which keys to group on, whether to aggregate
# at all, which columns to drop first) and the wizard works those out from what
# the step in front of it produced. they're wiring in every sense except that
# they're literals rather than `{{templates}}`, so `is_wired` can't see them.
#
# getting one wrong doesn't fail, which is the scary part. group row-level
# embeddings by a key that's unique per row and you get a file of the right
# shape, with the right columns and real numbers in them, that has aggregated
# nothing. we've shipped that bug once already.
_MERGE_SHAPE = ("root_dir", "pattern", "recursive", "aggregate", "group_by",
                "per_file", "stats", "add_source_path", "exclude_cols")

# vars that come from the source itself rather than from any one step. we keep
# the input path in a var (rather than baking it into five separate steps) so
# that someone can re-point a finished preset at next term's essays with
# `--var input_dir=...` instead of editing YAML.
SOURCE_VARS: Dict[str, Dict[str, dict]] = {
    "media": {},
    "txt_dir": {
        "input_dir": {
            "default": ".",
            "desc": "Folder of documents (.txt, .docx, .doc, .pdf) to analyze.",
        },
        "txt_pattern": {
            "default": "*.txt;*.docx;*.doc;*.pdf",
            "desc": "Which files to read, as globs separated by ';'.",
        },
    },
    "csv": {
        "input_csv": {
            "default": "",
            "desc": "Spreadsheet holding the text to analyze.",
        },
        # a variable rather than a hardcoded comma: the file picker offers
        # .tsv, and reading a tab-separated file with a comma reader gets you
        # one column whose name is the whole header line.
        "csv_delimiter": {
            "default": ",",
            "desc": "Column separator in that file: ',' for CSV, tab for TSV.",
        },
    },
}


def text_binding(
    source: str,
    *,
    text_cols: Sequence[str] = ("text",),
    id_cols: Sequence[str] = (),
    pass_through: bool = False,
    text_mode: str = "concat",
    group_by: Sequence[str] = (),
) -> dict:
    """
    Build the input arguments a text analyzer needs for a given source.

    The analyzers take exactly one of three input modes and the arguments for
    the other two are silently ignored, so this returns the *complete* group
    rather than a patch. Callers strip :data:`TEXT_INPUT_KEYS` first and merge
    this in, which makes it impossible for a leftover ``csv_path`` to sit
    alongside a fresh ``txt_dir``.

    Parameters
    ----------
    source : {"media", "txt_dir", "csv"}
        ``"media"`` returns ``{}``: on that path the transcript wiring already
        in the recipe is correct and must not be touched.
    text_cols, id_cols : sequence of str
        Only meaningful for ``"csv"`` -- which columns hold the text, and which
        identify the row. ``id_cols`` composes ``text_id``; with none, the
        gatherer falls back to ``row_<n>``.
    text_mode : {"concat", "separate"}
        What to do when ``text_cols`` names more than one column. ``concat``
        joins them into one piece of text per row; ``separate`` measures each
        column on its own, giving one output row per column per input row with
        a ``source_col`` saying which is which. Meaningless for a single
        column, where the two are identical.
    pass_through : bool
        Whether this step wants ``pass_through_cols`` (sentence embeddings
        does; the others do not).

    Returns
    -------
    dict
        Arguments to merge into a step's ``with:`` block.
    """
    if source == "media":
        return {}

    if source == "txt_dir":
        # the folder walker gets text_id from the filename, so there are no id
        # columns to choose and nothing to group by: one text per file.
        #
        # gathered_csv keeps the intermediate table inside the run's own
        # folder. left to itself the analyzer writes it beside the *source*,
        # so analyzing a folder in someone's Downloads would leave a stray
        # file in their Downloads.
        binding: dict = {
            "txt_dir": "{{var:input_dir}}",
            "gathered_csv": "gathered/texts.csv",
            "pattern": "{{var:txt_pattern}}",
            "recursive": True,
            "id_from": "stem",
            "include_source_path": True,
        }
        if pass_through:
            binding["pass_through_cols"] = []
        return binding

    if source == "csv":
        # one feature row per spreadsheet row unless the user asked us to
        # combine rows. `group_by` and `id_cols` are either/or, not both: the
        # gatherer composes `text_id` from `id_cols` *only when not grouping*,
        # so if we sent both, one of them would get silently ignored.
        binding = {
            "csv_path": "{{var:input_csv}}",
            "gathered_csv": "gathered/texts.csv",
            "text_cols": list(text_cols),
            "mode": text_mode,
            "delimiter": "{{var:csv_delimiter}}",
        }
        carried = list(group_by) if group_by else list(id_cols)
        if group_by:
            binding["group_by"] = list(group_by)
        elif id_cols:
            # `id_cols` compose `text_id` -- for EVERY text step, the
            # pass-through one included. the sentence-embedding step used to
            # keep a synthetic `row_<n>` id instead, to protect its merge
            # from a repeating id column; but the metadata gather and every
            # other analyzer build theirs from the id columns, so the
            # embeddings never joined anything. this bit us once: an
            # embeddings + classification run died at the join with "no key
            # value appears in every input" (938 rows, 0 matched). so now it's
            # one identity everywhere, and we catch a repeating id before the
            # run instead -- the wizard checks the column, and the join
            # refuses duplicates.
            binding["id_cols"] = list(id_cols)
        if pass_through:
            # `source_col` is what tells a headline row from a body row under
            # `mode="separate"`. the analyzer only writes `text_id` plus
            # whatever we tell it to pass through, so without this the two
            # become indistinguishable rows sharing an id -- and the merge
            # downstream then averages them into one, undoing the very
            # separation the user asked for.
            binding["pass_through_cols"] = (
                carried + ["source_col"] if text_mode == "separate" else carried
            )
        return binding

    raise KeyError(f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}")



# vars shared by the statistics steps. same discipline as the matrix-shape
# vars: one named answer, referenced by template, so the metadata gather, the
# assemble step and the analyses can never disagree about which columns the
# statistics run on.
_STATS_DIR_VAR: Dict[str, dict] = {
    "stats_dir": {
        "default": "stats_results",
        "desc": "Folder (inside the pipeline's folder) for the statistical "
                "results: tidy CSVs plus a plain-English report.md.",
    },
}

_STATS_META_VARS: Dict[str, dict] = {
    "stats_meta_carry": {
        "default": [],
        "desc": "Spreadsheet columns copied through to the metadata table "
                "unchanged (group labels, categorical covariates). When rows "
                "are combined, a carried column is kept only where every "
                "combined row agrees.",
    },
    "stats_meta_agg": {
        "default": {},
        "desc": "Numeric spreadsheet columns summarized per combined row, as "
                "{column: statistic}. Outcomes become <column>_mean (with a "
                "<column>_n count) when rows are combined.",
    },
}

# referenced by both the step that writes the merged table and the step that
# tidies up after the analyses. that's what gets the wizard to show it as a
# shared setting -- the vocabulary this UI already has for "global".
_STATS_KEEP_VAR: Dict[str, dict] = {
    "stats_keep_table": {
        "default": True,
        "desc": "Keep the merged analysis table (features joined to your "
                "metadata) after the run. It is the dataset the statistics "
                "were computed on -- the file to open in R or hand to a "
                "colleague -- but on a wide feature set it is the largest "
                "thing the run produces.",
    },
}

_STATS_FILTER_VAR: Dict[str, dict] = {
    "stats_filters": {
        "default": [],
        "desc": "Row filters applied before any statistics, as [column, "
                "operator, value] triples a row must all satisfy to stay -- "
                "e.g. [word_count, >=, 25]. Blank cells fail every filter.",
    },
}

_STATS_GROUP_VAR: Dict[str, dict] = {
    "stats_group_col": {
        "default": "",
        "desc": "The metadata column naming each row's group, for the "
                "group-differences step.",
    },
}

_STATS_OUTCOME_VAR: Dict[str, dict] = {
    "stats_outcome_cols": {
        "default": [],
        "desc": "The metadata column(s) holding numeric outcomes, for the "
                "correlation and prediction steps. When rows are combined, "
                "use the post-combination names (<column>_mean).",
    },
}

# shared by every analysis that corrects for multiple comparisons, so we make
# the choice once and it can't differ between two tables in one report.
_STATS_ADJUST_VAR: Dict[str, dict] = {
    "stats_p_adjust": {
        "default": "fdr_bh",
        "desc": "How to adjust p-values for the number of tests: fdr_bh "
                "(Benjamini-Hochberg), fdr_by (valid under any dependence "
                "between features), holm, bonferroni, or none.",
    },
}

# set when a spreadsheet's text columns were measured one at a time. each
# analysis then runs once per column, because a participant contributes one
# row per column and pooling them would count one person's answers as
# several independent observations (which they aren't).
_STATS_SPLIT_VAR: Dict[str, dict] = {
    "stats_split_col": {
        "default": "",
        "desc": "Run every analysis once per value of this column, keeping "
                "the results apart. Set to 'source_col' when each text "
                "column was measured separately; empty otherwise.",
    },
}

# shared by every analysis, so "controlling for age and gender" means the
# same sample and the same adjustment in the group comparison, the
# correlations and the prediction. three different answers to one question
# would be worse than not offering it at all.
_STATS_CONTROL_VAR: Dict[str, dict] = {
    "stats_control_cols": {
        "default": [],
        "desc": "Metadata column(s) to hold constant. Group comparisons "
                "become ANCOVA with adjusted means, correlations become "
                "partial correlations, and prediction fits the controls "
                "alone as well so you can see what the language added over "
                "them. Empty for none.",
    },
    "stats_categorical_controls": {
        "default": [],
        "desc": "Which of the control columns are categories rather than "
                "measurements (gender, condition, site). These are dummy "
                "coded against their alphabetically first level. Columns "
                "holding words are treated as categorical whether or not "
                "they are listed here; listing matters for a category that "
                "happens to be numbered, like a 1/2 gender code.",
    },
}

# shared, so "reduce the features?" gets answered once for the run -- and then
# tweaked per analysis on the options screen if the answer differs between
# them, which it often does: raw variables read better in a correlation
# table, while a ridge over hundreds of collinear measures is exactly what
# components are for. a shared default plus a per-step override is how every
# other setting in this app behaves anyway.
_STATS_PCA_VARS: Dict[str, dict] = {
    "stats_pca": {
        "default": "off",
        "desc": "Analyze components instead of the raw measures: off, all "
                "(reduce every feature set), or the name(s) of the feature "
                "sets to reduce. Naming some is the useful case -- a hundred "
                "dictionary categories are worth reducing and eight "
                "readability indices are not.",
    },
    "stats_pca_components": {
        "default": 0,
        "desc": "How many components to keep when reducing. 0 decides by "
                "the rule below; a number is honored as asked. Either "
                "way, look at the eigenvalues.",
    },
    "stats_pca_retain": {
        "default": "parallel",
        "desc": "How 0 decides the number of components. parallel: "
                "parallel analysis, keep a component while its eigenvalue "
                "beats what random data of the same size would give -- the "
                "recommended rule, and memory-light. kaiser: keep every "
                "eigenvalue above 1, the older rule of thumb, which keeps "
                "most of them on a wide table.",
    },
    "stats_pca_rotation": {
        "default": True,
        "desc": "Rotate the components (varimax) so each loads on a small "
                "cluster of features and can be given a name.",
    },
}

#: One answer for what to do with the count columns every step writes beside
#: its measures -- the matrix's token_count, the dictionaries' WC, a
#: readability step's raw sentence and syllable counts. Read by the assemble
#: step, which every analysis is downstream of.
_STATS_BOOKKEEPING_VAR: Dict[str, dict] = {
    "stats_bookkeeping": {
        "default": "aside",
        "desc": "Count columns the steps write beside their measures -- "
                "token_count, word counts, sentence counts. 'aside' (the "
                "default): joined into the table for filters and inspection, "
                "not analyzed. 'features': analyzed like any other measure, "
                "for a design where length is a predictor.",
    },
    "stats_bookkeeping_cols": {
        "default": [],
        "desc": "Extra count columns to set aside, for a table whose record "
                "does not declare them: the row counts a grouped spreadsheet "
                "gather writes beside each averaged column.",
    },
}

#: Cross-validation, once for both prediction steps: the ridge and the
#: classifier used to take their fold counts as separate per-step settings.
_STATS_CV_VARS: Dict[str, dict] = {
    "stats_n_folds": {
        "default": 5,
        "desc": "How many cross-validation folds. Every reported score is "
                "out-of-fold: each row is predicted by a model that never "
                "saw it. Five is the usual answer; ten for a small sample.",
    },
    "stats_stratify": {
        "default": True,
        "desc": "Make the folds alike -- the same spread of the outcome, or "
                "the same mix of classes, in every fold. Off: the folds are "
                "a random shuffle, which can put most of the high scores or "
                "all of a small class in one fold.",
    },
}

_STATS_CLASS_VAR: Dict[str, dict] = {
    "stats_class_cols": {
        "default": [],
        "desc": "The metadata column(s) holding a category to predict -- a "
                "diagnosis, a condition, male/female -- for the "
                "classification step. Labels are taken as written. Kept "
                "apart from stats_outcome_cols because predicting a class "
                "and predicting a number are different questions with "
                "different answers, and each step refuses the other's.",
    },
}

#: How tables analyzed together are combined by the two fitters.
_STATS_COMBOS_VAR: Dict[str, dict] = {
    "stats_set_combos": {
        "default": "subsets",
        "desc": "When the feature tables are analyzed together, which "
                "models to fit: subsets -- every combination of the tables, "
                "each alone, every pair and so on up to all of them, so you "
                "can see what each table adds (up to five tables); "
                "each_and_all -- each alone and all together; none -- all "
                "together only.",
    },
}

_STATS_SETS_VAR: Dict[str, dict] = {
    "stats_feature_sets": {
        "default": None,
        "desc": "How to batch the features: empty for one combined analysis "
                "over everything, per_table to repeat the analysis for each "
                "feature table and compare them.",
    },
}


#: What every text analyzer's step declares: the input it reads is the
#: pipeline's gathered table, and the source stage decides the rest. Eight
#: recipes spelled these out identically; a ninth got one key wrong.
_TEXT_STEP: Dict[str, object] = {
    "hidden": ("txt_dir", "analysis_csv"),
    "sources": ("media", "txt_dir", "csv"),
    "text_input": True,
}
_TEXT_INPUT_WITH: Dict[str, object] = {
    "csv_path": "{{transcripts_all}}",
    "text_cols": ["text"],
    "id_cols": list(_ID_COLS),
    "group_by": list(_ID_COLS),
    "mode": "concat",
    "delimiter": ",",
    "encoding": "utf-8-sig",
    "overwrite_existing": "{{var:overwrite_existing}}",
    "workers": "{{var:workers}}",
}

#: What every analysis step reads and where it writes: the assembled table,
#: the shared feature-set / split / PCA / control variables, the results
#: folder. The four analyses repeated this ten-key block.
_STATS_ANALYSIS_WITH: Dict[str, object] = {
    "table_csv": "{{analysis_table}}",
    "feature_sets": "{{var:stats_feature_sets}}",
    "split_col": "{{var:stats_split_col}}",
    "pca": "{{var:stats_pca}}",
    "pca_components": "{{var:stats_pca_components}}",
    "pca_rotation": "{{var:stats_pca_rotation}}",
    "pca_retain": "{{var:stats_pca_retain}}",
    "control_cols": "{{var:stats_control_cols}}",
    "categorical_controls": "{{var:stats_categorical_controls}}",
    "out_dir": "{{var:stats_dir}}",
    "overwrite_existing": "{{var:overwrite_existing}}",
}


RECIPES: List[Recipe] = [
    # ---------------------------------------------------------------- inputs
    Recipe(
        id="convert_to_wav",
        gpu_use="cpu",
        label="Convert media to WAV",
        help="Standardize every input to PCM WAV. Everything else needs this.",
        call="potato.audio.convert_to_wav",
        target="taters.audio.convert_to_wav:convert_audio_to_wav",
        scope="item",
        save_as="wav",
        produces=frozenset({"wav"}),
        needs_ffmpeg=True,
        user_facing=False,
        with_={
            "input_path": "{{input}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    ),

    # ------------------------------------------------------------ transcripts
    Recipe(
        id="transcribe",
        tags=("transcription",),
        # faster-whisper caches one CTranslate2 model and every worker shares
        # it. we measured VRAM and it sat flat at 272 MiB from 1 worker to 4.
        gpu_use="gpu_one_model",
        label="Transcript (one speaker)",
        help="faster-whisper on the whole file. Works with the base install. "
             "Best for lectures, voice memos, dictation, single-mic interviews.",
        call="potato.audio.transcribe_with_whisper",
        target="taters.audio.transcribe_with_whisper:transcribe_with_whisper",
        scope="item",
        save_as="diar",
        requires=frozenset({"wav"}),
        produces=frozenset({"transcript_csv"}),
        vars={
            **_TRANSCRIPTS_DIR_VAR,
            "whisper_model": {
                "default": "base.en",
                "desc": "faster-whisper model size. Use a name without '.en' "
                        "(e.g. 'base', 'small') for anything but English.",
            },
            "language": {"default": None, "desc": "Force a language code (e.g. en). Null auto-detects."},
            "translate": {
                "default": False,
                "desc": "Write the transcript in English whatever language is "
                        "spoken. Needs a multilingual model: a name without "
                        "'.en'. Only the one-speaker transcript can do this; "
                        "the diarizer aligns words to the audio, which a "
                        "translation cannot be.",
            },
            "speaker_label": {
                "default": "Speaker 0",
                "desc": "Value written to the speaker column. There is only one "
                        "speaker, but the column is kept so later steps match "
                        "the diarized pipeline.",
            },
        },
        with_={
            "audio_path": "{{wav}}",
            "out_dir": "{{var:transcripts_dir}}",
            "whisper_model": "{{var:whisper_model}}",
            "language": "{{var:language}}",
            "translate": "{{var:translate}}",
            "device": "{{var:device}}",
            "speaker_label": "{{var:speaker_label}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    ),
    Recipe(
        id="diarize",
        tags=("transcription", "diarization"),
        # this one runs the vendored diarizer as a subprocess, which loads
        # NeMo, Demucs and its own aligner for every single file.
        gpu_use="gpu_model_each",
        label="Transcript with speaker labels (several speakers)",
        help="Transcribe and work out who spoke when. Needs the diarization "
             "extra, which is a large install (NeMo) plus three git packages.",
        call="potato.audio.diarize_with_thirdparty",
        target="taters.audio.diarizer.whisper_diar_wrapper:run_whisper_diarization_repo",
        scope="item",
        save_as="diar",
        requires=frozenset({"wav"}),
        produces=frozenset({"transcript_csv"}),
        extras=("diarization",),
        vars={
            **_TRANSCRIPTS_DIR_VAR,
            "whisper_model": {
                "default": "base",
                "desc": "Whisper model size used for the transcription half.",
            },
            "num_speakers": {
                "default": None,
                "desc": "Fix the speaker count if you know it. Null lets the "
                        "clustering decide, which is usually fine.",
            },
        },
        with_={
            "audio_path": "{{wav}}",
            "out_dir": "{{var:transcripts_dir}}",
            "whisper_model": "{{var:whisper_model}}",
            "device": "{{var:device}}",
            "batch_size": 0,
            "use_custom": True,
            "keep_temp": False,
            "num_speakers": "{{var:num_speakers}}",
        },
    ),

    # --------------------------------------------------------- audio features
    Recipe(
        id="split_by_speaker",
        gpu_use="cpu",
        label="One audio file per speaker",
        help="Cut the WAV into a separate track per speaker, using the "
             "transcript's timings. Only interesting when there is more than one.",
        call="potato.audio.split_wav_by_speaker",
        target="taters.audio.split_wav_by_speaker:make_speaker_wavs_from_csv",
        scope="item",
        save_as="speaker_wav",
        requires=frozenset({"wav", "transcript_csv"}),
        produces=frozenset({"speaker_wavs"}),
        with_={
            "source_wav": "{{wav}}",
            "transcript_csv_path": "{{pick:diar.raw_files.csv}}",
            "time_unit": "ms",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    ),
    Recipe(
        id="acoustics",
        gpu_use="cpu",
        label="Acoustic measures (pitch, loudness, pauses)",
        help="Voice measurements per speaker, guided by the transcript timings.",
        call="potato.audio.analyze_vocal_acoustics",
        target="taters.audio.analyze_vocal_acoustics:analyze_acoustics",
        scope="item",
        save_as="acoustics",
        requires=frozenset({"wav", "transcript_csv"}),
        produces=frozenset({"acoustics_csv"}),
        auto_with=("gather_acoustics",),
        extras=("vocalacoustics",),
        vars={**_FEATURES_DIR_VAR, **_ACOUSTICS_VARS},
        with_={
            "wav_path": "{{wav}}",
            "transcript_csv": "{{pick:diar.raw_files.csv}}",
            "time_unit": "ms",
            "group_by": ["speaker"],
            "extra_id_cols": list(_ID_COLS),
            "out_dir": "{{var:features_dir}}/acoustics",
            "include_framewise": False,
            "mode": "{{var:acoustics_mode}}",
            "summarize_on_voiced_segments_ms": "{{var:acoustics_voiced_ms}}",
            "preprocess": "{{var:acoustics_preprocess}}",
            "target_sr": "{{var:acoustics_target_sr}}",
            "target_dbfs": "{{var:acoustics_target_dbfs}}",
            "remove_dc": "{{var:acoustics_remove_dc}}",
            "pause_top_db": "{{var:acoustics_pause_top_db}}",
            "pause_frame_length": "{{var:acoustics_pause_frame_length}}",
            "pause_hop_length": "{{var:acoustics_pause_hop_length}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    ),
    Recipe(
        id="whisper_embeddings",
        # `run_in_subprocess=True` by default, so each file gets its own
        # process and its own copy of the encoder. we measured VRAM at
        # 240 -> 479 -> 893 MiB for 1, 2 and 4 workers -- and it got slower
        # at every step up, too.
        gpu_use="gpu_model_each",
        label="Whisper audio embeddings",
        help="A numeric fingerprint of how each stretch of speech sounds. "
             "Useful as model input; not directly interpretable.",
        call="potato.audio.extract_whisper_embeddings",
        target="taters.audio.extract_whisper_embeddings:extract_whisper_embeddings",
        scope="item",
        save_as="whisper_feats",
        requires=frozenset({"wav", "transcript_csv"}),
        produces=frozenset({"whisper_embeddings_csv"}),
        auto_with=("gather_whisper_embeddings",),
        vars={**_FEATURES_DIR_VAR},
        with_={
            "source_wav": "{{wav}}",
            "transcript_csv": "{{pick:diar.raw_files.csv}}",
            # spelled out because the paired gather reads
            # {{var:features_dir}}/whisper-embeddings. left to its own default
            # the extractor wrote to ./features/... while a redirected gather
            # scanned somewhere else, and the run died on "No files matched".
            "output_dir": "{{var:features_dir}}/whisper-embeddings",
            "time_unit": "ms",
            "model_name": "{{var:whisper_model}}",
            "device": "{{var:device}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    ),

    # -------------------------------------------------------------- gathering
    Recipe(
        id="gather_acoustics",
        feature_table=True,
        gpu_use="cpu",
        hidden=_MERGE_SHAPE,
        label="Merge acoustic measures",
        help="Collect the per-file acoustic summaries into one table.",
        call="potato.helpers.feature_gather",
        target="taters.helpers.feature_gather:feature_gather",
        scope="global",
        save_as="acoustics_all",
        requires=frozenset({"acoustics_csv"}),
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        with_={
            "root_dir": "{{var:features_dir}}/acoustics",
            "pattern": "*_summary_by_speaker.csv",
            "recursive": False,
            "aggregate": False,
            "add_source_path": True,
            # where a segment sat in the recording is bookkeeping, not voice.
            "exclude_cols": ["start_s_mean", "start_s_std", "end_s_mean",
                             "end_s_std", "segment_index_mean",
                             "segment_index_std", "start_s", "end_s",
                             "segment_index", "utterance_text"],
            "overwrite_existing": "{{var:overwrite_existing}}",
            "out_csv": "{{var:features_dir}}/acoustics_summary.csv",
        },
    ),
    Recipe(
        id="gather_whisper_embeddings",
        feature_table=True,
        gpu_use="cpu",
        hidden=_MERGE_SHAPE,
        label="Merge Whisper embeddings",
        help="Average the segment-level embeddings to one row per speaker per file.",
        call="potato.helpers.feature_gather",
        target="taters.helpers.feature_gather:feature_gather",
        scope="global",
        save_as="whisper_gathered",
        requires=frozenset({"whisper_embeddings_csv"}),
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        with_={
            "root_dir": "{{var:features_dir}}/whisper-embeddings",
            "pattern": "*.csv",
            "aggregate": True,
            "group_by": ["speaker"],
            "per_file": True,
            "stats": ["mean"],
            # same as the sentence-embedding merge: a segment's timing isn't
            # an embedding, and once averaged it was sneaking in as a
            # predictor.
            "exclude_cols": ["text_id", "start_time", "end_time", "text"],
            "overwrite_existing": "{{var:overwrite_existing}}",
            "out_csv": "{{var:features_dir}}/whisper-embeddings_aggregated.csv",
        },
    ),
    Recipe(
        id="gather_transcripts",
        gpu_use="cpu",
        hidden=_MERGE_SHAPE,
        label="Merge transcripts",
        help="Stack every transcript into one table. The text measures read this.",
        call="potato.helpers.feature_gather",
        target="taters.helpers.feature_gather:feature_gather",
        scope="global",
        save_as="transcripts_all",
        requires=frozenset({"transcript_csv"}),
        produces=frozenset({"unified_transcripts_csv"}),
        user_facing=False,
        vars={**_TRANSCRIPTS_DIR_VAR},
        with_={
            "root_dir": "{{var:transcripts_dir}}",
            "pattern": "*.csv",
            "recursive": True,
            "aggregate": False,
            "add_source_path": True,
            "overwrite_existing": "{{var:overwrite_existing}}",
            # goes inside the transcripts folder, beside the per-file
            # transcripts it merges. it used to land in the pipeline root (the
            # module's own default leaking through), where it sat as the one
            # loose file among the tidy per-purpose folders.
            "out_csv": "{{var:transcripts_dir}}/all_transcripts.csv",
        },
    ),

    # ---------------------------------------------------------- text measures
    Recipe(
        id="dictionaries",
        feature_table=True,
        library={"dict_paths": "dictionaries"},
        gpu_use="cpu",
        label="Dictionary counts (LIWC-style)",
        help="Count words from LIWC-format dictionaries. Point it at a folder "
             "of .dicx / .dic / .csv files you already have.",
        call="potato.text.analyze_with_dictionaries",
        target="taters.text.analyze_with_dictionaries:analyze_with_dictionaries",
        scope="global",
        save_as="dict_features",
        requires=frozenset({"unified_transcripts_csv"}),
        vars={
            **_FEATURES_DIR_VAR,
            "dictionaries_path": {
                "default": "dictionaries/liwc",
                "desc": "Folder of LIWC-formatted dictionary files (.dicx, .dic, .csv).",
            },
        },
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/dictionary.csv",
            "dict_paths": ["{{var:dictionaries_path}}"],
            "relative_freq": True,
        },
    ),
    Recipe(
        id="cohesion",
        feature_table=True,
        library={"connective_lists": "connectives"},
        # left alone, exactly the shipped category lists apply. injecting
        # "everything in the library" would double-apply them plus whatever
        # the user imported; extra imported lists join by being picked.
        library_defaults={"connective_lists": (
            "basic_connectives.txt", "conjunctions.txt", "disjunctions.txt",
            "lexical_subordinators.txt", "coordinating_conjuncts.txt",
            "addition.txt", "sentence_linking.txt", "order.txt",
            "reason_and_purpose.txt", "all_causal.txt", "positive_causal.txt",
            "opposition.txt", "determiners.txt", "all_additive.txt",
            "all_logical.txt", "positive_logical.txt", "negative_logical.txt",
            "all_temporal.txt", "positive_intentional.txt",
            "all_positive.txt", "all_negative.txt", "all_connective.txt",
        )},
        vars={
            **_FEATURES_DIR_VAR,
            **_ENGINE_VARS,
        },
        # stanza (when chosen) and the sentence-embedding model both run as
        # one global call, one model.
        gpu_use="gpu_one_model",
        # note that `device` is NOT gated on the stanza engine here: the
        # semantic similarity block runs an embedding model under either
        # engine.
        param_when={"stanza_lang": ("engine", "stanza")},
        label="Text cohesion",
        help="Text cohesion: overlap between adjacent sentences and paragraphs, "
             "connectives, givenness, semantic similarity (the TAACO/Coh-Metrix "
             "families, about 150 columns).",
        text_help="Text cohesion: overlap between adjacent sentences and paragraphs, "
                  "connectives, givenness, semantic similarity (the TAACO/Coh-Metrix "
                  "families, about 150 columns).",
        call="potato.text.analyze_cohesion",
        target="taters.text.analyze_cohesion:analyze_cohesion",
        scope="global",
        save_as="cohesion",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/cohesion.csv",
            "engine": "{{var:engine}}",
            "tokenizer": "{{var:tokenizer}}",
            "stanza_lang": "{{var:stanza_lang}}",
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="word_count",
        feature_table=True,
        # not on the feature checklist. nobody sets out to extract a word
        # count -- it's a thing you filter on, and the filter question adds
        # this step when you pick it. offering it alongside cohesion and
        # readability would put a one-column table on a list of analyses.
        user_facing=False,
        label="Word count",
        help="How many words each text has",
        text_help="How many words each text has",
        call="potato.text.analyze_word_count",
        target="taters.text.analyze_word_count:analyze_word_count",
        scope="global",
        save_as="word_counts",
        requires=frozenset({"unified_transcripts_csv"}),
        sources=("media", "txt_dir", "csv"),
        text_input=True,
        gpu_use="cpu",
        hidden=("txt_dir", "analysis_csv"),
        with_={
            "csv_path": "{{transcripts_all}}",
            "out_features_csv": "{{var:features_dir}}/word_count.csv",
            "text_cols": ["text"],
            "id_cols": _ID_COLS,
            "group_by": _ID_COLS,
            "mode": "concat",
            "delimiter": ",",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        vars={**_FEATURES_DIR_VAR},
    ),
    Recipe(
        id="readability",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="cpu",
        label="Readability scores",
        help="Flesch-Kincaid and friends, per speaker per file.",
        text_help="Flesch-Kincaid and friends, one row per text.",
        call="potato.text.analyze_readability",
        target="taters.text.analyze_readability:analyze_readability",
        scope="global",
        save_as="readability_features",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/readability.csv",
        },
    ),
    Recipe(
        id="sentiment_vader",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="cpu",
        label="Sentiment (VADER)",
        help="Positive, negative and overall tone, per speaker per file.",
        text_help="Positive, negative and overall tone, one row per text.",
        call="potato.text.analyze_sentiment_vader",
        target="taters.text.analyze_sentiment_vader:analyze_sentiment_vader",
        scope="global",
        save_as="vader_features",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/sentiment_vader.csv",
        },
    ),
    Recipe(
        id="lexical_richness",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="cpu",
        label="Lexical richness / vocabulary diversity",
        help="TTR, MTLD, HD-D and related measures, per speaker per file.",
        text_help="TTR, MTLD, HD-D and related measures, one row per text.",
        call="potato.text.analyze_lexical_richness",
        target="taters.text.analyze_lexical_richness:analyze_lexical_richness",
        scope="global",
        save_as="lexrich_features",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/lexical-richness.csv",
        },
    ),
    Recipe(
        id="ngram_frequencies",
        vars={
            **_FEATURES_DIR_VAR,
            **_ENGINE_VARS,
            "lemmatize": _LEMMATIZE_VAR,
            "keep_punctuation": _KEEP_PUNCTUATION_VAR,
            "pos_tagged": _POS_TAGGED_VAR,
        },
        # one global call; under engine=stanza that call holds one model on
        # the GPU (when the device allows), never one per worker.
        gpu_use="gpu_one_model",
        param_when=dict(_ENGINE_PARAM_WHEN),
        library={"stoplist_paths": "stoplists"},
        # the usual "everything in the library" default would apply all 22
        # languages' stopwords at once -- German's "die" quietly stripping
        # English text. left alone, exactly these three apply: punctuation and
        # English. that's what nearly everyone wants without any setup.
        library_defaults={"stoplist_paths": _DEFAULT_STOPLISTS},
        label="N-gram frequency list",
        help="A corpus-wide frequency list of words and phrases across all "
             "speakers, with NPMI and logDice collocation scores for phrases.",
        text_help="A corpus-wide frequency list of words and phrases across "
                  "all texts, with NPMI and logDice collocation scores for "
                  "phrases.",
        call="potato.text.analyze_ngram_frequencies",
        target="taters.text.analyze_ngram_frequencies:analyze_ngram_frequencies",
        scope="global",
        save_as="ngram_freqs",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"ngram_freq_csv"}),
        auto_with=("ngram_frequency_wordclouds",),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/ngram_frequencies.csv",
            "ngram_n": 1,
            "lemmatize": "{{var:lemmatize}}",
            "keep_punctuation": "{{var:keep_punctuation}}",
            "pos_tagged": "{{var:pos_tagged}}",
            **_ENGINE_WITH,
        },
    ),
    Recipe(
        id="ngram_frequency_wordclouds",
        label="Word cloud of the most frequent terms",
        help="The corpus's most frequent words and phrases as one word cloud, "
             "drawn from the frequency list",
        call="potato.figures.frequency_wordclouds",
        target="taters.figures.wordclouds:frequency_wordclouds",
        scope="global",
        save_as="ngram_wordcloud",
        # never on the checklist: it tags along with the frequency list, and
        # the shared `wordclouds` switch turns it off.
        user_facing=False,
        requires=frozenset({"ngram_freq_csv"}),
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "freq_csv": "{{ngram_freqs}}",
            "out_dir": "{{var:features_dir}}/figures/wordclouds/ngram_frequencies",
            "enabled": "{{var:wordclouds}}",
            "top_words": "{{var:wordcloud_words}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("freq_csv", "out_dir"),
        vars={**_FEATURES_DIR_VAR, **_WORDCLOUD_VARS},
    ),
    Recipe(
        id="doc_term_matrix",
        vars={
            **_FEATURES_DIR_VAR,
            # the same vars the frequency list reads. the DTM scans text
            # against that list's vocabulary, and a lemmatized (or tagged)
            # vocabulary can't be found in an unprepared stream -- it fails
            # silently, as near-zero matches. sharing the answers keeps the
            # two steps agreeing.
            "lemmatize": _LEMMATIZE_VAR,
            "keep_punctuation": _KEEP_PUNCTUATION_VAR,
            "pos_tagged": _POS_TAGGED_VAR,
            **_ENGINE_VARS,
            **_MATRIX_SHAPE_VARS,
        },
        gpu_use="gpu_one_model",
        param_when={**_ENGINE_PARAM_WHEN, **_VOCAB_PARAM_WHEN},
        label="Document-term matrix",
        help="One row per speaker per file, one column per frequent word or "
             "phrase. Uses the n-gram frequency list as its vocabulary, so "
             "that step comes with this one.",
        text_help="One row per text, one column per frequent word or phrase. "
                  "Uses the n-gram frequency list as its vocabulary, so that "
                  "step comes with this one.",
        call="potato.text.build_doc_term_matrix",
        target="taters.text.build_doc_term_matrix:build_doc_term_matrix",
        scope="global",
        save_as="doc_term_matrix",
        # one row per text keyed by text_id, so it joins the analysis table
        # like any other measure -- and predicting an outcome from n-gram
        # frequencies (reduced to components first, or not) is the classic
        # open-vocabulary design. we'd originally left it out as "a matrix,
        # not features", and then the wizard refused someone's run that picked
        # it alongside statistics. the n-gram *frequency list* stays out
        # though: that one is a table of the corpus, not of each text.
        feature_table=True,
        requires=frozenset({"unified_transcripts_csv", "ngram_freq_csv"}),
        produces=frozenset({"doc_term_matrix_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "freq_list_csv": "{{ngram_freqs}}",
            # the weighting rides in the filename so that rerunning with a
            # different one writes a sibling file instead of skipping or
            # clobbering -- count/binary/relfreq/tfidf matrices can coexist.
            "out_features_csv":
                "{{var:features_dir}}/doc_term_matrix_{{var:weighting}}.csv",
            "lemmatize": "{{var:lemmatize}}",
            "keep_punctuation": "{{var:keep_punctuation}}",
            "pos_tagged": "{{var:pos_tagged}}",
            **_MATRIX_SHAPE_WITH,
            **_ENGINE_WITH,
        },
    ),
    Recipe(
        id="topic_model_mem",
        # the same stop lists the n-gram step applies, and for the same
        # reason: left unfiltered, the most frequent words in any corpus are
        # function words, and every topic comes out as "the, and, of". this
        # step builds its own frequency list, so it has to ask for them
        # itself -- it used to inherit them from a shared n-gram step.
        library={"stoplist_paths": "stoplists"},
        library_defaults={"stoplist_paths": _DEFAULT_STOPLISTS},
        feature_table=True,
        vars={
            **_FEATURES_DIR_VAR,
            # everything that decided what the matrix LOOKS like is shared
            # with the matrix step. the model file records these settings so
            # the themes can be re-applied to new texts later, and a recorded
            # setting that disagrees with the matrix would be a lie.
            **_vocab_vars("mem"),
            **_ENGINE_VARS,
            "mem_components": {
                "default": 0,
                "desc": "How many themes to keep. 0 decides by the rule "
                        "below; a number is honored as asked.",
            },
            "mem_k_selection": {
                "default": "parallel",
                "desc": "How 0 decides the number of themes. parallel: keep "
                        "a theme while its eigenvalue beats what a random "
                        "matrix of the same shape would give -- the "
                        "recommended rule, because a wide matrix makes large "
                        "eigenvalues by chance alone. kaiser: keep every "
                        "theme whose eigenvalue beats the fixed cutoff "
                        "below. Both read numbers the fit already has. "
                        "coherence and coherence_exclusivity instead score "
                        "the themes' words at each count, which is how LDA "
                        "and NMF choose, and costs a rotation per count.",
            },
            "mem_kaiser_cutoff": {
                "default": 1.5,
                "desc": "The eigenvalue a theme has to beat under kaiser. "
                        "The textbook value is 1, but on a vocabulary of "
                        "hundreds of terms that keeps almost everything "
                        "(one real corpus gave 101 themes).",
            },
            **_k_sweep_vars("mem"),
        },
        # the fit is pure linear algebra, but the step is not: it builds its
        # own frequency list and matrix now, which means it tokenizes -- and
        # with engine="stanza" that loads a model that can sit on the GPU. this
        # said "cpu" for a while after the matrix moved in here, which was a
        # leftover from when somebody else did the tokenizing. same declaration
        # as `ngram_frequencies` and `doc_term_matrix`, because it now does
        # exactly what they do.
        gpu_use="gpu_one_model",
        # `device` only means anything under stanza, same as every other step
        param_when={**_ENGINE_PARAM_WHEN, **_VOCAB_PARAM_WHEN,
                    # the rule decides nothing once a count is given.
                    "k_selection": ("n_components", 0),
                    # and the cutoff belongs to one rule in particular, so it
                    # takes both conditions
                    "kaiser_cutoff": [("n_components", 0),
                                      ("k_selection", "kaiser")],
                    # the two scoring rules are the only ones that fit
                    # anything, so these belong to neither eigenvalue rule
                    "k_values": _MEM_SWEEPING,
                    "coherence_metric": _MEM_SWEEPING,
                    "top_terms": _MEM_SWEEPING},
        label="Topic model: meaning extraction method",
        help="Themes from the document-term matrix (Chung & Pennebaker's "
             "meaning extraction method): a score per theme per speaker, the "
             "words behind each theme, and a saved model.",
        text_help="Themes from the document-term matrix (Chung & Pennebaker's "
                  "meaning extraction method): a score per theme per text, the "
                  "words behind each theme, and a saved model.",
        call="potato.text.topic_model_mem",
        target="taters.text.topic_model_mem:topic_model_mem",
        scope="global",
        save_as="mem_topics",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"mem_loadings_csv"}),
        auto_with=("topic_model_mem_wordclouds",),
        # it reads the corpus and builds its own frequency list and matrix,
        # into a `<stem>_matrix` folder beside its results. it used to be wired to a
        # shared matrix step, which is wrong as soon as there is more than one
        # topic model: LDA needs counts, NMF wants tf-idf, MEM takes either,
        # and the shared matrix's filename carried only the weighting -- so two
        # of them asking for different vocabularies rebuilt over each other.
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/topic_model_mem/topic_model_mem.csv",
            # named here rather than left to the analyzer's default, because
            # the record needs to know where the fitted model went: a ridge
            # fitted on these themes carries that file so it can score a
            # corpus the themes were never fitted on.
            "out_model_json":
                "{{var:features_dir}}/topic_model_mem/topic_model_mem_model.json",
            # named for the same reason: a step of its own draws the theme
            # word clouds from this file.
            "out_loadings_csv":
                "{{var:features_dir}}/topic_model_mem/topic_model_mem_loadings.csv",
            "n_components": "{{var:mem_components}}",
            "k_selection": "{{var:mem_k_selection}}",
            "kaiser_cutoff": "{{var:mem_kaiser_cutoff}}",
            "k_values": "{{var:mem_k_values}}",
            "coherence_metric": "{{var:mem_coherence_metric}}",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            **_vocab_with("mem"),
            "engine": "{{var:engine}}",
            "tokenizer": "{{var:tokenizer}}",
            "stanza_lang": "{{var:stanza_lang}}",
            # the step declares `gpu_one_model` because it tokenizes, so it has
            # to be told where to run -- without this it took "auto" whatever
            # the shared device setting said.
            "device": "{{var:device}}",
            # deliberately *not* `_MATRIX_SHAPE_WITH`: those variables are
            # the document-term-matrix step's, and MEM shared them back when it
            # scanned the matrix that step built. It builds its own now, so
            # sharing them only means changing one screen changes the other's
            # model -- and it would override this step's own defaults, which
            # differ on purpose (a topic model ranks its vocabulary by how many
            # documents use a term, a feature table by raw count).
        },
    ),
    Recipe(
        id="topic_model_mem_apply",
        feature_table=True,
        # never offered on the checklist: this is the shape a saved model's
        # replay of a topic-model table takes. a ridge fitted on themes
        # carries the theme model, and the composer builds a private step
        # from this recipe that applies it to the new text. refitting would
        # pick different themes and then the ridge couldn't be scored.
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        # the saved model says which tokenizer built its matrix; a stanza
        # model loads once and tags every text, same as the tagging step.
        gpu_use="gpu_one_model",
        label="Topic model: apply saved themes",
        help="Score texts on the themes of a saved meaning-extraction model.",
        call="potato.text.apply_mem_model",
        target="taters.text.topic_model_mem:apply_mem_model",
        scope="global",
        save_as="mem_topics_applied",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "device": "{{var:device}}",
            "out_features_csv":
                "{{var:features_dir}}/topic_model_mem/topic_model_mem_applied.csv",
        },
    ),
    Recipe(
        id="topic_model_mem_wordclouds",
        label="Word clouds of the themes",
        help="One word cloud per theme of the topic model: the terms that "
             "define it, sized by their loading, blue loading positively and "
             "red negatively",
        call="potato.figures.theme_wordclouds",
        target="taters.figures.wordclouds:theme_wordclouds",
        scope="global",
        save_as="mem_wordclouds",
        user_facing=False,
        requires=frozenset({"mem_loadings_csv"}),
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "loadings_csv": "{{var:features_dir}}/topic_model_mem/topic_model_mem_loadings.csv",
            "out_dir": "{{var:features_dir}}/figures/wordclouds/topic_model_mem",
            "enabled": "{{var:wordclouds}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("loadings_csv", "out_dir"),
        vars={**_FEATURES_DIR_VAR, **_WORDCLOUD_VARS},
    ),
    Recipe(
        id="topic_model_lda",
        # the same stop lists the n-gram step applies, and for the same
        # reason: left unfiltered, the most frequent words in any corpus are
        # function words, and every topic comes out as "the, and, of". this
        # step builds its own frequency list, so it has to ask for them
        # itself -- it used to inherit them from a shared n-gram step.
        library={"stoplist_paths": "stoplists"},
        library_defaults={"stoplist_paths": _DEFAULT_STOPLISTS},
        feature_table=True,
        vars={
            **_FEATURES_DIR_VAR,
            # the same settings the matrix is built from, because this step
            # builds it -- and the model records them so the topics can be
            # applied to another corpus later.
            **_vocab_vars("lda"),
            **_ENGINE_VARS,
            "lda_topics": {
                "default": 20,
                "desc": "How many topics to fit. There is no right answer -- "
                        "more topics means narrower ones. 0 chooses by "
                        "fitting at several counts and scoring them, which "
                        "is the slow way round but leaves the evidence "
                        "behind.",
            },
            "lda_k_selection": {
                "default": "coherence_exclusivity",
                "desc": _K_SELECTION_DESC,
            },
            **_k_sweep_vars("lda"),
            "lda_passes": {
                "default": 10,
                "desc": "How many times to go over the corpus. More is "
                        "steadier and slower; 10 is plenty for most corpora.",
            },
        },
        # it tokenizes to build its own matrix, and under engine="stanza" that
        # is a model that can sit on the GPU -- same as the n-gram family.
        gpu_use="gpu_one_model",
        param_when={"stanza_lang": ("engine", "stanza"), **_VOCAB_PARAM_WHEN,
                    # none of these decides anything once a count is given
                    "k_selection": ("n_topics", 0),
                    "k_values": ("n_topics", 0),
                    "coherence_metric": ("n_topics", 0),
                    "top_terms": ("n_topics", 0)},
        label="Topic model: latent Dirichlet allocation",
        help="Topics from the corpus (LDA, the topic model most papers mean): a share of each topic per speaker, the words behind each topic, and a saved model.",
        text_help="Topics from the corpus (LDA, the topic model most papers mean): a share of each topic per text, the words behind each topic, and a saved model.",
        call="potato.text.topic_model_lda",
        target="taters.text.topic_model_lda:topic_model_lda",
        scope="global",
        save_as="lda_topics",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"lda_loadings_csv"}),
        auto_with=("topic_model_lda_wordclouds",),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/topic_model_lda/topic_model_lda.csv",
            # named rather than defaulted: the record has to know where the
            # fitted model went, so a ridge trained on these columns can carry
            # it and score a corpus the topics were never fitted on.
            "out_model_json": "{{var:features_dir}}/topic_model_lda/topic_model_lda_model.json",
            # named for the same reason: the word-cloud step reads this file.
            "out_loadings_csv": "{{var:features_dir}}/topic_model_lda/topic_model_lda_loadings.csv",
            "n_topics": "{{var:lda_topics}}",
            "k_selection": "{{var:lda_k_selection}}",
            "k_values": "{{var:lda_k_values}}",
            "coherence_metric": "{{var:lda_coherence_metric}}",
            "passes": "{{var:lda_passes}}",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            **_vocab_with("lda"),
            "engine": "{{var:engine}}",
            "tokenizer": "{{var:tokenizer}}",
            "stanza_lang": "{{var:stanza_lang}}",
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="topic_model_lda_apply",
        feature_table=True,
        label="Score texts with saved LDA topics",
        help="Measure a new corpus with topics fitted somewhere else.",
        call="potato.text.apply_lda_model",
        target="taters.text.topic_model_lda:apply_lda_model",
        scope="global",
        save_as="lda_topics_applied",
        user_facing=False,
        requires=frozenset({"unified_transcripts_csv"}),
        gpu_use="gpu_one_model",
        library={"model_json": "models"},
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "out_features_csv": "{{var:features_dir}}/topic_model_lda/topic_model_lda_applied.csv",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            "device": "{{var:device}}",
        },
        vars={**_FEATURES_DIR_VAR},
    ),
    Recipe(
        id="topic_model_lda_wordclouds",
        label="Word clouds of the topics",
        help="One word cloud per LDA topic: the words that make it up, sized by how much they belong to it",
        call="potato.figures.theme_wordclouds",
        target="taters.figures.wordclouds:theme_wordclouds",
        scope="global",
        save_as="lda_topics_wordclouds",
        user_facing=False,
        requires=frozenset({"lda_loadings_csv"}),
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "loadings_csv": "{{var:features_dir}}/topic_model_lda/topic_model_lda_loadings.csv",
            "out_dir": "{{var:features_dir}}/figures/wordclouds/topic_model_lda",
            "enabled": "{{var:wordclouds}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("loadings_csv", "out_dir"),
        vars={**_FEATURES_DIR_VAR, **_WORDCLOUD_VARS},
    ),
    Recipe(
        id="topic_model_nmf",
        # the same stop lists the n-gram step applies, and for the same
        # reason: left unfiltered, the most frequent words in any corpus are
        # function words, and every topic comes out as "the, and, of". this
        # step builds its own frequency list, so it has to ask for them
        # itself -- it used to inherit them from a shared n-gram step.
        library={"stoplist_paths": "stoplists"},
        library_defaults={"stoplist_paths": _DEFAULT_STOPLISTS},
        feature_table=True,
        vars={
            **_FEATURES_DIR_VAR,
            # the same settings the matrix is built from, because this step
            # builds it -- and the model records them so the topics can be
            # applied to another corpus later.
            **_vocab_vars("nmf"),
            **_ENGINE_VARS,
            "nmf_factors": {
                "default": 20,
                "desc": "How many factors to fit. There is no right answer -- "
                        "more factors means narrower ones. 0 chooses by "
                        "fitting at several counts and scoring them, which "
                        "is the slow way round but leaves the evidence "
                        "behind.",
            },
            "nmf_k_selection": {
                "default": "coherence_exclusivity",
                "desc": _K_SELECTION_DESC,
            },
            **_k_sweep_vars("nmf"),
        },
        # it tokenizes to build its own matrix, and under engine="stanza" that
        # is a model that can sit on the GPU -- same as the n-gram family.
        gpu_use="gpu_one_model",
        param_when={"stanza_lang": ("engine", "stanza"), **_VOCAB_PARAM_WHEN,
                    # none of these decides anything once a count is given
                    "k_selection": ("n_topics", 0),
                    "k_values": ("n_topics", 0),
                    "coherence_metric": ("n_topics", 0),
                    "top_terms": ("n_topics", 0)},
        label="Topic model: non-negative matrix factorization",
        help="Topics from the corpus (NMF, often sharper on short texts): a weight for each factor per speaker, the words behind each topic, and a saved model.",
        text_help="Topics from the corpus (NMF, often sharper on short texts): a weight for each factor per text, the words behind each topic, and a saved model.",
        call="potato.text.topic_model_nmf",
        target="taters.text.topic_model_nmf:topic_model_nmf",
        scope="global",
        save_as="nmf_topics",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"nmf_loadings_csv"}),
        auto_with=("topic_model_nmf_wordclouds",),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/topic_model_nmf/topic_model_nmf.csv",
            # named rather than defaulted: the record has to know where the
            # fitted model went, so a ridge trained on these columns can carry
            # it and score a corpus the topics were never fitted on.
            "out_model_json": "{{var:features_dir}}/topic_model_nmf/topic_model_nmf_model.json",
            # named for the same reason: the word-cloud step reads this file.
            "out_loadings_csv": "{{var:features_dir}}/topic_model_nmf/topic_model_nmf_loadings.csv",
            "n_topics": "{{var:nmf_factors}}",
            "k_selection": "{{var:nmf_k_selection}}",
            "k_values": "{{var:nmf_k_values}}",
            "coherence_metric": "{{var:nmf_coherence_metric}}",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            **_vocab_with("nmf"),
            "engine": "{{var:engine}}",
            "tokenizer": "{{var:tokenizer}}",
            "stanza_lang": "{{var:stanza_lang}}",
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="topic_model_nmf_apply",
        feature_table=True,
        label="Score texts with saved NMF topics",
        help="Measure a new corpus with topics fitted somewhere else.",
        call="potato.text.apply_nmf_model",
        target="taters.text.topic_model_nmf:apply_nmf_model",
        scope="global",
        save_as="nmf_topics_applied",
        user_facing=False,
        requires=frozenset({"unified_transcripts_csv"}),
        gpu_use="gpu_one_model",
        library={"model_json": "models"},
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "out_features_csv": "{{var:features_dir}}/topic_model_nmf/topic_model_nmf_applied.csv",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            "device": "{{var:device}}",
        },
        vars={**_FEATURES_DIR_VAR},
    ),
    Recipe(
        id="topic_model_nmf_wordclouds",
        label="Word clouds of the factors",
        help="One word cloud per NMF factor: the words that make it up, sized by their weight",
        call="potato.figures.theme_wordclouds",
        target="taters.figures.wordclouds:theme_wordclouds",
        scope="global",
        save_as="nmf_topics_wordclouds",
        user_facing=False,
        requires=frozenset({"nmf_loadings_csv"}),
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "loadings_csv": "{{var:features_dir}}/topic_model_nmf/topic_model_nmf_loadings.csv",
            "out_dir": "{{var:features_dir}}/figures/wordclouds/topic_model_nmf",
            "enabled": "{{var:wordclouds}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("loadings_csv", "out_dir"),
        vars={**_FEATURES_DIR_VAR, **_WORDCLOUD_VARS},
    ),
    Recipe(
        id="word_vectors_train",
        feature_table=True,
        label="Word vectors: train on these texts",
        help="Train word2vec or fastText on the speakers' own language: each "
             "speaker's mean word vector, plus similarity to concept "
             "dictionaries. Saves a model that scores new data.",
        text_help="Train word2vec or fastText on these texts: each text's mean "
                  "word vector, plus similarity to concept dictionaries. Saves a "
                  "model that scores new data.",
        call="potato.text.train_word_vectors",
        target="taters.text.word_vectors:train_word_vectors",
        scope="global",
        save_as="word_vectors",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"word_vectors_model_json",
                            "word_vectors_neighbors_csv"}),
        auto_with=("word_vectors_wordclouds",),
        extras=("vectors",),
        # concepts are LIWC-22 dictionaries from the library, one sim_ column
        # per category -- and they're optional, so the default we declare is
        # *no* dictionary rather than the whole library.
        library={"concept_dicts": "dictionaries"},
        library_defaults={"concept_dicts": ()},
        # tokenizing under engine=stanza loads one model; gensim itself is
        # CPU-only, so this declaration is really about the tokenizer.
        gpu_use="gpu_one_model",
        param_when=dict(_ENGINE_PARAM_WHEN),
        **_TEXT_STEP,
        vars={
            **_FEATURES_DIR_VAR,
            "lemmatize": _LEMMATIZE_VAR,
            "keep_punctuation": _KEEP_PUNCTUATION_VAR,
            **_ENGINE_VARS,
            "wv_family": {
                "default": "word2vec",
                "desc": "word2vec learns a vector per word; fasttext also "
                        "learns from character n-grams while training, which "
                        "helps with rare and misspelt words.",
            },
            "wv_algorithm": {
                "default": "skipgram",
                "desc": "skipgram does better on small corpora and rare "
                        "words; cbow is faster on very large ones.",
            },
            "wv_size": {
                "default": 100,
                "desc": "Dimensions per word vector (and wv_ columns). 50–300 "
                        "is the usual range; more needs more text.",
            },
            "wv_window": {
                "default": 5,
                "desc": "Context words either side of a word that count as "
                        "its company.",
            },
            "wv_min_count": {
                "default": 5,
                "desc": "The fewest times a word must occur to get a vector. "
                        "Words below it are skipped everywhere.",
            },
            "wv_epochs": {
                "default": 5,
                "desc": "Passes over the corpus. More helps a small corpus; "
                        "the report shows the loss per epoch.",
            },
            "wv_weighting": {
                "default": "tokens",
                "desc": "How words are averaged into a text's vector: every "
                        "occurrence (tokens), each distinct word once (types), "
                        "or smooth inverse frequency (sif), where the "
                        "commonest words count least.",
            },
            "wv_seed": {
                "default": 42,
                "desc": "Random seed for training. The same seed on one thread "
                        "reproduces the vectors exactly; on several threads, "
                        "only in distribution.",
            },
        },
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/word_vectors.csv",
            # named so the record knows where the model went: a ridge fitted
            # on these features carries the model and can score a corpus
            # the vectors were never trained on. the matrix lands beside it.
            "out_model_json": "{{var:features_dir}}/models/word_vectors.json",
            "out_neighbors_csv":
                "{{var:features_dir}}/models/word_vectors_neighbors.csv",
            "out_report_md": "{{var:features_dir}}/models/word_vectors_report.md",
            "lemmatize": "{{var:lemmatize}}",
            "keep_punctuation": "{{var:keep_punctuation}}",
            **_ENGINE_WITH,
            "family": "{{var:wv_family}}",
            "algorithm": "{{var:wv_algorithm}}",
            "vector_size": "{{var:wv_size}}",
            "window": "{{var:wv_window}}",
            "min_count": "{{var:wv_min_count}}",
            "epochs": "{{var:wv_epochs}}",
            "weighting": "{{var:wv_weighting}}",
            "seed": "{{var:wv_seed}}",
        },
    ),
    Recipe(
        id="adapt_encoder",
        stage="train",
        encoder_param="base_model",
        label="Adapt a language model to these texts",
        help="Continue an encoder's own pretraining on your texts, no labels "
             "needed, so it speaks their dialect before embedding or "
             "predicting. Saves a text encoder with a report.",
        call="potato.text.adapt_encoder",
        target="taters.text.adapt_encoder:adapt_encoder",
        scope="global",
        save_as="adapted_encoder",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"adapted_encoder_json"}),
        gpu_use="gpu_one_model",
        **_TEXT_STEP,
        vars={
            **_FEATURES_DIR_VAR,
            "encoder": {
                "default": "distilroberta-base",
                "desc": "The encoder to start from: a Hugging Face name "
                        "(distilroberta-base; sentence-transformers/"
                        "all-MiniLM-L6-v2 for a laptop; roberta-base; "
                        "bert-base-uncased; microsoft/deberta-v3-base), a "
                        "checkpoint folder, or a text encoder from your "
                        "library (its .json file) to adapt further.",
            },
            "encoder_name": {
                "default": "adapted_encoder",
                "desc": "What to call the adapted encoder (its file name and "
                        "its name in menus).",
            },
            "adapt_epochs": {
                "default": 3,
                "desc": "Passes over the corpus. One to three is usual; the "
                        "report's held-out loss per epoch shows when more "
                        "stopped helping.",
            },
            "adapt_max_length": {
                "default": 256,
                "desc": "Tokens per training window. Longer texts are cut "
                        "into windows so every token is trained on.",
            },
            "adapt_batch_size": {
                "default": 16,
                "desc": "Windows per forward pass. Halved automatically if "
                        "the GPU runs out of memory.",
            },
            "adapt_learning_rate": {
                "default": 5e-5,
                "desc": "Peak learning rate (AdamW, linear warm-up and decay).",
            },
            "adapt_train_layers": {
                "default": 0,
                "desc": "Train only the top this-many layers; 0 trains all. "
                        "2 is a good compromise on a CPU.",
            },
            "adapt_seed": {
                "default": 42,
                "desc": "Seeds the held-out split, the shuffles and the masks.",
            },
        },
        with_={
            **_TEXT_INPUT_WITH,
            "out_model_json": "{{var:features_dir}}/models/{{var:encoder_name}}.json",
            "out_report_md":
                "{{var:features_dir}}/models/{{var:encoder_name}}_report.md",
            "base_model": "{{var:encoder}}",
            "name": "{{var:encoder_name}}",
            "epochs": "{{var:adapt_epochs}}",
            "max_length": "{{var:adapt_max_length}}",
            "batch_size": "{{var:adapt_batch_size}}",
            "learning_rate": "{{var:adapt_learning_rate}}",
            "train_layers": "{{var:adapt_train_layers}}",
            "seed": "{{var:adapt_seed}}",
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="finetune_text_predictor",
        stage="train",
        encoder_param="base_model",
        label="Fine-tune a transformer to predict outcomes from text",
        help="Fine-tune a transformer to predict one or more of your "
             "spreadsheet's columns from the text, cross-validated. Saves a "
             "model that scores new texts, with a full report.",
        call="potato.text.finetune_text_predictor",
        target="taters.text.finetune_predictor:finetune_text_predictor",
        scope="global",
        save_as="text_predictor",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"text_predictor_json"}),
        gpu_use="gpu_one_model",
        hidden=("analysis_csv",),
        sources=("csv",),
        text_input=True,
        vars={
            **_FEATURES_DIR_VAR,
            "encoder": {
                "default": "distilroberta-base",
                "desc": "The encoder to start from: a Hugging Face name "
                        "(distilroberta-base; sentence-transformers/"
                        "all-MiniLM-L6-v2 for a laptop; roberta-base; "
                        "bert-base-uncased; microsoft/deberta-v3-base), a "
                        "checkpoint folder, a text encoder from your library, "
                        "or a fine-tuned predictor from your library to warm-"
                        "start from (its heads are reused for outcomes of the "
                        "same name).",
            },
            "predictor_name": {
                "default": "",
                "desc": "What to call the fine-tuned model (its file name and "
                        "its name in menus). Left blank, it is named after "
                        "the outcomes it predicts.",
            },
            "predictor_outcomes": {
                "default": [],
                "desc": "The column(s) to predict. A column of numbers is a "
                        "regression, a column of labels a classification; "
                        "several columns train one model with a head each.",
            },
            "predictor_categorical": {
                "default": [],
                "desc": "Which of the outcomes are categories even though "
                        "they look like numbers (a 0/1 condition code).",
            },
            "predictor_folds": {
                "default": 5,
                "desc": "Cross-validation folds. Every reported number is out "
                        "of fold.",
            },
            "predictor_epochs": {
                "default": 3,
                "desc": "The most passes over the training rows per fold; the "
                        "epoch with the lowest validation loss is kept.",
            },
            "predictor_max_length": {
                "default": 256,
                "desc": "Tokens per window. A longer text is cut into windows, "
                        "each trained with the text's labels (weighted so the "
                        "text counts once), and predicted as the mean over its "
                        "windows; the share of texts needing more than one is "
                        "reported.",
            },
            "predictor_batch_size": {
                "default": 16,
                "desc": "Texts per forward pass. Halved automatically if the "
                        "GPU runs out of memory.",
            },
            "predictor_learning_rate": {
                "default": 2e-5,
                "desc": "Peak learning rate (AdamW, linear warm-up and decay).",
            },
            "predictor_train_layers": {
                "default": 0,
                "desc": "Train only the top this-many encoder layers and the "
                        "heads; 0 trains all. 2 is a good compromise on a CPU.",
            },
            "predictor_seed": {
                "default": 42,
                "desc": "Seeds the folds, the shuffles and the heads.",
            },
        },
        with_={
            "csv_path": "{{transcripts_all}}",
            "text_cols": ["text"],
            "id_cols": list(_ID_COLS),
            "group_by": list(_ID_COLS),
            "mode": "concat",
            "delimiter": ",",
            "encoding": "utf-8-sig",
            "overwrite_existing": "{{var:overwrite_existing}}",
            "workers": "{{var:workers}}",
            "out_dir": "{{var:features_dir}}/../stats_results",
            "out_models_dir": "{{var:features_dir}}/models",
            "name": "{{var:predictor_name}}",
            "outcome_cols": "{{var:predictor_outcomes}}",
            "categorical_outcomes": "{{var:predictor_categorical}}",
            "base_model": "{{var:encoder}}",
            "n_folds": "{{var:predictor_folds}}",
            "epochs": "{{var:predictor_epochs}}",
            "max_length": "{{var:predictor_max_length}}",
            "batch_size": "{{var:predictor_batch_size}}",
            "learning_rate": "{{var:predictor_learning_rate}}",
            "train_layers": "{{var:predictor_train_layers}}",
            "seed": "{{var:predictor_seed}}",
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="text_predictor_apply",
        feature_table=True,
        # never offered on the checklist: this is the shape a saved
        # predictor's scoring takes when the model is picked from the library.
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="gpu_one_model",
        label="Text predictor: apply a saved model",
        help="Score texts with a fine-tuned text predictor: a prediction per "
             "outcome, classes written with their labels.",
        call="potato.text.apply_text_predictor",
        target="taters.text.finetune_predictor:apply_text_predictor",
        scope="global",
        save_as="text_predictor_applied",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "device": "{{var:device}}",
            "out_features_csv":
                "{{var:features_dir}}/text_predictor_applied.csv",
        },
    ),
    Recipe(
        id="hf_classifier_apply",
        feature_table=True,
        # never offered on the checklist either: the shape a Hugging Face
        # classifier's scoring takes when it's picked from the library.
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="gpu_one_model",
        label="Hugging Face classifier: apply an imported model",
        help="Score texts with a classifier or regressor from the Hugging "
             "Face hub: a predicted class (with its probabilities) or a "
             "number per text.",
        call="potato.text.apply_hf_classifier",
        target="taters.text.hf_classifier:apply_hf_classifier",
        scope="global",
        save_as="hf_classifier_applied",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "device": "{{var:device}}",
            "out_features_csv":
                "{{var:features_dir}}/hf_classifier_applied.csv",
        },
    ),
    Recipe(
        id="word_vectors_apply",
        feature_table=True,
        # never offered on the checklist: this is the shape a saved model's
        # replay of a word-vector table takes, and the step the scoring
        # task composes for a word-vector model picked from the library.
        user_facing=False,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="gpu_one_model",
        label="Word vectors: apply a saved model",
        help="Represent texts with a saved word-vector model: the mean "
             "vector of each text's words and its similarity to the model's "
             "concepts.",
        call="potato.text.apply_word_vectors",
        target="taters.text.word_vectors:apply_word_vectors",
        scope="global",
        save_as="word_vectors_applied",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": "",
            "device": "{{var:device}}",
            "out_features_csv":
                "{{var:features_dir}}/word_vectors_applied.csv",
        },
    ),
    Recipe(
        id="word_vectors_wordclouds",
        label="Word clouds of nearest neighbors",
        help="One word cloud per probe word of the trained word vectors: the "
             "words closest to it, sized by similarity",
        call="potato.figures.neighbor_wordclouds",
        target="taters.figures.wordclouds:neighbor_wordclouds",
        scope="global",
        save_as="word_vectors_wordclouds",
        user_facing=False,
        requires=frozenset({"word_vectors_neighbors_csv"}),
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "neighbors_csv":
                "{{var:features_dir}}/models/word_vectors_neighbors.csv",
            "out_dir": "{{var:features_dir}}/figures/wordclouds/word_vectors",
            "enabled": "{{var:wordclouds}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("neighbors_csv", "out_dir"),
        vars={**_FEATURES_DIR_VAR, **_WORDCLOUD_VARS},
    ),
    Recipe(
        id="parts_of_speech",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR, **_ENGINE_VARS},
        gpu_use="gpu_one_model",
        param_when=dict(_ENGINE_PARAM_WHEN),
        label="Parts of speech",
        help="How much of each speaker's language is nouns, verbs, adjectives "
             "and so on -- relative frequency per part-of-speech tag.",
        text_help="How much of each text is nouns, verbs, adjectives and so "
                  "on -- relative frequency per part-of-speech tag.",
        call="potato.text.analyze_parts_of_speech",
        target="taters.text.analyze_parts_of_speech:analyze_parts_of_speech",
        scope="global",
        save_as="pos_features",
        requires=frozenset({"unified_transcripts_csv"}),
        **_TEXT_STEP,
        with_={
            **_TEXT_INPUT_WITH,
            "out_features_csv": "{{var:features_dir}}/pos.csv",
            "tagset": "penn",
            "relative_freq": True,
            "sngram_n": 1,
            **_ENGINE_WITH,
        },
    ),
    Recipe(
        id="archetypes",
        feature_table=True,
        library={"archetype_csvs": "archetypes"},
        # `ArchetypeQuantifier` loads a sentence-transformer. a `global` step
        # is one call, so the cap changes nothing today -- we declare it
        # anyway because the fact is about the code, not the scope.
        gpu_use="gpu_one_model",
        label="Archetype similarity",
        help="Score each speaker's text against archetype dictionaries using "
             "sentence embeddings. Downloads a model on first run.",
        text_help="Score each text against archetype dictionaries using sentence "
                  "embeddings. Downloads a model on first run.",
        call="potato.text.analyze_with_archetypes",
        target="taters.text.analyze_with_archetypes:analyze_with_archetypes",
        scope="global",
        save_as="archetype_features",
        requires=frozenset({"unified_transcripts_csv"}),
        vars={
            **_FEATURES_DIR_VAR,
            "archetypes_dict_path": {
                "default": "dictionaries/archetypes",
                "desc": "Folder of .csv archetype dictionaries.",
            },
        },
        **_TEXT_STEP,
        with_={
            "csv_path": "{{transcripts_all}}",
            "out_features_csv": "{{var:features_dir}}/archetypes.csv",
            "text_cols": ["text"],
            "id_cols": list(_ID_COLS),
            "group_by": list(_ID_COLS),
            "archetype_csvs": ["{{var:archetypes_dict_path}}"],
            "model_name": "sentence-transformers/all-roberta-large-v1",
            "device": "{{var:device}}",
            "mean_center_vectors": True,
            "rounding": 4,
            "overwrite_existing": "{{var:overwrite_existing}}",
            "workers": "{{var:workers}}",
        },
    ),
    Recipe(
        id="sentence_embeddings",
        feature_table=True,
        # loads a sentence-transformer. also a `global` step; declared for
        # the same reason as `archetypes`.
        gpu_use="gpu_one_model",
        label="Sentence embeddings (meaning-tuned model)",
        help="A numeric fingerprint of what was said, from a model trained so "
             "that utterances meaning the same get similar numbers. The usual "
             "choice. Downloads a model on first run.",
        text_help="A numeric fingerprint of each text, from a model trained so "
                  "that texts meaning the same get similar numbers. The usual "
                  "choice. Downloads a model on first run.",
        call="potato.text.extract_sentence_embeddings",
        target="taters.text.extract_sentence_embeddings:extract_sentence_embeddings",
        scope="global",
        save_as="sent_embeds",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"sentence_embeddings_csv"}),
        auto_with=("gather_sentence_embeddings",),
        **_TEXT_STEP,
        with_={
            "csv_path": "{{transcripts_all}}",
            "text_cols": ["text"],
            "mode": "concat",
            # named, like every other feature table. left to the analyzer's
            # default the file took the gathered table's name, and someone's
            # statistics then called this feature set "texts". the stem is
            # the set's name everywhere downstream, so it matters.
            "out_features_csv": "{{var:features_dir}}/sentence_embeddings.csv",
            "model_name": "sentence-transformers/all-roberta-large-v1",
            "device": "{{var:device}}",
            "normalize_l2": True,
            "pass_through_cols": list(_ID_COLS),
            "overwrite_existing": "{{var:overwrite_existing}}",
            "workers": "{{var:workers}}",
        },
    ),
    Recipe(
        id="transformer_embeddings",
        feature_table=True,
        encoder_param="model_name_or_path",
        gpu_use="gpu_one_model",
        label="Transformer embeddings (raw, any encoder)",
        help="The same fingerprint, read straight out of any encoder, including "
             "one you adapted or fine-tuned here. Not trained to capture "
             "meaning: the specialist's tool.",
        text_help="The same fingerprint, read straight out of any encoder, including "
                  "one you adapted or fine-tuned here. Not trained to capture "
                  "meaning: the specialist's tool.",
        call="potato.text.extract_transformer_embeddings",
        target="taters.text.transformer_embeddings:extract_transformer_embeddings",
        scope="global",
        save_as="transformer_embeds",
        requires=frozenset({"unified_transcripts_csv"}),
        produces=frozenset({"transformer_embeddings_csv"}),
        auto_with=("gather_transformer_embeddings",),
        vars={
            **_FEATURES_DIR_VAR,
            "encoder": {
                "default": "distilroberta-base",
                "desc": "The encoder: a Hugging Face name (distilroberta-base; "
                        "sentence-transformers/all-MiniLM-L6-v2 for a laptop; "
                        "roberta-base; bert-base-uncased; "
                        "microsoft/deberta-v3-base), a checkpoint folder, or "
                        "a text encoder / fine-tuned predictor from your "
                        "library (its .json file).",
            },
            "encoder_layers": {
                "default": "second_to_last",
                "desc": "Which hidden layers are read. second_to_last is the "
                        "recommended reading of an encoder that was not "
                        "fine-tuned (the last layer is specialized to "
                        "predicting masked words); last; last4_mean; "
                        "last4_concat (four times the width); or a list like "
                        "-1,-2.",
            },
            "encoder_pooling": {
                "default": "mean",
                "desc": "How a sentence's token vectors become one: mean over "
                        "its tokens (recommended), the [CLS] position, or the "
                        "element-wise max.",
            },
        },
        **_TEXT_STEP,
        with_={
            "csv_path": "{{transcripts_all}}",
            "text_cols": ["text"],
            "mode": "concat",
            "out_features_csv": "{{var:features_dir}}/transformer_embeddings.csv",
            "model_name_or_path": "{{var:encoder}}",
            "layers": "{{var:encoder_layers}}",
            "pooling": "{{var:encoder_pooling}}",
            "device": "{{var:device}}",
            "pass_through_cols": list(_ID_COLS),
            "overwrite_existing": "{{var:overwrite_existing}}",
            "workers": "{{var:workers}}",
        },
    ),
    Recipe(
        id="gather_transformer_embeddings",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="cpu",
        hidden=_MERGE_SHAPE,
        label="Merge transformer embeddings",
        help="Average utterance embeddings to one row per speaker per file.",
        call="potato.helpers.feature_gather",
        target="taters.helpers.feature_gather:feature_gather",
        scope="global",
        save_as="transformer_embeds_agg",
        requires=frozenset({"transformer_embeddings_csv"}),
        user_facing=False,
        sources=("media", "txt_dir", "csv"),
        source_with={
            "txt_dir": {
                "group_by": TEXT_IDENTITY,
                "exclude_cols": ["text", "source_path"],
            },
            "csv": {
                "group_by": TEXT_IDENTITY,
                "exclude_cols": ["text"],
            },
        },
        with_={
            "root_dir": "{{transformer_embeds}}",
            "aggregate": True,
            "group_by": ["source", "speaker"],
            "per_file": False,
            "stats": ["mean"],
            "add_source_path": False,
            "exclude_cols": ["text_id", "start_time", "end_time", "text",
                             "file_source", "token_count", "sentence_count"],
            "overwrite_existing": "{{var:overwrite_existing}}",
            "out_csv": "{{var:features_dir}}/transformer_embeddings_aggregated.csv",
        },
    ),
    Recipe(
        id="gather_sentence_embeddings",
        feature_table=True,
        vars={**_FEATURES_DIR_VAR},
        gpu_use="cpu",
        hidden=_MERGE_SHAPE,
        label="Merge sentence embeddings",
        help="Average utterance embeddings to one row per speaker per file.",
        call="potato.helpers.feature_gather",
        target="taters.helpers.feature_gather:feature_gather",
        scope="global",
        save_as="sent_embeds_agg",
        requires=frozenset({"sentence_embeddings_csv"}),
        user_facing=False,
        sources=("media", "txt_dir", "csv"),
        # there's no `speaker` column when the input was essays or a
        # spreadsheet, and grouping by a column that isn't there is an error
        # rather than a no-op. `text_id` is the one identifier every input
        # mode produces; for those two it's also already one row per text, so
        # collapsing on it changes nothing and the step just tidies columns.
        source_with={
            "txt_dir": {
                "group_by": TEXT_IDENTITY,
                "exclude_cols": ["text", "source_path"],
            },
            "csv": {
                "group_by": TEXT_IDENTITY,
                "exclude_cols": ["text"],
            },
        },
        with_={
            "root_dir": "{{sent_embeds}}",
            "aggregate": True,
            # (source, speaker), and very much *not* text_id. the row-level
            # embeddings are one row per utterance, so text_id is unique in
            # every one of them -- add it to the key and each group has a
            # single member, which turns the whole step into a copy. and you
            # can't see that failure from the output: the file is the right
            # shape, has the right columns, and the numbers are real. they're
            # just the unaggregated ones.
            "group_by": ["source", "speaker"],
            "per_file": False,
            "stats": ["mean"],
            "add_source_path": False,
            "exclude_cols": ["text_id", "start_time", "end_time", "text", "file_source"],
            "overwrite_existing": "{{var:overwrite_existing}}",
            "out_csv": "{{var:features_dir}}/sentence-embeddings_aggregated.csv",
        },
    ),

    # ------------------------------------------------------------- statistics
    # the stats stage. all of these are csv-source only in v1: a folder of
    # documents has no metadata columns to group on or correlate with, and
    # media transcripts only carry source/speaker. they sit at the END of the
    # catalog on purpose -- among ready global steps the composer breaks ties
    # by catalog position, and that's what lands them after every feature
    # step whose artifacts they read.
    Recipe(
        id="score_with_model",
        feature_table=True,
        # any saved model, and the file says which kind it is. one entry
        # rather than one per kind: "score this dataset with a model I
        # already have" is a single thing to want, and a checklist showing
        # "score with a topic model" beside "score with a prediction model"
        # made people choose a *format* before they'd even chosen a model.
        library={"model_json": "models"},
        # a text model holds one model on the chosen device while scoring,
        # never one per worker. the feature models ignore both.
        gpu_use="gpu_one_model",
        label="Score with models I already have",
        help="Score with models from your library: a ridge, a classifier, a "
             "topic model, word vectors, a fine-tuned or imported transformer. "
             "Several merge into one table.",
        text_help="Score each text with models from your library: a ridge, a "
                  "classifier, a topic model, word vectors, a fine-tuned or "
                  "imported transformer. Several merge into one table.",
        call="potato.score_with_model",
        target="taters.score_model:score_with_model",
        scope="global",
        save_as="model_scores",
        requires=frozenset({"unified_transcripts_csv"}),
        # a prediction model reads features, so we hand it this run's
        # feature tables -- but only some models need them, so an empty list
        # isn't an error here the way it is for the statistics.
        consumes_feature_tables=True,
        feature_tables_optional=True,
        hidden=("txt_dir", "analysis_csv", "key_cols", "feature_csvs",
                "metadata_csv"),
        sources=("media", "txt_dir", "csv"),
        text_input=True,
        vars={
            **_FEATURES_DIR_VAR,
            "model_path": {
                "default": "models",
                "desc": "One or more saved model files (*.json), or a folder "
                        "holding them (every model in it is scored). The "
                        "wizard points this at your library automatically.",
            },
        },
        with_={
            **_TEXT_INPUT_WITH,
            "model_json": ["{{var:model_path}}"],
            "out_csv": "{{var:features_dir}}/model_scores.csv",
            # the join key -- what every feature table is keyed on. not the
            # gather's id_cols above, which say how text_id gets composed in
            # the first place.
            "key_cols": ["text_id"],
            "device": "{{var:device}}",
        },
    ),
    Recipe(
        id="spreadsheet_columns",
        label="Use my spreadsheet's own columns as the measures",
        help="Takes the columns you picked as the predictors, exactly as they "
             "are in your file. Nothing is measured -- the numbers are "
             "already there.",
        call="potato.helpers.csv_to_analysis_ready_csv",
        target="taters.helpers.text_gather:csv_to_analysis_ready_csv",
        scope="global",
        save_as="spreadsheet_features",
        feature_table=True,
        # there is no text to combine, but "one row per participant" is still
        # the user's call -- so the level question gets asked, and the answer
        # decides whether the predictors are carried or averaged
        takes_level=True,
        # text_id has to compose exactly as it does in the metadata gather, or
        # the two tables join on nothing
        keys_like_metadata=True,
        # never on the checklist: this is what "Run analyses" selects for you,
        # and it means nothing without the columns that flow comes with
        user_facing=False,
        sources=("csv",),
        gpu_use="cpu",
        with_={
            "csv_path": "{{var:input_csv}}",
            "out_csv": "{{var:features_dir}}/spreadsheet_columns.csv",
            # no text at all: the whole point is that the measures are
            # already in the file. one row out per row in
            "text_cols": [],
            # ungrouped the predictors are carried as they are; grouped they
            # are averaged, and the wizard fills in whichever applies
            "carry_cols": "{{var:analysis_predictor_cols}}",
            "agg_cols": "{{var:analysis_predictor_agg}}",
            # the id columns identify a row, they do not measure it. left on
            # (the default) they would ride into a *feature* table, and a
            # numeric participant number reads as a predictor from there on
            "include_id_cols": False,
            "delimiter": "{{var:csv_delimiter}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("csv_path", "out_csv", "text_cols", "id_cols", "group_by",
                "mode", "include_id_cols", "carry_cols", "agg_cols"),
        vars={
            **_FEATURES_DIR_VAR,
            "analysis_predictor_cols": {
                "default": [],
                "desc": "The spreadsheet columns to analyze as predictors, "
                        "carried through as they are.",
            },
            "analysis_predictor_agg": {
                "default": {},
                "desc": "The predictors to average instead, when rows are "
                        "combined: {column: 'mean'}.",
            },
            # the same two the metadata gather redeclares, and for the same
            # reason: every {{var:}} a step names has to be declared by some
            # recipe, and the source contributes these at compose time
            "input_csv": {
                "default": "",
                "desc": "Spreadsheet holding the columns to analyze.",
            },
            "csv_delimiter": {
                "default": ",",
                "desc": "Column separator in that file: ',' for CSV, tab for TSV.",
            },
        },
    ),
    Recipe(
        id="describe_features",
        label="Describe every feature table",
        help="Descriptive statistics -- count, missing, mean, standard "
             "deviation, quartiles, range, skewness, kurtosis -- for every "
             "numeric measure of every feature table, under "
             "stats_descriptives/",
        call="potato.stats.describe_features",
        target="taters.stats.describe:describe_features",
        scope="global",
        save_as="descriptives",
        # tags along with every feature table (the composer adds it whenever
        # one is in the run), never on the checklist; a run with no table to
        # describe simply has nothing to do here.
        consumes_feature_tables=True,
        feature_tables_optional=True,
        user_facing=False,
        sources=("media", "txt_dir", "csv"),
        gpu_use="cpu",
        with_={
            "feature_csvs": [],
            "out_dir": "{{var:descriptives_dir}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("feature_csvs",),
        vars={
            "descriptives_dir": {
                "default": "stats_descriptives",
                "desc": "Folder (inside the pipeline's folder) for the "
                        "descriptive statistics of every feature table.",
            },
        },
    ),
    Recipe(
        id="stats_gather_metadata",
        label="Collect the grouping/outcome columns",
        help="Reads the group and outcome columns out of your spreadsheet, "
             "one row per analyzed text",
        call="potato.helpers.csv_to_analysis_ready_csv",
        target="taters.helpers.text_gather:csv_to_analysis_ready_csv",
        scope="global",
        save_as="stats_metadata",
        produces=frozenset({"stats_metadata_csv"}),
        keys_like_metadata=True,
        user_facing=False,
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        # the same source spreadsheet the text gather reads, with the same
        # id/group binding (compose wires that part), but with NO text
        # columns: the output is text_id plus the metadata the statistics
        # need. so text_id composes identically here and in every feature
        # table, and that's what makes the join structural rather than a
        # happy coincidence.
        with_={
            "csv_path": "{{var:input_csv}}",
            "out_csv": "gathered/metadata.csv",
            "text_cols": [],
            "delimiter": "{{var:csv_delimiter}}",
            "carry_cols": "{{var:stats_meta_carry}}",
            "agg_cols": "{{var:stats_meta_agg}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        # all wiring: the columns come from the wizard's analysis stage (or
        # the vars), and compose binds the input group exactly like the text
        # gather's. offering these rows would just collect answers the run
        # ignores.
        hidden=("csv_path", "out_csv", "text_cols", "id_cols", "group_by",
                "mode", "include_id_cols", "carry_cols", "agg_cols"),
        vars={
            **_STATS_META_VARS,
            # mirrors SOURCE_VARS["csv"]: this step references the same
            # variables the source contributes, and the declared-vars test
            # (rightly) wants every reference declared by *some* recipe.
            "input_csv": {
                "default": "",
                "desc": "Spreadsheet holding the text to analyze.",
            },
            "csv_delimiter": {
                "default": ",",
                "desc": "Column separator in that file: ',' for CSV, tab for TSV.",
            },
        },
    ),
    Recipe(
        id="stats_assemble",
        label="Assemble the analysis table",
        help="Joins the chosen feature tables with your metadata into one "
             "wide table, applying any row filters",
        call="potato.stats.assemble_analysis_table",
        target="taters.stats.assemble:assemble_analysis_table",
        scope="global",
        save_as="analysis_table",
        requires=frozenset({"stats_metadata_csv"}),
        produces=frozenset({"analysis_table_csv"}),
        consumes_feature_tables=True,
        user_facing=False,
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        with_={
            # a placeholder: compose fills this in with the selected feature
            # steps' artifacts ("{{cohesion}}", ...). we keep it in the recipe
            # so a hand-written preset gets the step's honest empty-list
            # refusal rather than a missing-argument crash.
            "feature_csvs": [],
            # compose fills this in when a filter names a measure the run has
            # to compute for it; joined so the filter works, never analyzed.
            "filter_csvs": [],
            "bookkeeping": "{{var:stats_bookkeeping}}",
            # the gather writes a count beside every averaged column, and a
            # row count per group. they describe the combining, not the
            # participant -- left alone they would be predictors, and "how
            # many rows this person had" is not a finding anybody wants
            "bookkeeping_cols": "{{var:stats_bookkeeping_cols}}",
            "metadata_csv": "{{stats_metadata}}",
            "key_cols": ["text_id"],
            "split_col": "{{var:stats_split_col}}",
            "filters": "{{var:stats_filters}}",
            "keep_table": "{{var:stats_keep_table}}",
            "out_dir": "{{var:stats_dir}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("feature_csvs", "filter_csvs", "metadata_csv", "key_cols",
                "split_col", "out_csv", "metadata_cols", "bookkeeping_cols"),
        vars={**_STATS_DIR_VAR, **_STATS_FILTER_VAR, **_STATS_KEEP_VAR,
              **_STATS_SPLIT_VAR, **_STATS_BOOKKEEPING_VAR},
    ),
    Recipe(
        id="stats_group_differences",
        outcome_kind="labels",
        label="Group differences (ANOVA + post-hoc)",
        help="Tests every feature for differences between the groups in one "
             "of your columns, with post-hoc pairwise tests, effect sizes "
             "and multiple-comparison correction",
        call="potato.stats.analyze_group_differences",
        target="taters.stats.group_differences:analyze_group_differences",
        scope="global",
        save_as="group_differences",
        requires=frozenset({"analysis_table_csv"}),
        auto_with=("stats_wordclouds", "stats_report"),
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        with_={
            **_STATS_ANALYSIS_WITH,
            "group_col": "{{var:stats_group_col}}",
            "p_adjust": "{{var:stats_p_adjust}}",
        },
        # everything after the plumbing gets answered by the analysis stage
        # -- which column, which tables, together or apart, what to hold
        # constant, how to correct. when we offered them again on the options
        # screen as free text ("feature sets: ›") they read as a second,
        # unrelated question nobody could answer -- and somebody told us so.
        # shared with the other three analyses via the same variables.
        param_when=dict(_STATS_PCA_PARAM_WHEN_WITH_MISSING),
        hidden=("table_csv",
                "group_col",
                "feature_sets",
                "split_col",
                "control_cols",
                "categorical_controls",
                "p_adjust"),
        vars={**_STATS_DIR_VAR, **_STATS_GROUP_VAR, **_STATS_SETS_VAR,
              **_STATS_ADJUST_VAR, **_STATS_SPLIT_VAR,
              **_STATS_CONTROL_VAR,
              **_STATS_PCA_VARS},
    ),
    Recipe(
        id="stats_correlations",
        outcome_kind="numeric",
        label="Correlations with outcomes",
        help="Correlates every feature with your outcome columns -- r, p, "
             "FDR-adjusted p and the pairwise N for each pair",
        call="potato.stats.analyze_correlations",
        target="taters.stats.correlations:analyze_correlations",
        scope="global",
        save_as="correlations",
        requires=frozenset({"analysis_table_csv"}),
        auto_with=("stats_wordclouds", "stats_report"),
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        with_={
            **_STATS_ANALYSIS_WITH,
            "outcome_cols": "{{var:stats_outcome_cols}}",
            "p_adjust": "{{var:stats_p_adjust}}",
        },
        # answered by the analysis stage; see stats_group_differences.
        param_when=dict(_STATS_PCA_PARAM_WHEN_WITH_MISSING),
        hidden=("table_csv",
                "outcome_cols",
                "feature_sets",
                "split_col",
                "control_cols",
                "categorical_controls",
                "p_adjust"),
        vars={**_STATS_DIR_VAR, **_STATS_OUTCOME_VAR, **_STATS_SETS_VAR,
              **_STATS_ADJUST_VAR, **_STATS_SPLIT_VAR,
              **_STATS_CONTROL_VAR,
              **_STATS_PCA_VARS},
    ),
    Recipe(
        id="stats_ridge_fit",
        outcome_kind="numeric",
        label="Prediction model (cross-validated ridge)",
        help="Trains a model to predict your outcome column(s) from the "
             "features, checks it on held-out rows, and saves it so you can "
             "score other datasets with it later",
        call="potato.stats.fit_ridge_csv",
        target="taters.stats.ridge:fit_ridge_csv",
        scope="global",
        save_as="ridge_model",
        requires=frozenset({"analysis_table_csv"}),
        auto_with=("stats_wordclouds", "stats_report"),
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        with_={
            **_STATS_ANALYSIS_WITH,
            "outcome_cols": "{{var:stats_outcome_cols}}",
            "n_folds": "{{var:stats_n_folds}}",
            "set_combos": "{{var:stats_set_combos}}",
            "stratify": "{{var:stats_stratify}}",
        },
        # answered by the analysis stage; see stats_group_differences.
        param_when={**_STATS_PCA_PARAM_WHEN,
                    # combining the tables means nothing once each one is
                    # analyzed on its own.
                    "set_combos": ("feature_sets", "!=", "per_table")},
        hidden=("table_csv",
                "out_models_dir",
                "outcome_cols",
                "feature_sets",
                "split_col",
                "control_cols",
                "categorical_controls"),
        vars={**_STATS_CV_VARS, **_STATS_DIR_VAR, **_STATS_OUTCOME_VAR, **_STATS_SETS_VAR, **_STATS_COMBOS_VAR,
              **_STATS_SPLIT_VAR,
              **_STATS_CONTROL_VAR,
              **_STATS_PCA_VARS},
    ),
    Recipe(
        id="stats_classify_fit",
        outcome_kind="labels",
        label="Classification model (cross-validated logistic)",
        help="Predict which category a row is in (a diagnosis, a condition) "
             "from the features, checked on held-out rows, and saved to score "
             "other datasets later.",
        call="potato.stats.fit_classifier_csv",
        target="taters.stats.classify:fit_classifier_csv",
        scope="global",
        save_as="classifier_model",
        requires=frozenset({"analysis_table_csv"}),
        auto_with=("stats_wordclouds", "stats_report"),
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        with_={
            **_STATS_ANALYSIS_WITH,
            # its own variable, not stats_outcome_cols: a run can sensibly
            # correlate features with a test score *and* predict a diagnosis,
            # and handing either column to the other step is a category
            # error that each one refuses by name.
            "outcome_cols": "{{var:stats_class_cols}}",
            "n_folds": "{{var:stats_n_folds}}",
            "set_combos": "{{var:stats_set_combos}}",
            "stratify": "{{var:stats_stratify}}",
        },
        # answered by the analysis stage; see stats_group_differences.
        param_when={**_STATS_PCA_PARAM_WHEN,
                    # combining the tables means nothing once each one is
                    # analyzed on its own.
                    "set_combos": ("feature_sets", "!=", "per_table")},
        hidden=("table_csv",
                "out_models_dir",
                "outcome_cols",
                "feature_sets",
                "split_col",
                "control_cols",
                "categorical_controls"),
        vars={**_STATS_CV_VARS, **_STATS_DIR_VAR, **_STATS_CLASS_VAR, **_STATS_SETS_VAR, **_STATS_COMBOS_VAR,
              **_STATS_SPLIT_VAR,
              **_STATS_CONTROL_VAR,
              **_STATS_PCA_VARS},
    ),
    Recipe(
        id="stats_wordclouds",
        label="Word clouds of the results",
        help="Draws a word cloud for every result: the features predicting "
             "higher and lower scores, correlating with each outcome, "
             "differing between groups, loading on each component -- and, "
             "with a group column and a document-term matrix, each group's "
             "most frequent terms",
        call="potato.figures.stats_wordclouds",
        target="taters.figures.wordclouds:stats_wordclouds",
        scope="global",
        save_as="stats_wordclouds",
        user_facing=False,
        requires=frozenset({"analysis_table_csv"}),
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        # second to last in the catalog: after every analysis whose tables
        # it draws, before the report that shows off the pictures.
        with_={
            "stats_dir": "{{var:stats_dir}}",
            "enabled": "{{var:wordclouds}}",
            "max_words": "{{var:wordcloud_words}}",
            "group_col": "{{var:stats_group_col}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        # answered by the analysis stage, like the group-differences step's.
        hidden=("group_col",),
        vars={**_STATS_DIR_VAR, **_WORDCLOUD_VARS, **_STATS_GROUP_VAR},
    ),
    Recipe(
        id="stats_report",
        label="Write the statistics report",
        help="Gathers every analysis's summary into one plain-English "
             "report.md",
        call="potato.stats.write_stats_report",
        target="taters.stats.report:write_stats_report",
        scope="global",
        save_as="stats_report",
        user_facing=False,
        sources=("csv",),
        gpu_use="cpu",
        tags=("stats",),
        stage="analyze",
        # last in the catalog, so the tiebreak lands it after the analyses
        # whose section fragments it stitches together.
        with_={
            "stats_dir": "{{var:stats_dir}}",
            "keep_table": "{{var:stats_keep_table}}",
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
        hidden=("out_md",),
        vars={**_STATS_DIR_VAR, **_STATS_KEEP_VAR},
    ),
]


#: Lookup cache for :func:`by_id`. Refreshed on a miss rather than built once
#: at import, because `RECIPES` is a list that can be appended to -- a test
#: adding a recipe, or one day a plugin registering one. Built once, the cache
#: silently disagreed with the catalog and `by_id` raised "unknown recipe" for
#: something sitting in `RECIPES`.
_BY_ID: Dict[str, Recipe] = {}


def by_id(recipe_id: str) -> Recipe:
    """
    Look a recipe up by id.

    Raises
    ------
    KeyError
        With the list of valid ids, because this is almost always a typo in a
        caller and the bare id is not enough to fix it.
    """
    if recipe_id not in _BY_ID:
        _BY_ID.update({r.id: r for r in RECIPES})
    try:
        return _BY_ID[recipe_id]
    except KeyError:
        raise KeyError(
            f"unknown recipe {recipe_id!r}. "
            f"Known: {', '.join(sorted(r.id for r in RECIPES))}"
        ) from None


@dataclass(frozen=True)
class FeatureCategory:
    """One heading on the feature checklist, and what sits under it."""

    id: str
    label: str
    help: str
    members: Tuple[str, ...]


#: The feature checklist, grouped. One flat list of 22 rows in catalog order
#: said nothing about which rows belong together -- that the three topic models
#: and the topic-count sweep are one family, or that readability and lexical
#: richness answer related questions -- and every new extractor made it worse.
#:
#: Grouped by what a measure is *about* rather than by how it is computed, so
#: that somebody looks for the thing they want to know rather than for the
#: machinery that produces it. That is why `archetypes` sits with the
#: dictionaries: it runs on sentence embeddings, but what it gives you is a
#: score against theory-driven categories, which is the dictionary question.
#:
#: One central table rather than a field on each recipe, the way MODEL_TYPES
#: and SETTING_LABELS are central: the whole taxonomy reads in one block, and
#: the order and the headings have somewhere to live. `tags` was the other
#: candidate and is deliberately not used -- compose reads it into every saved
#: preset's `meta.tags`, so headings would leak onto disk.
#:
#: A test pins every user-facing extract recipe to exactly one category, so a
#: new extractor that names no home fails the build rather than quietly
#: vanishing from the checklist.
FEATURE_CATEGORIES: Tuple[FeatureCategory, ...] = (
    FeatureCategory(
        "transcription", "Transcription & speakers",
        "Turn the audio into text, and work out who said what. Everything "
        "under the text headings below can then run on the result.",
        ("transcribe", "diarize", "split_by_speaker")),
    FeatureCategory(
        "voice", "Voice & audio measures",
        "Measures of how it sounded rather than what was said.",
        ("acoustics", "whisper_embeddings")),
    FeatureCategory(
        "style", "Style & readability",
        "How the language is put together -- how hard it is to read, how "
        "varied the vocabulary, which parts of speech, how it hangs together.",
        ("readability", "lexical_richness", "parts_of_speech", "cohesion")),
    FeatureCategory(
        "content", "Content categories & sentiment",
        "Score the text against categories somebody defined in advance: a "
        "dictionary, a sentiment lexicon, a set of archetypes.",
        ("dictionaries", "sentiment_vader", "archetypes")),
    FeatureCategory(
        "topics", "Topics & themes",
        "Let the corpus tell you what it is about, by finding the words that "
        "rise and fall together.",
        ("topic_model_mem", "topic_model_lda", "topic_model_nmf")),
    FeatureCategory(
        "frequencies", "Word & phrase frequencies",
        "Counts over the whole corpus rather than measures of each text.",
        ("ngram_frequencies", "doc_term_matrix")),
    FeatureCategory(
        "vectors", "Embeddings & vectors",
        "Turn each text into numbers that put similar meanings close "
        "together.",
        ("sentence_embeddings", "transformer_embeddings", "word_vectors_train")),
    FeatureCategory(
        "saved_models", "Score with saved models",
        "Measure this corpus with something fitted somewhere else.",
        ("score_with_model",)),
)


def categories_for(source: str = "media",
                   stage: str = "extract") -> List[Tuple[FeatureCategory, List[Recipe]]]:
    """
    :func:`user_facing`, grouped into headings, in table order.

    A category with nothing in it for this source is dropped rather than shown
    empty -- the two audio headings simply are not there for a folder of
    essays. Members come back in category order rather than catalog order,
    because the heading is the promise about what is under it.
    """
    offered = {r.id: r for r in user_facing(source, stage)}
    grouped = []
    for category in FEATURE_CATEGORIES:
        members = [offered[m] for m in category.members if m in offered]
        if members:
            grouped.append((category, members))
    return grouped


def user_facing(source: str = "media", stage: str = "extract") -> List[Recipe]:
    """
    The recipes to show in one wizard checklist, in catalog order.

    Filtered by source, so someone who said "I have a folder of essays" is
    never offered vocal acoustics -- an option that could only ever fail for
    them, and that costs a multi-gigabyte install to find out. And by stage:
    the feature checklist and the statistics stage are different questions,
    and mixing "extract cohesion" with "run an ANOVA" on one screen buries
    both.
    """
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}")
    return [r for r in RECIPES
            if r.user_facing and source in r.sources and r.stage == stage]


def providers_of(capability: str) -> List[Recipe]:
    """
    Every recipe that can satisfy a capability, in catalog order.

    More than one means the user has a choice to make -- ``transcript_csv`` is
    the case that matters, with plain transcription and diarization both able
    to produce it.
    """
    return [r for r in RECIPES if capability in r.produces]

