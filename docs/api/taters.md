# Core Class: `Taters`

The core class of Taters is `Taters()`, the lightweight front door to the library. It does not introduce new functionality; it organizes the public API into namespaces (`audio`, `text`, `stats`, `figures`, `helpers`) and forwards calls to the real implementations with argument validation. This keeps imports fast, error messages clear, and usage consistent whether you are working in a notebook, a script, or a YAML pipeline.

## What it provides

* **Clean namespaces:**

  * `t.audio`: media I/O, transcription and diarization, Whisper embeddings, vocal acoustics.
  * `t.text`: dictionary coding, archetypes, readability, word counts, lexical richness, n-grams, parts of speech, cohesion, document-term matrices and topic models, word vectors, sentence and transformer embeddings, adapting and fine-tuning encoders, subtitle conversion.
  * `t.stats`: the analysis table, group differences, correlations, ridge and classifier fits, PCA, descriptives, the report.
  * `t.figures`: word clouds from results, themes, frequencies and word-vector neighbors.
  * `t.helpers`: file discovery, text gathering, feature gathering.
  * `t.score_with_model`, `t.describe_model`, `t.rename_model`: apply, inspect and rename saved models.

* **Back-compat pass-throughs:** Top-level methods mirror the namespaced ones (e.g., `t.convert_to_wav(...)` just calls `t.audio.convert_to_wav(...)`). Existing notebooks and scripts keep working while the namespaced style becomes the norm.

* **Helpful errors:** Calls are validated against the target function's signature **before** execution. If a parameter is missing or misspelled, you get a clear `TypeError` listing the allowed parameters. Note that every facade method takes **keyword arguments only** — write `t.audio.convert_to_wav(input_path=...)`, not `t.audio.convert_to_wav("input.mp4")`.

* **Lazy imports:** Targets are imported inside the forwarding method, so simply constructing `Taters()` does not pull heavy dependencies into memory. This plays nicely with environments that mix CPU/GPU or optional extras.

## How forwarding works (under the hood)

Each namespaced method loads the real function (for example, `audio.convert_to_wav.convert_audio_to_wav`) and passes your kwargs through a small `_forward(...)` helper. `_forward` binds the kwargs to the function's signature with `inspect.signature(...).bind_partial(...)`; if binding fails, it raises a readable error that includes the "Allowed params" from the target's signature. Then it executes the call.

## Typical usage

```python
from taters import Taters
t = Taters()

# Namespaced (preferred)
wav = t.audio.convert_to_wav(input_path="input.mp4", sample_rate=16000)

# Back-compat (still supported)
wav = t.convert_to_wav(input_path="input.mp4", sample_rate=16000)

# Text workflows
dict_csv = t.text.analyze_with_dictionaries(
    csv_path="transcripts/X.csv",
    text_cols=["text"],
    id_cols=["speaker"],
    group_by=["speaker"],
    dict_paths=["dictionaries/liwc"]
)

# Helpers (facade calls are keyword-only)
found = t.helpers.find_files(root_dir="videos/", file_type="video", ffprobe_verify=True)
```

## Where it fits with pipelines

Pipeline presets refer to these namespaced methods directly (e.g., `call: potato.audio.convert_to_wav`). A single `Taters()` instance is shared across steps, keeping behavior consistent while letting you swap in different calls or parameters without changing your code layout.

## Methods exposed through the core class

All functionality lives in modules under `taters.audio`, `taters.text`, `taters.stats`, `taters.figures` and `taters.helpers`, plus `taters.score_model`; the core class just routes to them.

* **Audio:** `convert_to_wav`, `extract_wavs_from_video`, `split_wav_by_speaker`, `extract_whisper_embeddings`, `transcribe_with_whisper`, `diarize_with_thirdparty`, `analyze_vocal_acoustics` → `t.audio.<name>(...)` (also available at top level for back-compat).
* **Text:** `analyze_with_dictionaries`, `analyze_with_archetypes`, `analyze_readability`, `analyze_word_count`, `analyze_lexical_richness`, `analyze_ngram_frequencies`, `analyze_parts_of_speech`, `analyze_cohesion`, `build_doc_term_matrix`, `topic_model_mem`, `apply_mem_model`, `topic_model_lda`, `apply_lda_model`, `topic_model_nmf`, `apply_nmf_model`, `sweep_topic_count`, `train_word_vectors`, `import_word_vectors`, `apply_word_vectors`, `describe_word_vectors`, `extract_sentence_embeddings`, `extract_transformer_embeddings`, `adapt_encoder`, `pretrain_encoder`, `finetune_text_predictor`, `apply_text_predictor`, `convert_subtitles` → `t.text.<name>(...)`.
* **Statistics:** `assemble_analysis_table`, `analyze_group_differences`, `analyze_correlations`, `fit_ridge_csv`, `apply_ridge_csv`, `fit_classifier_csv`, `apply_classifier_csv`, `fit_pca_csv`, `apply_pca_csv`, `describe_features`, `write_stats_report` → `t.stats.<name>(...)`.
* **Figures:** `stats_wordclouds`, `theme_wordclouds`, `frequency_wordclouds`, `neighbor_wordclouds` → `t.figures.<name>(...)`.
* **Saved models:** `score_with_model` (one model or several; several merge into one table with each model's name prefixed to its columns), `describe_model`, `rename_model` → `t.<name>(...)`.
* **Helpers:** `txt_folder_to_analysis_ready_csv`, `csv_to_analysis_ready_csv`, `find_files`, `feature_gather` → `t.helpers.<name>(...)`.

In short, `Taters` is a coherent surface that stays stable while the individual modules evolve, with validation and lazy imports to keep everyday use straightforward.
