"""
The programmatic front door: ``Taters().text.analyze_readability(...)``.

Every public function in the package is reachable from one object, in four
namespaces (``audio``, ``text``, ``stats``, ``helpers``) plus flat aliases for
the older ``Taters().analyze_readability(...)`` spelling. Each method imports
its module on first use, so constructing a ``Taters`` costs nothing and the
wizard can start before torch has loaded. Pipeline presets name these methods
as ``potato.text.analyze_readability``; ``tests/test_facade.py`` holds the
list of them to the package.
"""

from __future__ import annotations
from typing import Any
import inspect

# these are the arguments the pipeline runner supplies, rather than a user or
# a preset. they're capabilities a target may or may not have, so we drop them
# for targets that don't take them -- unlike a user-supplied name, where an
# unexpected argument is a mistake worth complaining about.
#
# the runner can't tell which of these a target accepts: every method here
# takes `**kwargs`, so the real signature is only visible one frame further
# in, which is right here.
_INJECTED_PARAMS = frozenset({"on_progress", "verbose", "workers"})


def _forward(func, kwargs: dict[str, Any]):
    sig = inspect.signature(func)

    unsupported = [
        name for name in _INJECTED_PARAMS
        if name in kwargs and name not in sig.parameters
    ]
    if unsupported:
        kwargs = {k: v for k, v in kwargs.items() if k not in unsupported}

    try:
        # validate the names & required args; we don't want to run defaults here
        sig.bind_partial(**kwargs)
    except TypeError as e:
        allowed = ", ".join([str(p) for p in sig.parameters.values()])
        raise TypeError(f"{func.__module__}.{func.__name__}: {e}\nAllowed params: {allowed}")
    return func(**kwargs)


class Taters:
    def __init__(self):
        self.audio = _AudioAPI()
        self.text = _TextAPI()
        self.helpers = _HelpersAPI()
        self.stats = _StatsAPI()
        self.figures = _FiguresAPI()

    # back-compat pass-throughs
    
    # audio
    def convert_to_wav(self, **kwargs):                         return self.audio.convert_to_wav(**kwargs)
    def extract_wavs_from_video(self, **kwargs):                return self.audio.extract_wavs_from_video(**kwargs)
    def split_wav_by_speaker(self, **kwargs):                   return self.audio.split_wav_by_speaker(**kwargs)
    def extract_whisper_embeddings(self, **kwargs):             return self.audio.extract_whisper_embeddings(**kwargs)
    def transcribe_with_whisper(self, **kwargs):                return self.audio.transcribe_with_whisper(**kwargs)
    def diarize_with_thirdparty(self, **kwargs):                return self.audio.diarize_with_thirdparty(**kwargs)
    def analyze_vocal_acoustics(self, **kwargs):               return self.audio.analyze_vocal_acoustics(**kwargs)
    
    #text
    def analyze_with_dictionaries(self, **kwargs):              return self.text.analyze_with_dictionaries(**kwargs)
    def analyze_with_norms(self, **kwargs):                     return self.text.analyze_with_norms(**kwargs)
    def analyze_with_archetypes(self, **kwargs):                return self.text.analyze_with_archetypes(**kwargs)
    def analyze_readability(self, **kwargs):                    return self.text.analyze_readability(**kwargs)
    def analyze_word_count(self, **kwargs):                     return self.text.analyze_word_count(**kwargs)
    def analyze_lexical_richness(self, **kwargs):               return self.text.analyze_lexical_richness(**kwargs)
    def analyze_entropy(self, **kwargs):                        return self.text.analyze_entropy(**kwargs)
    def analyze_sentiment_vader(self, **kwargs):                return self.text.analyze_sentiment_vader(**kwargs)
    def analyze_ngram_frequencies(self, **kwargs):              return self.text.analyze_ngram_frequencies(**kwargs)
    def analyze_parts_of_speech(self, **kwargs):                return self.text.analyze_parts_of_speech(**kwargs)
    def build_doc_term_matrix(self, **kwargs):                  return self.text.build_doc_term_matrix(**kwargs)
    def topic_model_mem(self, **kwargs):                        return self.text.topic_model_mem(**kwargs)
    def apply_mem_model(self, **kwargs):                        return self.text.apply_mem_model(**kwargs)
    def topic_model_lda(self, **kwargs):                        return self.text.topic_model_lda(**kwargs)
    def apply_lda_model(self, **kwargs):                        return self.text.apply_lda_model(**kwargs)
    def topic_model_nmf(self, **kwargs):                        return self.text.topic_model_nmf(**kwargs)
    def apply_nmf_model(self, **kwargs):                        return self.text.apply_nmf_model(**kwargs)
    def sweep_topic_count(self, **kwargs):                      return self.text.sweep_topic_count(**kwargs)
    def train_word_vectors(self, **kwargs):                     return self.text.train_word_vectors(**kwargs)
    def apply_word_vectors(self, **kwargs):                     return self.text.apply_word_vectors(**kwargs)
    def import_word_vectors(self, **kwargs):                    return self.text.import_word_vectors(**kwargs)
    def describe_word_vectors(self, **kwargs):                  return self.text.describe_word_vectors(**kwargs)
    def extract_transformer_embeddings(self, **kwargs):         return self.text.extract_transformer_embeddings(**kwargs)
    def adapt_encoder(self, **kwargs):                          return self.text.adapt_encoder(**kwargs)
    def pretrain_encoder(self, **kwargs):                       return self.text.pretrain_encoder(**kwargs)
    def finetune_text_predictor(self, **kwargs):                return self.text.finetune_text_predictor(**kwargs)
    def apply_text_predictor(self, **kwargs):                   return self.text.apply_text_predictor(**kwargs)
    def import_hf_classifier(self, **kwargs):                   return self.text.import_hf_classifier(**kwargs)
    def apply_hf_classifier(self, **kwargs):                    return self.text.apply_hf_classifier(**kwargs)
    def analyze_cohesion(self, **kwargs):                       return self.text.analyze_cohesion(**kwargs)
    def extract_sentence_embeddings(self, **kwargs):            return self.text.extract_sentence_embeddings(**kwargs)
    def convert_subtitles(self, **kwargs):                      return self.text.convert_subtitles(**kwargs)
    
    # stats
    def assemble_analysis_table(self, **kwargs):                return self.stats.assemble_analysis_table(**kwargs)
    def analyze_group_differences(self, **kwargs):              return self.stats.analyze_group_differences(**kwargs)
    def analyze_correlations(self, **kwargs):                   return self.stats.analyze_correlations(**kwargs)
    def write_stats_report(self, **kwargs):                     return self.stats.write_stats_report(**kwargs)
    def fit_ridge_csv(self, **kwargs):                          return self.stats.fit_ridge_csv(**kwargs)
    def apply_ridge_csv(self, **kwargs):                        return self.stats.apply_ridge_csv(**kwargs)
    def fit_classifier_csv(self, **kwargs):                     return self.stats.fit_classifier_csv(**kwargs)
    def apply_classifier_csv(self, **kwargs):                   return self.stats.apply_classifier_csv(**kwargs)
    def fit_pca_csv(self, **kwargs):                            return self.stats.fit_pca_csv(**kwargs)
    def apply_pca_csv(self, **kwargs):                          return self.stats.apply_pca_csv(**kwargs)
    def describe_features(self, **kwargs):                      return self.stats.describe_features(**kwargs)

    # figures
    def stats_wordclouds(self, **kwargs):                       return self.figures.stats_wordclouds(**kwargs)
    def theme_wordclouds(self, **kwargs):                       return self.figures.theme_wordclouds(**kwargs)
    def frequency_wordclouds(self, **kwargs):                   return self.figures.frequency_wordclouds(**kwargs)
    def neighbor_wordclouds(self, **kwargs):                   return self.figures.neighbor_wordclouds(**kwargs)

    # models -- one entry for every kind of saved model. this is why it lives
    # in no namespace: the whole point is not having to know which kind.
    def score_with_model(self, **kwargs):
        from .score_model import score_with_model
        return _forward(score_with_model, kwargs)

    def describe_model(self, **kwargs):
        from .helpers.model_spec import describe
        return _forward(describe, kwargs)

    def rename_model(self, **kwargs):
        from .helpers.model_spec import rename_model
        return _forward(rename_model, kwargs)

    # helpers
    def txt_folder_to_analysis_ready_csv(self, **kwargs):       return self.helpers.txt_folder_to_analysis_ready_csv(**kwargs)
    def csv_to_analysis_ready_csv(self, **kwargs):              return self.helpers.csv_to_analysis_ready_csv(**kwargs)
    def find_files(self, **kwargs):                             return self.helpers.find_files(**kwargs)
    def feature_gather(self, **kwargs):                         return self.helpers.feature_gather(**kwargs)
    def average_feature_table(self, **kwargs):                  return self.helpers.average_feature_table(**kwargs)
    




class _AudioAPI:
    def convert_to_wav(self, **kwargs):
        from .audio.convert_to_wav import convert_audio_to_wav
        return _forward(convert_audio_to_wav, kwargs)

    def extract_wavs_from_video(self, **kwargs):
        from .audio.extract_wav_from_video import split_audio_streams_to_wav
        return _forward(split_audio_streams_to_wav, kwargs)

    def split_wav_by_speaker(self, **kwargs):
        from .audio.split_wav_by_speaker import make_speaker_wavs_from_csv
        return _forward(make_speaker_wavs_from_csv, kwargs)

    def extract_whisper_embeddings(self, **kwargs):
        from .audio.extract_whisper_embeddings import extract_whisper_embeddings
        return _forward(extract_whisper_embeddings, kwargs)

    def transcribe_with_whisper(self, **kwargs):
        from .audio.transcribe_with_whisper import transcribe_with_whisper
        return _forward(transcribe_with_whisper, kwargs)

    def diarize_with_thirdparty(self, **kwargs):
        from .audio.diarizer.whisper_diar_wrapper import run_whisper_diarization_repo
        return _forward(run_whisper_diarization_repo, kwargs)
    
    def analyze_vocal_acoustics(self, **kwargs):
        from .audio.analyze_vocal_acoustics import analyze_acoustics
        return _forward(analyze_acoustics, kwargs)


class _TextAPI:
    def analyze_with_dictionaries(self, **kwargs):
        from .text.analyze_with_dictionaries import analyze_with_dictionaries
        return _forward(analyze_with_dictionaries, kwargs)

    def analyze_with_norms(self, **kwargs):
        from .text.analyze_with_norms import analyze_with_norms
        return _forward(analyze_with_norms, kwargs)

    def analyze_with_archetypes(self, **kwargs):
        from .text.analyze_with_archetypes import analyze_with_archetypes
        return _forward(analyze_with_archetypes, kwargs)
    
    def analyze_readability(self, **kwargs):
        from .text.analyze_readability import analyze_readability
        return _forward(analyze_readability, kwargs)

    def analyze_sentiment_vader(self, **kwargs):
        from .text.analyze_sentiment_vader import analyze_sentiment_vader
        return _forward(analyze_sentiment_vader, kwargs)

    def analyze_word_count(self, **kwargs):
        from .text.analyze_word_count import analyze_word_count
        return _forward(analyze_word_count, kwargs)

    def analyze_lexical_richness(self, **kwargs):
        from .text.analyze_lexical_richness import analyze_lexical_richness
        return _forward(analyze_lexical_richness, kwargs)

    def analyze_entropy(self, **kwargs):
        from .text.analyze_entropy import analyze_entropy
        return _forward(analyze_entropy, kwargs)

    def analyze_ngram_frequencies(self, **kwargs):
        from .text.analyze_ngram_frequencies import analyze_ngram_frequencies
        return _forward(analyze_ngram_frequencies, kwargs)

    def analyze_parts_of_speech(self, **kwargs):
        from .text.analyze_parts_of_speech import analyze_parts_of_speech
        return _forward(analyze_parts_of_speech, kwargs)

    def build_doc_term_matrix(self, **kwargs):
        from .text.build_doc_term_matrix import build_doc_term_matrix
        return _forward(build_doc_term_matrix, kwargs)

    def topic_model_mem(self, **kwargs):
        from .text.topic_model_mem import topic_model_mem
        return _forward(topic_model_mem, kwargs)

    def apply_mem_model(self, **kwargs):
        from .text.topic_model_mem import apply_mem_model
        return _forward(apply_mem_model, kwargs)

    def topic_model_lda(self, **kwargs):
        from .text.topic_model_lda import topic_model_lda
        return _forward(topic_model_lda, kwargs)

    def apply_lda_model(self, **kwargs):
        from .text.topic_model_lda import apply_lda_model
        return _forward(apply_lda_model, kwargs)

    def topic_model_nmf(self, **kwargs):
        from .text.topic_model_nmf import topic_model_nmf
        return _forward(topic_model_nmf, kwargs)

    def apply_nmf_model(self, **kwargs):
        from .text.topic_model_nmf import apply_nmf_model
        return _forward(apply_nmf_model, kwargs)

    def sweep_topic_count(self, **kwargs):
        from .text.topic_count_sweep import sweep_topic_count
        return _forward(sweep_topic_count, kwargs)

    def train_word_vectors(self, **kwargs):
        from .text.word_vectors import train_word_vectors
        return _forward(train_word_vectors, kwargs)

    def apply_word_vectors(self, **kwargs):
        from .text.word_vectors import apply_word_vectors
        return _forward(apply_word_vectors, kwargs)

    def import_word_vectors(self, **kwargs):
        from .text.word_vectors import import_word_vectors
        return _forward(import_word_vectors, kwargs)

    def describe_word_vectors(self, **kwargs):
        from .text.word_vectors import describe_word_vectors
        return _forward(describe_word_vectors, kwargs)

    def extract_transformer_embeddings(self, **kwargs):
        from .text.transformer_embeddings import extract_transformer_embeddings
        return _forward(extract_transformer_embeddings, kwargs)

    def adapt_encoder(self, **kwargs):
        from .text.adapt_encoder import adapt_encoder
        return _forward(adapt_encoder, kwargs)

    def pretrain_encoder(self, **kwargs):
        from .text.pretrain_encoder import pretrain_encoder
        return _forward(pretrain_encoder, kwargs)

    def finetune_text_predictor(self, **kwargs):
        from .text.finetune_predictor import finetune_text_predictor
        return _forward(finetune_text_predictor, kwargs)

    def import_hf_classifier(self, **kwargs):
        from .text.hf_classifier import import_hf_classifier
        return _forward(import_hf_classifier, kwargs)

    def apply_hf_classifier(self, **kwargs):
        from .text.hf_classifier import apply_hf_classifier
        return _forward(apply_hf_classifier, kwargs)

    def apply_text_predictor(self, **kwargs):
        from .text.finetune_predictor import apply_text_predictor
        return _forward(apply_text_predictor, kwargs)

    def analyze_cohesion(self, **kwargs):
        from .text.analyze_cohesion import analyze_cohesion
        return _forward(analyze_cohesion, kwargs)

    def extract_sentence_embeddings(self, **kwargs):
        from .text.extract_sentence_embeddings import extract_sentence_embeddings
        return _forward(extract_sentence_embeddings, kwargs)
    
    def convert_subtitles(self, **kwargs):
        from .text.subtitle_parser import convert_subtitles
        return _forward(convert_subtitles, kwargs)


class _StatsAPI:
    def assemble_analysis_table(self, **kwargs):
        from .stats.assemble import assemble_analysis_table
        return _forward(assemble_analysis_table, kwargs)

    def analyze_group_differences(self, **kwargs):
        from .stats.group_differences import analyze_group_differences
        return _forward(analyze_group_differences, kwargs)

    def analyze_correlations(self, **kwargs):
        from .stats.correlations import analyze_correlations
        return _forward(analyze_correlations, kwargs)

    def write_stats_report(self, **kwargs):
        from .stats.report import write_stats_report
        return _forward(write_stats_report, kwargs)

    def fit_ridge_csv(self, **kwargs):
        from .stats.ridge import fit_ridge_csv
        return _forward(fit_ridge_csv, kwargs)

    def apply_ridge_csv(self, **kwargs):
        from .stats.ridge import apply_ridge_csv
        return _forward(apply_ridge_csv, kwargs)

    def fit_classifier_csv(self, **kwargs):
        from .stats.classify import fit_classifier_csv
        return _forward(fit_classifier_csv, kwargs)

    def apply_classifier_csv(self, **kwargs):
        from .stats.classify import apply_classifier_csv
        return _forward(apply_classifier_csv, kwargs)

    def fit_pca_csv(self, **kwargs):
        from .stats.pca import fit_pca_csv
        return _forward(fit_pca_csv, kwargs)

    def apply_pca_csv(self, **kwargs):
        from .stats.pca import apply_pca_csv
        return _forward(apply_pca_csv, kwargs)

    def describe_features(self, **kwargs):
        from .stats.describe import describe_features
        return _forward(describe_features, kwargs)


class _FiguresAPI:
    def stats_wordclouds(self, **kwargs):
        from .figures.wordclouds import stats_wordclouds
        return _forward(stats_wordclouds, kwargs)

    def theme_wordclouds(self, **kwargs):
        from .figures.wordclouds import theme_wordclouds
        return _forward(theme_wordclouds, kwargs)

    def frequency_wordclouds(self, **kwargs):
        from .figures.wordclouds import frequency_wordclouds
        return _forward(frequency_wordclouds, kwargs)

    def neighbor_wordclouds(self, **kwargs):
        from .figures.wordclouds import neighbor_wordclouds
        return _forward(neighbor_wordclouds, kwargs)


class _HelpersAPI:
    
    def txt_folder_to_analysis_ready_csv(self, **kwargs):
        from .helpers.text_gather import txt_folder_to_analysis_ready_csv
        return _forward(txt_folder_to_analysis_ready_csv, kwargs)
    
    def csv_to_analysis_ready_csv(self, **kwargs):
        from .helpers.text_gather import csv_to_analysis_ready_csv
        return _forward(csv_to_analysis_ready_csv, kwargs)
    
    def find_files(self, **kwargs):
        from .helpers.find_files import find_files
        return _forward(find_files, kwargs)
    
    def feature_gather(self, **kwargs):
        from .helpers.feature_gather import feature_gather
        return _forward(feature_gather, kwargs)

    def average_feature_table(self, **kwargs):
        from .helpers.feature_average import average_feature_table
        return _forward(average_feature_table, kwargs)
    
