# taters/Taters.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import inspect

def _abs(p: str | Path) -> str:
    return str(Path(p).resolve())

def _forward(func, kwargs: dict[str, Any]):
    sig = inspect.signature(func)
    try:
        # validate names & requireds; don't execute defaults here
        sig.bind_partial(**kwargs)
    except TypeError as e:
        allowed = ", ".join([str(p) for p in sig.parameters.values()])
        raise TypeError(f"{func.__module__}.{func.__name__}: {e}\nAllowed params: {allowed}")
    return func(**kwargs)


class Taters:
    def __init__(self):
        self.audio = _AudioAPI(self)
        self.text = _TextAPI(self)
        self.helpers = _HelpersAPI(self)

    # back-compat pass-throughs
    
    # audio
    def convert_to_wav(self, **kwargs):                         return self.audio.convert_to_wav(**kwargs)
    def extract_wavs_from_video(self, **kwargs):                return self.audio.extract_wavs_from_video(**kwargs)
    def split_wav_by_speaker(self, **kwargs):                   return self.audio.split_wav_by_speaker(**kwargs)
    def extract_whisper_embeddings(self, **kwargs):             return self.audio.extract_whisper_embeddings(**kwargs)
    def diarize_with_thirdparty(self, **kwargs):                return self.audio.diarize_with_thirdparty(**kwargs)
    
    #text
    def analyze_with_dictionaries(self, **kwargs):              return self.text.analyze_with_dictionaries(**kwargs)
    def analyze_with_archetypes(self, **kwargs):                return self.text.analyze_with_archetypes(**kwargs)
    def analyze_readability(self, **kwargs):                    return self.text.analyze_readability(**kwargs)
    def analyze_lexical_richness(self, **kwargs):               return self.text.analyze_lexical_richness(**kwargs)
    def extract_sentence_embeddings(self, **kwargs):            return self.text.extract_sentence_embeddings(**kwargs)
    def convert_subtitles(self, **kwargs):                      return self.text.convert_subtitles(**kwargs)
    
    # helpers
    def txt_folder_to_analysis_ready_csv(self, **kwargs):       return self.helpers.txt_folder_to_analysis_ready_csv(**kwargs)
    def csv_to_analysis_ready_csv(self, **kwargs):              return self.helpers.csv_to_analysis_ready_csv(**kwargs)
    def find_files(self, **kwargs):                             return self.helpers.find_files(**kwargs)
    def feature_gather(self, **kwargs):                         return self.helpers.feature_gather(**kwargs)
    




class _AudioAPI:
    def __init__(self, parent: Taters): self._cs = parent

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

    def diarize_with_thirdparty(self, **kwargs):
        from .audio.diarizer.whisper_diar_wrapper import run_whisper_diarization_repo
        return _forward(run_whisper_diarization_repo, kwargs)


class _TextAPI:
    def __init__(self, parent: Taters): self._cs = parent

    def analyze_with_dictionaries(self, **kwargs):
        from .text.analyze_with_dictionaries import analyze_with_dictionaries
        return _forward(analyze_with_dictionaries, kwargs)

    def analyze_with_archetypes(self, **kwargs):
        from .text.analyze_with_archetypes import analyze_with_archetypes
        return _forward(analyze_with_archetypes, kwargs)
    
    def analyze_readability(self, **kwargs):
        from .text.analyze_readability import analyze_readability
        return _forward(analyze_readability, kwargs)

    def analyze_lexical_richness(self, **kwargs):
        from .text.analyze_lexical_richness import analyze_lexical_richness
        return _forward(analyze_lexical_richness, kwargs)

    def extract_sentence_embeddings(self, **kwargs):
        from .text.extract_sentence_embeddings import analyze_with_sentence_embeddings
        return _forward(analyze_with_sentence_embeddings, kwargs)
    
    def convert_subtitles(self, **kwargs):
        from .text.subtitle_parser import convert_subtitles
        return _forward(convert_subtitles, kwargs)


class _HelpersAPI:
    def __init__(self, parent: Taters): self._cs = parent
    
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
    
