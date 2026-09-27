"""
Video feature extraction.

Taters has always taken video in and thrown the pictures away -- the media path
runs ``ffmpeg -vn``, keeps the audio, and everything downstream is a transcript.
This package measures what was on screen.

Nothing here needs a dependency that was not already installed. The structural
measures come out of ffmpeg filters that ship with ffmpeg itself, which Taters
already requires; the face and embedding work runs on ``onnxruntime`` and
``transformers``, both of which arrive with the audio and text stacks.

There are deliberately **no action units**. Every open FACS implementation is
either licensed for non-commercial research only (OpenFace 3.0, libreface,
Py-Feat's own multitask model) or drags a torch-version-locked dependency that
would fight the install people already have. Approximating AUs from something
else and calling it FACS would be worse than not shipping it.
"""
