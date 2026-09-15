# Third-party licenses

Taters is MIT-licensed (see `LICENSE`). Three components inside the package are
someone else's work under their own terms, reproduced here so that anyone
redistributing Taters can see them without opening the source.

## happierfuntokenizing

`src/taters/text/happierfuntokenizing.py` — the tokenizer behind the n-gram
frequency list, the document-term matrix and the parts-of-speech step.

* Original: Christopher Potts, "Happy Fun Tokenizer" (2011).
  Updated by H. Andrew Schwartz and Maarten Sap for DLATK.
* License: **Creative Commons Attribution-NonCommercial-ShareAlike 3.0
  Unported** (CC BY-NC-SA 3.0),
  <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
* Modifications by Taters: the `&amp;` entity replacement was moved out of
  the loop over the other entities so it runs once regardless of what else
  the text contains.

The NonCommercial and ShareAlike terms apply to this file. If you need the
n-gram, document-term-matrix or parts-of-speech features under different
terms, this is the file to replace.

## whisper-diarization

`src/taters/audio/diarizer/whisper-diarization/` — the speaker-diarization
scripts the `diarize` step runs in a subprocess.

* Author: Mahmoud Ashraf, <https://github.com/MahmoudAshraf97/whisper-diarization>.
* License: **BSD 2-Clause**; the full text ships beside the code as
  `whisper-diarization/LICENSE`.
* Modifications by Taters: `diarize_custom.py` and the files under
  `diarization/msdd/` adapt the pipeline to Taters' inputs and outputs; the
  upstream files are otherwise unchanged.

## DejaVu Sans Bold

`src/taters/figures/fonts/DejaVuSans-Bold.ttf` — the typeface the word
clouds are drawn in.

* Authors: Bitstream, Inc. (Bitstream Vera) and the DejaVu contributors,
  <https://dejavu-fonts.github.io/>.
* License: **Bitstream Vera Fonts License** (permissive; DejaVu's changes are
  in the public domain). The full text ships beside the font as
  `figures/fonts/LICENSE`.
* Modifications by Taters: none.
