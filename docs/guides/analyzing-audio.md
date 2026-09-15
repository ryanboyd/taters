# Analyzing Audio

The spoken word is where much of social life actually happens — interviews,
conversations, therapy sessions, meetings, media — and recordings of it carry
two layers of information at once. There is *what* was said, which you can
transcribe and then analyze with everything in the [text guide](analyzing-text.md).
And there is *how* it was said — pitch, pace, pauses, voice quality — which
lives in the sound itself and is thrown away the moment you keep only the
transcript. The audio side of Taters exists to get you both layers without a
degree in signal processing: recordings go in, transcripts and features come
out.

A general orientation to voice and speech as psychological data:

* Scherer, K. R. (1986). Vocal affect expression: A review and a model for future research. *Psychological Bulletin, 99*(2), 143–165. [https://doi.org/10.1037/0033-2909.99.2.143](https://doi.org/10.1037/0033-2909.99.2.143)

* Weidman, A. (2014). Anthropology and voice. *Annual Review of Anthropology, 43*, 37–51. [https://doi.org/10.1146/annurev-anthro-102313-030050](https://doi.org/10.1146/annurev-anthro-102313-030050)


---

## First, the unglamorous part: getting clean audio

Real recordings arrive as a zoo — video containers with multiple audio
tracks, phone memos, compressed formats of every vintage. Before anything
scientific happens, they get standardized into plain WAV files (16 kHz,
16-bit, mono is the lingua franca of speech models). This is "plumbing," not
method, but it is the plumbing that decides whether everything downstream
works.

**In Taters:** this happens automatically inside any pipeline that starts
from audio or video; the standalone converters are there if you want them.

---

## Transcription (speech-to-text)

Automatic speech recognition has crossed a threshold in the last few years:
models like OpenAI's Whisper transcribe clean recorded speech at close to
human accuracy, in dozens of languages, on ordinary hardware. For research,
this means transcription — long the most expensive step in working with
spoken data — is now nearly free.

Two things every researcher should know before trusting a machine
transcript, though. First, accuracy is not uniform: error rates climb with
background noise, crosstalk, strong accents, and clinical or child speech —
and those errors are *structured*, not random, so they can correlate with
exactly the group differences you study. Second, ASR models can
"hallucinate": on silence or noise they sometimes emit fluent text that
nobody said. Spot-check transcripts against audio, especially in the
populations you care about.

* Radford, A., Kim, J. W., Xu, T., Brockman, G., Mcleavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning*, 28492–28518. [https://proceedings.mlr.press/v202/radford23a.html](https://proceedings.mlr.press/v202/radford23a.html)


* Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. [https://doi.org/10.1073/pnas.1915768117](https://doi.org/10.1073/pnas.1915768117)


* Koenecke, A., Choi, A. S. G., Mei, K. X., Schellmann, H., & Sloane, M. (2024). Careless Whisper: Speech-to-text hallucination harms. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency*, FAccT ’24, 1672–1681. [https://doi.org/10.1145/3630106.3658996](https://doi.org/10.1145/3630106.3658996)


**In Taters:** the transcription step runs Whisper locally (your data never
leaves your machine) and writes a timestamped transcript CSV that every text
step can consume directly. The one-speaker transcript can also **translate**:
turn `translate` on under its settings and the transcript comes out in
English whatever language was spoken, using the same models (they were
trained to do this as a second task). It needs a multilingual model, so the
app moves you off an English-only `.en` model when you turn it on. The
diarized transcript cannot translate, because the diarizer aligns each word
to the audio and a translation has no words to align.

---

## Speaker diarization (who spoke when)

A transcript of a conversation is only half the story if you cannot tell
the voices apart. Diarization is the task of segmenting a recording by
speaker — no enrollment, no voice samples, just clustering "voices that
sound alike" — so that every stretch of speech gets a speaker label. Once
you have that, a whole family of questions opens up: per-person language
measures, turn-taking dynamics, who dominates the floor, how a dyad's
language converges over a session.

It is also the most error-prone step in the chain. Overlapping speech,
similar voices, and short back-channels ("mm-hm") all confuse it, and
diarization errors propagate into every per-speaker measure you compute
afterward. Treat speaker counts and labels as estimates worth verifying,
not ground truth.

* Park, T. J., Kanda, N., Dimitriadis, D., Han, K. J., Watanabe, S., & Narayanan, S. (2022). A review of speaker diarization: Recent advances with deep learning. *Computer Speech & Language, 72*, 101317. [https://doi.org/10.1016/j.csl.2021.101317](https://doi.org/10.1016/j.csl.2021.101317)


**In Taters:** two roads to a transcript, writing the *same* output format
so everything downstream is identical. Pick by your question:

| | Transcribe | Diarize + transcribe |
| --- | --- | --- |
| **Speakers** | one (or "don't care") | several, clustered automatically |
| **Right for** | lectures, monologues, dictation | interviews, conversations, groups |
| **Cost** | fast, base install | slower, extra installs |


At the time of this writing, Taters uses this repo for diarized transcription:
[https://github.com/MahmoudAshraf97/whisper-diarization/](https://github.com/MahmoudAshraf97/whisper-diarization/)

Why this repo? Well, honestly, because [people who know a lot more than I do](https://adithya8.github.io/) said that it has given them the best results.

---

## Speech embeddings

The same neural networks that transcribe speech build, along the way, a
rich numerical summary of each stretch of audio — a vector that encodes not
just the words but accent, affect, recording conditions, speaker
characteristics. Exported as features, these embeddings are the audio
counterpart of the sentence embeddings in the text guide: opaque
individually, but powerful as inputs to clustering, similarity, and
prediction — and they capture paralinguistic information no transcript
retains.

* Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems, 33*, 12449–12460. [https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)


**In Taters:** the Whisper-embeddings step exports one vector per transcript
segment (or per fixed window of raw audio), ready for the feature-gathering
and PCA tools.

---

## Vocal acoustics (prosody and voice quality)

This is the classic, interpretable layer of "how it was said":
fundamental frequency (pitch) and its variability, formants, loudness,
harshness and breathiness measures (jitter, shimmer, harmonics-to-noise
ratio), speech rate, and pauses. Decades of work tie these to arousal and
affect, to social perception (dominance, warmth), and to clinical states —
depressed speech, for example, reliably shows reduced pitch variability and
longer pauses. The measures come from Praat, the same tool phoneticians
have used for decades, so they connect directly to that literature.

* Scherer, K. R. (1986). Vocal affect expression: A review and a model for future research. *Psychological Bulletin, 99*(2), 143–165. [https://doi.org/10.1037/0033-2909.99.2.143](https://doi.org/10.1037/0033-2909.99.2.143)

* Cummins, N., Scherer, S., Krajewski, J., Schnieder, S., Epps, J., & Quatieri, T. F. (2015). A review of depression and suicide risk assessment using speech analysis. *Speech Communication, 71*, 10–49. [https://doi.org/10.1016/j.specom.2015.03.004](https://doi.org/10.1016/j.specom.2015.03.004)

* Boersma, P., & Weenink, D. *Praat: Doing phonetics by computer*. [praat.org](https://www.praat.org)

!!! warning "Treat these features with caution"
    Acoustics is not my area of expertise. This module is newer and less
    battle-tested than the rest of Taters; validate its output against a
    known reference before relying on it.

**In Taters:** the vocal-acoustics step analyzes a whole recording or,
given a transcript, each speaker's turns — summaries per file or per
speaker, framewise tracks if you want them.

---

## Per-speaker audio

A small workflow idea that earns its keep: once a recording is diarized,
you can reassemble one audio file *per speaker* — everything person A said,
spliced together. That is what makes speaker-level acoustic and embedding
features possible from a single shared microphone, and it gives human
coders clean per-person audio to listen to.

**In Taters:** the split-by-speaker step, fed by any diarized transcript.

---

## The shape of a typical study

Most audio projects in Taters reduce to the same arc: standardize the
recordings, transcribe (diarized if multiple voices matter), then fan out —
text features from the transcript via everything in the
[text guide](analyzing-text.md), acoustic and embedding features from the
sound, aggregated per file or per speaker. The app (`taters`) builds that
whole chain from a handful of questions; the [pipelines guide](pipelines.md)
shows what it looks like written down.
