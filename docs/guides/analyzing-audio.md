# Analyzing Audio

A lot of what people do with language, they do out loud. Sometimes, *way* too much
out loud. Interviews, conversations, therapy sessions, meetings, podcasts.
A recording of any of those things carries two things at once: *what* was said,
and *how* it was vocalized.

The first one you can transcribe, then run through everything in the
[text guide](analyzing-text.md). The second one — pitch, pace, pauses, voice
quality — only exists in the sound itself, and it is gone the moment you keep
the transcript and throw the audio away. So, this side of Taters is here to get
you both. Recordings in, transcripts and features out.

If you want a sense of why anybody treats the voice as data in the first place,
start here:

* Scherer, K. R. (1986). Vocal affect expression: A review and a model for future research. *Psychological Bulletin, 99*(2), 143–165. [https://doi.org/10.1037/0033-2909.99.2.143](https://doi.org/10.1037/0033-2909.99.2.143)

* Weidman, A. (2014). Anthropology and voice. *Annual Review of Anthropology, 43*, 37–51. [https://doi.org/10.1146/annurev-anthro-102313-030050](https://doi.org/10.1146/annurev-anthro-102313-030050)


---

## First, the boring part: getting clean audio

Real recordings show up in a hundred different shapes. Video files with three
audio tracks, phone memos, compressed formats of every vintage. Remember RealAudio?
Or ATRAC? Yikes. Speech models are picky and nearly all of them want the same thing,
so that's what everything gets turned into first: plain WAV, 16 kHz, 16-bit, mono.
This isn't science so much as basic housekeeping, but it's important that everything
is standardized so that the science and engineering parts of what we do here will
actually work.

**In Taters:** this happens on its own inside any pipeline that starts from
audio or video. The standalone converters are there if you'd rather do it
yourself.

---

## Transcription (speech-to-text)

Speech recognition got good *real* fast. Not "good for a computer" good — models like
OpenAI's Whisper transcribe clean, recorded speech at close to human accuracy,
in dozens of languages, on ordinary hardware. Transcription used to be the most
expensive part of working with spoken data, and now it's basically free. If you
did any transcription (automated or not) even like 5-6 years ago, you'll know just
how awful it was not that long ago.

Two things to know before you trust a machine transcript, though.

First, the accuracy isn't the same for everybody. Error rates climb with
background noise, crosstalk, strong accents, and clinical or child speech. And
those errors aren't random — they're *structured*, which means they can line up
with exactly the group differences you were hoping to study. You've been warned.

Second, these models hallucinate. Handed silence or noise, they will sometimes
produce perfectly fluent text that nobody said. So spot-check transcripts
against the audio, especially for the people you actually care about measuring.

* Radford, A., Kim, J. W., Xu, T., Brockman, G., Mcleavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning*, 28492–28518. [https://proceedings.mlr.press/v202/radford23a.html](https://proceedings.mlr.press/v202/radford23a.html)


* Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. [https://doi.org/10.1073/pnas.1915768117](https://doi.org/10.1073/pnas.1915768117)


* Koenecke, A., Choi, A. S. G., Mei, K. X., Schellmann, H., & Sloane, M. (2024). Careless Whisper: Speech-to-text hallucination harms. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency*, FAccT ’24, 1672–1681. [https://doi.org/10.1145/3630106.3658996](https://doi.org/10.1145/3630106.3658996)


**In Taters:** the transcription step runs Whisper locally — your data never
leaves your machine — and writes a timestamped transcript CSV that every text
module in Taters can read directly.

The one-speaker transcript can also **translate**: turn `translate` on under its
settings and the transcript comes out in English whatever language was spoken,
using the same models (they were trained to do this as a second job). It needs a
multilingual model, so the app moves you off an English-only `.en` model when you
turn it on. The diarized transcript can't translate — the diarizer lines each
word up against the audio, and a translation has no words to line up.

---

## Speaker diarization (who spoke when)

If you care about measuring each *person* in a conversation, a transcript of That
conversation isn't much use if you can't tell the voices apart. Diarization is the
task of splitting up a recording up by speaker — no enrollment, no voice samples,
just clustering the stretches that sound like the same person, resulting in every bit
of speech getting assigned to a person. Once you have that, you can do per-person
language measures, turn-taking, who's hogging the floor, whether a pair's language 
converges over a session, and so on.

It's also the flakiest step in the whole chain. Overlapping speech, similar
voices and short back-channels ("mm-hm") can all trip it up, and a diarization error
doesn't stay put — it contaminates every per-speaker measure you compute
afterward. So treat the speaker count and the labels as estimates a "best guess" that
you should definitely double-check, and not as ground truth.

* Park, T. J., Kanda, N., Dimitriadis, D., Han, K. J., Watanabe, S., & Narayanan, S. (2022). A review of speaker diarization: Recent advances with deep learning. *Computer Speech & Language, 72*, 101317. [https://doi.org/10.1016/j.csl.2021.101317](https://doi.org/10.1016/j.csl.2021.101317)


**In Taters:** there are two paths to a transcript, and they write the *same*
output format, so everything downstream is identical either way. Pick by your
question:

| | Transcribe | Diarize + transcribe |
| --- | --- | --- |
| **Speakers** | one (or "don't care") | several, clustered automatically |
| **Right for** | lectures, monologues, dictation | interviews, conversations, groups |
| **Cost** | fast, base install | slower, extra installs |


At the time of this writing, Taters uses this repo for diarized transcription:
[https://github.com/MahmoudAshraf97/whisper-diarization/](https://github.com/MahmoudAshraf97/whisper-diarization/)

Why this repo? Well, honestly, because [people who know a lot more than I do](https://adithya8.github.io/) said
that it has given them the best results. Your mileage may vary.

---

## Speech embeddings

The same models that do the transcribing build a numeric/geometric summary of each
stretch of audio along the way. That vector holds more than the words — accent,
affect, recording conditions, whatever is distinctive about the speaker.

Pulled out as features, these are the audio version of the sentence embeddings
in the text guide: not much use one at a time, quite useful as input to
clustering, similarity, and prediction. And they keep the paralinguistic stuff
that a transcript never had in the first place.

* Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems, 33*, 12449–12460. [https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)

* Rao, R., V Ganesan, A., Kjell, O., Luby, J., Raghavan, A., Feltman, S. M., Ringwald, W., Boyd, R. L., Luft, B. J., Ruggero, C. J., Ryant, N., Kotov, R., & Schwartz, H. (2025). WhiSPA: Semantically and psychologically aligned Whisper with self-supervised contrastive and student-teacher learning. In W. Che, J. Nabende, E. Shutova, & M. T. Pilehvar (Eds.), *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 22529–22544). Association for Computational Linguistics. [https://aclanthology.org/2025.acl-long.1098/](https://aclanthology.org/2025.acl-long.1098/)


**In Taters:** the Whisper-embeddings step writes one vector per transcript
segment, or per fixed window of raw audio, ready for the feature-gathering and
PCA tools.

---

## Vocal acoustics (prosody and voice quality)

This is the old, interpretable layer of "how it was said." Pitch (fundamental
frequency) and how much it moves around, formants, loudness, the harshness and
breathiness measures (jitter, shimmer, harmonics-to-noise ratio), speech rate,
and pauses.

There are decades of work tying these to arousal and affect, to social
perception like dominance and warmth, and to clinical states — depressed
speech, for example, pretty reliably shows less pitch variability and longer
pauses. The measures come out of Praat, which is what phoneticians have been
using for ages, so they connect straight to that literature. That's "phoneticians,"
not "Phoenicians," although there's a possibility that there is some overlap
between the two groups. I don't know, I'm not a historian.

* Scherer, K. R. (1986). Vocal affect expression: A review and a model for future research. *Psychological Bulletin, 99*(2), 143–165. [https://doi.org/10.1037/0033-2909.99.2.143](https://doi.org/10.1037/0033-2909.99.2.143)

* Cummins, N., Scherer, S., Krajewski, J., Schnieder, S., Epps, J., & Quatieri, T. F. (2015). A review of depression and suicide risk assessment using speech analysis. *Speech Communication, 71*, 10–49. [https://doi.org/10.1016/j.specom.2015.03.004](https://doi.org/10.1016/j.specom.2015.03.004)

* Boersma, P., & Weenink, D. *Praat: Doing phonetics by computer*. [praat.org](https://www.praat.org)

!!! warning "Treat these features with caution"
    Acoustics is not my area of expertise. This module is newer and less
    battle-tested than the rest of Taters; validate its output against a
    known reference before relying on it.

**In Taters:** the vocal-acoustics step will do a whole recording, or, given a
transcript, each speaker's turns on their own — summaries per file or per
speaker, plus framewise tracks if you want them.

---

## Per-speaker audio

Here's a small thing that hopefully pays for itself. Once a recording is diarized, you can
put it back together as one audio file *per speaker* — everything person A said,
spliced end to end. That's what makes speaker-level acoustic and embedding
features possible when everybody shared one microphone, and it gives human
coders clean per-person audio to listen to.

**In Taters:** the split-by-speaker step, fed by any diarized transcript.

---

## The design of a typical study

Most audio projects in Taters end up looking the same. Standardize the
recordings, transcribe them (diarized, if it matters who said what), and then go
two directions at once: text features from the transcript, using anything in the
[text guide](analyzing-text.md), and acoustic and embedding features from the
sound, summarized per file or per speaker.

The app (`taters`) will build that whole chain out of a handful of questions.
The [pipelines guide](pipelines.md) shows what it looks like written down.
