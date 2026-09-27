# Analyzing Video

Taters has always taken video in and thrown the pictures away. It pulled the
audio out, transcribed it, and everything after that was text. This part
measures what was actually on screen.

Three things, and none of them costs you an install. Everything here runs on
software that was already in the box — ffmpeg, which Taters has always
required, and the ONNX and transformer machinery that arrives with the audio and
text sides. There is no extra to install and nothing new to resolve.

---

## First, the thing that isn't here

**There are no FACS action units.** That's the obvious thing to want, and I went
looking for it first. Here's why it didn't make the cut.

Every open implementation is either licensed in a way that stops us shipping it,
or drags a dependency that would break the install you already have. OpenFace
3.0's license says the software "may be used for your own noncommercial internal
research purposes" and forbids redistribution. libreface is under a USC research
license *and* pins `torch==2.0.0`, `opencv-python==4.10.0.84`, dlib and cmake —
installing it would wreck a working environment. Py-Feat is MIT and genuinely
good, but its flagship multitask model is "non-commercial research only" too,
and the package pulls `torchvision`, which pins an exact torch version and would
happily drag your GPU build out from under you.

I could have approximated AUs from face-mesh blendshapes and called it FACS.
Plenty of tools do. But blendshapes are ARKit animation coefficients, they are
not validated against FACS coding, and putting that label on them would be the
kind of plausible-looking wrong number this whole codebase keeps trying to avoid.

So: no AUs. If that's what you need, OpenFace or Py-Feat run perfectly well
beside Taters — analyze your video with them and bring the results in as a
spreadsheet.

---

## Shot structure and pacing

The structural half: not what is in the picture, but how the picture behaves.

**Average shot length** is the headline. It has a real measurement tradition
behind it — the cinemetrics literature, Barry Salt's shot-length statistics,
Cutting and colleagues on the pacing of Hollywood film — and it's the covariate
anyone studying edited media reaches for first. If you're claiming something
about *content*, you usually want to have held pacing constant before you say it.

Beside it sits something less common and, I think, more interesting. ffmpeg
scores every frame for how different it is from the last, and instead of asking
for a list of cuts we keep the whole distribution. That's a measure of **visual
volatility**: a locked-off interview and a handheld chase can have identical
shot lengths and nothing else in common.

You also get black and frozen stretches — leaders, transitions, static slides, a
webcam that stalled.

!!! warning "One honest limit"
    The scene detector works on **luminance**. A cut between two shots of
    similar brightness scores low, and a cut from a red screen to a green one at
    matched brightness is invisible to it. On real footage that's rarely the
    situation; on synthetic or heavily graded material it can be. The per-frame
    scores are summarized in the output, so you can see where yours sit before
    trusting the cut count.

---

## Faces

Who is on screen, how big they are in frame, which way their head is tilted, and
an eight-way emotion distribution per face.

Three small models do the work, and all three are licensed for any use,
commercial included — which is why they were chosen over better-known
alternatives:

| what | model | license |
|---|---|---|
| finding faces | YuNet (OpenCV Zoo) | MIT |
| who a face is | SFace (OpenCV Zoo) | Apache-2.0 |
| what a face is doing | HSEmotion / EmotiEffLib | Apache-2.0 |

HSEmotion is an EfficientNet-B0 trained on AffectNet, and it's about as well
validated as open facial emotion gets: state of the art on the EmotiW 2019 and
2020 challenges and the ABAW CVPR/ECCV 2022 ones, and first place in Expression
Recognition at the eighth ABAW competition.

!!! warning "What this measures, and what it doesn't"
    Emotion recognition from faces is contested, and you should say so in your
    methods section. These models are trained on posed or crowd-labeled images.
    The mapping from a facial configuration to a felt emotion is **not**
    one-to-one — a smile is not happiness, it's a smile. Barrett et al. (2019)
    is the standard reference for why.

    What this measures is **facial appearance**. Treat it as that, and it's
    useful. Treat it as an emotion readout and you'll publish something you'll
    regret.

### Head pose

You get **roll** — head tilt, from the angle between the eyes. That's honest
geometry and it's labeled as such. You do not get calibrated yaw and pitch,
because those need a fitted 3D model and the estimate you'd get without one
looks precise and isn't.

---

## Who is who

If you ask for per-face results, Taters works out which faces are the same
person. This is the part with real error in it, so it's worth understanding.

The naive approach — embed every face, cluster the lot — scatters one person
across a dozen identities, because a single face varies more across poses and
lighting than two similar-looking faces vary from each other. What the video
face clustering literature actually does, and what this does, is three stages:

1. **Track within shots.** Faces are linked frame to frame by overlap and
   similarity. A track never crosses a cut — a new camera angle looks like
   smooth motion to a box tracker, and that's where naive tracking fails. The
   shot boundaries come from the same scene detection described above.
2. **One embedding per track**, weighted by detection quality — confidence, face
   size, sharpness. A blurred, tiny, half-turned face shouldn't be what decides
   who somebody is.
3. **Cluster the tracks**, with one hard rule: **two faces visible in the same
   frame are two people**, whatever their embeddings say. That rule is certain
   rather than probabilistic, and it rules out exactly the merge a similarity
   threshold gets wrong.

### Telling it how many people there are

Two ways, and the first is better whenever you can use it.

**Give it a count.** A dyadic interview is two. A therapy session is two. A
focus group is six. If you know, say so — it's far more reliable than any
threshold.

Taters will refuse a count smaller than the number of faces it can see together
in a single frame, and tell you which frame, because that request is impossible
rather than merely unwise.

**Or give it a threshold.** How different two faces must look to be different
people, as a cosine distance. Lower splits one person into several; higher
merges two people into one. For calibration: the same face at different scales
sits around 0.00–0.02 apart, the same face tilted or mirrored around 0.12–0.17,
and two genuinely different people considerably further. The default of 0.45
sits in that gap.

!!! note "Verifying the identities is your job"
    This is the honest position and I'd rather state it than bury it. Taters
    reports how many people it found, how many frames each was seen in, and —
    importantly — the **margin**, which is how close a call each assignment was.
    A margin near your threshold means that identity is a coin flip.

    Those numbers exist to make checking quick, not to make it unnecessary.
    Nobody should publish per-person results without spot-checking that the
    clusters are the people they think they are.

---

## Video embeddings

Each video becomes a vector, the same way the text side turns a document into
one. Good for similarity, clustering, and as predictors in a model. Not good for
explaining anything — a CLIP vector will tell you two videos look alike and
won't tell you what about them.

Any Hugging Face vision encoder works. The default is small, fast on CPU, and
the one most other work uses, which makes your numbers comparable to theirs.

---

## Coverage, and why the counts matter

Every mean comes with the number of frames it was taken over, and those counts
are marked as bookkeeping — you can filter on them, but the statistics stage
will never let them become predictors.

That matters more than it sounds. A mean happiness across the 11% of frames
where a face was visible is a completely different quantity from one across 95%,
and nothing else in the output would tell them apart. The same goes the other
way: a model that discovered your outcome correlates with frame width, or with
how many frames had a face in them, has discovered how you collected your
videos.

A video with no detectable face gets **empty cells, not zeros**. It doesn't have
a mean emotion of nought; it doesn't have one at all.

## Where to go next

- [Analyzing audio](analyzing-audio.md) — the other half of a media file
- [Running analyses](running-analyses.md) — what to do with the table
- [The Taters app](wizard.md) — all of this without writing code
