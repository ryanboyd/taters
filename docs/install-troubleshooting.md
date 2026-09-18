# Installation troubleshooting

Everything that does not belong on the [install page](install-guide.md).
Nothing here is required reading — come looking when something breaks.

## Speaker diarization

Diarization works out *who spoke when*. It is the most involved thing to
install, because it pulls in NVIDIA's NeMo toolkit.

**You need Python 3.13 or older.** On 3.14 its dependencies do not exist yet.
`pip install taters[all]` still succeeds there and installs everything else;
**Check my setup** will show `nemo-toolkit · not available for Python 3.14`.

```bash
pip install taters[diarization]
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

The three GitHub packages are not on PyPI, so `pip install taters[diarization]`
cannot fetch them for you.

??? note "Why the Python limit, and why not to work around it?"

    Every NeMo release TATERS can use requires
    `nv-one-logger-pytorch-lightning-integration`, which is published for
    Python 3.13 and below. On 3.14 there is no version to install.

    Pinning an older NeMo yourself moves the failure rather than fixing it:
    NeMo 2.6 and below need `numpy<2`, which also has no 3.14 release, so pip
    tries to *build* numpy from source and stops on a missing C compiler. The
    error names numpy, which is a bystander.

    Taters also caps NeMo below 3.0, because 3.0 removed the speaker-clustering
    model the bundled code uses. That cap is upstream's too: whisper-diarization
    pins `nemo_toolkit[asr] >=2.5.0, <3` in its own requirements.

## Glottal features

`pip install taters[glottal]` adds three vocal-acoustics measures — HRF, NAQ
and OQ. Acoustics works fine without it; those three are skipped, with a
warning.

It is separate, and left out of `taters[all]`, because one of its dependencies
ships no pre-built packages, so pip has to compile things. Without Visual
Studio Build Tools (Windows) or a working compiler (macOS/Linux) it fails — and
it can fail with an error naming some unrelated package, because pip finds the
problem while working out versions for everything else.

Skip it unless you need those three measures.

## Training word vectors

Training word2vec or fastText models needs `gensim`
(`pip install taters[vectors]`, also part of `taters[standard]` and
`taters[all]`). Applying a saved word-vector model needs nothing extra.

**You need Python 3.13 or older.** gensim has no pre-built packages for 3.14
yet, so asking for it there makes pip try to compile it, and the install stops
with `Microsoft Visual C++ 14.0 or greater is required` (or a missing
compiler on macOS/Linux) after several hundred lines of output. Taters does
not request gensim on 3.14: the install succeeds, the training step is grayed
out on the checklist with the reason, and **Check my setup** shows
`gensim · not available for Python 3.14`. Installing the build tools is not
the fix; a 3.13 (or older) environment is.

## GPU problems

### It says my GPU works, but transcription is slow

Open the app, choose **Check my setup**, and look at the last table. It has two
rows on purpose:

```
│ PyTorch can use the GPU │ yes │ 1 device(s): NVIDIA RTX ...            │
│ Transcription on GPU    │ no  │ the card is visible but the CUDA maths │
│                         │     │ libraries are not loadable ...         │
```

If the second row says no while the first says yes, the report tells you the
command to fix it — usually reinstalling PyTorch on `cu128`.

??? note "Why transcription is fussier than everything else"

    Two separate pieces of software want your GPU, and they do not share
    libraries.

    PyTorch handles embeddings and diarization, and brings its own copy of
    NVIDIA's math libraries. Transcription instead runs on **CTranslate2**,
    which is built against CUDA 12 and looks for CUDA 12's copies. Install
    PyTorch on CUDA 13 and it brings CUDA 13's, which CTranslate2 will not load.

    Two things follow, and both surprise people:

    - `torch.cuda.is_available()` being `True` tells you nothing about
      transcription.
    - Both libraries load at the first file rather than at startup, so a broken
      setup starts up perfectly and fails minutes later.

### My card is newer than my PyTorch

A CUDA build of PyTorch only carries code for the graphics architectures it
was compiled for. A card newer than the build — an RTX 50-series (Blackwell,
`sm_120`) under a `cu126` build, say — is *visible* to PyTorch, is named
correctly, and then has nothing to run: the first real operation fails, and
TATERS falls back to the CPU. Nothing looks broken. The run is just twenty
times slower.

**Check my setup** catches this now: the "PyTorch can use the GPU" row says
`sm_120, which this build has no kernels for`, and gives the reinstall
command. The fix is a newer CUDA build; `sm_120` needs `cu128` or later.

While a run is going, every step that uses a model says which device it got —
"encoding on the GPU", or "encoding on the CPU" with the reason. If you see
the CPU where you expected the card, that is the same problem.

### `pip install` looked like it worked and changed nothing

This is the most common way to end up on the CPU after installing a CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If a `torch` is already installed, pip decides the requirement is satisfied and
never looks at the index. It reports success. Use:

```bash
pip install --force-reinstall --no-cache-dir torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

Check which one you have — a GPU build is numbered `2.13.0+cu128`, and the
`+cuNNN` is the part that matters:

```bash
pip show torch | grep -i version      # macOS / Linux
pip show torch | findstr /i version   # Windows
```

### Which `cuNNN` do I need?

`cu128` on almost every machine. The exception is an older driver — check what
yours supports:

```bash
nvidia-smi | grep "CUDA Version"      # macOS / Linux
nvidia-smi | findstr "CUDA Version"   # Windows
```

If that is 12.8 or higher, use `cu128`. If it is lower, use your number
instead: a driver reporting 12.4 wants `cu124`.

Do not simply match the newest CUDA your driver allows. Transcription needs
CUDA 12, so a CUDA 13 build looks like a working GPU setup and is not.

### No GPU was found

If `nvidia-smi` is not installed, there is no NVIDIA driver, and reinstalling
PyTorch will not help. Everything still runs on the CPU — slower, but correct.
Apple silicon is detected separately and uses Metal.

## Installing from a git clone

```bash
git clone https://github.com/ryanboyd/taters
cd taters
python -m venv venv-taters
source venv-taters/bin/activate      # Windows: venv-taters\Scripts\activate
pip install -e .
```

Use `pip install -e .`, not `python setup.py install`. Since Python 3.12 a
fresh virtual environment has no `setuptools` in it, so `setup.py` fails on its
first line, while `pip` fetches what the build needs.

## Errors that name the wrong package

pip works out all its versions before installing anything, so a problem with
one package often surfaces as an error about a different one. If an install
fails naming something you have never heard of — `contourpy`, `numpy`,
`ctc_segmentation` — the cause is usually an optional extra, not that package.

Install the base first and add extras one at a time:

```bash
pip install taters
pip install taters[vocalacoustics]
```
