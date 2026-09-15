# Install Taters

Three steps. If you have ever installed any software in your life, you are
overqualified.

## 1. Install Python

Taters needs **Python 3.10 or newer**. If you are choosing today, take
**3.13**: two things have no Python 3.14 release yet — speaker diarization and
training word vectors — so on 3.14 everything else installs and works, and
those two are simply missing. If you already have 3.10–3.13, skip ahead.

* **Windows** — grab the installer from
  [python.org/downloads](https://www.python.org/downloads/) and run it.
  **Tick the "Add Python to PATH" box** on the first screen; that is the one
  checkbox that saves you a headache later.
* **macOS** — grab the installer from
  [python.org/downloads](https://www.python.org/downloads/) and run it. (If you
  use Homebrew, `brew install python@3.13` works too.)
* **Linux** — you almost certainly have it. If not:
  `sudo apt-get install python3 python3-pip`.

## 2. Install Taters

If you already use Python for other things, glance at **"I like keeping my
Python installs tidy"** in the
[notes below](#optional-notes-skip-these-unless-they-describe-you) first — a
virtual environment costs one extra line now and saves you about 10,000
headaches later. Trust me.

But, if you don't want to use a virtual environment... go ahead and open a
terminal (Windows: search for "PowerShell"; macOS: search for
"Terminal") and type:

```bash
pip install taters
```

That's it. Pip fetches Taters and everything it needs.

## 3. Run it

In that same terminal, type:

```bash
taters
```

Voilà — the Taters app opens right there in your terminal and walks you
through everything else: what your data is, what you want out of it, and
where the results should go. You never have to write a line of code.

---

## Optional notes (skip these unless they describe you)

??? note "I have an NVIDIA GPU and want to use it"

    Do these **before** step 2, in this order:

    1. Install a current NVIDIA driver (and CUDA support) from
       [nvidia.com/drivers](https://www.nvidia.com/drivers).
    2. Install the CUDA 12.8 build of PyTorch:

        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        ```

    3. *Then* `pip install taters`.

    **Use `cu128`, not the newest CUDA your driver allows.** PyTorch's own
    installer page will offer you a newer one, and transcription cannot use
    it. `cu128` covers every current card, so there is nothing to gain by
    going higher —
    [Installation troubleshooting](install-troubleshooting.md) explains why.

    Already installed Taters first? No problem — inside the app, go to
    **Check my setup** and it will tell you exactly what to run to switch
    PyTorch onto your GPU.

??? note "I'll be working with audio or video files"

    Taters uses **FFmpeg** to read media files — it is the one thing pip
    can't install for you.

    * Windows: `winget install Gyan.FFmpeg` (then close and reopen the terminal)
    * macOS: `brew install ffmpeg`
    * Linux: `sudo apt-get install ffmpeg`

    Working with text only? You don't need this.

??? note "I like keeping my Python installs tidy"

    A virtual environment keeps Taters and its dependencies from mingling
    with your other Python projects — like keeping the mashed potatoes out
    of the fruit salad:

    ```bash
    python -m venv venv-taters
    # Windows:  venv-taters\Scripts\activate
    # macOS/Linux:  source venv-taters/bin/activate
    pip install taters
    ```

    Just remember to `activate` again whenever you open a new terminal.

---

Something not working? See [installation troubleshooting](install-troubleshooting.md),
or run `taters` and pick **Check my setup** — it inspects your machine and
tells you, in plain words, what (if anything) needs fixing.
