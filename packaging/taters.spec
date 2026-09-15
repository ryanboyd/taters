# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for a standalone `taters` wizard.

Build with:

    pip install pyinstaller
    pyinstaller packaging/taters.spec

The result lands in `dist/taters/` — a folder you can zip and hand to someone
who has no Python at all. They still need ffmpeg on their PATH; bundling it is a
licensing question, not a technical one, so it is left out deliberately.

Scope: the BASE install only
----------------------------
This freezes plain transcription and the text measures. It does **not** freeze
the `diarization` extra, and that is a considered decision rather than an
oversight. NeMo pulls in torch, hydra and a large stack of dynamically imported
plugins, and three of its dependencies exist only as git URLs — freezing that
reliably is a materially harder problem than everything else here combined.
Anyone who needs diarization is better served by a normal `pip install`.

What still downloads at runtime
-------------------------------
Model weights. faster-whisper fetches from Hugging Face on first use and caches
under the user's home directory. The frozen app is the code, not the models, so
the first run needs a network connection.
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# The shipped presets are package data. Without them `--list-presets` comes back
# empty and the wizard's output has nothing to sit alongside.
datas = collect_data_files("taters", includes=["**/*.yaml", "**/*.yml"])

# faster-whisper ships its VAD model as data, and CTranslate2 is a compiled
# extension whose shared libraries PyInstaller does not find on its own.
datas += collect_data_files("faster_whisper")
binaries = collect_dynamic_libs("ctranslate2")

hiddenimports = [
    # The Taters facade imports every analysis module lazily, inside the method
    # that uses it, so static analysis sees none of them.
    *collect_submodules("taters"),
    # questionary builds its prompts through prompt_toolkit's plugin machinery.
    *collect_submodules("prompt_toolkit"),
    "questionary",
    "rich",
    # Pulled in dynamically by the tokenizer and audio stacks.
    "tokenizers",
    "av",
    "onnxruntime",
]

excludes = [
    # Not part of the base install; listed so a developer machine that happens
    # to have them does not silently produce a multi-gigabyte bundle.
    "torch", "torchaudio", "nemo", "nemo_toolkit", "pytorch_lightning",
    "matplotlib", "tkinter", "IPython", "pytest",
]


a = Analysis(
    ["../src/taters/ui/wizard.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="taters",
    debug=False,
    strip=False,
    upx=False,          # UPX and CTranslate2's shared libraries do not get along
    console=True,       # it is a console wizard; a windowed build would show nothing
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="taters",
)
