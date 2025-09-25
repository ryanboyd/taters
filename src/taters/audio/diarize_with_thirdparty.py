"""Thin CLI shim for the vendored Whisper diarization wrapper.

This module exists so you can run:

    python -m taters.audio.diarize_with_thirdparty --audio_path ...

It simply delegates to the real implementation in
``taters/audio/diarizer/whisper_diar_wrapper.py``. :contentReference[oaicite:0]{index=0}
"""

# thin alias (args still pass through)
from .diarizer.whisper_diar_wrapper import main as main

if __name__ == "__main__":
    main()
