"""
The acoustics step's contract at the door: what it refuses before touching
audio. The measurements themselves are exercised in test_slow_audio.py.
"""

import pytest

parselmouth = pytest.importorskip("parselmouth", reason="needs praat-parselmouth")


def test_a_tremor_mode_without_a_script_is_refused_before_any_audio(tmp_path):
    """
    The docstring always said the Praat script is required for tremor and
    advanced; the per-clip code quietly returned the simple set instead, so a
    user who picked "tremor" got no tremor columns and no word about it,
    after the whole run. Refused at the door, before a file is opened.
    """
    from taters.audio.analyze_vocal_acoustics import analyze_acoustics

    for mode in ("tremor", "advanced"):
        with pytest.raises(ValueError, match="tremor_script"):
            analyze_acoustics(wav_path=tmp_path / "missing.wav", mode=mode,
                              out_dir=tmp_path)
    assert not list(tmp_path.iterdir()), "something was written before refusing"


def _tone(path, seconds=1.0, sr=16000, hz=140.0):
    """A one-second voiced-ish tone: enough for the pitch tracker to run."""
    import numpy as np
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * hz * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    sf.write(str(path), y.astype("float32"), sr)
    return path


def test_the_step_reports_progress_and_keeps_quiet_when_told(tmp_path, capsys):
    """
    Every other analysis takes `verbose` and `on_progress`, and the runner
    injects both only into signatures that accept them. Acoustics accepted
    neither, so its skip-path print landed inside the live display and the
    per-turn loop -- the longest CPU step in a conversation pipeline --
    reported nothing while it ran.
    """
    from taters.audio.analyze_vocal_acoustics import analyze_acoustics

    wav = _tone(tmp_path / "clip.wav")
    seen = []
    analyze_acoustics(wav_path=wav, out_dir=tmp_path / "a", include_framewise=False,
                      verbose=False, on_progress=lambda *a: seen.append(a))
    assert seen and seen[0][2] == "measuring the recording"
    # now the skip path, which should stay quiet.
    analyze_acoustics(wav_path=wav, out_dir=tmp_path / "a", include_framewise=False,
                      verbose=False)
    assert capsys.readouterr().out == ""

    transcript = tmp_path / "turns.csv"
    transcript.write_text("speaker,start_time,end_time,text\n"
                          "A,0,450,hello\nB,500,1000,there\n", encoding="utf-8")
    seen.clear()
    analyze_acoustics(wav_path=wav, transcript_csv=transcript, time_unit="ms",
                      out_dir=tmp_path / "b", include_framewise=False,
                      verbose=False, on_progress=lambda *a: seen.append(a))
    ticks = [a for a in seen if a[2] == "measuring turns" and a[1] == 2]
    assert [a[0] for a in ticks] == [1, 2], seen
    assert not list((tmp_path / "b").glob("*.part*")), "a scratch file was left"
