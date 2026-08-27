"""Tests for taters.helpers.find_files.

Pure filesystem logic — no media required, so we build little trees of empty
files with the right names and check what comes back.
"""

from pathlib import Path

import pytest

from taters.helpers.find_files import GROUPS, find_files


@pytest.fixture
def tree(tmp_path) -> Path:
    """A folder with one file of each interesting kind, plus a nested subfolder."""
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "hidden_dir").mkdir()
    names = [
        "clip.mp4", "clip.mkv", "shout.WAV", "song.mp3", "still.png",
        "captions.srt", "bundle.zip", "notes.txt",
        "nested/deep.mp4", "nested/deep.wav",
    ]
    for n in names:
        (root / n).touch()
    (root / ".secret.mp4").touch()
    dot = root / ".dotdir"
    dot.mkdir()
    (dot / "buried.mp4").touch()
    return root


def names(paths) -> set[str]:
    return {Path(p).name for p in paths}


# --- group selection --------------------------------------------------------

def test_video_group_matches_only_video_extensions(tree):
    assert names(find_files(tree, file_type="video")) == {"clip.mp4", "clip.mkv", "deep.mp4"}


def test_audio_group_is_case_insensitive(tree):
    # shout.WAV is uppercase on disk and must still be found.
    assert names(find_files(tree, file_type="audio")) == {"shout.WAV", "song.mp3", "deep.wav"}


def test_image_and_subtitle_and_archive_groups(tree):
    assert names(find_files(tree, file_type="image")) == {"still.png"}
    assert names(find_files(tree, file_type="subtitle")) == {"captions.srt"}
    assert names(find_files(tree, file_type="archive")) == {"bundle.zip"}


def test_any_covers_media_but_not_archives(tree):
    """`any` is audio+video+image+subtitle — deliberately not archives."""
    found = names(find_files(tree, file_type="any"))
    assert "clip.mp4" in found and "still.png" in found and "captions.srt" in found
    assert "bundle.zip" not in found
    assert "notes.txt" not in found


def test_unknown_file_type_raises(tree):
    with pytest.raises(ValueError, match="Unknown kind"):
        find_files(tree, file_type="movies")


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_files(tmp_path / "does_not_exist")


# --- explicit extensions ----------------------------------------------------

@pytest.mark.parametrize("exts", [[".wav"], ["wav"], [".WAV"], ["WAV"]])
def test_extensions_accept_any_spelling(tree, exts):
    """Leading dot optional, case irrelevant."""
    assert names(find_files(tree, extensions=exts)) == {"shout.WAV", "deep.wav"}


def test_extensions_override_the_group(tree):
    # file_type says video, but the explicit extension list wins.
    assert names(find_files(tree, file_type="video", extensions=[".txt"])) == {"notes.txt"}


def test_extensions_can_span_several_kinds(tree):
    found = names(find_files(tree, extensions=[".png", ".zip"]))
    assert found == {"still.png", "bundle.zip"}


# --- traversal --------------------------------------------------------------

def test_recursive_is_on_by_default(tree):
    assert "deep.mp4" in names(find_files(tree, file_type="video"))


def test_non_recursive_stays_in_the_top_folder(tree):
    found = names(find_files(tree, file_type="video", recursive=False))
    assert found == {"clip.mp4", "clip.mkv"}


def test_hidden_files_and_folders_are_skipped_by_default(tree):
    found = names(find_files(tree, file_type="video"))
    assert ".secret.mp4" not in found
    assert "buried.mp4" not in found      # lives inside .dotdir


def test_include_hidden_picks_up_dotfiles_and_dotdirs(tree):
    found = names(find_files(tree, file_type="video", include_hidden=True))
    assert ".secret.mp4" in found
    assert "buried.mp4" in found


# --- glob filters -----------------------------------------------------------

def test_include_globs_narrow_the_result(tree):
    found = names(find_files(tree, file_type="video", include_globs=["*deep*"]))
    assert found == {"deep.mp4"}


def test_include_globs_are_or_semantics(tree):
    found = names(find_files(tree, file_type="video", include_globs=["*deep*", "*.mkv"]))
    assert found == {"deep.mp4", "clip.mkv"}


def test_exclude_globs_remove_matches(tree):
    found = names(find_files(tree, file_type="video", exclude_globs=["*deep*"]))
    assert found == {"clip.mp4", "clip.mkv"}


def test_exclude_wins_over_include(tree):
    found = find_files(
        tree, file_type="video", include_globs=["*deep*"], exclude_globs=["*deep*"]
    )
    assert found == []


# --- output shape -----------------------------------------------------------

def test_absolute_paths_by_default(tree):
    assert all(Path(p).is_absolute() for p in find_files(tree, file_type="video"))


def test_absolute_false_keeps_paths_rooted_at_the_given_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("data")
    root.mkdir()
    (root / "a.mp4").touch()
    (found,) = find_files(root, file_type="video", absolute=False)
    assert not Path(found).is_absolute()
    assert Path(found) == root / "a.mp4"


def test_results_are_sorted_case_insensitively(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    for n in ["b.mp4", "A.mp4", "c.mp4"]:
        (root / n).touch()
    assert [Path(p).name for p in find_files(root, file_type="video")] == ["A.mp4", "b.mp4", "c.mp4"]


def test_sort_false_still_returns_everything(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    for n in ["b.mp4", "A.mp4", "c.mp4"]:
        (root / n).touch()
    assert len(find_files(root, file_type="video", sort=False)) == 3


def test_empty_folder_returns_empty_list(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert find_files(root, file_type="video") == []


# --- the extension groups themselves ---------------------------------------

def test_groups_are_lowercase_and_dotless():
    """The matcher lowercases and strips dots, so the tables must too."""
    for group, exts in GROUPS.items():
        for e in exts:
            assert e == e.lower(), f"{group}: {e!r} is not lowercase"
            assert not e.startswith("."), f"{group}: {e!r} should not start with a dot"


def test_common_formats_are_covered():
    assert {"mp4", "mkv", "mov", "webm"} <= GROUPS["video"]
    assert {"wav", "mp3", "m4a", "flac", "opus"} <= GROUPS["audio"]


# --- ffprobe verification ---------------------------------------------------

@pytest.mark.needs_ffmpeg
def test_ffprobe_verify_drops_files_without_the_requested_stream(tmp_path, tiny_video):
    """
    A video-only container should be rejected when asking for audio — this is
    exactly the case that bit the yt-dlp downloads.
    """
    root = tmp_path / "media"
    root.mkdir()
    import shutil, subprocess
    shutil.copy(tiny_video, root / "with_audio.mp4")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(tiny_video), "-an", "-c:v", "copy", str(root / "silent.mp4")],
        check=True, capture_output=True,
    )

    # By extension alone, both files look like video.
    assert names(find_files(root, file_type="video")) == {"with_audio.mp4", "silent.mp4"}

    # Asking for audio *and* verifying keeps only the one that really has an
    # audio stream. (An explicit extension list is needed because "mp4" is not
    # in the audio extension group.)
    assert names(
        find_files(root, file_type="audio", extensions=[".mp4"], ffprobe_verify=True)
    ) == {"with_audio.mp4"}

    # Without the explicit extensions, no .mp4 is even a candidate for audio.
    assert names(find_files(root, file_type="audio", ffprobe_verify=True)) == set()

    # Both files genuinely carry a video stream, so both survive video verification.
    assert names(find_files(root, file_type="video", ffprobe_verify=True)) == {
        "with_audio.mp4", "silent.mp4",
    }
