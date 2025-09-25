# taters/helpers/find_files.py
from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import Iterable, Sequence, List, Optional, Set

# --- Extension groups (lowercase, without the dot) ---------------------------
AUDIO_EXTS: Set[str] = {
    # lossless
    "wav", "flac", "alac", "aiff", "aif", "aifc",
    # lossy
    "mp3", "m4a", "aac", "ogg", "oga", "opus", "wma",
    # containers that often hold audio
    "mka",
}

VIDEO_EXTS: Set[str] = {
    "mp4", "m4v", "mov", "avi", "mkv", "webm", "wmv", "flv", "mpeg",
    "mpg", "m2ts", "ts", "3gp", "3g2", "ogv",
}

IMAGE_EXTS: Set[str] = {
    "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif", "heic", "heif",
}

SUBTITLE_EXTS: Set[str] = {
    "srt", "vtt", "ass", "ssa", "sub",
}

ARCHIVE_EXTS: Set[str] = {
    "zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar",
}

GROUPS = {
    "audio": AUDIO_EXTS,
    "video": VIDEO_EXTS,
    "image": IMAGE_EXTS,
    "subtitle": SUBTITLE_EXTS,
    "archive": ARCHIVE_EXTS,
    "any": AUDIO_EXTS | VIDEO_EXTS | IMAGE_EXTS | SUBTITLE_EXTS,
}

def _norm_ext(e: str) -> str:
    """Normalize an extension to 'ext' (lowercase, no dot)."""
    e = e.strip().lower()
    return e[1:] if e.startswith(".") else e

def _match_ext(path: Path, allowed: Set[str]) -> bool:
    try:
        ext = path.suffix.lower().lstrip(".")
        return ext in allowed
    except Exception:
        return False

def _iter_files(
    root: Path,
    *,
    recursive: bool,
    follow_symlinks: bool,
    include_hidden: bool,
) -> Iterable[Path]:
    if not recursive:
        for p in root.iterdir():
            if p.is_file() or (follow_symlinks and p.is_symlink() and p.exists()):
                if include_hidden or not p.name.startswith("."):
                    yield p
    else:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            dpath = Path(dirpath)
            if not include_hidden:
                # prune hidden directories
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if not include_hidden and fn.startswith("."):
                    continue
                yield dpath / fn

def _glob_filter(paths: Iterable[Path], includes: Sequence[str], excludes: Sequence[str]) -> Iterable[Path]:
    if not includes and not excludes:
        yield from paths
        return
    # include first (logical OR), then exclude
    for p in paths:
        inc_ok = True if not includes else any(p.match(gl) for gl in includes)
        if not inc_ok:
            continue
        if any(p.match(gl) for gl in excludes):
            continue
        yield p

def _ffprobe_has_stream(path: Path, kind: str) -> bool:
    """
    Return True if ffprobe sees at least one stream of the requested kind ('audio'|'video').
    """
    stream_sel = {"audio": "a", "video": "v"}.get(kind, "")
    if not stream_sel:
        return False
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", stream_sel,
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False

def find_files(
    root_dir: str | os.PathLike,
    *,
    file_type: str = "video",                            # 'audio' | 'video' | 'image' | 'subtitle' | 'archive' | 'any'
    extensions: Optional[Sequence[str]] = None,     # explicit extensions override group (e.g., ['.wav','.flac'])
    recursive: bool = True,
    follow_symlinks: bool = False,
    include_hidden: bool = False,
    include_globs: Optional[Sequence[str]] = None,  # e.g., ['**/*session*']
    exclude_globs: Optional[Sequence[str]] = None,  # e.g., ['**/temp/**']
    absolute: bool = True,
    sort: bool = True,
    ffprobe_verify: bool = False,                   # confirm stream presence via ffprobe (audio/video only)
) -> List[Path]:
    """
    Discover media files under a folder using smart, FFmpeg-friendly filters.

    You can either (a) choose a built-in **group** of extensions via `file_type`
    (`"audio"|"video"|"image"|"subtitle"|"archive"|"any"`) or (b) pass an explicit
    list of `extensions` to match. Matching is case-insensitive; dots are optional
    (e.g., `".wav"` and `"wav"` are equivalent). Hidden files and directories are
    excluded by default.

    For audio/video, `ffprobe_verify=True` additionally checks that at least one
    corresponding stream is present (e.g., exclude MP4s with no audio when
    `file_type="audio"`). This is slower but robust when your dataset contains
    “container only” files. :contentReference[oaicite:0]{index=0}

    Parameters
    ----------
    root_dir
        Folder to scan.
    file_type
        Built-in group selector. Ignored if `extensions` is provided.
    extensions
        Explicit extensions to include (e.g., `[".wav",".flac"]`). Overrides `file_type`.
    recursive
        Recurse into subfolders. Default: `True`.
    follow_symlinks
        Follow directory symlinks during traversal. Default: `False`.
    include_hidden
        Include dot-files and dot-dirs. Default: `False`.
    include_globs / exclude_globs
        Additional glob filters applied after extension filtering; `include_globs`
        uses OR-semantics, then `exclude_globs` removes matches.
    absolute
        Return absolute paths when `True` (default) else relative to `root_dir`.
    sort
        Sort lexicographically (case-insensitive). Default: `True`.
    ffprobe_verify
        For `audio`/`video`, keep only files where `ffprobe` reports ≥1 matching
        stream.

    Returns
    -------
    list[pathlib.Path]
        The matched files.

    Raises
    ------
    FileNotFoundError
        If `root_dir` does not exist.
    ValueError
        If `file_type` is not one of the supported groups.

    Examples
    --------
    Find all videos (recursive), as absolute paths:

    >>> find_files("dataset", file_type="video")

    Use explicit extensions and keep paths relative:

    >>> find_files("dataset", extensions=[".wav",".flac"], absolute=False)

    Only include files matching a glob and exclude temp folders:

    >>> find_files("dataset", file_type="audio",
    ...            include_globs=["**/*session*"], exclude_globs=["**/tmp/**"])

    Verify playable audio streams exist:

    >>> find_files("dataset", file_type="audio", ffprobe_verify=True)
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root path not found: {root_dir}")

    if extensions:
        allowed = {_norm_ext(e) for e in extensions}
    else:
        if file_type not in GROUPS:
            raise ValueError(f"Unknown kind '{file_type}'. Choose from {', '.join(GROUPS.keys())}.")
        allowed = set(GROUPS[file_type])

    cand = (
        p for p in _iter_files(root_dir, recursive=recursive, follow_symlinks=follow_symlinks, include_hidden=include_hidden)
        if p.is_file() and _match_ext(p, allowed)
    )

    cand = _glob_filter(
        cand,
        includes=include_globs or [],
        excludes=exclude_globs or [],
    )

    out: List[Path] = []
    for p in cand:
        if ffprobe_verify and file_type in ("audio", "video"):
            if not _ffprobe_has_stream(p, file_type):
                continue
        out.append(p.resolve() if absolute else p)

    if sort:
        out.sort(key=lambda x: str(x).lower())
    return out

# --- CLI ---------------------------------------------------------------------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="List media files under a folder (audio/video/image/subtitle/any) or by explicit extension(s)."
    )
    p.add_argument("root_dir", help="Root folder to scan")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--file_type", choices=GROUPS.keys(), default="video",
                   help="Built-in media kind (default: audio)")
    g.add_argument("--ext", dest="extensions", action="append",
                   help="Explicit extension to include (repeatable), e.g., --ext .wav --ext .flac")

    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse into subfolders")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow directory symlinks")
    p.add_argument("--include-hidden", action="store_true", help="Include hidden files and folders")

    p.add_argument("--include", dest="include_globs", action="append",
                   help="Glob to include (repeatable). Example: --include '**/*session*'")
    p.add_argument("--exclude", dest="exclude_globs", action="append",
                   help="Glob to exclude (repeatable). Example: --exclude '**/temp/**'")

    p.add_argument("--relative", dest="absolute", action="store_false", help="Print relative paths instead of absolute")
    p.add_argument("--unsorted", dest="sort", action="store_false", help="Keep filesystem order")

    p.add_argument("--ffprobe-verify", action="store_true",
                   help="Verify at least one matching stream exists (audio/video only)")

    p.add_argument("--out", default=None, help="Write the list to a file instead of stdout")
    return p

def main():
    args = _build_arg_parser().parse_args()
    paths = find_files(
        root_dir=args.root_dir,
        file_type=args.file_type if args.extensions is None else "any",
        extensions=args.extensions,
        recursive=args.recursive,
        follow_symlinks=args.follow_symlinks,
        include_hidden=args.include_hidden,
        include_globs=args.include_globs,
        exclude_globs=args.exclude_globs,
        absolute=args.absolute,
        sort=args.sort,
        ffprobe_verify=args.ffprobe_verify,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
        print(f"Wrote {len(paths)} paths to {out_path}")
    else:
        for p in paths:
            print(p)

if __name__ == "__main__":
    main()
