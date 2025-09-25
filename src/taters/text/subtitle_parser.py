# taters/text/subtitle_parser.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Literal, Union
import argparse
import csv
import re

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubtitleSegment:
    """
    Normalized subtitle cue spanning a time interval.

    Parameters
    ----------
    number : int or None
        SRT block index if present; ``None`` for VTT or SRTs without explicit numbering.
    start_ms : int
        Start time in milliseconds.
    end_ms : int
        End time in milliseconds.
    text : str
        Cue text content. May contain embedded newlines if the source had multiple lines.
    name : str or None, optional
        Optional speaker/name field (not populated by the built-in parsers).

    Notes
    -----
    Instances are immutable (``frozen=True``) so they can be safely shared and hashed.
    """

    number: Optional[int]           # SRT index if present; None for VTT/no-number
    start_ms: int                   # start time in milliseconds
    end_ms: int                     # end time in milliseconds
    text: str                       # text content (possibly multi-line)
    name: Optional[str] = None      # optional speaker name (VTT-style notes/inline tags not parsed here)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

_TS_SRT = re.compile(r"^(\d{1,2}):([0-5]\d):([0-5]\d)[,.](\d{3})$")
_TS_VTT = _TS_SRT  # Same format; VTT uses '.' between seconds and ms, but we accept both , and .

def _to_ms(hh: int, mm: int, ss: int, ms: int) -> int:
    """
    Convert separate time components to milliseconds.

    Parameters
    ----------
    hh : int
        Hours.
    mm : int
        Minutes (0–59).
    ss : int
        Seconds (0–59).
    ms : int
        Milliseconds (0–999).

    Returns
    -------
    int
        Time in milliseconds.
    """

    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms

def _parse_timestamp(ts: str) -> int:
    """
    Parse a timestamp string into milliseconds.

    Accepts both ``HH:MM:SS,mmm`` and ``HH:MM:SS.mmm`` forms (SRT/VTT).

    Parameters
    ----------
    ts : str
        Timestamp string.

    Returns
    -------
    int
        Time in milliseconds.

    Raises
    ------
    ValueError
        If the timestamp does not match the expected pattern.
    """

    m = _TS_SRT.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid timestamp: {ts!r}")
    hh, mm, ss, ms = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return _to_ms(hh, mm, ss, ms)

def _fmt_ms_srt(ms: int) -> str:
    """
    Format milliseconds as an SRT timestamp.

    Parameters
    ----------
    ms : int
        Time in milliseconds.

    Returns
    -------
    str
        Timestamp formatted as ``HH:MM:SS,mmm``.
    """

    hh = ms // 3_600_000
    mm = (ms // 60_000) % 60
    ss = (ms // 1000) % 60
    mmm = ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02},{mmm:03}"

def _fmt_ms_vtt(ms: int) -> str:
    """
    Format milliseconds as a WebVTT timestamp.

    Parameters
    ----------
    ms : int
        Time in milliseconds.

    Returns
    -------
    str
        Timestamp formatted as ``HH:MM:SS.mmm``.
    """

    hh = ms // 3_600_000
    mm = (ms // 60_000) % 60
    ss = (ms // 1000) % 60
    mmm = ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02}.{mmm:03}"

def _detect_encoding(path: Path) -> str:
    """
    Best-effort file encoding detection.

    Uses ``chardet.UniversalDetector`` when available; otherwise falls back to ``"utf-8"``.

    Parameters
    ----------
    path : pathlib.Path
        Path to a text file.

    Returns
    -------
    str
        Detected encoding (or ``"utf-8"`` if detection is unavailable).
    """

    try:
        import chardet  # type: ignore
    except Exception:
        return "utf-8"
    with path.open("rb") as f:
        detector = chardet.UniversalDetector()
        for chunk in iter(lambda: f.read(4096), b""):
            detector.feed(chunk)
            if detector.done:
                break
        detector.close()
        enc = detector.result.get("encoding") or "utf-8"
        return enc

def _default_out_dir() -> Path:
    """
    Return the default output directory for converted subtitle files.

    Returns
    -------
    pathlib.Path
        ``<cwd>/features/subtitles``.
    """

    return Path.cwd() / "features" / "subtitles"


# ---------------------------------------------------------------------------
# SRT parser
# ---------------------------------------------------------------------------

_SRT_TS_LINE = re.compile(r"^\s*(?P<start>[^ ]+)\s*-->\s*(?P<end>[^ ]+).*$")

def parse_srt(text: str) -> List[SubtitleSegment]:
    """
    Parse SRT content into normalized subtitle segments.

    The parser tolerates extra whitespace and the optional numeric index line.
    Each cue must include a timestamp line of the form
    ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` (a dot separator for milliseconds is also
    accepted for robustness).

    Parameters
    ----------
    text : str
        Entire SRT file content.

    Returns
    -------
    list[SubtitleSegment]
        Parsed cues with millisecond times and original (joined) text.

    Raises
    ------
    ValueError
        If a well-formed timestamp line is missing where expected.
    """

    lines = [ln.rstrip("\r") for ln in text.splitlines()]
    i = 0
    n = len(lines)
    out: List[SubtitleSegment] = []

    while i < n:
        # Skip blank lines
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        # Optional numeric index
        number = None
        maybe_num = lines[i].strip()
        ts_line_idx = i
        if maybe_num.isdigit():
            number = int(maybe_num)
            i += 1
            ts_line_idx = i

        if i >= n:
            break

        # Timestamp line
        m = _SRT_TS_LINE.match(lines[ts_line_idx].strip())
        if not m:
            # Some SRTs omit numeric indices—allow timestamps immediately
            m = _SRT_TS_LINE.match(lines[i].strip())
            if not m:
                raise ValueError(f"SRT parse error: expected timestamp near line {ts_line_idx+1}")
            ts_line_idx = i
        i = ts_line_idx + 1

        start_ms = _parse_timestamp(m.group("start"))
        end_ms = _parse_timestamp(m.group("end"))

        # Content lines until blank
        content: List[str] = []
        while i < n and lines[i].strip() != "":
            content.append(lines[i])
            i += 1

        if not content:
            # SRT often allows empty entries, but we'll keep it consistent:
            # accept empty text as empty string.
            content = [""]

        text_block = "\n".join(content)
        out.append(SubtitleSegment(number=number, start_ms=start_ms, end_ms=end_ms, text=text_block, name=None))

        # Skip the trailing blank between blocks
        while i < n and lines[i].strip() == "":
            i += 1

    return out


# ---------------------------------------------------------------------------
# WebVTT parser
# ---------------------------------------------------------------------------

def parse_vtt(text: str) -> List[SubtitleSegment]:
    """
    Parse WebVTT content into normalized subtitle segments.

    Behavior:
    - Skips the ``WEBVTT`` header and any header metadata.
    - Skips ``NOTE`` and ``STYLE`` blocks.
    - Ignores optional cue identifiers.
    - Requires a timestamp line of the form
    ``HH:MM:SS.mmm --> HH:MM:SS.mmm`` (comma also accepted).

    Parameters
    ----------
    text : str
        Entire VTT file content.

    Returns
    -------
    list[SubtitleSegment]
        Parsed cues with millisecond times and original (joined) text.

    Raises
    ------
    ValueError
        If a required timestamp line is malformed or missing.
    """

    lines = [ln.rstrip("\r") for ln in text.splitlines()]
    i = 0
    n = len(lines)

    # Header
    if i < n and lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
        # Skip header meta until blank line
        while i < n and lines[i].strip() != "":
            i += 1
        while i < n and lines[i].strip() == "":
            i += 1

    out: List[SubtitleSegment] = []

    while i < n:
        # Skip NOTE/STYLE blocks
        if lines[i].strip().startswith("NOTE") or lines[i].strip().upper() == "STYLE":
            # Skip until blank line
            i += 1
            while i < n and lines[i].strip() != "":
                i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            continue

        # Optional cue identifier line (not used here)
        # If next non-empty line contains '-->' treat as timestamp; else it's an ID.
        # Lookahead 2 lines max
        if i < n and "-->" not in lines[i]:
            # Might be identifier; check next line
            if i + 1 < n and "-->" in lines[i + 1]:
                i += 1  # consume ID; ignore value
            # else fall through; if invalid, timestamp line will fail below

        if i >= n:
            break

        # Timestamp line
        line = lines[i].strip()
        if "-->" not in line:
            raise ValueError(f"VTT parse error: expected timestamp at line {i+1}")
        parts = [p.strip() for p in line.split("-->")]
        if len(parts) < 2:
            raise ValueError(f"VTT parse error: invalid timestamp at line {i+1}")

        start_ms = _parse_timestamp(parts[0])
        end_ms = _parse_timestamp(parts[1].split(" ")[0])  # drop cue settings if present
        i += 1

        # Content until blank
        content: List[str] = []
        while i < n and lines[i].strip() != "":
            content.append(lines[i])
            i += 1

        if not content:
            content = [""]

        text_block = "\n".join(content)
        out.append(SubtitleSegment(number=None, start_ms=start_ms, end_ms=end_ms, text=text_block, name=None))

        while i < n and lines[i].strip() == "":
            i += 1

    return out


# ---------------------------------------------------------------------------
# I/O layer
# ---------------------------------------------------------------------------

def parse_subtitles(input_path: Union[str, Path], *, encoding: Optional[str] = None) -> List[SubtitleSegment]:
    """
    Auto-detect and parse a subtitle file by extension.

    ``.vtt`` files are parsed as WebVTT; ``.srt`` and unknown extensions are
    parsed as SRT. Input encoding is detected with ``chardet`` when available,
    otherwise UTF-8 is assumed. Decoding errors are replaced.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to an SRT or VTT file.
    encoding : str, optional
        Override input encoding. If omitted, try detect then fall back to UTF-8.

    Returns
    -------
    list[SubtitleSegment]
        Normalized subtitle segments.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {path}")

    enc = encoding or _detect_encoding(path)
    raw = path.read_text(encoding=enc, errors="replace")

    ext = path.suffix.lower()
    if ext == ".vtt":
        return parse_vtt(raw)
    else:
        # Default to SRT for .srt or any other unknown extension (common in the wild)
        return parse_srt(raw)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_to_csv(segs: Iterable[SubtitleSegment], out_path: Union[str, Path], *, include_name: bool = False) -> Path:
    """
    Write segments to a CSV file.

    The CSV schema is:

    ``start_time,end_time[,name],text``

    Times are written as integer milliseconds (stringified) to preserve exact
    alignment for downstream tools.

    Parameters
    ----------
    segs : Iterable[SubtitleSegment]
        Segments to write.
    out_path : str or pathlib.Path
        Output CSV path.
    include_name : bool, default=False
        Include a ``name`` column (useful if upstream added speaker names).

    Returns
    -------
    pathlib.Path
        Path to the written CSV file.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["start_time", "end_time"]
        if include_name:
            header.append("name")
        header.append("text")
        w.writerow(header)
        for s in segs:
            row: List[str] = [f"{s.start_ms}", f"{s.end_ms}"]
            if include_name:
                row.append(s.name or "")
            row.append(s.text)
            w.writerow(row)
    return out_path

def render_to_srt(segs: Iterable[SubtitleSegment], out_path: Union[str, Path]) -> Path:
    """
    Write segments to SRT format.

    Blocks are 1-indexed and use ``HH:MM:SS,mmm`` timestamps.

    Parameters
    ----------
    segs : Iterable[SubtitleSegment]
        Segments to write.
    out_path : str or pathlib.Path
        Output ``.srt`` path.

    Returns
    -------
    pathlib.Path
        Path to the written SRT file.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for i, s in enumerate(segs, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ms_srt(s.start_ms)} --> {_fmt_ms_srt(s.end_ms)}\n")
            f.write(f"{s.text}\n\n")
    return out_path

def render_to_vtt(segs: Iterable[SubtitleSegment], out_path: Union[str, Path]) -> Path:
    """
    Write segments to WebVTT format.

    Includes a standard ``WEBVTT`` header and uses ``HH:MM:SS.mmm`` timestamps.

    Parameters
    ----------
    segs : Iterable[SubtitleSegment]
        Segments to write.
    out_path : str or pathlib.Path
        Output ``.vtt`` path.

    Returns
    -------
    pathlib.Path
        Path to the written VTT file.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{_fmt_ms_vtt(s.start_ms)} --> {_fmt_ms_vtt(s.end_ms)}\n")
            f.write(f"{s.text}\n\n")
    return out_path


# ---------------------------------------------------------------------------
# Primary Function
# ---------------------------------------------------------------------------

def convert_subtitles(
    *,
    input: Union[str, Path],
    to: Literal["csv", "srt", "vtt"],
    output: Optional[Union[str, Path]] = None,
    encoding: Optional[str] = None,
    include_name: bool = False,
) -> Path:
    """
    Convert an SRT/VTT file to CSV/SRT/VTT.

    Reads a subtitle file, parses into normalized segments, and renders to the
    requested format. When ``output`` is omitted, a default path is created at
    ``./features/subtitles/<input_stem>.<ext>``.

    Parameters
    ----------
    input : str or pathlib.Path
        Path to the input ``.srt`` or ``.vtt`` file.
    to : {'csv', 'srt', 'vtt'}
        Desired output format.
    output : str or pathlib.Path, optional
        Explicit output path. If ``None``, use the default location.
    encoding : str, optional
        Input encoding override; otherwise auto-detected (or UTF-8).
    include_name : bool, default=False
        When ``to='csv'``, include a ``name`` column if available.

    Returns
    -------
    pathlib.Path
        Path to the written output file.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the output format is unsupported or input content is malformed.
    """

    in_path = Path(input).resolve()
    segs = parse_subtitles(in_path, encoding=encoding)

    # Default output location if not provided
    if output is not None:
        out_path = Path(output)
    else:
        out_dir = _default_out_dir()
        ext = {"csv": ".csv", "srt": ".srt", "vtt": ".vtt"}[to]
        out_path = out_dir / f"{in_path.stem}{ext}"

    # Render
    if to == "csv":
        return render_to_csv(segs, out_path, include_name=include_name)
    elif to == "srt":
        return render_to_srt(segs, out_path)
    else:  # "vtt"
        return render_to_vtt(segs, out_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Create an ``argparse.ArgumentParser`` for the subtitle converter CLI.

    The parser supports:
    - ``--input``: path to ``.srt`` or ``.vtt``
    - ``--to``: output format (``csv``, ``srt``, ``vtt``)
    - ``--output``: optional explicit output path
    - ``--encoding``: optional input encoding override
    - ``--include-name``: include ``name`` column when writing CSV

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    p = argparse.ArgumentParser(
        prog="taters.text.subtitle_parser",
        description="Parse SRT/WebVTT subtitles and export to CSV/SRT/VTT."
    )
    p.add_argument("--input", required=True, help="Path to input .srt or .vtt")
    p.add_argument("--to", required=True, choices=("csv", "srt", "vtt"), help="Output format")
    p.add_argument("--output", default=None, help="Output file path. If omitted, uses ./features/subtitles/<stem>.<ext>")
    p.add_argument("--encoding", default=None, help="Force input encoding (otherwise chardet or utf-8).")
    p.add_argument("--include-name", action="store_true", default=False, help="Include 'name' column in CSV (if available).")
    return p

def main():
    """
    Command-line entry point for subtitle parsing and conversion.

    Parses arguments via :func:`_build_arg_parser`, calls
    :func:`convert_subtitles`, and prints the resulting output path.

    Examples
    --------
    $ python -m taters.text.subtitle_parser \
        --input transcript.srt --to csv \
        --output features/subtitles/transcript.csv
    """

    args = _build_arg_parser().parse_args()
    out = convert_subtitles(
        input=args.input,
        to=args.to,
        output=args.output,
        encoding=args.encoding,
        include_name=args.include_name,
    )
    print(str(out))


if __name__ == "__main__":
    main()
