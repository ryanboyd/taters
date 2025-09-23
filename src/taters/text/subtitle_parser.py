# taters/text/subtitle_parser.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import argparse
import csv
import html
import io
import re
import sys
import textwrap

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubtitleSegment:
    """One subtitle entry (normalized across formats)."""
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
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms

def _parse_timestamp(ts: str) -> int:
    """
    Accepts 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' (SRT & VTT).
    """
    m = _TS_SRT.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid timestamp: {ts!r}")
    hh, mm, ss, ms = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return _to_ms(hh, mm, ss, ms)

def _fmt_ms_srt(ms: int) -> str:
    hh = ms // 3_600_000
    mm = (ms // 60_000) % 60
    ss = (ms // 1000) % 60
    mmm = ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02},{mmm:03}"

def _fmt_ms_vtt(ms: int) -> str:
    hh = ms // 3_600_000
    mm = (ms // 60_000) % 60
    ss = (ms // 1000) % 60
    mmm = ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02}.{mmm:03}"

def _detect_encoding(path: Path) -> str:
    """
    Try chardet if available; otherwise utf-8.
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
    return Path.cwd() / "features" / "subtitles"


# ---------------------------------------------------------------------------
# SRT parser
# ---------------------------------------------------------------------------

_SRT_TS_LINE = re.compile(r"^\s*(?P<start>[^ ]+)\s*-->\s*(?P<end>[^ ]+).*$")

def parse_srt(text: str) -> List[SubtitleSegment]:
    """
    Minimal, robust SRT parser that tolerates blank lines and extra whitespace.
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
    Minimal WebVTT parser: skips 'WEBVTT' header and NOTE/STYLE blocks.
    Speaker names (if present) are not parsed specifically; keep text as-is.
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
    Auto-detect format by extension; returns normalized segments list.
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for i, s in enumerate(segs, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ms_srt(s.start_ms)} --> {_fmt_ms_srt(s.end_ms)}\n")
            f.write(f"{s.text}\n\n")
    return out_path

def render_to_vtt(segs: Iterable[SubtitleSegment], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{_fmt_ms_vtt(s.start_ms)} --> {_fmt_ms_vtt(s.end_ms)}\n")
            f.write(f"{s.text}\n\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
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
    args = _build_arg_parser().parse_args()
    in_path = Path(args.input).resolve()
    segs = parse_subtitles(in_path, encoding=args.encoding)

    # Default output location if not provided
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = _default_out_dir()
        ext = {"csv": ".csv", "srt": ".srt", "vtt": ".vtt"}[args.to]
        out_path = out_dir / f"{in_path.stem}{ext}"

    # Render
    if args.to == "csv":
        path = render_to_csv(segs, out_path, include_name=args.include_name)
    elif args.to == "srt":
        path = render_to_srt(segs, out_path)
    else:
        path = render_to_vtt(segs, out_path)

    print(str(path))

if __name__ == "__main__":
    main()
