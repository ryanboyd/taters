"""
Read the text out of a document, whatever the document is.

The folder source used to mean "a folder of .txt files"; it now means a folder
of *documents* -- plain text, Word, PDF -- and this module is the one place
that knows how to get text out of each. No UI imports, so the gatherer, the
runner, and any future GUI read documents the same way.

Deliberate limits:

* **No OCR.** A PDF with no machine-readable text layer has, for our
  purposes, zero text. The gatherer warns and moves on.
* **Legacy ``.doc`` needs help.** The old binary Word format has no reliable
  pure-Python reader. When ``antiword`` is on the PATH it is used; otherwise
  the file is refused with the fix in the message (open it in Word and Save
  As .docx). Refusing beats the alternative -- a byte-soup "extraction" that
  quietly poisons every word count downstream.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Union

__all__ = ["DOCUMENT_SUFFIXES", "DOCUMENT_PATTERN", "DocumentReadError",
           "read_document_text"]

PathLike = Union[str, Path]

#: What counts as a document, lowercase with the dot.
DOCUMENT_SUFFIXES = (".txt", ".docx", ".doc", ".pdf")

#: The glob set matching :data:`DOCUMENT_SUFFIXES` -- semicolon-separated,
#: which is what the folder gatherer's ``pattern`` accepts.
DOCUMENT_PATTERN = "*.txt;*.docx;*.doc;*.pdf"


class DocumentReadError(Exception):
    """A document whose text cannot be extracted, with the reason and -- where
    one exists -- the fix. Callers skip-and-warn rather than dying an hour in."""


def read_document_text(path: PathLike, encoding: str = "utf-8") -> str:
    """
    The text of one document.

    Parameters
    ----------
    path
        A ``.txt``, ``.docx``, ``.doc`` or ``.pdf`` file.
    encoding
        Used for plain text (and antiword output) only; Word and PDF carry
        their own encodings.

    Returns
    -------
    str
        The extracted text. Empty when the document genuinely has none --
        e.g. a scanned PDF with no text layer (no OCR is attempted).

    Raises
    ------
    DocumentReadError
        When extraction is impossible: a corrupt or password-protected file,
        or a legacy ``.doc`` on a machine without ``antiword``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _scrub_surrogates(_docx_text(path))

    if suffix == ".pdf":
        return _scrub_surrogates(_pdf_text(path))

    if suffix == ".doc":
        return _scrub_surrogates(_doc_text(path, encoding))

    # everything else we read as plain text -- .txt, and whatever a custom
    # pattern matched (.md, .log): if the user gave us the pattern, they're
    # telling us these files are text. one guard stays, though: a binary file
    # wearing a text name (an image, a zip) wouldn't fail here -- errors="ignore"
    # happily decodes it into token soup that poisons every count downstream.
    # NUL bytes are the tell: no text encoding produces them, every common
    # binary format does.
    head = path.read_bytes()[:4096]
    if b"\x00" in head:
        raise DocumentReadError(
            f"'{path.name}' has a text-file name but binary contents -- "
            "treated as having no text."
        )
    return path.read_text(encoding=encoding, errors="ignore")


def _docx_text(path: Path) -> str:
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as e:
        raise DocumentReadError(
            f"'{path.name}' could not be read as a Word document "
            f"({type(e).__name__}: {e})."
        ) from e
    # paragraphs, then tables: table text lives outside `paragraphs`, and a
    # Word document that's mostly a table would otherwise come back empty.
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(part for part in parts if part.strip())


def _scrub_surrogates(text: str) -> str:
    """
    Make extracted text safely UTF-8-encodable.

    pypdf extracts mathematical-alphabet characters (the 𝑥 and 𝑦 of formulas)
    from some PDFs as UTF-16 *surrogate* code points. Python strings hold them
    happily; writing them to a CSV then dies with "surrogates not allowed" --
    after minutes of gathering, from one glyph in one paper (a real run).

    Properly paired surrogates are rescued into the characters they encode;
    unpaired ones -- unrescuable by definition -- become U+FFFD. Text that was
    already clean passes through untouched, which is the only path .txt files
    ever take.
    """
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        pass
    try:
        repaired = text.encode("utf-16", "surrogatepass").decode("utf-16")
        repaired.encode("utf-8")
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text.encode("utf-8", "replace").decode("utf-8")


class _CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):  # noqa: D102 - the count is the whole point
        self.count += 1


@contextlib.contextmanager
def _quiet_pypdf():
    """
    Swallow pypdf's structural grumbling, but keep count.

    A slightly malformed PDF makes pypdf log dozens of lines like "Ignoring
    wrong pointing object 12 0 (offset 0)" straight to stderr -- which, from a
    folder of real-world PDFs, scrolled a wall of noise that named no file and
    buried the progress display. The text still extracts fine. So the log
    records are counted instead of printed, and the caller turns a nonzero
    count into ONE warning that names the file.
    """
    logger = logging.getLogger("pypdf")
    handler = _CountingHandler()
    old_propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.propagate = old_propagate


def _pdf_text(path: Path) -> str:
    import pypdf

    try:
        with _quiet_pypdf() as grumbles:
            reader = pypdf.PdfReader(str(path))
            if reader.is_encrypted:
                # an empty password opens some "protected" PDFs; a real one
                # we have no way of knowing.
                try:
                    reader.decrypt("")
                except Exception:
                    raise DocumentReadError(
                        f"'{path.name}' is password-protected."
                    ) from None
            pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentReadError:
        raise
    except Exception as e:
        raise DocumentReadError(
            f"'{path.name}' could not be read as a PDF "
            f"({type(e).__name__}: {e})."
        ) from e
    if grumbles.count:
        warnings.warn(
            f"'{path.name}' is slightly malformed ({grumbles.count} "
            "structural notice(s) from the PDF reader); its text was still "
            "extracted.", stacklevel=2)
    return "\n".join(page for page in pages if page.strip())


def _doc_text(path: Path, encoding: str) -> str:
    antiword = shutil.which("antiword")
    if antiword is None:
        raise DocumentReadError(
            f"'{path.name}' is a legacy .doc file, which needs either "
            "`antiword` installed or the file saved as .docx (open it in "
            "Word: File -> Save As -> .docx)."
        )
    try:
        result = subprocess.run([antiword, str(path)], capture_output=True,
                                timeout=120)
    except Exception as e:
        raise DocumentReadError(
            f"'{path.name}': antiword failed ({type(e).__name__}: {e})."
        ) from e
    if result.returncode != 0:
        raise DocumentReadError(
            f"'{path.name}': antiword could not read it "
            f"({result.stderr.decode(encoding, errors='ignore').strip()})."
        )
    return result.stdout.decode(encoding, errors="ignore")
