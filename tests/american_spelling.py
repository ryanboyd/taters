"""
British spellings to keep out of comments, docstrings and docs.

The codebase is written in American English. This is the word list and the
scanner behind ``test_spelling.py``: it walks comments and docstrings (by
token, so identifiers and string literals are never looked at), skips code
spans in prose (``like_this``, `roles`), and skips anything that touches an
underscore or a dot -- a quoted parameter or file name keeps the code's
spelling whatever that is.

Two deliberate omissions: "analyses" is the American plural of analysis,
which is what this codebase means by it; and the "cancelled" family names
the ``Cancelled`` exception class.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple

ISE_STEMS = """
normal token lemma standard summar recogn organ initial serial minim maxim optim
visual categor capital custom emphas final general priorit random real special
stabil util vector sanit memor character author critic apolog item parameter
parametr symbol synthes theor hypothes penal equal local global formal digit
central neutral personal legal actual factor familiar harmon human ideal
internal material mobil modern national natural popular public rational regular
scrutin sensit signal steril subsid synchron systemat trivial urban verbal vocal
""".split()
ISE_SUFFIXES = [("isations", "izations"), ("isation", "ization"), ("ising", "izing"),
                ("ised", "ized"), ("ises", "izes"), ("isers", "izers"),
                ("iser", "izer"), ("ise", "ize")]

PAIRS: Dict[str, str] = {
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "analyser": "analyzer", "paralyse": "paralyze", "catalyse": "catalyze",
    "neighbour": "neighbor", "neighbours": "neighbors", "neighbouring": "neighboring",
    "neighbourhood": "neighborhood",
    "colour": "color", "colours": "colors", "coloured": "colored",
    "colouring": "coloring", "colourful": "colorful",
    "behaviour": "behavior", "behaviours": "behaviors", "behavioural": "behavioral",
    "favour": "favor", "favours": "favors", "favoured": "favored",
    "favourite": "favorite", "favourable": "favorable",
    "honour": "honor", "honours": "honors", "honoured": "honored",
    "honouring": "honoring", "honourable": "honorable",
    "humour": "humor", "labour": "labor", "flavour": "flavor", "harbour": "harbor",
    "rumour": "rumor", "armour": "armor", "endeavour": "endeavor", "savour": "savor",
    "vigour": "vigor", "odour": "odor", "rigour": "rigor", "tumour": "tumor",
    "vapour": "vapor", "glamour": "glamor",
    "grey": "gray", "greys": "grays", "greyed": "grayed", "greying": "graying",
    "centre": "center", "centres": "centers", "centred": "centered",
    "centring": "centering", "metre": "meter", "metres": "meters", "litre": "liter",
    "fibre": "fiber", "theatre": "theater", "calibre": "caliber",
    "licence": "license", "licences": "licenses", "defence": "defense",
    "offence": "offense", "pretence": "pretense",
    "catalogue": "catalog", "catalogues": "catalogs", "analogue": "analog",
    "programme": "program", "programmes": "programs",
    "labelled": "labeled", "labelling": "labeling", "unlabelled": "unlabeled",
    "relabelled": "relabeled", "relabelling": "relabeling",
    "modelled": "modeled", "modelling": "modeling", "travelled": "traveled",
    "travelling": "traveling", "signalled": "signaled", "signalling": "signaling",
    "levelled": "leveled", "levelling": "leveling", "channelled": "channeled",
    "totalled": "totaled", "totalling": "totaling", "fuelled": "fueled",
    "marvellous": "marvelous", "focussed": "focused", "focussing": "focusing",
    "whilst": "while", "amongst": "among",
    "mould": "mold", "sceptic": "skeptic", "sceptical": "skeptical",
    "manoeuvre": "maneuver", "aluminium": "aluminum", "artefact": "artifact",
    "artefacts": "artifacts", "cosy": "cozy", "draught": "draft", "storey": "story",
    "sulphur": "sulfur", "ageing": "aging", "judgement": "judgment",
    "judgements": "judgments", "enrol": "enroll", "enrolment": "enrollment",
    "fulfil": "fulfill", "fulfilment": "fulfillment", "instil": "instill",
    "skilful": "skillful", "wilful": "willful", "learnt": "learned",
    "spelt": "spelled", "dreamt": "dreamed", "practise": "practice",
    "practised": "practiced", "practising": "practicing", "maths": "math",
    "tyre": "tire", "kerb": "curb", "cheque": "check", "plough": "plow",
    "tonne": "ton", "tonnes": "tons", "distil": "distill", "instalment": "installment",
    "encyclopaedia": "encyclopedia", "mediaeval": "medieval",
    "kilometre": "kilometer", "sombre": "somber", "acknowledgement": "acknowledgment",
    "acknowledgements": "acknowledgments", "dialled": "dialed", "dialling": "dialing",
    "equalled": "equaled", "equalling": "equaling", "initialled": "initialed",
    "marshalled": "marshaled", "marshalling": "marshaling", "remodelled": "remodeled",
    "remodelling": "remodeling", "trialled": "trialed", "trialling": "trialing",
    "unravelled": "unraveled", "unravelling": "unraveling", "worshipped": "worshiped",
    "spiralled": "spiraled", "spiralling": "spiraling", "tunnelled": "tunneled",
    "tunnelling": "tunneling", "panelled": "paneled", "panelling": "paneling",
    "pencilled": "penciled", "stencilled": "stenciled", "shrivelled": "shriveled",
}
for _stem in set(ISE_STEMS):
    for _br, _us in ISE_SUFFIXES:
        PAIRS.setdefault(_stem + _br, _stem + _us)
PAIRS = {k: v for k, v in PAIRS.items() if k != v}
for _skip in ("analyses", "cancelled", "cancelling"):
    PAIRS.pop(_skip, None)

BRITISH: Set[str] = set(PAIRS)

#: A word, but not a piece of an identifier or a dotted name: nothing that
#: touches an underscore, and nothing followed by ".letter" (a file's
#: extension or an attribute). A full stop at the end of a sentence is fine.
WORD_RE = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z]+(?![A-Za-z0-9_]|\.[A-Za-z0-9])")
#: Code quoted in prose: RST literals, roles, Markdown spans.
CODE_SPAN_RE = re.compile(r"``[^`\n]+``|:[a-z]+:`[^`\n]+`|`[^`\n]+`")
FENCE_RE = re.compile(r"^(```|~~~)")

#: Folders and files the scan never enters: vendored code and the reference
#: repositories, which are not ours to respell.
SKIP_PARTS = {"reference", "whisper-diarization", "venv", ".venv", "site",
              "__pycache__", ".git"}
# the demo (hashbrowns.py) was written elsewhere and is kept exactly as it
# arrived, the one word this list would flag included
SKIP_FILES = {"happierfuntokenizing.py", "hashbrowns.py"}


def british_words(text: str) -> List[str]:
    """The British spellings in one piece of prose, code spans excluded."""
    found: List[str] = []
    last = 0
    pieces = []
    for m in CODE_SPAN_RE.finditer(text):
        pieces.append(text[last:m.start()])
        last = m.end()
    pieces.append(text[last:])
    for piece in pieces:
        for m in WORD_RE.finditer(piece):
            if m.group(0).lower() in BRITISH:
                found.append(m.group(0))
    return found


def docstring_starts(source: str) -> Set[Tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    starts = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                starts.add((first.value.lineno, first.value.col_offset))
    return starts


def scan_python(path: Path) -> Iterator[Tuple[int, str]]:
    """``(line, word)`` for every British spelling in the file's comments and
    docstrings. Identifiers and other string literals are never read."""
    source = path.read_text(encoding="utf-8")
    starts = docstring_starts(source)
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT or (
                tok.type == tokenize.STRING and tok.start in starts):
            for word in british_words(tok.string):
                yield tok.start[0], word


def scan_markdown(path: Path) -> Iterator[Tuple[int, str]]:
    """``(line, word)`` for every British spelling in a Markdown file's
    prose; fenced code and code spans are skipped."""
    in_fence = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for word in british_words(line):
            yield n, word


def files_under(roots: Sequence[Path], suffixes: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in suffixes:
                out.append(root)
            continue
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in suffixes:
                continue
            if any(p in SKIP_PARTS for p in f.parts) or f.name in SKIP_FILES:
                continue
            out.append(f)
    return sorted(out)
