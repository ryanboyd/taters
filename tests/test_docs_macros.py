"""
The docs go through mkdocs-macros, which reads every page as a Jinja
template before it's Markdown. So a literal `{{` in prose (a prompt's
escaped brace, a pipeline placeholder) is an expression to it, and a page
that doesn't parse is built as an error page with none of its headings --
every link to one of its anchors then breaks too. A real build hit exactly
that on analyzing-text.md. Literal braces go inside {% raw %}...{% endraw %},
the way pipelines.md does it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted(DOCS.rglob("*.md"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_every_page_parses_as_the_macros_plugin_reads_it(page):
    text = page.read_text(encoding="utf-8")
    try:
        jinja2.Environment().parse(text)
    except jinja2.TemplateSyntaxError as e:
        line = text.splitlines()[e.lineno - 1] if e.lineno else ""
        pytest.fail(f"{page.relative_to(DOCS)}:{e.lineno}: {e.message}\n  {line[:160]}\n"
                    f"Wrap literal braces in {{% raw %}}...{{% endraw %}}.")


def test_the_guard_catches_the_brace_that_broke_the_build():
    with pytest.raises(jinja2.TemplateSyntaxError):
        jinja2.Environment().parse("`{{` and `}}` are literal braces.")
    jinja2.Environment().parse("{% raw %}`{{` and `}}`{% endraw %} are literal braces.")
