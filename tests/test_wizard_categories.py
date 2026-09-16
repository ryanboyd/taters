"""
The feature checklist, grouped under headings.

Why this file exists
--------------------
The checklist was one flat list in catalog order -- 22 rows on a media source
-- and nothing on it said which rows belonged together. The three topic models
and the topic-count sweep are one family; readability and lexical richness
answer related questions; none of that was visible, and every new extractor
made the screen worse.

What is pinned here is the *semantic* half of the fix, deliberately kept off
the terminal widget. `LivePrompter` cascades a heading's tick to its children
so the boxes move, but the answer is expanded again in `ask_features`, which
is what `--plain` mode and the scripted prompter rely on. That means a bug in
the key binding cannot produce a wrong selection -- only a wrong-looking one
-- and it is why these tests can drive the real function with no terminal at
all. The binding's own behavior is pinned in `test_live.py`.
"""

from __future__ import annotations

import pytest

from taters.ui import wizard as wiz
from taters.ui.prompts import ScriptedPrompter
from taters.ui.recipes import FEATURE_CATEGORIES, by_id, categories_for


def rows(source: str = "media", analyses: bool = False):
    """The checklist as it would be offered, without answering it."""
    p = ScriptedPrompter([["readability"]])
    wiz.ask_features(p, source=source, analyses=analyses)
    return p.offered_choices("Which features do you want to extract?")


# ---------------------------------------------------------------------------
# A heading is a way to tick rows, never a pick of its own
# ---------------------------------------------------------------------------

def test_ticking_a_heading_picks_everything_under_it():
    p = ScriptedPrompter([["category:topics"]])
    assert wiz.ask_features(p, source="csv") == [
        "topic_model_mem", "topic_model_lda", "topic_model_nmf",
        "topic_count_sweep"]


def test_a_heading_never_escapes_as_a_pick():
    """
    Everything downstream of here calls `recipes.by_id()` on what it is given
    and raises on anything it does not know, so a heading's value reaching
    `compose` would be a crash rather than a wrong answer.
    """
    for source in ("media", "csv"):
        for category in FEATURE_CATEGORIES:
            value = wiz.category_value(category.id)
            head = next((c for c in rows(source) if c.value == value), None)
            if head is None or head.disabled:
                continue          # not offered for this source, or nothing live under it
            picked = wiz.ask_features(ScriptedPrompter([[value]]), source=source)
            assert picked, f"{category.id} picked nothing"
            for pick in picked:
                assert not pick.startswith(wiz.CATEGORY_PREFIX)
                by_id(pick)                   # raises if it is not a real step


def test_a_heading_and_one_of_its_own_rows_is_not_two_picks():
    """Both tickable, and ticking both is the same request as ticking one."""
    p = ScriptedPrompter([["category:frequencies", "ngram_frequencies"]])
    assert wiz.ask_features(p, source="csv") == [
        "ngram_frequencies", "doc_term_matrix"]


def test_picks_come_back_in_catalog_order_not_in_the_order_they_were_ticked():
    """A stable answer, so that two people who tick the same rows in a
    different order get the same pipeline."""
    a = wiz.ask_features(ScriptedPrompter([["readability", "category:frequencies"]]),
                         source="csv")
    b = wiz.ask_features(ScriptedPrompter([["category:frequencies", "readability"]]),
                         source="csv")
    assert a == b == ["readability", "ngram_frequencies", "doc_term_matrix"]


# ---------------------------------------------------------------------------
# What the screen offers
# ---------------------------------------------------------------------------

def test_every_row_is_a_heading_or_a_step_and_every_step_is_under_one():
    for source in ("media", "csv"):
        offered = rows(source)
        headings = [c for c in offered if c.children]
        steps = [c for c in offered if not c.children]
        assert headings, "the checklist lost its headings"
        assert all(c.depth == 1 for c in steps), "a step is not indented"
        assert all(c.depth == 0 for c in headings), "a heading is indented"
        # every step sits under exactly one heading, and no heading points at
        # a row that is not on the screen
        under = [v for c in headings for v in c.children]
        assert sorted(under) == sorted(c.value for c in steps)
        assert len(under) == len(set(under))


def test_the_two_audio_headings_are_absent_for_a_folder_of_essays():
    on_text = {c.value for c in rows("csv") if c.children}
    assert "category:transcription" not in on_text
    assert "category:voice" not in on_text
    assert "category:topics" in on_text


def test_a_heading_with_nothing_available_is_itself_unavailable():
    """
    Scoring with a saved model is the case: with an empty library its only row
    is grayed out, and a heading that still looked pickable would promise
    something the screen underneath it cannot deliver.
    """
    saved = next(c for c in rows("csv") if c.value == "category:saved_models")
    children = [c for c in rows("csv") if c.value in saved.children]
    assert (bool(saved.disabled)) == all(bool(c.disabled) for c in children)


def test_a_heading_that_can_offer_nothing_asks_again():
    """
    A heading is a box like any other, so the "tick at least one" guard counts
    it -- but it may stand for nothing available, and returning an empty pick
    would send the run on with no features at all. The screen grays such a
    heading out so the pointer cannot reach it; this is the second lock.
    """
    saved = next(c for c in rows("csv") if c.value == "category:saved_models")
    if not saved.disabled:
        pytest.skip("this library has a saved model, so the heading is live")

    p = ScriptedPrompter([["category:saved_models"], ["readability"]])
    assert wiz.ask_features(p, source="csv") == ["readability"]
    assert any("Nothing under that is available" in line for line in p.output)


def test_a_heading_warns_about_statistics_only_when_all_its_rows_do():
    """
    The `not for statistics` mark says a row describes the corpus rather than
    each text. On a heading it can only be honest if it is true of everything
    underneath -- otherwise it would wave someone off a branch that does have
    what they need.
    """
    for heading in (c for c in rows("media", analyses=True) if c.children):
        under = [c for c in rows("media", analyses=True)
                 if c.value in heading.children]
        assert bool(heading.annotation) == all(bool(c.annotation) for c in under), \
            heading.label


def test_a_heading_of_corpus_wide_rows_cannot_carry_the_statistics():
    """The re-ask guard still fires when the only pick was a heading -- it
    runs on the expanded recipe ids, so it never sees a heading at all."""
    p = ScriptedPrompter([["category:transcription"], ["readability"]])
    assert wiz.ask_features(p, source="media", analyses=True) == ["readability"]
    assert any("per-text table" in line for line in p.output)


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_the_headings_come_out_in_table_order():
    """The order is a claim about what people look for first; it must not be
    at the mercy of dict iteration or catalog order."""
    from taters.ui.recipes import user_facing

    offered = {r.id for r in user_facing("media")}
    wanted = [c.id for c in FEATURE_CATEGORIES
              if any(m in offered for m in c.members)]
    assert [c.id for c, _m in categories_for("media")] == wanted


@pytest.mark.parametrize("source", ["media", "txt_dir", "csv"])
def test_grouping_adds_no_steps_and_loses_none(source):
    """The headings are presentation. The set of steps a user can reach has
    to be exactly what the flat list offered."""
    from taters.ui.recipes import user_facing

    grouped = {r.id for _c, members in categories_for(source) for r in members}
    assert grouped == {r.id for r in user_facing(source)}
