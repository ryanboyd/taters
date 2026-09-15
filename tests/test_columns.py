"""
Reading a spreadsheet's shape for the screens that ask about its columns.
"""

import pytest

from taters.ui.columns import peek_csv, sniff_delimiter


def test_a_settled_delimiter_is_used_rather_than_sniffed_again(tmp_path,
                                                              monkeypatch):
    """
    The source stage sniffs once and the preset carries the answer; the
    analysis stage used to sniff again, unwhitelisted, so on a file where the
    two disagreed it offered columns the run would never see. With a settled
    delimiter in hand, nothing is sniffed -- and the answer is honored even
    when it is the wrong one, because that is the run's reading too.
    """
    from taters.ui import columns

    path = tmp_path / "eu.csv"
    path.write_text("id;text\n1;hello there\n2;more words\n", encoding="utf-8")
    monkeypatch.setattr(columns, "sniff_delimiter",
                        lambda p: pytest.fail("sniffed although settled"))
    cols, rows = columns.peek_csv(path, delimiter=";")
    assert cols == ["id", "text"] and rows[0]["text"] == "hello there"
    wrong, _rows = columns.peek_csv(path, delimiter=",")
    assert wrong == ["id;text"], "the settled delimiter was second-guessed"


def test_the_sniff_never_returns_a_letter(tmp_path):
    """csv.Sniffer, handed prose, returns "e" or a space; the whitelist keeps
    such answers from being baked into a pipeline."""
    path = tmp_path / "prose.csv"
    path.write_text("text\nthe quick brown fox eats every evening\n"
                    "even elephants eat eagerly\n", encoding="utf-8")
    assert sniff_delimiter(path) in (",", "\t", ";", "|")
    columns, _rows = peek_csv(path)
    assert columns == ["text"]
