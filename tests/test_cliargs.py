"""Tests for taters.helpers.cliargs.

This is the simplest test module in the suite, so it doubles as a tour of the
three things you need to know:

  1. A test is a function whose name starts with `test_`. pytest finds it, runs
     it, and calls it a pass if it returns without raising.
  2. You check things with a bare `assert`. No special assertion methods —
     pytest rewrites `assert` so a failure prints both sides of the comparison.
  3. `pytest.raises` is how you assert that something *should* blow up.
"""

import argparse

import pytest

from taters.helpers.cliargs import add_bool_argument, str2bool


# `parametrize` runs the same test body once per case, and reports each one
# separately — so a failure tells you *which* input broke rather than just
# "the loop failed somewhere".
@pytest.mark.parametrize(
    "text",
    ["true", "True", "TRUE", "t", "yes", "y", "on", "1"],
)
def test_str2bool_accepts_truthy_spellings(text):
    assert str2bool(text) is True


@pytest.mark.parametrize(
    "text",
    ["false", "False", "FALSE", "f", "no", "n", "off", "0"],
)
def test_str2bool_accepts_falsy_spellings(text):
    assert str2bool(text) is False


def test_str2bool_passes_through_real_booleans():
    assert str2bool(True) is True
    assert str2bool(False) is False


def test_str2bool_tolerates_surrounding_whitespace():
    assert str2bool("  true  ") is True


@pytest.mark.parametrize("text", ["maybe", "", "2", "yeah", "none", "file.mp4"])
def test_str2bool_rejects_anything_ambiguous(text):
    # The point of this function is to fail loudly. Silently reading "maybe"
    # as False is how someone loses an overnight run.
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool(text)


def test_str2bool_error_message_names_the_bad_value():
    with pytest.raises(argparse.ArgumentTypeError, match="banana"):
        str2bool("banana")


class TestAddBoolArgument:
    """
    Tests can be grouped in a class for readability. The only rule is that the
    class name starts with `Test` and it has no `__init__`.
    """

    @staticmethod
    def _parser(default=False):
        p = argparse.ArgumentParser()
        add_bool_argument(p, "--overwrite_existing", default=default, help="x")
        return p

    def test_absent_flag_uses_the_default(self):
        assert self._parser(default=False).parse_args([]).overwrite_existing is False
        assert self._parser(default=True).parse_args([]).overwrite_existing is True

    def test_bare_flag_means_true(self):
        assert self._parser().parse_args(["--overwrite_existing"]).overwrite_existing is True

    @pytest.mark.parametrize("value,expected", [("false", False), ("true", True), ("0", False)])
    def test_explicit_value_is_honored(self, value, expected):
        ns = self._parser().parse_args(["--overwrite_existing", value])
        assert ns.overwrite_existing is expected

    def test_bare_flag_does_not_swallow_the_next_option(self):
        p = self._parser()
        p.add_argument("--sr", type=int, default=16000)
        ns = p.parse_args(["--overwrite_existing", "--sr", "48000"])
        assert ns.overwrite_existing is True
        assert ns.sr == 48000

    def test_junk_value_exits_with_an_error(self):
        # argparse calls sys.exit() on a bad argument, which raises SystemExit.
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--overwrite_existing", "banana"])

    def test_custom_dest_is_respected(self):
        p = argparse.ArgumentParser()
        add_bool_argument(p, "--voiced-segments", dest="voiced_segments", default=True)
        assert p.parse_args([]).voiced_segments is True
        assert p.parse_args(["--voiced-segments", "no"]).voiced_segments is False
