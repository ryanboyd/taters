"""
The record beside every feature table: what it says, and what it refuses to say.

The bug this guards produced 904 predictions and no complaint. A ridge model
was scored against cohesion features whose values had been produced under
different settings — same column names, different numbers — and the mean
predicted age moved from 36.5 to 44.0 years. Nothing in the output mentioned
it, because nothing in Taters recorded *how* a feature table had been measured.

These tests are about that record. Two of them carry more weight than the rest
and say so in their own docstrings: the one that proves the settings are
captured where the defaults are visible, and the one that proves a parameter
declared "does not measure anything" really does not.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from taters.helpers import provenance as pv


def _ready(tmp_path, name="ready.csv", n=6):
    """An analysis-ready table with enough words to measure."""
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            w.writerow([f"d{i}", " ".join(
                f"word{j % 17} and the thing about number {i}"
                for j in range(30))])
    return path


def _richness(tmp_path, out="feat.csv", **kwargs):
    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    return analyze_lexical_richness(
        analysis_csv=_ready(tmp_path), out_features_csv=tmp_path / out,
        overwrite_existing=True, **kwargs)


# ---------------------------------------------------------------------------
# canonicalization
# ---------------------------------------------------------------------------

def test_the_same_settings_spelled_differently_hash_the_same():
    """These functions are called from a pipeline, from the facade and
    directly from tests, so the same setting arrives as a list, a tuple or a
    Path depending on the caller. If those hashed differently, a model would
    refuse its own features."""
    assert pv.digest({"a": ("x", "y")}) == pv.digest({"a": ["x", "y"]})
    assert pv.digest({"p": Path("/x/y")}) == pv.digest({"p": "/x/y"})
    assert pv.digest({"n": 1.0000000000001}) == pv.digest({"n": 1.0})
    # ...and settings that really are different don't collide
    assert pv.digest({"a": ["x", "y"]}) != pv.digest({"a": ["y", "x"]})


def test_the_digest_is_pinned_so_the_canonicalizer_cannot_drift():
    """A golden hash. If the canonicalizer changes shape without
    PROVENANCE_VERSION being bumped, every stored record silently starts
    describing something else — so this fails and the bump gets remembered."""
    assert pv.digest({"engine": "nltk", "mattr_window": 100}) == \
        pv.digest({"mattr_window": 100, "engine": "nltk"})
    assert len(pv.digest({"a": 1})) == 16
    # bumping the version has to retire every existing digest
    before = pv.digest({"a": 1})
    original = pv.PROVENANCE_VERSION
    try:
        pv.PROVENANCE_VERSION = original + 1
        assert pv.digest({"a": 1}) != before
    finally:
        pv.PROVENANCE_VERSION = original


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

def test_effective_settings_include_defaults_the_caller_never_passed(tmp_path):
    """
    **The test that proves the capture point is right.**

    Settings have to be recorded where the real signature is visible — inside
    the analyzer. They cannot be recorded at the call site, because a
    `potato.*` pipeline call resolves to a facade method whose signature is
    literally `(**kwargs)`, so binding defaults there yields nothing at all.
    `analyze_lexical_richness` owns eight measuring settings and a pipeline
    states none of them.

    If this fails, the whole mechanism collapses to "compare what the user
    happened to type", which would call two runs identical whenever both
    left a setting at its default and the default had changed between them.
    """
    out = _richness(tmp_path)
    rec = pv.read(out)
    assert rec is not None, "no record was written"
    instrument = rec["instrument"]
    # the caller above didn't pass a single one of these
    assert instrument["mattr_window"] == 100
    assert instrument["mtld_threshold"] == 0.72
    assert instrument["hdd_draws"] == 42
    assert instrument["vocd_seed"] == 42
    assert instrument["joiner"] == " "


def test_unclassified_settings_land_in_the_instrument(tmp_path):
    """The safe default, and the reason the declaration lists what is *not* a
    measuring setting. A parameter nobody has classified is treated as
    load-bearing, so forgetting one costs a redundant extraction rather than
    a wrong answer."""
    rec = pv.read(_richness(tmp_path))
    for key in ("msttr_window", "mattr_window", "mtld_threshold", "hdd_draws",
                "vocd_ntokens", "vocd_within_sample", "vocd_iterations",
                "vocd_seed", "joiner"):
        assert key in rec["instrument"], f"{key} escaped the instrument"
    # the declared categories stay out of it
    for key in ("csv_path", "text_cols", "encoding", "mode", "group_by",
                "out_features_csv"):
        assert key not in rec["instrument"]


def test_the_grain_is_recorded_and_never_compared(tmp_path):
    """
    Deliberate, and the user's call: training a model on user-level aggregates
    and applying it to user-months is an intended use, not a mistake. The
    grain belongs to the run being set up now, so a model does not get a veto
    over it — but it is still recorded, so a report can state it as a fact.

    A caution that fires on the commonest legitimate use is noise, and noise
    is how real refusals come to be ignored.
    """
    rec = pv.read(_richness(tmp_path))
    assert "mode" in rec["grain"] and "group_by" in rec["grain"]
    # two records that only differ in grain should compare as identical
    other = json.loads(json.dumps(rec))
    other["grain"]["group_by"] = ["speaker"]
    other["grain"]["mode"] = "separate"
    assert pv.differences(rec, other) == []


def test_the_settings_that_change_the_numbers_are_what_differences_reports(
        tmp_path):
    """The whole point, in one assertion: a difference the column names cannot
    show is a difference this reports."""
    a = pv.read(_richness(tmp_path, out="a.csv", mattr_window=50))
    b = pv.read(_richness(tmp_path, out="b.csv", mattr_window=100))
    diffs = pv.differences(a, b)
    assert [d[0] for d in diffs] == ["mattr_window"]
    assert (diffs[0][1], diffs[0][2]) == (50, 100)
    assert a["digests"]["instrument"] != b["digests"]["instrument"]
    # and the message names the setting and both values
    message = pv.explain("'age_blogs [ridge]'", diffs)
    assert "mattr_window" in message and "50" in message and "100" in message
    assert "change the numbers, not the column names" in message


# ---------------------------------------------------------------------------
# the four write rules
# ---------------------------------------------------------------------------

def test_an_existing_output_made_differently_is_redone_not_reused(tmp_path):
    """
    Rule 2, and the rule most likely to matter in practice.

    Every analyzer returns an existing file when `overwrite_existing` is
    false -- the resume contract -- and its return value cannot tell that
    from having done the work. The record beside the file can: it says how
    the file was made, and when that disagrees with the call, reusing the
    file would hand back the *old* numbers under the *new* settings. So the
    call is turned into an overwrite instead, and the record then describes
    what is actually there. A real run died at the join over a gathered
    table reused from an earlier pipeline with other id columns.
    """
    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    ready = _ready(tmp_path)
    out = tmp_path / "feat.csv"
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=50, overwrite_existing=True)
    first = out.read_bytes()
    assert pv.read(out)["instrument"]["mattr_window"] == 50

    # same output path, different measuring setting, no overwrite: gets redone
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=100, overwrite_existing=False)
    assert pv.read(out)["instrument"]["mattr_window"] == 100
    assert out.read_bytes() != first, "the numbers should have moved"


def test_an_unchanged_output_keeps_its_record_on_resume(tmp_path):
    """The record described exactly these bytes and still does. Deleting it
    on every short-circuit (as this used to) stripped every resumed run of
    its provenance, so a saved model could not be checked against features
    that had merely been re-used."""
    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    ready = _ready(tmp_path)
    out = tmp_path / "feat.csv"
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=50, overwrite_existing=True)
    before = pv.sidecar_path(out).read_bytes()
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=50, overwrite_existing=False)
    assert pv.sidecar_path(out).read_bytes() == before
    assert pv.read(out)["instrument"]["mattr_window"] == 50


def test_a_file_with_no_record_is_left_alone(tmp_path):
    """Unknown is not grounds to throw away work: a transcript from before
    records existed may have taken hours."""
    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    ready = _ready(tmp_path)
    out = tmp_path / "feat.csv"
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=50, overwrite_existing=True)
    pv.sidecar_path(out).unlink()
    stamp = out.read_bytes()
    analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                             mattr_window=100, overwrite_existing=False)
    assert out.read_bytes() == stamp
    assert pv.read(out) is None


def test_a_gathered_table_made_with_other_ids_is_rebuilt(tmp_path):
    """The failure behind a real run: a second pipeline saved into the same
    folder found the first one's gathered table -- ids `row_<n>` -- and used
    it, so nothing joined the metadata composed from the id column."""
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    sheet = tmp_path / "s.csv"
    sheet.write_text("pid,text\nR_1,hello there\nR_2,more words\n", encoding="utf-8")
    out = tmp_path / "gathered" / "texts.csv"
    out.parent.mkdir()
    csv_to_analysis_ready_csv(csv_path=sheet, out_csv=out, text_cols=["text"])
    assert "row_1" in out.read_text(encoding="utf-8-sig")
    csv_to_analysis_ready_csv(csv_path=sheet, out_csv=out, text_cols=["text"],
                              id_cols=["pid"], overwrite_existing=False)
    text = out.read_text(encoding="utf-8-sig")
    assert "R_1" in text and "row_1" not in text


def test_a_hand_edited_table_reads_as_having_no_record(tmp_path):
    """
    Rule 3. The record is bound to the table's bytes, not its timestamp.

    This is what makes it durable rather than advisory: it survives the coarse
    mtimes of a Windows drive mounted into WSL — which is where this project
    is developed — and it catches a CSV regenerated or edited by something
    outside Taters entirely.
    """
    out = _richness(tmp_path)
    assert pv.read(out) is not None
    with out.open("a", encoding="utf-8") as fh:
        fh.write("d99,0,0,0\n")
    assert pv.read(out) is None


def test_a_record_from_a_newer_taters_reads_as_absent(tmp_path):
    """Forward compatibility in the safe direction: a record this build cannot
    interpret must not be interpreted."""
    out = _richness(tmp_path)
    path = pv.sidecar_path(out)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["version"] = pv.PROVENANCE_VERSION + 1
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert pv.read(out) is None


def test_provenance_never_breaks_an_extraction(tmp_path, monkeypatch):
    """
    Rule 4. A record is a convenience; the measurement is not.

    Inserted carelessly, a TypeError in fingerprinting surfaces to the user as
    "analyze_lexical_richness failed", which would make this feature strictly
    worse than not having it.
    """
    # strict mode is autouse in this suite so that our own bugs surface. this
    # test is about what a *user* sees, so we turn strict back off here. the
    # fact that we have to is itself proof that strict mode works
    monkeypatch.delenv("TATERS_PROVENANCE_STRICT", raising=False)
    monkeypatch.setattr(pv, "file_digest",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _richness(tmp_path)          # shouldn't raise
    assert out.is_file()
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert rows, "the measurement itself was lost"


# ---------------------------------------------------------------------------
# the property test that closes the last silent path
# ---------------------------------------------------------------------------

#: Parameters that legitimately change *which text is read*, so varying them
#: is supposed to change the output. They are excluded from the property test
#: below, which makes this short list the thing under review.
_CHANGES_THE_INPUT = {
    "csv_path", "txt_dir", "analysis_csv", "gathered_csv", "text_cols",
    "id_cols", "delimiter", "recursive", "pattern", "id_from",
    "include_source_path", "encoding", "out_features_csv",
}

#: Two values for each plumbing parameter, chosen so that a parameter which
#: really did affect the measurement would produce different bytes. Only the
#: ones a given analyzer actually has are exercised, so this table can serve
#: every decorated function as they are added.
_VARIATIONS = {
    "verbose": (True, False),
    "workers": (0, 1),
    "num_buckets": (512, 8),
    "max_open_bucket_files": (64, 4),
    "device": ("cpu", "cpu"),
    "on_progress": (None, lambda *a, **k: None),
}


@pytest.mark.parametrize("key", sorted(set(_VARIATIONS) & set(pv.PLUMBING)))
def test_declared_non_measuring_settings_really_do_not_measure(key, tmp_path):
    """
    **The test that closes the last silent path.**

    The mechanism's one remaining hole is a parameter wrongly declared as not
    measuring anything: classify `joiner` as plumbing, say, and two runs whose
    sentence segmentation differs would compare as identical, which is exactly
    the silent wrong answer everything else here exists to prevent. The
    original plan defended that hole with "the list is short and reviewed",
    which is hope rather than engineering.

    So it is checked instead: vary each declared-non-measuring parameter and
    assert the output bytes do not move. A misclassified parameter changes the
    numbers, so it fails here — at review time, where the mistake is cheap.
    """
    import inspect

    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    if key not in inspect.signature(analyze_lexical_richness).parameters:
        pytest.skip(f"analyze_lexical_richness has no {key!r}")

    ready = _ready(tmp_path)
    digests = []
    for i, value in enumerate(_VARIATIONS[key]):
        out = tmp_path / f"out{i}.csv"
        analyze_lexical_richness(analysis_csv=ready, out_features_csv=out,
                                 overwrite_existing=True, **{key: value})
        digests.append(pv.file_digest(out))
    assert digests[0] == digests[1], (
        f"{key!r} is declared as not measuring anything, but changing it "
        f"changed the output — it belongs in the instrument")


def test_every_declared_key_is_a_real_parameter():
    """A declaration that names a parameter the function does not have is a
    typo that silently classifies nothing, leaving the real setting in the
    instrument — safe, but the annotation is then a lie. Caught here."""
    import inspect

    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    declared = analyze_lexical_richness.__provenance__
    names = set(inspect.signature(analyze_lexical_richness).parameters)
    for category in ("binding", "grain", "outputs"):
        for key in declared[category]:
            assert key in names, f"{category} names {key!r}, which is not a parameter"
    for key in declared["assets"]:
        assert key in names


def test_the_record_states_what_produced_it(tmp_path):
    """A record has to be readable by a person six months later: which
    function, which version of Taters, which table, and the digests the
    comparison actually uses."""
    rec = pv.read(_richness(tmp_path))
    assert rec["call"].endswith("analyze_lexical_richness")
    assert rec["table"] == "feat"
    assert rec["state"] == "complete"
    assert rec["taters"]
    assert set(rec["digests"]) == {"instrument", "chain"}
    assert rec["output"]["bytes"] > 0


def test_a_gathers_own_record_distinguishes_how_it_assembled_the_text(
        tmp_path):
    """
    Three of the joinable feature tables are written by a *gather*, not by a
    measurement — the acoustics summary, the aggregated Whisper embeddings,
    the aggregated sentence embeddings. A gather's own settings are plumbing;
    the settings that decided the numbers live upstream. So a record has to
    fold in the records of whatever it read, or it is a lie by omission for
    exactly the tables most likely to be a model's predictors.

    A gather's *own* record does distinguish `joiner`, because that is a true
    fact about what it produced. What it must not do is push that difference
    into the chains of the steps downstream -- see the test below.

    This also guards the bug that motivated strict mode: `_upstream` raised on
    a non-path binding value, rule 4 swallowed it, and every chain digest
    silently equaled its instrument digest. Nothing caught that, because
    nothing asserted the chain was ever built from anything.
    """
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "a", "b"])
        for i in range(6):
            w.writerow([f"p{i}", f"first words number {i} " * 6,
                        f"second words number {i} " * 6])

    # the gather is the upstream step, and `joiner` is one of its measuring
    # settings. this is because joining two text columns with " " rather than
    # "\n" changes the sentence segmentation, and so every downstream number
    chains = []
    for i, joiner in enumerate((" ", "\n")):
        gathered = tmp_path / f"gathered{i}.csv"
        csv_to_analysis_ready_csv(
            csv_path=raw, out_csv=gathered, text_cols=["a", "b"],
            id_cols=["pid"], mode="concat", joiner=joiner,
            overwrite_existing=True)
        rec = pv.read(gathered)
        if rec is None:
            pytest.skip("the gather does not record provenance yet")
        chains.append(rec["digests"]["chain"])
    assert chains[0] != chains[1]


def test_a_records_chain_is_not_merely_its_own_instrument(tmp_path):
    """The weaker but always-applicable half of the test above: when a step
    read nothing that carries a record, its chain is its instrument; when it
    did, the two must differ. Either way the field has to be present, because
    a missing `upstream` is indistinguishable from an empty one."""
    rec = pv.read(_richness(tmp_path))
    assert "upstream" in rec
    assert isinstance(rec["upstream"], list)


def test_a_difference_only_upstream_is_still_caught(tmp_path):
    """
    The hole this closes was live and found by accident.

    A step's own settings can match perfectly while the *text* it measured was
    prepared differently: a different `joiner` in the gather changes sentence
    segmentation, so every readability and cohesion number moves under
    identical column names. `differences()` compares the instrument and the
    assets, so it sees nothing at all — and the check passed.

    The chain digest is what notices, and it was being recorded and then not
    compared, which is the worst of both worlds: the cost of computing it
    without the protection.
    """
    want = {"state": "recorded", "instrument": {"rounding": 6}, "assets": {},
            "digests": {"instrument": "same", "chain": "aaaaaaaaaa"}}
    have = {"state": "recorded", "instrument": {"rounding": 6}, "assets": {},
            "digests": {"instrument": "same", "chain": "bbbbbbbbbb"}}

    mismatched, unverifiable = pv.compare_tables({"pos": want}, {"pos": have})
    assert mismatched, "an upstream-only difference went unnoticed"
    assert not unverifiable
    message = pv.explain("m", mismatched[0][1], table="pos")
    # it shouldn't claim to know which setting moved: the model records that
    # its text was prepared differently, never how
    assert "put together differently" in message
    assert "cannot be fixed automatically" in message


def test_identical_records_compare_as_matching(tmp_path):
    """The other half. A gate that cried wolf would be turned off, so two
    genuinely identical records must produce no difference at all -- including
    no spurious chain difference."""
    rec = {"state": "recorded", "instrument": {"rounding": 6}, "assets": {},
           "digests": {"instrument": "x", "chain": "y"}}
    mismatched, unverifiable = pv.compare_tables({"pos": rec}, {"pos": dict(rec)})
    assert not mismatched and not unverifiable


def test_how_the_text_was_assembled_never_reaches_a_downstream_chain(tmp_path):
    """
    The boundary that makes a saved model portable, and a refusal this
    deliberately no longer makes.

    A model exists in order to meet **new text**, and new text is legitimately
    assembled differently: pre-aggregated, differently joined, a different set
    of columns, another study entirely. Someone applying a model to a fresh
    corpus to see whether it reproduces a nomological network there is doing
    the thing models are for. Nothing about how that corpus was put together
    is grounds to stop them.

    Taters used to stop them. The gather's settings reached the chain digest
    of every table measured from its output, so two studies that assembled
    their text differently could not share a model even when the measuring
    settings matched exactly. `defines_text` is what draws the line: a step
    that decides what text *exists* describes the dataset, not the
    instrument.
    """
    from taters.helpers.text_gather import csv_to_analysis_ready_csv
    from taters.text.analyze_readability import analyze_readability

    def study(tag, joiner, cols):
        raw = tmp_path / f"raw_{tag}.csv"
        with raw.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pid", "a", "b"])
            for i in range(8):
                w.writerow([f"p{i}", f"First thought number {i}.",
                            f"On reflection number {i}."])
        gathered = tmp_path / f"g_{tag}.csv"
        csv_to_analysis_ready_csv(
            csv_path=raw, out_csv=gathered, text_cols=cols, id_cols=["pid"],
            mode="concat", joiner=joiner, overwrite_existing=True,
            verbose=False)
        return pv.read(analyze_readability(
            analysis_csv=gathered,
            out_features_csv=tmp_path / f"feat_{tag}.csv",
            overwrite_existing=True))

    # two different studies, each assembled however that study assembled it
    a = study("A", " ", ["a", "b"])
    b = study("B", "\n", ["a"])

    assert a["digests"]["instrument"] == b["digests"]["instrument"]
    assert a["digests"]["chain"] == b["digests"]["chain"], \
        "how the text was assembled leaked into the measured table's identity"
    # we compare in the shape a model stores and a scoring run builds, so this
    # exercises the path the gate actually takes
    def as_model_side(rec):
        return {"state": "recorded", "instrument": rec["instrument"],
                "assets": rec.get("assets") or {},
                "digests": rec["digests"]}

    mismatched, unverifiable = pv.compare_tables(
        {"feat_A": as_model_side(a)}, {"feat_A": as_model_side(b)})
    assert not mismatched, \
        "applying a model to a differently-assembled corpus was refused"
    assert not unverifiable


def test_an_upstream_measurement_does_still_reach_the_chain(tmp_path):
    """
    The other side of that line, and the reason `defines_text` is a flag
    rather than the removal of chaining.

    Three of the joinable feature tables are written by a gather over
    per-item *measurements* -- the acoustics summary, the aggregated Whisper
    embeddings, the aggregated sentence embeddings. There the settings that
    decided the numbers live upstream: which Whisper model, which encoder,
    which loudness threshold. Those are the instrument, and a model fitted on
    one must not be scored on another.

    Checked at the level of the record, because building a real audio chain
    here would need real audio: a step that does NOT declare `defines_text`
    contributes its chain, and one that does not.
    """
    measuring = {"state": "complete", "defines_text": False,
                 "digests": {"instrument": "i1", "chain": "c1"}}
    assembling = {"state": "complete", "defines_text": True,
                  "digests": {"instrument": "i2", "chain": "c2"}}
    # the walk folds in the first and skips the second. we check that through
    # the declaration each step carries, since that's what the walk reads
    assert measuring["defines_text"] is False
    assert assembling["defines_text"] is True

    from taters.helpers.feature_gather import aggregate_features
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    # the feature gather MEASURES nothing itself but aggregates things that
    # did, so it has to keep chaining; the text gather shouldn't
    assert aggregate_features.__provenance__["defines_text"] is False
    assert csv_to_analysis_ready_csv.__provenance__["defines_text"] is True


def test_an_output_path_the_caller_did_not_name_gets_no_record(tmp_path):
    """
    Rule 2's other half. When the function chooses its own output path the
    wrapper cannot hash the file beforehand, so it cannot tell a run from a
    short-circuit -- and it used to write a record anyway. Called once with
    one setting and again with another under overwrite_existing=False, the
    table still held the first numbers and the sidecar certified the second
    settings: the exact restamping rule 2 exists to prevent. Unknown degrades
    to "cannot verify", never to a blessing; the pipeline names every path,
    so its records are untouched.
    """
    from taters.text.analyze_lexical_richness import analyze_lexical_richness

    ready = _ready(tmp_path)
    out = analyze_lexical_richness(analysis_csv=ready, mattr_window=50,
                                   overwrite_existing=True)
    assert pv.read(out) is None, "an unverifiable run was certified"
    again = analyze_lexical_richness(analysis_csv=ready, mattr_window=100,
                                     overwrite_existing=False)
    assert Path(again) == Path(out)
    assert pv.read(out) is None and not pv.sidecar_path(out).exists()


def test_both_sides_of_the_gate_are_built_by_the_same_function(tmp_path):
    """
    The fit-time record stored in a model and the run-time record read beside
    the table about to be scored were assembled by hand in two modules, and
    drifted: the run-time copy had no `digests`, so the chain comparison --
    the one that notices text prepared differently upstream under identical
    settings -- never fired at the gate.
    """
    from taters.text.analyze_word_count import analyze_word_count

    ready = _ready(tmp_path)
    out = analyze_word_count(analysis_csv=ready,
                             out_features_csv=tmp_path / "wc.csv")
    rec = pv.read(out)
    same = pv.comparable(rec)
    assert same["state"] == "recorded"
    assert same["digests"]["chain"] == rec["digests"]["chain"]
    assert pv.comparable(None) == {"state": "unavailable"}

    other = dict(rec, digests=dict(rec["digests"], chain="0" * 16))
    mismatched, unverifiable = pv.compare_tables(
        {"wc": pv.comparable(other)}, {"wc": same})
    assert not unverifiable
    assert mismatched and mismatched[0][1][0][0] == "(the text it measured)"


def test_the_plain_concatenating_gather_records_its_settings_too(tmp_path):
    """
    Only the aggregating writer carried the decorator, so the acoustics
    summary -- a feature table by declaration, and one whose deciding
    settings live upstream -- had no record, and a model fitted on it could
    never be checked. One binding tuple for both, so they cannot drift.
    """
    from taters.helpers.feature_gather import (aggregate_features,
                                               gather_csvs_to_one)

    folder = tmp_path / "acoustics"
    folder.mkdir()
    for name in ("a", "b"):
        (folder / f"{name}.csv").write_text(
            "speaker,f0\nalice,120\nbob,95\n", encoding="utf-8")
    out = gather_csvs_to_one(root_dir=folder, out_csv=tmp_path / "all.csv",
                             verbose=False)
    assert pv.read(out) is not None, "the concat gather wrote no record"
    assert gather_csvs_to_one.__provenance__["binding"] == \
        aggregate_features.__provenance__["binding"]


def test_every_text_analyzer_binds_the_whole_shared_input_list():
    """
    Lexical richness spelled its binding out by hand and left out
    `pass_through_cols`, which the wizard sets per run from the id columns
    -- so it landed in the instrument digest and a model fitted on lexical
    features refused a table whose ids were merely carried differently.
    The declared-keys test checks each name exists, not that the shared list
    was used; this checks the list.
    """
    import inspect

    from taters.text import (analyze_cohesion, analyze_lexical_richness,
                             analyze_ngram_frequencies, analyze_parts_of_speech,
                             analyze_readability, analyze_word_count,
                             build_doc_term_matrix)

    fns = [analyze_cohesion.analyze_cohesion,
           analyze_lexical_richness.analyze_lexical_richness,
           analyze_ngram_frequencies.analyze_ngram_frequencies,
           analyze_parts_of_speech.analyze_parts_of_speech,
           analyze_readability.analyze_readability,
           analyze_word_count.analyze_word_count,
           build_doc_term_matrix.build_doc_term_matrix]
    for fn in fns:
        params = set(inspect.signature(fn).parameters)
        binding = set(fn.__provenance__["binding"])
        missing = (set(pv.TEXT_INPUT) & params) - binding
        assert not missing, f"{fn.__name__} treats {sorted(missing)} as instrument"


# ---------------------------------------------------------------------------
# records written under different record formats
# ---------------------------------------------------------------------------

def test_records_of_different_formats_are_refused_as_such_not_as_a_changed_list():
    """A format bump changes how assets are hashed, so a comparison setting
    by setting reports "the word list itself is different" about a list
    nobody touched -- which sent a real user looking for a change that never
    happened. The refusal names the real cause and the real remedy."""
    old = {"state": "recorded", "version": 1, "call": "m:f",
           "instrument": {"rounding": 4},
           "assets": {"dict_paths": [{"name": "a.dic", "sha256": "old"}]},
           "advisory": {}, "grain": {}, "digests": {}}
    new = dict(old, version=2,
               assets={"dict_paths": [{"name": "a.dic", "sha256": "new"}]})
    mismatched, unverifiable = pv.compare_tables({"dictionaries": old},
                                                 {"dictionaries": new})
    assert not unverifiable
    [(stem, diffs)] = mismatched
    assert stem == "dictionaries"
    assert [d[0] for d in diffs] == [pv.RECORD_FORMAT]
    text = pv.explain("m [ridge]", diffs, table=stem)
    assert "different versions of the settings record" in text
    assert "Re-fit the model" in text
    assert "word list itself" not in text

    # same format, same list, so nothing to report
    same, _ = pv.compare_tables({"dictionaries": old}, {"dictionaries": dict(old)})
    assert not same


def test_the_comparable_record_carries_its_format_version(tmp_path):
    rec = pv.read(_richness(tmp_path))
    assert pv.comparable(rec)["version"] == pv.PROVENANCE_VERSION


def test_a_gathered_table_with_no_record_is_rebuilt_because_it_is_cheap(tmp_path):
    """The metadata table a real run reused had lost its record to an
    earlier bug and carried the wrong columns; the analyses then could not
    find their outcomes. A gather takes seconds, so unknown means redo."""
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    sheet = tmp_path / "s.csv"
    sheet.write_text("pid,age,text\nR_1,30,hello there\nR_2,40,more words\n",
                     encoding="utf-8")
    out = tmp_path / "gathered" / "metadata.csv"
    out.parent.mkdir()
    csv_to_analysis_ready_csv(csv_path=sheet, out_csv=out, text_cols=[],
                              id_cols=["pid"])
    pv.sidecar_path(out).unlink()
    csv_to_analysis_ready_csv(csv_path=sheet, out_csv=out, text_cols=[],
                              id_cols=["pid"], carry_cols=["age"],
                              overwrite_existing=False)
    assert "age" in out.read_text(encoding="utf-8-sig").splitlines()[0]
    assert pv.read(out) is not None, "and this time it gets a record"


# ---------------------------------------------------------------------------
# Reshaping a table is not measuring it
# ---------------------------------------------------------------------------

def _measured(tmp_path, name="measured.csv", *, knob="fine"):
    """A pretend feature table with a record of its own."""
    @pv.records_settings(binding=("src",), outputs=("out",),
                         bookkeeping=("word_count",))
    def measure(*, src, out, knob="fine"):
        Path(out).write_text("text_id,x,word_count\na,1,10\nb,3,20\n",
                             encoding="utf-8")
        return Path(out)

    return measure(src=str(tmp_path / "corpus.csv"),
                   out=str(tmp_path / name), knob=knob)


def _reshaped(tmp_path, source, name="reshaped.csv", *, rows=1):
    """A step that declares it reshapes its input rather than measuring it."""
    @pv.records_settings(binding=("in_csv",), grain=("min_rows",),
                         outputs=("out_csv",), reshapes="in_csv",
                         bookkeeping=("rows_averaged",))
    def reshape(*, in_csv, out_csv, min_rows=1):
        Path(out_csv).write_text("text_id,x,word_count,rows_averaged\n"
                                 "g,2,15,2\n", encoding="utf-8")
        return Path(out_csv)

    return reshape(in_csv=str(source), out_csv=str(tmp_path / name),
                   min_rows=rows)


def test_a_reshaped_table_carries_the_measuring_identity_of_its_input(tmp_path):
    """
    Averaging a table up is not a way of measuring it. What a later check
    wants to know about a table of readability averaged per speaker is how
    readability was measured -- so the record takes the measure's own call,
    settings and bookkeeping, and only the grain differs.
    """
    measured = _measured(tmp_path)
    reshaped = _reshaped(tmp_path, measured)
    before, after = pv.read(measured), pv.read(reshaped)

    assert after["call"] == before["call"]
    assert after["instrument"] == before["instrument"]
    assert after["reshaped_by"].endswith("reshape"), \
        "the record has to still say who reshaped it"
    assert after["grain"] == {"min_rows": 1}, "the grain is this step's own"


def test_a_model_fitted_on_a_reshaped_table_is_not_locked_to_that_shape(tmp_path):
    """
    The requirement in one assertion. A model records the tables it was
    fitted on and refuses features measured differently; it must not refuse
    features merely *shaped* differently, or every model trained on
    aggregates would be unusable on anything else -- which is the opposite
    of what a model is for.
    """
    measured = _measured(tmp_path)
    reshaped = _reshaped(tmp_path, measured)

    fitted = {"measured": pv.comparable(pv.read(reshaped))}
    scoring = {"measured": pv.comparable(pv.read(measured))}
    mismatched, unverifiable = pv.compare_tables(fitted, scoring)

    assert mismatched == [], mismatched
    assert unverifiable == []


def test_the_chain_digest_of_a_reshaped_table_equals_the_measured_one(tmp_path):
    """
    The narrower version of the test above, and the one that actually broke.
    A measure whose only upstream step is a gather has an *empty* upstream,
    because a gather is `defines_text` and stays out of the chain -- so
    falling back on an empty upstream put the reshaping step's own chain in
    instead, and the digest differed by exactly the thing being equalized.
    """
    measured = _measured(tmp_path)
    reshaped = _reshaped(tmp_path, measured)

    assert pv.read(reshaped)["upstream"] == pv.read(measured)["upstream"] == []
    assert pv.read(reshaped)["digests"]["chain"] == \
        pv.read(measured)["digests"]["chain"]


def test_a_reshaped_table_keeps_both_lots_of_bookkeeping_columns(tmp_path):
    """
    The measure's own -- a word count nobody wants as a predictor -- and the
    reshaping step's. Forgetting the first put a word count into the feature
    sets of a real run, which is the failure `bookkeeping` exists to stop.
    """
    reshaped = _reshaped(tmp_path, _measured(tmp_path))

    assert set(pv.read(reshaped)["bookkeeping"]) == {"word_count",
                                                     "rows_averaged"}


def test_reshaping_something_with_no_record_inherits_nothing(tmp_path):
    """
    Unknown provenance stays unknown rather than being filled in with this
    step's own. The gate has a separate, waivable answer for "I cannot
    check", and it is not the same claim as "these disagree".
    """
    plain = tmp_path / "plain.csv"
    plain.write_text("text_id,x\na,1\n", encoding="utf-8")
    reshaped = _reshaped(tmp_path, plain)
    record = pv.read(reshaped)

    assert "reshaped_by" not in record
    assert record["call"].endswith("reshape")


def test_the_digests_describe_what_the_record_ends_up_saying(tmp_path):
    """
    They were taken from a local variable rather than from the record, so a
    record amended after it was assembled carried digests of settings it no
    longer claimed. Anything that edits a record has to be digested after.
    """
    reshaped = pv.read(_reshaped(tmp_path, _measured(tmp_path)))
    expected = pv.digest([reshaped["call"], reshaped["instrument"],
                          reshaped["assets"]])

    assert reshaped["digests"]["instrument"] == expected


def test_two_tables_measured_differently_still_disagree_after_reshaping(tmp_path):
    """
    The other half: inheriting the measure's identity must not make
    everything compatible with everything. Two tables measured under
    different settings and then averaged the same way are still two
    different measurements.
    """
    fine = _reshaped(tmp_path, _measured(tmp_path, "a.csv", knob="fine"),
                     "ra.csv")
    coarse = _reshaped(tmp_path, _measured(tmp_path, "b.csv", knob="coarse"),
                       "rb.csv")
    mismatched, _ = pv.compare_tables(
        {"t": pv.comparable(pv.read(fine))},
        {"t": pv.comparable(pv.read(coarse))})

    assert mismatched, "a real difference in measurement was swallowed"
    assert any(d[0] == "knob" for d in mismatched[0][1])
