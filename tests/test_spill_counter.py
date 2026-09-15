"""
The spill counter: exact counts at any RAM budget.

The contract under test is blunt -- whatever the budget, however many spills
and compactions happen, the totals that come back are the totals a plain dict
would have produced, in word order. Everything else (NPMI correctness on the
disk path, byte-identical output files) is layered on that in
test_ngram_frequencies.py.
"""

from __future__ import annotations

from pathlib import Path

from taters.helpers.spill_counter import _MAX_SHARDS, SpillCounter


def _reference(doc_dicts):
    """What a plain in-memory dict says: gram -> (frequency, documents)."""
    ref = {}
    for doc in doc_dicts:
        for gram, count in doc.items():
            entry = ref.setdefault(gram, [0, 0])
            entry[0] += count
            entry[1] += 1
    return {g: tuple(e) for g, e in ref.items()}


def _fill(counter, doc_dicts):
    for doc in doc_dicts:
        counter.add_doc(dict(doc))
    return counter


def test_a_zero_budget_spills_every_document_and_stays_exact():
    docs = [
        {"the": 3, "cat": 2, "the cat": 2, "cat sat": 1},
        {"the": 1, "dog": 2, "the dog": 1, "cat sat": 2},
        {"cat": 5, "the cat": 1, "dog": 1},
    ]
    counter = _fill(SpillCounter(0), docs)
    try:
        assert counter.spills == len(docs), "budget 0 must spill per document"
        got = {g: (f, d) for g, f, d in counter.sorted_items()}
        assert got == _reference(docs)
    finally:
        counter.close()


def test_documents_are_tallied_once_per_document_across_shards():
    """A gram spread over documents on both sides of a spill boundary: the
    merge must sum per-shard document tallies, not take one of them."""
    docs = [{"peanut butter": 2}, {"peanut butter": 1}, {"peanut butter": 4}]
    counter = _fill(SpillCounter(0), docs)
    try:
        [(gram, freq, ndocs)] = list(counter.sorted_items())
        assert (gram, freq, ndocs) == ("peanut butter", 7, 3)
    finally:
        counter.close()


def test_ordering_is_by_words_not_by_raw_string():
    """The tag separator \\x1f sorts below space, so raw-string order would
    put "a b<TAG>" *before* "a b c" -- breaking the prefix-stack invariant
    scoring relies on. Word order keeps every prefix ahead of its extensions
    with nothing foreign in between."""
    tagged = "a b\x1fNN"
    docs = [{tagged: 1}, {"a b c": 1}, {"a b": 1}]
    counter = _fill(SpillCounter(0), docs)
    try:
        order = [g for g, _f, _d in counter.sorted_items()]
        assert order == ["a b", "a b c", tagged]
        assert sorted(order) != order, (
            "raw-string sort agrees here, so this corpus proves nothing"
        )
    finally:
        counter.close()


def test_compaction_bounds_the_shard_count_and_keeps_totals():
    docs = [{"w": 1, f"only-{i}": 2, "shared pair": 1}
            for i in range(_MAX_SHARDS + 5)]
    counter = _fill(SpillCounter(0), docs)
    try:
        assert counter.compactions >= 1
        assert len(counter._shards) < _MAX_SHARDS
        got = {g: (f, d) for g, f, d in counter.sorted_items()}
        assert got == _reference(docs)
    finally:
        counter.close()


def test_sorted_items_can_be_consumed_twice():
    """The scorer reads the stream once to build the unigram index and once
    to score; a generator that exhausts on the first pass would silently
    score nothing."""
    docs = [{"a": 1, "a b": 1}, {"b": 2, "a b": 1}]
    counter = _fill(SpillCounter(0), docs)
    try:
        first = list(counter.sorted_items())
        second = list(counter.sorted_items())
        assert first == second and first
    finally:
        counter.close()


def test_unigram_lookup_agrees_between_memory_and_disk():
    docs = [{"the": 10, "cat": 3, "the cat": 3}, {"the": 5, "dog": 1}]
    in_ram = _fill(SpillCounter(10**9), docs)
    on_disk = _fill(SpillCounter(0), docs)
    try:
        assert not in_ram.spilled and on_disk.spilled
        ram_of, disk_of = in_ram.unigram_lookup(), on_disk.unigram_lookup()
        for word, want in [("the", 15), ("cat", 3), ("dog", 1)]:
            assert ram_of(word) == want
            assert disk_of(word) == want
        for missing in (ram_of, disk_of):
            try:
                missing("nowhere")
            except KeyError:
                pass
            else:
                raise AssertionError("a missing word must raise, not guess")
    finally:
        in_ram.close()
        on_disk.close()


def test_close_removes_the_temporary_folder():
    counter = _fill(SpillCounter(0), [{"a": 1}, {"b": 1}])
    counter.unigram_lookup()            # this is what actually creates the sqlite file
    tmp = Path(counter._tmp.name)
    assert tmp.exists()
    counter.close()
    assert not tmp.exists()
    counter.close()                     # closing twice shouldn't hurt anything
