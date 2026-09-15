"""
Exact n-gram counting at a fixed RAM budget.

The frequency list's counting table (every distinct n-gram with its corpus
frequency and document count) is the one structure in the text steps whose
size follows the *vocabulary*, not the row count -- and n-gram vocabularies
explode: a corpus of books can hold billions of distinct bigrams. Holding the
whole table in a dict is fastest and right for normal corpora; on a huge one
it is an out-of-memory crash with hours of work behind it.

This counter is the middle road the statistics allow, because NPMI demands
**full, unpruned counts** (see analyze_ngram_frequencies): lossy tricks like
dropping rare grams mid-count are off the table. So instead:

* Counting happens in an ordinary dict until it outgrows a byte budget.
* Then the dict is written to disk as one **sorted batch** (a "shard") and
  emptied -- a single sequential write, not a database's per-key seeking.
* At the end, the shards are merged in one streaming pass (`heapq.merge`),
  summing the partial counts of each gram. Sorted runs make that merge both
  exact and RAM-flat.

Under the budget nothing spills and the dict behaves like the dict it is --
the fast path costs one integer comparison per document.

Two contracts callers lean on:

* ``add_doc`` takes a whole document's counts at once, so a document's
  contribution to the *document* count lands in exactly one shard and the
  merge can simply sum. Spilling mid-document would double-count.
* ``sorted_items`` yields grams ordered by their **words** (``split(" ")``),
  not by the raw string. The distinction is load-bearing: in word order,
  every gram's ``(n-1)``-word prefix arrives before any of its extensions
  and nothing that is not an extension sits between them, so a scorer can
  resolve prefix counts with a small stack instead of a lookup table. Raw
  string order almost agrees -- until a token contains a character below
  space (the pos-tagged separator ``\\x1f`` is one), where it silently
  breaks the prefix property.
"""

from __future__ import annotations

import heapq
import pickle
import sqlite3
import tempfile
from itertools import groupby, islice
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

__all__ = ["SpillCounter"]

#: Rough bytes one dict entry costs beyond its characters: the string object,
#: the [freq, docs] list, and the dict slot. The budget is a courtesy, not an
#: audit -- being wrong by a third moves the spill point, never the counts.
_ENTRY_OVERHEAD = 240

#: Rows per pickle block inside a shard file. Blocks keep reading streamed
#: (one block in memory at a time) while writing stays a few big dumps.
_BLOCK_ROWS = 65536

#: Merging holds one open file per shard. At this many, existing shards are
#: compacted into one, so file handles stay bounded however small the budget
#: or however large the corpus.
_MAX_SHARDS = 32


def _sort_key(gram: str) -> List[str]:
    """Word order, not raw-string order -- see the module docstring."""
    return gram.split(" ")


def _row_key(row: Tuple[str, int, int]) -> List[str]:
    return row[0].split(" ")


def _write_shard(path: Path, rows) -> None:
    with path.open("wb") as fh:
        it = iter(rows)
        while True:
            block = list(islice(it, _BLOCK_ROWS))
            if not block:
                return
            pickle.dump(block, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _read_shard(path: Path) -> Iterator[Tuple[str, int, int]]:
    with path.open("rb") as fh:
        while True:
            try:
                block = pickle.load(fh)
            except EOFError:
                return
            yield from block


class SpillCounter:
    """
    ``gram -> (frequency, documents)`` counter, exact at any RAM budget.

    Feed it one document at a time with :meth:`add_doc`; read it back with
    :meth:`sorted_items` (word-ordered, totals exact, re-iterable) and
    :meth:`unigram_lookup` (random access to single-word counts). Call
    :meth:`close` when done -- it owns temporary files once it has spilled.

    Parameters
    ----------
    budget_bytes
        Approximate RAM the in-memory table may use before spilling a sorted
        batch to disk. ``0`` spills after every document -- correct, slow,
        and exactly what a test forcing the disk path wants.
    tmp_root
        Where the temporary shard folder is created (default: the system
        temp directory).
    on_spill
        Called with the running batch number after each spill -- the hook a
        progress display uses to say the disk is now involved.
    """

    def __init__(self, budget_bytes: int, *,
                 tmp_root: Optional[Union[str, Path]] = None,
                 on_spill: Optional[Callable[[int], None]] = None):
        self._budget = max(0, int(budget_bytes))
        self._data: Dict[str, List[int]] = {}
        self._approx = 0
        self._tmp_root = tmp_root
        self._on_spill = on_spill
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._shards: List[Path] = []
        self._shard_seq = 0
        self._sqlite: Optional[sqlite3.Connection] = None
        #: How many sorted batches went to disk / how many compactions ran.
        self.spills = 0
        self.compactions = 0

    # -- counting ------------------------------------------------------------

    @property
    def data(self) -> Dict[str, List[int]]:
        """The live table -- complete only while nothing has spilled."""
        return self._data

    @property
    def spilled(self) -> bool:
        return bool(self._shards)

    def add_doc(self, doc_counts: Dict[str, int]) -> None:
        """
        Fold one document's gram counts in: frequency by the count, the
        document tally by one. Whole documents only -- the merge sums the
        per-shard document tallies, which is the true document count exactly
        because no document straddles a shard.
        """
        data = self._data
        approx = self._approx
        for gram, count in doc_counts.items():
            entry = data.get(gram)
            if entry is None:
                data[gram] = [count, 1]
                approx += _ENTRY_OVERHEAD + len(gram)
            else:
                entry[0] += count
                entry[1] += 1
        self._approx = approx
        if approx > self._budget and data:
            self._spill()

    def _tmp_dir(self) -> Path:
        if self._tmp is None:
            root = str(self._tmp_root) if self._tmp_root else None
            self._tmp = tempfile.TemporaryDirectory(
                prefix="taters-ngram-counts-", dir=root)
        return Path(self._tmp.name)

    def _spill(self) -> None:
        rows = sorted(((g, e[0], e[1]) for g, e in self._data.items()),
                      key=_row_key)
        self._shard_seq += 1
        path = self._tmp_dir() / f"shard-{self._shard_seq:06d}.pkl"
        _write_shard(path, rows)
        self._shards.append(path)
        self._data = {}
        self._approx = 0
        self.spills += 1
        if self._on_spill is not None:
            self._on_spill(self.spills)
        if len(self._shards) >= _MAX_SHARDS:
            self._compact()

    def _compact(self) -> None:
        """Fold every shard into one, so a tiny budget over a huge corpus
        cannot accumulate more open files than the merge may hold."""
        self._shard_seq += 1
        path = self._tmp_dir() / f"compact-{self._shard_seq:06d}.pkl"
        _write_shard(path, self._merged(self._shards, None))
        for old in self._shards:
            old.unlink()
        self._shards = [path]
        self.compactions += 1

    # -- reading back ----------------------------------------------------------

    @staticmethod
    def _merged(paths: List[Path], live_rows) -> Iterator[Tuple[str, int, int]]:
        sources = [_read_shard(p) for p in paths]
        if live_rows is not None:
            sources.append(iter(live_rows))
        merged = heapq.merge(*sources, key=_row_key)
        for gram, group in groupby(merged, key=lambda row: row[0]):
            freq = docs = 0
            for _gram, f, d in group:
                freq += f
                docs += d
            yield gram, freq, docs

    def sorted_items(self) -> Iterator[Tuple[str, int, int]]:
        """
        Every gram with its exact totals, in word order. Each call starts a
        fresh pass (spilled shards are re-read), so it may be consumed more
        than once; do not interleave with further ``add_doc`` calls.
        """
        live = sorted(((g, e[0], e[1]) for g, e in self._data.items()),
                      key=_row_key)
        if not self._shards:
            return iter(live)
        return self._merged(self._shards, live)

    def unigram_lookup(self) -> Callable[[str], int]:
        """
        Random access to single-word frequencies, however the counts are held.

        In memory that is the dict. Spilled, the unigrams (the one slice of
        the vocabulary that does not explode) are loaded once, in sorted
        order, into a SQLite table in the shard folder: indexed lookups whose
        hot pages -- the Zipf head that answers most queries -- live in
        SQLite's page cache, while the long tail stays on disk. A small dict
        cache in front keeps repeat lookups from paying even that.
        """
        if not self._shards:
            data = self._data
            return lambda word: data[word][0]

        if self._sqlite is None:
            con = sqlite3.connect(str(self._tmp_dir() / "unigrams.sqlite3"))
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            con.execute("PRAGMA cache_size=-65536")  # 64 MB of pages
            con.execute("CREATE TABLE unigrams "
                        "(gram TEXT PRIMARY KEY, freq INTEGER) WITHOUT ROWID")
            batch: List[Tuple[str, int]] = []
            for gram, freq, _docs in self.sorted_items():
                if " " in gram:
                    continue
                batch.append((gram, freq))
                if len(batch) >= 50_000:
                    con.executemany("INSERT INTO unigrams VALUES (?, ?)", batch)
                    batch.clear()
            if batch:
                con.executemany("INSERT INTO unigrams VALUES (?, ?)", batch)
            con.commit()
            self._sqlite = con

        con = self._sqlite
        cache: Dict[str, int] = {}

        def lookup(word: str) -> int:
            hit = cache.get(word)
            if hit is not None:
                return hit
            row = con.execute("SELECT freq FROM unigrams WHERE gram = ?",
                              (word,)).fetchone()
            if row is None:
                raise KeyError(word)
            if len(cache) >= 200_000:
                cache.clear()
            cache[word] = row[0]
            return row[0]

        return lookup

    def close(self) -> None:
        """Drop the table and delete the shard folder. Safe to call twice."""
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self._shards = []
        self._data = {}
        self._approx = 0
