"""corpus: the claim cache answers what reading the files answers, and reads a file again
only once it changes."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.harness import fixture_env

import claims  # noqa: E402
import corpus  # noqa: E402
import manifest  # noqa: E402

OLD = time.time() - 60          # past the racy window, as a settled corpus is
FACT = "Silhouette glyphs scale by wingspan across the comparison cards on every page.\n"


class ClaimCache(unittest.TestCase):
    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))
        (self.root / "archive").mkdir()
        for name, body in (("facts.md", FACT), ("notes.md", "Turbine overhaul programs cover the engine "
                                                          "reserve entirely on this airframe.\n"),
                           ("archive/old.md", "The superseded glyph rule scaled every silhouette by its "
                                              "length instead.\n")):
            self.settle(name, body)
        corpus._CACHES.clear()
        self.addCleanup(corpus._CACHES.clear)
        self.reads = []
        real = corpus._file_claims

        def counted(p):
            self.reads.append(Path(p).name)
            return real(p)
        patch = mock.patch.object(corpus, "_file_claims", counted)
        patch.start()
        self.addCleanup(patch.stop)

    def settle(self, name, body):
        p = self.root / name
        p.write_text(body)
        os.utime(p, (OLD, OLD))

    def fresh(self):
        """A new process: nothing held in memory, only what the cache file kept."""
        corpus._CACHES.clear()
        self.reads.clear()

    def index_of(self):
        ix = corpus.claim_index(self.root)
        return {n: sorted(s for f, s, _ in ix.items if ix.names[f] == n) for n in ix.names}

    def test_a_warm_cache_reads_no_file_and_answers_the_same(self):
        want = {n: sorted(s for s, _ in claims.claims(b)) for n, b in corpus.corpus(self.root).items()}
        self.assertEqual(self.index_of(), want)
        corpus.archived_claims(self.root)
        self.fresh()
        self.assertEqual(self.index_of(), want)
        self.assertEqual(self.reads, [])
        self.assertEqual(len(corpus.archived_claims(self.root)), 1)
        self.assertEqual(self.reads, [])

    def test_a_changed_file_is_read_again(self):
        self.index_of()
        self.fresh()
        self.settle("facts.md", FACT + "Each card names its source beside the figure it carries today.\n")
        self.assertEqual(len(self.index_of()["facts.md"]), 2)
        self.assertEqual(self.reads, ["facts.md"])

    def test_a_file_written_this_instant_is_never_cached(self):
        (self.root / "facts.md").write_text(FACT)            # mtime now: racily clean
        self.index_of()
        self.fresh()
        self.index_of()
        self.assertEqual(self.reads, ["facts.md"])

    def test_a_removed_file_leaves_the_cache(self):
        self.index_of()
        (self.root / "notes.md").unlink()
        self.settle("facts.md", FACT + "Each card names its source beside the figure it carries today.\n")
        self.index_of()
        kept = json.loads((corpus.cache_dir(self.root) / corpus.CLAIM_CACHE).read_text())["files"]
        self.assertFalse([k for k in kept if k.endswith("notes.md")], kept)

    def test_a_damaged_or_foreign_cache_reads_as_empty(self):
        self.index_of()
        path = corpus.cache_dir(self.root) / corpus.CLAIM_CACHE
        for junk in ("{", json.dumps({"version": -1, "files": {}}), json.dumps([1, 2])):
            path.write_text(junk)
            self.fresh()
            self.assertIn("facts.md", self.index_of())
            self.assertIn("facts.md", self.reads)

    def test_a_damaged_entry_under_a_current_fingerprint_is_a_miss(self):
        self.index_of()
        path = corpus.cache_dir(self.root) / corpus.CLAIM_CACHE
        good = json.loads(path.read_text())
        key = next(k for k in good["files"] if k.endswith("facts.md"))
        fp = good["files"][key][0]
        for entry in (5, [fp, "abc"], [fp, [["s", "a b"], ["q"]]], [fp, [["s", 5]]], [fp, [["s", [["t", "u v"]]]]]):
            data = json.loads(json.dumps(good))
            data["files"][key] = entry
            path.write_text(json.dumps(data))
            self.fresh()
            self.assertEqual(len(self.index_of()["facts.md"]), 1, entry)
            self.assertIn("facts.md", self.reads, entry)

    def test_a_csv_that_stops_parsing_keeps_the_rows_before_the_fault(self):
        rows = self.root / "R.csv"
        rows.write_text("Ruling,Area\nChock both mains before the tug is disconnected from the nosewheel.,ops\n"
                        '"' + "x" * 200_000 + '",ops\n')
        (self.root / manifest.MANIFEST).write_text(json.dumps({
            "index_files": [], "ledger_surfaces": [],
            "sources": [{"path": "R.csv", "text": "Ruling", "label": "Area"}]}))
        corpus.corpora.clear_settings()           # the manifest changed on disk
        self.assertEqual(len(corpus.csv_rows(self.root)), 1)
        self.assertEqual(len(self.index_of().get("R.csv#ops", [])), 1)


if __name__ == "__main__":
    unittest.main()
