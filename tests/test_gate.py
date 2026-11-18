"""gate: one write, one archive read, and the 2e gate reads it only once the session's
question is known to be unasked."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.harness import fixture_env

import corpus  # noqa: E402
import gate  # noqa: E402
import manifest  # noqa: E402

NOVEL = ("Cold-soaked turbines need a longer motoring cycle before light-off on the "
         "northern routes, because the oil viscosity at the scavenge pump stalls the "
         "starter below minus thirty. ")
UNASKED = {"_host": "mcp", "_asked": False}
ASKED = {"_host": "mcp", "_asked": True}


class ArchiveReads(unittest.TestCase):
    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))
        (self.root / "facts.md").write_text("Silhouette glyphs scale by wingspan across the comparison cards.\n")
        self.reads = 0
        real = corpus.archived_claims

        def counted(root):
            self.reads += 1
            return real(root)
        patch = mock.patch.object(corpus, "archived_claims", counted)
        patch.start()
        self.addCleanup(patch.stop)

    def test_one_write_reads_the_archive_once(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL * 2}, payload=UNASKED)
        self.assertTrue(d.blocking and d.checked == "store", d)
        self.assertEqual(self.reads, 1)

    def test_an_asked_session_never_reads_it_for_the_2e_gate(self):
        msg = gate.memfind_gate(ASKED, "new.md", {}, NOVEL * 2, "", self.root)
        self.assertIsNone(msg)
        self.assertEqual(self.reads, 0)

    def test_a_short_write_reads_nothing(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL[:120]}, payload=UNASKED)
        self.assertEqual((d.message, d.blocking, self.reads), (None, False, 0))


class Decisions(unittest.TestCase):
    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))

    def test_a_check_that_raises_lets_the_write_through(self):
        d = gate.evaluate_write(str(self.root / "a\0b.md"), {"content": NOVEL * 2}, payload=UNASKED)
        self.assertEqual(d, gate.Decision(None, False, "error"))

    def test_an_autonomous_write_is_never_asked_the_question(self):
        d = gate.evaluate_write(str(self.root / "new.md"), {"content": NOVEL * 2}, autonomous=True,
                                payload=UNASKED)
        self.assertEqual((d.message, d.blocking), (None, False))

    def test_an_edit_in_a_subdirectory_never_stands_in_for_a_root_file_of_its_name(self):
        (self.root / "MEMORY.md").write_text(NOVEL * 2)
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "MEMORY.md").write_text("A line about the hangar door schedule, kept short.\n")
        d = gate.evaluate_write(str(sub / "MEMORY.md"), {"old_string": "A line about the hangar door",
                                                         "new_string": NOVEL * 3}, payload=ASKED)
        self.assertTrue(d.message and "MEMORY.md" in d.message, d)

    def test_an_append_is_checked_against_the_ledger_header(self):
        hist = self.root / "HISTORY-Proj-Sessions.csv"
        hist.write_text("date,title,summary,area,repo,commit_hash,issue,pr\n2026-09-20,Ok,Fine.,api,x,5ab427a,,\n")
        good = gate.evaluate_write(str(hist), {"content": "2026-09-21,Two,Fine.,api,x,6bc5380,,\n"}, append=True)
        self.assertEqual((good.blocking, good.checked), (False, "ledger"), good)
        bad = gate.evaluate_write(str(hist), {"content": "not-a-date,Two,Fine.,api,x,6bc5380,,\n"}, append=True)
        self.assertTrue(bad.blocking and "date" in bad.message, bad)


if __name__ == "__main__":
    unittest.main()
