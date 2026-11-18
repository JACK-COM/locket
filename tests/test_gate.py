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
        self._env = fixture_env()
        self._env.__enter__()
        self.root = Path(tempfile.mkdtemp())
        (self.root / manifest.MANIFEST).write_text(json.dumps({"index_files": [], "ledger_surfaces": []}))
        (self.root / "facts.md").write_text("Silhouette glyphs scale by wingspan across the comparison cards.\n")
        self.reads = 0
        real = corpus.archived_claims

        def counted(root):
            self.reads += 1
            return real(root)
        self._patch = mock.patch.object(corpus, "archived_claims", counted)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._env.__exit__(None, None, None)

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


if __name__ == "__main__":
    unittest.main()
