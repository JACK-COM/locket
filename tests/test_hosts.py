"""hosts: a Paths built without a host names a home that never exists, never a KeyError."""
import tempfile
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import hosts  # noqa: E402
import paths  # noqa: E402
import trust  # noqa: E402
import corpora  # noqa: E402


class MissingHost(unittest.TestCase):
    def test_absent_hosts_match_nothing(self):
        with tempfile.TemporaryDirectory() as t, paths.use(paths.Paths.under(Path(t) / "locket")):
            for home in (hosts.claude(), hosts.hermes(), hosts.codex()):
                self.assertFalse(home.exists(), home)
            mem = Path(t) / "projects" / "-x" / "memory"
            mem.mkdir(parents=True)
            self.assertFalse(trust.granted(mem), "a store under no host's projects/ was trusted")
            self.assertEqual(corpora.raw_corpora(), {})
            self.assertEqual(trust.session_dirs(), set())


if __name__ == "__main__":
    unittest.main()
