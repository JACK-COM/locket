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

    def test_a_home_through_a_link_is_the_host(self):
        """A host home named through a symlink keeps its members and exclusions."""
        with tempfile.TemporaryDirectory() as t:
            home, link = Path(t) / "claude", Path(t) / "via"
            home.mkdir()
            link.symlink_to(home)
            with paths.use(paths.Paths.under(Path(t) / "locket", host_stores={"council": home, "hermes": Path(t) / "h"})):
                self.assertEqual(hosts.name_of(link), "council")
                self.assertEqual(hosts.members_of(link), hosts.HOST_MEMBERS["council"])
                self.assertEqual(hosts.excluded_of(link), {"synced"})
                self.assertIsNone(hosts.name_of(Path(t)), "a parent of a home was read as the home")
                self.assertFalse(hosts.foreign(link))
                self.assertTrue(hosts.foreign(Path(t) / "h"), "Hermes's home took the apparatus's defaults")


if __name__ == "__main__":
    unittest.main()
