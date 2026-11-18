"""hosts: a Paths built without a host names a home that never exists, never a KeyError."""
import tempfile
import unittest
from pathlib import Path

import tests.harness  # noqa: F401  (puts the source under test on sys.path)

import adapters  # noqa: E402
import hosts  # noqa: E402
import paths  # noqa: E402
import trust  # noqa: E402
import corpora  # noqa: E402


class Table(unittest.TestCase):
    def test_every_row_has_its_module_and_every_module_its_row(self):
        """A host is one row in hosts.TABLE and one module under adapters/; each finds
        the other, and the host every payload falls back to is asked last."""
        self.assertEqual([m.__name__ for m in adapters.TARGETS], [f"adapters.{h.adapter}" for h in hosts.TABLE])
        for m in adapters.HOSTS:
            self.assertIs(hosts.row(m.__name__), m.ROW)
        self.assertEqual([m.BASE for m in adapters.HOSTS], [False] * (len(adapters.HOSTS) - 1) + [True])
        self.assertEqual(set(hosts.HOST_MEMBERS), {h.name for h in hosts.TABLE if h.home})


class MissingHost(unittest.TestCase):
    def test_absent_hosts_match_nothing(self):
        with tempfile.TemporaryDirectory() as t, paths.use(paths.Paths.under(Path(t) / "locket")):
            for home in (hosts.home(h.name) for h in hosts.TABLE if h.home):
                self.assertFalse(home.exists(), home)
            mem = Path(t) / "projects" / "-x" / "memory"
            mem.mkdir(parents=True)
            self.assertFalse(trust.granted(mem), "a store under no host's projects/ was trusted")
            self.assertEqual(corpora.raw_corpora(), {})
            self.assertEqual(adapters.session_cwds(), set())

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
