"""What every suite shares: the source under test, the command under test, and the
scratch environment the behaviour suites run in.

LOCKET_CMD names the command under test (default: this checkout's locket.py), and
LOCKET_SRC the source the in-process suites import. Point both at an older release to
see a case fail there:  LOCKET_CMD=/opt/homebrew/bin/locket LOCKET_SRC=<its src/locket>.
The subprocess suites (test_trust, test_bounds, test_output, test_reads) run against any
release; the in-process behaviour suites need a source that has `paths.py` (0.13+).
"""
import contextlib
import os
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("LOCKET_SRC", REPO / "src/locket"))
CMD = shlex.split(os.environ.get("LOCKET_CMD", f"{sys.executable} {REPO / 'src/locket/locket.py'}"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@contextlib.contextmanager
def fixture_env():
    """Every Locket path under a scratch directory, in this process and in each child it
    spawns, so the live registry, cache, hook health, grep log and refusal records are
    never read or written. Synthetic stores under the temp dir act as trusted and a
    person's `yes` is assumed, so a behaviour suite reads a store as a person's own; the
    trust boundary itself is test_trust.py's, against the real check."""
    import corpora
    import paths
    import trust
    with tempfile.TemporaryDirectory(prefix="locket-test-") as t:
        # every host home a scratch directory, so no suite reads a live store's layout
        import hosts
        homes = {h.name: Path(t) / "hosts" / h.name for h in hosts.TABLE if h.home}
        p = paths.Paths.under(Path(t) / "locket", host_stores=homes)
        tmp = Path(tempfile.gettempdir()).resolve()
        real_trusted = trust.trusted

        def fixture_trusted(d):
            q = Path(d).resolve()
            return q == tmp or tmp in q.parents or real_trusted(d)

        env = {"LOCKET_REGISTRY": str(p.registry), "LOCKET_HEALTH_DIR": str(p.health_dir),
               "LOCKET_GREPASSIST_LOG": str(p.grepassist_log)}
        yes = lambda *_a, **_k: True                       # noqa: E731
        # Patched where each name lives: every caller reads `trust.<name>`, so one patch
        # reaches them all.
        with paths.use(p), mock.patch.dict(os.environ, env), \
                mock.patch.object(trust, "trusted", fixture_trusted), \
                mock.patch.object(trust, "person_confirms", yes):
            corpora.clear_settings()
            try:
                yield p
            finally:
                corpora.clear_settings()


# Three files the claim suites score against: a and b share one claim, c is a pointer.
DOCS = {
    "a.md": "The default execution path is the dossier pipeline run interactively.\n"
            "Unrelated sentence about hangar rates and insurance quotes entirely.",
    "b.md": "Something else here about propeller pitch enumerations and defaults.\n"
            "The default execution path is the dossier pipeline, run interactively.",
    "c.md": "See project doc basis per engine class for the whole seam and ladder.",
}


def shown(report):
    """Print a report as the CLI does and return its finding count."""
    report.show()
    return report.count


class StoreCase(unittest.TestCase):
    """A behaviour case inside `fixture_env`, with a scratch directory `self.tmp`.

    With `COUNCIL` set, the council store is a synthetic one at `self.council`: a test
    that reads the live store encodes that machine's manifest, and fails on a store with
    other ledger and holding-space names. Host tables change only through `hosts()`,
    which restores them on the way out, failed or not."""
    COUNCIL = True

    def setUp(self):
        env = fixture_env()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        self.tmp = Path(tempfile.mkdtemp(prefix="locket-case-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.council = self.tmp / "memory"
        if self.COUNCIL:
            self.council.mkdir()
            self.hosts(stores={"council": self.council})

    def hosts(self, stores=None, members=None):
        """Add or replace host stores (`paths.PATHS.host_stores`) and host members
        (`hosts.HOST_MEMBERS`) for this case only."""
        import corpora
        import hosts
        import paths
        for table, entries in ((paths.PATHS.host_stores, stores), (hosts.HOST_MEMBERS, members)):
            if entries is not None:
                patch = mock.patch.dict(table, entries)
                patch.start()
                self.addCleanup(patch.stop)
        corpora.clear_settings()
        self.addCleanup(corpora.clear_settings)
